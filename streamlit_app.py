import importlib.util
import json
import sys
import uuid
from pathlib import Path

import plotly.io as pio
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
SQL_RUNNER_PATH = SRC_DIR / "02_run_sql_on_sqlite.py"

sys.path.append(str(SRC_DIR))

from chart_agent import generate_chart_for_result
from chat_store import (
    append_message,
    get_chat,
    get_or_create_chat,
    init_chat_store,
    list_chats,
    load_chat_memory,
    load_chat_messages,
    update_chat_memory,
)
from langfuse_tracing import (
    conversation_turn_trace,
    create_conversation_trace_id,
    flush_langfuse,
    safe_update_observation,
)
from pipeline import (
    clear_conversation_memory,
    generate_sql_for_question,
    get_conversation_memory,
    record_sql_execution_for_thread,
    restore_conversation_memory,
    summarize_sql_result,
)


def load_sql_runner():
    spec = importlib.util.spec_from_file_location("sql_runner", SQL_RUNNER_PATH)
    sql_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sql_runner)
    return sql_runner


def init_session_state():
    init_chat_store()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None


def reset_conversation():
    clear_conversation_memory(st.session_state.thread_id)
    st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"
    st.session_state.messages = []
    st.session_state.active_chat_id = None


def ensure_active_chat(user_question):
    if st.session_state.active_chat_id:
        return get_chat(st.session_state.active_chat_id)

    trace_id = create_conversation_trace_id(st.session_state.thread_id)
    chat = get_or_create_chat(
        thread_id=st.session_state.thread_id,
        name=user_question,
        langfuse_trace_id=trace_id,
    )
    st.session_state.active_chat_id = chat["id"]
    return chat


def serialize_assistant_message(message):
    stored_message = {
        key: value
        for key, value in message.items()
        if key != "fig"
    }

    if message.get("fig") is not None:
        stored_message["fig_json"] = message["fig"].to_json()

    return stored_message


def deserialize_stored_message(stored_message):
    content = stored_message["content"]

    if stored_message["role"] == "user":
        if isinstance(content, str):
            return {"role": "user", "content": content}

        return {
            "role": "user",
            "content": content.get("content", ""),
        }

    message = dict(content)
    fig_json = message.pop("fig_json", None)

    if fig_json:
        try:
            message["fig"] = pio.from_json(fig_json)
        except ValueError:
            message["fig"] = None

    return message


def persist_memory_snapshot():
    chat_id = st.session_state.active_chat_id
    if not chat_id:
        return

    update_chat_memory(
        chat_id,
        get_conversation_memory(st.session_state.thread_id),
    )


def load_saved_chat(chat_id):
    chat = get_chat(chat_id)
    if not chat:
        return

    st.session_state.active_chat_id = chat["id"]
    st.session_state.thread_id = chat["thread_id"]
    st.session_state.messages = [
        deserialize_stored_message(message)
        for message in load_chat_messages(chat_id)
    ]
    restore_conversation_memory(chat["thread_id"], load_chat_memory(chat_id))


def run_pipeline(user_question, thread_id, chat_name, turn_index):
    with conversation_turn_trace(
        thread_id=thread_id,
        user_question=user_question,
        chat_name=chat_name,
        turn_index=turn_index,
    ) as turn_span:
        try:
            sql_output = generate_sql_for_question(user_question, thread_id=thread_id)
            generated_sql = sql_output.get("SQL")

            if not generated_sql:
                record_sql_execution_for_thread(thread_id, None)
                safe_update_observation(
                    turn_span,
                    output={
                        "sql_output": sql_output,
                        "sql_execution": summarize_sql_result(None),
                        "chart": None,
                    },
                )
                return sql_output, None, None, None, None, None

            sql_runner = load_sql_runner()
            sql_result = sql_runner.run_query(generated_sql)
            record_sql_execution_for_thread(thread_id, sql_result)

            if isinstance(sql_result, str):
                safe_update_observation(
                    turn_span,
                    output={
                        "sql_output": sql_output,
                        "sql_execution": summarize_sql_result(sql_result),
                        "chart": None,
                    },
                    level="ERROR",
                    status_message=sql_result,
                )
                return sql_output, sql_result, None, None, None, None

            try:
                fig, chart_plan, chart_path = generate_chart_for_result(
                    user_question=sql_output.get("Resolved_Question") or user_question,
                    chart_hint=sql_output.get("Chart"),
                    sql_result=sql_result,
                )
                chart_error = None
            except Exception as e:
                fig, chart_plan, chart_path = None, None, None
                chart_error = str(e)

            safe_update_observation(
                turn_span,
                output={
                    "sql_output": sql_output,
                    "sql_execution": summarize_sql_result(sql_result),
                    "chart": {
                        "plan": chart_plan,
                        "path": str(chart_path) if chart_path else None,
                        "error": chart_error,
                    },
                },
                level="WARNING" if chart_error else "DEFAULT",
                status_message=f"Chart generation skipped: {chart_error}" if chart_error else None,
            )
            return sql_output, sql_result, fig, chart_plan, chart_path, chart_error
        except Exception as e:
            safe_update_observation(
                turn_span,
                output={"error": str(e)},
                level="ERROR",
                status_message=str(e),
            )
            raise
        finally:
            flush_langfuse()


def is_empty_value(value):
    return value is None or value == "" or value == [] or value == {}


def show_sql_generation_output(sql_output):
    st.subheader("SQL Generation Output")

    non_sql_keys = [key for key in sql_output.keys() if key != "SQL"]
    if not non_sql_keys:
        st.info("No non-SQL generation fields were returned.")
        return

    for key in non_sql_keys:
        value = sql_output.get(key)
        st.markdown(f"**{key}**")

        if is_empty_value(value):
            st.info("Empty")
        elif isinstance(value, (dict, list)):
            st.json(value)
        else:
            st.write(value)


def render_assistant_message(message):
    if message.get("error"):
        st.error(message["error"])
        return

    sql_output = message["sql_output"]
    sql_result = message["sql_result"]
    fig = message.get("fig")
    chart_plan = message.get("chart_plan")
    chart_path = message.get("chart_path")
    chart_error = message.get("chart_error")

    show_sql_generation_output(sql_output)

    generated_sql = sql_output.get("SQL")
    if generated_sql:
        st.subheader("Generated SQL")
        st.code(generated_sql, language="sql")

    st.subheader("SQL Running Result")
    if sql_result is None:
        st.info("SQL was not generated, so query execution was skipped.")
    elif isinstance(sql_result, str):
        st.error(sql_result)
    else:
        st.dataframe(sql_result, use_container_width=True)

    if fig is not None:
        st.subheader("Chart")
        st.plotly_chart(fig, use_container_width=True)

        if chart_plan:
            with st.expander("Chart Agent Plan"):
                st.json(chart_plan)

        if chart_path:
            st.caption(f"Stored chart: {chart_path}")
    elif chart_error:
        st.warning(f"Chart generation skipped: {chart_error}")

    with st.expander("Raw SQL Running Result"):
        st.code(json.dumps(sql_result, indent=2, default=str), language="json")


st.set_page_config(page_title="NL to SQL", layout="wide")
init_session_state()

st.title("NL to SQL")

with st.sidebar:
    if st.button("New conversation", type="secondary"):
        reset_conversation()
        st.rerun()

    memory_turns = get_conversation_memory(st.session_state.thread_id)
    st.caption(f"Memory turns: {len(memory_turns)}")

    saved_chats = list_chats()
    if saved_chats:
        chat_lookup = {chat["id"]: chat for chat in saved_chats}
        selected_chat_id = st.selectbox(
            "Saved chats",
            options=["", *chat_lookup.keys()],
            format_func=lambda chat_id: "Select a chat" if not chat_id else chat_lookup[chat_id]["name"],
        )

        if st.button("Load selected chat", disabled=not selected_chat_id):
            load_saved_chat(selected_chat_id)
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message)

submitted_question = st.chat_input("Ask a question about the AP data")

if submitted_question:
    user_question = submitted_question.strip()

    if not user_question:
        st.warning("Please enter a question.")
    else:
        user_message = {"role": "user", "content": user_question}
        chat = ensure_active_chat(user_question)
        append_message(chat["id"], "user", user_message)
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL, running query, and planning chart..."):
                try:
                    (
                        sql_output,
                        sql_result,
                        fig,
                        chart_plan,
                        chart_path,
                        chart_error,
                    ) = run_pipeline(
                        user_question=user_question,
                        thread_id=st.session_state.thread_id,
                        chat_name=chat["name"],
                        turn_index=len([
                            message
                            for message in st.session_state.messages
                            if message["role"] == "user"
                        ]),
                    )
                    assistant_message = {
                        "role": "assistant",
                        "sql_output": sql_output,
                        "sql_result": sql_result,
                        "fig": fig,
                        "chart_plan": chart_plan,
                        "chart_path": str(chart_path) if chart_path else None,
                        "chart_error": chart_error,
                    }
                except Exception as e:
                    assistant_message = {
                        "role": "assistant",
                        "error": f"Pipeline failed: {e}",
                    }

            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)
            append_message(
                chat["id"],
                "assistant",
                serialize_assistant_message(assistant_message),
            )
            persist_memory_snapshot()
