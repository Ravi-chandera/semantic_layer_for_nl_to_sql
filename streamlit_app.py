import importlib.util
import json
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
SQL_RUNNER_PATH = SRC_DIR / "02_run_sql_on_sqlite.py"

sys.path.append(str(SRC_DIR))

from chart_agent import generate_chart_for_result
from benchmark_store import (
    BENCHMARK_QUESTIONS,
    append_benchmark_record,
    build_benchmark_record,
    init_benchmark_store,
    new_benchmark_run_id,
    utc_now as benchmark_utc_now,
    write_benchmark_dashboard,
)
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
    get_conversation_memory,
    restore_conversation_memory,
    summarize_sql_result,
)
from analysis_workflow import run_ai_native_analysis


@lru_cache(maxsize=1)
def load_sql_runner():
    spec = importlib.util.spec_from_file_location("sql_runner", SQL_RUNNER_PATH)
    sql_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sql_runner)
    return sql_runner


def init_session_state():
    init_chat_store()
    init_benchmark_store()

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
            analysis_result = run_ai_native_analysis(
                user_question,
                thread_id=thread_id,
            )
            sql_output = analysis_result["sql_output"]
            sql_result = analysis_result.get("primary_result")
            analysis = analysis_result.get("analysis") or sql_output.get("Analysis")

            if isinstance(sql_result, str):
                safe_update_observation(
                    turn_span,
                    output={
                        "sql_output": sql_output,
                        "sql_execution": summarize_sql_result(sql_result),
                        "analysis": analysis,
                        "chart": None,
                    },
                    level="ERROR",
                    status_message=sql_result,
                )
                return sql_output, sql_result, None, None, None, None

            if sql_output.get("Requires_Clarification") or sql_output.get("Clarification_Limit_Reached"):
                safe_update_observation(
                    turn_span,
                    output={
                        "sql_output": sql_output,
                        "sql_execution": summarize_sql_result(sql_result),
                        "analysis": analysis,
                        "chart": None,
                    },
                )
                return sql_output, sql_result, None, None, None, None

            try:
                fig, chart_plan, chart_path = generate_chart_for_result(
                    user_question=analysis.get("Resolved_Question") or sql_output.get("Resolved_Question") or user_question,
                    chart_hint=sql_output.get("Chart"),
                    sql_result=sql_result or [],
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
                    "analysis": analysis,
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


def record_benchmark_run(
    *,
    run_id,
    source,
    question,
    started_at,
    ended_at,
    latency_ms,
    sql_output=None,
    sql_result=None,
    thread_id=None,
    chat_id=None,
    category=None,
    expected_capability=None,
    chart_path=None,
    chart_error=None,
    error_message=None,
):
    record = build_benchmark_record(
        run_id=run_id,
        source=source,
        question=question,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        sql_output=sql_output,
        sql_result=sql_result,
        thread_id=thread_id,
        chat_id=chat_id,
        category=category,
        expected_capability=expected_capability,
        chart_path=chart_path,
        chart_error=chart_error,
        error_message=error_message,
    )
    append_benchmark_record(record)
    write_benchmark_dashboard()
    return record


def run_single_benchmark_question(item, run_id, turn_index):
    question = item["question"]
    thread_id = f"benchmark-{uuid.uuid4()}"
    started_at = benchmark_utc_now()
    started_perf = time.perf_counter()
    sql_output = None
    sql_result = None
    chart_path = None
    chart_error = None
    error_message = None

    try:
        (
            sql_output,
            sql_result,
            _fig,
            _chart_plan,
            chart_path,
            chart_error,
        ) = run_pipeline(
            user_question=question,
            thread_id=thread_id,
            chat_name=f"Benchmark: {item['category']}",
            turn_index=turn_index,
        )
    except Exception as e:
        error_message = str(e)

    ended_at = benchmark_utc_now()
    latency_ms = (time.perf_counter() - started_perf) * 1000

    return record_benchmark_run(
        run_id=run_id,
        source="benchmark_suite",
        question=question,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=latency_ms,
        sql_output=sql_output,
        sql_result=sql_result,
        thread_id=thread_id,
        category=item["category"],
        expected_capability=item["expected_capability"],
        chart_path=chart_path,
        chart_error=chart_error,
        error_message=error_message,
    )


def is_empty_value(value):
    return value is None or value == "" or value == [] or value == {}


def render_kv_list(items, empty_message="None"):
    if not items:
        st.info(empty_message)
        return

    for item in items:
        st.markdown(f"- {item}")


def render_definition(definition):
    if not definition:
        return

    metric = definition.get("metric")
    description = definition.get("description")
    sql = definition.get("sql")
    filters = definition.get("filters")
    unit = definition.get("result_unit")

    st.markdown(f"**{metric}**")
    if description:
        st.write(description)
    details = []
    if sql:
        details.append(f"Formula: `{sql}`")
    if filters:
        details.append(f"Filter: `{filters}`")
    if unit:
        details.append(f"Unit: `{unit}`")
    render_kv_list(details)


def render_period_comparison(period_comparison):
    if not period_comparison or period_comparison.get("status") != "ok":
        return

    current = period_comparison["current_period"]
    previous = period_comparison["previous_period"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"{current['label']} revenue",
            f"INR {current['value']:,.2f}",
            f"{period_comparison['absolute_change']:,.2f}",
        )
    with col2:
        st.metric(
            f"{previous['label']} revenue",
            f"INR {previous['value']:,.2f}",
        )
    with col3:
        percent_change = period_comparison.get("percent_change")
        st.metric(
            "MoM change",
            "n/a" if percent_change is None else f"{percent_change:,.1f}%",
        )


def render_citations(citations):
    citations = citations or {}
    metric_defs = citations.get("metrics") or []
    tables = citations.get("tables") or []
    columns = citations.get("columns") or []

    if metric_defs:
        st.markdown("**Metrics**")
        for metric in metric_defs:
            if metric:
                st.markdown(f"- `{metric.get('metric')}`: {metric.get('description')}")

    if tables:
        st.markdown("**Tables**")
        for table in tables:
            st.markdown(f"- `{table.get('table')}`: {table.get('description')}")

    if columns:
        st.markdown("**Columns**")
        for column in columns:
            st.markdown(f"- `{column.get('column')}`: {column.get('description')}")


def render_evidence(evidence_items):
    if not evidence_items:
        st.info("No evidence queries were run.")
        return

    for item in evidence_items:
        status = item.get("status", "unknown")
        label = f"{item.get('name', 'evidence')} - {status}"
        with st.expander(label, expanded=False):
            st.write(item.get("purpose"))
            checks = item.get("checks") or []
            if checks:
                st.markdown("**Checks**")
                for check in checks:
                    st.markdown(
                        f"- `{check.get('status')}` {check.get('name')}: {check.get('detail')}"
                    )

            rows = item.get("result_preview") or []
            if rows:
                st.dataframe(rows, use_container_width=True)
            elif item.get("error"):
                st.error(item["error"])
            else:
                st.info("No rows returned.")

            st.markdown("**Supporting SQL**")
            st.code(item.get("sql") or "", language="sql")


def render_next_queries(analysis):
    backed_suggestions = analysis.get("Suggested_Next_Query_Evidence") or []
    if backed_suggestions:
        for item in backed_suggestions:
            st.markdown(f"**{item.get('question')}**")
            if item.get("why"):
                st.caption(item["why"])

            supporting_facts = item.get("supporting_facts") or {}
            if supporting_facts:
                st.json(supporting_facts)

            source = item.get("source_evidence")
            tables = item.get("tables") or []
            if source or tables:
                st.caption(
                    "Backed by "
                    + (f"`{source}`" if source else "evidence")
                    + (f" using {', '.join(f'`{table}`' for table in tables)}" if tables else "")
                )
        return

    render_kv_list(
        analysis.get("Suggested_Next_Queries") or [],
        empty_message="No data-backed next queries were generated from the current evidence.",
    )


def render_analysis_output(sql_output, sql_result):
    analysis = sql_output.get("Analysis")
    if not analysis:
        return False

    st.subheader("Answer")
    st.write(analysis.get("Executive_Answer") or "No answer was returned.")

    clarification = analysis.get("Clarification") or {}
    if clarification.get("needed"):
        st.subheader("Clarification Needed")
        st.info(clarification.get("question") or sql_output.get("Clarification_Question"))
        return True

    render_period_comparison(analysis.get("Period_Comparison"))

    tabs = st.tabs([
        "Assumptions",
        "Evidence",
        "Citations",
        "Confidence",
        "Next",
    ])

    with tabs[0]:
        st.markdown("**Definitions**")
        definitions = analysis.get("Definitions") or []
        if definitions:
            for definition in definitions:
                render_definition(definition)
        else:
            st.info("No semantic metric definition was required.")

        st.markdown("**Assumptions**")
        render_kv_list(analysis.get("Assumptions") or [])

        anomalies = analysis.get("Anomalies") or []
        if anomalies:
            st.markdown("**Anomaly Checks**")
            for anomaly in anomalies:
                st.markdown(f"- `{anomaly.get('severity')}` {anomaly.get('message')}")

    with tabs[1]:
        render_evidence(analysis.get("Evidence") or [])
        generated_sql = analysis.get("Generated_SQL_Evidence")
        if generated_sql:
            with st.expander("Original generated SQL evidence", expanded=False):
                if generated_sql.get("error"):
                    st.error(generated_sql["error"])
                elif generated_sql.get("result_preview"):
                    st.dataframe(generated_sql["result_preview"], use_container_width=True)
                st.code(generated_sql.get("sql") or "", language="sql")

    with tabs[2]:
        render_citations(analysis.get("Citations"))

    with tabs[3]:
        confidence = analysis.get("Confidence") or {}
        st.metric(
            "Confidence",
            confidence.get("level", "unknown"),
            confidence.get("score"),
        )
        render_kv_list(confidence.get("reasons") or [])
        st.markdown("**Limitations**")
        render_kv_list(analysis.get("Limitations") or [])

    with tabs[4]:
        render_next_queries(analysis)

    if sql_result is not None and not isinstance(sql_result, str):
        with st.expander("Primary result table", expanded=False):
            st.dataframe(sql_result, use_container_width=True)

    return True


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

    rendered_analysis = render_analysis_output(sql_output, sql_result)

    if sql_output.get("Requires_Clarification"):
        if not rendered_analysis:
            st.subheader("Clarification Needed")
            st.info(sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions"))
        return

    if sql_output.get("Clarification_Limit_Reached"):
        if not rendered_analysis:
            st.subheader("Clarification Limit Reached")
            st.warning(sql_output.get("Assumptions") or "The question is still underspecified.")
        return

    if not rendered_analysis:
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

    with st.expander("Raw Analysis Payload"):
        st.code(json.dumps(sql_output.get("Analysis") or sql_output, indent=2, default=str), language="json")


def render_chat_tab():
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
                run_id = new_benchmark_run_id()
                started_at = benchmark_utc_now()
                started_perf = time.perf_counter()
                sql_output = None
                sql_result = None
                chart_path = None
                chart_error = None
                error_message = None

                with st.spinner("Running analysis, checking evidence, and preparing the answer..."):
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
                        error_message = str(e)
                        assistant_message = {
                            "role": "assistant",
                            "error": f"Pipeline failed: {e}",
                        }

                ended_at = benchmark_utc_now()
                latency_ms = (time.perf_counter() - started_perf) * 1000
                record_benchmark_run(
                    run_id=run_id,
                    source="chat",
                    question=user_question,
                    started_at=started_at,
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    sql_output=sql_output,
                    sql_result=sql_result,
                    thread_id=st.session_state.thread_id,
                    chat_id=chat["id"],
                    chart_path=chart_path,
                    chart_error=chart_error,
                    error_message=error_message,
                )

            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)
            append_message(
                chat["id"],
                "assistant",
                serialize_assistant_message(assistant_message),
            )
            persist_memory_snapshot()


def render_benchmark_tab():
    st.subheader("Benchmark Dashboard")
    st.caption(
        "Records are append-only in data/benchmark_results.db. "
        "The HTML dashboard is regenerated from those records."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        run_suite = st.button("Run fixed benchmark suite", type="primary")
    with col2:
        refresh_dashboard = st.button("Refresh dashboard")

    if run_suite:
        run_id = new_benchmark_run_id()
        progress = st.progress(0)
        status = st.empty()
        records = []

        for index, item in enumerate(BENCHMARK_QUESTIONS, start=1):
            status.write(f"Running {index}/{len(BENCHMARK_QUESTIONS)}: {item['question']}")
            records.append(run_single_benchmark_question(item, run_id, index))
            progress.progress(index / len(BENCHMARK_QUESTIONS))

        status.success(f"Appended {len(records)} benchmark records.")

    if refresh_dashboard:
        write_benchmark_dashboard()

    dashboard_path, html_text = write_benchmark_dashboard()
    st.caption(f"Dashboard file: {dashboard_path}")
    components.html(html_text, height=900, scrolling=True)


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

chat_tab, benchmark_tab = st.tabs(["Chat", "Benchmark"])

with chat_tab:
    render_chat_tab()

with benchmark_tab:
    render_benchmark_tab()
