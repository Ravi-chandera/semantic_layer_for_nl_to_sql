import json
import html
import tempfile
import time
import uuid
from datetime import date
from pathlib import Path

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parent
APP_NAME = "Data Investigation Analyst"

from src.chart_agent import generate_chart_for_result
from src.benchmark_store import (
    BENCHMARK_QUESTIONS,
    append_benchmark_record,
    build_benchmark_record,
    init_benchmark_store,
    new_benchmark_run_id,
    utc_now as benchmark_utc_now,
    write_benchmark_dashboard,
)
from src.chat_store import (
    append_message,
    get_chat,
    get_or_create_chat,
    init_chat_store,
    list_chats,
    load_chat_memory,
    load_chat_messages,
    update_chat_memory,
)
from src.langfuse_tracing import (
    conversation_turn_trace,
    create_conversation_trace_id,
    flush_langfuse,
    safe_update_observation,
)
from src.pipeline import (
    clear_conversation_memory,
    get_conversation_memory,
    restore_conversation_memory,
    summarize_sql_result,
)
from src.analysis_workflow import run_ai_native_analysis
from src.dataset_onboarding import (
    ACTIVE_DB_PATH,
    DATASET_MANIFEST_PATH,
    apply_dataset_understanding_to_review,
    build_review_template,
    build_semantic_layer,
    discover_sqlite_dataset,
    generate_dataset_understanding,
    get_active_db_path,
    save_onboarded_dataset,
)
from src.glossary_manager import (
    ambiguity_rule_records_from_layer,
    apply_glossary_records,
    load_semantic_layer,
    metric_records_from_layer,
    save_semantic_layer,
    validate_metric_records,
)
from src.feedback_store import (
    add_feedback,
    init_feedback_store,
    list_feedback,
    summarize_corrections,
)
from src.certified_question_store import (
    init_certified_question_store,
    list_certified_questions,
    save_certified_question,
    set_certified_question_active,
)
from src.data_settings import (
    DEFAULT_DATA_SETTINGS,
    format_currency,
    load_data_settings,
    save_data_settings,
)
from src.answer_modes import (
    ANSWER_MODE_ANALYST,
    ANSWER_MODE_AUDIT,
    ANSWER_MODE_EXECUTIVE,
    ANSWER_MODE_SQL_DEBUG,
    answer_mode_label,
    answer_mode_options,
    normalize_answer_mode,
)
from src.edge_cases import classify_edge_cases
from src.pipeline_semantic_context import clarification_options_from_rule, clear_sql_context_cache


def init_session_state():
    init_chat_store()
    init_benchmark_store()
    init_feedback_store()
    init_certified_question_store()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if "data_settings" not in st.session_state:
        st.session_state.data_settings = load_data_settings()

    if "answer_mode" not in st.session_state:
        st.session_state.answer_mode = ANSWER_MODE_ANALYST


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
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant":
            st.session_state.answer_mode = answer_mode_from_message(message)
            break
    restore_conversation_memory(chat["thread_id"], load_chat_memory(chat_id))


def run_pipeline(user_question, thread_id, chat_name, turn_index, answer_mode):
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
                answer_mode=answer_mode,
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
            answer_mode=ANSWER_MODE_ANALYST,
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


FEEDBACK_CATEGORY_LABELS = {
    "wrong_join": "Wrong join",
    "wrong_metric": "Wrong metric",
    "wrong_date": "Wrong date",
    "missing_filter": "Missing filter",
}


def _analysis_from_sql_output(sql_output):
    if not isinstance(sql_output, dict):
        return {}
    analysis = sql_output.get("Analysis")
    return analysis if isinstance(analysis, dict) else {}


def answer_mode_from_message(message):
    sql_output = message.get("sql_output") or {}
    analysis = _analysis_from_sql_output(sql_output)
    return normalize_answer_mode(
        message.get("answer_mode")
        or sql_output.get("Answer_Mode")
        or analysis.get("Answer_Mode")
    )


def feedback_payload_from_message(message):
    sql_output = message.get("sql_output") or {}
    analysis = _analysis_from_sql_output(sql_output)
    return {
        "question": (
            message.get("question")
            or sql_output.get("Original_Question")
            or analysis.get("Question")
        ),
        "resolved_question": (
            analysis.get("Resolved_Question")
            or sql_output.get("Resolved_Question")
        ),
        "generated_sql": sql_output.get("SQL"),
        "metrics": sql_output.get("Selected_Metrics") or [],
        "tables": sql_output.get("Selected_Tables") or [],
        "chat_id": message.get("chat_id"),
        "thread_id": message.get("thread_id"),
        "turn_index": message.get("turn_index"),
        "message_id": message.get("message_id"),
    }


def certified_question_payload_from_message(message):
    sql_output = message.get("sql_output") or {}
    analysis = _analysis_from_sql_output(sql_output)
    question = (
        analysis.get("Resolved_Question")
        or sql_output.get("Resolved_Question")
        or message.get("question")
        or sql_output.get("Original_Question")
        or analysis.get("Question")
    )
    return {
        "question": question,
        "approved_sql": sql_output.get("SQL"),
        "source_chat_id": message.get("chat_id"),
        "source_message_id": message.get("message_id"),
    }


def render_certified_question_save_action(message):
    if message.get("error"):
        return

    payload = certified_question_payload_from_message(message)
    if not payload["question"]:
        return

    message_id = message.get("message_id") or str(abs(hash(json.dumps(message, default=str))))
    form_key = f"certified-question-{message_id}"
    default_title = payload["question"][:80]

    with st.expander("Save as certified question", expanded=False):
        with st.form(form_key, clear_on_submit=True):
            title = st.text_input("Title", value=default_title, key=f"{form_key}-title")
            question = st.text_area("Question template", value=payload["question"], key=f"{form_key}-question")
            category = st.text_input("Category", value="General", key=f"{form_key}-category")
            owner = st.text_input("Owner", value="", key=f"{form_key}-owner")
            tags = st.text_input("Tags", value="", help="Comma-separated tags.", key=f"{form_key}-tags")
            approved_sql = st.text_area(
                "Approved SQL",
                value=payload.get("approved_sql") or "",
                key=f"{form_key}-approved-sql",
            )
            notes = st.text_area(
                "Admin notes",
                value="",
                placeholder="Optional business context, caveats, or approval notes.",
                key=f"{form_key}-notes",
            )
            certified = st.checkbox("Certified", value=True, key=f"{form_key}-certified")
            active = st.checkbox("Active quick action", value=True, key=f"{form_key}-active")
            submitted = st.form_submit_button("Save certified question")

        if submitted:
            try:
                saved = save_certified_question(
                    title=title,
                    question=question,
                    category=category,
                    owner=owner,
                    tags=tags,
                    approved_sql=approved_sql,
                    notes=notes,
                    certified=certified,
                    active=active,
                    source_chat_id=payload.get("source_chat_id"),
                    source_message_id=payload.get("source_message_id"),
                )
                st.success(f"Saved certified question: {saved['title']}")
            except Exception as e:
                st.error(f"Could not save certified question: {e}")


def render_result_feedback(message):
    if message.get("error"):
        return

    message_id = message.get("message_id") or str(abs(hash(json.dumps(message, default=str))))
    form_key = f"result-feedback-{message_id}"

    with st.expander("Does this look right?", expanded=False):
        with st.form(form_key, clear_on_submit=True):
            sentiment = st.radio(
                "Result validation",
                options=["up", "down"],
                format_func=lambda value: "Thumbs up" if value == "up" else "Thumbs down",
                horizontal=True,
                key=f"{form_key}-sentiment",
            )
            selected_labels = st.multiselect(
                "What needs correction?",
                options=list(FEEDBACK_CATEGORY_LABELS.keys()),
                format_func=lambda value: FEEDBACK_CATEGORY_LABELS[value],
                disabled=sentiment == "up",
                key=f"{form_key}-categories",
            )
            note = st.text_area(
                "Optional note",
                placeholder="Example: use invoice_date instead of due_date, or join vendors through purchase_orders.",
                key=f"{form_key}-note",
            )
            submitted = st.form_submit_button("Save feedback")

        if submitted:
            try:
                add_feedback(
                    sentiment=sentiment,
                    categories=selected_labels if sentiment == "down" else [],
                    note=note.strip() or None,
                    **feedback_payload_from_message(message),
                )
                clear_sql_context_cache()
                st.success("Feedback saved for future semantic corrections.")
            except Exception as e:
                st.error(f"Could not save feedback: {e}")


def render_period_comparison(period_comparison):
    if not period_comparison or period_comparison.get("status") != "ok":
        return

    data_settings = load_data_settings()
    current = period_comparison["current_period"]
    previous = period_comparison["previous_period"]
    metric_label = str(period_comparison.get("metric") or "value").replace("_", " ")
    unit = period_comparison.get("unit")
    format_value = (
        (lambda value: format_currency(value, data_settings))
        if unit and str(unit).upper() == data_settings.get("default_currency")
        else (lambda value: "n/a" if value is None else f"{float(value):,.2f}")
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"{current['label']} {metric_label}",
            format_value(current["value"]),
            format_value(period_comparison["absolute_change"]),
        )
    with col2:
        st.metric(
            f"{previous['label']} {metric_label}",
            format_value(previous["value"]),
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


def render_evidence(evidence_items, expanded=False, show_sql=True):
    if not evidence_items:
        st.info("No evidence queries were run.")
        return

    for item in evidence_items:
        status = item.get("status", "unknown")
        row_count = item.get("row_count")
        row_text = f" | rows: {row_count}" if row_count is not None else ""
        label = f"{item.get('name', 'evidence')} - {status}{row_text}"
        with st.expander(label, expanded=expanded):
            st.write(item.get("purpose"))
            if item.get("tables"):
                st.caption("Tables: " + ", ".join(f"`{table}`" for table in item.get("tables", [])))
            if item.get("columns"):
                st.caption("Columns: " + ", ".join(f"`{column}`" for column in item.get("columns", [])))
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

            if show_sql:
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


def build_investigation_report(analysis):
    confidence = analysis.get("Confidence") or {}
    confidence_lines = [
        f"{confidence.get('badge') or confidence.get('level', 'unknown')} ({confidence.get('score', 'n/a')})",
        confidence.get("summary") or "",
    ]
    for reason in confidence.get("reason_codes") or []:
        confidence_lines.append(f"- {reason.get('message')} [{reason.get('code')}]")

    lines = [
        f"# {APP_NAME} Report",
        "",
        f"Question: {analysis.get('Question')}",
        f"Resolved question: {analysis.get('Resolved_Question')}",
        "",
        "## Business Answer",
        analysis.get("Executive_Answer") or "No answer was returned.",
        "",
        "## Confidence",
        *[line for line in confidence_lines if line],
        "",
        "## Evidence",
    ]

    for item in analysis.get("Evidence") or []:
        lines.extend(
            [
                f"- {item.get('name')} ({item.get('status')}): {item.get('purpose')}",
                f"  Rows: {item.get('row_count')}",
                "  SQL:",
                "```sql",
                item.get("sql") or "",
                "```",
            ]
        )

    lines.extend(["", "## Limitations"])
    for limitation in analysis.get("Limitations") or []:
        lines.append(f"- {limitation}")

    return "\n".join(lines)


def render_confidence_badge(confidence):
    confidence = confidence or {}
    level = str(confidence.get("level") or "unknown").lower()
    label = confidence.get("badge") or level.title()
    score = confidence.get("score", "n/a")
    summary = confidence.get("summary")
    colors = {
        "high": ("#0f766e", "#ecfdf5", "#99f6e4"),
        "medium": ("#a16207", "#fffbeb", "#fde68a"),
        "low": ("#b91c1c", "#fef2f2", "#fecaca"),
    }
    text_color, bg_color, border_color = colors.get(level, ("#475569", "#f8fafc", "#cbd5e1"))
    st.markdown(
        (
            "<div style='display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;"
            "margin:0.35rem 0 0.4rem 0;'>"
            f"<span style='font-weight:700;color:{text_color};background:{bg_color};"
            f"border:1px solid {border_color};border-radius:999px;padding:0.18rem 0.65rem;'>"
            f"{html.escape(str(label))} confidence</span>"
            f"<span style='color:#475569;'>Score: {html.escape(str(score))}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if summary:
        st.caption(summary)

    reason_codes = confidence.get("reason_codes") or []
    if reason_codes:
        for reason in reason_codes[:4]:
            st.markdown(f"- {reason.get('message')}")


def clarification_options_from_sql_output(sql_output):
    options = sql_output.get("Clarification_Options") or []
    if options:
        return options

    decision = sql_output.get("Clarification_Decision") or {}
    matched_rule = decision.get("matched_rule")
    if not matched_rule:
        return []

    try:
        semantic_layer = load_semantic_layer(ROOT_DIR / "data" / "semantic_layer.json")
    except Exception:
        return []

    rule = semantic_layer.get("ambiguity_rules", {}).get(matched_rule)
    if not rule:
        return []

    return clarification_options_from_rule(matched_rule, rule)


def clarification_resolution(option, remember_default=False):
    resolution_text = option.get("resolution_text") or option.get("label") or ""
    if remember_default:
        return f"{resolution_text} Use this as my default next time."
    return resolution_text


def render_clickable_clarifications(sql_output, question):
    if not question:
        return

    options = clarification_options_from_sql_output(sql_output)
    if not options:
        return

    decision = sql_output.get("Clarification_Decision") or {}
    is_entity_disambiguation = decision.get("matched_rule") == "entity_disambiguation"
    remember_default = False
    if not is_entity_disambiguation:
        remember_key = f"clarification-default-{sql_output.get('Clarification_Attempts', 0)}-{question}"
        remember_default = st.checkbox("Use this as default next time", key=remember_key)
    st.caption("Choose one to continue:")
    cols = st.columns(min(len(options), 3))
    for index, option in enumerate(options):
        col = cols[index % len(cols)]
        with col:
            with st.container(border=True):
                st.markdown(f"**{option.get('label') or option.get('id')}**")
                detail = option.get("detail")
                if detail:
                    st.caption(detail)
                if st.button("Choose", key=f"clarification-{option.get('id')}-{index}"):
                    st.session_state.pending_clarification = clarification_resolution(
                        option,
                        remember_default=remember_default,
                    )
                    st.rerun()


def render_edge_cases(edge_cases):
    if not edge_cases:
        return

    st.subheader("Result Notes")
    for edge_case in edge_cases:
        with st.container(border=True):
            st.markdown(f"**{edge_case.get('title') or edge_case.get('type')}**")
            if edge_case.get("explanation"):
                st.write(edge_case["explanation"])
            next_actions = edge_case.get("next_actions") or []
            if next_actions:
                st.markdown("**Next action**")
                for action in next_actions:
                    st.markdown(f"- {action}")


def render_generated_sql_evidence(analysis):
    generated_sql = analysis.get("Generated_SQL_Evidence")
    if not generated_sql:
        return

    with st.expander("Original generated SQL evidence", expanded=False):
        if generated_sql.get("error"):
            st.error(generated_sql["error"])
        elif generated_sql.get("result_preview"):
            st.dataframe(generated_sql["result_preview"], use_container_width=True)
        st.code(generated_sql.get("sql") or "", language="sql")


def render_primary_result_table(sql_result, expanded=False):
    if sql_result is None or isinstance(sql_result, str):
        return

    with st.expander("Primary result table", expanded=expanded):
        st.dataframe(sql_result, use_container_width=True)


def render_source_selection(sql_output, analysis):
    tables = sql_output.get("Selected_Tables") or []
    metrics = sql_output.get("Selected_Metrics") or []
    citations = analysis.get("Citations") or {}

    if tables:
        st.markdown("**Selected tables**")
        render_kv_list([f"`{table}`" for table in tables])
    elif citations.get("tables"):
        st.markdown("**Cited tables**")
        for table in citations.get("tables") or []:
            st.markdown(f"- `{table.get('table')}`: {table.get('description')}")

    if metrics:
        st.markdown("**Selected metrics**")
        render_kv_list([f"`{metric}`" for metric in metrics])


def render_debug_metadata(sql_output):
    debug_payload = {
        "cache": {
            "hit": sql_output.get("Cache_Hit"),
            "strategy": sql_output.get("Cache_Strategy"),
            "score": sql_output.get("Cache_Score"),
        },
        "clarification": {
            "requires_clarification": sql_output.get("Requires_Clarification"),
            "question": sql_output.get("Clarification_Question"),
            "attempts": sql_output.get("Clarification_Attempts"),
            "limit_reached": sql_output.get("Clarification_Limit_Reached"),
            "decision": sql_output.get("Clarification_Decision"),
            "options": sql_output.get("Clarification_Options"),
        },
        "entity_search": sql_output.get("Entity_Search"),
        "memory": {
            "is_followup": sql_output.get("Is_Followup"),
            "memory_used": sql_output.get("Memory_Used"),
        },
        "answer_mode": {
            "key": sql_output.get("Answer_Mode"),
            "label": sql_output.get("Answer_Mode_Label"),
        },
    }
    st.json(debug_payload)


def render_executive_summary_output(sql_output, sql_result, analysis):
    limitations = analysis.get("Limitations") or []
    if limitations:
        st.markdown("**Watchouts**")
        render_kv_list(limitations[:4])

    assumptions = analysis.get("Assumptions") or []
    if assumptions:
        with st.expander("Assumptions and details", expanded=False):
            render_kv_list(assumptions)
            render_source_selection(sql_output, analysis)

    if analysis.get("Evidence") or analysis.get("Generated_SQL_Evidence"):
        with st.expander("Evidence and SQL", expanded=False):
            render_evidence(analysis.get("Evidence") or [], expanded=False, show_sql=False)
            render_generated_sql_evidence(analysis)

    render_primary_result_table(sql_result, expanded=False)


def render_audit_evidence_output(sql_output, sql_result, analysis):
    st.subheader("Evidence Trail")
    render_source_selection(sql_output, analysis)

    st.markdown("**Assumptions**")
    render_kv_list(analysis.get("Assumptions") or [])

    st.markdown("**Limitations**")
    render_kv_list(analysis.get("Limitations") or [])

    confidence = analysis.get("Confidence") or {}
    reason_codes = confidence.get("reason_codes") or []
    if reason_codes:
        st.markdown("**Confidence reasons**")
        for reason in reason_codes:
            st.markdown(f"- `{reason.get('code')}` {reason.get('message')}")

    render_evidence(analysis.get("Evidence") or [], expanded=True, show_sql=True)
    render_generated_sql_evidence(analysis)
    render_primary_result_table(sql_result, expanded=False)


def render_sql_debug_output(sql_output, sql_result, analysis):
    render_confidence_badge(analysis.get("Confidence") or {})

    generated_sql = sql_output.get("SQL")
    if generated_sql:
        st.subheader("Generated SQL")
        st.code(generated_sql, language="sql")
    else:
        st.info("No generated SQL is available for this turn.")

    st.subheader("SQL Running Result")
    if sql_result is None:
        st.info("SQL was not generated, so query execution was skipped.")
    elif isinstance(sql_result, str):
        st.error(sql_result)
    else:
        st.dataframe(sql_result, use_container_width=True)

    st.subheader("Debug Metadata")
    render_debug_metadata(sql_output)

    with st.expander("Raw SQL output payload", expanded=True):
        st.code(json.dumps(sql_output, indent=2, default=str), language="json")


def render_analysis_output(sql_output, sql_result, chart_error=None, answer_mode=ANSWER_MODE_ANALYST):
    answer_mode = normalize_answer_mode(answer_mode)
    analysis = sql_output.get("Analysis")
    if not analysis:
        return False

    st.caption(f"Answer mode: {answer_mode_label(answer_mode)}")
    st.subheader("Business Answer")
    st.write(analysis.get("Executive_Answer") or "No answer was returned.")
    render_edge_cases(
        classify_edge_cases(
            question=analysis.get("Question") or analysis.get("Resolved_Question"),
            sql_output=sql_output,
            sql_result=sql_result,
            analysis=analysis,
            chart_error=chart_error,
        )
    )

    clarification = analysis.get("Clarification") or {}
    if clarification.get("needed"):
        st.subheader("Clarification Needed")
        question = clarification.get("question") or sql_output.get("Clarification_Question")
        st.info(question)
        render_clickable_clarifications(sql_output, question)
        return True

    render_period_comparison(analysis.get("Period_Comparison"))
    confidence = analysis.get("Confidence") or {}
    render_confidence_badge(confidence)

    if answer_mode == ANSWER_MODE_EXECUTIVE:
        render_executive_summary_output(sql_output, sql_result, analysis)
        return True

    if answer_mode == ANSWER_MODE_AUDIT:
        render_audit_evidence_output(sql_output, sql_result, analysis)
        st.download_button(
            "Export investigation report",
            data=build_investigation_report(analysis),
            file_name="ap_investigation_report.md",
            mime="text/markdown",
        )
        return True

    if answer_mode == ANSWER_MODE_SQL_DEBUG:
        render_sql_debug_output(sql_output, sql_result, analysis)
        return True

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
        render_generated_sql_evidence(analysis)

    with tabs[2]:
        render_citations(analysis.get("Citations"))

    with tabs[3]:
        confidence = analysis.get("Confidence") or {}
        st.metric(
            "Confidence",
            confidence.get("level", "unknown"),
            confidence.get("score"),
        )
        reason_codes = confidence.get("reason_codes") or []
        if reason_codes:
            st.markdown("**Reason codes**")
            for reason in reason_codes:
                st.markdown(f"- `{reason.get('code')}` {reason.get('message')}")
        st.markdown("**Detailed reasons**")
        render_kv_list(confidence.get("reasons") or [])
        st.markdown("**Limitations**")
        render_kv_list(analysis.get("Limitations") or [])

    with tabs[4]:
        render_next_queries(analysis)

    st.download_button(
        "Export investigation report",
        data=build_investigation_report(analysis),
        file_name="ap_investigation_report.md",
        mime="text/markdown",
    )

    render_primary_result_table(sql_result, expanded=False)

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

    answer_mode = answer_mode_from_message(message)
    sql_output = message["sql_output"]
    sql_result = message["sql_result"]
    fig = message.get("fig")
    chart_plan = message.get("chart_plan")
    chart_path = message.get("chart_path")
    chart_error = message.get("chart_error")

    rendered_analysis = render_analysis_output(
        sql_output,
        sql_result,
        chart_error,
        answer_mode=answer_mode,
    )

    if sql_output.get("Requires_Clarification"):
        if not rendered_analysis:
            st.subheader("Clarification Needed")
            question = sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions")
            st.info(question)
            render_clickable_clarifications(sql_output, question)
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
        render_edge_cases(
            classify_edge_cases(
                question=sql_output.get("Original_Question") or sql_output.get("Resolved_Question"),
                sql_output=sql_output,
                sql_result=sql_result,
                analysis=sql_output.get("Analysis"),
                chart_error=chart_error,
            )
        )
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

    if answer_mode == ANSWER_MODE_SQL_DEBUG:
        with st.expander("Raw Analysis Payload", expanded=True):
            st.code(json.dumps(sql_output.get("Analysis") or sql_output, indent=2, default=str), language="json")
    elif answer_mode != ANSWER_MODE_EXECUTIVE:
        with st.expander("Raw Analysis Payload"):
            st.code(json.dumps(sql_output.get("Analysis") or sql_output, indent=2, default=str), language="json")

    render_result_feedback(message)
    render_certified_question_save_action(message)


def render_certified_quick_actions():
    templates = list_certified_questions(active_only=True, certified_only=True, limit=12)
    if not templates:
        return

    st.markdown("**Certified questions**")
    categories = ["All"]
    for template in templates:
        category = template.get("category") or "General"
        if category not in categories:
            categories.append(category)

    selected_category = st.selectbox(
        "Filter certified questions",
        options=categories,
        label_visibility="collapsed",
        key="certified-question-category-filter",
    )
    visible_templates = [
        template
        for template in templates
        if selected_category == "All" or (template.get("category") or "General") == selected_category
    ]

    columns = st.columns(3)
    for index, template in enumerate(visible_templates):
        with columns[index % 3]:
            if st.button(
                template["title"],
                key=f"launch-certified-question-{template['id']}",
                help=template["question"],
                use_container_width=True,
            ):
                st.session_state.pending_template_question = template["question"]
                st.rerun()


def render_chat_tab():
    selected_answer_mode = st.selectbox(
        "Answer mode",
        options=answer_mode_options(),
        index=answer_mode_options().index(normalize_answer_mode(st.session_state.answer_mode)),
        format_func=answer_mode_label,
        help="Choose how new assistant answers are presented. Saved messages keep the mode used when they were created.",
        key="answer_mode",
    )
    st.caption(
        {
            ANSWER_MODE_EXECUTIVE: "Lead with the answer, confidence, and watchouts.",
            ANSWER_MODE_ANALYST: "Show the standard analyst view with result tables, chart, and evidence tabs.",
            ANSWER_MODE_AUDIT: "Lead with evidence, assumptions, limitations, validation checks, and row counts.",
            ANSWER_MODE_SQL_DEBUG: "Expose SQL, raw payloads, cache, clarification, and entity metadata.",
        }[normalize_answer_mode(selected_answer_mode)]
    )

    render_certified_quick_actions()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_message(message)

    submitted_question = st.session_state.pop("pending_clarification", None)
    if submitted_question is None:
        submitted_question = st.session_state.pop("pending_template_question", None)
    if submitted_question is None:
        submitted_question = st.chat_input("Ask a question about the active dataset")

    if submitted_question:
        user_question = submitted_question.strip()

        if not user_question:
            st.warning("Please enter a question.")
        else:
            answer_mode = normalize_answer_mode(st.session_state.answer_mode)
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
                            answer_mode=answer_mode,
                        )
                        assistant_message = {
                            "role": "assistant",
                            "message_id": str(uuid.uuid4()),
                            "question": user_question,
                            "answer_mode": answer_mode,
                            "answer_mode_label": answer_mode_label(answer_mode),
                            "chat_id": chat["id"],
                            "thread_id": st.session_state.thread_id,
                            "turn_index": len([
                                message
                                for message in st.session_state.messages
                                if message["role"] == "user"
                            ]) - 1,
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
                            "message_id": str(uuid.uuid4()),
                            "question": user_question,
                            "answer_mode": answer_mode,
                            "answer_mode_label": answer_mode_label(answer_mode),
                            "chat_id": chat["id"],
                            "thread_id": st.session_state.thread_id,
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


def render_corrections_tab():
    st.subheader("Semantic Corrections")
    st.caption(
        "Analyst result feedback is append-only in data/result_feedback.db and is summarized into future SQL context."
    )

    summary = summarize_corrections()
    col1, col2 = st.columns(2)
    col1.metric("Feedback items", summary["total_feedback"])
    col2.metric("Corrections", summary["negative_feedback"])

    if summary["by_category"]:
        st.markdown("**Corrections by category**")
        st.dataframe(
            [
                {"category": FEEDBACK_CATEGORY_LABELS.get(category, category), "count": count}
                for category, count in summary["by_category"].items()
            ],
            use_container_width=True,
        )

    if summary["by_metric"] or summary["by_table"]:
        metric_col, table_col = st.columns(2)
        with metric_col:
            st.markdown("**By metric**")
            st.dataframe(
                [{"metric": metric, "count": count} for metric, count in summary["by_metric"].items()],
                use_container_width=True,
            )
        with table_col:
            st.markdown("**By table**")
            st.dataframe(
                [{"table": table, "count": count} for table, count in summary["by_table"].items()],
                use_container_width=True,
            )

    records = list_feedback(limit=50)
    if not records:
        st.info("No result feedback has been saved yet.")
        return

    st.markdown("**Recent feedback**")
    st.dataframe(
        [
            {
                "created_at": record["created_at"],
                "sentiment": record["sentiment"],
                "categories": ", ".join(
                    FEEDBACK_CATEGORY_LABELS.get(category, category)
                    for category in record.get("categories", [])
                ),
                "question": record.get("resolved_question") or record.get("question"),
                "metrics": ", ".join(record.get("metrics") or []),
                "tables": ", ".join(record.get("tables") or []),
                "note": record.get("note"),
            }
            for record in records
        ],
        use_container_width=True,
    )


def render_certified_questions_tab():
    st.subheader("Certified Questions")
    st.caption(
        "Reusable business-approved question templates are stored in data/certified_questions.db."
    )

    with st.expander("Create certified question", expanded=False):
        with st.form("create-certified-question", clear_on_submit=True):
            title = st.text_input("Title", key="create-certified-title")
            question = st.text_area("Question", key="create-certified-question-text")
            category = st.text_input("Category", value="General", key="create-certified-category")
            owner = st.text_input("Owner", key="create-certified-owner")
            tags = st.text_input("Tags", help="Comma-separated tags.", key="create-certified-tags")
            approved_sql = st.text_area("Approved SQL", key="create-certified-approved-sql")
            notes = st.text_area("Notes", key="create-certified-notes")
            certified = st.checkbox("Certified", value=True, key="create-certified-certified")
            active = st.checkbox("Active quick action", value=True, key="create-certified-active")
            submitted = st.form_submit_button("Create")

        if submitted:
            try:
                saved = save_certified_question(
                    title=title,
                    question=question,
                    category=category,
                    owner=owner,
                    tags=tags,
                    approved_sql=approved_sql,
                    notes=notes,
                    certified=certified,
                    active=active,
                )
                st.success(f"Created certified question: {saved['title']}")
            except Exception as e:
                st.error(f"Could not create certified question: {e}")

    records = list_certified_questions(limit=200)
    if not records:
        st.info("No certified questions have been saved yet.")
        return

    st.dataframe(
        [
            {
                "title": record["title"],
                "category": record.get("category") or "General",
                "owner": record.get("owner"),
                "certified": record["certified"],
                "active": record["active"],
                "tags": ", ".join(record.get("tags") or []),
                "question": record["question"],
                "updated_at": record["updated_at"],
            }
            for record in records
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Activate or deactivate**")
    record_lookup = {record["id"]: record for record in records}
    selected_id = st.selectbox(
        "Certified question",
        options=list(record_lookup.keys()),
        format_func=lambda question_id: record_lookup[question_id]["title"],
    )
    selected = record_lookup[selected_id]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Activate", disabled=selected["active"]):
            set_certified_question_active(selected_id, True)
            st.rerun()
    with col2:
        if st.button("Deactivate", disabled=not selected["active"]):
            set_certified_question_active(selected_id, False)
            st.rerun()


def editor_records(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return []


def comma_join(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value or ""


def normalize_review(tables, columns, joins, metrics):
    return {
        "tables": [
            {
                "table_name": row.get("table_name"),
                "business_name": row.get("business_name"),
                "synonyms": comma_join(row.get("synonyms")),
            }
            for row in editor_records(tables)
        ],
        "columns": [
            {
                "table_name": row.get("table_name"),
                "column_name": row.get("column_name"),
                "business_name": row.get("business_name"),
                "synonyms": comma_join(row.get("synonyms")),
                "is_metric": bool(row.get("is_metric")),
                "is_sensitive": bool(row.get("is_sensitive")),
            }
            for row in editor_records(columns)
        ],
        "joins": [
            {
                "left_table": row.get("left_table"),
                "left_column": row.get("left_column"),
                "right_table": row.get("right_table"),
                "right_column": row.get("right_column"),
                "approved": bool(row.get("approved")),
                "source": row.get("source") or "reviewed",
            }
            for row in editor_records(joins)
        ],
        "metrics": [
            {
                "metric_name": row.get("metric_name"),
                "description": row.get("description"),
                "sql": row.get("sql"),
                "filters": row.get("filters"),
                "synonyms": comma_join(row.get("synonyms")),
                "result_unit": row.get("result_unit"),
                "tables": comma_join(row.get("tables")),
                "enabled": bool(row.get("enabled", True)),
            }
            for row in editor_records(metrics)
        ],
    }


def render_discovery_preview(discovered):
    st.subheader("Discovered Tables")
    rows = [
        {
            "table": table["table_name"],
            "business_name": table.get("business_name"),
            "rows": table.get("row_count"),
            "columns": len(table.get("columns", [])),
            "primary_key": ", ".join(table.get("primary_keys") or []),
            "foreign_keys": len(table.get("foreign_keys", [])),
        }
        for table in discovered.get("tables", [])
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    selected_table = st.selectbox(
        "Preview table",
        options=[table["table_name"] for table in discovered.get("tables", [])],
    )
    if selected_table:
        table = next(table for table in discovered["tables"] if table["table_name"] == selected_table)
        column_rows = [
            {
                "column": column["name"],
                "type": column.get("type"),
                "metric": column.get("is_metric"),
                "sensitive": column.get("is_sensitive"),
                "samples": ", ".join(str(sample) for sample in column.get("sample_values", [])),
            }
            for column in table.get("columns", [])
        ]
        st.dataframe(column_rows, use_container_width=True, hide_index=True)


def render_onboarding_tab():
    st.subheader("Dataset Onboarding")
    st.caption(
        f"Active DB: {get_active_db_path()} | Semantic layer: {ROOT_DIR / 'data' / 'semantic_layer.json'}"
    )

    uploaded_db = st.file_uploader("Upload SQLite database", type=["db", "sqlite", "sqlite3"])
    db_path_text = st.text_input("Or connect to a local SQLite path", value="")
    use_llm_understanding = st.checkbox(
        "Use AI to infer dataset meaning",
        value=True,
        help="Uses the configured LLM to propose names, metrics, synonyms, ambiguity rules, and suggested questions from schema and samples.",
    )

    discover_clicked = st.button("Discover dataset", type="primary")
    if discover_clicked:
        try:
            if uploaded_db is not None:
                upload_path = Path(tempfile.gettempdir()) / f"semantic_layer_upload_{uuid.uuid4().hex}.db"
                upload_path.write_bytes(uploaded_db.getbuffer())
                source_path = upload_path
            elif db_path_text.strip():
                source_path = Path(db_path_text.strip())
            else:
                st.warning("Upload a SQLite DB or enter a local DB path.")
                return

            discovered = discover_sqlite_dataset(source_path)
            review = build_review_template(discovered)
            if use_llm_understanding:
                try:
                    with st.spinner("Inferring dataset meaning from schema and samples..."):
                        understanding = generate_dataset_understanding(discovered)
                    review = apply_dataset_understanding_to_review(review, understanding)
                    st.session_state.onboarding_understanding = understanding
                except Exception as e:
                    st.warning(f"AI understanding failed; using schema heuristics. {e}")
            st.session_state.onboarding_source_path = str(source_path)
            st.session_state.onboarding_discovered = discovered
            st.session_state.onboarding_review = review
        except Exception as e:
            st.error(f"Could not discover SQLite dataset: {e}")
            return

    discovered = st.session_state.get("onboarding_discovered")
    review = st.session_state.get("onboarding_review")
    if not discovered or not review:
        if DATASET_MANIFEST_PATH.exists():
            st.info(f"Current onboarded dataset manifest: {DATASET_MANIFEST_PATH}")
        else:
            st.info("No custom dataset is active. The app is using the bundled AP sample database.")
        return

    render_discovery_preview(discovered)
    if st.session_state.get("onboarding_understanding"):
        with st.expander("AI dataset understanding", expanded=False):
            st.json(st.session_state.onboarding_understanding)

    st.subheader("Semantic Review")
    review_tabs = st.tabs(["Tables", "Columns", "Joins", "Metrics", "Publish"])

    with review_tabs[0]:
        edited_tables = st.data_editor(
            [
                {
                    **row,
                    "synonyms": comma_join(row.get("synonyms")),
                }
                for row in review.get("tables", [])
            ],
            use_container_width=True,
            hide_index=True,
            key="onboarding_tables_editor",
        )

    with review_tabs[1]:
        edited_columns = st.data_editor(
            [
                {
                    **row,
                    "synonyms": comma_join(row.get("synonyms")),
                }
                for row in review.get("columns", [])
            ],
            use_container_width=True,
            hide_index=True,
            key="onboarding_columns_editor",
        )

    with review_tabs[2]:
        edited_joins = st.data_editor(
            review.get("joins", []),
            use_container_width=True,
            hide_index=True,
            key="onboarding_joins_editor",
        )

    with review_tabs[3]:
        edited_metrics = st.data_editor(
            [
                {
                    **row,
                    "synonyms": comma_join(row.get("synonyms")),
                    "tables": comma_join(row.get("tables")),
                }
                for row in review.get("metrics", [])
            ],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="onboarding_metrics_editor",
        )

    current_review = normalize_review(edited_tables, edited_columns, edited_joins, edited_metrics)
    st.session_state.onboarding_review = current_review

    with review_tabs[4]:
        semantic_layer = build_semantic_layer(discovered, current_review)
        st.metric("Tables", len(semantic_layer.get("tables", {})))
        st.metric("Approved joins", len(semantic_layer.get("join_paths", {})))
        st.metric("Enabled metrics", len(semantic_layer.get("metrics", {})))

        with st.expander("Semantic layer preview", expanded=False):
            st.json(semantic_layer)

        if st.button("Publish semantic layer and activate DB", type="primary"):
            try:
                manifest = save_onboarded_dataset(
                    st.session_state.onboarding_source_path,
                    semantic_layer,
                    discovered,
                    target_db_path=ACTIVE_DB_PATH,
                )
                st.success(
                    "Published semantic_layer.json, schema.json, and active dataset manifest. "
                    "New investigation questions will use the onboarded DB."
                )
                st.json(manifest)
            except Exception as e:
                st.error(f"Could not publish onboarded dataset: {e}")


def render_data_settings_tab():
    st.subheader("Data Settings")
    st.caption("Global defaults used in SQL prompts, relative-period interpretation, and display formatting.")

    settings = load_data_settings()
    fiscal_start = settings.get("fiscal_year_start", {})
    display_format = settings.get("display_format", {})

    with st.form("data-settings-form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            currency = st.text_input(
                "Default currency",
                value=settings.get("default_currency", DEFAULT_DATA_SETTINGS["default_currency"]),
                max_chars=8,
            )
            timezone = st.text_input(
                "Timezone",
                value=settings.get("timezone", DEFAULT_DATA_SETTINGS["timezone"]),
            )
            today_anchor = st.date_input(
                "Today anchor",
                value=date.fromisoformat(settings.get("today_anchor")),
            )
        with col2:
            fiscal_month = st.number_input(
                "Fiscal year start month",
                min_value=1,
                max_value=12,
                value=int(fiscal_start.get("month", 4)),
                step=1,
            )
            fiscal_day = st.number_input(
                "Fiscal year start day",
                min_value=1,
                max_value=31,
                value=int(fiscal_start.get("day", 1)),
                step=1,
            )
            quarter_definition = st.selectbox(
                "Quarter definition",
                options=["fiscal", "calendar"],
                index=0 if settings.get("quarter_definition") == "fiscal" else 1,
            )
        with col3:
            currency_style_options = ["code", "symbol", "code_suffix"]
            currency_style = st.selectbox(
                "Currency display",
                options=currency_style_options,
                index=currency_style_options.index(display_format.get("currency", "code")),
            )
            decimal_places = st.number_input(
                "Decimal places",
                min_value=0,
                max_value=6,
                value=int(display_format.get("decimal_places", 2)),
                step=1,
            )
            date_format = st.text_input(
                "Date format",
                value=display_format.get("date", DEFAULT_DATA_SETTINGS["display_format"]["date"]),
            )
            month_format = st.text_input(
                "Month format",
                value=display_format.get("month", DEFAULT_DATA_SETTINGS["display_format"]["month"]),
            )

        submitted = st.form_submit_button("Save data settings", type="primary")

    preview_settings = {
        "default_currency": currency,
        "timezone": timezone,
        "today_anchor": today_anchor.isoformat(),
        "fiscal_year_start": {"month": fiscal_month, "day": fiscal_day},
        "month_definition": "calendar",
        "quarter_definition": quarter_definition,
        "display_format": {
            "currency": currency_style,
            "date": date_format,
            "month": month_format,
            "decimal_places": decimal_places,
        },
    }

    if submitted:
        try:
            saved = save_data_settings(preview_settings)
            st.session_state.data_settings = saved
            clear_sql_context_cache()
            st.success("Saved data settings. New SQL generations will use the updated context.")
        except Exception as e:
            st.error(f"Could not save data settings: {e}")

    st.markdown("**Preview**")
    st.write(
        {
            "currency": format_currency(1234567.89, preview_settings),
            "today_anchor": preview_settings["today_anchor"],
            "quarter_definition": preview_settings["quarter_definition"],
        }
    )


def render_glossary_tab():
    st.subheader("Business Glossary")
    semantic_layer_path = ROOT_DIR / "data" / "semantic_layer.json"
    st.caption(f"Editing metrics and ambiguity rules in {semantic_layer_path}")

    if (
        "glossary_semantic_layer" not in st.session_state
        or st.session_state.get("glossary_semantic_layer_path") != str(semantic_layer_path)
    ):
        try:
            st.session_state.glossary_semantic_layer = load_semantic_layer(semantic_layer_path)
            st.session_state.glossary_semantic_layer_path = str(semantic_layer_path)
        except Exception as e:
            st.error(f"Could not load semantic layer: {e}")
            return

    if st.button("Reload glossary from disk"):
        try:
            st.session_state.glossary_semantic_layer = load_semantic_layer(semantic_layer_path)
            st.success("Reloaded semantic layer from disk.")
        except Exception as e:
            st.error(f"Could not reload semantic layer: {e}")
            return

    semantic_layer = st.session_state.glossary_semantic_layer
    glossary_tabs = st.tabs(["Metrics", "Ambiguity Rules", "Save"])

    with glossary_tabs[0]:
        st.caption("Formula is stored as the metric SQL expression; existing `sql` fields remain available to routing and SQL context.")
        edited_metrics = st.data_editor(
            metric_records_from_layer(semantic_layer),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="glossary_metrics_editor",
        )

    with glossary_tabs[1]:
        st.caption("Use JSON for ambiguous_dimensions, for example [{\"label\": \"paid\", \"sql_hint\": \"invoices.status = 'paid'\"}].")
        edited_rules = st.data_editor(
            ambiguity_rule_records_from_layer(semantic_layer),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="glossary_rules_editor",
        )

    with glossary_tabs[2]:
        available_tables = set(semantic_layer.get("tables", {}))
        validation = validate_metric_records(edited_metrics, available_tables=available_tables)

        for warning in validation["warnings"]:
            st.warning(warning)
        for error in validation["errors"]:
            st.error(error)

        try:
            updated_semantic_layer = apply_glossary_records(
                semantic_layer,
                edited_metrics,
                edited_rules,
            )
        except Exception as e:
            st.error(f"Could not normalize glossary edits: {e}")
            return

        st.metric("Metrics", len(updated_semantic_layer.get("metrics", {})))
        st.metric("Ambiguity rules", len(updated_semantic_layer.get("ambiguity_rules", {})))

        with st.expander("Glossary JSON preview", expanded=False):
            st.json(
                {
                    "metrics": updated_semantic_layer.get("metrics", {}),
                    "ambiguity_rules": updated_semantic_layer.get("ambiguity_rules", {}),
                }
            )

        if st.button(
            "Save glossary",
            type="primary",
            disabled=bool(validation["errors"]),
        ):
            try:
                save_semantic_layer(updated_semantic_layer, semantic_layer_path)
                st.session_state.glossary_semantic_layer = updated_semantic_layer
                st.success("Saved glossary metrics and ambiguity rules to semantic_layer.json.")
            except Exception as e:
                st.error(f"Could not save glossary: {e}")


st.set_page_config(page_title=APP_NAME, layout="wide")
init_session_state()

st.title(APP_NAME)
st.caption("Turns ambiguous data questions into evidence-backed SQL analysis.")

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

chat_tab, certified_tab, onboarding_tab, settings_tab, glossary_tab, corrections_tab, benchmark_tab = st.tabs(
    [
        "Investigation",
        "Certified Questions",
        "Dataset Onboarding",
        "Data Settings",
        "Business Glossary",
        "Corrections",
        "Benchmark",
    ]
)

with chat_tab:
    render_chat_tab()

with certified_tab:
    render_certified_questions_tab()

with onboarding_tab:
    render_onboarding_tab()

with settings_tab:
    render_data_settings_tab()

with glossary_tab:
    render_glossary_tab()

with corrections_tab:
    render_corrections_tab()

with benchmark_tab:
    render_benchmark_tab()
