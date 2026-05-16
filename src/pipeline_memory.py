import json

try:
    from .pipeline_config import (
        MAX_CLARIFICATION_ATTEMPTS,
        MAX_MEMORY_FIELD_CHARS,
        MAX_MEMORY_SQL_CHARS,
        MAX_MEMORY_TURNS,
        RESULT_SAMPLE_ROWS,
    )
except ImportError:
    from pipeline_config import (
        MAX_CLARIFICATION_ATTEMPTS,
        MAX_MEMORY_FIELD_CHARS,
        MAX_MEMORY_SQL_CHARS,
        MAX_MEMORY_TURNS,
        RESULT_SAMPLE_ROWS,
    )


def truncate_text(value, max_chars):
    if value is None:
        return None

    text = str(value)
    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}... [truncated]"


def summarize_sql_result(sql_result):
    if sql_result is None:
        return {
            "status": "skipped",
            "reason": "SQL was not generated or execution was skipped.",
        }

    if isinstance(sql_result, str):
        return {
            "status": "error",
            "message": truncate_text(sql_result, MAX_MEMORY_FIELD_CHARS),
        }

    if isinstance(sql_result, list):
        sample_rows = sql_result[:RESULT_SAMPLE_ROWS]
        columns = []

        if sample_rows and isinstance(sample_rows[0], dict):
            columns = list(sample_rows[0].keys())

        return {
            "status": "ok",
            "row_count": len(sql_result),
            "columns": columns,
            "sample_rows": sample_rows,
        }

    return {
        "status": "unknown",
        "value": truncate_text(sql_result, MAX_MEMORY_FIELD_CHARS),
    }


def compact_turn_for_memory(turn):
    return {
        "user_question": truncate_text(turn.get("user_question"), MAX_MEMORY_FIELD_CHARS),
        "resolved_question": truncate_text(turn.get("resolved_question"), MAX_MEMORY_FIELD_CHARS),
        "selected_tables": turn.get("selected_tables", []),
        "selected_metrics": turn.get("selected_metrics", []),
        "sql": truncate_text(turn.get("sql"), MAX_MEMORY_SQL_CHARS),
        "explanation": truncate_text(turn.get("explanation"), MAX_MEMORY_FIELD_CHARS),
        "assumptions": truncate_text(turn.get("assumptions"), MAX_MEMORY_FIELD_CHARS),
        "followup_questions": truncate_text(turn.get("followup_questions"), MAX_MEMORY_FIELD_CHARS),
        "requires_clarification": turn.get("requires_clarification", False),
        "clarification_question": truncate_text(turn.get("clarification_question"), MAX_MEMORY_FIELD_CHARS),
        "clarification_attempts": turn.get("clarification_attempts", 0),
        "clarification_limit_reached": turn.get("clarification_limit_reached", False),
        "chart": turn.get("chart"),
        "result_summary": turn.get("result_summary"),
        "cache_hit": turn.get("cache_hit", False),
        "cache_strategy": turn.get("cache_strategy"),
    }


def format_conversation_context(conversation_turns):
    if not conversation_turns:
        return "No prior conversation."

    formatted_turns = []
    for index, turn in enumerate(conversation_turns[-MAX_MEMORY_TURNS:], start=1):
        compact_turn = compact_turn_for_memory(turn)
        formatted_turns.append(
            f"Turn {index}:\n{json.dumps(compact_turn, indent=2, default=str)}"
        )

    return "\n\n".join(formatted_turns)


def trim_conversation_turns(conversation_turns):
    return conversation_turns[-MAX_MEMORY_TURNS:]


def active_conversation_context(state):
    question_resolution = state.get("question_resolution", {})

    if (
        question_resolution.get("is_follow_up")
        or question_resolution.get("memory_used")
        or question_resolution.get("clarification_needed")
    ):
        return state.get("conversation_context", "No prior conversation.")

    return "No prior conversation is relevant to this turn."


def sql_generation_conversation_context(state):
    conversation_context = active_conversation_context(state)
    clarification_response = state.get("clarification_response", {})
    default_assumption = clarification_response.get("default_assumption")

    if default_assumption:
        return (
            f"{conversation_context}\n\n"
            "Current turn clarification default assumption:\n"
            f"{default_assumption}"
        )

    return conversation_context


def latest_pending_clarification_turn(conversation_turns):
    if not conversation_turns:
        return None

    latest_turn = conversation_turns[-1]
    if latest_turn.get("requires_clarification") and not latest_turn.get("sql"):
        return latest_turn

    return None


def clarification_attempts_for_current_question(state):
    pending_turn = latest_pending_clarification_turn(state.get("conversation_turns", []))
    question_resolution = state.get("question_resolution", {})

    if not pending_turn:
        return 0

    if question_resolution.get("is_follow_up") or question_resolution.get("memory_used"):
        return int(pending_turn.get("clarification_attempts") or 1)

    return 0
