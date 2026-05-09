import re

from pipeline_config import (
    INSUFFICIENT_DATA_PREFIX,
    MAX_CLARIFICATION_ATTEMPTS,
    MAX_MEMORY_FIELD_CHARS,
)
from pipeline_memory import clarification_attempts_for_current_question, truncate_text


EXECUTABLE_SQL_PATTERN = re.compile(
    r"^\s*(?:(?:--[^\n]*(?:\n|$)|/\*.*?\*/)\s*)*(?:SELECT|WITH)\b",
    re.IGNORECASE | re.DOTALL,
)


def build_sql_response(
    *,
    sql,
    explanation,
    assumptions,
    followup_questions=None,
    chart="none",
    requires_clarification=False,
    clarification_question=None,
    clarification_attempts=0,
    clarification_limit_reached=False,
):
    return {
        "SQL": sql,
        "Explanation": explanation,
        "Assumptions": assumptions,
        "Followup_Questions": followup_questions,
        "Chart": chart,
        "Requires_Clarification": requires_clarification,
        "Clarification_Question": clarification_question,
        "Clarification_Attempts": clarification_attempts,
        "Clarification_Limit_Reached": clarification_limit_reached,
    }


def no_valid_tables_response():
    return build_sql_response(
        sql=None,
        explanation="No valid database table was selected for this question.",
        assumptions="The router did not map the question to any table present in the semantic layer.",
        followup_questions=(
            "Can you rephrase the question using available business entities like invoices, "
            "payments, vendors, purchase orders, products, departments, companies, GRNs, "
            "or approval matrix?"
        ),
    )


def clarification_needed_response(clarifying_question, clarification_attempts=1, reason=None):
    return build_sql_response(
        sql=None,
        explanation=reason or "The question needs one more business detail before SQL can be generated safely.",
        assumptions="No SQL was generated because the latest message is underspecified.",
        followup_questions=clarifying_question,
        requires_clarification=True,
        clarification_question=clarifying_question,
        clarification_attempts=clarification_attempts,
    )


def clarification_limit_response(reason=None):
    return build_sql_response(
        sql=None,
        explanation=(
            "I could not resolve the question safely after the clarification attempt."
        ),
        assumptions=reason or (
            "No SQL was generated because a required business detail is still missing, "
            "and the clarification limit has been reached."
        ),
        followup_questions=None,
        clarification_limit_reached=True,
    )


def is_executable_sql(sql):
    if not isinstance(sql, str):
        return False

    return bool(EXECUTABLE_SQL_PATTERN.match(sql))


def non_executable_sql_response(sql_response):
    sql_text = str(sql_response.get("SQL") or "").strip()
    lower_sql_text = sql_text.lower()

    if lower_sql_text.startswith(INSUFFICIENT_DATA_PREFIX):
        return build_sql_response(
            sql=None,
            explanation=sql_text,
            assumptions=(
                sql_response.get("Assumptions")
                or "No SQL was generated because required data is not available in the selected semantic context."
            ),
            followup_questions=None,
            chart=sql_response.get("Chart") or "none",
        )

    return build_sql_response(
        sql=None,
        explanation="The model did not return executable SQL.",
        assumptions=(
            sql_response.get("Assumptions")
            or f"Rejected non-SQL response from SQL generation: {truncate_text(sql_text, MAX_MEMORY_FIELD_CHARS)}"
        ),
        followup_questions=sql_response.get("Followup_Questions"),
        chart=sql_response.get("Chart") or "none",
    )


def normalize_sql_response_after_generation(sql_response, state):
    sql_response = dict(sql_response)
    if sql_response.get("SQL"):
        if not is_executable_sql(sql_response.get("SQL")):
            if not sql_response.get("Followup_Questions"):
                return non_executable_sql_response(sql_response)

            sql_response["SQL"] = None

    if sql_response.get("SQL"):
        sql_response.setdefault("Requires_Clarification", False)
        sql_response.setdefault("Clarification_Question", None)
        sql_response.setdefault("Clarification_Attempts", clarification_attempts_for_current_question(state))
        sql_response.setdefault("Clarification_Limit_Reached", False)
        return sql_response

    followup_question = sql_response.get("Followup_Questions")
    if not followup_question:
        sql_response.setdefault("Requires_Clarification", False)
        sql_response.setdefault("Clarification_Question", None)
        sql_response.setdefault("Clarification_Attempts", clarification_attempts_for_current_question(state))
        sql_response.setdefault("Clarification_Limit_Reached", False)
        return sql_response

    clarification_attempts = clarification_attempts_for_current_question(state)
    if clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
        return clarification_limit_response(sql_response.get("Explanation") or sql_response.get("Assumptions"))

    return clarification_needed_response(
        followup_question,
        clarification_attempts=clarification_attempts + 1,
        reason=sql_response.get("Explanation"),
    )
