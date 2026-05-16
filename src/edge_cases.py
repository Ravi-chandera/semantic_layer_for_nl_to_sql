from __future__ import annotations

from typing import Any


DEFAULT_TOO_MANY_ROWS_THRESHOLD = 500

DATE_COLUMN_MARKERS = ("date", "_dt", "dt_", "month", "year", "period")
NAME_COLUMN_MARKERS = ("vendor", "customer")
ENTITY_ID_MARKERS = ("id", "code", "number", "no")


def _edge_case(case_type, title, explanation, next_actions):
    return {
        "type": case_type,
        "title": title,
        "explanation": explanation,
        "next_actions": list(next_actions or []),
    }


def _text_parts(*values):
    parts = []
    for value in values:
        if isinstance(value, dict):
            parts.extend(_text_parts(*value.values()))
        elif isinstance(value, list):
            parts.extend(_text_parts(*value))
        elif value is not None:
            parts.append(str(value))
    return parts


def _combined_text(*values):
    return " ".join(_text_parts(*values)).lower()


def _rows_from_result(sql_result):
    if isinstance(sql_result, str) or sql_result is None:
        return []
    if isinstance(sql_result, list):
        return [row for row in sql_result if isinstance(row, dict)]
    try:
        return [row for row in list(sql_result) if isinstance(row, dict)]
    except TypeError:
        return []


def _has_date_context(question, sql_output, analysis):
    text = _combined_text(
        question,
        sql_output.get("SQL") if isinstance(sql_output, dict) else None,
        sql_output.get("Resolved_Question") if isinstance(sql_output, dict) else None,
        analysis.get("Assumptions") if isinstance(analysis, dict) else None,
    )
    return any(
        token in text
        for token in (
            "date",
            "period",
            "month",
            "quarter",
            "year",
            "week",
            "today",
            "yesterday",
            "last ",
            "this ",
            "between",
        )
    )


def _date_columns(rows):
    if not rows:
        return []
    columns = rows[0].keys()
    return [
        column
        for column in columns
        if any(marker in str(column).lower() for marker in DATE_COLUMN_MARKERS)
    ]


def _currency_columns(rows):
    if not rows:
        return []
    return [
        column
        for column in rows[0].keys()
        if "currency" in str(column).lower()
    ]


def _entity_name_columns(rows):
    if not rows:
        return []
    columns = rows[0].keys()
    return [
        column
        for column in columns
        if "name" in str(column).lower()
        and any(marker in str(column).lower() for marker in NAME_COLUMN_MARKERS)
    ]


def _candidate_id_columns(rows, name_column):
    name_prefix = str(name_column).lower().replace("_name", "").replace("name", "")
    candidates = []
    for column in rows[0].keys():
        normalized = str(column).lower()
        if column == name_column:
            continue
        if not any(marker in normalized for marker in ENTITY_ID_MARKERS):
            continue
        if name_prefix and name_prefix.strip("_") in normalized:
            candidates.append(column)
    if candidates:
        return candidates
    return [
        column
        for column in rows[0].keys()
        if column != name_column and str(column).lower().endswith("_id")
    ]


def _duplicate_entity_names(rows):
    duplicates = []
    for name_column in _entity_name_columns(rows):
        id_columns = _candidate_id_columns(rows, name_column)
        if not id_columns:
            continue
        seen = {}
        for row in rows:
            name_value = row.get(name_column)
            if not name_value:
                continue
            key = str(name_value).strip().lower()
            seen.setdefault(key, {"name": str(name_value), "ids": set()})
            for id_column in id_columns:
                identifier = row.get(id_column)
                if identifier not in (None, ""):
                    seen[key]["ids"].add(str(identifier))
        for item in seen.values():
            if len(item["ids"]) > 1:
                duplicates.append(item["name"])
    return sorted(set(duplicates))


def _analysis_suggests_ambiguity(sql_output, analysis):
    text = _combined_text(sql_output, analysis)
    return "ambiguous" in text or "clarification" in text and "needed" in text


def classify_edge_cases(
    *,
    question=None,
    sql_output=None,
    sql_result=None,
    analysis=None,
    chart_error=None,
    too_many_rows_threshold=DEFAULT_TOO_MANY_ROWS_THRESHOLD,
):
    sql_output = sql_output or {}
    analysis = analysis or {}
    rows = _rows_from_result(sql_result)
    cases = []

    if chart_error:
        cases.append(
            _edge_case(
                "chart_not_possible",
                "Chart not possible",
                f"The result is still available, but the chart step could not render it: {chart_error}",
                [
                    "Review the result table instead.",
                    "Ask for a simpler chart with one date or category and one numeric measure.",
                ],
            )
        )

    if isinstance(sql_result, str):
        cases.append(
            _edge_case(
                "sql_error",
                "SQL could not run",
                sql_result,
                [
                    "Rephrase with the exact entity, metric, filter, and time period.",
                    "Use the generated SQL panel to inspect the failed query.",
                ],
            )
        )
        return cases

    if sql_output.get("Requires_Clarification") or analysis.get("Clarification", {}).get("needed"):
        cases.append(
            _edge_case(
                "ambiguous_entity_name",
                "Question needs clarification",
                sql_output.get("Clarification_Question")
                or sql_output.get("Followup_Questions")
                or analysis.get("Clarification", {}).get("question")
                or "The question can map to more than one business interpretation.",
                [
                    "Choose one of the clarification options when shown.",
                    "Ask again with the exact vendor, customer, metric, or time period.",
                ],
            )
        )
        return cases

    generated_sql = sql_output.get("SQL")
    supported_no_sql_modes = {
        "data_overview",
        "planned_dataset_analysis",
        "clarification_required",
        "metric_driver_investigation",
    }
    analysis_mode = analysis.get("Mode") or sql_output.get("Analysis_Mode")
    if not generated_sql and analysis_mode not in supported_no_sql_modes:
        explanation = (
            sql_output.get("Explanation")
            or sql_output.get("Assumptions")
            or "No executable SQL was generated for this request."
        )
        cases.append(
            _edge_case(
                "unsupported_question",
                "Question is not supported by the current data",
                explanation,
                [
                    "Ask about entities, metrics, statuses, dates, or categories visible in the active dataset.",
                    "Use 'What can I ask you?' to see supported examples.",
                ],
            )
        )

    if sql_result is not None and not rows:
        if _has_date_context(question, sql_output, analysis):
            cases.append(
                _edge_case(
                    "no_data_for_period",
                    "No data for that period",
                    "The query ran successfully, but no rows matched the requested time period or date filter.",
                    [
                        "Try a broader date range.",
                        "Ask what date range is available for this dataset.",
                    ],
                )
            )
        else:
            cases.append(
                _edge_case(
                    "no_rows",
                    "No matching rows",
                    "The query ran successfully, but no rows matched the filters.",
                    [
                        "Relax one filter or ask for a broader summary.",
                        "Check whether the entity name or status value matches the data.",
                    ],
                )
            )

    if len(rows) > too_many_rows_threshold:
        cases.append(
            _edge_case(
                "too_many_rows",
                "Result is large",
                f"The query returned {len(rows)} rows, which is more than the display guidance threshold of {too_many_rows_threshold}.",
                [
                    "Add a date range, status, vendor, customer, or top-N limit.",
                    "Ask for a grouped summary before drilling into row detail.",
                ],
            )
        )

    duplicate_names = _duplicate_entity_names(rows)
    if duplicate_names:
        shown = ", ".join(duplicate_names[:3])
        cases.append(
            _edge_case(
                "duplicate_entity_names",
                "Duplicate vendor or customer names detected",
                f"At least one entity name appears with multiple identifiers: {shown}.",
                [
                    "Use the ID column to identify the exact entity.",
                    "Ask for a disambiguation table with name, ID, city, tax ID, and recent activity.",
                ],
            )
        )

    date_columns = _date_columns(rows)
    null_date_columns = [
        column
        for column in date_columns
        if any(row.get(column) in (None, "") for row in rows)
    ]
    if null_date_columns:
        cases.append(
            _edge_case(
                "null_dates",
                "Some date values are missing",
                f"Date-based interpretation may be incomplete because these columns contain blanks: {', '.join(null_date_columns[:4])}.",
                [
                    "Filter to rows where the date is present.",
                    "Ask for a count of records with missing dates by table or status.",
                ],
            )
        )

    mixed_currency_columns = []
    for column in _currency_columns(rows):
        values = {
            str(row.get(column)).strip().upper()
            for row in rows
            if row.get(column) not in (None, "")
        }
        if len(values) > 1:
            mixed_currency_columns.append(f"{column}: {', '.join(sorted(values)[:5])}")
    if mixed_currency_columns:
        cases.append(
            _edge_case(
                "mixed_currencies",
                "Mixed currencies detected",
                f"The result includes multiple currencies ({'; '.join(mixed_currency_columns)}), so totals may not be directly comparable.",
                [
                    "Filter to one currency before comparing amounts.",
                    "Ask for the result grouped by currency.",
                ],
            )
        )

    if not cases and _analysis_suggests_ambiguity(sql_output, analysis):
        cases.append(
            _edge_case(
                "ambiguous_entity_name",
                "Possible ambiguity detected",
                "The analysis mentions ambiguity or clarification, so treat the answer as conditional.",
                [
                    "Ask again with the exact entity name or ID.",
                    "Use a follow-up that names the metric and period explicitly.",
                ],
            )
        )

    return cases
