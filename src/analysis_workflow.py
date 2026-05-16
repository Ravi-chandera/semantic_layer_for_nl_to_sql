import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
    from .answer_modes import (
        DEFAULT_ANSWER_MODE,
        apply_answer_mode_to_sql_output,
        normalize_answer_mode,
    )
    from .data_settings import format_currency, format_date, load_data_settings
    from . import sqlite_runner
    from .langfuse_tracing import safe_update_observation, traced_span
    from .model_config import get_default_model_name
    from .pipeline import (
        gemini_call,
        generate_sql_for_question,
        load_json,
        load_string_as_json,
        record_sql_execution_for_thread,
    )
    from .pipeline_config import SEMANTIC_LAYER_PATH
except ImportError:
    from answer_modes import (
        DEFAULT_ANSWER_MODE,
        apply_answer_mode_to_sql_output,
        normalize_answer_mode,
    )
    from data_settings import format_currency, format_date, load_data_settings
    import sqlite_runner
    from langfuse_tracing import safe_update_observation, traced_span
    from model_config import get_default_model_name
    from pipeline import (
        gemini_call,
        generate_sql_for_question,
        load_json,
        load_string_as_json,
        record_sql_execution_for_thread,
    )
    from pipeline_config import SEMANTIC_LAYER_PATH

MAX_PREVIEW_ROWS = 20


@dataclass(frozen=True)
class EvidenceQuery:
    name: str
    purpose: str
    sql: str
    tables: list[str]
    columns: list[str]
    metric: str | None = None


def _round_number(value, digits=2):
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _safe_float(value, default=0.0):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_percent_change(current_value, previous_value):
    current = _safe_float(current_value)
    previous = _safe_float(previous_value)
    if previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100.0


def _format_money(value, unit=None, settings=None):
    resolved_settings = load_data_settings() if settings is None else settings
    if unit:
        resolved_settings = {
            **resolved_settings,
            "default_currency": unit,
        }
    return format_currency(value, resolved_settings)


def _format_month_label(value, settings=None):
    return format_date(value, settings or load_data_settings(), kind="month")


def _format_percent(value):
    if value is None:
        return "n/a"
    return f"{_safe_float(value):,.1f}%"


def _extract_sql_tables(sql):
    if not isinstance(sql, str):
        return []

    cte_names = set(
        re.findall(
            r"(?:WITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(",
            sql,
            flags=re.IGNORECASE,
        )
    )
    table_names = set(
        re.findall(
            r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
            sql,
            flags=re.IGNORECASE,
        )
    )
    return sorted(table_names - cte_names)


def _quote_identifier(identifier):
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def _sql_table(table_name, alias=None):
    if alias:
        return f"{_quote_identifier(table_name)} AS {_quote_identifier(alias)}"
    return _quote_identifier(table_name)


def _sql_col(column_name, alias=None):
    if alias:
        return f"{_quote_identifier(alias)}.{_quote_identifier(column_name)}"
    return _quote_identifier(column_name)


def _is_numeric_semantic_column(column_info):
    column_type = str(column_info.get("type") or "").upper()
    return bool(column_info.get("is_metric")) or any(
        part in column_type
        for part in ("INT", "REAL", "NUM", "DEC", "DOUBLE", "FLOAT")
    )


def _is_text_semantic_column(column_info):
    column_type = str(column_info.get("type") or "").upper()
    return any(part in column_type for part in ("CHAR", "TEXT", "CLOB", "VARCHAR"))


def _is_date_semantic_column(column_name, column_info):
    normalized = str(column_name or "").lower()
    column_type = str(column_info.get("type") or "").upper()
    return (
        "DATE" in column_type
        or "TIME" in column_type
        or normalized.endswith("_date")
        or normalized.endswith("_at")
        or normalized in {"date", "month", "year"}
    )


def _is_category_semantic_column(column_name, column_info):
    normalized = str(column_name or "").lower()
    if column_info.get("enum_values"):
        return True
    if not _is_text_semantic_column(column_info):
        return False
    category_terms = ("status", "type", "category", "segment", "state", "mode", "class")
    return any(term in normalized for term in category_terms)


def _display_name_for_table(table_name, table_info):
    return (
        table_info.get("business_name")
        or table_info.get("description")
        or table_name.replace("_", " ")
    )


def _semantic_profile(semantic_layer):
    profiles = {}
    for table_name, table_info in semantic_layer.get("tables", {}).items():
        columns = table_info.get("columns", {}) or {}
        profile = {
            "table": table_name,
            "description": table_info.get("description"),
            "business_context": table_info.get("business_context"),
            "metric_columns": [],
            "numeric_columns": [],
            "date_columns": [],
            "category_columns": [],
            "display_columns": [],
        }
        primary_keys = {
            part.strip()
            for part in str(table_info.get("primary_key") or "").split(",")
            if part.strip()
        }
        for column_name, column_info in columns.items():
            normalized_column = str(column_name or "").lower()
            is_identifier = (
                normalized_column == "id"
                or normalized_column.endswith("_id")
                or column_name in primary_keys
            )
            if _is_numeric_semantic_column(column_info) and not is_identifier:
                profile["numeric_columns"].append(column_name)
                if column_info.get("is_metric") and not is_identifier:
                    profile["metric_columns"].append(column_name)
            if _is_date_semantic_column(column_name, column_info):
                profile["date_columns"].append(column_name)
            if _is_category_semantic_column(column_name, column_info):
                profile["category_columns"].append(column_name)
            if _is_text_semantic_column(column_info) and not column_info.get("is_sensitive"):
                profile["display_columns"].append(column_name)
        if not profile["metric_columns"]:
            profile["metric_columns"] = profile["numeric_columns"][:3]
        profiles[table_name] = profile
    return profiles


def _table_analysis_score(table_name, profile, semantic_layer):
    table_info = semantic_layer.get("tables", {}).get(table_name, {})
    relationship_count = len(table_info.get("relationships") or [])
    incoming_relationship_count = sum(
        1
        for other_table in semantic_layer.get("tables", {}).values()
        for relationship in other_table.get("relationships") or []
        if relationship.get("target_table") == table_name
    )
    return (
        len(profile.get("metric_columns") or []) * 4
        + len(profile.get("date_columns") or []) * 3
        + len(profile.get("category_columns") or []) * 2
        + relationship_count
        + incoming_relationship_count
    )


def _ordered_profiles_for_analysis(semantic_layer, profiles):
    return sorted(
        profiles.items(),
        key=lambda item: (
            _table_analysis_score(item[0], item[1], semantic_layer),
            item[0],
        ),
        reverse=True,
    )


def _safe_alias(value):
    alias = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_").lower()
    if not alias or alias[0].isdigit():
        alias = f"field_{alias}"
    return alias[:48]


def _count_query_for_table(table_name):
    return EvidenceQuery(
        name=f"{_safe_alias(table_name)}_row_count",
        purpose=f"Count available records in {table_name}.",
        sql=f"SELECT COUNT(*) AS row_count FROM {_quote_identifier(table_name)};",
        tables=[table_name],
        columns=[],
    )


def _metric_definition(semantic_layer, metric_name):
    metric = semantic_layer.get("metrics", {}).get(metric_name)
    if not metric:
        return None

    return {
        "metric": metric_name,
        "description": metric.get("description"),
        "sql": metric.get("sql"),
        "filters": metric.get("filters"),
        "result_unit": metric.get("result_unit"),
        "tables": metric.get("tables") or [],
    }


def _table_column_citations(semantic_layer, tables, columns):
    cited_tables = []
    cited_columns = []

    for table_name in tables:
        table_info = semantic_layer.get("tables", {}).get(table_name, {})
        cited_tables.append(
            {
                "table": table_name,
                "description": table_info.get("description"),
            }
        )

    for column_ref in columns:
        if "." not in column_ref:
            cited_columns.append({"column": column_ref, "description": None})
            continue

        table_name, column_name = column_ref.split(".", 1)
        column_info = (
            semantic_layer.get("tables", {})
            .get(table_name, {})
            .get("columns", {})
            .get(column_name, {})
        )
        cited_columns.append(
            {
                "column": column_ref,
                "description": column_info.get("description"),
            }
        )

    return cited_tables, cited_columns


@lru_cache(maxsize=1)
def load_default_sql_runner():
    return sqlite_runner


def _run_evidence_query(query: EvidenceQuery, sql_runner):
    result = sql_runner.run_query(query.sql)
    is_error = isinstance(result, str)
    rows = [] if is_error else list(result or [])
    row_count = None if is_error else len(rows)

    checks = [
        {
            "name": "read_only_guardrail_and_query_plan",
            "status": "failed" if is_error else "passed",
            "detail": result if is_error else "SQLite guardrails, table validation, and query plan checks passed.",
        },
        {
            "name": "result_shape",
            "status": "failed" if is_error else "passed",
            "detail": result if is_error else f"Returned {row_count} row(s).",
        },
    ]

    if not is_error:
        checks.append(
            {
                "name": "non_empty_result",
                "status": "warning" if row_count == 0 else "passed",
                "detail": "No rows returned." if row_count == 0 else "Result contains data.",
            }
        )

    return {
        "name": query.name,
        "purpose": query.purpose,
        "status": "error" if is_error else "ok",
        "error": result if is_error else None,
        "row_count": row_count,
        "result_preview": rows[:MAX_PREVIEW_ROWS],
        "sql": query.sql,
        "tables": query.tables,
        "columns": query.columns,
        "metric": query.metric,
        "checks": checks,
    }


def build_metric_driver_evidence_queries(metric_name=None, semantic_layer=None):
    semantic_layer = semantic_layer or load_json(str(SEMANTIC_LAYER_PATH))
    metric = semantic_layer.get("metrics", {}).get(metric_name or "")
    if metric and metric.get("tables"):
        source_tables = [table for table in metric.get("tables", []) if table in semantic_layer.get("tables", {})]
        focused_layer = {
            **semantic_layer,
            "tables": {
                table: semantic_layer["tables"][table]
                for table in source_tables
            },
            "metrics": {metric_name: metric},
        }
        queries = build_dataset_analysis_plan(focused_layer, max_queries=5)
        return [
            EvidenceQuery(
                name=query.name,
                purpose=query.purpose,
                sql=query.sql,
                tables=query.tables,
                columns=query.columns,
                metric=query.metric or metric_name,
            )
            for query in queries
        ]

    return build_dataset_analysis_plan(semantic_layer, max_queries=5)


def build_revenue_drop_evidence_queries():
    """Backward-compatible wrapper; evidence is now semantic-layer driven."""
    return build_metric_driver_evidence_queries("revenue")


def infer_analysis_intent(user_question, semantic_layer):
    normalized = str(user_question or "").lower()
    overview_phrases = [
        "what can i ask",
        "what can i ask you",
        "what questions can i ask",
        "what data",
        "what does the data contain",
        "what is in the data",
        "what do you know",
        "help me explore",
    ]
    broad_analysis_phrases = [
        "whole analysis",
        "full analysis",
        "complete analysis",
        "overall analysis",
        "analyze the data",
        "analyse the data",
        "analyze all data",
        "analyse all data",
        "give me insights",
        "business review",
    ]
    why_terms = ["why", "reason", "driver", "cause", "explain", "root cause"]
    change_terms = ["drop", "decline", "fell", "fall", "decrease", "down", "increase", "up", "change", "spike"]
    if any(phrase in normalized for phrase in overview_phrases):
        return {
            "mode": "data_overview",
            "metric": None,
            "reason": "The user is asking what the system can answer, so summarize the available data and example questions.",
        }

    if any(phrase in normalized for phrase in broad_analysis_phrases):
        return {
            "mode": "planned_dataset_analysis",
            "metric": None,
            "reason": "The user is asking for a broad analysis, so plan and run multiple focused evidence questions.",
        }

    if any(term in normalized for term in why_terms) and any(term in normalized for term in change_terms):
        for metric_name, metric in semantic_layer.get("metrics", {}).items():
            metric_terms = [metric_name.replace("_", " "), *(metric.get("synonyms") or [])]
            if any(str(term).lower() in normalized for term in metric_terms if term):
                return {
                    "mode": "metric_driver_investigation",
                    "metric": metric_name,
                    "reason": "The question asks for drivers behind a metric change, so run semantic-layer driven evidence queries.",
                }

    return {
        "mode": "single_query_analysis",
        "metric": None,
        "reason": "The question can be answered from the generated SQL result.",
    }


def _semantic_table_groups(semantic_layer):
    groups = []
    for table_name, table_info in semantic_layer.get("tables", {}).items():
        metric_columns = [
            column_name
            for column_name, column_info in table_info.get("columns", {}).items()
            if column_info.get("is_metric")
        ]
        groups.append(
            {
                "table": table_name,
                "description": table_info.get("description"),
                "business_context": table_info.get("business_context"),
                "metric_columns": metric_columns[:5],
                "synonyms": table_info.get("synonyms") or [],
            }
        )
    return groups


def _example_questions_from_semantic_layer(semantic_layer):
    metrics = semantic_layer.get("metrics", {})
    profiles = _semantic_profile(semantic_layer)
    examples = []

    for metric_name, metric in metrics.items():
        description = metric.get("description")
        display_metric = metric_name.replace("_", " ")
        metric_tables = metric.get("tables") or []
        if metric_tables:
            examples.append(f"Show {display_metric} by month if a date field exists.")
            examples.append(f"Which records have the highest {display_metric}?")
        elif description:
            examples.append(f"Show {display_metric}: {description.lower()}.")

    for table_name, profile in profiles.items():
        table_label = table_name.replace("_", " ")
        examples.append(f"How many {table_label} records are in the data?")
        if profile["category_columns"]:
            category = profile["category_columns"][0].replace("_", " ")
            examples.append(f"Break down {table_label} by {category}.")
        if profile["metric_columns"]:
            metric = profile["metric_columns"][0].replace("_", " ")
            examples.append(f"What is the total {metric} in {table_label}?")
        if profile["date_columns"]:
            date_col = profile["date_columns"][0].replace("_", " ")
            examples.append(f"Show {table_label} trends by {date_col}.")

    return examples[:8]


def _overview_profile_queries(semantic_layer):
    queries = []
    for table_name in semantic_layer.get("tables", {}):
        queries.append(_count_query_for_table(table_name))

    profiles = _semantic_profile(semantic_layer)
    for table_name, profile in profiles.items():
        for date_column in profile["date_columns"][:2]:
            queries.append(
                EvidenceQuery(
                    name=f"{_safe_alias(table_name)}_{_safe_alias(date_column)}_range",
                    purpose=f"Find available date coverage for {table_name}.",
                    sql=(
                        f"SELECT MIN({_quote_identifier(date_column)}) AS min_date, "
                        f"MAX({_quote_identifier(date_column)}) AS max_date "
                        f"FROM {_quote_identifier(table_name)};"
                    ),
                    tables=[table_name],
                    columns=[f"{table_name}.{date_column}"],
                )
            )
    return queries


def _build_data_overview_analysis(user_question, evidence_items, semantic_layer):
    table_groups = _semantic_table_groups(semantic_layer)
    row_counts = {}
    date_ranges = {}
    for item in evidence_items:
        rows = _rows(item)
        if not rows:
            continue
        if item["name"].endswith("_row_count"):
            table_name = item["name"][: -len("_row_count")]
            matched_table = next(
                (
                    table["table"]
                    for table in table_groups
                    if _safe_alias(table["table"]) == table_name
                ),
                table_name,
            )
            row_counts[matched_table] = rows[0].get("row_count")
        elif item["name"].endswith("_range"):
            date_ranges[item["name"]] = rows[0]

    top_tables = [
        f"{table['table']} ({row_counts.get(table['table'], 'unknown')} rows): {table.get('description')}"
        for table in table_groups[:8]
    ]
    metric_names = list(semantic_layer.get("metrics", {}).keys())
    examples = _example_questions_from_semantic_layer(semantic_layer)

    table_count = len(table_groups)
    metric_count = len(metric_names)
    answer = (
        f"This dataset has {table_count} table(s) and {metric_count} semantic metric(s). "
        "You can ask for counts, breakdowns, trends, rankings, joins, and broader reviews "
        "using the entities and metrics discovered from the active database."
    )

    analysis = {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "data_overview",
        "Question": user_question,
        "Resolved_Question": "What data is available and what can be asked?",
        "Executive_Answer": answer,
        "Clarification": {"needed": False, "question": None, "decision": None},
        "Assumptions": [
            "This overview is built from the semantic layer plus lightweight row-count and date-range checks.",
            "Example questions are limited to tables and metrics present in the current semantic layer.",
        ],
        "Definitions": [
            _metric_definition(semantic_layer, metric_name)
            for metric_name in metric_names
            if _metric_definition(semantic_layer, metric_name)
        ],
        "Dataset_Overview": {
            "tables": table_groups,
            "row_counts": row_counts,
            "date_ranges": date_ranges,
            "metrics": metric_names,
            "example_questions": examples,
            "summary_bullets": top_tables,
        },
        "Period_Comparison": None,
        "Anomalies": [],
        "Citations": {
            "metrics": [
                _metric_definition(semantic_layer, metric_name)
                for metric_name in metric_names
                if _metric_definition(semantic_layer, metric_name)
            ],
            "tables": [
                {"table": table["table"], "description": table.get("description")}
                for table in table_groups
            ],
            "columns": [],
        },
        "Evidence": evidence_items,
        "Confidence": {
            "level": "high",
            "score": 0.9,
            "reasons": ["The overview uses semantic metadata and direct database profile queries."],
        },
        "Limitations": [
            "The overview explains the active database from schema metadata and sample values; it does not infer facts outside the uploaded data."
        ],
        "Suggested_Next_Queries": examples,
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }
    return _augment_confidence_reasons(analysis, semantic_layer)


def _metric_name_for_table_column(semantic_layer, table_name, column_name):
    for metric_name, metric in semantic_layer.get("metrics", {}).items():
        metric_tables = metric.get("tables") or []
        metric_sql = str(metric.get("sql") or "").lower()
        if table_name in metric_tables and column_name.lower() in metric_sql:
            return metric_name
    return None


def _category_breakdown_query(table_name, category_column, metric_column=None, semantic_layer=None):
    category_alias = _safe_alias(category_column)
    table_alias = "t"
    select_parts = [
        f"{_sql_col(category_column, table_alias)} AS {_quote_identifier(category_alias)}",
        "COUNT(*) AS row_count",
    ]
    order_by = "row_count"
    columns = [f"{table_name}.{category_column}"]
    metric_name = None
    if metric_column:
        metric_alias = f"total_{_safe_alias(metric_column)}"
        select_parts.append(
            f"ROUND(SUM(COALESCE({_sql_col(metric_column, table_alias)}, 0)), 2) AS {_quote_identifier(metric_alias)}"
        )
        order_by = _quote_identifier(metric_alias)
        columns.append(f"{table_name}.{metric_column}")
        if semantic_layer:
            metric_name = _metric_name_for_table_column(semantic_layer, table_name, metric_column)

    select_sql = ",\n  ".join(select_parts)
    sql = f"""
SELECT
  {select_sql}
FROM {_sql_table(table_name, table_alias)}
GROUP BY {_sql_col(category_column, table_alias)}
ORDER BY {order_by} DESC
LIMIT 20;
""".strip()
    return EvidenceQuery(
        name=f"{_safe_alias(table_name)}_by_{_safe_alias(category_column)}",
        purpose=f"Break down {table_name} by {category_column}.",
        sql=sql,
        tables=[table_name],
        columns=columns,
        metric=metric_name,
    )


def _monthly_trend_query(table_name, date_column, metric_column=None, semantic_layer=None):
    table_alias = "t"
    select_parts = [
        f"strftime('%Y-%m', {_sql_col(date_column, table_alias)}) AS month",
        "COUNT(*) AS row_count",
    ]
    order_by = "month"
    columns = [f"{table_name}.{date_column}"]
    metric_name = None
    if metric_column:
        metric_alias = f"total_{_safe_alias(metric_column)}"
        select_parts.append(
            f"ROUND(SUM(COALESCE({_sql_col(metric_column, table_alias)}, 0)), 2) AS {_quote_identifier(metric_alias)}"
        )
        columns.append(f"{table_name}.{metric_column}")
        if semantic_layer:
            metric_name = _metric_name_for_table_column(semantic_layer, table_name, metric_column)

    select_sql = ",\n  ".join(select_parts)
    sql = f"""
SELECT
  {select_sql}
FROM {_sql_table(table_name, table_alias)}
WHERE {_sql_col(date_column, table_alias)} IS NOT NULL
GROUP BY strftime('%Y-%m', {_sql_col(date_column, table_alias)})
ORDER BY {order_by}
LIMIT 36;
""".strip()
    return EvidenceQuery(
        name=f"{_safe_alias(table_name)}_{_safe_alias(date_column)}_monthly_trend",
        purpose=f"Review the monthly trend for {table_name} using {date_column}.",
        sql=sql,
        tables=[table_name],
        columns=columns,
        metric=metric_name,
    )


def _numeric_summary_query(table_name, metric_column, semantic_layer=None):
    table_alias = "t"
    metric_alias = _safe_alias(metric_column)
    sql = f"""
SELECT
  COUNT(*) AS row_count,
  ROUND(SUM(COALESCE({_sql_col(metric_column, table_alias)}, 0)), 2) AS total_{metric_alias},
  ROUND(AVG({_sql_col(metric_column, table_alias)}), 2) AS avg_{metric_alias},
  MIN({_sql_col(metric_column, table_alias)}) AS min_{metric_alias},
  MAX({_sql_col(metric_column, table_alias)}) AS max_{metric_alias}
FROM {_sql_table(table_name, table_alias)};
""".strip()
    return EvidenceQuery(
        name=f"{_safe_alias(table_name)}_{metric_alias}_summary",
        purpose=f"Summarize {metric_column} in {table_name}.",
        sql=sql,
        tables=[table_name],
        columns=[f"{table_name}.{metric_column}"],
        metric=_metric_name_for_table_column(semantic_layer or {}, table_name, metric_column),
    )


def _relationship_breakdown_query(semantic_layer, profiles):
    for left_table, left_profile in _ordered_profiles_for_analysis(semantic_layer, profiles):
        table_info = semantic_layer.get("tables", {}).get(left_table, {})
        metric_column = next(iter(left_profile.get("metric_columns") or []), None)
        if not metric_column:
            continue
        for relationship in table_info.get("relationships") or []:
            right_table = relationship.get("target_table")
            right_profile = profiles.get(right_table, {})
            display_column = next(iter(right_profile.get("display_columns") or []), None)
            join_condition = relationship.get("join_condition")
            if not right_table or not display_column or not join_condition:
                continue
            sql = f"""
SELECT
  r.{_quote_identifier(display_column)} AS {_quote_identifier(_safe_alias(display_column))},
  COUNT(*) AS row_count,
  ROUND(SUM(COALESCE(l.{_quote_identifier(metric_column)}, 0)), 2) AS total_{_safe_alias(metric_column)}
FROM {_sql_table(left_table, "l")}
INNER JOIN {_sql_table(right_table, "r")}
  ON {join_condition.replace(left_table + ".", "l.").replace(right_table + ".", "r.")}
GROUP BY r.{_quote_identifier(display_column)}
ORDER BY total_{_safe_alias(metric_column)} DESC
LIMIT 20;
""".strip()
            return EvidenceQuery(
                name=f"{_safe_alias(left_table)}_by_{_safe_alias(right_table)}",
                purpose=f"Find how {left_table} metric values break down by {right_table}.",
                sql=sql,
                tables=[left_table, right_table],
                columns=[
                    f"{left_table}.{metric_column}",
                    f"{right_table}.{display_column}",
                ],
                metric=_metric_name_for_table_column(semantic_layer, left_table, metric_column),
            )
    return None


def build_dataset_analysis_plan(semantic_layer=None, max_queries=6):
    if semantic_layer is None:
        semantic_layer = load_json(str(SEMANTIC_LAYER_PATH))

    profiles = _semantic_profile(semantic_layer)
    queries = []

    if semantic_layer.get("tables"):
        table_count_selects = [
            f"SELECT '{table_name.replace(chr(39), chr(39) * 2)}' AS table_name, COUNT(*) AS row_count FROM {_quote_identifier(table_name)}"
            for table_name in semantic_layer.get("tables", {})
        ]
        queries.append(
            EvidenceQuery(
                name="dataset_row_counts",
                purpose="Compare row counts across all discovered tables.",
                sql="\nUNION ALL\n".join(table_count_selects) + "\nORDER BY row_count DESC;",
                tables=list(semantic_layer.get("tables", {}).keys()),
                columns=[],
            )
        )

    relationship_query = _relationship_breakdown_query(semantic_layer, profiles)
    if relationship_query:
        queries.append(relationship_query)

    for table_name, profile in _ordered_profiles_for_analysis(semantic_layer, profiles):
        if len(queries) >= max_queries:
            break
        metric_column = next(iter(profile["metric_columns"]), None)
        if profile["category_columns"]:
            queries.append(
                _category_breakdown_query(
                    table_name,
                    profile["category_columns"][0],
                    metric_column,
                    semantic_layer,
                )
            )
            continue
        if profile["date_columns"]:
            queries.append(
                _monthly_trend_query(
                    table_name,
                    profile["date_columns"][0],
                    metric_column,
                    semantic_layer,
                )
            )
            continue
        if metric_column:
            queries.append(_numeric_summary_query(table_name, metric_column, semantic_layer))
            continue
        queries.append(_count_query_for_table(table_name))

    return queries[:max_queries]


def _planned_analysis_answer(evidence_items):
    parts = []
    for item in evidence_items:
        if item.get("status") != "ok":
            continue
        rows = _rows(item)
        if not rows:
            continue
        if item.get("name") == "dataset_row_counts":
            top = rows[0]
            parts.append(
                f"The largest table by row count is {top.get('table_name')} with {top.get('row_count')} row(s)."
            )
            continue
        row = rows[-1] if "trend" in item.get("name", "") else rows[0]
        numeric_fields = [
            key for key, value in row.items()
            if key != "row_count" and isinstance(value, (int, float))
        ]
        label_fields = [key for key in row if key not in numeric_fields and key != "row_count"]
        if numeric_fields and label_fields:
            parts.append(
                f"{item.get('purpose')} Leading result: {label_fields[0]}={row.get(label_fields[0])}, "
                f"{numeric_fields[0]}={row.get(numeric_fields[0])}."
            )
        elif "row_count" in row:
            parts.append(f"{item.get('purpose')} Returned {row.get('row_count')} row(s).")

    if not parts:
        return "I planned a broad dataset review, but the evidence queries did not return enough rows to summarize."

    return " ".join(parts)


def _build_planned_dataset_analysis(user_question, evidence_items, semantic_layer):
    plan_questions = [
        {
            "question": item.get("purpose"),
            "evidence": item.get("name"),
            "why": "This evidence query was selected from the active semantic layer.",
        }
        for item in evidence_items
    ]
    failed = [item for item in evidence_items if item.get("status") != "ok"]
    cited_tables, cited_columns = _table_column_citations(
        semantic_layer,
        sorted({table for item in evidence_items for table in item.get("tables", [])}),
        sorted({column for item in evidence_items for column in item.get("columns", [])}),
    )
    metric_defs = [
        _metric_definition(semantic_layer, metric)
        for metric in sorted({item.get("metric") for item in evidence_items if item.get("metric")})
        if _metric_definition(semantic_layer, metric)
    ]
    confidence_score = 0.86 - min(0.4, len(failed) * 0.12)

    analysis = {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "planned_dataset_analysis",
        "Question": user_question,
        "Resolved_Question": "Broad dataset analysis",
        "Executive_Answer": _planned_analysis_answer(evidence_items),
        "Clarification": {"needed": False, "question": None, "decision": None},
        "Assumptions": [
            "A broad analysis should be decomposed into focused, answerable business questions.",
            "The first-pass plan is generated from tables, metrics, relationships, date columns, and categorical columns in the semantic layer.",
            "Each planned question is backed by its own read-only SQL evidence query.",
        ],
        "Definitions": metric_defs,
        "Analysis_Plan": plan_questions,
        "Period_Comparison": None,
        "Anomalies": [],
        "Citations": {
            "metrics": metric_defs,
            "tables": cited_tables,
            "columns": cited_columns,
        },
        "Evidence": evidence_items,
        "Confidence": {
            "level": "high" if confidence_score >= 0.8 else "medium",
            "score": _round_number(confidence_score, 2),
            "reasons": [
                "The answer comes from multiple planned evidence queries.",
                f"{len(evidence_items) - len(failed)} of {len(evidence_items)} planned evidence queries completed successfully.",
            ],
        },
        "Limitations": [
            "This is an initial broad review; deeper root-cause analysis should follow the strongest signals.",
            "The plan is constrained to tables and metrics present in the active semantic layer.",
        ],
        "Suggested_Next_Queries": _example_questions_from_semantic_layer(semantic_layer)[:4],
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }
    return _augment_confidence_reasons(analysis, semantic_layer)


def _evidence_by_name(evidence_items, name):
    for item in evidence_items:
        if item.get("name") == name:
            return item
    return None


def _rows(item):
    if not item or item.get("status") != "ok":
        return []
    return item.get("result_preview") or []


def _question_with_evidence(*, question, why, evidence_name, supporting_facts, tables, columns):
    return {
        "question": question,
        "why": why,
        "source_evidence": evidence_name,
        "supporting_facts": supporting_facts,
        "tables": tables,
        "columns": columns,
    }


def _confidence_for_analysis(evidence_items, period_comparison, limitations):
    error_count = sum(1 for item in evidence_items if item.get("status") != "ok")
    warning_count = sum(
        1
        for item in evidence_items
        for check in item.get("checks", [])
        if check.get("status") == "warning"
    )

    score = 0.92
    reasons = []

    if period_comparison.get("status") != "ok":
        score -= 0.35
        reasons.append("The required period comparison was unavailable.")
    else:
        reasons.append("The main comparison query executed successfully and returned both periods.")

    if error_count:
        score -= min(0.4, error_count * 0.15)
        reasons.append(f"{error_count} evidence query failed.")

    if warning_count:
        score -= min(0.15, warning_count * 0.05)
        reasons.append(f"{warning_count} evidence check produced a warning.")

    if limitations:
        score -= min(0.2, len(limitations) * 0.04)
        reasons.append("The answer has explicit data or semantic limitations.")

    score = max(0.05, min(0.99, score))

    if score >= 0.8:
        level = "high"
    elif score >= 0.55:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": _round_number(score, 2),
        "reasons": reasons,
    }


def _confidence_level(score):
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _unique_preserving_order(values):
    seen = set()
    unique = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _analysis_tables(analysis, sql_output=None):
    tables = []
    for item in analysis.get("Evidence") or []:
        tables.extend(item.get("tables") or [])

    citations = analysis.get("Citations") or {}
    tables.extend(item.get("table") for item in citations.get("tables") or [])

    if sql_output:
        tables.extend(sql_output.get("Selected_Tables") or [])

    return _unique_preserving_order(tables)


def _analysis_metrics(analysis, sql_output=None):
    metrics = []
    for definition in analysis.get("Definitions") or []:
        metrics.append(definition.get("metric"))

    for item in analysis.get("Evidence") or []:
        metrics.append(item.get("metric"))

    if sql_output:
        metrics.extend(sql_output.get("Selected_Metrics") or [])

    return _unique_preserving_order(metrics)


def _has_reviewed_join(analysis):
    for item in analysis.get("Evidence") or []:
        sql = item.get("sql") or ""
        if len(item.get("tables") or []) > 1:
            return True
        if re.search(r"\bJOIN\b", sql, flags=re.IGNORECASE):
            return True
    return False


def _date_range_was_inferred(analysis, sql_output=None):
    text_parts = []
    text_parts.extend(analysis.get("Assumptions") or [])
    if sql_output and sql_output.get("Assumptions"):
        text_parts.append(sql_output.get("Assumptions"))

    joined = " ".join(str(part).lower() for part in text_parts)
    date_signals = [
        "latest complete",
        "latest invoice date",
        "treated as incomplete",
        "last month",
        "date bucketing",
        "available data",
        "inferred",
    ]
    return any(signal in joined for signal in date_signals)


def _add_reason(reason_codes, code, message):
    if any(reason["code"] == code for reason in reason_codes):
        return
    reason_codes.append({"code": code, "message": message})


def _augment_confidence_reasons(analysis, semantic_layer=None, sql_output=None):
    confidence = dict(analysis.get("Confidence") or {})
    score = confidence.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5

    reason_codes = []
    metrics = _analysis_metrics(analysis, sql_output)
    evidence_items = analysis.get("Evidence") or []
    limitations = analysis.get("Limitations") or []
    tables = _analysis_tables(analysis, sql_output)
    evidence_errors = [item for item in evidence_items if item.get("status") != "ok"]
    evidence_warnings = [
        check
        for item in evidence_items
        for check in item.get("checks", [])
        if check.get("status") == "warning"
    ]
    reviewed_join = _has_reviewed_join(analysis)
    date_inferred = _date_range_was_inferred(analysis, sql_output)

    if metrics:
        _add_reason(
            reason_codes,
            "exact_metric_found",
            "Exact metric found: " + ", ".join(metrics) + ".",
        )
    else:
        _add_reason(
            reason_codes,
            "no_exact_metric",
            "No exact semantic metric was selected for this answer.",
        )

    if evidence_items and not evidence_errors:
        _add_reason(
            reason_codes,
            "evidence_passed",
            f"{len(evidence_items)} evidence query(s) completed successfully.",
        )
    elif evidence_errors:
        _add_reason(
            reason_codes,
            "evidence_failed",
            f"{len(evidence_errors)} evidence query(s) failed or were unavailable.",
        )

    if evidence_warnings:
        _add_reason(
            reason_codes,
            "evidence_warning",
            f"{len(evidence_warnings)} evidence check(s) returned a warning.",
        )

    if date_inferred:
        _add_reason(
            reason_codes,
            "date_range_inferred",
            "Date range was inferred from the question or available data.",
        )
        score = min(score, 0.79)

    if len(tables) > 1 and reviewed_join:
        _add_reason(
            reason_codes,
            "join_path_reviewed",
            "Join path was reviewed across: " + ", ".join(tables[:4]) + ".",
        )
    elif len(tables) > 1:
        _add_reason(
            reason_codes,
            "missing_join_path_review",
            "No join path was reviewed for the selected tables.",
        )
        score = min(score, 0.54)
    else:
        _add_reason(
            reason_codes,
            "single_table_answer",
            "No join path was needed for the selected table.",
        )

    if limitations:
        _add_reason(
            reason_codes,
            "limitations_present",
            f"{len(limitations)} limitation(s) apply to this answer.",
        )

    score = max(0.05, min(0.99, score))
    level = _confidence_level(score)
    lead_priority = [
        "missing_join_path_review",
        "evidence_failed",
        "date_range_inferred",
        "evidence_warning",
        "limitations_present",
        "exact_metric_found",
        "evidence_passed",
        "single_table_answer",
        "no_exact_metric",
    ]
    lead_reason = next(
        (
            reason["message"]
            for code in lead_priority
            for reason in reason_codes
            if reason["code"] == code
        ),
        "Confidence is based on available evidence.",
    )
    summary = f"{level.title()} confidence because {lead_reason[:1].lower()}{lead_reason[1:]}"

    confidence["score"] = _round_number(score, 2)
    confidence["level"] = level
    confidence["badge"] = level.title()
    confidence["summary"] = summary.rstrip(".") + "."
    confidence["reason_codes"] = reason_codes
    analysis["Confidence"] = confidence
    return analysis


def _build_revenue_drop_analysis(user_question, sql_output, evidence_items, semantic_layer):
    analysis = _build_planned_dataset_analysis(user_question, evidence_items, semantic_layer)
    analysis["Mode"] = "metric_driver_investigation"
    analysis["Resolved_Question"] = sql_output.get("Resolved_Question") or user_question
    analysis["Clarification"] = {
        "needed": False,
        "question": sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions"),
        "decision": sql_output.get("Clarification_Decision"),
        "handled_as_assumption": bool(sql_output.get("Requires_Clarification")),
    }
    if sql_output.get("Assumptions"):
        analysis.setdefault("Assumptions", []).append(sql_output.get("Assumptions"))
    return _augment_confidence_reasons(analysis, semantic_layer, sql_output)


def _generic_analysis_from_sql(user_question, sql_output, sql_result, generated_sql, semantic_layer):
    no_sql = not generated_sql
    is_error = isinstance(sql_result, str) or no_sql
    rows = [] if is_error else list(sql_result or [])
    tables = sql_output.get("Selected_Tables") or _extract_sql_tables(generated_sql)
    metrics = sql_output.get("Selected_Metrics") or []
    metric_defs = [
        _metric_definition(semantic_layer, metric)
        for metric in metrics
        if _metric_definition(semantic_layer, metric)
    ]
    cited_tables, _ = _table_column_citations(semantic_layer, tables, [])

    if no_sql:
        answer = "I could not produce a reliable answer because no executable SQL was generated."
    elif is_error:
        answer = "I could not produce a reliable answer because the generated SQL failed during validation or execution."
    elif rows:
        answer = f"I found {len(rows)} row(s) matching the question. Review the answer table and supporting SQL evidence below."
    else:
        answer = "The query executed successfully but returned no rows for the stated assumptions."

    evidence = [
        {
            "name": "generated_sql_result",
            "purpose": "Answer the user's question with the generated query.",
            "status": "error" if is_error else "ok",
            "error": "No executable SQL was generated." if no_sql else (sql_result if is_error else None),
            "row_count": None if is_error else len(rows),
            "result_preview": rows[:MAX_PREVIEW_ROWS],
            "sql": generated_sql,
            "tables": tables,
            "columns": [],
            "metric": metrics[0] if metrics else None,
            "checks": [
                {
                    "name": "read_only_guardrail_and_query_plan",
                    "status": "failed" if is_error else "passed",
                    "detail": (
                        "No executable SQL was generated."
                        if no_sql
                        else sql_result if is_error
                        else "SQLite guardrails, table validation, and query plan checks passed."
                    ),
                },
                {
                    "name": "result_shape",
                    "status": "failed" if is_error else "passed",
                    "detail": (
                        "No result was available because no executable SQL was generated."
                        if no_sql
                        else sql_result if is_error
                        else f"Returned {len(rows)} row(s)."
                    ),
                },
            ],
        }
    ]

    limitations = []
    if no_sql:
        limitations.append("The SQL generation step did not return executable SQL.")
    elif is_error:
        limitations.append("The generated SQL failed validation or execution.")
    if not rows and not is_error:
        limitations.append("No rows matched the generated query filters.")

    confidence = _confidence_for_analysis(
        evidence,
        {"status": "ok" if not is_error else "unavailable"},
        limitations,
    )

    analysis = {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "single_query_analysis",
        "Question": user_question,
        "Resolved_Question": sql_output.get("Resolved_Question") or user_question,
        "Executive_Answer": answer,
        "Clarification": {
            "needed": False,
            "question": None,
            "decision": sql_output.get("Clarification_Decision"),
        },
        "Assumptions": [
            sql_output.get("Assumptions") or "Used the assumptions returned by the SQL generation agent."
        ],
        "Definitions": metric_defs,
        "Period_Comparison": None,
        "Anomalies": [],
        "Citations": {
            "metrics": metric_defs,
            "tables": cited_tables,
            "columns": [],
        },
        "Evidence": evidence,
        "Confidence": confidence,
        "Limitations": limitations,
        "Suggested_Next_Queries": [],
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }
    return _augment_confidence_reasons(analysis, semantic_layer, sql_output)


def _clarification_analysis(user_question, sql_output):
    question = sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions")
    analysis = {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "clarification_required",
        "Question": user_question,
        "Resolved_Question": sql_output.get("Resolved_Question") or user_question,
        "Executive_Answer": "I need one clarification before running analysis.",
        "Clarification": {
            "needed": True,
            "question": question,
            "decision": sql_output.get("Clarification_Decision"),
        },
        "Assumptions": [],
        "Definitions": [],
        "Period_Comparison": None,
        "Anomalies": [],
        "Citations": {
            "metrics": [],
            "tables": [],
            "columns": [],
        },
        "Evidence": [],
        "Confidence": {
            "level": "low",
            "score": 0.1,
            "reasons": ["Analysis is paused until the ambiguity is resolved."],
        },
        "Limitations": [sql_output.get("Assumptions") or "The question is underspecified."],
        "Suggested_Next_Queries": [question] if question else [],
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }
    return _augment_confidence_reasons(analysis, sql_output=sql_output)


def _analysis_synthesis_prompt(analysis):
    compact = dict(analysis)
    compact["Evidence"] = [
        {
            "name": item.get("name"),
            "purpose": item.get("purpose"),
            "status": item.get("status"),
            "row_count": item.get("row_count"),
            "result_preview": item.get("result_preview", [])[:8],
            "tables": item.get("tables"),
            "columns": item.get("columns"),
        }
        for item in analysis.get("Evidence", [])
    ]

    return f"""
You are the final analyst in an AI-native NL-to-SQL system.
Use the structured evidence to improve the executive answer without inventing facts.

Return strict JSON:
{{
  "Executive_Answer": "<concise business answer>",
  "Confidence_Adjustment": "<optional short note or null>",
  "Limitations": ["<keep or refine limitations>"]
}}

Do not generate or edit `Suggested_Next_Queries`; those are produced by deterministic
evidence checks and must remain unchanged.

Structured evidence:
{json.dumps(compact, indent=2, default=str)}
""".strip()


def _try_llm_synthesis(analysis, model_name):
    if analysis.get("Clarification", {}).get("needed"):
        return analysis

    try:
        response_text = gemini_call(
            model_name,
            _analysis_synthesis_prompt(analysis),
            trace_name="gemini-analysis-synthesis",
        )
        synthesized = load_string_as_json(response_text)
    except Exception as exc:
        analysis.setdefault("Synthesis_Warnings", []).append(
            f"LLM synthesis failed; deterministic analysis was used. Error: {exc}"
        )
        return analysis

    if synthesized.get("Executive_Answer"):
        analysis["Executive_Answer"] = synthesized["Executive_Answer"]
    if isinstance(synthesized.get("Limitations"), list) and synthesized["Limitations"]:
        analysis["Limitations"] = synthesized["Limitations"]
    if synthesized.get("Confidence_Adjustment"):
        analysis.setdefault("Confidence", {}).setdefault("reasons", []).append(
            synthesized["Confidence_Adjustment"]
        )
    return analysis


def run_ai_native_analysis(
    user_question,
    *,
    model_name=None,
    thread_id=None,
    sql_runner=None,
    use_llm_synthesis=True,
    answer_mode=DEFAULT_ANSWER_MODE,
):
    model_name = model_name or get_default_model_name()
    sql_runner = sql_runner or load_default_sql_runner()
    answer_mode = normalize_answer_mode(answer_mode)

    with traced_span(
        "ai-native-analysis-workflow",
        input={
            "user_question": user_question,
            "thread_id": thread_id,
            "answer_mode": answer_mode,
        },
    ) as span:
        semantic_layer = load_json(str(SEMANTIC_LAYER_PATH))
        intent = infer_analysis_intent(user_question, semantic_layer)

        if intent["mode"] == "data_overview":
            evidence_items = [
                _run_evidence_query(query, sql_runner)
                for query in _overview_profile_queries(semantic_layer)
            ]
            analysis = _build_data_overview_analysis(
                user_question,
                evidence_items,
                semantic_layer,
            )
            sql_output = {
                "SQL": None,
                "Resolved_Question": analysis["Resolved_Question"],
                "Requires_Clarification": False,
                "Analysis": analysis,
                "Analysis_Mode": analysis.get("Mode"),
                "Executive_Answer": analysis.get("Executive_Answer"),
            }
            sql_output = apply_answer_mode_to_sql_output(sql_output, answer_mode)
            analysis = sql_output["Analysis"]
            record_sql_execution_for_thread(thread_id, None)
            safe_update_observation(span, output={"sql_output": sql_output, "analysis": analysis})
            return {
                "sql_output": sql_output,
                "sql_result": None,
                "analysis": analysis,
                "primary_result": None,
            }

        if intent["mode"] == "planned_dataset_analysis":
            evidence_items = [
                _run_evidence_query(query, sql_runner)
                for query in build_dataset_analysis_plan(semantic_layer)
            ]
            analysis = _build_planned_dataset_analysis(
                user_question,
                evidence_items,
                semantic_layer,
            )
            sql_output = {
                "SQL": None,
                "Resolved_Question": analysis["Resolved_Question"],
                "Requires_Clarification": False,
                "Analysis": analysis,
                "Analysis_Mode": analysis.get("Mode"),
                "Executive_Answer": analysis.get("Executive_Answer"),
            }
            sql_output = apply_answer_mode_to_sql_output(sql_output, answer_mode)
            analysis = sql_output["Analysis"]
            record_sql_execution_for_thread(thread_id, None)
            safe_update_observation(span, output={"sql_output": sql_output, "analysis": analysis})
            return {
                "sql_output": sql_output,
                "sql_result": None,
                "analysis": analysis,
                "primary_result": None,
            }

        sql_output = generate_sql_for_question(
            user_question,
            model_name=model_name,
            thread_id=thread_id,
        )
        sql_output = apply_answer_mode_to_sql_output(sql_output, answer_mode)

        if (
            intent["mode"] != "metric_driver_investigation"
            and (sql_output.get("Requires_Clarification") or sql_output.get("Clarification_Limit_Reached"))
        ):
            analysis = _clarification_analysis(user_question, sql_output)
            sql_output["Analysis"] = analysis
            sql_output = apply_answer_mode_to_sql_output(sql_output, answer_mode)
            analysis = sql_output["Analysis"]
            record_sql_execution_for_thread(thread_id, None)
            safe_update_observation(span, output={"sql_output": sql_output, "analysis": analysis})
            return {
                "sql_output": sql_output,
                "sql_result": None,
                "analysis": analysis,
                "primary_result": None,
            }

        generated_sql = sql_output.get("SQL")
        generated_sql_result = None
        if generated_sql:
            generated_sql_result = sql_runner.run_query(generated_sql)
            record_sql_execution_for_thread(thread_id, generated_sql_result)
        else:
            record_sql_execution_for_thread(thread_id, None)

        if intent["mode"] == "metric_driver_investigation":
            evidence_items = [
                _run_evidence_query(query, sql_runner)
                for query in build_metric_driver_evidence_queries(intent.get("metric"), semantic_layer)
            ]
            analysis = _build_revenue_drop_analysis(
                user_question,
                sql_output,
                evidence_items,
                semantic_layer,
            )
            primary_result = (
                evidence_items[0] if evidence_items else {}
            ).get("result_preview")
        else:
            analysis = _generic_analysis_from_sql(
                user_question,
                sql_output,
                generated_sql_result,
                generated_sql,
                semantic_layer,
            )
            primary_result = generated_sql_result

        if generated_sql:
            analysis["Generated_SQL_Evidence"] = {
                "sql": generated_sql,
                "status": "error" if isinstance(generated_sql_result, str) else "ok",
                "result_preview": [] if isinstance(generated_sql_result, str) else list(generated_sql_result or [])[:MAX_PREVIEW_ROWS],
                "error": generated_sql_result if isinstance(generated_sql_result, str) else None,
            }

        if use_llm_synthesis:
            analysis = _try_llm_synthesis(analysis, model_name)

        analysis = _augment_confidence_reasons(analysis, semantic_layer, sql_output)

        if analysis.get("Clarification", {}).get("handled_as_assumption"):
            sql_output["Original_Requires_Clarification"] = bool(sql_output.get("Requires_Clarification"))
            sql_output["Non_Blocking_Clarification_Question"] = (
                sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions")
            )
            sql_output["Requires_Clarification"] = False
            sql_output["Clarification_Question"] = None

        sql_output["Analysis"] = analysis
        sql_output["Analysis_Mode"] = analysis.get("Mode")
        sql_output["Executive_Answer"] = analysis.get("Executive_Answer")
        sql_output = apply_answer_mode_to_sql_output(sql_output, answer_mode)
        analysis = sql_output["Analysis"]

        safe_update_observation(
            span,
            output={
                "analysis_mode": analysis.get("Mode"),
                "answer_mode": answer_mode,
                "confidence": analysis.get("Confidence"),
                "evidence_count": len(analysis.get("Evidence", [])),
                "sql_output": sql_output,
            },
        )
        return {
            "sql_output": sql_output,
            "sql_result": generated_sql_result,
            "analysis": analysis,
            "primary_result": primary_result,
        }
