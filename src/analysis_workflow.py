import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
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


def _format_money(value, unit="INR"):
    if value is None:
        return f"{unit} n/a"

    amount = _safe_float(value)
    return f"{unit} {amount:,.2f}"


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


def _data_freshness_sql():
    return """
SELECT
  MIN(invoice_date) AS min_invoice_date,
  MAX(invoice_date) AS max_invoice_date,
  COUNT(*) AS invoice_rows,
  SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) AS paid_invoice_rows
FROM invoices;
""".strip()


def _revenue_period_comparison_sql():
    return """
WITH bounds AS (
  SELECT
    MAX(invoice_date) AS max_invoice_date,
    date(MAX(invoice_date), 'start of month', '-1 month') AS current_start,
    date(MAX(invoice_date), 'start of month') AS current_end_exclusive,
    date(MAX(invoice_date), 'start of month', '-2 months') AS previous_start,
    date(MAX(invoice_date), 'start of month', '-1 month') AS previous_end_exclusive
  FROM invoices
),
periods AS (
  SELECT
    'previous_period' AS period_key,
    previous_start AS period_start,
    previous_end_exclusive AS period_end_exclusive,
    max_invoice_date
  FROM bounds
  UNION ALL
  SELECT
    'current_period' AS period_key,
    current_start AS period_start,
    current_end_exclusive AS period_end_exclusive,
    max_invoice_date
  FROM bounds
)
SELECT
  p.period_key,
  p.period_start,
  date(p.period_end_exclusive, '-1 day') AS period_end,
  p.max_invoice_date AS data_max_invoice_date,
  ROUND(COALESCE(SUM(CASE WHEN i.status = 'paid' THEN COALESCE(i.grand_total, 0) ELSE 0 END), 0), 2) AS revenue,
  SUM(CASE WHEN i.status = 'paid' THEN 1 ELSE 0 END) AS paid_invoice_count,
  ROUND(AVG(CASE WHEN i.status = 'paid' THEN COALESCE(i.grand_total, 0) END), 2) AS average_paid_invoice_value
FROM periods AS p
LEFT JOIN invoices AS i
  ON i.invoice_date >= p.period_start
 AND i.invoice_date < p.period_end_exclusive
GROUP BY
  p.period_key,
  p.period_start,
  p.period_end_exclusive,
  p.max_invoice_date
ORDER BY
  p.period_start;
""".strip()


def _revenue_monthly_trend_sql():
    return """
WITH bounds AS (
  SELECT
    date(MAX(invoice_date), 'start of month') AS incomplete_month_start,
    date(MAX(invoice_date), 'start of month', '-8 months') AS trend_start
  FROM invoices
)
SELECT
  strftime('%Y-%m', i.invoice_date) AS month,
  ROUND(SUM(CASE WHEN i.status = 'paid' THEN COALESCE(i.grand_total, 0) ELSE 0 END), 2) AS revenue,
  SUM(CASE WHEN i.status = 'paid' THEN 1 ELSE 0 END) AS paid_invoice_count,
  ROUND(AVG(CASE WHEN i.status = 'paid' THEN COALESCE(i.grand_total, 0) END), 2) AS average_paid_invoice_value
FROM invoices AS i
CROSS JOIN bounds AS b
WHERE i.invoice_date >= b.trend_start
  AND i.invoice_date < b.incomplete_month_start
GROUP BY
  strftime('%Y-%m', i.invoice_date)
ORDER BY
  month;
""".strip()


def _revenue_vendor_driver_sql():
    return """
WITH bounds AS (
  SELECT
    date(MAX(invoice_date), 'start of month', '-1 month') AS current_start,
    date(MAX(invoice_date), 'start of month') AS current_end_exclusive,
    date(MAX(invoice_date), 'start of month', '-2 months') AS previous_start,
    date(MAX(invoice_date), 'start of month', '-1 month') AS previous_end_exclusive
  FROM invoices
),
vendor_revenue AS (
  SELECT
    v.id AS vendor_id,
    v.name AS vendor_name,
    ROUND(SUM(CASE
      WHEN i.status = 'paid'
       AND i.invoice_date >= b.previous_start
       AND i.invoice_date < b.previous_end_exclusive
      THEN COALESCE(i.grand_total, 0)
      ELSE 0
    END), 2) AS previous_revenue,
    ROUND(SUM(CASE
      WHEN i.status = 'paid'
       AND i.invoice_date >= b.current_start
       AND i.invoice_date < b.current_end_exclusive
      THEN COALESCE(i.grand_total, 0)
      ELSE 0
    END), 2) AS current_revenue,
    SUM(CASE
      WHEN i.status = 'paid'
       AND i.invoice_date >= b.previous_start
       AND i.invoice_date < b.previous_end_exclusive
      THEN 1
      ELSE 0
    END) AS previous_paid_invoice_count,
    SUM(CASE
      WHEN i.status = 'paid'
       AND i.invoice_date >= b.current_start
       AND i.invoice_date < b.current_end_exclusive
      THEN 1
      ELSE 0
    END) AS current_paid_invoice_count
  FROM invoices AS i
  INNER JOIN vendors AS v
    ON i.vendor_id = v.id
  CROSS JOIN bounds AS b
  WHERE i.invoice_date >= b.previous_start
    AND i.invoice_date < b.current_end_exclusive
  GROUP BY
    v.id,
    v.name
)
SELECT
  vendor_id,
  vendor_name,
  previous_revenue,
  current_revenue,
  ROUND(current_revenue - previous_revenue, 2) AS revenue_change,
  previous_paid_invoice_count,
  current_paid_invoice_count,
  current_paid_invoice_count - previous_paid_invoice_count AS paid_invoice_count_change
FROM vendor_revenue
WHERE ABS(current_revenue - previous_revenue) > 0
ORDER BY
  revenue_change ASC
LIMIT 10;
""".strip()


def _revenue_status_mix_sql():
    return """
WITH bounds AS (
  SELECT
    date(MAX(invoice_date), 'start of month', '-1 month') AS current_start,
    date(MAX(invoice_date), 'start of month') AS current_end_exclusive,
    date(MAX(invoice_date), 'start of month', '-2 months') AS previous_start,
    date(MAX(invoice_date), 'start of month', '-1 month') AS previous_end_exclusive
  FROM invoices
),
status_periods AS (
  SELECT
    CASE
      WHEN i.invoice_date >= b.previous_start AND i.invoice_date < b.previous_end_exclusive THEN 'previous_period'
      WHEN i.invoice_date >= b.current_start AND i.invoice_date < b.current_end_exclusive THEN 'current_period'
    END AS period_key,
    i.status,
    COUNT(*) AS invoice_count,
    ROUND(SUM(COALESCE(i.grand_total, 0)), 2) AS invoice_value
  FROM invoices AS i
  CROSS JOIN bounds AS b
  WHERE i.invoice_date >= b.previous_start
    AND i.invoice_date < b.current_end_exclusive
  GROUP BY
    period_key,
    i.status
)
SELECT
  period_key,
  status,
  invoice_count,
  invoice_value
FROM status_periods
WHERE period_key IS NOT NULL
ORDER BY
  period_key,
  invoice_value DESC;
""".strip()


def build_revenue_drop_evidence_queries():
    common_columns = ["invoices.invoice_date", "invoices.status", "invoices.grand_total"]
    return [
        EvidenceQuery(
            name="data_freshness",
            purpose="Inspect the available invoice date range before interpreting relative periods.",
            sql=_data_freshness_sql(),
            tables=["invoices"],
            columns=["invoices.invoice_date", "invoices.status"],
        ),
        EvidenceQuery(
            name="period_comparison",
            purpose="Compare revenue for the latest complete invoice month against the prior month.",
            sql=_revenue_period_comparison_sql(),
            tables=["invoices"],
            columns=common_columns,
            metric="revenue",
        ),
        EvidenceQuery(
            name="monthly_trend",
            purpose="Check whether the latest complete month is unusual against recent monthly history.",
            sql=_revenue_monthly_trend_sql(),
            tables=["invoices"],
            columns=common_columns,
            metric="revenue",
        ),
        EvidenceQuery(
            name="vendor_drivers",
            purpose="Find vendors contributing most to the month-over-month revenue change.",
            sql=_revenue_vendor_driver_sql(),
            tables=["invoices", "vendors"],
            columns=[
                "invoices.invoice_date",
                "invoices.status",
                "invoices.grand_total",
                "invoices.vendor_id",
                "vendors.id",
                "vendors.name",
            ],
            metric="revenue",
        ),
        EvidenceQuery(
            name="status_mix",
            purpose="Check whether invoices moved out of paid status in the comparison month.",
            sql=_revenue_status_mix_sql(),
            tables=["invoices"],
            columns=common_columns,
            metric="revenue",
        ),
    ]


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
    revenue_terms = ["revenue", "paid invoice value", "realized invoice value", "paid bills"]
    why_terms = ["why", "reason", "driver", "cause", "explain"]
    drop_terms = ["drop", "decline", "fell", "fall", "decrease", "down"]

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

    if (
        any(term in normalized for term in revenue_terms)
        and any(term in normalized for term in why_terms)
        and any(term in normalized for term in drop_terms)
    ):
        return {
            "mode": "metric_driver_investigation",
            "metric": "revenue",
            "reason": "The question asks for drivers behind a revenue decline, which needs comparison and evidence queries.",
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
    examples = [
        "Which vendors have the highest paid invoice value this year?",
        "Which invoices are overdue or still unpaid?",
        "How much spend is outstanding by status?",
        "Which purchase orders are closed versus still open?",
        "Which products or receipts have the highest rejection rate?",
    ]

    for metric_name, metric in metrics.items():
        description = metric.get("description")
        if description:
            examples.append(f"Show {metric_name.replace('_', ' ')}: {description.lower()}.")

    return examples[:8]


def _overview_profile_queries(semantic_layer):
    queries = []
    for table_name in semantic_layer.get("tables", {}):
        queries.append(
            EvidenceQuery(
                name=f"{table_name}_row_count",
                purpose=f"Count available records in {table_name}.",
                sql=f"SELECT COUNT(*) AS row_count FROM {table_name};",
                tables=[table_name],
                columns=[],
            )
        )

    date_profiles = {
        "invoices": "invoice_date",
        "purchase_orders": "po_date",
        "payments": "payment_date",
        "grns": "grn_date",
    }
    for table_name, date_column in date_profiles.items():
        if table_name not in semantic_layer.get("tables", {}):
            continue
        queries.append(
            EvidenceQuery(
                name=f"{table_name}_date_range",
                purpose=f"Find available date coverage for {table_name}.",
                sql=(
                    f"SELECT MIN({date_column}) AS min_date, "
                    f"MAX({date_column}) AS max_date FROM {table_name};"
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
            row_counts[table_name] = rows[0].get("row_count")
        elif item["name"].endswith("_date_range"):
            table_name = item["name"][: -len("_date_range")]
            date_ranges[table_name] = rows[0]

    top_tables = [
        f"{table['table']} ({row_counts.get(table['table'], 'unknown')} rows): {table.get('description')}"
        for table in table_groups[:8]
    ]
    metric_names = list(semantic_layer.get("metrics", {}).keys())
    examples = _example_questions_from_semantic_layer(semantic_layer)

    answer = (
        "You can ask about accounts-payable operations: vendors, invoices, payments, "
        "purchase orders, goods receipts, departments, companies, products, approval rules, "
        "spend, liability, paid invoice value, invoice status, fulfillment, and rejection rate. "
        "I can also run a broader review by planning several focused questions and checking the data behind each one."
    )

    return {
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
            "The overview explains available AP data; it does not infer business context outside the database."
        ],
        "Suggested_Next_Queries": examples,
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }


def build_dataset_analysis_plan():
    return [
        EvidenceQuery(
            name="invoice_status_liability",
            purpose="Quantify invoice workload and value by status.",
            sql="""
SELECT
  status,
  COUNT(*) AS invoice_count,
  ROUND(SUM(COALESCE(grand_total, 0)), 2) AS invoice_value
FROM invoices
GROUP BY status
ORDER BY invoice_value DESC;
""".strip(),
            tables=["invoices"],
            columns=["invoices.status", "invoices.grand_total"],
            metric="total_liability",
        ),
        EvidenceQuery(
            name="monthly_paid_invoice_value",
            purpose="Review the paid invoice value trend by month.",
            sql="""
SELECT
  strftime('%Y-%m', invoice_date) AS month,
  ROUND(SUM(CASE WHEN status = 'paid' THEN COALESCE(grand_total, 0) ELSE 0 END), 2) AS paid_invoice_value,
  SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) AS paid_invoice_count
FROM invoices
GROUP BY strftime('%Y-%m', invoice_date)
ORDER BY month;
""".strip(),
            tables=["invoices"],
            columns=["invoices.invoice_date", "invoices.status", "invoices.grand_total"],
            metric="revenue",
        ),
        EvidenceQuery(
            name="top_vendors_by_paid_value",
            purpose="Find vendors with the largest paid invoice value.",
            sql="""
SELECT
  v.id AS vendor_id,
  v.name AS vendor_name,
  ROUND(SUM(CASE WHEN i.status = 'paid' THEN COALESCE(i.grand_total, 0) ELSE 0 END), 2) AS paid_invoice_value,
  SUM(CASE WHEN i.status = 'paid' THEN 1 ELSE 0 END) AS paid_invoice_count
FROM invoices AS i
INNER JOIN vendors AS v ON i.vendor_id = v.id
GROUP BY v.id, v.name
ORDER BY paid_invoice_value DESC
LIMIT 10;
""".strip(),
            tables=["invoices", "vendors"],
            columns=["invoices.vendor_id", "invoices.status", "invoices.grand_total", "vendors.name"],
            metric="revenue",
        ),
        EvidenceQuery(
            name="purchase_order_fulfillment",
            purpose="Summarize purchase order status and value.",
            sql="""
SELECT
  status,
  COUNT(*) AS purchase_order_count,
  ROUND(SUM(COALESCE(total_amount, 0)), 2) AS purchase_order_value
FROM purchase_orders
GROUP BY status
ORDER BY purchase_order_value DESC;
""".strip(),
            tables=["purchase_orders"],
            columns=["purchase_orders.status", "purchase_orders.total_amount"],
            metric="order_fulfillment_count",
        ),
        EvidenceQuery(
            name="goods_rejection_rate",
            purpose="Measure rejected received quantity by product.",
            sql="""
SELECT
  p.id AS product_id,
  p.name AS product_name,
  ROUND(SUM(COALESCE(gli.quantity_rejected, 0)), 2) AS rejected_quantity,
  ROUND(SUM(COALESCE(gli.quantity_received, 0)), 2) AS received_quantity,
  ROUND(
    SUM(COALESCE(gli.quantity_rejected, 0)) * 100.0 /
    NULLIF(SUM(COALESCE(gli.quantity_received, 0) + COALESCE(gli.quantity_rejected, 0)), 0),
    2
  ) AS rejection_rate
FROM grn_line_items AS gli
INNER JOIN products AS p ON gli.product_id = p.id
GROUP BY p.id, p.name
HAVING rejected_quantity > 0
ORDER BY rejection_rate DESC, rejected_quantity DESC
LIMIT 10;
""".strip(),
            tables=["grn_line_items", "products"],
            columns=[
                "grn_line_items.quantity_rejected",
                "grn_line_items.quantity_received",
                "products.name",
            ],
            metric="rejection_rate",
        ),
    ]


def _planned_analysis_answer(evidence_items):
    status_rows = _rows(_evidence_by_name(evidence_items, "invoice_status_liability"))
    vendor_rows = _rows(_evidence_by_name(evidence_items, "top_vendors_by_paid_value"))
    po_rows = _rows(_evidence_by_name(evidence_items, "purchase_order_fulfillment"))
    rejection_rows = _rows(_evidence_by_name(evidence_items, "goods_rejection_rate"))
    monthly_rows = _rows(_evidence_by_name(evidence_items, "monthly_paid_invoice_value"))

    largest_status = max(
        status_rows,
        key=lambda row: _safe_float(row.get("invoice_value")),
        default={},
    )
    top_vendor = vendor_rows[0] if vendor_rows else {}
    latest_month = monthly_rows[-1] if monthly_rows else {}
    largest_po_status = max(
        po_rows,
        key=lambda row: _safe_float(row.get("purchase_order_value")),
        default={},
    )
    top_rejection = rejection_rows[0] if rejection_rows else {}

    parts = []
    if largest_status:
        parts.append(
            f"The largest invoice status bucket is {largest_status.get('status')} "
            f"at {_format_money(largest_status.get('invoice_value'))} across "
            f"{largest_status.get('invoice_count')} invoice(s)."
        )
    if latest_month:
        parts.append(
            f"The latest invoice month in the trend is {latest_month.get('month')} with "
            f"{_format_money(latest_month.get('paid_invoice_value'))} paid invoice value."
        )
    if top_vendor:
        parts.append(
            f"The top paid-value vendor is {top_vendor.get('vendor_name')} at "
            f"{_format_money(top_vendor.get('paid_invoice_value'))}."
        )
    if largest_po_status:
        parts.append(
            f"Purchase order value is most concentrated in {largest_po_status.get('status')} "
            f"orders at {_format_money(largest_po_status.get('purchase_order_value'))}."
        )
    if top_rejection:
        parts.append(
            f"The highest product rejection signal is {top_rejection.get('product_name')} "
            f"with {_format_percent(top_rejection.get('rejection_rate'))} rejected quantity."
        )

    if not parts:
        return "I planned a broad dataset review, but the evidence queries did not return enough rows to summarize."

    return " ".join(parts)


def _build_planned_dataset_analysis(user_question, evidence_items, semantic_layer):
    plan_questions = [
        {
            "question": "What is the invoice value and count by status?",
            "evidence": "invoice_status_liability",
            "why": "Starts with operational exposure and outstanding workload.",
        },
        {
            "question": "How is paid invoice value trending by month?",
            "evidence": "monthly_paid_invoice_value",
            "why": "Shows whether paid value is rising, falling, or volatile over time.",
        },
        {
            "question": "Which vendors contribute the most paid invoice value?",
            "evidence": "top_vendors_by_paid_value",
            "why": "Identifies concentration and vendor-level drivers.",
        },
        {
            "question": "How are purchase orders distributed by status?",
            "evidence": "purchase_order_fulfillment",
            "why": "Checks procurement execution and open order value.",
        },
        {
            "question": "Which products have the highest receipt rejection rate?",
            "evidence": "goods_rejection_rate",
            "why": "Surfaces quality or receiving issues.",
        },
    ]
    failed = [item for item in evidence_items if item.get("status") != "ok"]
    cited_tables, cited_columns = _table_column_citations(
        semantic_layer,
        sorted({table for item in evidence_items for table in item.get("tables", [])}),
        sorted({column for item in evidence_items for column in item.get("columns", [])}),
    )
    metric_defs = [
        _metric_definition(semantic_layer, metric)
        for metric in ["total_liability", "revenue", "order_fulfillment_count", "rejection_rate"]
        if _metric_definition(semantic_layer, metric)
    ]
    confidence_score = 0.86 - min(0.4, len(failed) * 0.12)

    return {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "planned_dataset_analysis",
        "Question": user_question,
        "Resolved_Question": "Broad AP dataset analysis",
        "Executive_Answer": _planned_analysis_answer(evidence_items),
        "Clarification": {"needed": False, "question": None, "decision": None},
        "Assumptions": [
            "A broad analysis should be decomposed into focused, answerable business questions.",
            "The first-pass plan covers invoice exposure, paid-value trend, vendor concentration, purchase-order status, and receipt quality.",
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
            "The plan is constrained to the AP tables and metrics present in the semantic layer.",
        ],
        "Suggested_Next_Queries": [
            "Explain the largest unpaid or pending invoice status bucket by vendor.",
            "Investigate month-over-month changes in paid invoice value for the top vendors.",
            "List high-value open purchase orders and their requesting departments.",
            "Show receipt rejection details for the products with the highest rejection rate.",
        ],
        "Suggested_Next_Query_Evidence": [],
        "SQL_Is_Supporting_Evidence": True,
    }


def _evidence_by_name(evidence_items, name):
    for item in evidence_items:
        if item.get("name") == name:
            return item
    return None


def _rows(item):
    if not item or item.get("status") != "ok":
        return []
    return item.get("result_preview") or []


def _period_comparison_from_evidence(evidence_items):
    period_rows = _rows(_evidence_by_name(evidence_items, "period_comparison"))
    previous_row = next((row for row in period_rows if row.get("period_key") == "previous_period"), None)
    current_row = next((row for row in period_rows if row.get("period_key") == "current_period"), None)

    if not previous_row or not current_row:
        return {
            "status": "unavailable",
            "reason": "The period comparison query did not return both current and previous periods.",
        }

    previous_revenue = _safe_float(previous_row.get("revenue"))
    current_revenue = _safe_float(current_row.get("revenue"))
    absolute_change = current_revenue - previous_revenue
    percent_change = _safe_percent_change(current_revenue, previous_revenue)

    if absolute_change < 0:
        direction = "down"
    elif absolute_change > 0:
        direction = "up"
    else:
        direction = "flat"

    return {
        "status": "ok",
        "metric": "revenue",
        "unit": "INR",
        "current_period": {
            "label": str(current_row.get("period_start"))[:7],
            "start": current_row.get("period_start"),
            "end": current_row.get("period_end"),
            "value": _round_number(current_revenue),
            "paid_invoice_count": current_row.get("paid_invoice_count"),
            "average_paid_invoice_value": current_row.get("average_paid_invoice_value"),
        },
        "previous_period": {
            "label": str(previous_row.get("period_start"))[:7],
            "start": previous_row.get("period_start"),
            "end": previous_row.get("period_end"),
            "value": _round_number(previous_revenue),
            "paid_invoice_count": previous_row.get("paid_invoice_count"),
            "average_paid_invoice_value": previous_row.get("average_paid_invoice_value"),
        },
        "absolute_change": _round_number(absolute_change),
        "percent_change": _round_number(percent_change),
        "direction": direction,
        "data_max_invoice_date": current_row.get("data_max_invoice_date") or previous_row.get("data_max_invoice_date"),
    }


def _anomalies_from_evidence(evidence_items, period_comparison):
    anomalies = []
    trend_rows = _rows(_evidence_by_name(evidence_items, "monthly_trend"))

    if period_comparison.get("status") != "ok":
        return [
            {
                "severity": "warning",
                "message": "Could not evaluate anomalies because the comparison period was unavailable.",
            }
        ]

    current_label = period_comparison["current_period"]["label"]
    current_value = _safe_float(period_comparison["current_period"]["value"])
    baseline_values = [
        _safe_float(row.get("revenue"))
        for row in trend_rows
        if row.get("month") != current_label and row.get("revenue") is not None
    ]

    if len(baseline_values) < 3:
        anomalies.append(
            {
                "severity": "info",
                "message": "Recent history has fewer than three baseline months, so statistical anomaly confidence is limited.",
            }
        )
        return anomalies

    baseline_mean = sum(baseline_values) / len(baseline_values)
    variance = sum((value - baseline_mean) ** 2 for value in baseline_values) / len(baseline_values)
    baseline_stddev = math.sqrt(variance)
    z_score = None if baseline_stddev == 0 else (current_value - baseline_mean) / baseline_stddev

    if z_score is None:
        anomalies.append(
            {
                "severity": "info",
                "message": "Historical monthly revenue had no variance, so a z-score anomaly test was not meaningful.",
                "baseline_mean": _round_number(baseline_mean),
            }
        )
    elif z_score <= -1.5:
        anomalies.append(
            {
                "severity": "high" if z_score <= -2.0 else "medium",
                "message": "The current period revenue is materially below the recent monthly baseline.",
                "z_score": _round_number(z_score),
                "baseline_mean": _round_number(baseline_mean),
                "baseline_stddev": _round_number(baseline_stddev),
            }
        )
    else:
        anomalies.append(
            {
                "severity": "info",
                "message": "The current period does not cross the configured revenue anomaly threshold.",
                "z_score": _round_number(z_score),
                "baseline_mean": _round_number(baseline_mean),
                "baseline_stddev": _round_number(baseline_stddev),
            }
        )

    if period_comparison.get("direction") != "down":
        anomalies.append(
            {
                "severity": "info",
                "message": "The data did not confirm a revenue drop for the analyzed periods.",
            }
        )

    return anomalies


def _top_drivers_from_evidence(evidence_items):
    driver_rows = _rows(_evidence_by_name(evidence_items, "vendor_drivers"))
    negative_drivers = [
        row for row in driver_rows
        if _safe_float(row.get("revenue_change")) < 0
    ]
    negative_drivers.sort(key=lambda row: _safe_float(row.get("revenue_change")))
    return negative_drivers[:5]


def _status_rows_for_period(evidence_items, period_key):
    rows = _rows(_evidence_by_name(evidence_items, "status_mix"))
    return [
        row for row in rows
        if row.get("period_key") == period_key
    ]


def _question_with_evidence(*, question, why, evidence_name, supporting_facts, tables, columns):
    return {
        "question": question,
        "why": why,
        "source_evidence": evidence_name,
        "supporting_facts": supporting_facts,
        "tables": tables,
        "columns": columns,
    }


def _data_backed_next_questions_for_revenue_drop(evidence_items, period_comparison):
    if period_comparison.get("status") != "ok":
        return []

    suggestions = []
    current = period_comparison["current_period"]
    previous = period_comparison["previous_period"]
    current_label = current["label"]
    previous_label = previous["label"]
    drivers = _top_drivers_from_evidence(evidence_items)

    for row in drivers[:2]:
        vendor_name = row.get("vendor_name")
        if not vendor_name:
            continue

        suggestions.append(
            _question_with_evidence(
                question=(
                    f"Show paid invoices for {vendor_name} in {previous_label} and {current_label} "
                    "to verify the vendor-level revenue change."
                ),
                why=(
                    f"`vendor_drivers` shows {vendor_name} changed by "
                    f"{_format_money(row.get('revenue_change'))}."
                ),
                evidence_name="vendor_drivers",
                supporting_facts={
                    "vendor_id": row.get("vendor_id"),
                    "vendor_name": vendor_name,
                    "previous_revenue": row.get("previous_revenue"),
                    "current_revenue": row.get("current_revenue"),
                    "revenue_change": row.get("revenue_change"),
                    "previous_paid_invoice_count": row.get("previous_paid_invoice_count"),
                    "current_paid_invoice_count": row.get("current_paid_invoice_count"),
                },
                tables=["invoices", "vendors"],
                columns=[
                    "invoices.vendor_id",
                    "invoices.invoice_date",
                    "invoices.status",
                    "invoices.grand_total",
                    "vendors.name",
                ],
            )
        )

    current_status_rows = _status_rows_for_period(evidence_items, "current_period")
    excluded_status_rows = [
        row for row in current_status_rows
        if row.get("status") != "paid" and _safe_float(row.get("invoice_value")) > 0
    ]
    excluded_status_rows.sort(key=lambda row: _safe_float(row.get("invoice_value")), reverse=True)

    for row in excluded_status_rows[:2]:
        status = row.get("status")
        suggestions.append(
            _question_with_evidence(
                question=(
                    f"List {status} invoices in {current_label} to see which invoice value is "
                    "excluded from paid-invoice revenue."
                ),
                why=(
                    f"`status_mix` shows {row.get('invoice_count')} {status} invoice(s) worth "
                    f"{_format_money(row.get('invoice_value'))} in {current_label}."
                ),
                evidence_name="status_mix",
                supporting_facts={
                    "period_key": row.get("period_key"),
                    "period": current_label,
                    "status": status,
                    "invoice_count": row.get("invoice_count"),
                    "invoice_value": row.get("invoice_value"),
                },
                tables=["invoices"],
                columns=["invoices.invoice_date", "invoices.status", "invoices.grand_total"],
            )
        )

    if current.get("paid_invoice_count") is not None and previous.get("paid_invoice_count") is not None:
        suggestions.append(
            _question_with_evidence(
                question=(
                    f"List the paid invoices in {current_label} and {previous_label} to inspect "
                    "the exact transaction mix behind the revenue change."
                ),
                why=(
                    "`period_comparison` shows paid invoice count changed from "
                    f"{previous.get('paid_invoice_count')} to {current.get('paid_invoice_count')}."
                ),
                evidence_name="period_comparison",
                supporting_facts={
                    "previous_period": previous_label,
                    "current_period": current_label,
                    "previous_paid_invoice_count": previous.get("paid_invoice_count"),
                    "current_paid_invoice_count": current.get("paid_invoice_count"),
                    "previous_average_paid_invoice_value": previous.get("average_paid_invoice_value"),
                    "current_average_paid_invoice_value": current.get("average_paid_invoice_value"),
                },
                tables=["invoices"],
                columns=["invoices.invoice_date", "invoices.status", "invoices.grand_total"],
            )
        )

    return suggestions[:5]


def _revenue_drop_answer(period_comparison, evidence_items):
    if period_comparison.get("status") != "ok":
        return "I could not confirm whether revenue dropped because the period comparison did not return enough data."

    current = period_comparison["current_period"]
    previous = period_comparison["previous_period"]
    absolute_change = period_comparison["absolute_change"]
    percent_change = period_comparison["percent_change"]
    drivers = _top_drivers_from_evidence(evidence_items)

    if period_comparison["direction"] == "down":
        opening = (
            f"Revenue dropped from {_format_money(previous['value'])} in {previous['label']} "
            f"to {_format_money(current['value'])} in {current['label']} "
            f"({_format_money(absolute_change)}, {_format_percent(percent_change)})."
        )
    elif period_comparison["direction"] == "up":
        opening = (
            f"The available data does not show a drop: revenue increased from "
            f"{_format_money(previous['value'])} in {previous['label']} to "
            f"{_format_money(current['value'])} in {current['label']} "
            f"({_format_money(absolute_change)}, {_format_percent(percent_change)})."
        )
    else:
        opening = (
            f"Revenue was flat at {_format_money(current['value'])} in {current['label']} "
            f"versus {previous['label']}."
        )

    volume_detail = (
        f"Paid invoice count moved from {previous.get('paid_invoice_count')} to "
        f"{current.get('paid_invoice_count')}, and average paid invoice value moved from "
        f"{_format_money(previous.get('average_paid_invoice_value'))} to "
        f"{_format_money(current.get('average_paid_invoice_value'))}."
    )

    if drivers:
        driver_bits = [
            f"{row.get('vendor_name')} ({_format_money(row.get('revenue_change'))})"
            for row in drivers[:3]
        ]
        driver_detail = "Largest negative vendor-level contributors: " + "; ".join(driver_bits) + "."
    else:
        driver_detail = "No negative vendor-level contributors were found in the driver query."

    return f"{opening} {volume_detail} {driver_detail}"


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


def _build_revenue_drop_analysis(user_question, sql_output, evidence_items, semantic_layer):
    metric = _metric_definition(semantic_layer, "revenue")
    period_comparison = _period_comparison_from_evidence(evidence_items)
    anomalies = _anomalies_from_evidence(evidence_items, period_comparison)
    cited_tables, cited_columns = _table_column_citations(
        semantic_layer,
        ["invoices", "vendors"],
        [
            "invoices.invoice_date",
            "invoices.status",
            "invoices.grand_total",
            "invoices.vendor_id",
            "vendors.id",
            "vendors.name",
        ],
    )

    data_max_invoice_date = period_comparison.get("data_max_invoice_date")
    assumptions = [
        "Revenue means total value of paid invoices, using SUM(invoices.grand_total) where invoices.status = 'paid'.",
        "Date bucketing uses invoices.invoice_date because the revenue metric is defined on invoices.",
        (
            "For 'last month', the analysis uses the latest complete invoice month in the available data "
            "instead of the machine calendar month, so incomplete data is not treated as a business drop."
        ),
        "No breakdown dimension was provided, so the first-pass investigation checks vendor drivers, status mix, and recent monthly trend.",
    ]
    if data_max_invoice_date:
        assumptions.append(
            f"The latest invoice date found was {data_max_invoice_date}; the month containing that date is treated as incomplete."
        )

    limitations = [
        "This sample schema is accounts-payable oriented; its 'revenue' metric is paid vendor invoice value, not customer revenue.",
        "Causality is inferred from available invoice, vendor, status, and amount fields; external factors are not present in the database.",
    ]
    if period_comparison.get("status") == "ok":
        current = period_comparison["current_period"]
        previous = period_comparison["previous_period"]
        if current.get("paid_invoice_count", 0) < 3 or previous.get("paid_invoice_count", 0) < 3:
            limitations.append("The comparison period has very few paid invoices, so vendor-driver conclusions can be sensitive to one invoice.")

    confidence = _confidence_for_analysis(evidence_items, period_comparison, limitations)
    answer = _revenue_drop_answer(period_comparison, evidence_items)
    next_query_evidence = _data_backed_next_questions_for_revenue_drop(
        evidence_items,
        period_comparison,
    )

    return {
        "Analysis_Version": "ai_native_analysis_v1",
        "Mode": "metric_driver_investigation",
        "Question": user_question,
        "Resolved_Question": sql_output.get("Resolved_Question") or user_question,
        "Executive_Answer": answer,
        "Clarification": {
            "needed": False,
            "question": sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions"),
            "decision": sql_output.get("Clarification_Decision"),
            "handled_as_assumption": bool(sql_output.get("Requires_Clarification")),
        },
        "Assumptions": assumptions,
        "Definitions": [metric] if metric else [],
        "Period_Comparison": period_comparison,
        "Anomalies": anomalies,
        "Citations": {
            "metrics": [metric] if metric else [],
            "tables": cited_tables,
            "columns": cited_columns,
        },
        "Evidence": evidence_items,
        "Confidence": confidence,
        "Limitations": limitations,
        "Suggested_Next_Queries": [
            item["question"] for item in next_query_evidence
        ],
        "Suggested_Next_Query_Evidence": next_query_evidence,
        "SQL_Is_Supporting_Evidence": True,
    }


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

    return {
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


def _clarification_analysis(user_question, sql_output):
    question = sql_output.get("Clarification_Question") or sql_output.get("Followup_Questions")
    return {
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
):
    model_name = model_name or get_default_model_name()
    sql_runner = sql_runner or load_default_sql_runner()

    with traced_span(
        "ai-native-analysis-workflow",
        input={"user_question": user_question, "thread_id": thread_id},
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
                for query in build_dataset_analysis_plan()
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

        if (
            intent["mode"] != "metric_driver_investigation"
            and (sql_output.get("Requires_Clarification") or sql_output.get("Clarification_Limit_Reached"))
        ):
            analysis = _clarification_analysis(user_question, sql_output)
            sql_output["Analysis"] = analysis
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

        if intent["mode"] == "metric_driver_investigation" and intent["metric"] == "revenue":
            evidence_items = [
                _run_evidence_query(query, sql_runner)
                for query in build_revenue_drop_evidence_queries()
            ]
            analysis = _build_revenue_drop_analysis(
                user_question,
                sql_output,
                evidence_items,
                semantic_layer,
            )
            primary_result = (
                _evidence_by_name(evidence_items, "period_comparison") or {}
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

        safe_update_observation(
            span,
            output={
                "analysis_mode": analysis.get("Mode"),
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
