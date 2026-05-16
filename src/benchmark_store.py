import html
import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .model_config import get_default_model_name
except ImportError:
    from model_config import get_default_model_name


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DB_PATH = ROOT_DIR / "data" / "benchmark_results.db"
BENCHMARK_DASHBOARD_PATH = ROOT_DIR / "data" / "benchmark_dashboard.html"

_BENCHMARK_INIT_LOCK = threading.Lock()
_BENCHMARK_INITIALIZED = False


BENCHMARK_QUESTIONS = [
    {
        "category": "Simple (Single Table)",
        "question": "How many invoices were raised last month?",
        "expected_capability": "Single-table temporal filter on invoices.",
    },
    {
        "category": "Simple (Single Table)",
        "question": "List all vendors on the watchlist.",
        "expected_capability": "Single-table filter on vendors.",
    },
    {
        "category": "Joins",
        "question": "Which vendors have overdue invoices greater than INR 1,00,000?",
        "expected_capability": "Join invoices to vendors and filter overdue high-value invoices.",
    },
    {
        "category": "Joins",
        "question": "Show me all invoices for the Engineering department.",
        "expected_capability": "Join invoices to purchase_orders to departments.",
    },
    {
        "category": "Aggregation + Conditions",
        "question": "What is the total outstanding amount across all vendors?",
        "expected_capability": "Aggregate outstanding invoice amount.",
    },
    {
        "category": "Aggregation + Conditions",
        "question": "Which product has the highest total invoiced value?",
        "expected_capability": "Join invoice_line_items to products, group, rank by value.",
    },
    {
        "category": "Window Functions",
        "question": "Rank vendors by total invoice value.",
        "expected_capability": "Use ranking over aggregated vendor invoice values.",
    },
    {
        "category": "Window Functions",
        "question": "For each vendor, show the running total of payments received.",
        "expected_capability": "Use SUM window ordered by payment date per vendor.",
    },
    {
        "category": "Window Functions",
        "question": "Show each invoice alongside the previous invoice amount for the same vendor.",
        "expected_capability": "Use LAG over invoices partitioned by vendor.",
    },
    {
        "category": "Synonyms & Business Metrics",
        "question": "What was our revenue last quarter?",
        "expected_capability": "Resolve revenue to paid invoice grand_total over prior quarter.",
    },
    {
        "category": "Synonyms & Business Metrics",
        "question": "Show me all unpaid bills.",
        "expected_capability": "Resolve bills to invoices and unpaid to status filter.",
    },
    {
        "category": "Ambiguous / Temporal",
        "question": "Who are our top 5 vendors?",
        "expected_capability": "Ask a clarification because top can mean value, count, rating, etc.",
    },
    {
        "category": "Ambiguous / Temporal",
        "question": "Compare this quarter's invoice volume with last quarter.",
        "expected_capability": "Compare invoice counts across current and previous quarter.",
    },
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_benchmark_run_id():
    return str(uuid.uuid4())


def get_connection():
    BENCHMARK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(BENCHMARK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_benchmark_store():
    global _BENCHMARK_INITIALIZED

    if _BENCHMARK_INITIALIZED and BENCHMARK_DB_PATH.exists():
        return

    with _BENCHMARK_INIT_LOCK:
        if _BENCHMARK_INITIALIZED and BENCHMARK_DB_PATH.exists():
            return

        with closing(get_connection()) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        category TEXT NOT NULL,
                        question TEXT NOT NULL,
                        expected_capability TEXT,
                        thread_id TEXT,
                        chat_id TEXT,
                        model_name TEXT,
                        started_at TEXT NOT NULL,
                        ended_at TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        requires_clarification INTEGER NOT NULL DEFAULT 0,
                        clarification_attempts INTEGER NOT NULL DEFAULT 0,
                        clarification_question TEXT,
                        clarification_limit_reached INTEGER NOT NULL DEFAULT 0,
                        sql_generated INTEGER NOT NULL DEFAULT 0,
                        sql_executed INTEGER NOT NULL DEFAULT 0,
                        sql_error TEXT,
                        result_status TEXT NOT NULL,
                        result_row_count INTEGER,
                        cache_hit INTEGER NOT NULL DEFAULT 0,
                        cache_strategy TEXT,
                        selected_tables_json TEXT NOT NULL DEFAULT '[]',
                        selected_metrics_json TEXT NOT NULL DEFAULT '[]',
                        generated_sql TEXT,
                        chart_generated INTEGER NOT NULL DEFAULT 0,
                        chart_error TEXT,
                        error_message TEXT,
                        raw_sql_output_json TEXT NOT NULL DEFAULT '{}'
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created ON benchmark_runs(id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_benchmark_runs_question ON benchmark_runs(question);"
                )

        _BENCHMARK_INITIALIZED = True


def classify_benchmark_question(question):
    normalized_question = question.strip().lower()
    for item in BENCHMARK_QUESTIONS:
        if item["question"].strip().lower() == normalized_question:
            return item["category"], item["expected_capability"]

    return "Ad hoc", "User-entered question outside the fixed benchmark suite."


def summarize_sql_execution(sql_result):
    if sql_result is None:
        return "skipped", None, None

    if isinstance(sql_result, str):
        return "error", None, sql_result

    if isinstance(sql_result, list):
        return "ok", len(sql_result), None

    return "unknown", None, str(sql_result)


def build_benchmark_record(
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
    model_name=None,
    chart_path=None,
    chart_error=None,
    error_message=None,
):
    inferred_category, inferred_capability = classify_benchmark_question(question)
    sql_output = sql_output or {}
    result_status, result_row_count, sql_error = summarize_sql_execution(sql_result)
    generated_sql = sql_output.get("SQL")

    return {
        "run_id": run_id,
        "source": source,
        "category": category or inferred_category,
        "question": question,
        "expected_capability": expected_capability or inferred_capability,
        "thread_id": thread_id,
        "chat_id": chat_id,
        "model_name": model_name or get_default_model_name(),
        "started_at": started_at,
        "ended_at": ended_at,
        "latency_ms": float(latency_ms),
        "requires_clarification": bool(sql_output.get("Requires_Clarification")),
        "clarification_attempts": int(sql_output.get("Clarification_Attempts") or 0),
        "clarification_question": sql_output.get("Clarification_Question"),
        "clarification_limit_reached": bool(sql_output.get("Clarification_Limit_Reached")),
        "sql_generated": bool(generated_sql),
        "sql_executed": result_status == "ok",
        "sql_error": sql_error,
        "result_status": "exception" if error_message else result_status,
        "result_row_count": result_row_count,
        "cache_hit": bool(sql_output.get("Cache_Hit")),
        "cache_strategy": sql_output.get("Cache_Strategy"),
        "selected_tables": sql_output.get("Selected_Tables") or [],
        "selected_metrics": sql_output.get("Selected_Metrics") or [],
        "generated_sql": generated_sql,
        "chart_generated": bool(chart_path),
        "chart_error": chart_error,
        "error_message": error_message,
        "raw_sql_output": sql_output,
    }


def append_benchmark_record(record):
    init_benchmark_store()

    with closing(get_connection()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO benchmark_runs (
                    run_id, source, category, question, expected_capability, thread_id, chat_id,
                    model_name, started_at, ended_at, latency_ms, requires_clarification,
                    clarification_attempts, clarification_question, clarification_limit_reached,
                    sql_generated, sql_executed, sql_error, result_status, result_row_count,
                    cache_hit, cache_strategy, selected_tables_json, selected_metrics_json,
                    generated_sql, chart_generated, chart_error, error_message, raw_sql_output_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record["run_id"],
                    record["source"],
                    record["category"],
                    record["question"],
                    record.get("expected_capability"),
                    record.get("thread_id"),
                    record.get("chat_id"),
                    record.get("model_name"),
                    record["started_at"],
                    record["ended_at"],
                    record["latency_ms"],
                    int(record.get("requires_clarification", False)),
                    int(record.get("clarification_attempts") or 0),
                    record.get("clarification_question"),
                    int(record.get("clarification_limit_reached", False)),
                    int(record.get("sql_generated", False)),
                    int(record.get("sql_executed", False)),
                    record.get("sql_error"),
                    record.get("result_status") or "unknown",
                    record.get("result_row_count"),
                    int(record.get("cache_hit", False)),
                    record.get("cache_strategy"),
                    json.dumps(record.get("selected_tables") or [], default=str),
                    json.dumps(record.get("selected_metrics") or [], default=str),
                    record.get("generated_sql"),
                    int(record.get("chart_generated", False)),
                    record.get("chart_error"),
                    record.get("error_message"),
                    json.dumps(record.get("raw_sql_output") or {}, default=str),
                ),
            )


def list_benchmark_records(limit=500):
    init_benchmark_store()

    with closing(get_connection()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM benchmark_runs
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    records = []
    for row in rows:
        record = dict(row)
        record["selected_tables"] = json.loads(record.pop("selected_tables_json") or "[]")
        record["selected_metrics"] = json.loads(record.pop("selected_metrics_json") or "[]")
        record["raw_sql_output"] = json.loads(record.pop("raw_sql_output_json") or "{}")
        records.append(record)

    return records


def get_benchmark_summary():
    records = list(reversed(list_benchmark_records(limit=5000)))
    total = len(records)

    if not total:
        return {
            "total_runs": 0,
            "avg_latency_ms": 0,
            "avg_clarifying_questions": 0,
            "sql_generation_rate": 0,
            "sql_execution_success_rate": 0,
            "cache_hit_rate": 0,
            "chart_generation_rate": 0,
            "records": [],
            "by_category": [],
            "by_question": [],
        }

    def average(values):
        return sum(values) / len(values) if values else 0

    category_names = sorted({record["category"] for record in records})
    by_category = []
    for category in category_names:
        category_records = [record for record in records if record["category"] == category]
        by_category.append(
            {
                "category": category,
                "runs": len(category_records),
                "avg_latency_ms": average([record["latency_ms"] for record in category_records]),
                "avg_clarifying_questions": average(
                    [record["clarification_attempts"] for record in category_records]
                ),
                "sql_execution_success_rate": average(
                    [1 if record["result_status"] == "ok" else 0 for record in category_records]
                ),
            }
        )

    question_names = []
    for record in records:
        if record["question"] not in question_names:
            question_names.append(record["question"])

    by_question = []
    for question in question_names:
        question_records = [record for record in records if record["question"] == question]
        latest = question_records[-1]
        by_question.append(
            {
                "question": question,
                "category": latest["category"],
                "runs": len(question_records),
                "latest_status": latest["result_status"],
                "latest_latency_ms": latest["latency_ms"],
                "avg_latency_ms": average([record["latency_ms"] for record in question_records]),
                "avg_clarifying_questions": average(
                    [record["clarification_attempts"] for record in question_records]
                ),
                "sql_generation_rate": average(
                    [1 if record["sql_generated"] else 0 for record in question_records]
                ),
                "sql_execution_success_rate": average(
                    [1 if record["result_status"] == "ok" else 0 for record in question_records]
                ),
            }
        )

    return {
        "total_runs": total,
        "avg_latency_ms": average([record["latency_ms"] for record in records]),
        "avg_clarifying_questions": average(
            [record["clarification_attempts"] for record in records]
        ),
        "sql_generation_rate": average([1 if record["sql_generated"] else 0 for record in records]),
        "sql_execution_success_rate": average(
            [1 if record["result_status"] == "ok" else 0 for record in records]
        ),
        "cache_hit_rate": average([1 if record["cache_hit"] else 0 for record in records]),
        "chart_generation_rate": average(
            [1 if record["chart_generated"] else 0 for record in records]
        ),
        "records": records,
        "by_category": by_category,
        "by_question": by_question,
    }


def format_ms(value):
    return f"{value / 1000:.2f}s"


def format_pct(value):
    return f"{value * 100:.1f}%"


def _metric_card(label, value):
    return f"""
    <section class="metric-card">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(str(value))}</div>
    </section>
    """


def _status_class(status):
    if status == "ok":
        return "status-ok"
    if status in {"error", "exception"}:
        return "status-error"
    return "status-muted"


def render_benchmark_dashboard_html(summary=None):
    summary = summary or get_benchmark_summary()
    records = summary["records"]
    latest_records = list(reversed(records[-25:]))

    category_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row["category"])}</td>
          <td>{row["runs"]}</td>
          <td>{format_ms(row["avg_latency_ms"])}</td>
          <td>{row["avg_clarifying_questions"]:.2f}</td>
          <td>{format_pct(row["sql_execution_success_rate"])}</td>
        </tr>
        """
        for row in summary["by_category"]
    )

    question_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(row["category"])}</td>
          <td>{html.escape(row["question"])}</td>
          <td>{row["runs"]}</td>
          <td><span class="status {_status_class(row["latest_status"])}">{html.escape(row["latest_status"])}</span></td>
          <td>{format_ms(row["avg_latency_ms"])}</td>
          <td>{row["avg_clarifying_questions"]:.2f}</td>
          <td>{format_pct(row["sql_generation_rate"])}</td>
          <td>{format_pct(row["sql_execution_success_rate"])}</td>
        </tr>
        """
        for row in summary["by_question"]
    )

    latest_rows = "\n".join(
        f"""
        <tr>
          <td>{record["id"]}</td>
          <td>{html.escape(record["source"])}</td>
          <td>{html.escape(record["category"])}</td>
          <td>{html.escape(record["question"])}</td>
          <td><span class="status {_status_class(record["result_status"])}">{html.escape(record["result_status"])}</span></td>
          <td>{format_ms(record["latency_ms"])}</td>
          <td>{record["clarification_attempts"]}</td>
          <td>{"yes" if record["sql_generated"] else "no"}</td>
          <td>{"yes" if record["cache_hit"] else "no"}</td>
          <td>{html.escape(record["started_at"])}</td>
        </tr>
        """
        for record in latest_records
    )

    empty_state = ""
    if not records:
        empty_state = "<p class=\"empty\">No benchmark records yet. Ask a question or run the benchmark suite.</p>"

    generated_at = utc_now()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NL-to-SQL Benchmark Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #607083;
      --line: #dce3ea;
      --accent: #146c94;
      --ok: #136f3a;
      --error: #a83232;
      --warn: #9a6200;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.4;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 24px 42px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    .subtle {{
      color: var(--muted);
      margin: 7px 0 0;
      font-size: 14px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 86px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      min-height: 34px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    section.table-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 18px;
      overflow: hidden;
    }}
    h2 {{
      margin: 0;
      padding: 16px 18px 8px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #fbfcfe;
      white-space: nowrap;
    }}
    td:nth-child(4) {{
      min-width: 230px;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 650;
      background: #edf1f5;
      color: var(--muted);
    }}
    .status-ok {{
      background: #e8f5ee;
      color: var(--ok);
    }}
    .status-error {{
      background: #faeaea;
      color: var(--error);
    }}
    .status-muted {{
      background: #f2f4f7;
      color: var(--muted);
    }}
    .empty {{
      margin: 18px 0;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    @media (max-width: 1100px) {{
      .metrics {{ grid-template-columns: repeat(3, minmax(150px, 1fr)); }}
      header {{ display: block; }}
      table {{ min-width: 920px; }}
      section.table-section {{ overflow-x: auto; }}
    }}
    @media (max-width: 700px) {{
      main {{ padding: 20px 14px 32px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>NL-to-SQL Benchmark Dashboard</h1>
        <p class="subtle">Append-only metrics from chat questions and fixed benchmark suite runs.</p>
      </div>
      <p class="subtle">Generated {html.escape(generated_at)}</p>
    </header>

    {empty_state}

    <div class="metrics">
      {_metric_card("Total runs", summary["total_runs"])}
      {_metric_card("Avg latency", format_ms(summary["avg_latency_ms"]))}
      {_metric_card("Avg clarifying questions", f'{summary["avg_clarifying_questions"]:.2f}')}
      {_metric_card("SQL generated", format_pct(summary["sql_generation_rate"]))}
      {_metric_card("SQL execution success", format_pct(summary["sql_execution_success_rate"]))}
      {_metric_card("Cache hit rate", format_pct(summary["cache_hit_rate"]))}
    </div>

    <section class="table-section">
      <h2>Category Performance</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Runs</th>
            <th>Avg latency</th>
            <th>Avg clarifications</th>
            <th>Execution success</th>
          </tr>
        </thead>
        <tbody>{category_rows}</tbody>
      </table>
    </section>

    <section class="table-section">
      <h2>Question Performance</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Question</th>
            <th>Runs</th>
            <th>Latest status</th>
            <th>Avg latency</th>
            <th>Avg clarifications</th>
            <th>SQL generated</th>
            <th>Execution success</th>
          </tr>
        </thead>
        <tbody>{question_rows}</tbody>
      </table>
    </section>

    <section class="table-section">
      <h2>Latest Append-Only Runs</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Source</th>
            <th>Category</th>
            <th>Question</th>
            <th>Status</th>
            <th>Latency</th>
            <th>Clarifications</th>
            <th>SQL</th>
            <th>Cache</th>
            <th>Started</th>
          </tr>
        </thead>
        <tbody>{latest_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""


def write_benchmark_dashboard():
    html_text = render_benchmark_dashboard_html()
    BENCHMARK_DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DASHBOARD_PATH.write_text(html_text, encoding="utf-8")
    return BENCHMARK_DASHBOARD_PATH, html_text
