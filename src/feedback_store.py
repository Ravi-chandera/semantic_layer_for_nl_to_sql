import json
import sqlite3
import threading
import uuid
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_DB_PATH = ROOT_DIR / "data" / "result_feedback.db"
VALID_SENTIMENTS = {"up", "down"}
VALID_CATEGORIES = {"wrong_join", "wrong_metric", "wrong_date", "missing_filter"}
_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS = set()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value)


def _normalize_categories(categories):
    normalized = []
    for category in _json_list(categories):
        category = str(category or "").strip().lower()
        if category in VALID_CATEGORIES and category not in normalized:
            normalized.append(category)
    return normalized


def get_connection(db_path=None):
    path = Path(db_path or FEEDBACK_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_store(db_path=None):
    path = str(Path(db_path or FEEDBACK_DB_PATH))
    if path in _INITIALIZED_PATHS and Path(path).exists():
        return

    with _INIT_LOCK:
        if path in _INITIALIZED_PATHS and Path(path).exists():
            return

        with closing(get_connection(path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_feedback (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    sentiment TEXT NOT NULL CHECK (sentiment IN ('up', 'down')),
                    categories_json TEXT NOT NULL DEFAULT '[]',
                    note TEXT,
                    question TEXT,
                    resolved_question TEXT,
                    generated_sql TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '[]',
                    tables_json TEXT NOT NULL DEFAULT '[]',
                    chat_id TEXT,
                    thread_id TEXT,
                    turn_index INTEGER,
                    message_id TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_result_feedback_created_at ON result_feedback(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_result_feedback_sentiment ON result_feedback(sentiment);"
            )
            conn.commit()

        _INITIALIZED_PATHS.add(path)


def add_feedback(
    *,
    sentiment,
    categories=None,
    note=None,
    question=None,
    resolved_question=None,
    generated_sql=None,
    metrics=None,
    tables=None,
    chat_id=None,
    thread_id=None,
    turn_index=None,
    message_id=None,
    db_path=None,
):
    sentiment = str(sentiment or "").strip().lower()
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"sentiment must be one of {sorted(VALID_SENTIMENTS)}")

    normalized_categories = _normalize_categories(categories)
    feedback_id = str(uuid.uuid4())
    init_feedback_store(db_path)

    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO result_feedback (
                id, created_at, sentiment, categories_json, note, question,
                resolved_question, generated_sql, metrics_json, tables_json,
                chat_id, thread_id, turn_index, message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                feedback_id,
                utc_now(),
                sentiment,
                json.dumps(normalized_categories),
                note,
                question,
                resolved_question,
                generated_sql,
                json.dumps(_json_list(metrics)),
                json.dumps(_json_list(tables)),
                chat_id,
                thread_id,
                turn_index,
                message_id,
            ),
        )
        conn.commit()

    return feedback_id


def _decode_row(row):
    record = dict(row)
    for key in ("categories_json", "metrics_json", "tables_json"):
        output_key = key.removesuffix("_json")
        try:
            record[output_key] = json.loads(record.pop(key) or "[]")
        except json.JSONDecodeError:
            record[output_key] = []
    return record


def list_feedback(limit=100, db_path=None):
    init_feedback_store(db_path)
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM result_feedback
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
    return [_decode_row(row) for row in rows]


def summarize_corrections(limit=200, db_path=None):
    records = list_feedback(limit=limit, db_path=db_path)
    negative_records = [record for record in records if record["sentiment"] == "down"]

    category_counts = Counter()
    metric_counts = Counter()
    table_counts = Counter()
    question_counts = Counter()
    grouped_examples = defaultdict(list)

    for record in negative_records:
        categories = record.get("categories") or ["uncategorized"]
        for category in categories:
            category_counts[category] += 1
            if len(grouped_examples[category]) < 5:
                grouped_examples[category].append(
                    {
                        "question": record.get("resolved_question") or record.get("question"),
                        "note": record.get("note"),
                        "metrics": record.get("metrics") or [],
                        "tables": record.get("tables") or [],
                    }
                )

        for metric in record.get("metrics") or []:
            metric_counts[metric] += 1
        for table in record.get("tables") or []:
            table_counts[table] += 1
        question = record.get("resolved_question") or record.get("question")
        if question:
            question_counts[question] += 1

    return {
        "total_feedback": len(records),
        "negative_feedback": len(negative_records),
        "by_category": dict(category_counts.most_common()),
        "by_metric": dict(metric_counts.most_common()),
        "by_table": dict(table_counts.most_common()),
        "by_question": dict(question_counts.most_common(20)),
        "examples_by_category": dict(grouped_examples),
    }


def build_semantic_correction_context(selected_tables=None, selected_metrics=None, limit=200, db_path=None):
    if db_path is None and not FEEDBACK_DB_PATH.exists():
        return {}

    summary = summarize_corrections(limit=limit, db_path=db_path)
    selected_tables = set(selected_tables or [])
    selected_metrics = set(selected_metrics or [])

    relevant_examples = {}
    for category, examples in summary["examples_by_category"].items():
        filtered = []
        for example in examples:
            example_tables = set(example.get("tables") or [])
            example_metrics = set(example.get("metrics") or [])
            if (
                not selected_tables
                and not selected_metrics
                or example_tables.intersection(selected_tables)
                or example_metrics.intersection(selected_metrics)
            ):
                filtered.append(example)
        if filtered:
            relevant_examples[category] = filtered

    if not summary["negative_feedback"]:
        return {}

    return {
        "analyst_correction_summary": {
            "negative_feedback": summary["negative_feedback"],
            "by_category": summary["by_category"],
            "by_metric": {
                metric: count
                for metric, count in summary["by_metric"].items()
                if not selected_metrics or metric in selected_metrics
            },
            "by_table": {
                table: count
                for table, count in summary["by_table"].items()
                if not selected_tables or table in selected_tables
            },
            "examples_by_category": relevant_examples,
        }
    }
