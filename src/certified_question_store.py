import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CERTIFIED_QUESTION_DB_PATH = ROOT_DIR / "data" / "certified_questions.db"
_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS = set()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        raw_tags = tags.split(",")
    else:
        raw_tags = list(tags)

    normalized = []
    for tag in raw_tags:
        tag = str(tag or "").strip().lower()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def get_connection(db_path=None):
    path = Path(db_path or CERTIFIED_QUESTION_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_certified_question_store(db_path=None):
    path = str(Path(db_path or CERTIFIED_QUESTION_DB_PATH))
    if path in _INITIALIZED_PATHS and Path(path).exists():
        return

    with _INIT_LOCK:
        if path in _INITIALIZED_PATHS and Path(path).exists():
            return

        with closing(get_connection(path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS certified_questions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    category TEXT,
                    owner TEXT,
                    approved_sql TEXT,
                    notes TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    active INTEGER NOT NULL DEFAULT 1,
                    certified INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_chat_id TEXT,
                    source_message_id TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_certified_questions_active ON certified_questions(active, certified, category);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_certified_questions_updated ON certified_questions(updated_at);"
            )
            conn.commit()

        _INITIALIZED_PATHS.add(path)


def _decode_row(row):
    record = dict(row)
    try:
        record["tags"] = json.loads(record.pop("tags_json") or "[]")
    except json.JSONDecodeError:
        record["tags"] = []
    record["active"] = bool(record["active"])
    record["certified"] = bool(record["certified"])
    return record


def save_certified_question(
    *,
    title,
    question,
    category=None,
    owner=None,
    approved_sql=None,
    notes=None,
    tags=None,
    active=True,
    certified=True,
    source_chat_id=None,
    source_message_id=None,
    question_id=None,
    db_path=None,
):
    title = str(title or "").strip()
    question = str(question or "").strip()
    if not title:
        raise ValueError("title is required")
    if not question:
        raise ValueError("question is required")

    init_certified_question_store(db_path)
    timestamp = utc_now()
    normalized_tags = normalize_tags(tags)
    question_id = question_id or str(uuid.uuid4())

    with closing(get_connection(db_path)) as conn:
        existing = conn.execute(
            "SELECT created_at FROM certified_questions WHERE id = ?;",
            (question_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else timestamp
        conn.execute(
            """
            INSERT INTO certified_questions (
                id, title, question, category, owner, approved_sql, notes, tags_json,
                active, certified, created_at, updated_at, source_chat_id, source_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                question = excluded.question,
                category = excluded.category,
                owner = excluded.owner,
                approved_sql = excluded.approved_sql,
                notes = excluded.notes,
                tags_json = excluded.tags_json,
                active = excluded.active,
                certified = excluded.certified,
                updated_at = excluded.updated_at,
                source_chat_id = excluded.source_chat_id,
                source_message_id = excluded.source_message_id;
            """,
            (
                question_id,
                title,
                question,
                str(category or "").strip() or None,
                str(owner or "").strip() or None,
                str(approved_sql or "").strip() or None,
                str(notes or "").strip() or None,
                json.dumps(normalized_tags),
                int(bool(active)),
                int(bool(certified)),
                created_at,
                timestamp,
                source_chat_id,
                source_message_id,
            ),
        )
        conn.commit()

    return get_certified_question(question_id, db_path=db_path)


def get_certified_question(question_id, db_path=None):
    init_certified_question_store(db_path)
    with closing(get_connection(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM certified_questions WHERE id = ?;",
            (question_id,),
        ).fetchone()
    return _decode_row(row) if row else None


def list_certified_questions(
    *,
    active_only=False,
    certified_only=False,
    category=None,
    limit=100,
    db_path=None,
):
    init_certified_question_store(db_path)
    clauses = []
    params = []
    if active_only:
        clauses.append("active = 1")
    if certified_only:
        clauses.append("certified = 1")
    if category:
        clauses.append("category = ?")
        params.append(category)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM certified_questions
            {where_sql}
            ORDER BY category COLLATE NOCASE, title COLLATE NOCASE
            LIMIT ?;
            """,
            params,
        ).fetchall()
    return [_decode_row(row) for row in rows]


def list_template_categories(db_path=None):
    records = list_certified_questions(
        active_only=True,
        certified_only=True,
        limit=1000,
        db_path=db_path,
    )
    categories = []
    for record in records:
        category = record.get("category") or "General"
        if category not in categories:
            categories.append(category)
    return categories


def set_certified_question_active(question_id, active, db_path=None):
    init_certified_question_store(db_path)
    timestamp = utc_now()
    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            UPDATE certified_questions
            SET active = ?, updated_at = ?
            WHERE id = ?;
            """,
            (int(bool(active)), timestamp, question_id),
        )
        conn.commit()
    return get_certified_question(question_id, db_path=db_path)
