import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CHAT_DB_PATH = ROOT_DIR / "data" / "chat_history.db"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_store():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                langfuse_trace_id TEXT,
                memory_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content_json TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id);"
        )


def create_chat(thread_id, name, langfuse_trace_id=None):
    init_chat_store()
    chat_id = str(uuid.uuid4())
    timestamp = utc_now()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chats (id, thread_id, name, langfuse_trace_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (chat_id, thread_id, name, langfuse_trace_id, timestamp, timestamp),
        )

    return get_chat(chat_id)


def get_chat(chat_id):
    init_chat_store()

    with get_connection() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id = ?;", (chat_id,)).fetchone()

    return dict(row) if row else None


def get_chat_by_thread_id(thread_id):
    init_chat_store()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM chats WHERE thread_id = ?;",
            (thread_id,),
        ).fetchone()

    return dict(row) if row else None


def get_or_create_chat(thread_id, name, langfuse_trace_id=None):
    existing_chat = get_chat_by_thread_id(thread_id)
    if existing_chat:
        return existing_chat
    return create_chat(thread_id, name, langfuse_trace_id)


def list_chats(limit=50):
    init_chat_store()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, thread_id, name, langfuse_trace_id, created_at, updated_at
            FROM chats
            ORDER BY updated_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def append_message(chat_id, role, content: Any):
    init_chat_store()
    timestamp = utc_now()

    with get_connection() as conn:
        user_message_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND role = 'user';",
            (chat_id,),
        ).fetchone()[0]
        next_turn_index = user_message_count if role == "user" else max(user_message_count - 1, 0)

        conn.execute(
            """
            INSERT INTO messages (chat_id, role, content_json, turn_index, created_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (chat_id, role, json.dumps(content, default=str), next_turn_index, timestamp),
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?;",
            (timestamp, chat_id),
        )


def load_chat_messages(chat_id):
    init_chat_store()

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content_json, turn_index, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC;
            """,
            (chat_id,),
        ).fetchall()

    messages = []
    for row in rows:
        message = dict(row)
        message["content"] = json.loads(message.pop("content_json"))
        messages.append(message)

    return messages


def update_chat_memory(chat_id, conversation_turns):
    init_chat_store()
    timestamp = utc_now()

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE chats
            SET memory_json = ?, updated_at = ?
            WHERE id = ?;
            """,
            (json.dumps(conversation_turns, default=str), timestamp, chat_id),
        )


def load_chat_memory(chat_id):
    chat = get_chat(chat_id)
    if not chat:
        return []
    return json.loads(chat.get("memory_json") or "[]")
