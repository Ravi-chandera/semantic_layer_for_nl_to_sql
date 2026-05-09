import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_DB_PATH = ROOT_DIR / "data" / "nl_to_sql_cache.db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SEMANTIC_MATCH_THRESHOLD = 0.98

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "give",
    "had",
    "has",
    "have",
    "having",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "more",
    "most",
    "my",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "please",
    "same",
    "show",
    "so",
    "some",
    "such",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def cache_connection():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_cache_store():
    with cache_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id TEXT PRIMARY KEY,
                semantic_layer_hash TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                keyword_signature TEXT NOT NULL,
                question_text TEXT NOT NULL,
                representative_original_question TEXT,
                model_name TEXT NOT NULL,
                sql_response_json TEXT NOT NULL,
                selected_tables_json TEXT NOT NULL,
                selected_metrics_json TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_hit_at TEXT,
                hit_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cache_semantic_question
            ON cache_entries (semantic_layer_hash, normalized_question);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_keyword_signature
            ON cache_entries (semantic_layer_hash, keyword_signature);
            """
        )


def stable_json_dumps(value: Any):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def semantic_layer_hash(semantic_layer: dict[str, Any]):
    return hashlib.sha256(stable_json_dumps(semantic_layer).encode("utf-8")).hexdigest()


def normalize_question(question: str):
    text = (question or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(question: str):
    normalized = normalize_question(question)
    tokens = [
        token
        for token in normalized.split()
        if len(token) > 1 and token not in STOPWORDS
    ]
    return tokens


def keyword_signature(question: str):
    tokens = sorted(set(extract_keywords(question)))
    return " ".join(tokens)


def _keyword_signature_can_match(signature: str):
    return len(signature.split()) >= 2


def _disable_embedding_progress_bars():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        logger.debug("Unable to disable Hugging Face Hub progress bars.", exc_info=True)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except Exception:
        logger.debug("Unable to disable Transformers progress bars.", exc_info=True)


@lru_cache(maxsize=1)
def get_embedding_model():
    _disable_embedding_progress_bars()

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_question(question: str):
    model = get_embedding_model()
    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return [float(value) for value in embedding]


def cosine_similarity(left: list[float], right: list[float]):
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if not left_norm or not right_norm:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _load_cache_row(row: sqlite3.Row, strategy: str, score: float | None = None):
    return {
        "id": row["id"],
        "strategy": strategy,
        "score": score,
        "question_text": row["question_text"],
        "sql_response": json.loads(row["sql_response_json"]),
        "selected_tables": json.loads(row["selected_tables_json"]),
        "selected_metrics": json.loads(row["selected_metrics_json"]),
    }


def _record_cache_hit(cache_id: str):
    timestamp = utc_now()
    with cache_connection() as conn:
        conn.execute(
            """
            UPDATE cache_entries
            SET hit_count = hit_count + 1,
                last_hit_at = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (timestamp, timestamp, cache_id),
        )


def _backfill_cache_embedding(cache_id: str, question: str):
    try:
        embedding = embed_question(question)
        embedding_json = json.dumps(embedding)
    except Exception as e:
        logger.warning(
            "Semantic cache embedding backfill failed for cache entry %s: %s",
            cache_id,
            e,
            exc_info=True,
        )
        return False

    timestamp = utc_now()
    with cache_connection() as conn:
        conn.execute(
            """
            UPDATE cache_entries
            SET embedding_model = ?,
                embedding_json = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            (EMBEDDING_MODEL_NAME, embedding_json, timestamp, cache_id),
        )

    return True


def lookup_cache(question: str, layer_hash: str):
    init_cache_store()
    normalized_question = normalize_question(question)
    signature = keyword_signature(question)
    keyword_hit = None

    with cache_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM cache_entries
            WHERE semantic_layer_hash = ?
              AND normalized_question = ?
            ORDER BY updated_at DESC
            LIMIT 1;
            """,
            (layer_hash, normalized_question),
        ).fetchone()

        if row:
            keyword_hit = _load_cache_row(row, strategy="keyword_exact", score=1.0)
        elif _keyword_signature_can_match(signature):
            row = conn.execute(
                """
                SELECT *
                FROM cache_entries
                WHERE semantic_layer_hash = ?
                  AND keyword_signature = ?
                ORDER BY updated_at DESC
                LIMIT 1;
                """,
                (layer_hash, signature),
            ).fetchone()

            if row:
                keyword_hit = _load_cache_row(row, strategy="keyword_signature", score=1.0)

        if keyword_hit:
            candidate_rows = []
        else:
            candidate_rows = conn.execute(
                """
                SELECT *
                FROM cache_entries
                WHERE semantic_layer_hash = ?
                  AND embedding_model = ?
                  AND embedding_json IS NOT NULL;
                """,
                (layer_hash, EMBEDDING_MODEL_NAME),
            ).fetchall()

    if keyword_hit:
        _record_cache_hit(keyword_hit["id"])
        if row["embedding_json"] is None:
            _backfill_cache_embedding(row["id"], row["question_text"])
        return keyword_hit

    if not candidate_rows:
        return None

    try:
        query_embedding = embed_question(question)
    except Exception as e:
        logger.warning(
            "Semantic cache lookup skipped because embedding failed: %s",
            e,
            exc_info=True,
        )
        return None

    best_row = None
    best_score = 0.0

    for row in candidate_rows:
        try:
            candidate_embedding = json.loads(row["embedding_json"])
        except json.JSONDecodeError:
            continue

        score = cosine_similarity(query_embedding, candidate_embedding)
        if score > best_score:
            best_row = row
            best_score = score

    if best_row and best_score > SEMANTIC_MATCH_THRESHOLD:
        _record_cache_hit(best_row["id"])
        return _load_cache_row(best_row, strategy="semantic", score=best_score)

    return None


def store_cache_entry(
    question: str,
    original_question: str,
    layer_hash: str,
    model_name: str,
    sql_response: dict[str, Any],
    selected_tables: list[str],
    selected_metrics: list[str],
):
    if not sql_response.get("SQL"):
        return {"stored": False, "reason": "SQL response did not include SQL."}

    init_cache_store()

    normalized_question = normalize_question(question)
    if not normalized_question:
        return {"stored": False, "reason": "Question normalized to an empty string."}

    signature = keyword_signature(question)
    timestamp = utc_now()

    try:
        embedding = embed_question(question)
        embedding_json = json.dumps(embedding)
    except Exception as e:
        logger.warning(
            "Semantic cache embedding failed; storing keyword-only cache entry: %s",
            e,
            exc_info=True,
        )
        embedding_json = None

    cache_id = str(uuid.uuid4())
    with cache_connection() as conn:
        conn.execute(
            """
            INSERT INTO cache_entries (
                id,
                semantic_layer_hash,
                normalized_question,
                keyword_signature,
                question_text,
                representative_original_question,
                model_name,
                sql_response_json,
                selected_tables_json,
                selected_metrics_json,
                embedding_model,
                embedding_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(semantic_layer_hash, normalized_question) DO UPDATE SET
                keyword_signature = excluded.keyword_signature,
                question_text = excluded.question_text,
                representative_original_question = excluded.representative_original_question,
                model_name = excluded.model_name,
                sql_response_json = excluded.sql_response_json,
                selected_tables_json = excluded.selected_tables_json,
                selected_metrics_json = excluded.selected_metrics_json,
                embedding_model = excluded.embedding_model,
                embedding_json = excluded.embedding_json,
                updated_at = excluded.updated_at;
            """,
            (
                cache_id,
                layer_hash,
                normalized_question,
                signature,
                question,
                original_question,
                model_name,
                json.dumps(sql_response, default=str),
                json.dumps(selected_tables, default=str),
                json.dumps(selected_metrics, default=str),
                EMBEDDING_MODEL_NAME,
                embedding_json,
                timestamp,
                timestamp,
            ),
        )

    return {
        "stored": True,
        "semantic_enabled": embedding_json is not None,
        "keyword_signature": signature,
    }


def delete_cache_entry(question: str, layer_hash: str):
    normalized_question = normalize_question(question)
    if not normalized_question:
        return {"deleted": False, "reason": "Question normalized to an empty string."}

    init_cache_store()
    with cache_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM cache_entries
            WHERE semantic_layer_hash = ?
              AND normalized_question = ?;
            """,
            (layer_hash, normalized_question),
        )

    return {"deleted": cursor.rowcount > 0, "deleted_count": cursor.rowcount}
