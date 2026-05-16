import re
import sqlite3
from contextlib import closing
from difflib import SequenceMatcher
from pathlib import Path

try:
    from .dataset_onboarding import get_active_db_path
    from .pipeline_semantic_context import (
        entity_alias_for_table,
        label_alias_for_entity,
        pick_display_column,
    )
except ImportError:
    from dataset_onboarding import get_active_db_path
    from pipeline_semantic_context import (
        entity_alias_for_table,
        label_alias_for_entity,
        pick_display_column,
    )


ENTITY_ROW_LIMIT = 500
MAX_ENTITY_MATCHES = 12
MIN_FUZZY_SCORE = 0.82
AMBIGUITY_SCORE_DELTA = 0.12

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "by",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "list",
    "me",
    "of",
    "on",
    "our",
    "show",
    "the",
    "to",
    "total",
    "what",
    "which",
    "with",
}


def quote_identifier(identifier):
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def normalize_entity_text(value):
    text = re.sub(r"[^0-9a-zA-Z]+", " ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def entity_search_phrases(question, max_words=4):
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(question or ""))
    phrases = []

    for start in range(len(tokens)):
        for width in range(min(max_words, len(tokens) - start), 0, -1):
            phrase_tokens = tokens[start : start + width]
            normalized = normalize_entity_text(" ".join(phrase_tokens))
            if not normalized:
                continue
            if width == 1 and normalized in STOPWORDS:
                continue
            if width == 1 and len(normalized) < 3:
                continue
            if normalized not in phrases:
                phrases.append(normalized)

    full_question = normalize_entity_text(question)
    if full_question and len(tokens) <= max_words and full_question not in phrases:
        phrases.insert(0, full_question)

    return phrases


def entity_candidate_columns(semantic_layer):
    candidates = []
    for table_name, table_info in (semantic_layer.get("tables") or {}).items():
        columns = table_info.get("columns") or {}
        primary_key = table_info.get("primary_key")
        display_column = pick_display_column(table_info)
        if not primary_key or "," in str(primary_key) or not display_column:
            continue
        if display_column not in columns:
            continue

        entity_name = entity_alias_for_table(table_name)
        candidates.append(
            {
                "table": table_name,
                "id_column": primary_key,
                "display_column": display_column,
                "entity": entity_name,
                "label_alias": label_alias_for_entity(entity_name, display_column),
                "description": table_info.get("description"),
            }
        )

    return candidates


def _table_exists(conn, table_name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_rows(conn, candidate, row_limit):
    table_name = candidate["table"]
    id_column = candidate["id_column"]
    display_column = candidate["display_column"]
    query = (
        f"SELECT {quote_identifier(id_column)} AS entity_id, "
        f"{quote_identifier(display_column)} AS entity_label "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(display_column)} IS NOT NULL "
        f"LIMIT ?;"
    )
    return conn.execute(query, (row_limit,)).fetchall()


def _score_phrase_against_value(phrase, value):
    normalized_value = normalize_entity_text(value)
    if not phrase or not normalized_value:
        return 0.0, "none"
    if phrase == normalized_value:
        return 1.0, "exact"
    if normalized_value.startswith(phrase):
        return 0.94, "prefix"
    if phrase in normalized_value or normalized_value in phrase:
        return 0.9, "contains"

    score = SequenceMatcher(None, phrase, normalized_value).ratio()
    if score >= MIN_FUZZY_SCORE:
        return score, "fuzzy"
    return 0.0, "none"


def search_entities(question, semantic_layer, db_path=None, *, row_limit=ENTITY_ROW_LIMIT):
    phrases = entity_search_phrases(question)
    if not phrases:
        return {
            "matches": [],
            "ambiguous": False,
            "clarifying_question": None,
            "options": [],
            "context": [],
        }

    resolved_db_path = Path(db_path or get_active_db_path())
    candidates = entity_candidate_columns(semantic_layer)
    matches_by_key = {}

    with closing(sqlite3.connect(resolved_db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for candidate in candidates:
            if not _table_exists(conn, candidate["table"]):
                continue

            try:
                rows = _column_rows(conn, candidate, row_limit)
            except sqlite3.Error:
                continue

            for row in rows:
                value = row["entity_label"]
                for phrase in phrases:
                    score, match_type = _score_phrase_against_value(phrase, value)
                    if score <= 0:
                        continue

                    key = (
                        candidate["table"],
                        candidate["display_column"],
                        str(row["entity_id"]),
                        normalize_entity_text(value),
                    )
                    existing = matches_by_key.get(key)
                    if existing:
                        existing_phrase_len = len(existing.get("matched_phrase") or "")
                        current_phrase_len = len(phrase)
                        if existing["score"] > score + 0.08:
                            continue
                        if existing_phrase_len > current_phrase_len and score <= existing["score"] + 0.08:
                            continue
                        if existing["score"] >= score and existing_phrase_len >= current_phrase_len:
                            continue

                    matches_by_key[key] = {
                        "id": f"{candidate['table']}:{candidate['display_column']}:{row['entity_id']}",
                        "table": candidate["table"],
                        "entity": candidate["entity"],
                        "id_column": candidate["id_column"],
                        "display_column": candidate["display_column"],
                        "entity_id": row["entity_id"],
                        "value": value,
                        "matched_phrase": phrase,
                        "score": round(score, 3),
                        "match_type": match_type,
                        "label": f"{value} {candidate['entity']}",
                        "detail": (
                            f"Filter {candidate['table']}.{candidate['display_column']} "
                            f"= {value!r}"
                        ),
                        "resolution_text": f"Use {candidate['entity']} {value}.",
                        "sql_hint": (
                            f"{candidate['table']}.{candidate['display_column']} = "
                            f"'{str(value).replace(chr(39), chr(39) * 2)}'"
                        ),
                    }

    matches = sorted(
        matches_by_key.values(),
        key=lambda item: (
            -item["score"],
            len(str(item["value"])),
            item["table"],
            str(item["value"]),
        ),
    )[:MAX_ENTITY_MATCHES]

    ambiguity = find_entity_ambiguity(matches)
    context = build_entity_match_context(matches, ambiguity)
    return {
        "matches": matches,
        "ambiguous": bool(ambiguity),
        "clarifying_question": ambiguity.get("clarifying_question") if ambiguity else None,
        "options": ambiguity.get("options") if ambiguity else [],
        "matched_phrase": ambiguity.get("matched_phrase") if ambiguity else None,
        "context": context,
    }


def find_entity_ambiguity(matches):
    by_phrase = {}
    for match in matches:
        by_phrase.setdefault(match["matched_phrase"], []).append(match)

    best = None
    for phrase, phrase_matches in by_phrase.items():
        ordered = sorted(phrase_matches, key=lambda item: -item["score"])
        if len(ordered) < 2:
            continue

        top_score = ordered[0]["score"]
        if top_score < MIN_FUZZY_SCORE:
            continue

        close_matches = [
            match
            for match in ordered
            if top_score - match["score"] <= AMBIGUITY_SCORE_DELTA
        ]
        distinct_entities = {
            (match["table"], normalize_entity_text(match["value"]))
            for match in close_matches
        }
        if len(distinct_entities) < 2:
            continue

        candidate = {
            "matched_phrase": phrase,
            "score": top_score,
            "matches": close_matches[:5],
        }
        if not best or candidate["score"] > best["score"] or len(phrase) > len(best["matched_phrase"]):
            best = candidate

    if not best:
        return None

    options = []
    for position, match in enumerate(best["matches"]):
        option = {
            "id": match["id"],
            "label": match["label"],
            "detail": match["detail"],
            "resolution_text": match["resolution_text"],
            "sql_hint": match["sql_hint"],
            "entity_match": match,
            "position": position,
        }
        options.append(option)

    phrase_label = best["matched_phrase"].title()
    return {
        "matched_phrase": best["matched_phrase"],
        "clarifying_question": f"Did you mean one of these matches for {phrase_label}?",
        "options": options,
    }


def build_entity_match_context(matches, ambiguity=None):
    if ambiguity:
        return []

    context = []
    seen_phrases = set()
    for match in matches:
        if match["matched_phrase"] in seen_phrases:
            continue
        if match["score"] < MIN_FUZZY_SCORE:
            continue
        seen_phrases.add(match["matched_phrase"])
        context.append(
            {
                "matched_phrase": match["matched_phrase"],
                "match_type": match["match_type"],
                "score": match["score"],
                "table": match["table"],
                "id_column": match["id_column"],
                "display_column": match["display_column"],
                "entity_id": match["entity_id"],
                "value": match["value"],
                "sql_hint": match["sql_hint"],
            }
        )
    return context[:5]
