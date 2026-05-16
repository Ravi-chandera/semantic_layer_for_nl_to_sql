import json
import re
import shutil
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "assignment.db"
ACTIVE_DB_PATH = DATA_DIR / "onboarded_dataset.db"
SCHEMA_PATH = DATA_DIR / "schema.json"
SEMANTIC_LAYER_PATH = DATA_DIR / "semantic_layer.json"
DATASET_MANIFEST_PATH = DATA_DIR / "dataset_onboarding.json"

SENSITIVE_NAME_PARTS = (
    "email",
    "phone",
    "mobile",
    "address",
    "ssn",
    "tax",
    "gstin",
    "pan",
    "account",
    "iban",
    "routing",
    "password",
    "token",
)
METRIC_NAME_PARTS = (
    "amount",
    "total",
    "price",
    "cost",
    "value",
    "qty",
    "quantity",
    "count",
    "balance",
    "rate",
)
NUMERIC_TYPES = ("INT", "REAL", "NUM", "DEC", "DOUBLE", "FLOAT")

DATASET_UNDERSTANDING_PROMPT = """
You are helping an NL-to-SQL app understand a newly uploaded SQLite database.
Infer the dataset's domain, business meaning, useful metrics, safe synonyms, and likely analysis questions from schema metadata and non-sensitive samples.

Return strict JSON only:
{
  "dataset_summary": "<one paragraph>",
  "domain": "<short domain label or unknown>",
  "table_updates": [
    {"table_name": "<table>", "business_name": "<human name>", "synonyms": ["<term>"], "description": "<meaning>"}
  ],
  "column_updates": [
    {"table_name": "<table>", "column_name": "<column>", "business_name": "<human name>", "synonyms": ["<term>"], "is_metric": true, "is_sensitive": false}
  ],
  "metric_updates": [
    {"metric_name": "<snake_case>", "description": "<meaning>", "sql": "<SQLite aggregate expression>", "filters": null, "synonyms": ["<term>"], "result_unit": "<number|count|percentage|currency code|days>", "tables": ["<table>"], "enabled": true}
  ],
  "ambiguity_rules": {
    "<rule_name>": {
      "trigger_phrases": ["<phrase>"],
      "ambiguous_dimensions": [{"label": "<choice>", "sql_hint": "<column or expression>"}],
      "clarification_question": "<question>",
      "default_assumption": null,
      "applies_to_tables": ["<table>"]
    }
  },
  "suggested_questions": ["<question>"]
}

Rules:
- Use only tables and columns present in the provided profile.
- Do not assume this is sales, finance, education, healthcare, or any other domain unless the schema supports it.
- Do not mark identifiers such as id or *_id as metrics.
- Avoid exposing sensitive sample values.
- Metric SQL must use table.column references and valid SQLite aggregate expressions.
""".strip()


def quote_identifier(identifier):
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def humanize_name(name):
    words = re.sub(r"[_\W]+", " ", str(name or "")).strip()
    return words.title() if words else ""


def split_synonyms(value):
    if isinstance(value, list):
        items = value
    else:
        items = str(value or "").split(",")
    return [item.strip() for item in items if item and item.strip()]


def _compact_profile_for_llm(discovered):
    compact_tables = []
    for table in discovered.get("tables", []):
        compact_columns = []
        for column in table.get("columns", []):
            compact_column = {
                "name": column.get("name"),
                "type": column.get("type"),
                "nullable": column.get("nullable"),
                "is_part_of_pk": column.get("is_part_of_pk"),
                "is_metric_guess": column.get("is_metric"),
                "is_sensitive_guess": column.get("is_sensitive"),
            }
            if not column.get("is_sensitive"):
                compact_column["sample_values"] = column.get("sample_values", [])[:3]
            compact_columns.append(compact_column)

        compact_tables.append(
            {
                "table_name": table.get("table_name"),
                "row_count": table.get("row_count"),
                "primary_keys": table.get("primary_keys"),
                "foreign_keys": table.get("foreign_keys"),
                "columns": compact_columns,
            }
        )
    return {
        "source_path": discovered.get("source_path"),
        "tables": compact_tables,
        "join_candidates": discovered.get("join_candidates", []),
    }


def _load_llm_json(response_text):
    cleaned = str(response_text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return json.loads(cleaned)


def generate_dataset_understanding(discovered, model_name=None, llm_call=None):
    """Ask an LLM for domain understanding of an uploaded dataset.

    The caller can inject `llm_call` in tests. If omitted, the project Gemini
    wrapper is imported lazily so discovery still works without an API key.
    """
    if llm_call is None:
        try:
            from .model_config import get_default_model_name
            from .pipeline import gemini_call
        except ImportError:
            from model_config import get_default_model_name
            from pipeline import gemini_call

        llm_call = gemini_call
        model_name = model_name or get_default_model_name()

    prompt = (
        DATASET_UNDERSTANDING_PROMPT
        + "\n\nDataset profile:\n"
        + json.dumps(_compact_profile_for_llm(discovered), indent=2, default=str)
    )
    response_text = llm_call(model_name, prompt, trace_name="gemini-dataset-understanding")
    understanding = _load_llm_json(response_text)
    return understanding if isinstance(understanding, dict) else {}


def apply_dataset_understanding_to_review(review, understanding):
    updated = json.loads(json.dumps(review, default=str))
    understanding = understanding or {}

    table_updates = {
        item.get("table_name"): item
        for item in understanding.get("table_updates", [])
        if isinstance(item, dict) and item.get("table_name")
    }
    for table in updated.get("tables", []):
        table_update = table_updates.get(table.get("table_name"))
        if not table_update:
            continue
        table["business_name"] = table_update.get("business_name") or table.get("business_name")
        if table_update.get("synonyms") is not None:
            table["synonyms"] = split_synonyms(table_update.get("synonyms"))

    column_updates = {
        (item.get("table_name"), item.get("column_name")): item
        for item in understanding.get("column_updates", [])
        if isinstance(item, dict) and item.get("table_name") and item.get("column_name")
    }
    for column in updated.get("columns", []):
        column_update = column_updates.get((column.get("table_name"), column.get("column_name")))
        if not column_update:
            continue
        column["business_name"] = column_update.get("business_name") or column.get("business_name")
        if column_update.get("synonyms") is not None:
            column["synonyms"] = split_synonyms(column_update.get("synonyms"))
        if column_update.get("is_metric") is not None:
            column["is_metric"] = bool(column_update.get("is_metric"))
        if column_update.get("is_sensitive") is not None:
            column["is_sensitive"] = bool(column_update.get("is_sensitive"))

    metric_updates = [
        metric
        for metric in understanding.get("metric_updates", [])
        if isinstance(metric, dict) and metric.get("metric_name") and metric.get("sql")
    ]
    if metric_updates:
        updated["metrics"] = metric_updates

    updated["dataset_understanding"] = {
        "dataset_summary": understanding.get("dataset_summary"),
        "domain": understanding.get("domain"),
        "suggested_questions": understanding.get("suggested_questions") or [],
        "ambiguity_rules": understanding.get("ambiguity_rules") or {},
    }
    return updated


def is_numeric_type(sqlite_type):
    normalized = str(sqlite_type or "").upper()
    return any(part in normalized for part in NUMERIC_TYPES)


def is_sensitive_column(column_name):
    normalized = str(column_name or "").lower()
    return any(part in normalized for part in SENSITIVE_NAME_PARTS)


def is_metric_column(column_name, sqlite_type):
    normalized = str(column_name or "").lower()
    return is_numeric_type(sqlite_type) and any(part in normalized for part in METRIC_NAME_PARTS)


def _connect_read_only(db_path):
    return sqlite3.connect(Path(db_path).resolve())


def _table_names(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    return [row[0] for row in cursor.fetchall()]


def discover_sqlite_dataset(db_path, sample_size=3):
    conn = _connect_read_only(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        discovered = {"source_path": str(Path(db_path)), "tables": []}
        for table_name in _table_names(cursor):
            quoted_table = quote_identifier(table_name)
            cursor.execute(f"PRAGMA table_info({quoted_table});")
            columns = cursor.fetchall()
            primary_keys = [
                row["name"]
                for row in sorted(columns, key=lambda item: item["pk"])
                if row["pk"] > 0
            ]

            cursor.execute(f"SELECT COUNT(*) AS row_count FROM {quoted_table};")
            row_count = cursor.fetchone()["row_count"]

            table = {
                "table_name": table_name,
                "business_name": humanize_name(table_name),
                "row_count": row_count,
                "columns": [],
                "primary_keys": primary_keys,
                "foreign_keys": [],
            }

            for column in columns:
                column_name = column["name"]
                quoted_column = quote_identifier(column_name)
                cursor.execute(
                    f"SELECT DISTINCT {quoted_column} AS sample_value "
                    f"FROM {quoted_table} "
                    f"WHERE {quoted_column} IS NOT NULL "
                    f"LIMIT ?;",
                    (sample_size,),
                )
                samples = [row["sample_value"] for row in cursor.fetchall()]
                sqlite_type = column["type"] or ""
                table["columns"].append(
                    {
                        "name": column_name,
                        "business_name": humanize_name(column_name),
                        "type": sqlite_type,
                        "nullable": not bool(column["notnull"]),
                        "is_part_of_pk": column["pk"] > 0,
                        "sample_values": samples,
                        "is_metric": is_metric_column(column_name, sqlite_type),
                        "is_sensitive": is_sensitive_column(column_name),
                        "synonyms": [],
                    }
                )

            cursor.execute(f"PRAGMA foreign_key_list({quoted_table});")
            for fk in cursor.fetchall():
                table["foreign_keys"].append(
                    {
                        "column": fk["from"],
                        "references_table": fk["table"],
                        "references_column": fk["to"],
                    }
                )

            discovered["tables"].append(table)

        discovered["join_candidates"] = infer_join_candidates(discovered)
        return discovered
    finally:
        conn.close()


def infer_join_candidates(discovered):
    tables = {table["table_name"]: table for table in discovered.get("tables", [])}
    candidates = []
    seen = set()

    def add_candidate(left_table, left_column, right_table, right_column, source):
        if left_table not in tables or right_table not in tables:
            return
        key = (left_table, left_column, right_table, right_column)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": right_column,
                "approved": source == "foreign_key",
                "source": source,
            }
        )

    for table in tables.values():
        for fk in table.get("foreign_keys", []):
            add_candidate(
                table["table_name"],
                fk["column"],
                fk["references_table"],
                fk["references_column"],
                "foreign_key",
            )

    primary_key_by_table = {
        table_name: table["primary_keys"][0]
        for table_name, table in tables.items()
        if len(table.get("primary_keys") or []) == 1
    }
    table_aliases = {}
    for table_name in tables:
        table_aliases[table_name] = table_name
        if table_name.endswith("ies"):
            table_aliases[table_name[:-3] + "y"] = table_name
        if table_name.endswith("s"):
            table_aliases[table_name[:-1]] = table_name

    for table in tables.values():
        for column in table.get("columns", []):
            column_name = column["name"]
            if not column_name.endswith("_id"):
                continue
            alias = column_name[:-3]
            target_table = table_aliases.get(alias)
            target_pk = primary_key_by_table.get(target_table)
            if target_table and target_pk:
                add_candidate(table["table_name"], column_name, target_table, target_pk, "name_match")

    return candidates


def build_review_template(discovered):
    metrics = []
    for table in discovered.get("tables", []):
        for column in table.get("columns", []):
            if not column.get("is_metric"):
                continue
            metric_name = f"total_{table['table_name']}_{column['name']}"
            metrics.append(
                {
                    "metric_name": metric_name,
                    "description": f"Sum of {column['business_name']} from {table['business_name']}.",
                    "sql": f"SUM({table['table_name']}.{column['name']})",
                    "filters": None,
                    "synonyms": [],
                    "result_unit": "count" if "count" in column["name"].lower() else "number",
                    "tables": [table["table_name"]],
                    "enabled": True,
                }
            )

    return {
        "tables": [
            {
                "table_name": table["table_name"],
                "business_name": table.get("business_name") or humanize_name(table["table_name"]),
                "synonyms": [],
            }
            for table in discovered.get("tables", [])
        ],
        "columns": [
            {
                "table_name": table["table_name"],
                "column_name": column["name"],
                "business_name": column.get("business_name") or humanize_name(column["name"]),
                "synonyms": column.get("synonyms") or [],
                "is_metric": bool(column.get("is_metric")),
                "is_sensitive": bool(column.get("is_sensitive")),
            }
            for table in discovered.get("tables", [])
            for column in table.get("columns", [])
        ],
        "joins": discovered.get("join_candidates", []),
        "metrics": metrics,
    }


def _indexed_review(review, key_name):
    return {item[key_name]: item for item in review if item.get(key_name)}


def build_semantic_layer(discovered, review=None):
    review = review or build_review_template(discovered)
    dataset_understanding = review.get("dataset_understanding") or {}
    table_reviews = _indexed_review(review.get("tables", []), "table_name")
    column_reviews = {
        (item.get("table_name"), item.get("column_name")): item
        for item in review.get("columns", [])
    }
    approved_joins = [join for join in review.get("joins", []) if join.get("approved")]

    semantic_layer = {
        "tables": {},
        "join_paths": {},
        "metrics": {},
        "synonyms": {"entity_synonyms": {}},
        "onboarding": {
            "source_path": discovered.get("source_path"),
            "generated_by": "dataset_onboarding",
        },
    }
    if dataset_understanding:
        semantic_layer["dataset_context"] = {
            "domain": dataset_understanding.get("domain"),
            "summary": dataset_understanding.get("dataset_summary"),
            "suggested_questions": dataset_understanding.get("suggested_questions") or [],
        }
        if dataset_understanding.get("ambiguity_rules"):
            semantic_layer["ambiguity_rules"] = dataset_understanding.get("ambiguity_rules")

    for table in discovered.get("tables", []):
        table_name = table["table_name"]
        table_review = table_reviews.get(table_name, {})
        business_name = table_review.get("business_name") or table.get("business_name") or humanize_name(table_name)
        table_synonyms = split_synonyms(table_review.get("synonyms"))
        semantic_layer["synonyms"]["entity_synonyms"][table_name] = table_synonyms
        primary_key = ", ".join(table.get("primary_keys") or [])

        semantic_table = {
            "description": f"Discovered table for {business_name}.",
            "synonyms": table_synonyms,
            "business_context": f"Use this table when questions refer to {business_name}.",
            "primary_key": primary_key,
            "columns": {},
            "relationships": [],
        }

        for column in table.get("columns", []):
            column_review = column_reviews.get((table_name, column["name"]), {})
            column_business_name = column_review.get("business_name") or column.get("business_name") or humanize_name(column["name"])
            semantic_table["columns"][column["name"]] = {
                "type": column.get("type") or "",
                "description": f"{column_business_name} on {business_name}.",
                "synonyms": split_synonyms(column_review.get("synonyms") or column.get("synonyms")),
                "is_filterable": True,
                "is_metric": bool(column_review.get("is_metric", column.get("is_metric"))),
                "is_sensitive": bool(column_review.get("is_sensitive", column.get("is_sensitive"))),
                "sample_values": column.get("sample_values", []),
            }

        for join in approved_joins:
            if join.get("left_table") != table_name:
                continue
            semantic_table["relationships"].append(
                {
                    "target_table": join.get("right_table"),
                    "type": "many_to_one",
                    "join_condition": (
                        f"{join.get('left_table')}.{join.get('left_column')} = "
                        f"{join.get('right_table')}.{join.get('right_column')}"
                    ),
                }
            )

        semantic_layer["tables"][table_name] = semantic_table

    for join in approved_joins:
        path_name = f"{join.get('left_table')}_to_{join.get('right_table')}"
        semantic_layer["join_paths"][path_name] = {
            "description": f"Join {humanize_name(join.get('left_table'))} to {humanize_name(join.get('right_table'))}.",
            "use_when": f"Use when a question needs fields from both {join.get('left_table')} and {join.get('right_table')}.",
            "steps": [
                {
                    "from": join.get("left_table"),
                    "to": join.get("right_table"),
                    "on": (
                        f"{join.get('left_table')}.{join.get('left_column')} = "
                        f"{join.get('right_table')}.{join.get('right_column')}"
                    ),
                }
            ],
        }

    for metric in review.get("metrics", []):
        if not metric.get("enabled", True) or not metric.get("metric_name"):
            continue
        semantic_layer["metrics"][metric["metric_name"]] = {
            "description": metric.get("description") or f"Metric {metric['metric_name']}.",
            "sql": metric.get("sql") or "",
            "filters": metric.get("filters") or None,
            "synonyms": split_synonyms(metric.get("synonyms")),
            "result_unit": metric.get("result_unit") or "number",
            "tables": split_synonyms(metric.get("tables")) if isinstance(metric.get("tables"), str) else metric.get("tables", []),
        }

    return semantic_layer


def save_onboarded_dataset(source_db_path, semantic_layer, discovered, target_db_path=ACTIVE_DB_PATH):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_db_path = Path(source_db_path)
    target_db_path = Path(target_db_path)
    if source_db_path.resolve() != target_db_path.resolve():
        shutil.copyfile(source_db_path, target_db_path)

    schema_payload = {
        "tables": [
            {
                "table_name": table["table_name"],
                "columns": table.get("columns", []),
                "primary_keys": table.get("primary_keys", []),
                "foreign_keys": table.get("foreign_keys", []),
            }
            for table in discovered.get("tables", [])
        ]
    }
    SCHEMA_PATH.write_text(json.dumps(schema_payload, indent=2, default=str), encoding="utf-8")
    SEMANTIC_LAYER_PATH.write_text(json.dumps(semantic_layer, indent=2, default=str), encoding="utf-8")

    manifest = {
        "active_db_path": str(target_db_path),
        "source_db_path": str(source_db_path),
        "table_count": len(discovered.get("tables", [])),
        "semantic_layer_path": str(SEMANTIC_LAYER_PATH),
        "schema_path": str(SCHEMA_PATH),
    }
    DATASET_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def get_active_db_path():
    if DATASET_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
            active_path = Path(manifest.get("active_db_path", ""))
            if active_path.exists():
                return active_path
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_DB_PATH
