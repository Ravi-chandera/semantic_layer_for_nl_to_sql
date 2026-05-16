import sqlite3
import logging
import time
from pathlib import Path
from urllib.parse import quote

import sqlglot
from sqlglot import errors as sqlglot_errors
from sqlglot import exp

try:
    from .dataset_onboarding import get_active_db_path
    from .logging_config import configure_logging
    from .langfuse_tracing import safe_update_observation, traced_span
except ImportError:
    from dataset_onboarding import get_active_db_path
    from logging_config import configure_logging
    from langfuse_tracing import safe_update_observation, traced_span

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "assignment.db"
SEMANTIC_LAYER_PATH = ROOT_DIR / "data" / "semantic_layer.json"

configure_logging()
logger = logging.getLogger(__name__)

MAX_RESULT_ROWS = 500
QUERY_TIMEOUT_SECONDS = 5.0
PROGRESS_HANDLER_OPCODES = 1_000

MAX_TRACE_RESULT_ROWS = 50

SQLITE_WRITE_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}

BLOCKED_FUNCTIONS = {"load_extension"}


def connect_read_only(db_name):
    if str(db_name) == ":memory:":
        return sqlite3.connect(db_name)

    db_path = Path(db_name).resolve()
    db_uri = f"file:{quote(db_path.as_posix(), safe='/:')}?mode=ro"
    return sqlite3.connect(db_uri, uri=True, timeout=1.0)


def build_sqlite_authorizer():
    def authorizer(action_code, arg1, arg2, database_name, trigger_name):
        if action_code in SQLITE_WRITE_ACTIONS:
            logger.error(
                "SQLite authorizer blocked action=%s arg1=%s arg2=%s db=%s trigger=%s",
                action_code,
                arg1,
                arg2,
                database_name,
                trigger_name,
            )
            return sqlite3.SQLITE_DENY

        if action_code == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in BLOCKED_FUNCTIONS:
            logger.error("SQLite authorizer blocked function: %s", arg2)
            return sqlite3.SQLITE_DENY

        return sqlite3.SQLITE_OK

    return authorizer


def install_query_timeout(conn, timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = QUERY_TIMEOUT_SECONDS

    deadline = time.monotonic() + timeout_seconds

    def progress_handler():
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(progress_handler, PROGRESS_HANDLER_OPCODES)


def summarize_result_for_trace(result):
    if isinstance(result, str):
        return {"status": "error", "message": result}

    if isinstance(result, list):
        return {
            "status": "ok",
            "row_count": len(result),
            "rows": result[:MAX_TRACE_RESULT_ROWS],
            "truncated": len(result) > MAX_TRACE_RESULT_ROWS,
        }

    return {"status": "unknown", "value": result}


def parse_sql_statement(query):
    try:
        statements = [statement for statement in sqlglot.parse(query, read="sqlite") if statement]
    except sqlglot_errors.ParseError as e:
        logger.error("SQL guardrail could not parse query: %s", e)
        return None, f"SQL Guardrail Error: could not parse SQL: {e}"

    if len(statements) != 1:
        logger.error("SQL guardrail blocked statement count: %s", len(statements))
        return None, "SQL Guardrail Error: exactly one read-only SQL statement is allowed."

    return statements[0], None


def is_read_only_statement(statement):
    return isinstance(statement, exp.Query)


def validate_sql_guardrails(query):
    statement, parse_error = parse_sql_statement(query)

    if parse_error:
        return None, parse_error

    if not is_read_only_statement(statement):
        logger.error("SQL guardrail blocked non-read query")
        return None, "SQL Guardrail Error: only SELECT/WITH read queries are allowed."

    logger.info("SQL guardrail passed")
    return statement, None


def get_db_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return {row["name"] for row in cursor.fetchall()}


def find_query_tables(statement):
    cte_tables = {
        cte.alias_or_name
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }
    query_tables = {
        table.name
        for table in statement.find_all(exp.Table)
        if table.name
    }

    return query_tables - cte_tables


def validate_query_tables(statement, cursor):
    db_tables = get_db_tables(cursor)
    query_tables = find_query_tables(statement)
    missing_tables = sorted(query_tables - db_tables)

    logger.info("Tables used by SQL: %s", sorted(query_tables))

    if missing_tables:
        logger.error("SQL uses tables not present in DB: %s", missing_tables)
        return f"SQL Error: tables not present in DB: {missing_tables}"

    return None


def validate_query_plan(query, cursor):
    cursor.execute(f"EXPLAIN QUERY PLAN {query}")


def quote_identifier(identifier):
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def _singularize_table_name(table_name):
    if table_name.endswith("ies"):
        return f"{table_name[:-3]}y"
    if table_name.endswith("s"):
        return table_name[:-1]
    return table_name


def _pick_display_column(table_info):
    columns = table_info.get("columns", {}) or {}
    preferred = (
        "name",
        "title",
        "label",
        "code",
        "number",
        "reference",
        "reference_number",
    )
    for column_name in preferred:
        if column_name in columns and not columns[column_name].get("is_sensitive"):
            return column_name
    for column_name, column_info in columns.items():
        column_type = str(column_info.get("type") or "").upper()
        if column_info.get("is_sensitive"):
            continue
        if any(part in column_type for part in ("CHAR", "TEXT", "CLOB", "VARCHAR")):
            return column_name
    return None


def _load_display_label_lookups(cursor):
    if not SEMANTIC_LAYER_PATH.exists():
        return {}
    try:
        import json

        semantic_layer = json.loads(SEMANTIC_LAYER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    db_tables = get_db_tables(cursor)
    lookups = {}
    for table_name, table_info in semantic_layer.get("tables", {}).items():
        if table_name not in db_tables:
            continue
        primary_key = table_info.get("primary_key")
        if not primary_key or "," in str(primary_key):
            continue
        display_column = _pick_display_column(table_info)
        if not display_column:
            continue
        entity_name = _singularize_table_name(table_name)
        id_alias = f"{entity_name}_id"
        label_alias = (
            display_column
            if display_column.endswith("_number")
            else f"{entity_name}_{display_column}"
        )
        lookups[id_alias] = (table_name, primary_key, display_column, label_alias)
    return lookups


def enrich_results_with_display_labels(results, cursor):
    if not results:
        return results

    display_label_lookups = _load_display_label_lookups(cursor)
    for id_alias, (table_name, id_column, label_column, output_alias) in display_label_lookups.items():
        if id_alias not in results[0] or output_alias in results[0]:
            continue

        ids = sorted({
            row[id_alias]
            for row in results
            if row.get(id_alias) is not None
        })

        if not ids:
            continue

        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            (
                f"SELECT {quote_identifier(id_column)}, {quote_identifier(label_column)} "
                f"FROM {quote_identifier(table_name)} "
                f"WHERE {quote_identifier(id_column)} IN ({placeholders})"
            ),
            ids,
        )
        labels_by_id = {
            row[id_column]: row[label_column]
            for row in cursor.fetchall()
        }

        for row in results:
            row[output_alias] = labels_by_id.get(row.get(id_alias))

    return results


def run_query(query, db_name=None):
    if db_name is None:
        db_name = get_active_db_path()

    with traced_span(
        "execute-sqlite-query",
        input={
            "sql": query,
            "db_path": str(db_name),
        },
        metadata={"component": "sqlite"},
    ) as span:
        conn = connect_read_only(db_name)
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA query_only = ON;")
            conn.set_authorizer(build_sqlite_authorizer())
            install_query_timeout(conn)
            cursor = conn.cursor()
            statement, guardrail_error = validate_sql_guardrails(query)

            if guardrail_error:
                safe_update_observation(
                    span,
                    output=summarize_result_for_trace(guardrail_error),
                    level="ERROR",
                    status_message=guardrail_error,
                )
                return guardrail_error

            validation_error = validate_query_tables(statement, cursor)

            if validation_error:
                safe_update_observation(
                    span,
                    output=summarize_result_for_trace(validation_error),
                    level="ERROR",
                    status_message=validation_error,
                )
                return validation_error

            validate_query_plan(query, cursor)
            logger.info("Running SQL query")
            cursor.execute(query)
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

            results = [dict(row) for row in rows[:MAX_RESULT_ROWS]]
            results = enrich_results_with_display_labels(results, cursor)
            logger.info("Returned %s rows", len(results))
            safe_update_observation(span, output=summarize_result_for_trace(results))
            return results

        except sqlite3.Error as e:
            error_message = f"SQL Error: {e}"
            logger.error("SQLite failed to run SQL: %s", e)
            safe_update_observation(
                span,
                output=summarize_result_for_trace(error_message),
                level="ERROR",
                status_message=error_message,
            )
            return error_message
        finally:
            conn.close()


if __name__ == "__main__":
    my_sql = """
    SELECT\n  strftime('%Y-%m', date('now', 'start of month', '-1 month')) AS period_label,\n  COUNT(i.id) AS total_invoices\nFROM invoices AS i\nWHERE i.invoice_date >= date('now', 'start of month', '-1 month')\n  AND i.invoice_date < date('now', 'start of month');
    """

    data = run_query(my_sql)
    print(data)
