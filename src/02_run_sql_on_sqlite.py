import sqlite3
import logging
import re
from pathlib import Path

from logging_config import configure_logging

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "assignment.db"

configure_logging()
logger = logging.getLogger(__name__)

BLOCKED_SQL_OPERATIONS = ("INSERT", "UPDATE", "DELETE")

DISPLAY_LABEL_LOOKUPS = {
    "vendor_id": ("vendors", "id", "name", "vendor_name"),
    "company_id": ("companies", "id", "name", "company_name"),
    "department_id": ("departments", "id", "name", "department_name"),
    "product_id": ("products", "id", "name", "product_name"),
    "invoice_id": ("invoices", "id", "invoice_number", "invoice_number"),
    "po_id": ("purchase_orders", "id", "po_number", "po_number"),
    "grn_id": ("grns", "id", "grn_number", "grn_number"),
}


def remove_sql_literals_and_comments(query):
    without_comments = re.sub(r"--.*?$|/\*.*?\*/", " ", query, flags=re.MULTILINE | re.DOTALL)
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", without_comments)


def validate_sql_guardrails(query):
    searchable_query = remove_sql_literals_and_comments(query)
    blocked_pattern = rf"\b({'|'.join(BLOCKED_SQL_OPERATIONS)})\b"
    blocked_matches = sorted(set(re.findall(blocked_pattern, searchable_query, flags=re.IGNORECASE)))

    if blocked_matches:
        blocked_operations = [operation.upper() for operation in blocked_matches]
        logger.error("SQL guardrail blocked operation(s): %s", blocked_operations)
        return f"SQL Guardrail Error: write operations are not allowed: {blocked_operations}"

    logger.info("SQL guardrail passed")
    return None


def get_db_tables(cursor):
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return {row["name"] for row in cursor.fetchall()}


def find_query_tables(query):
    table_pattern = r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)"
    cte_pattern = r"(?:WITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\("

    query_tables = set(re.findall(table_pattern, query, flags=re.IGNORECASE))
    cte_tables = set(re.findall(cte_pattern, query, flags=re.IGNORECASE))

    return query_tables - cte_tables


def validate_query_tables(query, cursor):
    db_tables = get_db_tables(cursor)
    query_tables = find_query_tables(query)
    missing_tables = sorted(query_tables - db_tables)

    logger.info("Tables used by SQL: %s", sorted(query_tables))

    if missing_tables:
        logger.error("SQL uses tables not present in DB: %s", missing_tables)
        return f"SQL Error: tables not present in DB: {missing_tables}"

    return None


def validate_query_plan(query, cursor):
    cursor.execute(f"EXPLAIN QUERY PLAN {query}")


def enrich_results_with_display_labels(results, cursor):
    if not results:
        return results

    for id_alias, (table_name, id_column, label_column, output_alias) in DISPLAY_LABEL_LOOKUPS.items():
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
            f"SELECT {id_column}, {label_column} FROM {table_name} WHERE {id_column} IN ({placeholders})",
            ids,
        )
        labels_by_id = {
            row[id_column]: row[label_column]
            for row in cursor.fetchall()
        }

        for row in results:
            row[output_alias] = labels_by_id.get(row.get(id_alias))

    return results


def run_query(query, db_name=DB_PATH):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        guardrail_error = validate_sql_guardrails(query)

        if guardrail_error:
            return guardrail_error

        validation_error = validate_query_tables(query, cursor)

        if validation_error:
            return validation_error

        validate_query_plan(query, cursor)
        logger.info("Running SQL query")
        cursor.execute(query)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        results = enrich_results_with_display_labels(results, cursor)
        logger.info("Returned %s rows", len(results))
        return results

    except sqlite3.Error as e:
        logger.error("SQLite failed to run SQL: %s", e)
        return f"SQL Error: {e}"
    finally:
        conn.close()


if __name__ == "__main__":
    my_sql = """
    SELECT\n  strftime('%Y-%m', date('now', 'start of month', '-1 month')) AS period_label,\n  COUNT(i.id) AS total_invoices\nFROM invoices AS i\nWHERE i.invoice_date >= date('now', 'start of month', '-1 month')\n  AND i.invoice_date < date('now', 'start of month');
    """

    data = run_query(my_sql)
    print(data)
