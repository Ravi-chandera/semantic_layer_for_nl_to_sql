import sqlite3
import logging
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "assignment.db"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def run_query(query, db_name=DB_PATH):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        validation_error = validate_query_tables(query, cursor)

        if validation_error:
            return validation_error

        validate_query_plan(query, cursor)
        logger.info("Running SQL query")
        cursor.execute(query)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]
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
