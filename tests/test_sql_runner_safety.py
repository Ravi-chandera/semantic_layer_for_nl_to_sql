import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import sqlite_runner


class SqlRunnerSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql_runner = sqlite_runner

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "runner.db"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("CREATE TABLE invoices (id INTEGER PRIMARY KEY, amount INTEGER);")
            conn.executemany(
                "INSERT INTO invoices (amount) VALUES (?);",
                [(10,), (20,), (30,)],
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_select_returns_rows(self):
        result = self.sql_runner.run_query(
            "SELECT id, amount FROM invoices ORDER BY id;",
            self.db_path,
        )

        self.assertEqual(
            result,
            [
                {"id": 1, "amount": 10},
                {"id": 2, "amount": 20},
                {"id": 3, "amount": 30},
            ],
        )

    def test_multiple_statements_are_blocked_before_execution(self):
        result = self.sql_runner.run_query(
            "SELECT id FROM invoices; SELECT amount FROM invoices;",
            self.db_path,
        )

        self.assertIsInstance(result, str)
        self.assertIn("exactly one", result)

    def test_write_statement_is_blocked_and_database_is_unchanged(self):
        result = self.sql_runner.run_query("DELETE FROM invoices;", self.db_path)

        self.assertIsInstance(result, str)
        self.assertIn("only SELECT/WITH", result)

        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM invoices;").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(count, 3)

    def test_pragma_and_attach_are_blocked(self):
        pragma_result = self.sql_runner.run_query("PRAGMA table_info(invoices);", self.db_path)
        attach_result = self.sql_runner.run_query(
            "ATTACH DATABASE 'other.db' AS other;",
            self.db_path,
        )

        self.assertIsInstance(pragma_result, str)
        self.assertIn("only SELECT/WITH", pragma_result)
        self.assertIsInstance(attach_result, str)
        self.assertIn("only SELECT/WITH", attach_result)

    def test_missing_table_uses_parsed_table_detection(self):
        result = self.sql_runner.run_query(
            "WITH recent AS (SELECT * FROM invoices) SELECT * FROM missing_table;",
            self.db_path,
        )

        self.assertIsInstance(result, str)
        self.assertIn("tables not present", result)
        self.assertIn("missing_table", result)

    def test_result_rows_are_capped(self):
        result = self.sql_runner.run_query(
            """
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 600
            )
            SELECT n FROM nums ORDER BY n;
            """,
            self.db_path,
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), self.sql_runner.MAX_RESULT_ROWS)
        self.assertEqual(result[0], {"n": 1})
        self.assertEqual(result[-1], {"n": self.sql_runner.MAX_RESULT_ROWS})

    def test_long_running_query_times_out(self):
        original_timeout = self.sql_runner.QUERY_TIMEOUT_SECONDS
        self.sql_runner.QUERY_TIMEOUT_SECONDS = 0.001
        try:
            result = self.sql_runner.run_query(
                """
                WITH RECURSIVE nums(n) AS (
                    SELECT 1
                    UNION ALL
                    SELECT n + 1 FROM nums WHERE n < 1000000
                )
                SELECT SUM(a.n + b.n) AS total
                FROM nums AS a
                CROSS JOIN nums AS b;
                """,
                self.db_path,
            )
        finally:
            self.sql_runner.QUERY_TIMEOUT_SECONDS = original_timeout

        self.assertIsInstance(result, str)
        self.assertIn("interrupted", result.lower())


if __name__ == "__main__":
    unittest.main()
