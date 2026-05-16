import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import dataset_onboarding


class DatasetOnboardingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "sample.db"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE customers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    tax_id TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    customer_id INTEGER NOT NULL,
                    total_amount REAL NOT NULL,
                    status TEXT,
                    FOREIGN KEY(customer_id) REFERENCES customers(id)
                );
                """
            )
            conn.executemany(
                "INSERT INTO customers (id, name, email, tax_id) VALUES (?, ?, ?, ?);",
                [
                    (1, "Acme", "ap@acme.test", "TAX-1"),
                    (2, "Globex", "billing@globex.test", "TAX-2"),
                ],
            )
            conn.executemany(
                "INSERT INTO orders (id, customer_id, total_amount, status) VALUES (?, ?, ?, ?);",
                [
                    (10, 1, 125.50, "paid"),
                    (11, 2, 200.00, "open"),
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_sqlite_dataset_extracts_tables_columns_samples_and_flags(self):
        discovered = dataset_onboarding.discover_sqlite_dataset(self.db_path, sample_size=2)

        table_names = [table["table_name"] for table in discovered["tables"]]
        self.assertEqual(table_names, ["customers", "orders"])

        customers = discovered["tables"][0]
        self.assertEqual(customers["row_count"], 2)
        self.assertEqual(customers["primary_keys"], ["id"])

        columns = {column["name"]: column for column in customers["columns"]}
        self.assertTrue(columns["email"]["is_sensitive"])
        self.assertTrue(columns["tax_id"]["is_sensitive"])
        self.assertEqual(columns["name"]["sample_values"], ["Acme", "Globex"])

        orders = discovered["tables"][1]
        order_columns = {column["name"]: column for column in orders["columns"]}
        self.assertTrue(order_columns["total_amount"]["is_metric"])

    def test_join_candidates_include_approved_foreign_keys(self):
        discovered = dataset_onboarding.discover_sqlite_dataset(self.db_path)

        self.assertIn(
            {
                "left_table": "orders",
                "left_column": "customer_id",
                "right_table": "customers",
                "right_column": "id",
                "approved": True,
                "source": "foreign_key",
            },
            discovered["join_candidates"],
        )

    def test_build_semantic_layer_uses_reviewed_names_metrics_joins_and_sensitivity(self):
        discovered = dataset_onboarding.discover_sqlite_dataset(self.db_path)
        review = dataset_onboarding.build_review_template(discovered)
        review["tables"][0]["business_name"] = "Buyer"
        review["tables"][0]["synonyms"] = "client, account"
        review["columns"][2]["is_sensitive"] = True
        review["metrics"][0]["metric_name"] = "booked_order_amount"
        review["metrics"][0]["synonyms"] = "sales, bookings"

        semantic_layer = dataset_onboarding.build_semantic_layer(discovered, review)

        self.assertEqual(
            semantic_layer["tables"]["customers"]["synonyms"],
            ["client", "account"],
        )
        self.assertTrue(
            semantic_layer["tables"]["customers"]["columns"]["email"]["is_sensitive"]
        )
        self.assertEqual(
            semantic_layer["join_paths"]["orders_to_customers"]["steps"][0]["on"],
            "orders.customer_id = customers.id",
        )
        self.assertIn("booked_order_amount", semantic_layer["metrics"])
        self.assertEqual(
            semantic_layer["metrics"]["booked_order_amount"]["synonyms"],
            ["sales", "bookings"],
        )

    def test_dataset_understanding_can_enrich_review_and_semantic_context(self):
        discovered = dataset_onboarding.discover_sqlite_dataset(self.db_path)
        review = dataset_onboarding.build_review_template(discovered)
        understanding = {
            "dataset_summary": "Customer order tracking data.",
            "domain": "order management",
            "table_updates": [
                {
                    "table_name": "orders",
                    "business_name": "Purchases",
                    "synonyms": ["transactions"],
                }
            ],
            "column_updates": [
                {
                    "table_name": "orders",
                    "column_name": "total_amount",
                    "business_name": "Order Value",
                    "synonyms": ["booking value"],
                    "is_metric": True,
                }
            ],
            "metric_updates": [
                {
                    "metric_name": "total_order_value",
                    "description": "Total order value.",
                    "sql": "SUM(orders.total_amount)",
                    "filters": None,
                    "synonyms": ["bookings"],
                    "result_unit": "number",
                    "tables": ["orders"],
                    "enabled": True,
                }
            ],
            "suggested_questions": ["What is total order value by status?"],
            "ambiguity_rules": {},
        }

        enriched = dataset_onboarding.apply_dataset_understanding_to_review(
            review,
            understanding,
        )
        semantic_layer = dataset_onboarding.build_semantic_layer(discovered, enriched)

        self.assertEqual(enriched["tables"][1]["business_name"], "Purchases")
        self.assertIn("total_order_value", semantic_layer["metrics"])
        self.assertEqual(
            semantic_layer["dataset_context"]["domain"],
            "order management",
        )

    def test_save_onboarded_dataset_writes_manifest_schema_semantic_and_active_copy(self):
        data_dir = self.temp_path / "data"
        active_db = data_dir / "active.db"
        schema_path = data_dir / "schema.json"
        semantic_path = data_dir / "semantic_layer.json"
        manifest_path = data_dir / "dataset_onboarding.json"

        discovered = dataset_onboarding.discover_sqlite_dataset(self.db_path)
        semantic_layer = dataset_onboarding.build_semantic_layer(discovered)

        with patch.object(dataset_onboarding, "DATA_DIR", data_dir), \
             patch.object(dataset_onboarding, "ACTIVE_DB_PATH", active_db), \
             patch.object(dataset_onboarding, "SCHEMA_PATH", schema_path), \
             patch.object(dataset_onboarding, "SEMANTIC_LAYER_PATH", semantic_path), \
             patch.object(dataset_onboarding, "DATASET_MANIFEST_PATH", manifest_path):
            manifest = dataset_onboarding.save_onboarded_dataset(
                self.db_path,
                semantic_layer,
                discovered,
                target_db_path=active_db,
            )
            self.assertEqual(Path(manifest["active_db_path"]), active_db)
            self.assertTrue(active_db.exists())
            self.assertTrue(schema_path.exists())
            self.assertTrue(semantic_path.exists())
            self.assertEqual(dataset_onboarding.get_active_db_path(), active_db)

    def test_get_active_db_path_falls_back_when_manifest_is_missing(self):
        missing_manifest = self.temp_path / "missing.json"
        fallback_db = self.temp_path / "fallback.db"

        with patch.object(dataset_onboarding, "DATASET_MANIFEST_PATH", missing_manifest), \
             patch.object(dataset_onboarding, "DEFAULT_DB_PATH", fallback_db):
            self.assertEqual(dataset_onboarding.get_active_db_path(), fallback_db)


if __name__ == "__main__":
    unittest.main()
