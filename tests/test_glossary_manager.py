import tempfile
import unittest
from pathlib import Path

from src import glossary_manager
from src.pipeline_semantic_context import build_sql_context, find_required_clarification_rule


class GlossaryManagerTests(unittest.TestCase):
    def setUp(self):
        self.semantic_layer = {
            "tables": {
                "invoices": {
                    "description": "Invoices.",
                    "synonyms": [],
                    "business_context": "Invoice headers.",
                    "primary_key": "id",
                    "columns": {
                        "id": {"type": "INTEGER", "description": "ID."},
                        "grand_total": {"type": "REAL", "description": "Total."},
                        "status": {"type": "TEXT", "description": "Status."},
                    },
                    "relationships": [],
                }
            },
            "join_paths": {},
            "metrics": {
                "revenue": {
                    "description": "Total value of paid invoices.",
                    "sql": "SUM(invoices.grand_total)",
                    "filters": "invoices.status = 'paid'",
                    "synonyms": ["paid invoice value"],
                    "result_unit": "INR",
                    "tables": ["invoices"],
                    "legacy_field": "preserve me",
                }
            },
            "ambiguity_rules": {},
        }

    def test_metric_records_include_editable_business_glossary_fields(self):
        records = glossary_manager.metric_records_from_layer(self.semantic_layer)

        self.assertEqual(
            records,
            [
                {
                    "metric_name": "revenue",
                    "description": "Total value of paid invoices.",
                    "formula": "SUM(invoices.grand_total)",
                    "filters": "invoices.status = 'paid'",
                    "unit": "INR",
                    "owner": "",
                    "examples": "",
                    "ambiguity_rules": "",
                    "synonyms": "paid invoice value",
                    "tables": "invoices",
                    "enabled": True,
                }
            ],
        )

    def test_apply_glossary_records_preserves_existing_metric_fields_and_sql_contract(self):
        metric_rows = [
            {
                "metric_name": "revenue",
                "description": "Recognized paid invoice amount.",
                "formula": "SUM(invoices.grand_total)",
                "filters": "invoices.status = 'paid'",
                "unit": "INR",
                "owner": "Finance",
                "examples": "revenue last month, paid revenue by vendor",
                "ambiguity_rules": "Use paid invoices only.",
                "synonyms": "sales, paid bills",
                "tables": "invoices",
                "enabled": True,
            }
        ]

        updated = glossary_manager.apply_glossary_records(self.semantic_layer, metric_rows, [])
        revenue = updated["metrics"]["revenue"]

        self.assertEqual(revenue["sql"], "SUM(invoices.grand_total)")
        self.assertEqual(revenue["formula"], "SUM(invoices.grand_total)")
        self.assertEqual(revenue["result_unit"], "INR")
        self.assertEqual(revenue["unit"], "INR")
        self.assertEqual(revenue["owner"], "Finance")
        self.assertEqual(
            revenue["examples"],
            ["revenue last month", "paid revenue by vendor"],
        )
        self.assertEqual(revenue["ambiguity_rules"], "Use paid invoices only.")
        self.assertEqual(revenue["synonyms"], ["sales", "paid bills"])
        self.assertEqual(revenue["tables"], ["invoices"])
        self.assertEqual(revenue["legacy_field"], "preserve me")

    def test_apply_glossary_records_writes_ambiguity_rules_used_by_context_helpers(self):
        rule_rows = [
            {
                "rule_name": "revenue_definition",
                "trigger_phrases": "revenue",
                "resolved_by_phrases": "paid revenue",
                "applies_to_tables": "invoices",
                "ambiguous_dimensions_json": (
                    '[{"label": "paid_invoice_value", '
                    '"sql_hint": "SUM(invoices.grand_total) WHERE invoices.status = paid"}]'
                ),
                "clarification_question": "Do you mean paid invoices or all invoices?",
                "default_assumption": "",
                "reason": "Revenue definitions vary by business.",
                "enabled": True,
            }
        ]

        updated = glossary_manager.apply_glossary_records(self.semantic_layer, [], rule_rows)
        rule = find_required_clarification_rule(updated, ["invoices"], "Show revenue")

        self.assertIsNotNone(rule)
        self.assertEqual(rule["name"], "revenue_definition")
        self.assertEqual(
            rule["clarifying_question"],
            "Do you mean paid invoices or all invoices?",
        )

        resolved_rule = find_required_clarification_rule(
            updated,
            ["invoices"],
            "Show paid revenue",
        )
        self.assertIsNone(resolved_rule)

    def test_metric_glossary_fields_are_available_in_sql_context(self):
        metric_rows = [
            {
                "metric_name": "active_customer",
                "description": "Customer with a paid invoice in the last 90 days.",
                "formula": "COUNT(DISTINCT invoices.vendor_id)",
                "filters": "invoices.status = 'paid'",
                "unit": "count",
                "owner": "RevOps",
                "examples": "active customers this quarter",
                "ambiguity_rules": "Exclude prospects without paid invoices.",
                "synonyms": "active account",
                "tables": "invoices",
                "enabled": True,
            }
        ]

        updated = glossary_manager.apply_glossary_records(self.semantic_layer, metric_rows, [])
        context = build_sql_context(["invoices"], ["active_customer"], updated)

        self.assertIn('"owner": "RevOps"', context)
        self.assertIn('"examples": [', context)
        self.assertIn("Exclude prospects without paid invoices.", context)

    def test_validation_reports_duplicate_missing_formula_and_unknown_tables(self):
        result = glossary_manager.validate_metric_records(
            [
                {"metric_name": "revenue", "formula": "SUM(invoices.grand_total)", "tables": "invoices"},
                {"metric_name": "revenue", "formula": "", "tables": "missing_table"},
            ],
            available_tables={"invoices"},
        )

        self.assertIn("Metric 'revenue' is duplicated.", result["errors"])
        self.assertIn("Metric 'revenue' is missing a formula.", result["errors"])
        self.assertIn(
            "Metric 'revenue' references unknown tables: missing_table.",
            result["warnings"],
        )

    def test_save_and_load_semantic_layer_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "semantic_layer.json"

            glossary_manager.save_semantic_layer(self.semantic_layer, path)
            loaded = glossary_manager.load_semantic_layer(path)

        self.assertEqual(loaded, self.semantic_layer)


if __name__ == "__main__":
    unittest.main()
