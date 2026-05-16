import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import feedback_store
from src.pipeline_semantic_context import build_sql_context, clear_sql_context_cache


class FeedbackStoreTests(unittest.TestCase):
    def test_add_list_and_summarize_feedback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.db"

            feedback_store.add_feedback(
                sentiment="down",
                categories=["wrong_join", "wrong_metric", "ignored"],
                note="Use purchase_orders for the vendor join.",
                question="Top vendors",
                resolved_question="Top vendors by invoice total",
                generated_sql="SELECT * FROM invoices",
                metrics=["invoice_total"],
                tables=["invoices", "vendors"],
                chat_id="chat-1",
                thread_id="thread-1",
                turn_index=0,
                message_id="message-1",
                db_path=db_path,
            )
            feedback_store.add_feedback(
                sentiment="up",
                question="Looks good",
                db_path=db_path,
            )

            records = feedback_store.list_feedback(db_path=db_path)
            self.assertEqual(2, len(records))
            correction = next(record for record in records if record["sentiment"] == "down")
            self.assertEqual(["wrong_join", "wrong_metric"], correction["categories"])
            self.assertEqual(["invoice_total"], correction["metrics"])
            self.assertEqual(["invoices", "vendors"], correction["tables"])

            summary = feedback_store.summarize_corrections(db_path=db_path)
            self.assertEqual(2, summary["total_feedback"])
            self.assertEqual(1, summary["negative_feedback"])
            self.assertEqual(1, summary["by_category"]["wrong_join"])
            self.assertEqual(1, summary["by_metric"]["invoice_total"])
            self.assertEqual(1, summary["by_table"]["vendors"])
            self.assertIn("Top vendors by invoice total", summary["by_question"])

    def test_invalid_sentiment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "feedback.db"

            with self.assertRaises(ValueError):
                feedback_store.add_feedback(sentiment="maybe", db_path=db_path)

    def test_semantic_context_includes_relevant_corrections(self):
        clear_sql_context_cache()
        semantic_layer = {
            "tables": {
                "invoices": {"description": "Invoices", "columns": {}},
                "vendors": {"description": "Vendors", "columns": {}},
            },
            "metrics": {
                "invoice_total": {
                    "description": "Total invoice value",
                    "sql": "SUM(invoices.grand_total)",
                    "tables": ["invoices"],
                }
            },
            "join_paths": {},
        }
        correction_context = {
            "analyst_correction_summary": {
                "negative_feedback": 1,
                "by_category": {"missing_filter": 1},
                "by_metric": {"invoice_total": 1},
                "by_table": {"invoices": 1},
                "examples_by_category": {
                    "missing_filter": [
                        {
                            "question": "Paid invoice total",
                            "note": "Filter status to paid.",
                            "metrics": ["invoice_total"],
                            "tables": ["invoices"],
                        }
                    ]
                },
            }
        }

        with patch(
            "src.pipeline_semantic_context.build_semantic_correction_context",
            return_value=correction_context,
        ):
            context = build_sql_context(["invoices"], ["invoice_total"], semantic_layer)

        self.assertIn("semantic_corrections", context)
        self.assertIn("missing_filter", context)
        self.assertIn("Filter status to paid.", context)


if __name__ == "__main__":
    unittest.main()
