import tempfile
import unittest
from pathlib import Path

from src import certified_question_store


class CertifiedQuestionStoreTests(unittest.TestCase):
    def test_save_list_and_filter_templates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "certified_questions.db"

            first = certified_question_store.save_certified_question(
                title="Top vendors",
                question="Top vendors by outstanding amount",
                category="Vendors",
                owner="AP Ops",
                approved_sql="SELECT vendor_name, SUM(outstanding_amount) FROM invoices GROUP BY vendor_name",
                notes="Approved for finance reviews.",
                tags="vendors, outstanding, Vendors",
                source_chat_id="chat-1",
                source_message_id="message-1",
                db_path=db_path,
            )
            certified_question_store.save_certified_question(
                title="Draft overdue",
                question="Overdue invoices by department",
                category="Invoices",
                certified=False,
                active=True,
                db_path=db_path,
            )
            certified_question_store.save_certified_question(
                title="Inactive vendors",
                question="Inactive vendor template",
                category="Vendors",
                certified=True,
                active=False,
                db_path=db_path,
            )

            self.assertEqual(["vendors", "outstanding"], first["tags"])
            self.assertTrue(first["active"])
            self.assertTrue(first["certified"])
            self.assertEqual("chat-1", first["source_chat_id"])

            all_records = certified_question_store.list_certified_questions(db_path=db_path)
            templates = certified_question_store.list_certified_questions(
                active_only=True,
                certified_only=True,
                db_path=db_path,
            )
            vendor_templates = certified_question_store.list_certified_questions(
                active_only=True,
                certified_only=True,
                category="Vendors",
                db_path=db_path,
            )

            self.assertEqual(3, len(all_records))
            self.assertEqual(["Top vendors"], [record["title"] for record in templates])
            self.assertEqual(["Top vendors"], [record["title"] for record in vendor_templates])
            self.assertEqual(["Vendors"], certified_question_store.list_template_categories(db_path=db_path))

    def test_update_preserves_created_at_and_can_deactivate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "certified_questions.db"

            saved = certified_question_store.save_certified_question(
                question_id="question-1",
                title="Original",
                question="Original question",
                tags=["one"],
                db_path=db_path,
            )
            updated = certified_question_store.save_certified_question(
                question_id="question-1",
                title="Updated",
                question="Updated question",
                tags=["two"],
                db_path=db_path,
            )
            inactive = certified_question_store.set_certified_question_active(
                "question-1",
                False,
                db_path=db_path,
            )

            self.assertEqual(saved["created_at"], updated["created_at"])
            self.assertEqual("Updated", updated["title"])
            self.assertEqual(["two"], updated["tags"])
            self.assertFalse(inactive["active"])
            self.assertEqual(
                [],
                certified_question_store.list_certified_questions(
                    active_only=True,
                    certified_only=True,
                    db_path=db_path,
                ),
            )

    def test_title_and_question_are_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "certified_questions.db"

            with self.assertRaises(ValueError):
                certified_question_store.save_certified_question(
                    title="",
                    question="Valid question",
                    db_path=db_path,
                )
            with self.assertRaises(ValueError):
                certified_question_store.save_certified_question(
                    title="Valid title",
                    question="",
                    db_path=db_path,
                )


if __name__ == "__main__":
    unittest.main()
