import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import pipeline
from src.entity_search import search_entities


def build_semantic_layer():
    return {
        "tables": {
            "departments": {
                "description": "Internal departments.",
                "primary_key": "id",
                "columns": {
                    "id": {"type": "INTEGER"},
                    "name": {"type": "TEXT"},
                },
            },
            "vendors": {
                "description": "External vendors.",
                "primary_key": "id",
                "columns": {
                    "id": {"type": "INTEGER"},
                    "name": {"type": "TEXT"},
                },
            },
            "invoices": {
                "description": "Vendor invoices.",
                "primary_key": "id",
                "columns": {
                    "id": {"type": "INTEGER"},
                    "invoice_number": {"type": "TEXT"},
                },
            },
        },
        "metrics": {},
    }


def build_temp_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE vendors (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE invoices (id INTEGER PRIMARY KEY, invoice_number TEXT);
            INSERT INTO departments (id, name) VALUES (1, 'Engineering');
            INSERT INTO vendors (id, name) VALUES (10, 'Engineering Services');
            INSERT INTO vendors (id, name) VALUES (11, 'Acme Systems');
            INSERT INTO invoices (id, invoice_number) VALUES (100, 'INV-2026-001');
            """
        )
        conn.commit()
    finally:
        conn.close()


class EntitySearchTests(unittest.TestCase):
    def test_entity_search_returns_structured_ambiguity_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "entities.db"
            build_temp_db(db_path)

            result = search_entities("Engineering", build_semantic_layer(), db_path=db_path)

        self.assertTrue(result["ambiguous"])
        self.assertIn("Did you mean", result["clarifying_question"])
        self.assertEqual(
            [option["label"] for option in result["options"]],
            ["Engineering department", "Engineering Services vendor"],
        )
        self.assertEqual(
            result["options"][0]["entity_match"]["table"],
            "departments",
        )

    def test_entity_search_context_handles_misspelled_single_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "entities.db"
            build_temp_db(db_path)

            result = search_entities(
                "Show invoices for Acme Sistem",
                build_semantic_layer(),
                db_path=db_path,
            )

        self.assertFalse(result["ambiguous"])
        self.assertEqual(result["context"][0]["table"], "vendors")
        self.assertEqual(result["context"][0]["value"], "Acme Systems")
        self.assertEqual(result["context"][0]["match_type"], "fuzzy")

    def test_pipeline_blocks_sql_generation_for_entity_ambiguity(self):
        state = {
            "user_question": "Engineering",
            "resolved_question": "Engineering",
            "semantic_layer": build_semantic_layer(),
            "router_response": {"tables": ["departments"], "metrics": []},
        }
        entity_search = {
            "ambiguous": True,
            "clarifying_question": "Did you mean one of these matches for Engineering?",
            "options": [
                {
                    "id": "departments:name:1",
                    "label": "Engineering department",
                    "resolution_text": "Use department Engineering.",
                },
                {
                    "id": "vendors:name:10",
                    "label": "Engineering Services vendor",
                    "resolution_text": "Use vendor Engineering Services.",
                },
            ],
            "context": [],
        }

        with patch.object(pipeline, "search_entities", return_value=entity_search):
            result = pipeline.select_semantic_context_node(state)
            result.update(
                {
                    "user_question": state["user_question"],
                    "resolved_question": state["resolved_question"],
                    "semantic_layer": state["semantic_layer"],
                    "question_resolution": {"is_follow_up": False},
                    "conversation_turns": [],
                }
            )
            clarification = pipeline.evaluate_clarification_node(result)

        self.assertTrue(clarification["clarification_blocks_sql"])
        self.assertTrue(clarification["sql_response"]["Requires_Clarification"])
        self.assertEqual(
            clarification["sql_response"]["Clarification_Options"],
            entity_search["options"],
        )


if __name__ == "__main__":
    unittest.main()
