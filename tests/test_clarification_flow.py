import unittest


from src.pipeline import (
    MAX_CLARIFICATION_ATTEMPTS,
    clarification_attempts_for_current_question,
    is_executable_sql,
    load_json,
    lookup_cache_node,
    normalize_sql_response_after_generation,
)
from src.pipeline_config import SEMANTIC_LAYER_PATH
from src.pipeline_responses import clarification_needed_response
from src.pipeline_semantic_context import clarification_options_from_rule
from src.pipeline_semantic_context import find_required_clarification_rule
from src.pipeline_semantic_context import expand_selected_tables_for_context


class ClarificationFlowTests(unittest.TestCase):
    def test_no_pending_clarification_starts_at_zero_attempts(self):
        state = {
            "conversation_turns": [],
            "question_resolution": {"is_follow_up": False, "memory_used": None},
        }

        self.assertEqual(clarification_attempts_for_current_question(state), 0)

    def test_pending_clarification_counts_when_user_replies_to_it(self):
        state = {
            "conversation_turns": [
                {
                    "requires_clarification": True,
                    "sql": None,
                    "clarification_attempts": 1,
                }
            ],
            "question_resolution": {
                "is_follow_up": True,
                "memory_used": "Answered the prior clarification question.",
            },
        }

        self.assertEqual(clarification_attempts_for_current_question(state), 1)

    def test_pending_clarification_does_not_leak_to_new_question(self):
        state = {
            "conversation_turns": [
                {
                    "requires_clarification": True,
                    "sql": None,
                    "clarification_attempts": 1,
                }
            ],
            "question_resolution": {"is_follow_up": False, "memory_used": None},
        }

        self.assertEqual(clarification_attempts_for_current_question(state), 0)

    def test_sql_followup_is_normalized_to_single_clarification_attempt(self):
        state = {
            "conversation_turns": [],
            "question_resolution": {"is_follow_up": False, "memory_used": None},
        }
        sql_response = {
            "SQL": None,
            "Explanation": "The amount type is ambiguous.",
            "Assumptions": "No assumption selected.",
            "Followup_Questions": "Do you mean billed amount or paid amount?",
            "Chart": "none",
        }

        normalized = normalize_sql_response_after_generation(sql_response, state)

        self.assertTrue(normalized["Requires_Clarification"])
        self.assertEqual(normalized["Clarification_Attempts"], 1)
        self.assertEqual(
            normalized["Clarification_Question"],
            "Do you mean billed amount or paid amount?",
        )

    def test_sql_followup_stops_after_max_clarification_attempts(self):
        state = {
            "conversation_turns": [
                {
                    "requires_clarification": True,
                    "sql": None,
                    "clarification_attempts": MAX_CLARIFICATION_ATTEMPTS,
                }
            ],
            "question_resolution": {
                "is_follow_up": True,
                "memory_used": "Answered the prior clarification question.",
            },
        }
        sql_response = {
            "SQL": None,
            "Explanation": "Still ambiguous.",
            "Assumptions": "No assumption selected.",
            "Followup_Questions": "Which date do you mean?",
            "Chart": "none",
        }

        normalized = normalize_sql_response_after_generation(sql_response, state)

        self.assertFalse(normalized["Requires_Clarification"])
        self.assertTrue(normalized["Clarification_Limit_Reached"])
        self.assertIsNone(normalized["Followup_Questions"])

    def test_insufficient_data_sql_text_is_not_treated_as_sql(self):
        state = {
            "conversation_turns": [],
            "question_resolution": {"is_follow_up": False, "memory_used": None},
        }
        sql_response = {
            "SQL": "Insufficient data in context: invoices table.",
            "Explanation": "Could not answer with selected context.",
            "Assumptions": None,
            "Followup_Questions": None,
            "Chart": "none",
        }

        normalized = normalize_sql_response_after_generation(sql_response, state)

        self.assertIsNone(normalized["SQL"])
        self.assertFalse(normalized["Requires_Clarification"])
        self.assertEqual(
            normalized["Explanation"],
            "Insufficient data in context: invoices table.",
        )

    def test_executable_sql_detection_accepts_read_queries_only(self):
        self.assertTrue(is_executable_sql("SELECT * FROM invoices"))
        self.assertTrue(is_executable_sql("-- comment\nWITH totals AS (SELECT 1) SELECT * FROM totals"))
        self.assertFalse(is_executable_sql("Insufficient data in context: invoices table."))
        self.assertFalse(is_executable_sql("DELETE FROM invoices"))

    def test_approval_details_matches_no_default_clarification_rule(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        rule = find_required_clarification_rule(
            semantic_layer,
            ["approval_matrix"],
            "Show me approval details",
        )

        self.assertIsNotNone(rule)
        self.assertEqual(rule["name"], "approval_details_ambiguity")
        self.assertEqual(
            rule["clarifying_question"],
            "Do you want approval threshold bands or approver contact details?",
        )

    def test_cache_lookup_skips_no_default_clarification_rule(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        result = lookup_cache_node(
            {
                "user_question": "Show me approval details",
                "resolved_question": "Show me approval details",
                "semantic_layer": semantic_layer,
            }
        )

        self.assertFalse(result["cache_hit"])
        self.assertTrue(result["cache_lookup"]["skipped"])
        self.assertEqual(
            result["cache_lookup"]["matched_rule"],
            "approval_details_ambiguity",
        )

    def test_bridge_tables_are_added_for_department_invoice_queries(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        tables = expand_selected_tables_for_context(
            ["invoices", "departments"],
            [],
            semantic_layer,
        )

        self.assertIn("invoices", tables)
        self.assertIn("departments", tables)
        self.assertIn("purchase_orders", tables)

    def test_metric_source_tables_are_added_when_router_selects_metric_only(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        tables = expand_selected_tables_for_context(
            [],
            ["revenue"],
            semantic_layer,
        )

        self.assertEqual(tables, ["invoices"])

    def test_top_vendor_question_requires_clarification_until_metric_is_supplied(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        rule = find_required_clarification_rule(
            semantic_layer,
            ["vendors"],
            "Who are our top 5 vendors?",
        )

        self.assertIsNotNone(rule)
        self.assertEqual(rule["name"], "top_vendors_ambiguity")

        resolved_rule = find_required_clarification_rule(
            semantic_layer,
            ["vendors"],
            "Who are our top 5 vendors by total invoice value?",
        )

        self.assertIsNone(resolved_rule)

    def test_top_vendor_clarification_options_come_from_semantic_layer(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)

        rule = find_required_clarification_rule(
            semantic_layer,
            ["vendors"],
            "Who are our top 5 vendors?",
        )

        self.assertEqual(
            [option["label"] for option in rule["options"]],
            [
                "Rank by invoice value",
                "Rank by count",
                "Rank by vendor rating",
                "Rank by payment value",
            ],
        )
        self.assertEqual(
            rule["options"][0]["resolution_text"],
            "Rank top vendors by total invoice value.",
        )

    def test_clarification_options_helper_derives_labels_for_plain_dimensions(self):
        options = clarification_options_from_rule(
            "approval_details_ambiguity",
            {
                "ambiguous_dimensions": [
                    {
                        "label": "approval_threshold_bands",
                        "sql_hint": "approval_matrix.min_amount",
                    }
                ]
            },
        )

        self.assertEqual(options[0]["label"], "Use Approval Threshold Bands")
        self.assertEqual(options[0]["resolution_text"], "Use approval threshold bands.")

    def test_clarification_needed_response_preserves_structured_options(self):
        options = [{"id": "invoice_value", "label": "Rank by invoice value"}]

        response = clarification_needed_response(
            "How should I rank vendors?",
            clarification_options=options,
        )

        self.assertEqual(response["Clarification_Options"], options)


if __name__ == "__main__":
    unittest.main()
