import json
import unittest


from src import pipeline, sqlite_runner
from src.pipeline_responses import is_executable_sql


def load_sql_runner():
    return sqlite_runner


class ClarificationSmokeScript(unittest.TestCase):
    def setUp(self):
        self.original_gemini_call = pipeline.gemini_call
        self.original_lookup_cache = pipeline.lookup_cache
        self.original_store_cache_entry = pipeline.store_cache_entry
        self.thread_id = "test-clarification-script-thread"
        pipeline.clear_conversation_memory(self.thread_id)

    def tearDown(self):
        pipeline.gemini_call = self.original_gemini_call
        pipeline.lookup_cache = self.original_lookup_cache
        pipeline.store_cache_entry = self.original_store_cache_entry
        pipeline.clear_conversation_memory(self.thread_id)

    def test_one_question_clarification_then_answer_generates_executable_sql(self):
        calls = []

        def fake_gemini_call(model_name, contents, trace_name="gemini-generate-content"):
            calls.append(trace_name)

            if trace_name == "gemini-question-resolution":
                return json.dumps(
                    {
                        "is_follow_up": True,
                        "standalone_question": "Show approval threshold bands by approver",
                        "memory_used": "The user answered the previous clarification question.",
                        "clarification_needed": False,
                        "clarifying_question": None,
                    }
                )

            if trace_name == "gemini-semantic-router":
                return json.dumps(
                    {
                        "tables": ["approval_matrix"],
                        "metrics": [],
                        "reason": "Approval threshold bands live in approval_matrix.",
                    }
                )

            if trace_name == "gemini-clarification-gate":
                if "Existing Clarification Attempts For This Pending Question\n0" in contents:
                    return json.dumps(
                        {
                            "clarification_needed": True,
                            "clarifying_question": (
                                "Do you want approval threshold bands or approver contact details?"
                            ),
                            "can_proceed": False,
                            "default_assumption": None,
                            "reason": "Approval details is underspecified.",
                            "unanswerable": False,
                        }
                    )

                return json.dumps(
                    {
                        "clarification_needed": False,
                        "clarifying_question": None,
                        "can_proceed": True,
                        "default_assumption": None,
                        "reason": "The user selected approval threshold bands.",
                        "unanswerable": False,
                    }
                )

            if trace_name == "gemini-sql-generation":
                return json.dumps(
                    {
                        "SQL": (
                            "SELECT approver_role, approver_name, min_amount, max_amount "
                            "FROM approval_matrix ORDER BY min_amount"
                        ),
                        "Explanation": "Reads approval threshold bands from approval_matrix.",
                        "Assumptions": "The user clarified they want threshold bands.",
                        "Followup_Questions": None,
                        "Chart": "none",
                    }
                )

            raise AssertionError(f"Unexpected Gemini call: {trace_name}")

        pipeline.gemini_call = fake_gemini_call
        pipeline.lookup_cache = lambda question, layer_hash: None
        pipeline.store_cache_entry = lambda **kwargs: {
            "stored": False,
            "reason": "Skipped in deterministic clarification smoke script.",
        }

        first_response = pipeline.generate_sql_for_question(
            "Show me approval details",
            thread_id=self.thread_id,
        )

        self.assertTrue(first_response["Requires_Clarification"])
        self.assertIsNone(first_response["SQL"])
        self.assertEqual(first_response["Clarification_Attempts"], 1)
        self.assertEqual(
            first_response["Clarification_Question"],
            "Do you want approval threshold bands or approver contact details?",
        )

        second_response = pipeline.generate_sql_for_question(
            "approval threshold bands",
            thread_id=self.thread_id,
        )

        self.assertFalse(second_response["Requires_Clarification"])
        self.assertFalse(second_response["Clarification_Limit_Reached"])
        self.assertTrue(is_executable_sql(second_response["SQL"]))
        self.assertEqual(
            second_response["Resolved_Question"],
            "Show approval threshold bands by approver",
        )

        sql_runner = load_sql_runner()
        sql_result = sql_runner.run_query(second_response["SQL"])

        self.assertIsInstance(sql_result, list)
        self.assertGreater(len(sql_result), 0)
        self.assertIn("approver_role", sql_result[0])
        self.assertIn("min_amount", sql_result[0])
        self.assertIn("gemini-clarification-gate", calls)
        self.assertIn("gemini-sql-generation", calls)


if __name__ == "__main__":
    unittest.main()
