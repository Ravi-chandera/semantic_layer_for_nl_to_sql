import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from pipeline import (  # noqa: E402
    MAX_CLARIFICATION_ATTEMPTS,
    clarification_attempts_for_current_question,
    normalize_sql_response_after_generation,
)


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


if __name__ == "__main__":
    unittest.main()
