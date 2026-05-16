import unittest

from src.answer_modes import (
    ANSWER_MODE_ANALYST,
    ANSWER_MODE_AUDIT,
    ANSWER_MODE_EXECUTIVE,
    answer_mode_label,
    apply_answer_mode_to_sql_output,
    normalize_answer_mode,
)


class AnswerModeTests(unittest.TestCase):
    def test_normalize_accepts_keys_and_labels(self):
        self.assertEqual(
            normalize_answer_mode(ANSWER_MODE_EXECUTIVE),
            ANSWER_MODE_EXECUTIVE,
        )
        self.assertEqual(
            normalize_answer_mode("Audit evidence"),
            ANSWER_MODE_AUDIT,
        )
        self.assertEqual(
            normalize_answer_mode("unknown"),
            ANSWER_MODE_ANALYST,
        )

    def test_label_uses_normalized_mode(self):
        self.assertEqual(answer_mode_label(ANSWER_MODE_AUDIT), "Audit evidence")
        self.assertEqual(answer_mode_label(None), "Analyst detail")

    def test_apply_answer_mode_adds_output_and_analysis_metadata(self):
        sql_output = {
            "SQL": "SELECT 1",
            "Analysis": {
                "Executive_Answer": "One row.",
                "Metadata": {"source": "test"},
            },
            "Metadata": {"trace": "abc"},
        }

        updated = apply_answer_mode_to_sql_output(sql_output, ANSWER_MODE_EXECUTIVE)

        self.assertEqual(updated["Answer_Mode"], ANSWER_MODE_EXECUTIVE)
        self.assertEqual(updated["Answer_Mode_Label"], "Executive summary")
        self.assertEqual(updated["Metadata"]["answer_mode"], ANSWER_MODE_EXECUTIVE)
        self.assertEqual(updated["Metadata"]["trace"], "abc")
        self.assertEqual(
            updated["Analysis"]["Metadata"]["answer_mode"],
            ANSWER_MODE_EXECUTIVE,
        )
        self.assertEqual(updated["Analysis"]["Metadata"]["source"], "test")


if __name__ == "__main__":
    unittest.main()
