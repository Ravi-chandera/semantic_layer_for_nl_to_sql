import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT_DIR / "data" / "golden_eval_cases.json"


class EvalCliTests(unittest.TestCase):
    def test_golden_cases_have_required_feedback_fields(self):
        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(cases), 30)
        self.assertLessEqual(len(cases), 50)
        for case in cases:
            self.assertIn("expected_sql_patterns", case)
            self.assertIn("requires_clarification", case)
            self.assertIn("latency_budget_ms", case)
            self.assertIn("cost_budget_usd", case)

        categories = {case["category"] for case in cases}
        self.assertEqual(
            categories,
            {"ambiguity", "joins", "metrics", "follow-ups", "driver analysis"},
        )
        self.assertTrue(any(case["expected_result"] for case in cases))
        self.assertTrue(any(case["requires_clarification"] for case in cases))

    def test_json_eval_runs_offline_keyword_baseline(self):
        completed = subprocess.run(
            [sys.executable, "-m", "src.eval", "--json"],
            cwd=ROOT_DIR,
            check=True,
            text=True,
            capture_output=True,
        )

        report = json.loads(completed.stdout)
        self.assertEqual(len(report["cases"]), 40)
        self.assertEqual(report["summaries"][0]["model"], "golden_expected")
        self.assertEqual(report["summaries"][1]["model"], "keyword_rule_baseline")
        self.assertEqual(report["summaries"][1]["cost_usd"], 0.0)
        for category in ["ambiguity", "joins", "metrics", "follow-ups", "driver analysis"]:
            self.assertIn(category, report["summaries"][1])

    def test_table_output_includes_failure_categories(self):
        completed = subprocess.run(
            [sys.executable, "-m", "src.eval", "--show-failures"],
            cwd=ROOT_DIR,
            check=True,
            text=True,
            capture_output=True,
        )

        output = completed.stdout
        self.assertIn("keyword_rule_baseline", output)
        self.assertIn("ambiguity", output)
        self.assertIn("joins", output)
        self.assertIn("metrics", output)
        self.assertIn("follow-ups", output)
        self.assertIn("driver analysis", output)


if __name__ == "__main__":
    unittest.main()
