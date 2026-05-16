import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

import analysis_workflow  # noqa: E402
from pipeline import load_json  # noqa: E402
from pipeline_config import SEMANTIC_LAYER_PATH  # noqa: E402


class AnalysisWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic_layer = load_json(str(SEMANTIC_LAYER_PATH))
        cls.sql_runner = analysis_workflow.load_default_sql_runner()

    def test_revenue_drop_question_builds_multi_evidence_analysis(self):
        queries = analysis_workflow.build_revenue_drop_evidence_queries()
        evidence = [
            analysis_workflow._run_evidence_query(query, self.sql_runner)
            for query in queries
        ]
        sql_output = {
            "Resolved_Question": "Why did revenue drop last month?",
            "Clarification_Decision": {"clarification_needed": False},
        }

        analysis = analysis_workflow._build_revenue_drop_analysis(
            "Why did revenue drop last month?",
            sql_output,
            evidence,
            self.semantic_layer,
        )

        self.assertEqual(analysis["Mode"], "metric_driver_investigation")
        self.assertTrue(analysis["SQL_Is_Supporting_Evidence"])
        self.assertGreaterEqual(len(analysis["Assumptions"]), 3)
        self.assertIn("Period_Comparison", analysis)

        evidence_names = {item["name"] for item in analysis["Evidence"]}
        self.assertIn("period_comparison", evidence_names)
        self.assertIn("vendor_drivers", evidence_names)
        self.assertIn("monthly_trend", evidence_names)
        self.assertTrue(all(item["status"] == "ok" for item in analysis["Evidence"]))

        cited_tables = {item["table"] for item in analysis["Citations"]["tables"]}
        cited_columns = {item["column"] for item in analysis["Citations"]["columns"]}
        self.assertIn("invoices", cited_tables)
        self.assertIn("vendors", cited_tables)
        self.assertIn("invoices.grand_total", cited_columns)
        self.assertIn("invoices.invoice_date", cited_columns)
        self.assertIn("invoices.status", cited_columns)

        self.assertIn("level", analysis["Confidence"])
        self.assertGreaterEqual(len(analysis["Suggested_Next_Queries"]), 3)
        self.assertEqual(
            len(analysis["Suggested_Next_Queries"]),
            len(analysis["Suggested_Next_Query_Evidence"]),
        )
        for suggestion in analysis["Suggested_Next_Query_Evidence"]:
            self.assertIn("question", suggestion)
            self.assertIn("source_evidence", suggestion)
            self.assertIn("supporting_facts", suggestion)
            self.assertIn(
                suggestion["source_evidence"],
                {"vendor_drivers", "status_mix", "period_comparison"},
            )

        suggestion_text = " ".join(analysis["Suggested_Next_Queries"]).lower()
        self.assertNotIn("department", suggestion_text)

    def test_evidence_sql_safety_reports_non_read_and_missing_table_errors(self):
        unsafe_query = analysis_workflow.EvidenceQuery(
            name="unsafe",
            purpose="Prove non-read SQL is blocked.",
            sql="DELETE FROM invoices",
            tables=["invoices"],
            columns=["invoices.id"],
        )
        missing_table_query = analysis_workflow.EvidenceQuery(
            name="missing_table",
            purpose="Prove missing tables are reported.",
            sql="SELECT * FROM definitely_missing_table",
            tables=["definitely_missing_table"],
            columns=[],
        )

        unsafe_result = analysis_workflow._run_evidence_query(
            unsafe_query,
            self.sql_runner,
        )
        missing_table_result = analysis_workflow._run_evidence_query(
            missing_table_query,
            self.sql_runner,
        )

        self.assertEqual(unsafe_result["status"], "error")
        self.assertIn("only SELECT/WITH", unsafe_result["error"])
        self.assertEqual(missing_table_result["status"], "error")
        self.assertIn("tables not present", missing_table_result["error"])

    def test_generic_sql_output_wraps_as_analysis_with_sql_evidence(self):
        sql = "SELECT COUNT(*) AS invoice_count FROM invoices"
        sql_result = [{"invoice_count": 101}]
        sql_output = {
            "SQL": sql,
            "Resolved_Question": "How many invoices exist?",
            "Assumptions": "Count all invoice rows.",
            "Selected_Tables": ["invoices"],
            "Selected_Metrics": [],
        }

        analysis = analysis_workflow._generic_analysis_from_sql(
            "How many invoices exist?",
            sql_output,
            sql_result,
            sql,
            self.semantic_layer,
        )

        self.assertEqual(analysis["Mode"], "single_query_analysis")
        self.assertTrue(analysis["SQL_Is_Supporting_Evidence"])
        self.assertEqual(len(analysis["Evidence"]), 1)
        self.assertEqual(analysis["Evidence"][0]["sql"], sql)
        self.assertEqual(analysis["Evidence"][0]["result_preview"], sql_result)
        self.assertIn("101", str(analysis["Evidence"][0]["result_preview"]))

    def test_generic_analysis_reports_missing_sql_as_failed_evidence(self):
        sql_output = {
            "SQL": None,
            "Resolved_Question": "Can this be answered?",
            "Assumptions": "No SQL was returned.",
            "Selected_Tables": ["invoices"],
            "Selected_Metrics": [],
        }

        analysis = analysis_workflow._generic_analysis_from_sql(
            "Can this be answered?",
            sql_output,
            None,
            None,
            self.semantic_layer,
        )

        self.assertEqual(analysis["Mode"], "single_query_analysis")
        self.assertEqual(analysis["Evidence"][0]["status"], "error")
        self.assertIn("No executable SQL", analysis["Evidence"][0]["error"])
        self.assertIn("did not return executable SQL", analysis["Limitations"][0])

    def test_revenue_drop_workflow_treats_optional_breakdown_clarification_as_assumption(self):
        original_generator = analysis_workflow.generate_sql_for_question

        def fake_generate_sql_for_question(*args, **kwargs):
            return {
                "SQL": None,
                "Resolved_Question": "Why did revenue drop last month?",
                "Requires_Clarification": True,
                "Clarification_Question": (
                    "To help analyze the drop, would you like to see the revenue "
                    "breakdown by vendor, product, or department compared to the previous month?"
                ),
                "Followup_Questions": (
                    "To help analyze the drop, would you like to see the revenue "
                    "breakdown by vendor, product, or department compared to the previous month?"
                ),
                "Clarification_Decision": {"clarification_needed": True},
                "Selected_Tables": ["invoices"],
                "Selected_Metrics": ["revenue"],
            }

        analysis_workflow.generate_sql_for_question = fake_generate_sql_for_question
        try:
            result = analysis_workflow.run_ai_native_analysis(
                "Why did revenue drop last month?",
                sql_runner=self.sql_runner,
                use_llm_synthesis=False,
            )
        finally:
            analysis_workflow.generate_sql_for_question = original_generator

        analysis = result["analysis"]
        self.assertEqual(analysis["Mode"], "metric_driver_investigation")
        self.assertFalse(analysis["Clarification"]["needed"])
        self.assertTrue(analysis["Clarification"]["handled_as_assumption"])
        self.assertFalse(result["sql_output"]["Requires_Clarification"])
        self.assertTrue(result["sql_output"]["Original_Requires_Clarification"])
        self.assertGreaterEqual(len(analysis["Evidence"]), 4)
        self.assertTrue(all(item["status"] == "ok" for item in analysis["Evidence"]))

    def test_llm_synthesis_cannot_replace_data_backed_next_queries(self):
        analysis = {
            "Mode": "metric_driver_investigation",
            "Clarification": {"needed": False},
            "Executive_Answer": "Deterministic answer.",
            "Confidence": {"level": "high", "score": 0.9, "reasons": []},
            "Limitations": ["Original limitation."],
            "Evidence": [],
            "Suggested_Next_Queries": [
                "Show paid invoices for OmniTech Hardware in 2025-12 and 2026-01."
            ],
            "Suggested_Next_Query_Evidence": [
                {
                    "question": "Show paid invoices for OmniTech Hardware in 2025-12 and 2026-01.",
                    "source_evidence": "vendor_drivers",
                    "supporting_facts": {"vendor_name": "OmniTech Hardware"},
                }
            ],
        }
        original_gemini_call = analysis_workflow.gemini_call

        def fake_gemini_call(*args, **kwargs):
            return """
            {
              "Executive_Answer": "Synthesized answer.",
              "Confidence_Adjustment": "Synthesis stayed grounded.",
              "Limitations": ["Synthesized limitation."],
              "Suggested_Next_Queries": ["Analyze customer churn by region."]
            }
            """

        analysis_workflow.gemini_call = fake_gemini_call
        try:
            synthesized = analysis_workflow._try_llm_synthesis(
                analysis,
                "fake-model",
            )
        finally:
            analysis_workflow.gemini_call = original_gemini_call

        self.assertEqual(synthesized["Executive_Answer"], "Synthesized answer.")
        self.assertEqual(
            synthesized["Suggested_Next_Queries"],
            ["Show paid invoices for OmniTech Hardware in 2025-12 and 2026-01."],
        )
        self.assertEqual(
            synthesized["Suggested_Next_Query_Evidence"][0]["source_evidence"],
            "vendor_drivers",
        )


if __name__ == "__main__":
    unittest.main()
