import unittest


from src import analysis_workflow
from src.pipeline import load_json
from src.pipeline_config import SEMANTIC_LAYER_PATH


class AnalysisWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic_layer = load_json(str(SEMANTIC_LAYER_PATH))
        cls.sql_runner = analysis_workflow.load_default_sql_runner()

    def test_metric_change_question_builds_semantic_layer_evidence_analysis(self):
        queries = analysis_workflow.build_metric_driver_evidence_queries("revenue", self.semantic_layer)
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

        evidence_names = {item["name"] for item in analysis["Evidence"]}
        self.assertIn("dataset_row_counts", evidence_names)
        self.assertTrue(all(item["status"] == "ok" for item in analysis["Evidence"]))

        cited_tables = {item["table"] for item in analysis["Citations"]["tables"]}
        self.assertIn("invoices", cited_tables)

        self.assertIn("level", analysis["Confidence"])
        confidence_codes = {
            reason["code"] for reason in analysis["Confidence"]["reason_codes"]
        }
        self.assertIn("exact_metric_found", confidence_codes)
        self.assertGreaterEqual(len(analysis["Suggested_Next_Queries"]), 1)

    def test_what_can_i_ask_returns_data_overview_without_sql_generation(self):
        original_generator = analysis_workflow.generate_sql_for_question

        def fail_generate_sql_for_question(*args, **kwargs):
            raise AssertionError("Overview requests should not call SQL generation.")

        analysis_workflow.generate_sql_for_question = fail_generate_sql_for_question
        try:
            result = analysis_workflow.run_ai_native_analysis(
                "What can I ask you?",
                sql_runner=self.sql_runner,
                use_llm_synthesis=False,
            )
        finally:
            analysis_workflow.generate_sql_for_question = original_generator

        analysis = result["analysis"]
        self.assertEqual(analysis["Mode"], "data_overview")
        self.assertFalse(analysis["Clarification"]["needed"])
        self.assertIsNone(result["sql_output"]["SQL"])
        self.assertIn("Dataset_Overview", analysis)

        overview = analysis["Dataset_Overview"]
        self.assertIn("invoices", overview["row_counts"])
        self.assertIn("vendors", overview["row_counts"])
        self.assertIn("revenue", overview["metrics"])
        self.assertIn("total_spend", overview["metrics"])
        self.assertIn("total_liability", overview["metrics"])
        self.assertGreaterEqual(len(overview["example_questions"]), 5)

        examples = " ".join(overview["example_questions"]).lower()
        self.assertIn("records", examples)
        self.assertNotIn("customer churn", examples)

    def test_full_analysis_builds_and_runs_multi_question_plan(self):
        original_generator = analysis_workflow.generate_sql_for_question

        def fail_generate_sql_for_question(*args, **kwargs):
            raise AssertionError("Broad analysis requests should use the deterministic planner.")

        analysis_workflow.generate_sql_for_question = fail_generate_sql_for_question
        try:
            result = analysis_workflow.run_ai_native_analysis(
                "Do a full analysis of accounts payable data",
                sql_runner=self.sql_runner,
                use_llm_synthesis=False,
            )
        finally:
            analysis_workflow.generate_sql_for_question = original_generator

        analysis = result["analysis"]
        self.assertEqual(analysis["Mode"], "planned_dataset_analysis")
        self.assertFalse(analysis["Clarification"]["needed"])
        self.assertIsNone(result["sql_output"]["SQL"])
        self.assertGreaterEqual(len(analysis["Analysis_Plan"]), 5)
        self.assertGreaterEqual(len(analysis["Evidence"]), 5)
        self.assertTrue(all(item["status"] == "ok" for item in analysis["Evidence"]))

        plan_text = " ".join(item["question"] for item in analysis["Analysis_Plan"]).lower()
        self.assertIn("invoice", plan_text)
        self.assertIn("purchase", plan_text)
        self.assertNotIn("customer churn", plan_text)

        evidence_tables = {
            table
            for item in analysis["Evidence"]
            for table in item.get("tables", [])
        }
        self.assertIn("invoices", evidence_tables)
        self.assertIn("purchase_orders", evidence_tables)

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

    def test_confidence_reasons_lower_multi_table_answer_without_join_review(self):
        analysis = {
            "Confidence": {"level": "high", "score": 0.91, "reasons": []},
            "Definitions": [{"metric": "total_spend"}],
            "Evidence": [
                {
                    "status": "ok",
                    "tables": ["invoices"],
                    "checks": [],
                    "sql": "SELECT COUNT(*) FROM invoices",
                    "metric": "total_spend",
                }
            ],
            "Citations": {
                "tables": [{"table": "invoices"}, {"table": "vendors"}],
            },
            "Assumptions": [],
            "Limitations": [],
        }

        augmented = analysis_workflow._augment_confidence_reasons(
            analysis,
            self.semantic_layer,
            {"Selected_Tables": ["invoices", "vendors"], "Selected_Metrics": ["total_spend"]},
        )

        reason_codes = {
            reason["code"] for reason in augmented["Confidence"]["reason_codes"]
        }
        self.assertIn("missing_join_path_review", reason_codes)
        self.assertEqual(augmented["Confidence"]["level"], "low")
        self.assertLessEqual(augmented["Confidence"]["score"], 0.54)
        self.assertIn("no join path was reviewed", augmented["Confidence"]["summary"])

    def test_confidence_reasons_keep_exact_metric_and_join_as_high(self):
        analysis = {
            "Confidence": {"level": "high", "score": 0.91, "reasons": []},
            "Definitions": [{"metric": "revenue"}],
            "Evidence": [
                {
                    "status": "ok",
                    "tables": ["invoices", "vendors"],
                    "checks": [],
                    "sql": "SELECT v.name FROM invoices i JOIN vendors v ON i.vendor_id = v.id",
                    "metric": "revenue",
                }
            ],
            "Citations": {
                "tables": [{"table": "invoices"}, {"table": "vendors"}],
            },
            "Assumptions": [],
            "Limitations": [],
        }

        augmented = analysis_workflow._augment_confidence_reasons(
            analysis,
            self.semantic_layer,
            {"Selected_Tables": ["invoices", "vendors"], "Selected_Metrics": ["revenue"]},
        )

        reason_codes = {
            reason["code"] for reason in augmented["Confidence"]["reason_codes"]
        }
        self.assertIn("exact_metric_found", reason_codes)
        self.assertIn("join_path_reviewed", reason_codes)
        self.assertEqual(augmented["Confidence"]["level"], "high")

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
        self.assertGreaterEqual(len(analysis["Evidence"]), 2)
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
                "Show the strongest metric driver in the latest period."
            ],
            "Suggested_Next_Query_Evidence": [
                {
                    "question": "Show the strongest metric driver in the latest period.",
                    "source_evidence": "semantic_layer_plan",
                    "supporting_facts": {"metric": "revenue"},
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
            ["Show the strongest metric driver in the latest period."],
        )
        self.assertEqual(
            synthesized["Suggested_Next_Query_Evidence"][0]["source_evidence"],
            "semantic_layer_plan",
        )


if __name__ == "__main__":
    unittest.main()
