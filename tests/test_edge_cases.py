import unittest

from src.edge_cases import classify_edge_cases


class EdgeCaseClassificationTests(unittest.TestCase):
    def assert_case_type(self, cases, case_type):
        self.assertIn(case_type, {case["type"] for case in cases})

    def test_no_rows_with_date_context_classifies_as_no_data_for_period(self):
        cases = classify_edge_cases(
            question="Show invoices from last month",
            sql_output={
                "SQL": "SELECT * FROM invoices WHERE invoice_date >= '2026-04-01'",
                "Resolved_Question": "Show invoices from last month",
            },
            sql_result=[],
            analysis={"Assumptions": ["Using invoice_date for last month."]},
        )

        self.assert_case_type(cases, "no_data_for_period")
        self.assertIn("broader date range", " ".join(cases[0]["next_actions"]))

    def test_sql_error_short_circuits_with_next_action(self):
        cases = classify_edge_cases(
            sql_output={"SQL": "SELECT * FROM missing_table"},
            sql_result="SQL validation failed: tables not present: missing_table",
        )

        self.assertEqual([case["type"] for case in cases], ["sql_error"])
        self.assertIn("SQL validation failed", cases[0]["explanation"])

    def test_missing_sql_classifies_as_unsupported_question(self):
        cases = classify_edge_cases(
            question="What is customer churn?",
            sql_output={
                "SQL": None,
                "Explanation": "No valid database table was selected for this question.",
            },
            sql_result=None,
        )

        self.assert_case_type(cases, "unsupported_question")
        self.assertIn("What can I ask you?", " ".join(cases[0]["next_actions"]))

    def test_supported_no_sql_analysis_modes_are_not_marked_unsupported(self):
        cases = classify_edge_cases(
            question="What can I ask you?",
            sql_output={"SQL": None, "Analysis_Mode": "data_overview"},
            sql_result=None,
            analysis={"Mode": "data_overview"},
        )

        self.assertEqual(cases, [])

    def test_too_many_rows_is_threshold_based(self):
        rows = [{"invoice_id": index} for index in range(4)]

        cases = classify_edge_cases(
            sql_output={"SQL": "SELECT * FROM invoices"},
            sql_result=rows,
            too_many_rows_threshold=3,
        )

        self.assert_case_type(cases, "too_many_rows")

    def test_result_content_hints_detect_duplicate_names_null_dates_and_mixed_currency(self):
        rows = [
            {
                "vendor_id": 1,
                "vendor_name": "Acme Supplies",
                "invoice_date": None,
                "currency": "INR",
            },
            {
                "vendor_id": 2,
                "vendor_name": "Acme Supplies",
                "invoice_date": "2026-01-05",
                "currency": "USD",
            },
        ]

        cases = classify_edge_cases(
            sql_output={"SQL": "SELECT * FROM invoices"},
            sql_result=rows,
        )

        self.assert_case_type(cases, "duplicate_entity_names")
        self.assert_case_type(cases, "null_dates")
        self.assert_case_type(cases, "mixed_currencies")

    def test_chart_error_classifies_as_chart_not_possible_without_hiding_rows(self):
        cases = classify_edge_cases(
            sql_output={"SQL": "SELECT status, COUNT(*) AS count FROM invoices GROUP BY status"},
            sql_result=[{"status": "paid", "count": 5}],
            chart_error="Need at least one numeric axis.",
        )

        self.assertEqual(cases[0]["type"], "chart_not_possible")
        self.assertIn("result table", " ".join(cases[0]["next_actions"]))

    def test_clarification_classifies_as_ambiguous_entity_name(self):
        cases = classify_edge_cases(
            sql_output={
                "SQL": None,
                "Requires_Clarification": True,
                "Clarification_Question": "Which Acme vendor do you mean?",
            },
            sql_result=None,
            analysis={"Clarification": {"needed": True}},
        )

        self.assertEqual([case["type"] for case in cases], ["ambiguous_entity_name"])
        self.assertIn("Which Acme vendor", cases[0]["explanation"])


if __name__ == "__main__":
    unittest.main()
