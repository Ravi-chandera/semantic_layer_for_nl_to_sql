import tempfile
import unittest
from pathlib import Path


from src import benchmark_store


class BenchmarkStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        benchmark_store.BENCHMARK_DB_PATH = temp_path / "benchmark_results.db"
        benchmark_store.BENCHMARK_DASHBOARD_PATH = temp_path / "benchmark_dashboard.html"
        benchmark_store._BENCHMARK_INITIALIZED = False

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_only_records_and_summary_metrics(self):
        base = {
            "run_id": "run-1",
            "source": "chat",
            "question": "How many invoices were raised last month?",
            "started_at": "2026-05-09T00:00:00+00:00",
            "ended_at": "2026-05-09T00:00:02+00:00",
            "latency_ms": 2000,
            "sql_output": {
                "SQL": "SELECT COUNT(*) FROM invoices",
                "Clarification_Attempts": 0,
                "Requires_Clarification": False,
                "Cache_Hit": False,
                "Selected_Tables": ["invoices"],
            },
            "sql_result": [{"count": 10}],
        }

        first = benchmark_store.build_benchmark_record(**base)
        second = benchmark_store.build_benchmark_record(
            **{
                **base,
                "run_id": "run-2",
                "latency_ms": 4000,
                "sql_output": {
                    **base["sql_output"],
                    "Requires_Clarification": True,
                    "Clarification_Attempts": 1,
                    "Clarification_Question": "Which date field do you mean?",
                },
                "sql_result": None,
            }
        )

        benchmark_store.append_benchmark_record(first)
        benchmark_store.append_benchmark_record(second)

        records = benchmark_store.list_benchmark_records()
        summary = benchmark_store.get_benchmark_summary()

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["avg_latency_ms"], 3000)
        self.assertEqual(summary["avg_clarifying_questions"], 0.5)
        self.assertEqual(summary["sql_generation_rate"], 1)
        self.assertEqual(summary["sql_execution_success_rate"], 0.5)

    def test_dashboard_file_is_generated_from_records(self):
        record = benchmark_store.build_benchmark_record(
            run_id="run-1",
            source="benchmark_suite",
            question="Show me all unpaid bills.",
            started_at="2026-05-09T00:00:00+00:00",
            ended_at="2026-05-09T00:00:01+00:00",
            latency_ms=1000,
            sql_output={
                "SQL": "SELECT * FROM invoices WHERE status <> 'paid'",
                "Selected_Tables": ["invoices"],
            },
            sql_result=[],
        )
        benchmark_store.append_benchmark_record(record)

        dashboard_path, html_text = benchmark_store.write_benchmark_dashboard()

        self.assertTrue(dashboard_path.exists())
        self.assertIn("NL-to-SQL Benchmark Dashboard", html_text)
        self.assertIn("Show me all unpaid bills.", html_text)


if __name__ == "__main__":
    unittest.main()
