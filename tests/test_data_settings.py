import tempfile
import unittest
from pathlib import Path

from src.data_settings import (
    build_settings_context,
    fiscal_year_label,
    format_currency,
    load_data_settings,
    save_data_settings,
)
from src.pipeline import load_json, load_semantic_layer_bundle
from src.pipeline_config import SEMANTIC_LAYER_PATH
from src.pipeline_semantic_context import build_sql_context, clear_sql_context_cache


class DataSettingsTests(unittest.TestCase):
    def test_missing_store_returns_deterministic_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = load_data_settings(Path(tmpdir) / "missing.json")

        self.assertEqual(settings["default_currency"], "INR")
        self.assertEqual(settings["timezone"], "Asia/Kolkata")
        self.assertEqual(settings["today_anchor"], "2026-05-16")
        self.assertEqual(settings["fiscal_year_start"], {"month": 4, "day": 1})

    def test_save_normalizes_and_format_currency_uses_display_preferences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            settings = save_data_settings(
                {
                    "default_currency": "usd",
                    "fiscal_year_start": {"month": 13, "day": 0},
                    "today_anchor": "not-a-date",
                    "display_format": {
                        "currency": "code_suffix",
                        "decimal_places": 0,
                    },
                },
                path,
            )

            self.assertEqual(settings["default_currency"], "USD")
            self.assertEqual(settings["fiscal_year_start"], {"month": 12, "day": 1})
            self.assertEqual(settings["today_anchor"], "2026-05-16")
            self.assertEqual(format_currency(1234.56, settings), "1,235 USD")

    def test_settings_context_is_included_in_sql_context(self):
        semantic_layer = load_json(SEMANTIC_LAYER_PATH)
        settings = {
            "default_currency": "USD",
            "timezone": "America/New_York",
            "today_anchor": "2026-01-15",
            "fiscal_year_start": {"month": 7, "day": 1},
            "quarter_definition": "calendar",
        }
        clear_sql_context_cache()

        context = build_sql_context(["invoices"], ["revenue"], semantic_layer, settings)

        self.assertIn("global_data_settings", context)
        self.assertIn('"default_currency": "USD"', context)
        self.assertIn('"today_anchor": "2026-01-15"', context)
        self.assertIn('"fiscal_year_start": "07-01"', context)

    def test_fiscal_year_label_uses_configured_start(self):
        settings = {"fiscal_year_start": {"month": 4, "day": 1}}

        self.assertEqual(fiscal_year_label("2026-03-31", settings), "FY2025-26")
        self.assertEqual(fiscal_year_label("2026-04-01", settings), "FY2026-27")

    def test_semantic_layer_bundle_includes_settings_in_hash_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            semantic_path = Path(tmpdir) / "semantic.json"
            semantic_path.write_text(SEMANTIC_LAYER_PATH.read_text(encoding="utf-8"), encoding="utf-8")

            first = load_semantic_layer_bundle(str(semantic_path), semantic_path.stat().st_mtime_ns, 1)
            second = load_semantic_layer_bundle(str(semantic_path), semantic_path.stat().st_mtime_ns, 2)

        self.assertEqual(first["semantic_layer_hash"], second["semantic_layer_hash"])
        self.assertEqual(build_settings_context(first["data_settings"])["today_anchor"], "2026-05-16")


if __name__ == "__main__":
    unittest.main()
