"""Offline evaluation CLI for NL-to-SQL golden cases.

The evaluator is intentionally deterministic: it runs a keyword/rule baseline
over checked-in golden cases and never calls an external model or API.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = ROOT_DIR / "data" / "golden_eval_cases.json"
DEFAULT_DB_PATH = ROOT_DIR / "data" / "assignment.db"
FAILURE_CATEGORIES = ("ambiguity", "joins", "metrics", "follow-ups", "driver analysis")


@dataclass(frozen=True)
class BaselinePrediction:
    sql: str | None
    requires_clarification: bool = False
    clarification_question: str | None = None


class KeywordRuleBaseline:
    """Small non-AI baseline used to make eval regressions visible offline."""

    TOP_VENDOR_CLARIFICATION = (
        "Should I rank the top vendors by total invoice value, invoice count, "
        "payment value, or vendor rating?"
    )

    APPROVAL_CLARIFICATION = "Do you want approval threshold bands or approver contact details?"
    DATE_CLARIFICATION = (
        "Are you referring to the date the bill was issued, the date it is due, "
        "or the date it was paid?"
    )
    AMOUNT_CLARIFICATION = (
        "Do you mean the total billed amount including tax, the taxable subtotal, "
        "or the actual amount paid?"
    )

    def predict(self, question: str) -> BaselinePrediction:
        q = question.lower()

        if "approval details" in q or "approver details" in q:
            return BaselinePrediction(None, True, self.APPROVAL_CLARIFICATION)
        if "top" in q and ("vendor" in q or "supplier" in q) and not any(
            phrase in q
            for phrase in (
                "by total invoice value",
                "by invoice value",
                "by spend",
                "by payment value",
                "by invoice count",
                "by rating",
            )
        ):
            return BaselinePrediction(None, True, self.TOP_VENDOR_CLARIFICATION)
        if "by date" in q and "invoice" in q:
            return BaselinePrediction(None, True, self.DATE_CLARIFICATION)
        if re.search(r"\b(amount|value|total)\b", q) and "which amount" in q:
            return BaselinePrediction(None, True, self.AMOUNT_CLARIFICATION)

        rules: list[tuple[bool, str]] = [
            (
                ("count" in q or "how many invoices are in the system" in q)
                and "invoice" in q
                and "paid" not in q,
                "SELECT COUNT(*) AS invoice_count FROM invoices",
            ),
            (
                "paid" in q and "invoice" in q and ("count" in q or "many" in q),
                "SELECT COUNT(*) AS paid_invoice_count FROM invoices WHERE status = 'paid'",
            ),
            (
                ("total spend" in q or "cash outflow" in q) and "vendor" not in q,
                "SELECT ROUND(SUM(amount), 2) AS total_spend FROM payments",
            ),
            (
                "outstanding" in q or "total liability" in q,
                "SELECT ROUND(SUM(grand_total), 2) AS total_liability "
                "FROM invoices WHERE status IN ('approved','validated','on_hold','partially_paid','received')",
            ),
            (
                "average invoice" in q or "mean bill" in q,
                "SELECT ROUND(AVG(grand_total), 2) AS average_invoice_value FROM invoices",
            ),
            (
                "watchlist" in q and "vendor" in q,
                "SELECT COUNT(*) AS watchlist_vendor_count FROM vendors WHERE is_watchlist = 1",
            ),
            (
                ("how many vendors" in q or "count vendors" in q or "supplier count" in q),
                "SELECT COUNT(*) AS vendor_count FROM vendors",
            ),
            (
                "closed" in q and ("purchase order" in q or " po" in q),
                "SELECT COUNT(*) AS closed_po_count FROM purchase_orders WHERE status = 'closed'",
            ),
            (
                "rejection rate" in q or "defect rate" in q,
                "SELECT ROUND(SUM(quantity_rejected) * 100.0 / "
                "NULLIF(SUM(quantity_received + quantity_rejected), 0), 2) AS rejection_rate "
                "FROM grn_line_items",
            ),
            (
                "missing grn" in q,
                "SELECT COUNT(*) AS missing_grn_invoice_count FROM invoices WHERE deviation_type = 'Missing GRN'",
            ),
            (
                "rate mismatch" in q,
                "SELECT COUNT(*) AS rate_mismatch_invoice_count FROM invoices WHERE deviation_type = 'Rate Mismatch'",
            ),
            (
                "top" in q
                and ("vendor" in q or "supplier" in q)
                and ("invoice value" in q or "spend" in q),
                "SELECT v.name AS vendor_name, ROUND(SUM(i.grand_total), 2) AS total_invoice_value "
                "FROM invoices i JOIN vendors v ON i.vendor_id = v.id "
                "GROUP BY v.id, v.name ORDER BY SUM(i.grand_total) DESC LIMIT 5",
            ),
            (
                "top" in q and ("vendor" in q or "supplier" in q) and "payment value" in q,
                "SELECT v.name AS vendor_name, ROUND(SUM(p.amount), 2) AS total_payment_value "
                "FROM payments p JOIN invoices i ON p.invoice_id = i.id "
                "JOIN vendors v ON i.vendor_id = v.id "
                "GROUP BY v.id, v.name ORDER BY SUM(p.amount) DESC LIMIT 5",
            ),
            (
                "engineering" in q and "invoice" in q,
                "SELECT COUNT(*) AS engineering_invoice_count "
                "FROM invoices i JOIN purchase_orders po ON i.po_id = po.id "
                "JOIN departments d ON po.department_id = d.id WHERE d.name = 'Engineering'",
            ),
            (
                "rtgs" in q and "payment" in q,
                "SELECT COUNT(*) AS rtgs_payment_count FROM payments WHERE payment_mode = 'RTGS'",
            ),
            (
                "2025" in q and "invoice" in q,
                "SELECT COUNT(*) AS invoice_count_2025 FROM invoices WHERE strftime('%Y', invoice_date) = '2025'",
            ),
            (
                "march 2026" in q and "invoice" in q,
                "SELECT COUNT(*) AS invoice_count_march_2026 "
                "FROM invoices WHERE invoice_date >= '2026-03-01' AND invoice_date < '2026-04-01'",
            ),
            (
                "over 1000000" in q or "over 1,000,000" in q or "above 1000000" in q,
                "SELECT COUNT(*) AS po_count_over_1000000 FROM purchase_orders WHERE total_amount > 1000000",
            ),
            (
                "who approves" in q and "2500000" in q,
                "SELECT approver_name FROM approval_matrix WHERE 2500000 BETWEEN min_amount AND max_amount",
            ),
            (
                "payment mode" in q and ("breakdown" in q or "count" in q),
                "SELECT payment_mode, COUNT(*) AS payment_count FROM payments "
                "GROUP BY payment_mode ORDER BY payment_count DESC",
            ),
        ]

        for matched, sql in rules:
            if matched:
                return BaselinePrediction(sql)
        return BaselinePrediction("SELECT COUNT(*) AS row_count FROM invoices")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Golden eval file must contain a JSON list.")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    required = {
        "id",
        "question",
        "category",
        "expected_sql_patterns",
        "requires_clarification",
        "latency_budget_ms",
        "cost_budget_usd",
    }
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(f"Case {case.get('id', '<unknown>')} missing fields: {missing}")
    if case["category"] not in FAILURE_CATEGORIES:
        raise ValueError(f"Case {case['id']} has invalid category: {case['category']}")
    if not isinstance(case["expected_sql_patterns"], list):
        raise ValueError(f"Case {case['id']} expected_sql_patterns must be a list.")


def execute_sql(db_path: Path, sql: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql).fetchall()
    return [dict(row) for row in rows]


def value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 0.01
        except (TypeError, ValueError):
            return False
    return expected == actual


def expected_result_matches(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    if len(actual) < len(expected):
        return False
    for expected_row, actual_row in zip(expected, actual):
        for key, expected_value in expected_row.items():
            if key not in actual_row or not value_matches(expected_value, actual_row[key]):
                return False
    return True


def sql_patterns_match(patterns: list[str], sql: str | None) -> bool:
    if not patterns:
        return sql is None
    if sql is None:
        return False
    return all(re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def evaluate_case(case: dict[str, Any], db_path: Path, baseline: KeywordRuleBaseline) -> dict[str, Any]:
    started = time.perf_counter()
    prediction = baseline.predict(case["question"])
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    passed = True
    reasons: list[str] = []
    actual_result: list[dict[str, Any]] | None = None

    if prediction.requires_clarification != case["requires_clarification"]:
        passed = False
        reasons.append("clarification expectation mismatch")

    expected_clarification = case.get("expected_clarification")
    if expected_clarification:
        if not prediction.clarification_question or expected_clarification.lower() not in prediction.clarification_question.lower():
            passed = False
            reasons.append("clarification text mismatch")

    if not case["requires_clarification"]:
        if not sql_patterns_match(case["expected_sql_patterns"], prediction.sql):
            passed = False
            reasons.append("sql pattern mismatch")
        if prediction.sql and case.get("expected_result"):
            try:
                actual_result = execute_sql(db_path, prediction.sql)
            except sqlite3.Error as exc:
                passed = False
                reasons.append(f"sql execution failed: {exc}")
            else:
                if not expected_result_matches(case["expected_result"], actual_result):
                    passed = False
                    reasons.append("result mismatch")

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "failure_category": None if passed else case["category"],
        "reasons": reasons,
        "sql": prediction.sql,
        "requires_clarification": prediction.requires_clarification,
        "clarification_question": prediction.clarification_question,
        "latency_ms": latency_ms,
        "cost_usd": 0.0,
        "expected_result": case.get("expected_result"),
        "actual_result": actual_result,
    }


def summarize(model_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "model": model_name,
        "cases": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "latency_ms": round(sum(result["latency_ms"] for result in results), 3),
        "cost_usd": round(sum(result["cost_usd"] for result in results), 6),
    }
    for category in FAILURE_CATEGORIES:
        summary[category] = sum(1 for result in results if result["failure_category"] == category)
    return summary


def oracle_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "model": "golden_expected",
        "cases": len(cases),
        "passed": len(cases),
        "failed": 0,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }
    for category in FAILURE_CATEGORIES:
        summary[category] = 0
    return summary


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["model", "cases", "passed", "failed", *FAILURE_CATEGORIES, "latency_ms", "cost_usd"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep = "-+-".join("-" * widths[header] for header in headers)
    body = [
        " | ".join(str(row[header]).ljust(widths[header]) for header in headers)
        for row in rows
    ]
    return "\n".join([line, sep, *body])


def run_eval(cases_path: Path, db_path: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    baseline = KeywordRuleBaseline()
    results = [evaluate_case(case, db_path, baseline) for case in cases]
    summaries = [oracle_summary(cases), summarize("keyword_rule_baseline", results)]
    return {"cases": cases, "results": results, "summaries": summaries}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic offline NL-to-SQL evals.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--show-failures", action="store_true", help="Print failed baseline cases.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_eval(args.cases, args.db)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("Offline NL-to-SQL eval")
    print(f"Cases: {args.cases}")
    print(f"Database: {args.db}")
    print()
    print(format_table(report["summaries"]))

    if args.show_failures:
        failures = [result for result in report["results"] if not result["passed"]]
        if failures:
            print("\nBaseline failures")
            for failure in failures:
                reason = "; ".join(failure["reasons"])
                print(f"- {failure['id']} [{failure['category']}]: {reason}")
        else:
            print("\nBaseline failures: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
