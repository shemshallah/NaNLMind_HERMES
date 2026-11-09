import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import resident_benchmark as bench


def _key_tree(value):
    """Return only the key/type shape of a report, ignoring timing values."""
    if isinstance(value, dict):
        return {key: _key_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_key_tree(value[0])] if value else []
    return None


class BudgetTests(unittest.TestCase):
    def test_default_is_256_mib_and_hard_ceiling_is_1_gib(self):
        self.assertEqual(bench.DEFAULT_BUDGET_BYTES, 256 * 1024 * 1024)
        self.assertEqual(bench.HARD_BUDGET_BYTES, 1024 * 1024 * 1024)
        self.assertEqual(bench.resolve_budget(None), bench.DEFAULT_BUDGET_BYTES)
        self.assertEqual(bench.resolve_budget(None, profile="high"),
                         bench.DEFAULT_BUDGET_BYTES)

    def test_phone_profile_budgets_are_low_medium_high(self):
        self.assertEqual(set(bench.PHONE_PROFILES), {"low", "medium", "high"})
        self.assertEqual(bench.resolve_budget(None, profile="low"), 64 * 1024 * 1024)
        self.assertEqual(bench.resolve_budget(None, profile="medium"),
                         128 * 1024 * 1024)
        self.assertEqual(bench.resolve_budget(None, profile="high"),
                         256 * 1024 * 1024)

    def test_budget_accepts_hard_ceiling_and_rejects_beyond_it(self):
        self.assertEqual(bench.resolve_budget(bench.HARD_BUDGET_BYTES),
                         bench.HARD_BUDGET_BYTES)
        with self.assertRaises(bench.BudgetError):
            bench.resolve_budget(bench.HARD_BUDGET_BYTES + 1)

    def test_budget_rejects_invalid_and_below_minimum_values(self):
        for value in (0, -1, bench.MIN_BUDGET_BYTES - 1):
            with self.assertRaises(bench.BudgetError):
                bench.resolve_budget(value)
        for value in (True, 1.5, "256"):
            with self.assertRaises(bench.BudgetError):
                bench.resolve_budget(value)

    def test_budget_from_mb(self):
        self.assertEqual(bench.budget_from_mb(256), bench.DEFAULT_BUDGET_BYTES)
        self.assertEqual(bench.budget_from_mb(1024), bench.HARD_BUDGET_BYTES)
        with self.assertRaises(bench.BudgetError):
            bench.budget_from_mb(0)
        with self.assertRaises(bench.BudgetError):
            bench.budget_from_mb(float("inf"))

    def test_workload_clamps_records_to_the_budget(self):
        plan = bench.build_workload(
            "high", "full", budget_bytes=bench.MIN_BUDGET_BYTES)
        self.assertTrue(plan.budget_clamped)
        self.assertLess(plan.records, plan.requested_records)
        self.assertLessEqual(plan.dataset_bytes, bench.MIN_BUDGET_BYTES)
        self.assertEqual(plan.mapped_bytes, bench.MIN_BUDGET_BYTES)
        self.assertGreaterEqual(plan.records, 1)

    def test_unclamped_quick_workload_fits_default_budget(self):
        plan = bench.build_workload("high", "quick")
        self.assertFalse(plan.budget_clamped)
        self.assertEqual(plan.records, plan.requested_records)
        self.assertLessEqual(plan.dataset_bytes, bench.DEFAULT_BUDGET_BYTES)


class RunTests(unittest.TestCase):
    def _quick(self, profile="low", seed=4242):
        return bench.run_benchmark(
            profile=profile,
            quick=True,
            budget_bytes=bench.MIN_BUDGET_BYTES,
            seed=seed,
        )

    def test_quick_run_produces_required_metrics(self):
        report = self._quick()
        self.assertEqual(report["schema"], bench.REPORT_SCHEMA)
        self.assertEqual(report["mode"], "quick")
        self.assertEqual(report["profile"], "low")

        self.assertGreater(report["ingest"]["records"], 0)
        self.assertGreater(report["ingest"]["ops_per_sec"], 0.0)
        self.assertGreaterEqual(report["query_latency"]["samples"], 1)
        self.assertGreaterEqual(report["query_latency"]["p50_ms"], 0.0)
        self.assertGreaterEqual(report["query_latency"]["p95_ms"],
                                report["query_latency"]["p50_ms"])
        self.assertTrue(report["top_k"]["passed"])
        self.assertEqual(report["top_k"]["against_bruteforce"], True)
        self.assertGreaterEqual(report["top_k"]["recall_at_k"], 0.999)

        memory = report["memory"]
        self.assertLessEqual(memory["mapped_bytes"],
                             report["budget"]["effective_bytes"])
        self.assertEqual(memory["mapped_bytes"], memory["file_size_bytes"])
        self.assertLessEqual(memory["file_size_bytes"],
                             bench.HARD_BUDGET_BYTES)
        self.assertGreaterEqual(memory["resident_pages_estimate"], 1)

        self.assertGreaterEqual(report["checkpoint"]["total_ms"], 0.0)
        self.assertGreaterEqual(report["checkpoint"]["fsync_ms"], 0.0)
        self.assertGreaterEqual(report["reopen"]["reopen_ms"], 0.0)
        self.assertTrue(report["reopen"]["count_verified"])
        self.assertTrue(report["reopen"]["digest_verified"])
        self.assertTrue(report["verdict"]["passed"])

    def test_report_keys_are_deterministic_between_runs(self):
        first = self._quick(seed=73)
        second = self._quick(seed=73)
        self.assertEqual(_key_tree(first), _key_tree(second))
        self.assertEqual(list(first), list(second))
        for section in (
                "scope", "budget", "workload", "store", "memory", "ingest",
                "query_latency", "top_k", "checkpoint", "reopen", "verdict"):
            self.assertEqual(list(first[section]), list(second[section]))
        # Non-timing configuration must also match exactly.
        self.assertEqual(first["budget"], second["budget"])
        self.assertEqual(first["workload"], second["workload"])
        self.assertEqual(first["limitations"], second["limitations"])
        self.assertEqual(first["top_k"]["k"], second["top_k"]["k"])

    def test_json_and_markdown_reports_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "report.json"
            markdown_path = root / "report.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main([
                    "--profile", "low",
                    "--quick",
                    "--budget-mb", "1",
                    "--seed", "99",
                    "--json", str(json_path),
                    "--markdown", str(markdown_path),
                ])
            self.assertEqual(code, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], bench.REPORT_SCHEMA)
            self.assertTrue(data["verdict"]["passed"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertTrue(markdown.startswith("# Resident universe benchmark"))
            self.assertIn("query p50", markdown)
            self.assertIn("top-k", markdown)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["schema"], bench.REPORT_SCHEMA)

    def test_cli_rejects_over_hard_budget_without_creating_a_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "rejected.json"
            store_dir = root / "store"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = bench.main([
                    "--profile", "high",
                    "--quick",
                    "--budget-mb", "1025",
                    "--json", str(json_path),
                    "--store-dir", str(store_dir),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(json_path.exists())
            self.assertFalse(store_dir.exists())
            self.assertIn("hard 1 GiB ceiling", stderr.getvalue())

    def test_no_allocation_above_one_gib_is_attempted(self):
        self.assertEqual(bench.HARD_BUDGET_BYTES, 1024 ** 3)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(bench.BudgetError):
                bench.run_benchmark(
                    profile="high",
                    quick=True,
                    budget_bytes=bench.HARD_BUDGET_BYTES + 1,
                    store_dir=temporary,
                )
            self.assertEqual(list(Path(temporary).iterdir()), [])

        # The largest accepted plan maps exactly the ceiling, never more.
        plan = bench.build_workload(
            "high", "quick", budget_bytes=bench.HARD_BUDGET_BYTES)
        self.assertEqual(plan.mapped_bytes, bench.HARD_BUDGET_BYTES)
        self.assertLessEqual(plan.mapped_bytes, bench.HARD_BUDGET_BYTES)


if __name__ == "__main__":
    unittest.main()
