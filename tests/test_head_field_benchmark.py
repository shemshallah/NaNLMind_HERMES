"""Tests for the real resident head-field benchmark."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import head_field_benchmark as bench

CELLS_AVAILABLE = bench.cells_available()
requires_cells = unittest.skipUnless(
    CELLS_AVAILABLE, "nanlmind_hermes.resident.cells is unavailable")


def _run(heads=8, steps=4, seed=1234):
    return bench.run_benchmark(heads, steps, seed=seed)


def _without_timing(value):
    """Strip wall-clock/throughput fields, which are host measurements."""
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"timing", "rates"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


class ArgumentBoundsTests(unittest.TestCase):
    def test_heads_accepts_full_boundary_range(self):
        self.assertEqual(bench.validate_heads(1), 1)
        self.assertEqual(bench.validate_heads(256), 256)
        self.assertEqual(bench.MAX_HEADS, 256)

    def test_heads_rejects_outside_1_to_256(self):
        for value in (0, -1, 257, 10_000):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_heads(value)
        for value in (True, 1.5, "256", None):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_heads(value)

    def test_steps_accepts_full_boundary_range(self):
        self.assertEqual(bench.validate_steps(1), 1)
        self.assertEqual(bench.validate_steps(1_000_000), 1_000_000)
        self.assertEqual(bench.MAX_STEPS, 1_000_000)

    def test_steps_rejects_outside_1_to_1000000(self):
        for value in (0, -1, 1_000_001, 10**12):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_steps(value)
        for value in (True, 2.5, "10", None):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_steps(value)

    def test_cli_defaults_are_256_heads_and_64_steps(self):
        args = bench.build_parser().parse_args([])
        self.assertEqual(args.heads, 256)
        self.assertEqual(args.steps, 64)

    def test_cli_rejects_out_of_range_heads_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "bad.json"
            markdown_path = root / "bad.md"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = bench.main([
                    "--heads", "257", "--steps", "10",
                    "--json", str(json_path),
                    "--markdown", str(markdown_path),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(json_path.exists())
            self.assertFalse(markdown_path.exists())
            self.assertIn("heads must be in [1, 256]", stderr.getvalue())

    def test_cli_rejects_out_of_range_steps_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "bad.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = bench.main([
                    "--heads", "8", "--steps", "1000001",
                    "--json", str(json_path),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(json_path.exists())
            self.assertIn("steps must be in [1, 1000000]", stderr.getvalue())


@requires_cells
class RunTests(unittest.TestCase):
    def test_independent_mode_activates_more_than_one_head(self):
        report = _run(heads=8, steps=4)
        mode = report["modes"]["independent"]
        field = mode["measured"]["field"]
        self.assertGreater(field["active_head_count"], 1)
        self.assertEqual(field["active_head_count"], 8)
        self.assertEqual(field["updated_head_count"], 8)
        self.assertGreater(field["distinct_activation_levels"], 1)
        self.assertGreater(field["field_entropy_bits"], 0.0)
        self.assertEqual(
            sum(item["count"]
                for item in field["activation_share_histogram"]), 8)
        self.assertEqual(
            mode["measured"]["updates"]["per_head"], 8 * 4)
        self.assertTrue(mode["checks"]["independent_multi_head"])
        self.assertTrue(mode["passed"])

    def test_single_head_control_has_exactly_one_active_head(self):
        report = _run(heads=8, steps=4)
        mode = report["modes"]["single_head"]
        field = mode["measured"]["field"]
        self.assertEqual(field["active_head_count"], 1)
        self.assertEqual(field["active_head_ids"], [0])
        self.assertEqual(field["updated_head_count"], 1)
        self.assertEqual(field["distinct_activation_levels"], 1)
        self.assertEqual(field["field_entropy_bits"], 0.0)
        histogram = field["activation_share_histogram"]
        self.assertEqual(sum(item["count"] for item in histogram), 8)
        self.assertEqual(histogram[0]["count"], 7)
        self.assertEqual(mode["measured"]["updates"]["per_head"], 4)
        self.assertTrue(mode["checks"]["single_head_exactly_one_active"])
        self.assertTrue(mode["checks"]["single_head_is_head_zero"])
        self.assertTrue(mode["passed"])

    def test_heads_one_is_valid_and_all_modes_pass(self):
        report = _run(heads=1, steps=2)
        self.assertEqual(report["config"]["heads"], 1)
        for mode_name in bench.MODES:
            self.assertEqual(
                report["modes"][mode_name]["measured"]["field"][
                    "active_head_count"], 1)
            self.assertTrue(report["modes"][mode_name]["passed"])
        self.assertTrue(report["verdict"]["passed"])

    def test_full_256_head_capacity_activates_all_heads(self):
        report = _run(heads=256, steps=1)
        self.assertEqual(report["config"]["heads"], 256)
        self.assertEqual(report["config"]["head_capacity"], 256)
        independent = report["modes"]["independent"]["measured"]["field"]
        self.assertEqual(independent["active_head_count"], 256)
        self.assertEqual(independent["distinct_activation_levels"], 256)
        coupled = report["modes"]["coupled"]["measured"]
        self.assertEqual(coupled["updates"]["pair_coupling"], 2 * 256)
        self.assertEqual(coupled["field"]["active_head_count"], 256)
        self.assertTrue(report["verdict"]["passed"])

    def test_uniform_control_has_one_activation_level(self):
        report = _run(heads=8, steps=4)
        field = report["modes"]["uniform"]["measured"]["field"]
        self.assertEqual(field["active_head_count"], 8)
        self.assertEqual(field["distinct_activation_levels"], 1)
        self.assertAlmostEqual(field["field_entropy_normalized"], 1.0,
                               places=12)
        self.assertAlmostEqual(field["top_head_share"], 1.0 / 8.0,
                               places=12)
        self.assertTrue(report["modes"]["uniform"]["passed"])

    def test_coupled_mode_measures_real_pair_updates(self):
        report = _run(heads=8, steps=3)
        mode = report["modes"]["coupled"]
        measured = mode["measured"]
        self.assertEqual(measured["updates"]["pair_coupling"], 2 * 8 * 3)
        self.assertIsNotNone(measured["pair_field"])
        self.assertEqual(measured["pair_field"]["unique_cells"], 8)
        self.assertTrue(measured["pair_field"]["generation_matches_updates"])
        self.assertEqual(
            measured["pair_field"]["generation_sum"],
            measured["updates"]["pair_coupling"])
        addressing = measured["addressing"]
        self.assertEqual(addressing["address_collisions"], 0)
        self.assertEqual(addressing["address_collision_rate"], 0.0)
        self.assertEqual(addressing["rail_key_collisions"], 0)
        self.assertEqual(addressing["rail_collision_rate"], 0.0)
        self.assertAlmostEqual(addressing["canonical_aliasing_rate"], 0.5)
        self.assertGreater(measured["field"]["active_head_count"], 1)
        self.assertTrue(mode["checks"]["coupled_pair_updates_present"])
        self.assertTrue(mode["checks"]["coupled_multi_head"])
        self.assertTrue(mode["passed"])

    def test_memory_time_and_operations_are_measured(self):
        report = _run(heads=8, steps=4)
        self.assertGreater(
            report["totals"]["timing"]["total_wall_seconds"], 0.0)
        self.assertEqual(report["totals"]["measured"]["memory_bytes"],
                         2 * 1024 * 1024)
        self.assertEqual(
            report["totals"]["measured"]["total_update_operations"],
            sum(report["modes"][mode_name]["measured"]["updates"]["total"]
                for mode_name in bench.MODES))
        for mode_name in bench.MODES:
            measured = report["modes"][mode_name]["measured"]
            self.assertEqual(measured["memory_bytes"], 2 * 1024 * 1024)
            self.assertEqual(measured["memory"]["buffer_bytes"],
                             2 * 1024 * 1024)
            self.assertEqual(measured["memory"]["region_bytes"],
                             2 * 1024 * 1024)
            self.assertGreater(measured["timing"]["total_wall_seconds"], 0.0)
            self.assertGreaterEqual(measured["timing"]["update_seconds"], 0.0)
            self.assertGreater(measured["rates"]["update_operations_per_sec"],
                               0.0)
            self.assertGreater(
                measured["rates"]["end_to_end_operations_per_sec"], 0.0)
            updates = measured["updates"]
            self.assertEqual(
                updates["total"],
                updates["per_head"] + updates["pair_coupling"])
            self.assertTrue(updates["head_generations_match_updates"])
            self.assertTrue(updates["pair_generations_match_updates"])

    def test_deterministic_output_on_fixed_seed(self):
        first = _run(heads=8, steps=4, seed=4242)
        second = _run(heads=8, steps=4, seed=4242)
        self.assertEqual(first["deterministic_fingerprint"],
                         second["deterministic_fingerprint"])
        self.assertEqual(_without_timing(first), _without_timing(second))
        self.assertEqual(
            json.dumps(_without_timing(first), sort_keys=True),
            json.dumps(_without_timing(second), sort_keys=True))
        # The target labels themselves are part of the deterministic output.
        for mode_name in bench.MODES:
            self.assertEqual(first["modes"][mode_name]["target"],
                             second["modes"][mode_name]["target"])

    def test_different_seed_changes_deterministic_measurements(self):
        first = _run(heads=8, steps=4, seed=1)
        second = _run(heads=8, steps=4, seed=2)
        self.assertNotEqual(first["deterministic_fingerprint"],
                            second["deterministic_fingerprint"])

    def test_measured_and_target_labels_exist_for_each_mode(self):
        report = _run(heads=8, steps=3)
        for mode_name in bench.MODES:
            mode = report["modes"][mode_name]
            self.assertIn("measured", mode)
            self.assertIn("target", mode)
            self.assertIn("updates", mode["measured"])
            self.assertIn("field", mode["measured"])
            self.assertIn("addressing", mode["measured"])
            self.assertTrue(mode["checks"])
            self.assertIn("passed", mode)
        self.assertIn("measure", bench.render_markdown(report).lower())
        self.assertIn("target", bench.render_markdown(report).lower())

    def test_no_canonical_merge_is_reported_and_not_performed(self):
        report = _run(heads=8, steps=3)
        self.assertFalse(report["scope"]["canonical_merge_performed"])
        self.assertFalse(
            report["merge_contract"]["canonical_merge_performed"])
        self.assertFalse(
            report["merge_contract"]["canonical_merge_api_called"])
        self.assertTrue(report["verdict"]["checks"]["no_canonical_merge"])
        markdown = bench.render_markdown(report)
        self.assertIn("| canonical merge performed | no |", markdown)
        self.assertNotIn("| canonical merge performed | yes |", markdown)

    def test_no_false_native_or_phone_performance_claim(self):
        report = _run(heads=8, steps=3)
        self.assertFalse(report["scope"]["native_code_used"])
        self.assertFalse(report["scope"]["phone_performance_claimed"])
        self.assertFalse(report["scope"]["handset_measurement"])
        markdown = bench.render_markdown(report)
        self.assertIn("native-code benchmark", markdown)
        self.assertIn("phone/handset performance measurement or claim", markdown)
        self.assertIn("| native code used | no |", markdown)
        self.assertIn("| phone/handset performance claim | no |", markdown)

    def test_json_and_markdown_reports_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "report.json"
            markdown_path = root / "report.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main([
                    "--heads", "8",
                    "--steps", "3",
                    "--seed", "99",
                    "--json", str(json_path),
                    "--markdown", str(markdown_path),
                ])
            self.assertEqual(code, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], bench.SCHEMA)
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["verdict"]["passed"])
            self.assertEqual(data["seed"], 99)
            self.assertEqual(data["deterministic_fingerprint"],
                             bench.deterministic_fingerprint(data))
            for mode_name in bench.MODES:
                self.assertIn("measured", data["modes"][mode_name])
                self.assertIn("target", data["modes"][mode_name])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertTrue(markdown.startswith(
                "# Head-field activation benchmark"))
            self.assertIn("## Active-head distribution", markdown)
            self.assertIn("## Measured vs target", markdown)
            self.assertIn("## Boundaries", markdown)
            self.assertIn("`independent`", markdown)
            self.assertIn("`single_head`", markdown)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["schema"], bench.SCHEMA)
            self.assertTrue(summary["verdict_passed"])
            self.assertEqual(summary["deterministic_fingerprint"],
                             data["deterministic_fingerprint"])


class SkipTests(unittest.TestCase):
    def test_missing_resident_cells_skips_gracefully_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "skip.json"
            markdown_path = root / "skip.md"
            with mock.patch.object(bench, "resident_cells", None):
                self.assertFalse(bench.cells_available())
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = bench.main([
                        "--heads", "8", "--steps", "3",
                        "--json", str(json_path),
                        "--markdown", str(markdown_path),
                    ])
            self.assertEqual(code, 0)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "skipped")
            self.assertTrue(data["verdict"]["skipped"])
            self.assertFalse(data["scope"]["resident_cells_available"])
            self.assertFalse(data["scope"]["canonical_merge_performed"])
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("**SKIPPED**", markdown)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
