"""Tests for the NaN lane/cycle readout benchmark."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import nan_cycle_benchmark as bench

FULL_PREFIXES = (48, 256, bench.CYCLE_SPACE_DIMENSION)
SMALL_PREFIXES = (48, 256)
REPORT_CACHE = {}


def cached_report(prefixes=FULL_PREFIXES, seed=bench.DEFAULT_SEED, steps=1):
    key = (tuple(prefixes), seed, steps, "fake")
    if key not in REPORT_CACHE:
        REPORT_CACHE[key] = bench.run_benchmark(
            prefixes=prefixes, steps=steps, seed=seed, backend="fake")
    return REPORT_CACHE[key]


def fresh_report(prefixes=SMALL_PREFIXES, seed=bench.DEFAULT_SEED, steps=1):
    return bench.run_benchmark(
        prefixes=prefixes, steps=steps, seed=seed, backend="fake")


def without_timing(value):
    """Strip wall-clock/throughput fields, which are host measurements."""
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if key not in {"timing", "rates"}
            and not key.endswith("_seconds")
            and not key.endswith("_per_sec")
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


class BoundaryTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        self.assertEqual(bench.MAX_CYCLES, 32385)
        self.assertEqual(bench.MIN_CYCLES, 1)
        self.assertEqual(bench.MAX_STEPS, 1_000_000)
        self.assertEqual(bench.DEFAULT_PREFIXES, (48, 256, 32385))
        self.assertEqual(bench.CYCLE_SPACE_DIMENSION, 32385)
        self.assertEqual(bench.PAIR_CAPACITY, 32640)
        self.assertEqual(bench.HEADS, 256)

    def test_validate_prefixes_accepts_boundaries_and_defaults(self):
        self.assertEqual(bench.validate_prefixes((1,)), (1,))
        self.assertEqual(bench.validate_prefixes((32385,)), (32385,))
        self.assertEqual(bench.validate_prefixes(FULL_PREFIXES), FULL_PREFIXES)
        self.assertEqual(bench.parse_prefixes("48,256,32385"), FULL_PREFIXES)

    def test_validate_prefixes_rejects_out_of_range_and_bad_shapes(self):
        for value in (0, -1, 32386, 10**9):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_prefixes((value,))
        for value in (48.0, True, "48", None):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_prefixes((value,))
        for value in ((), (48, 48), (256, 48), [48, 32386]):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_prefixes(value)

    def test_parse_prefixes_rejects_malformed_text(self):
        for text in ("", "48,", ",48", "48,,256", "48,x", "48;256"):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.parse_prefixes(text)

    def test_validate_steps_accepts_boundaries(self):
        self.assertEqual(bench.validate_steps(1), 1)
        self.assertEqual(bench.validate_steps(1_000_000), 1_000_000)

    def test_validate_steps_rejects_outside_bounds(self):
        for value in (0, -1, 1_000_001, 10**12):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_steps(value)
        for value in (True, 2.0, "1", None):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_steps(value)

    def test_validate_backend_accepts_known_names_and_rejects_others(self):
        for value in ("auto", "resident", "fake"):
            self.assertEqual(bench.validate_backend(value), value)
        for value in ("phone", "", None, 1):
            with self.assertRaises(bench.BenchmarkConfigError):
                bench.validate_backend(value)


class CycleBasisTests(unittest.TestCase):
    def test_basis_has_full_cycle_space_dimension(self):
        self.assertEqual(len(bench.CYCLE_BASIS), 32385)
        self.assertEqual(len(set(bench.CYCLE_BASIS)), len(bench.CYCLE_BASIS))
        for first, second in bench.CYCLE_BASIS:
            self.assertGreaterEqual(first, 1)
            self.assertLess(first, second)
            self.assertLess(second, 256)

    def test_first_48_cycles_are_disjoint_round029_triangles(self):
        expected = tuple((1 + 2 * k, 2 + 2 * k) for k in range(48))
        self.assertEqual(bench.CYCLE_BASIS[:48], expected)
        vertices = []
        for first, second in expected:
            vertices.extend((first, second))
        self.assertEqual(len(vertices), len(set(vertices)))

    def test_wrong_vertex_is_distinct_and_deterministic(self):
        for first, second in bench.CYCLE_BASIS[:256]:
            decoy = bench.wrong_vertex(first, second)
            self.assertNotIn(decoy, (first, second))
            self.assertGreaterEqual(decoy, 1)
            self.assertLess(decoy, 256)
            self.assertEqual(decoy, bench.wrong_vertex(first, second))

    def test_full_prefix_requires_all_pair_lanes(self):
        lanes = bench.required_lanes(bench.CYCLE_BASIS)
        self.assertEqual(len(lanes), 32640)
        self.assertEqual(len(set(lanes)), len(lanes))
        self.assertIn((0, 1), lanes)
        self.assertIn((1, 2), lanes)
        self.assertIn((254, 255), lanes)


class FakeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = cached_report()

    def test_report_has_requested_prefixes_and_counts(self):
        config = self.report["config"]
        self.assertEqual(config["cycle_prefixes"], list(FULL_PREFIXES))
        self.assertEqual(config["backend"], "fake")
        self.assertEqual(config["lane_count"], 32640)
        self.assertEqual(config["lane_capacity"], 32640)
        self.assertEqual(self.report["measurement"]["lanes_written"], 32640)
        for count in FULL_PREFIXES:
            key = str(count)
            for mode_name in bench.MODES:
                measured = self.report["modes"][mode_name][
                    "measured"]["prefixes"][key]
                self.assertEqual(measured["cycles"], count)

    def test_cycle_is_exact_under_both_gauges(self):
        for count in FULL_PREFIXES:
            key = str(count)
            cycle = self.report["modes"]["cycle"]["measured"]["prefixes"][key]
            conventional = self.report["modes"]["conventional_cycle"][
                "measured"]["prefixes"][key]
            self.assertEqual(cycle["clean_exact_vs_truth"], count)
            self.assertEqual(cycle["gauge_linear_exact_invariance"], count)
            self.assertEqual(cycle["gauge_scrambled_exact_invariance"], count)
            self.assertEqual(cycle["gauge_linear_mismatches"], 0)
            self.assertEqual(cycle["gauge_scrambled_mismatches"], 0)
            self.assertEqual(conventional["gauge_linear_exact_invariance"],
                             count)
            self.assertEqual(conventional["gauge_scrambled_exact_invariance"],
                             count)
            self.assertTrue(
                self.report["modes"]["cycle"]["passed"])
            self.assertTrue(
                self.report["modes"]["conventional_cycle"]["passed"])

    def test_raw_and_wrong_fail_exactly_under_nonuniform_gauges(self):
        for count in FULL_PREFIXES:
            key = str(count)
            for mode_name in ("raw", "wrong_triangle"):
                measured = self.report["modes"][mode_name][
                    "measured"]["prefixes"][key]
                self.assertEqual(measured["clean_exact_vs_truth"], count)
                self.assertEqual(measured["gauge_linear_exact_invariance"], 0)
                self.assertEqual(measured["gauge_scrambled_exact_invariance"],
                                 0)
                self.assertEqual(measured["gauge_linear_mismatches"], count)
                self.assertEqual(measured["gauge_scrambled_mismatches"],
                                 count)
                self.assertFalse(
                    self.report["modes"][mode_name]["observed"][
                        "gauge_invariance_holds"])
                self.assertFalse(
                    self.report["modes"][mode_name]["target"][
                        "gauge_invariance_expected"])
                # The mode passes because the expected failure was observed.
                self.assertTrue(self.report["modes"][mode_name]["passed"])

    def test_uniform_and_noise_controls(self):
        for count in FULL_PREFIXES:
            key = str(count)
            uniform = self.report["modes"]["uniform"][
                "measured"]["prefixes"][key]
            noise = self.report["modes"]["noise"][
                "measured"]["prefixes"][key]
            self.assertEqual(uniform["cycle_exact_invariance"], count)
            self.assertEqual(uniform["raw_exact_invariance"], count)
            self.assertEqual(uniform["cycle_mismatches"], 0)
            self.assertEqual(uniform["raw_mismatches"], 0)
            self.assertEqual(noise["cycle_exact_invariance"], 0)
            self.assertEqual(noise["raw_exact_invariance"], 0)
            self.assertEqual(noise["cycle_mismatches"], count)
            self.assertEqual(noise["raw_mismatches"], count)
            self.assertTrue(self.report["modes"]["uniform"]["passed"])
            self.assertTrue(self.report["modes"]["noise"]["passed"])
            self.assertEqual(
                self.report["comparison"]["uniform_control"][key][
                    "cycle_exact"], count)
            self.assertEqual(
                self.report["comparison"]["noise_control"][key][
                    "cycle_exact"], 0)

    def test_conventional_cycle_ties_cycle_exactly(self):
        for count in FULL_PREFIXES:
            key = str(count)
            cycle = self.report["modes"]["cycle"]["measured"]["prefixes"][key]
            conventional = self.report["modes"]["conventional_cycle"][
                "measured"]["prefixes"][key]
            for digest in ("clean_digest", "gauge_linear_digest",
                           "gauge_scrambled_digest", "noise_digest"):
                self.assertEqual(cycle[digest], conventional[digest])
            self.assertTrue(
                self.report["comparison"]["prefixes"][key][
                    "conventional_ties_cycle"])

    def test_operations_and_bytes_are_measured_with_targets(self):
        expected_ops = {
            "cycle": 5,
            "raw": 1,
            "wrong_triangle": 5,
            "conventional_cycle": 5,
        }
        expected_reads = {
            "cycle": 3,
            "raw": 1,
            "wrong_triangle": 3,
            "conventional_cycle": 3,
        }
        for count in FULL_PREFIXES:
            key = str(count)
            for mode_name, per_cycle in expected_ops.items():
                measured = self.report["modes"][mode_name][
                    "measured"]["prefixes"][key]
                self.assertEqual(measured["readout_operations_per_pass"],
                                 per_cycle * count)
                self.assertEqual(measured["readout_operations"],
                                 per_cycle * count * 4)
                self.assertEqual(measured["working_set_bytes_per_pass"],
                                 expected_reads[mode_name] * 4 * count)
                self.assertGreater(measured["readout_operations_per_sec"], 0)
                self.assertGreater(measured["readout_seconds"], 0)
        self.assertEqual(self.report["config"]["backend_storage_bytes"],
                         4 * 32640)
        self.assertGreater(
            self.report["measurement"]["snapshot_read_operations"], 0)
        self.assertGreater(
            self.report["measurement"]["snapshot_read_operations_per_sec"], 0)
        self.assertGreater(
            self.report["totals"]["rates"]["readout_operations_per_sec"], 0)

    def test_identity_metrics_are_integers_not_floats(self):
        keys = (
            "cycles",
            "clean_exact_vs_truth",
            "gauge_linear_exact_invariance",
            "gauge_scrambled_exact_invariance",
            "uniform_exact_invariance",
            "noise_exact_invariance",
            "readout_operations_per_pass",
            "working_set_bytes_per_pass",
        )
        for mode in self.report["modes"].values():
            for measured in mode["measured"]["prefixes"].values():
                for key in keys:
                    if key in measured:
                        value = measured[key]
                        self.assertIsInstance(value, int)
                        self.assertNotIsInstance(value, bool)
        self.assertFalse(self.report["scope"]["float_used_for_identity"])
        self.assertTrue(bench._identity_metrics_are_int(self.report))

    def test_no_canonical_merge_and_no_false_claims(self):
        self.assertFalse(self.report["scope"]["canonical_merge_performed"])
        self.assertFalse(
            self.report["merge_contract"]["canonical_merge_performed"])
        self.assertFalse(
            self.report["merge_contract"]["canonical_merge_api_called"])
        self.assertFalse(self.report["scope"]["native_code_used"])
        self.assertFalse(self.report["scope"]["phone_performance_claimed"])
        self.assertFalse(self.report["scope"]["handset_measurement"])
        self.assertFalse(self.report["scope"]["uniqueness_claim"])
        for key in ("no_canonical_merge", "no_native_acceleration",
                    "no_phone_performance_claim", "no_uniqueness_claim"):
            self.assertTrue(self.report["verdict"]["checks"][key])
        markdown = bench.render_markdown(self.report)
        self.assertIn("| canonical merge performed | no |", markdown)
        self.assertNotIn("| canonical merge performed | yes |", markdown)
        self.assertIn("native-code benchmark", markdown)
        self.assertIn("phone/handset performance measurement or claim", markdown)
        self.assertIn("not** a uniqueness claim", markdown)

    def test_verdict_and_mode_outcomes_are_reported(self):
        self.assertTrue(self.report["status"], "ok")
        self.assertTrue(self.report["verdict"]["passed"])
        self.assertTrue(self.report["verdict"]["checks"][
            "all_modes_expected_outcome"])
        for mode_name in bench.MODES:
            mode = self.report["modes"][mode_name]
            self.assertIn("measured", mode)
            self.assertIn("target", mode)
            self.assertIn("checks", mode)
            self.assertTrue(mode["passed"])
            self.assertTrue(mode["checks"])
        markdown = bench.render_markdown(self.report)
        self.assertIn("## Measured vs target", markdown)
        self.assertIn("## Operations, rates, and bytes", markdown)
        self.assertIn("## Boundaries", markdown)

    def test_report_is_deterministic_for_fixed_seed(self):
        first = fresh_report()
        second = fresh_report()
        self.assertEqual(first["deterministic_fingerprint"],
                         second["deterministic_fingerprint"])
        self.assertEqual(without_timing(first), without_timing(second))
        self.assertEqual(
            json.dumps(without_timing(first), sort_keys=True),
            json.dumps(without_timing(second), sort_keys=True))
        self.assertEqual(bench.deterministic_fingerprint(first),
                         first["deterministic_fingerprint"])

    def test_different_seed_changes_deterministic_measurements(self):
        first = fresh_report(prefixes=(48,), seed=1)
        second = fresh_report(prefixes=(48,), seed=2)
        self.assertNotEqual(first["deterministic_fingerprint"],
                            second["deterministic_fingerprint"])
        self.assertNotEqual(
            first["modes"]["cycle"]["measured"]["prefixes"]["48"][
                "clean_digest"],
            second["modes"]["cycle"]["measured"]["prefixes"]["48"][
                "clean_digest"])


class FilesAndCliTests(unittest.TestCase):
    def test_json_and_markdown_files_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "cycle.json"
            markdown_path = root / "cycle.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bench.main([
                    "--cycles", "48,256",
                    "--steps", "1",
                    "--seed", "99",
                    "--backend", "fake",
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
                "# NaN lane/cycle readout benchmark"))
            self.assertIn("## Measured vs target", markdown)
            self.assertIn("## Operations, rates, and bytes", markdown)
            self.assertIn("## Boundaries", markdown)
            self.assertIn("`cycle`", markdown)
            self.assertIn("`raw`", markdown)
            self.assertIn("`wrong_triangle`", markdown)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["schema"], bench.SCHEMA)
            self.assertTrue(summary["verdict_passed"])
            self.assertEqual(summary["deterministic_fingerprint"],
                             data["deterministic_fingerprint"])

    def test_cli_rejects_invalid_cycles_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "bad.json"
            markdown_path = root / "bad.md"
            for cycles in ("0", "32386", "48,48", "256,48", "48,oops"):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = bench.main([
                        "--cycles", cycles,
                        "--backend", "fake",
                        "--json", str(json_path),
                        "--markdown", str(markdown_path),
                    ])
                self.assertEqual(code, 2)
                self.assertFalse(json_path.exists())
                self.assertFalse(markdown_path.exists())
                self.assertIn("nan_cycle_benchmark: error:", stderr.getvalue())

    def test_cli_rejects_invalid_steps_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "bad.json"
            for steps in ("0", "-1", "1000001"):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = bench.main([
                        "--cycles", "48",
                        "--steps", steps,
                        "--backend", "fake",
                        "--json", str(json_path),
                    ])
                self.assertEqual(code, 2)
                self.assertFalse(json_path.exists())
                self.assertIn("steps must be in [1, 1000000]",
                              stderr.getvalue())


class FallbackAndResidentTests(unittest.TestCase):
    def test_missing_resident_cells_fall_back_to_fake(self):
        with mock.patch.object(bench, "resident_cells", None):
            self.assertFalse(bench.resident_available())
            report = bench.run_benchmark(
                prefixes=(48,), seed=bench.DEFAULT_SEED, backend="auto")
        self.assertEqual(report["config"]["backend"], "fake")
        self.assertFalse(report["scope"]["resident_cells_available"])
        self.assertEqual(report["scope"]["raw_lane_source"],
                         "deterministic integer fake lane table")
        self.assertTrue(report["verdict"]["passed"])

    def test_forced_resident_backend_unavailable_raises(self):
        with mock.patch.object(bench, "resident_cells", None):
            with self.assertRaises(bench.BenchmarkUnavailable):
                bench.run_benchmark(prefixes=(48,), backend="resident")

    @unittest.skipUnless(bench.resident_available(),
                         "nanlmind_hermes.resident.cells is unavailable")
    def test_real_resident_lanes_match_fake_readouts(self):
        resident = bench.run_benchmark(
            prefixes=SMALL_PREFIXES, seed=bench.DEFAULT_SEED,
            backend="resident")
        fake = cached_report(prefixes=FULL_PREFIXES)
        self.assertEqual(resident["config"]["backend"], "resident")
        self.assertEqual(resident["config"]["backend_storage_bytes"],
                         2 * 1024 * 1024)
        self.assertEqual(resident["config"]["lane_storage_bytes"],
                         32_640 * 64)
        self.assertEqual(resident["config"]["lane_value_bytes"], 4)
        self.assertEqual(resident["config"]["lane_cell_bytes"], 64)
        for field, meta in resident["measurement"]["fields"].items():
            self.assertEqual(meta["snapshot_mismatched_lanes"], 0,
                             field)
            self.assertEqual(meta["snapshot_exact_lanes"],
                             resident["config"]["lane_count"], field)
        for count in SMALL_PREFIXES:
            key = str(count)
            self.assertEqual(
                without_timing(resident["modes"]["cycle"]["measured"][
                    "prefixes"][key]),
                without_timing(fake["modes"]["cycle"]["measured"][
                    "prefixes"][key]))
            self.assertEqual(
                without_timing(resident["modes"]["raw"]["measured"][
                    "prefixes"][key]),
                without_timing(fake["modes"]["raw"]["measured"][
                    "prefixes"][key]))
        self.assertTrue(resident["verdict"]["passed"])


if __name__ == "__main__":
    unittest.main()
