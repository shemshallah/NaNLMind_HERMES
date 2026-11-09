"""Tests for the Round 030 native-audit verification script.

The tests exercise only ``scripts/round030_native_audit.py`` and the supplied
``experiments/round030`` artifacts.  They do not import WordNet, torch, numpy
or scikit-learn.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "round030_native_audit.py"
EXPERIMENT = ROOT / "experiments" / "round030"

_spec = importlib.util.spec_from_file_location(
    "round030_native_audit_under_test", SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/round030_native_audit.py")
rna = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rna
_spec.loader.exec_module(rna)


def _fake_rerun_measurement() -> dict:
    return {
        "root_bytes": 8_388_608,
        "native_gradient_max_abs_error": 1.862645149230957e-09,
        "gauge_weight_max_abs_error": 7.854541763663292e-07,
        "mirror_antisymmetry": True,
        "snapshot_bitwise_restore": True,
        "predictions": 1000,
        "ctypes_wall_ms": 8.25,
        "memory_wordnet_predictor_is_native": True,
        "text_encoder_is_external": True,
    }


def _fake_rerun() -> dict:
    return {
        "label": "re-run",
        "status": "ok",
        "reason": None,
        "re_executed": True,
        "measurement": _fake_rerun_measurement(),
        "compiler": "/usr/bin/c++",
        "compile": {
            "command": ["c++", "-std=c++20"],
            "returncode": 0,
            "stderr": "",
        },
        "elapsed_seconds": 0.25,
        "snapshot_removed": True,
    }


@contextlib.contextmanager
def _quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


class SuppliedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = rna.run_audit(source_only=True)

    def test_supplied_artifacts_verify_and_parse(self):
        report = self.report
        self.assertEqual(report["schema"], rna.SCHEMA)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["exit_code"], rna.EXIT_OK)
        self.assertEqual(report["mode"], "source-only")
        self.assertEqual(
            report["evidence_label"],
            "measured artifact only (source-only)",
        )
        self.assertTrue(report["checks"]["all_targets_met"])
        self.assertGreaterEqual(len(report["check_details"]), 30)

        verification = report["verification"]
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(verification["entry_count"], 13)
        self.assertEqual(verification["verified_count"], 13)
        self.assertTrue(verification["required_files_covered"])

        paths = {entry["path"] for entry in verification["verified"]}
        for name in rna.REQUIRED_MANIFEST_FILES:
            self.assertIn(name, paths)
        for seed in rna.EXPECTED_SEEDS:
            self.assertIn(f"root_{seed}.bin", paths)

    def test_parsed_artifacts_have_expected_shapes(self):
        artifacts = self.report["artifacts"]
        native = artifacts["native_audit"]["data"]
        tuned = artifacts["tuned_results"]["data"]
        run_log = artifacts["run_log"]["data"]

        self.assertEqual(native["root_bytes"], 8_388_608)
        self.assertEqual(native["predictions"], 1000)
        self.assertEqual(
            sorted(result["seed"] for result in tuned["results"]),
            [17, 29, 41],
        )
        self.assertEqual(len(run_log["runs"]), 3)
        for run in run_log["runs"]:
            self.assertEqual(run["samples"]["train"], 2400)
            self.assertEqual(run["samples"]["val"], 560)
            self.assertEqual(run["samples"]["test"], 560)
            self.assertEqual(run["root_bytes"], 8_388_608)
        self.assertEqual(len(artifacts["snapshots"]), 3)

    def test_mean_accuracy_numbers_are_exact(self):
        tuned = self.report["observed"]["tuned"]
        expected = {
            "cycle_test": 0.42083333333333334,
            "shift_test": 0.42083333333333334,
            "raw_shift_test": 0.08214285714285714,
        }
        for key, value in expected.items():
            self.assertEqual(tuned["mean_accuracy"][key], value)
            self.assertAlmostEqual(
                tuned["mean_accuracy"][key],
                value,
                places=12,
            )
        self.assertAlmostEqual(
            tuned["mean_accuracy"]["cycle_test"],
            rna.MEAN_TUNED_ACCURACY,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            tuned["mean_accuracy"]["raw_shift_test"],
            rna.MEAN_RAW_SHIFT_ACCURACY,
            delta=1e-6,
        )
        for key in expected:
            self.assertEqual(
                tuned["mean_accuracy"][key],
                tuned["computed_mean_accuracy"][key],
            )

    def test_gradient_and_gauge_errors_are_exact_with_tolerance(self):
        observed = self.report["observed"]
        self.assertEqual(
            observed["native_gradient_max_abs_error"],
            1.862645149230957e-09,
        )
        self.assertLessEqual(
            abs(
                observed["native_gradient_max_abs_error"]
                - rna.GRADIENT_MAX_ABS_ERROR_EXACT
            ),
            rna.GRADIENT_MATCH_TOLERANCE,
        )
        self.assertAlmostEqual(
            observed["native_gradient_max_abs_error"],
            rna.GRADIENT_MAX_ABS_ERROR,
            delta=rna.GRADIENT_MATCH_TOLERANCE,
        )
        self.assertEqual(
            observed["gauge_weight_max_abs_error"],
            4.1653402149677277e-07,
        )
        self.assertLessEqual(
            abs(
                observed["gauge_weight_max_abs_error"]
                - rna.GAUGE_WEIGHT_MAX_ABS_ERROR_EXACT
            ),
            rna.GAUGE_MATCH_TOLERANCE,
        )
        self.assertAlmostEqual(
            observed["gauge_weight_max_abs_error"],
            rna.GAUGE_WEIGHT_MAX_ABS_ERROR,
            delta=rna.GAUGE_MATCH_TOLERANCE,
        )
        self.assertTrue(
            self.report["checks"]["native_audit.gradient_bound"])
        self.assertTrue(
            self.report["checks"]["native_audit.gauge_weight_bound"])

    def test_snapshot_restore_and_8mib_root(self):
        observed = self.report["observed"]
        self.assertEqual(observed["root_bytes"], 8_388_608)
        self.assertIs(observed["snapshot_bitwise_restore"], True)
        self.assertIs(observed["mirror_antisymmetry"], True)
        self.assertEqual(
            observed["run_log"]["restore_max_logit_error"],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            observed["run_log"]["root_bytes"],
            [8_388_608, 8_388_608, 8_388_608],
        )

        snapshots = observed["snapshots"]
        self.assertEqual(
            [item["relative_path"] for item in snapshots],
            ["root_17.bin", "root_29.bin", "root_41.bin"],
        )
        for item in snapshots:
            self.assertEqual(item["size"], 8_388_616)
            self.assertEqual(item["magic"], "NANCYC30")
            self.assertTrue(item["magic_ok"])
            self.assertIsNotNone(item["sha256"])
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(self.report["checks"]["snapshots.size_and_magic"])

    def test_conventional_clean_classifier_ge_cycle(self):
        run_log = self.report["observed"]["run_log"]
        for conventional, cycle in zip(
            run_log["conventional_accuracy"],
            run_log["cycle_before_accuracy"],
        ):
            self.assertGreaterEqual(conventional, cycle)
        self.assertGreaterEqual(
            run_log["mean_conventional_accuracy"],
            run_log["mean_cycle_accuracy"],
        )
        self.assertGreaterEqual(
            run_log["mean_conventional_accuracy"],
            self.report["observed"]["tuned"]["mean_accuracy"]["cycle_test"],
        )
        self.assertTrue(
            self.report["checks"][
                "run_log.conventional_clean_classifier_ge_cycle"
            ]
        )
        self.assertTrue(
            self.report["checks"][
                "run_log.conventional_mean_ge_tuned_cycle_mean"
            ]
        )

    def test_no_canonical_language_or_nan_unique_claim(self):
        claims = self.report["claims"]
        for key in (
            "canonical",
            "canonical_implementation",
            "language_model_win",
            "lm_win",
            "language_win",
            "nan_unique_advantage",
            "nan_unique_benefit",
            "semantic_superiority",
            "quantum_claim",
            "android_benchmark",
            "portable_throughput",
            "native_benchmark",
            "native_execution",
        ):
            self.assertFalse(claims[key], key)

        text = json.dumps(self.report).lower()
        for forbidden in (
            '"canonical": true',
            '"language_model_win": true',
            '"lm_win": true',
            '"language_win": true',
            '"nan_unique_advantage": true',
            '"nan_unique_benefit": true',
            '"semantic_superiority": true',
        ):
            self.assertNotIn(forbidden, text)

        markdown = rna.render_markdown(self.report)
        self.assertIn("measured artifact", markdown)
        self.assertIn("not a generative language model", markdown)
        self.assertIn("No language-model or LM win is claimed", markdown)
        self.assertIn("NaN-unique benefit", markdown)
        self.assertIn("conventional clean 16x24 linear classifier", markdown)
        self.assertIn(">=", markdown)
        self.assertIn("No canonical implementation", markdown)


class CliTests(unittest.TestCase):
    def test_json_and_markdown_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_markdown = Path(tmp) / "report.md"
            with _quiet() as stdout:
                code = rna.main([
                    "--source-only",
                    "--json", str(out_json),
                    "--markdown", str(out_markdown),
                ])
            self.assertEqual(code, rna.EXIT_OK)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["mode"], "source-only")
            self.assertEqual(receipt["exit_code"], rna.EXIT_OK)
            self.assertTrue(receipt["checks_all_targets_met"])

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], rna.SCHEMA)
            for key in (
                "status",
                "exit_code",
                "mode",
                "evidence_label",
                "evidence",
                "verification",
                "targets",
                "measurements",
                "observed",
                "artifacts",
                "checks",
                "check_details",
                "claims",
                "boundaries",
                "notes",
            ):
                self.assertIn(key, report)

            measurements = report["measurements"]
            self.assertEqual(
                measurements["mean_tuned_accuracy"], 0.42083333333333334)
            self.assertEqual(
                measurements["mean_raw_shift_accuracy"],
                0.08214285714285714,
            )
            self.assertEqual(
                measurements["native_gradient_max_abs_error"],
                1.862645149230957e-09,
            )
            self.assertEqual(
                measurements["gauge_weight_max_abs_error"],
                4.1653402149677277e-07,
            )
            self.assertTrue(
                measurements["conventional_clean_classifier_ge_cycle"])

            markdown = out_markdown.read_text(encoding="utf-8")
            for heading in (
                "# Round 030 native audit verification",
                "## Digest verification",
                "## Supplied measured artifact",
                "## Checks",
                "## Local re-run",
                "## Boundaries",
            ):
                self.assertIn(heading, markdown)
            self.assertIn("0.420833", markdown)
            self.assertIn("1.86e-9", markdown)
            self.assertIn("4.17e-7", markdown)

    def test_checksum_mismatch_denies_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "round030"
            experiment.mkdir()
            artifact = experiment / "native_audit.json"
            artifact.write_text('{"root_bytes": 8388608}\n', encoding="utf-8")
            wrong = hashlib.sha256(b"not the supplied artifact").hexdigest()
            manifest = experiment / "SHA256SUMS"
            manifest.write_text(
                f"{wrong}  native_audit.json\n", encoding="utf-8")
            out_json = Path(tmp) / "report.json"
            out_markdown = Path(tmp) / "report.md"

            with mock.patch.object(
                rna,
                "load_artifacts",
                side_effect=AssertionError("must not parse after denial"),
            ):
                with _quiet():
                    code = rna.main([
                        "--experiment-dir", str(experiment),
                        "--sha256sums", str(manifest),
                        "--source-only",
                        "--json", str(out_json),
                        "--markdown", str(out_markdown),
                    ])

            self.assertEqual(code, rna.EXIT_DENIED)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "denied")
            self.assertEqual(report["exit_code"], rna.EXIT_DENIED)
            self.assertEqual(report["verification"]["status"], "denied")
            self.assertIn(
                "mismatch", report["verification"]["reason"].lower())
            self.assertEqual(
                report["evidence"]["rerun"]["status"], "not_requested")
            self.assertFalse(report["claims"]["native_execution"])
            self.assertFalse(report["checks"]["all_targets_met"])
            self.assertIsNone(report["observed"])
            self.assertIsNone(report["artifacts"])
            self.assertTrue(out_markdown.is_file())
            self.assertIn("denied", out_markdown.read_text(
                encoding="utf-8").lower())

    def test_missing_optional_dependencies_yield_structured_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_markdown = Path(tmp) / "report.md"
            with mock.patch.object(
                rna,
                "missing_rerun_dependencies",
                return_value=["sklearn"],
            ), mock.patch.object(
                rna, "find_compiler", return_value="/usr/bin/c++"
            ), mock.patch.object(
                rna,
                "run_native_audit",
                side_effect=AssertionError("must not run with missing deps"),
            ):
                with _quiet():
                    code = rna.main([
                        "--json", str(out_json),
                        "--markdown", str(out_markdown),
                    ])

            self.assertEqual(code, rna.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["mode"], "artifact-only")
            self.assertEqual(
                report["evidence_label"],
                "measured artifact only (no local re-run)",
            )
            rerun = report["evidence"]["rerun"]
            self.assertEqual(rerun["status"], "skipped")
            self.assertEqual(rerun["label"], "re-run")
            self.assertEqual(rerun["missing_dependencies"], ["sklearn"])
            self.assertIn("sklearn", rerun["reason"])
            self.assertFalse(rerun["re_executed"])
            self.assertIsNone(rerun["measurement"])
            self.assertIn(
                "scikit-learn", rerun["dependencies"]["required"])
            self.assertFalse(rerun["dependencies"]["wordnet_required"])
            self.assertFalse(rerun["dependencies"]["torch_required"])
            self.assertFalse(report["claims"]["native_execution"])
            self.assertTrue(report["checks"]["all_targets_met"])
            self.assertIn(
                "sklearn", out_markdown.read_text(encoding="utf-8"))

    def test_source_only_does_not_probe_or_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            with mock.patch.object(
                rna,
                "missing_rerun_dependencies",
                side_effect=AssertionError("must not probe dependencies"),
            ), mock.patch.object(
                rna,
                "find_compiler",
                side_effect=AssertionError("must not probe compiler"),
            ), mock.patch.object(
                rna,
                "run_native_audit",
                side_effect=AssertionError("must not execute"),
            ):
                with _quiet():
                    code = rna.main([
                        "--source-only",
                        "--json", str(out_json),
                    ])

            self.assertEqual(code, rna.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "source-only")
            rerun = report["evidence"]["rerun"]
            self.assertEqual(rerun["status"], "not_requested")
            self.assertIn("source-only", rerun["reason"].lower())
            self.assertFalse(report["claims"]["native_execution"])

    def test_successful_rerun_is_labeled_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_markdown = Path(tmp) / "report.md"
            with mock.patch.object(
                rna, "missing_rerun_dependencies", return_value=[]
            ), mock.patch.object(
                rna, "find_compiler", return_value="/usr/bin/c++"
            ), mock.patch.object(
                rna, "run_native_audit", return_value=_fake_rerun()
            ):
                with _quiet():
                    code = rna.main([
                        "--json", str(out_json),
                        "--markdown", str(out_markdown),
                    ])

            self.assertEqual(code, rna.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["mode"], "re-run")
            self.assertEqual(
                report["evidence_label"],
                "measured artifact plus local re-run",
            )
            rerun = report["evidence"]["rerun"]
            self.assertEqual(rerun["status"], "ok")
            self.assertEqual(rerun["label"], "re-run")
            self.assertTrue(rerun["re_executed"])
            self.assertEqual(
                rerun["measurement"]["native_gradient_max_abs_error"],
                1.862645149230957e-09,
            )
            self.assertTrue(
                report["checks"]["rerun.structural_matches_artifact"])
            self.assertTrue(
                report["checks"]["rerun.gradient_max_abs_error_bound"])
            self.assertEqual(
                report["check_details"][-1]["source"],
                rna.SOURCE_RERUN,
            )
            self.assertTrue(report["claims"]["native_execution"])
            for key in (
                "canonical",
                "language_model_win",
                "lm_win",
                "nan_unique_advantage",
                "nan_unique_benefit",
                "semantic_superiority",
            ):
                self.assertFalse(report["claims"][key], key)

            markdown = out_markdown.read_text(encoding="utf-8")
            self.assertIn("**re-run**", markdown)
            self.assertIn("measured artifact plus local re-run", markdown)


class NegativeArtifactTests(unittest.TestCase):
    def _write_minimal_bundle(self, directory: Path, *, mutate=None) -> Path:
        for name in ("native_audit.json", "tuned_results.json", "run.log"):
            shutil.copy2(EXPERIMENT / name, directory / name)
        if mutate is not None:
            mutate(directory)
        lines = []
        for name in ("native_audit.json", "tuned_results.json", "run.log"):
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        manifest = directory / "SHA256SUMS"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    def test_number_drift_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "round030"
            experiment.mkdir()

            def mutate(directory: Path) -> None:
                path = directory / "tuned_results.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["mean_accuracy"]["cycle_test"] = 0.42
                path.write_text(json.dumps(data), encoding="utf-8")

            manifest = self._write_minimal_bundle(experiment, mutate=mutate)
            report = rna.run_audit(
                experiment,
                source_only=True,
                manifest=manifest,
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["exit_code"], rna.EXIT_FAILED)
            self.assertFalse(
                report["checks"]["tuned.mean_cycle_accuracy"])
            self.assertFalse(report["checks"]["all_targets_met"])

    def test_invalid_json_reports_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "round030"
            experiment.mkdir()

            def mutate(directory: Path) -> None:
                (directory / "native_audit.json").write_text(
                    "{not valid json\n", encoding="utf-8")

            manifest = self._write_minimal_bundle(experiment, mutate=mutate)
            report = rna.run_audit(
                experiment,
                source_only=True,
                manifest=manifest,
            )
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["exit_code"], rna.EXIT_ERROR)
            self.assertFalse(report["checks"]["artifact.parsed"])
            self.assertIsNone(report["observed"])
            self.assertIn("invalid json", report["notes"][0].lower())

    def test_manifest_path_traversal_is_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "round030"
            experiment.mkdir()
            outside = Path(tmp) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            manifest = experiment / "SHA256SUMS"
            manifest.write_text(
                f"{digest}  ../{outside.name}\n", encoding="utf-8")

            with self.assertRaises(rna.VerificationError) as caught:
                rna.verify_artifacts(experiment, manifest)
            self.assertIn("traversal", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
