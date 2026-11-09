"""Tests for the Round 031 full-cycle audit script.

The tests exercise the supplied artifact contract, checksum denial, the
256-versus-32,385 cycle-count identity, the deterministic synthetic full-basis
audit and the structured optional skip.  They do not train a language model and
make no language-win, canonical or NaN-unique claim.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "round031_full_cycle_audit.py"
EXPERIMENT = ROOT / "experiments" / "round031" / "nan_round031"

_spec = importlib.util.spec_from_file_location(
    "round031_full_cycle_audit_under_test", SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/round031_full_cycle_audit.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rc
_spec.loader.exec_module(rc)


def require_artifacts() -> None:
    missing = [
        str(path)
        for path in (
            EXPERIMENT / "SHA256SUMS",
            EXPERIMENT / "results.json",
            EXPERIMENT / "audit.json",
            EXPERIMENT / "transfer_results.json",
        )
        if not path.is_file()
    ]
    if missing:
        raise unittest.SkipTest(
            "Round 031 artifacts are absent: " + ", ".join(missing))


class Round031FullCycleAuditTests(unittest.TestCase):
    def test_files_exist_and_script_declares_boundaries(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(Path(__file__).is_file())
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("32,385", text)
        self.assertIn("256", text)
        self.assertIn("No language-model win", text)
        self.assertIn("structured skip", text.lower())

    def test_cycle_counts_are_256_and_32385(self):
        invariants = rc.cycle_count_invariants()
        self.assertEqual(rc.EFFECTIVE_CYCLES, 256)
        self.assertEqual(rc.EFFECTIVE_CYCLES, rc.D * rc.K)
        self.assertEqual(rc.EFFECTIVE_CYCLES, 16 * 16)
        self.assertEqual(rc.FULL_BASIS_CYCLES, 32_385)
        self.assertEqual(rc.FULL_BASIS_CYCLES, math.comb(255, 2))
        self.assertEqual(
            rc.FULL_BASIS_CYCLES, rc.PAIRS - (rc.HEADS - 1))
        self.assertEqual(rc.PAIRS, math.comb(256, 2))
        self.assertEqual(rc.UNUSED_CYCLES, 32_129)
        self.assertEqual(invariants["effective_cycles"], 256)
        self.assertEqual(invariants["full_basis_cycles"], 32_385)
        self.assertEqual(invariants["unused_cycles"], 32_129)
        self.assertEqual(invariants["gauge_dimension"], 255)

        pairs = rc.full_basis_pairs()
        self.assertEqual(len(pairs), 32_385)
        self.assertEqual(pairs[0], (1, 2))
        self.assertEqual(pairs[-1], (254, 255))
        self.assertEqual(len(set(pairs)), 32_385)
        self.assertTrue(all(1 <= first < second < 256
                            for first, second in pairs))

        targets = rc.deterministic_full_basis_targets(seed=2031)
        self.assertEqual(len(targets), 32_385)
        self.assertEqual(
            targets, rc.deterministic_full_basis_targets(seed=2031))
        self.assertNotEqual(
            targets, rc.deterministic_full_basis_targets(seed=2032))

    def test_artifact_parsing_and_measured_invariants(self):
        require_artifacts()
        verification = rc.verify_artifacts(EXPERIMENT)
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(verification["entry_count"], 23)
        self.assertEqual(verification["verified_count"], 23)
        self.assertTrue(verification["required_files_covered"])
        self.assertIn("results.json", verification["digests"])
        self.assertIn("native_cycle.cpp", verification["digests"])

        artifacts = rc.load_artifacts(EXPERIMENT, verification)
        self.assertEqual(
            artifacts["results"]["data"]["counts"],
            {"train": 1800, "val": 1800, "test": 1800},
        )
        self.assertEqual(
            set(artifacts["results"]["data"]["runs"]), set(rc.EXPECTED_RUNS))
        self.assertEqual(
            artifacts["audit"]["data"]["wordnet"]["synsets"], 117_570)
        self.assertEqual(
            artifacts["audit"]["data"]["wordnet"]["two_hop_candidates"],
            87_655,
        )
        self.assertEqual(
            set(artifacts["transfer"]["data"]), set(rc.EXPECTED_RUNS))
        self.assertEqual(len(artifacts["snapshots"]), 4)
        for snapshot in artifacts["snapshots"]:
            self.assertEqual(snapshot["size"], rc.SNAPSHOT_BYTES)
            self.assertEqual(snapshot["root_bytes"], rc.ROOT_BYTES)
            self.assertTrue(snapshot["magic_ok"])
            self.assertTrue(snapshot["sha256"])

        constants = artifacts["native_source"]["constants"]
        self.assertEqual(constants["heads"], 256)
        self.assertEqual(constants["dimension"], 16)
        self.assertEqual(constants["channels"], 16)
        self.assertEqual(constants["pairs"], 32_640)
        self.assertEqual(constants["root_bytes"], 8_388_608)
        self.assertEqual(constants["effective_cycles"], 256)
        self.assertEqual(constants["full_basis_cycles"], 32_385)
        self.assertTrue(constants["has_size_static_assert"])
        self.assertTrue(constants["has_flux_formula"])
        self.assertTrue(all(constants["exports"].values()))

        train_constants = artifacts["train_source"]["constants"]
        self.assertEqual(train_constants["readout_dimension"], 256)
        self.assertTrue(train_constants["has_16x16_pair_parameter"])
        self.assertEqual(train_constants["native_learning_rate"], 0.003)
        self.assertEqual(train_constants["neural_learning_rate"], 0.002)
        self.assertEqual(train_constants["neural_optimizer"], "AdamW")

        self.assertEqual(
            artifacts["readme"]["table"][17],
            {"plain_bits_per_byte": 3.89467,
             "native_bits_per_byte": 4.00756},
        )
        self.assertTrue(all(artifacts["readme"]["limits"].values()))

        details, observed = rc.evaluate_artifacts(artifacts, verification)
        self.assertTrue(all(item["passed"] or item["skipped"]
                            for item in details))

        wordnet = observed["wordnet_test"]["per_seed"]
        self.assertAlmostEqual(
            wordnet[17]["native_minus_plain_bits_per_byte"],
            0.1128893715843664,
            places=12,
        )
        self.assertAlmostEqual(
            wordnet[29]["native_minus_plain_bits_per_byte"],
            0.06467401165010322,
            places=12,
        )
        self.assertAlmostEqual(
            observed["wordnet_test"][
                "mean_native_minus_plain_bits_per_byte"],
            0.08878169161723481,
            places=12,
        )
        self.assertTrue(
            observed["wordnet_test"]["conventional_16x16_better_all_seeds"])
        self.assertGreater(
            wordnet[17]["native_minus_plain_bits_per_byte"], 0.0)
        self.assertGreater(
            wordnet[29]["native_minus_plain_bits_per_byte"], 0.0)

        self.assertEqual(observed["effective_cycles"], 256)
        self.assertEqual(observed["full_basis_cycles"], 32_385)
        self.assertEqual(observed["unused_cycles"], 32_129)

        for seed in rc.EXPECTED_SEEDS:
            gauge = observed["gauge"][seed]
            self.assertLessEqual(gauge["max_weight_diff"],
                                 rc.GAUGE_WEIGHT_BOUND)
            self.assertEqual(gauge["accuracy_abs_delta"], 0.0)
            self.assertTrue(gauge["audit"])
            ablation = observed["memory_ablation"]["per_seed"][seed]
            self.assertGreater(
                ablation["bits_per_byte_increase_without_memory"], 0.0)
            self.assertGreater(
                ablation["accuracy_change_without_memory"], 0.0)
            self.assertAlmostEqual(
                observed["params"][seed]["native_minus_plain"], 256.0)

        self.assertTrue(
            observed["memory_ablation"]["loss_worse_without_memory"])
        self.assertFalse(
            observed["memory_ablation"]["accuracy_benefit_all_seeds"])
        self.assertEqual(observed["snapshot"]["audit_snapshot_max_diff"], 0.0)
        self.assertEqual(observed["two_hop"]["synsets"], 117_570)
        self.assertEqual(observed["two_hop"]["two_hop_candidates"], 87_655)
        self.assertEqual(observed["two_hop"]["two_hop_proofs_checked"], 1_500)
        self.assertTrue(
            observed["two_hop"]["all_chain_edges_verified"])
        self.assertEqual(observed["two_hop"]["direct_edge_only_hits"], 0)
        for key in rc.EXPECTED_RUNS:
            row = observed["transfer"][key]
            self.assertGreater(
                row["wordnet_bits_per_byte_improvement"], 0.0)
            self.assertGreater(
                row["alice_validation_bits_per_byte_improvement"], 0.0)
            self.assertEqual(row["alice_test"]["n_bytes"], 288)
        for seed in rc.EXPECTED_SEEDS:
            self.assertLess(
                observed["transfer"][f"plain_{seed}"]["alice_test"][
                    "bits_per_byte"],
                observed["transfer"][f"native_{seed}"]["alice_test"][
                    "bits_per_byte"],
            )

    def test_checksum_mismatch_denies_without_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "round031"
            base.mkdir()
            results = base / "results.json"
            results.write_text("{}\n", encoding="utf-8")
            wrong = hashlib.sha256(b"not the supplied artifact").hexdigest()
            sums = base / "SHA256SUMS"
            sums.write_text(f"{wrong}  results.json\n", encoding="utf-8")

            with self.assertRaises(rc.VerificationError) as caught:
                rc.verify_artifacts(base, sums)
            self.assertIn("mismatch", str(caught.exception).lower())

            with mock.patch.object(
                rc,
                "load_artifacts",
                side_effect=AssertionError("must not parse after denial"),
            ):
                report = rc.run_audit(
                    base, manifest=sums, full_cycle_memory=None)

            self.assertEqual(report["status"], "denied")
            self.assertEqual(report["exit_code"], rc.EXIT_DENIED)
            self.assertEqual(report["verification"]["status"], "denied")
            self.assertFalse(report["checks"]["manifest.verified"])
            self.assertIsNone(report["observed"])
            self.assertIsNone(report["artifacts"])
            self.assertEqual(
                report["full_basis_synthetic_audit"]["status"], "skipped")
            self.assertFalse(report["claims"]["canonical"])

            outside = Path(tmp) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            sums.write_text(f"{digest}  ../outside.json\n", encoding="utf-8")
            with self.assertRaises(rc.VerificationError) as caught:
                rc.verify_artifacts(base, sums)
            self.assertIn("traversal", str(caught.exception).lower())

    def test_synthetic_full_basis_audit_uses_every_coefficient(self):
        audit = rc.run_full_basis_synthetic_audit(seed=2031)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["coefficient_count"], 32_385)
        self.assertEqual(audit["expected_coefficient_count"], 32_385)
        self.assertEqual(audit["pair_edge_count"], 32_640)
        self.assertEqual(audit["used_count"], 32_385)
        self.assertTrue(audit["all_coefficients_used"])
        self.assertEqual(audit["min_visit_count"], 1)
        self.assertEqual(audit["max_visit_count"], 1)
        self.assertEqual(audit["training"]["updates"], 32_385)
        self.assertEqual(audit["training"]["epochs"], 1)
        self.assertLessEqual(audit["maximum_abs_error"],
                             rc.SYNTHETIC_ERROR_BOUND)
        self.assertLessEqual(audit["gauge_invariance_max_abs_error"],
                             rc.SYNTHETIC_GAUGE_BOUND)
        self.assertTrue(audit["audit_passed"])
        self.assertEqual(len(audit["coefficients"]), 32_385)
        self.assertEqual(
            rc.coefficient_digest(audit["coefficients"]),
            audit["coefficients_sha256"],
        )
        targets = rc.deterministic_full_basis_targets(seed=2031)
        self.assertEqual(audit["coefficients"], targets)
        self.assertTrue(all(math.isfinite(value)
                            for value in audit["coefficients"]))

        repeated = rc.run_full_basis_synthetic_audit(seed=2031)
        self.assertEqual(audit["coefficients_sha256"],
                         repeated["coefficients_sha256"])
        self.assertEqual(audit["training"]["order_sha256"],
                         repeated["training"]["order_sha256"])
        self.assertEqual(audit["coefficients"], repeated["coefficients"])

    def test_report_with_builtin_full_basis(self):
        require_artifacts()
        report = rc.run_audit(full_cycle_memory="builtin")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["exit_code"], rc.EXIT_OK)
        self.assertEqual(report["mode"], "artifact+synthetic-full-basis")
        self.assertEqual(report["targets"]["effective_cycles"], 256)
        self.assertEqual(report["targets"]["full_basis_cycles"], 32_385)
        self.assertEqual(report["targets"]["unused_cycles"], 32_129)
        full_basis = report["full_basis_synthetic_audit"]
        self.assertEqual(full_basis["status"], "ok")
        self.assertEqual(full_basis["source"], "builtin-synthetic")
        self.assertEqual(len(full_basis["coefficients"]), 32_385)
        self.assertEqual(full_basis["used_count"], 32_385)
        self.assertTrue(full_basis["all_coefficients_used"])
        self.assertTrue(full_basis["audit_passed"])
        self.assertTrue(report["checks"]["full_basis.audit_passed"])
        self.assertTrue(report["checks"]["all_targets_met"])

        serialized = json.dumps(report, allow_nan=False)
        self.assertIn("32385", serialized)
        self.assertNotIn('"canonical": true', serialized.lower())

        markdown = rc.render_markdown(report)
        for heading in (
            "# Round 031 full-cycle audit",
            "## Digest verification",
            "## Effective weights versus the full basis",
            "## Supplied measured artifact",
            "## Full-basis synthetic audit",
            "## Checks",
            "## Claims",
            "## Boundaries",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("32,385", markdown)
        self.assertIn("32,129", markdown)
        self.assertIn("No language-model win", markdown)
        self.assertIn("32385 / 32385", markdown)

    def test_optional_full_basis_skip_is_structured(self):
        require_artifacts()
        report = rc.run_audit(full_cycle_memory=None)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["mode"], "artifact-only")
        self.assertEqual(
            report["full_basis_synthetic_audit"]["status"], "skipped")
        self.assertEqual(
            report["full_basis_synthetic_audit"][
                "expected_coefficient_count"],
            32_385,
        )
        self.assertIsNone(
            report["full_basis_synthetic_audit"]["coefficient_count"])
        self.assertIsNone(
            report["full_basis_synthetic_audit"]["coefficients"])
        self.assertIn(
            "disabled",
            report["full_basis_synthetic_audit"]["reason"].lower(),
        )
        self.assertFalse(
            report["claims"]["full_cycle_memory_available"])
        markdown = rc.render_markdown(report)
        self.assertIn("Structured skip", markdown)

        unavailable = {
            "status": "unavailable",
            "source": None,
            "targets": None,
            "searched": ["module:full_cycle_memory"],
            "reason": "full_cycle_memory provider is not present",
        }
        with mock.patch.object(
            rc, "discover_full_cycle_memory", return_value=unavailable):
            auto_report = rc.run_audit(full_cycle_memory="auto")
        self.assertEqual(auto_report["status"], "ok")
        self.assertEqual(auto_report["mode"], "artifact-only")
        full_basis = auto_report["full_basis_synthetic_audit"]
        self.assertEqual(full_basis["status"], "skipped")
        self.assertIn("not present", full_basis["reason"])
        self.assertEqual(
            full_basis["searched"], ["module:full_cycle_memory"])

    def test_no_canonical_language_or_nan_unique_claim(self):
        require_artifacts()
        report = rc.run_audit(full_cycle_memory=None)
        claims = report["claims"]
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
            "full_cycle_memory_used_in_supplied_model",
        ):
            self.assertFalse(claims[key], key)
        text = json.dumps(report, sort_keys=True, allow_nan=False).lower()
        self.assertNotIn('"canonical": true', text)
        self.assertNotIn('"language_model_win": true', text)
        self.assertNotIn('"nan_unique_advantage": true', text)
        markdown = rc.render_markdown(report)
        self.assertIn("No language-model win", markdown)
        self.assertIn("No canonical implementation", markdown)

    def test_cli_writes_json_and_markdown(self):
        require_artifacts()
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_markdown = Path(tmp) / "report.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = rc.main([
                    "--no-full-cycle-memory",
                    "--json", str(out_json),
                    "--markdown", str(out_markdown),
                ])
            self.assertEqual(code, rc.EXIT_OK)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["effective_cycles"], 256)
            self.assertEqual(receipt["full_basis_cycles"], 32_385)
            self.assertEqual(
                receipt["full_basis_synthetic_audit"], "skipped")

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], rc.SCHEMA)
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["checks"]["all_targets_met"])
            self.assertTrue(out_markdown.is_file())
            markdown = out_markdown.read_text(encoding="utf-8")
            self.assertIn("256", markdown)
            self.assertIn("32,385", markdown)
            self.assertIn("structured skip", markdown.lower())


if __name__ == "__main__":
    unittest.main()
