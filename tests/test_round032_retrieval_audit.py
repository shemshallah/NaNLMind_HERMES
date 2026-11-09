"""Tests for the Round 032 retrieval audit script.

The tests exercise ``scripts/round032_retrieval_audit.py`` against the supplied
``experiments/round032`` artifacts.  They do not import WordNet, torch or any
native extension.
"""

from __future__ import annotations

import contextlib
import copy
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
SCRIPT = ROOT / "scripts" / "round032_retrieval_audit.py"
EXPERIMENT = ROOT / "experiments" / "round032"

_spec = importlib.util.spec_from_file_location(
    "round032_retrieval_audit_under_test", SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/round032_retrieval_audit.py")
r032 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r032
_spec.loader.exec_module(r032)


@contextlib.contextmanager
def _quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


class SuppliedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = r032.run_audit(native=False)

    def test_supplied_artifacts_verify_and_parse(self):
        report = self.report
        self.assertEqual(report["schema"], r032.SCHEMA)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["exit_code"], r032.EXIT_OK)
        self.assertEqual(report["mode"], "artifact-only")
        self.assertTrue(report["checks"]["all_targets_met"])

        verification = report["verification"]
        self.assertEqual(verification["status"], "verified")
        self.assertEqual(verification["entry_count"], 7)
        self.assertEqual(verification["verified_count"], 7)
        self.assertTrue(verification["required_files_covered"])
        self.assertEqual(
            {entry["path"] for entry in verification["verified"]},
            {
                "native_recursive.cpp",
                "prepare.py",
                "long_context.py",
                "data/facts.tsv",
                "data/queries.tsv",
                "results.json",
                "long_context_results.json",
            },
        )

        results = report["artifacts"]["results"]["data"]
        self.assertEqual(results["facts"], 50_268)
        self.assertEqual(results["root_bytes"], 8_388_608)
        self.assertIs(results["pair_map_pass"], True)
        self.assertEqual(results["membership_count"], 50_268)
        self.assertEqual(results["queries"], 250_000)
        self.assertEqual(results["posting_found"], 250_000)
        self.assertEqual(results["hash_found"], 250_000)
        self.assertEqual(results["twohop_queries"], 2_000)
        self.assertEqual(results["twohop_found"], 2_000)
        self.assertEqual(results["direct_found"], 0)

        facts = report["artifacts"]["facts"]
        self.assertEqual(facts["summary"]["rows"], 50_268)
        self.assertEqual(facts["summary"]["sources"], 49_321)
        self.assertEqual(
            facts["summary"]["duplicate_source_relation_rows"], 947)
        self.assertEqual(facts["sha256"], verification["digests"]["data/facts.tsv"])

        queries = report["artifacts"]["queries"]
        self.assertEqual(queries["summary"]["rows"], 2_000)
        self.assertEqual(queries["summary"]["unique_rows"], 2_000)
        self.assertEqual(
            queries["sha256"], verification["digests"]["data/queries.tsv"])

        long_context = report["artifacts"]["long_context_results"]["data"]
        self.assertEqual(long_context["queries"], 1_000)
        self.assertEqual(long_context["distractor_facts"], 512)
        self.assertEqual(long_context["short_context_correct"], 0)
        self.assertEqual(long_context["pair_index_conceptual_correct"], 1_000)
        self.assertEqual(long_context["hash_index_correct"], 1_000)

    def test_re_run_matches_artifact_counts(self):
        benchmark = self.report["benchmark"]
        exact = benchmark["exact"]
        two_hop = benchmark["two_hop"]
        replay = benchmark["long_context"]
        results = self.report["artifacts"]["results"]["data"]
        long_context = self.report["artifacts"]["long_context_results"]["data"]

        self.assertEqual(benchmark["facts"], 50_268)
        self.assertEqual(benchmark["root_bytes"], 8_388_608)
        self.assertEqual(benchmark["query_trials"], 250_000)
        self.assertEqual(exact["queries"], 250_000)
        self.assertEqual(exact["posting_found"], 250_000)
        self.assertEqual(exact["hash_found"], 250_000)
        self.assertEqual(exact["posting_found"], exact["hash_found"])
        self.assertEqual(exact["identity_mismatches"], 0)

        self.assertEqual(two_hop["queries"], 2_000)
        self.assertEqual(two_hop["posting_found"], 2_000)
        self.assertEqual(two_hop["hash_found"], 2_000)
        self.assertEqual(
            two_hop["posting_found"], results["twohop_found"])

        self.assertEqual(replay["queries"], long_context["queries"])
        self.assertEqual(
            replay["distractor_facts"], long_context["distractor_facts"])
        self.assertEqual(
            replay["short_context_correct"],
            long_context["short_context_correct"],
        )
        self.assertEqual(
            replay["pair_index_conceptual_correct"],
            long_context["pair_index_conceptual_correct"],
        )
        self.assertEqual(
            replay["hash_index_correct"],
            long_context["hash_index_correct"],
        )
        self.assertEqual(replay["short_context_correct"], 0)

        layout = benchmark["root_layout"]
        self.assertEqual(layout["root_bytes"], r032.ROOT_BYTES)
        self.assertTrue(layout["matches_expected"])
        self.assertTrue(layout["pair_count_matches"])
        self.assertEqual(layout["computed_pair_cells"], 32_640)

    def test_no_canonical_decoder_or_pair_speed_claim(self):
        claims = self.report["claims"]
        for key in (
            "canonical",
            "canonical_implementation",
            "neural_decoder",
            "language_model",
            "language_model_win",
            "lm_win",
            "learned_generalization",
            "pair_index_faster",
            "mirror_performance_advantage",
            "quantum_claim",
            "nan_unique_benefit",
        ):
            self.assertIs(claims[key], False, key)
        self.assertIn("No canonical implementation", claims["scope"])
        self.assertIn("neural decoder", claims["scope"])

        benchmark_note = self.report["benchmark"]["note"].lower()
        self.assertIn("no pair-index speed advantage", benchmark_note)
        self.assertIn("supplied artifact reports it as faster", benchmark_note)
        self.assertEqual(
            self.report["assertions"]["pair_index_faster_claimed"], False)

        boundaries = " ".join(self.report["boundaries"]).lower()
        self.assertIn("no pair-index or mirror-geometry performance", boundaries)
        self.assertIn("does not test a neural decoder", boundaries)

    def test_markdown_and_json_are_deterministic(self):
        report = self.report
        first = r032.render_markdown(report)
        second = r032.render_markdown(copy.deepcopy(report))
        self.assertEqual(first, second)
        self.assertNotIn("generated_at", report)
        self.assertNotIn("generated_at_utc", first)
        self.assertIn("# Round 032 retrieval audit", first)
        self.assertIn("## Python replay", first)
        self.assertIn("No pair-index", first)

        with tempfile.TemporaryDirectory() as directory:
            first_json = Path(directory) / "first.json"
            second_json = Path(directory) / "second.json"
            r032._write_json(first_json, report)
            r032._write_json(second_json, report)
            self.assertEqual(
                first_json.read_bytes(), second_json.read_bytes())
            parsed = json.loads(first_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema"], r032.SCHEMA)
            self.assertEqual(parsed["status"], "ok")

    def test_checks_fail_cleanly_on_malformed_artifact_values(self):
        with self.assertRaises(r032.ArtifactError):
            r032._validate_results({
                "facts": 50_268,
                "root_bytes": 8_388_608,
                "root_locked": 1,
                "pair_map_pass": "yes",
                "membership_count": 50_268,
                "queries": 250_000,
                "posting_found": 250_000,
                "hash_found": 250_000,
                "twohop_queries": 2_000,
                "twohop_found": 2_000,
                "direct_found": 0,
                "ingest_ms": 1.0,
                "posting_ms": 2.0,
                "hash_ms": 1.0,
                "cycle_before": 0.0,
                "cycle_after_update": 1.0,
                "cycle_after_gauge": 1.0,
            })
        with self.assertRaises(r032.ArtifactError):
            r032._validate_long_context({
                "queries": 1_000,
                "distractor_facts": 512,
                "short_context_correct": 0,
                "pair_index_conceptual_correct": 1_000,
                "hash_index_correct": 1_000,
                "evaluation_ms": 1.0,
            })


class DenialTests(unittest.TestCase):
    def test_checksum_mismatch_denies_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "round032"
            shutil.copytree(EXPERIMENT, copied)
            results_path = copied / "results.json"
            results_path.write_text(
                results_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                r032,
                "load_artifacts",
                side_effect=AssertionError("artifacts must not be parsed"),
            ) as loader:
                report = r032.run_audit(
                    experiment_dir=copied,
                    native=False,
                    queries=10,
                    two_hop=False,
                    long_context=False,
                )
            loader.assert_not_called()
            self.assertEqual(report["status"], "denied")
            self.assertEqual(report["exit_code"], r032.EXIT_DENIED)
            self.assertEqual(report["verification"]["status"], "denied")
            self.assertEqual(report["verification"]["verified_count"], 0)
            self.assertIsNone(report["artifacts"])
            self.assertIsNone(report["benchmark"])
            self.assertIsNone(report["native"])
            self.assertFalse(report["checks"]["all_targets_met"])
            self.assertIn("SHA-256 mismatch", report["verification"]["reason"])
            self.assertIn(
                "No artifact was parsed", " ".join(report["notes"]))

    def test_missing_manifest_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "round032"
            shutil.copytree(EXPERIMENT, copied)
            (copied / "SHA256SUMS").unlink()
            report = r032.run_audit(
                experiment_dir=copied,
                native=False,
                queries=10,
                two_hop=False,
                long_context=False,
            )
            self.assertEqual(report["status"], "denied")
            self.assertEqual(report["exit_code"], r032.EXIT_DENIED)


class ParseTests(unittest.TestCase):
    def test_facts_and_queries_parse_three_positive_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            facts = Path(directory) / "facts.tsv"
            facts.write_text("1 2 3\n\n4 5 6\n", encoding="utf-8")
            self.assertEqual(
                r032.parse_facts_tsv(facts),
                [(1, 2, 3), (4, 5, 6)],
            )
            queries = Path(directory) / "queries.tsv"
            queries.write_text("7 1 9\n", encoding="utf-8")
            self.assertEqual(r032.parse_queries_tsv(queries), [(7, 1, 9)])

    def test_malformed_tsv_is_an_artifact_error(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.tsv"
            for text in (
                "1 2\n",
                "1 2 3 4\n",
                "1 x 3\n",
                "0 1 2\n",
                "1 2 -3\n",
                "",
            ):
                bad.write_text(text, encoding="utf-8")
                with self.assertRaises(r032.ArtifactError):
                    r032.parse_facts_tsv(bad)


class BoundedQueryTests(unittest.TestCase):
    def test_query_count_is_a_hard_bound(self):
        report = r032.run_audit(
            queries=37,
            two_hop=False,
            long_context=False,
            native=False,
        )
        self.assertEqual(report["status"], "ok")
        benchmark = report["benchmark"]
        self.assertEqual(benchmark["query_trials"], 37)
        self.assertEqual(benchmark["exact"]["queries"], 37)
        self.assertEqual(benchmark["exact"]["posting_found"], 37)
        self.assertEqual(benchmark["exact"]["hash_found"], 37)
        self.assertEqual(benchmark["exact"]["identity_mismatches"], 0)
        self.assertIsNone(benchmark["two_hop"])
        self.assertIsNone(benchmark["two_hop_requested"])
        self.assertIsNone(benchmark["long_context"])

    def test_two_hop_limit_is_a_hard_bound(self):
        report = r032.run_audit(
            queries=5,
            two_hop=True,
            two_hop_limit=7,
            long_context=False,
            native=False,
        )
        self.assertEqual(report["status"], "ok")
        two_hop = report["benchmark"]["two_hop"]
        self.assertEqual(two_hop["queries"], 7)
        self.assertEqual(two_hop["posting_found"], 7)
        self.assertEqual(two_hop["hash_found"], 7)
        self.assertEqual(report["benchmark"]["two_hop_requested"], 7)

    def test_invalid_query_counts_are_rejected(self):
        for value in (0, -1, r032.MAX_QUERY_TRIALS + 1):
            with self.assertRaises(ValueError):
                r032.run_audit(
                    queries=value,
                    two_hop=False,
                    long_context=False,
                    native=False,
                )


class CommandLineTests(unittest.TestCase):
    def test_json_and_markdown_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "report.json"
            markdown_path = Path(directory) / "report.md"
            with _quiet() as stdout:
                exit_code = r032.main([
                    "--queries",
                    "19",
                    "--no-two-hop",
                    "--no-long-context",
                    "--no-native",
                    "--json",
                    str(json_path),
                    "--markdown",
                    str(markdown_path),
                ])
            self.assertEqual(exit_code, r032.EXIT_OK)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["schema"], r032.SCHEMA)
            self.assertEqual(receipt["json"], str(json_path))
            self.assertEqual(receipt["markdown"], str(markdown_path))

            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["benchmark"]["exact"]["queries"], 19)
            self.assertTrue(report["checks"]["all_targets_met"])

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# Round 032 retrieval audit", markdown)
            self.assertIn("## Python replay", markdown)
            self.assertIn("| Exact lookups | 19 |", markdown)
            self.assertIn("conventional hash comparator", markdown)


if __name__ == "__main__":
    unittest.main()
