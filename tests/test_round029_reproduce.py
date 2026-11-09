"""Tests for the dependency-light Round 029 reproduction.

These tests exercise the new script only. They do not import torch, sklearn,
numpy, or any WordNet corpus.
"""

import contextlib
import functools
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from scripts import round029_reproduce as repro


@functools.lru_cache(maxsize=8)
def _report(seed=29, samples=12):
    return repro.run(seed=seed, samples_per_class=samples)


class DatasetTests(unittest.TestCase):
    def test_synthetic_dataset_is_deterministic(self):
        first = repro.make_synthetic_dataset(seed=29, samples_per_class=8)
        second = repro.make_synthetic_dataset(seed=29, samples_per_class=8)
        self.assertEqual(first["features"], second["features"])
        self.assertEqual(first["labels"], second["labels"])
        self.assertEqual(first["class_names"], second["class_names"])
        self.assertEqual(first["source"], "deterministic_synthetic_q8_8")

    def test_shape_and_fixed_point_values(self):
        dataset = repro.make_synthetic_dataset(seed=29, samples_per_class=8)
        self.assertEqual(dataset["dims"], 48)
        self.assertEqual(dataset["classes"], 16)
        self.assertEqual(len(dataset["features"]), 16 * 8)
        self.assertEqual(len(set(dataset["labels"])), 16)
        counts = {label: dataset["labels"].count(label) for label in range(16)}
        self.assertEqual(set(counts.values()), {8})
        for row in dataset["features"]:
            self.assertEqual(len(row), 48)
            self.assertTrue(all(isinstance(value, int) for value in row))
            self.assertTrue(
                all(-repro.PAYLOAD_SIGN <= value < repro.PAYLOAD_SIGN for value in row)
            )

    def test_different_seed_changes_features(self):
        first = repro.make_synthetic_dataset(seed=29, samples_per_class=8)
        second = repro.make_synthetic_dataset(seed=30, samples_per_class=8)
        self.assertNotEqual(first["features"], second["features"])

    def test_stratified_split_is_deterministic_and_disjoint(self):
        dataset = repro.make_synthetic_dataset(seed=29, samples_per_class=8)
        labels = dataset["labels"]
        first = repro.stratified_split(labels, seed=29)
        second = repro.stratified_split(labels, seed=29)
        self.assertEqual(first, second)

        combined = first["train"] + first["val"] + first["test"]
        self.assertEqual(sorted(combined), list(range(len(labels))))
        self.assertEqual(len(combined), len(set(combined)))
        for name in ("train", "val", "test"):
            self.assertTrue(first[name])
            self.assertEqual(
                {labels[index] for index in first[name]}, set(range(16))
            )


class PayloadCodecTests(unittest.TestCase):
    def test_signed_payload_roundtrip_and_nan_carrier(self):
        for value in (
            -repro.PAYLOAD_SIGN,
            -123456789,
            -1,
            0,
            1,
            123456789,
            repro.PAYLOAD_SIGN - 1,
        ):
            carrier = repro.pack_payload(value)
            self.assertTrue(math.isnan(carrier))
            self.assertEqual(repro.unpack_payload(carrier), value)
            self.assertEqual(repro.payload_bits(carrier) >> 48, 0x7FF8)

    def test_payload_range_is_enforced(self):
        with self.assertRaises(ValueError):
            repro.pack_payload(repro.PAYLOAD_SIGN)
        with self.assertRaises(ValueError):
            repro.pack_payload(-repro.PAYLOAD_SIGN - 1)

    def test_non_payload_carriers_are_rejected(self):
        for carrier in (0.0, -1.5, float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                repro.unpack_payload(carrier)


class GaugeTests(unittest.TestCase):
    def setUp(self):
        self.z = [((index * 37) % 201) - 100 for index in range(48)]
        rng_one = random.Random(7)
        rng_two = random.Random(11)
        self.gauge_one = [rng_one.randint(-10**6, 10**6) for _ in range(256)]
        self.gauge_two = [rng_two.randint(-10**6, 10**6) for _ in range(256)]

    def test_lanes_are_nan_payloads_with_the_expected_formula(self):
        lanes = repro.encode_payload_lanes(self.z, self.gauge_one)
        self.assertEqual(set(lanes), {"oa", "ab", "ob"})
        for lane in lanes.values():
            self.assertEqual(len(lane), 48)
            self.assertTrue(all(math.isnan(value) for value in lane))
        for index, value in enumerate(self.z):
            a, b = repro.triangle_indices(index)
            self.assertEqual(
                repro.unpack_payload(lanes["oa"][index]),
                self.gauge_one[0] - self.gauge_one[a],
            )
            self.assertEqual(
                repro.unpack_payload(lanes["ab"][index]),
                value + self.gauge_one[a] - self.gauge_one[b],
            )
            self.assertEqual(
                repro.unpack_payload(lanes["ob"][index]),
                self.gauge_one[0] - self.gauge_one[b],
            )

    def test_cycle_cancels_exactly_under_two_fresh_gauges(self):
        first_lanes = repro.encode_payload_lanes(self.z, self.gauge_one)
        second_lanes = repro.encode_payload_lanes(self.z, self.gauge_two)
        first = repro.decode_payload_lanes(first_lanes, "cycle")
        second = repro.decode_payload_lanes(second_lanes, "cycle")
        self.assertEqual(first, self.z)
        self.assertEqual(second, self.z)
        self.assertEqual(first, second)
        for left, right in zip(first, second):
            self.assertEqual(
                repro.payload_bits(repro.pack_payload(left)),
                repro.payload_bits(repro.pack_payload(right)),
            )

    def test_cycle_readout_repacks_into_nan_payload_carriers(self):
        lanes = repro.encode_payload_lanes(self.z, self.gauge_one)
        carriers = repro.readout_payload_lanes(lanes, "cycle")
        self.assertEqual(len(carriers), 48)
        self.assertTrue(all(math.isnan(carrier) for carrier in carriers))
        self.assertEqual(
            [repro.unpack_payload(carrier) for carrier in carriers], self.z
        )

    def test_raw_and_wrong_triangle_lose_the_gauge(self):
        first_lanes = repro.encode_payload_lanes(self.z, self.gauge_one)
        second_lanes = repro.encode_payload_lanes(self.z, self.gauge_two)
        self.assertNotEqual(
            repro.decode_payload_lanes(first_lanes, "raw"),
            repro.decode_payload_lanes(second_lanes, "raw"),
        )
        self.assertNotEqual(
            repro.decode_payload_lanes(first_lanes, "wrong_triangle"),
            repro.decode_payload_lanes(second_lanes, "wrong_triangle"),
        )
        self.assertEqual(
            repro.decode_payload_lanes(first_lanes, "cycle"), self.z
        )

    def test_zero_gauge_recovers_raw_and_wrong_triangle(self):
        zero = [0] * 256
        lanes = repro.encode_payload_lanes(self.z, zero)
        self.assertEqual(repro.decode_payload_lanes(lanes, "raw"), self.z)
        self.assertEqual(
            repro.decode_payload_lanes(lanes, "wrong_triangle"), self.z
        )
        self.assertEqual(repro.decode_payload_lanes(lanes, "cycle"), self.z)


class ConventionalTieTests(unittest.TestCase):
    def test_integer_lanes_tie_payload_lanes_for_every_readout(self):
        values = [((index * 19) % 401) - 200 for index in range(48)]
        rng = random.Random(13)
        gauge = [rng.randint(-10**5, 10**5) for _ in range(256)]
        payload_lanes = repro.encode_payload_lanes(values, gauge)
        integer_lanes = repro.encode_integer_lanes(values, gauge)
        for lane in ("oa", "ab", "ob"):
            self.assertEqual(
                [repro.unpack_payload(value) for value in payload_lanes[lane]],
                integer_lanes[lane],
            )
        for kind in repro.READOUTS:
            self.assertEqual(
                repro.decode_payload_lanes(payload_lanes, kind),
                repro.decode_integer_lanes(integer_lanes, kind),
            )


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _report(seed=29, samples=12)

    def test_measured_and_target_are_explicit(self):
        report = self.report
        self.assertIn("measured", report)
        self.assertIn("target", report)
        self.assertIn("checks", report)
        self.assertEqual(report["schema"], repro.SCHEMA)
        self.assertTrue(report["checks"]["all_targets_met"])
        self.assertIn("cycle_test_accuracy", report["measured"])
        self.assertIn("raw_test_accuracy", report["measured"])
        self.assertIn("wrong_triangle_test_accuracy", report["measured"])

    def test_exact_cycle_cancellation(self):
        exactness = self.report["exactness"]
        self.assertTrue(exactness["cycle_payload_bitwise_equals_semantics"])
        self.assertTrue(exactness["cycle_readout_carriers_are_nan"])
        self.assertTrue(
            exactness["cycle_readout_carrier_bits_equal_semantics"]
        )
        self.assertTrue(exactness["cycle_bitwise_gauge_invariant"])
        self.assertEqual(exactness["max_cycle_abs_error"], 0)
        self.assertEqual(exactness["cycle_mismatch_count"], 0)

    def test_raw_and_wrong_triangle_fail_under_large_gauge(self):
        measured = self.report["measured"]
        target = self.report["target"]
        self.assertLessEqual(
            measured["raw_test_accuracy"], target["raw_test_accuracy_max"]
        )
        self.assertLessEqual(
            measured["wrong_triangle_test_accuracy"],
            target["wrong_triangle_test_accuracy_max"],
        )
        self.assertFalse(self.report["exactness"]["raw_bitwise_gauge_invariant"])
        self.assertFalse(
            self.report["exactness"]["wrong_triangle_bitwise_gauge_invariant"]
        )
        self.assertGreater(self.report["exactness"]["raw_max_gauge_delta"], 0)
        self.assertGreater(
            self.report["exactness"]["wrong_triangle_max_gauge_delta"], 0
        )
        self.assertGreaterEqual(
            measured["raw_zero_gauge_accuracy"],
            target["raw_zero_gauge_accuracy_min"],
        )
        self.assertGreaterEqual(
            measured["wrong_triangle_zero_gauge_accuracy"],
            target["wrong_triangle_zero_gauge_accuracy_min"],
        )

    def test_conventional_graph_implementation_ties(self):
        ties = self.report["conventional_ties"]
        for kind in repro.READOUTS:
            self.assertTrue(ties[kind])
        self.assertTrue(ties["conventional_graph_implementation_ties"])
        self.assertTrue(self.report["conventional_graph_implementation_ties"])
        self.assertTrue(
            self.report["boundaries"][
                "conventional_graph_implementation_ties_nan_lanes"
            ]
        )

    def test_no_nan_unique_advantage_is_explicit(self):
        boundaries = self.report["boundaries"]
        self.assertFalse(self.report["nan_unique_advantage"])
        self.assertTrue(self.report["no_nan_unique_advantage"])
        self.assertFalse(boundaries["nan_unique_advantage"])
        self.assertTrue(boundaries["no_nan_unique_advantage"])
        self.assertFalse(boundaries["ieee_nan_arithmetic_used"])
        self.assertTrue(boundaries["payloads_are_storage_not_compute"])

    def test_default_dataset_does_not_claim_wordnet(self):
        dataset = self.report["dataset"]
        self.assertEqual(dataset["source"], "deterministic_synthetic_q8_8")
        self.assertFalse(dataset["wordnet_used"])
        self.assertFalse(dataset["claims_wordnet_result"])
        self.assertIn("No WordNet", dataset["wordnet_note"])
        self.assertFalse(self.report["boundaries"]["wordnet_result_claimed"])
        self.assertNotIn("wordnet_accuracy", self.report["measured"])

    def test_report_is_deterministic(self):
        first = repro.run(seed=41, samples_per_class=8)
        second = repro.run(seed=41, samples_per_class=8)
        self.assertEqual(first, second)


class CliOutputTests(unittest.TestCase):
    def test_cli_writes_valid_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "round029.json"
            markdown_path = Path(directory) / "round029.md"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = repro.main(
                    [
                        "--seed",
                        "29",
                        "--samples",
                        "10",
                        "--json",
                        str(json_path),
                        "--markdown",
                        str(markdown_path),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

            text = json_path.read_text(encoding="utf-8")
            for token in (": NaN", ": Infinity", ": -Infinity"):
                self.assertNotIn(token, text)
            report = json.loads(text)

            def assert_finite(value):
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))
                elif isinstance(value, dict):
                    for item in value.values():
                        assert_finite(item)
                elif isinstance(value, list):
                    for item in value:
                        assert_finite(item)

            assert_finite(report)
            self.assertEqual(report["schema"], repro.SCHEMA)
            self.assertIn("measured", report)
            self.assertIn("target", report)
            self.assertTrue(report["checks"]["all_targets_met"])
            self.assertTrue(report["conventional_graph_implementation_ties"])
            self.assertFalse(report["nan_unique_advantage"])
            self.assertTrue(report["no_nan_unique_advantage"])

            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# Round 029 reproduction", markdown)
            self.assertIn("## Exact cancellation", markdown)
            self.assertIn("## Measured vs target", markdown)
            self.assertIn("conventional", markdown.lower())
            self.assertIn("no NaN-unique advantage", markdown)
            self.assertIn("No WordNet", markdown)


class OptionalWordNetTests(unittest.TestCase):
    @staticmethod
    def _write_wordnet(path, classes=16, per_class=6, use_floats=False):
        lines = []
        for label in range(classes):
            for item in range(per_class):
                if use_floats:
                    vector = [
                        (label * 100 + item + dimension) / 256.0
                        for dimension in range(48)
                    ]
                else:
                    vector = [
                        label * 100 + item + dimension for dimension in range(48)
                    ]
                lines.append(
                    json.dumps(
                        {
                            "id": "n%02d%02d" % (label, item),
                            "class": label,
                            "vector": vector,
                            "gloss": "synthetic test record",
                        }
                    )
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_optional_local_jsonl_is_labeled_without_a_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wordnet_features.jsonl"
            self._write_wordnet(path)
            report = repro.run(
                seed=29, samples_per_class=8, wordnet_path=path
            )
            dataset = report["dataset"]
            self.assertEqual(dataset["source"], "local_wordnet_jsonl")
            self.assertTrue(dataset["wordnet_used"])
            self.assertFalse(dataset["claims_wordnet_result"])
            self.assertEqual(dataset["wordnet_path"], str(path))
            self.assertFalse(report["boundaries"]["wordnet_result_claimed"])
            self.assertNotIn("wordnet_accuracy", report["measured"])
            self.assertIn("not a WordNet benchmark", dataset["wordnet_note"])

    def test_float_vectors_are_converted_to_fixed_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wordnet_float.jsonl"
            self._write_wordnet(path, use_floats=True)
            dataset = repro.load_wordnet_dataset(path)
            self.assertEqual(dataset["source"], "local_wordnet_jsonl")
            self.assertTrue(dataset["wordnet_used"])
            self.assertFalse(dataset["claims_wordnet_result"])
            for row in dataset["features"]:
                self.assertTrue(all(isinstance(value, int) for value in row))

    def test_wrong_class_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wordnet_15.jsonl"
            self._write_wordnet(path, classes=15)
            with self.assertRaises(ValueError):
                repro.load_wordnet_dataset(path)


if __name__ == "__main__":
    unittest.main()
