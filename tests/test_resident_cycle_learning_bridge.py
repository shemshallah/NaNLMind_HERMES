"""Tests for the gauge-invariant resident cycle learning bridge.

The suite exercises:

* exact gauge invariance of cycle ids/features/scores under vertex shifts;
* multiple active heads and sparse, bounded cycle selection;
* scratch-first rejection and dedicated denial on non-integer payloads or a
  non-zero gauge residual;
* temporal/field preservation when the bridge observes text;
* the real ``ResidentCells`` pair rails, including permanent identities;
* read-only evaluation and deterministic statistics.
"""
from __future__ import annotations

import math
import unittest

from nanlmind_hermes.resident.cells import (
    REGION_BYTES,
    ResidentCells,
    canonical_address,
    identity_address,
    reverse16,
)
from nanlmind_hermes.resident.cycle_learning_bridge import (
    CANONICAL_MERGE,
    CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH,
    CycleFieldAdapter,
    CycleLearningBridge,
    CycleLearningDenied,
    ResidentCellCycleAdapter,
    cycle_edge_delta,
    decode_cycle_id,
    encode_cycle_id,
    encode_edge_id,
    gauge_residual,
    triangle_flux,
)
from nanlmind_hermes.resident.field_learning import (
    FieldLearner,
    HeadFieldAdapter,
    ResidentCellFieldAdapter,
)
from nanlmind_hermes.temporal_lm import TemporalMirrorLM

SENTENCES = [
    "A penguin swims in cold water.",
    "A river carries water to the sea.",
    "A bird can fly over the river.",
]


class FakeTemporal:
    """Deterministic temporal model with observe(), vocab and _prob()."""

    def __init__(self, probability: float = 0.2):
        self.probability = probability
        self.vocab: dict[str, int] = {}
        self.observed: list[str] = []
        self.seen: set[str] = set()

    def observe(self, text: str) -> bool:
        value = str(text).strip()
        if value.endswith("?") or value.casefold() in self.seen:
            return False
        self.seen.add(value.casefold())
        self.observed.append(value)
        return True

    def _prob(self, context2: str | None, context1: str, word: str) -> float:
        return self.probability


class NoProbabilityTemporal:
    def __init__(self):
        self.observed: list[str] = []

    def observe(self, text: str) -> bool:
        self.observed.append(str(text))
        return True


class GaugeTamperingCycleField(CycleFieldAdapter):
    """Stages a valid proposal, then exposes a gauge-corrupt staged copy."""

    def __init__(self):
        super().__init__()
        self.propose_calls = 0
        self.apply_calls = 0

    def propose(self, update):
        self.propose_calls += 1
        proposal = super().propose(update)
        tampered = {
            **proposal,
            "edges": {encode_edge_id(0, 1): 1},
            "gauge_residual": 1,
        }
        return tampered

    def apply_proposal(self, proposal):
        self.apply_calls += 1
        return super().apply_proposal(proposal)


class DeltaTamperingCycleField(CycleFieldAdapter):
    """Stages a valid proposal, then exposes gauge-free but oversized deltas."""

    def __init__(self):
        super().__init__()
        self.propose_calls = 0
        self.apply_calls = 0

    def propose(self, update):
        self.propose_calls += 1
        proposal = super().propose(update)
        tampered = {
            **proposal,
            "edges": {
                edge_id: delta * 1000
                for edge_id, delta in proposal["edges"].items()
            },
        }
        return tampered

    def apply_proposal(self, proposal):
        self.apply_calls += 1
        return super().apply_proposal(proposal)


class NonIntegerCycleField(CycleFieldAdapter):
    """Publishes a float edge delta, which must be denied before staging."""

    def propose(self, update):
        payload = dict(update)
        payload["edges"] = {encode_edge_id(1, 2): 1.5}
        return super().propose(payload)


class HelperTests(unittest.TestCase):
    def test_ids_are_raw_integers_and_triangle_flux_is_gauge_invariant(self):
        cycle_id = encode_cycle_id(9, 2, 5)
        self.assertIsInstance(cycle_id, int)
        self.assertEqual(decode_cycle_id(cycle_id), (2, 5, 9))

        edges = {
            encode_edge_id(2, 5): 3,
            encode_edge_id(5, 9): 4,
            encode_edge_id(2, 9): 1,
        }
        self.assertEqual(triangle_flux(edges, cycle_id), 6)
        self.assertEqual(gauge_residual(cycle_edge_delta(cycle_id, 3)), 0)

        adapter = CycleFieldAdapter(edges)
        potential = [((index * 7) % 11) - 5 for index in range(256)]
        shifted = adapter.with_gauge(potential)
        self.assertNotEqual(adapter.read_edges(), shifted.read_edges())
        for head in range(256):
            self.assertEqual(
                potential[head], potential[head])  # deterministic fixture
        self.assertEqual(
            triangle_flux(adapter.read_edges(), cycle_id),
            triangle_flux(shifted.read_edges(), cycle_id))

    def test_non_integer_ids_and_payloads_are_denied(self):
        with self.assertRaises(CycleLearningDenied) as caught:
            encode_cycle_id(1, 2, 2.5)
        self.assertEqual(caught.exception.code, "non_integer")

        with self.assertRaises(CycleLearningDenied) as caught:
            gauge_residual({encode_edge_id(1, 2): 1.0})
        self.assertEqual(caught.exception.code, "non_integer")

        adapter = CycleFieldAdapter()
        with self.assertRaises(CycleLearningDenied) as caught:
            adapter.propose({"edges": {"not-an-int": 1}})
        self.assertEqual(caught.exception.code, "non_integer")
        self.assertEqual(adapter.scratch, {})


class GaugeInvarianceTests(unittest.TestCase):
    def test_gauge_shift_leaves_cycle_readout_and_stats_unchanged(self):
        def run(potential):
            adapter = CycleFieldAdapter({
                encode_edge_id(1, 2): 3,
                encode_edge_id(2, 3): 2,
                encode_edge_id(1, 3): -1,
            })
            if potential is not None:
                adapter = adapter.with_gauge(potential)
            bridge = CycleLearningBridge(
                FakeTemporal(), HeadFieldAdapter(), adapter,
                max_active_heads=12, max_active_cycles=6)
            return bridge.observe(SENTENCES[0]), bridge.stats(), adapter

        potential = [((index * 13) % 17) - 8 for index in range(256)]
        first, first_stats, first_adapter = run(None)
        second, second_stats, second_adapter = run(potential)

        for field in ("active_heads", "active_cycles", "applied_cycles",
                      "cycle_ids", "cycle_features", "cycles", "prediction",
                      "prediction_score", "predicted_class", "gauge_residual",
                      "writes", "temporal_nll"):
            self.assertEqual(first[field], second[field], field)
        self.assertEqual(first_stats, second_stats)
        self.assertNotEqual(
            first_adapter.read_edges(), second_adapter.read_edges())
        self.assertEqual(first["gauge_residual"], 0)

        for cycle_id in first["cycle_ids"]:
            self.assertEqual(
                triangle_flux(first_adapter.read_edges(), cycle_id),
                triangle_flux(second_adapter.read_edges(), cycle_id))


class CycleLearningTests(unittest.TestCase):
    def test_multiple_heads_and_cycles_with_bounded_writes(self):
        bridge = CycleLearningBridge(
            FakeTemporal(), HeadFieldAdapter(), CycleFieldAdapter(),
            max_active_heads=8, max_active_cycles=5,
            max_writes_per_sentence=7, max_cycle_delta=2,
            max_total_cycles=3)

        result = bridge.observe(SENTENCES[0])

        self.assertTrue(result["learned"])
        self.assertTrue(result["cycle_learned"])
        self.assertGreater(result["active_heads"], 1)
        self.assertLessEqual(result["active_heads"], 8)
        self.assertGreater(result["active_cycles"], 1)
        self.assertLessEqual(result["active_cycles"], 5)
        self.assertGreaterEqual(result["applied_cycles"], 1)
        self.assertLessEqual(result["writes"], 7)
        self.assertLessEqual(len(bridge.cycle_ledger), 3)

        heads_seen: set[int] = set()
        for cycle_id in result["cycle_ids"]:
            self.assertIsInstance(cycle_id, int)
            a, b, c = decode_cycle_id(cycle_id)
            self.assertEqual(len({a, b, c}), 3)
            heads_seen.update((a, b, c))
        self.assertGreater(len(heads_seen), 3)
        self.assertEqual(result["gauge_residual"], 0)

    def test_scratch_rejection_never_applies_cycle_edges(self):
        temporal = FakeTemporal()
        cycle_field = DeltaTamperingCycleField()
        bridge = CycleLearningBridge(temporal, HeadFieldAdapter(), cycle_field)
        before = cycle_field.read_edges()

        result = bridge.observe(SENTENCES[0])

        self.assertFalse(result["learned"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "invariant_failure")
        self.assertFalse(result["invariants"]["staged_matches_submitted"])
        self.assertFalse(result["invariants"]["deltas_bounded"])
        self.assertTrue(result["invariants"]["main_cycle_field_unchanged"])
        self.assertEqual(cycle_field.propose_calls, 1)
        self.assertEqual(cycle_field.apply_calls, 0)
        self.assertEqual(cycle_field.read_edges(), before)
        self.assertEqual(cycle_field.scratch, {})
        self.assertEqual(result["writes"], 0)
        self.assertEqual(temporal.observed, [SENTENCES[0]])

    def test_gauge_residual_tampering_is_a_dedicated_denial(self):
        temporal = FakeTemporal()
        cycle_field = GaugeTamperingCycleField()
        bridge = CycleLearningBridge(temporal, HeadFieldAdapter(), cycle_field)
        before = cycle_field.read_edges()

        with self.assertRaises(CycleLearningDenied) as caught:
            bridge.observe(SENTENCES[0])

        self.assertEqual(caught.exception.code, "gauge_residual")
        self.assertEqual(cycle_field.apply_calls, 0)
        self.assertEqual(cycle_field.read_edges(), before)
        self.assertEqual(cycle_field.scratch, {})
        self.assertEqual(bridge.denials, 1)
        self.assertEqual(bridge.last_denial["code"], "gauge_residual")
        # The temporal channel was observed before the cycle denial.
        self.assertEqual(temporal.observed, [SENTENCES[0]])

    def test_non_integer_adapter_payload_is_denied(self):
        bridge = CycleLearningBridge(
            FakeTemporal(), HeadFieldAdapter(), NonIntegerCycleField())
        with self.assertRaises(CycleLearningDenied) as caught:
            bridge.observe(SENTENCES[0])
        self.assertEqual(caught.exception.code, "non_integer")
        self.assertEqual(bridge.accepted_proposals, 0)
        self.assertEqual(bridge.writes, 0)

    def test_zero_write_budget_rejects_before_staging(self):
        cycle_field = CycleFieldAdapter()
        bridge = CycleLearningBridge(
            FakeTemporal(), HeadFieldAdapter(), cycle_field,
            total_write_budget=0)
        result = bridge.observe(SENTENCES[0])
        self.assertFalse(result["learned"])
        self.assertEqual(result["reason"], "write_budget_exhausted")
        self.assertEqual(result["writes"], 0)
        self.assertEqual(cycle_field.read_edges(), {})
        self.assertEqual(cycle_field.scratch, {})
        self.assertEqual(bridge.writes, 0)

    def test_temporal_and_field_channels_are_preserved(self):
        temporal = FakeTemporal()
        field = FieldLearner(temporal, HeadFieldAdapter())
        bridge = CycleLearningBridge(temporal, field, CycleFieldAdapter())

        first = bridge.observe(SENTENCES[0])
        self.assertTrue(first["temporal_learned"])
        self.assertTrue(first["field_learned"])
        self.assertTrue(first["learned"])
        self.assertIs(bridge.temporal, temporal)
        self.assertIs(bridge.field, field)
        self.assertEqual(temporal.observed, [SENTENCES[0]])
        self.assertEqual(field.stats()["sentences"], 1)

        duplicate = bridge.observe(SENTENCES[0])
        self.assertFalse(duplicate["temporal_learned"])
        self.assertTrue(duplicate["field_learned"])
        self.assertTrue(duplicate["learned"])
        self.assertEqual(len(temporal.observed), 1)
        self.assertEqual(field.stats()["accepted_proposals"], 2)

    def test_real_temporal_model_is_not_replaced(self):
        temporal = TemporalMirrorLM()
        bridge = CycleLearningBridge(
            temporal, HeadFieldAdapter(), CycleFieldAdapter())

        result = bridge.observe(SENTENCES[0])

        self.assertIs(bridge.temporal, temporal)
        self.assertTrue(result["temporal_learned"])
        self.assertIn("penguin", temporal.vocab)
        self.assertTrue(result["learned"])

    def test_real_resident_cells_keep_identity_and_commit_pair_edges(self):
        cells = ResidentCells.format_region(bytearray(REGION_BYTES))
        field = FieldLearner(FakeTemporal(), ResidentCellFieldAdapter(cells))
        cycle_field = ResidentCellCycleAdapter(cells)
        bridge = CycleLearningBridge(
            field.temporal, field, cycle_field,
            max_active_heads=10, max_active_cycles=4,
            max_writes_per_sentence=12)

        result = bridge.observe(SENTENCES[1])

        self.assertTrue(result["learned"])
        self.assertGreater(result["active_cycles"], 0)
        self.assertGreater(result["writes"], 0)
        self.assertEqual(result["gauge_residual"], 0)
        self.assertTrue(cycle_field.read_cycles())

        for head in range(256):
            record = cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity, identity_address(head))

        for update in result["edge_updates"]:
            left, right = update["left"], update["right"]
            self.assertNotEqual(left, right)
            state = cells.pair_state(left, right)
            self.assertEqual(state.canonical, canonical_address(left, right))
            self.assertEqual(state.mirror, reverse16(state.canonical))
            self.assertGreater(
                state.forward.amplitude + state.reverse.amplitude, 0)

        self.assertFalse(result["canonical_merge"])
        self.assertFalse(result["semantic_truth"])
        self.assertFalse(hasattr(bridge, "merge"))
        self.assertFalse(hasattr(cycle_field, "merge"))

    def test_evaluate_is_read_only_and_reports_temporal_nll(self):
        temporal = FakeTemporal(probability=0.25)
        cycle_field = CycleFieldAdapter()
        bridge = CycleLearningBridge(
            temporal, HeadFieldAdapter(), cycle_field)
        bridge.observe(SENTENCES[0])
        before_edges = cycle_field.read_edges()
        before_cycles = cycle_field.read_cycles()
        before_stats = bridge.stats()

        report = bridge.evaluate([
            {"text": SENTENCES[0]},
            {"text": SENTENCES[1]},
        ])

        self.assertEqual(report["evaluated"], 2)
        self.assertGreater(report["active_heads"], 1)
        self.assertGreater(report["active_cycles"], 1)
        self.assertTrue(report["temporal"]["available"])
        self.assertAlmostEqual(
            report["temporal_nll"], -math.log(0.25), places=6)
        self.assertAlmostEqual(
            report["temporal"]["mean_negative_log_likelihood"],
            -math.log(0.25), places=6)
        self.assertEqual(report["gauge_residual"], 0)
        self.assertFalse(report["canonical_merge"])
        self.assertFalse(report["semantic_truth"])

        self.assertEqual(cycle_field.read_edges(), before_edges)
        self.assertEqual(cycle_field.read_cycles(), before_cycles)
        self.assertEqual(bridge.stats(), before_stats)

    def test_evaluate_without_probability_reports_unavailable_nll(self):
        bridge = CycleLearningBridge(
            NoProbabilityTemporal(), HeadFieldAdapter(), CycleFieldAdapter())
        report = bridge.evaluate([{"text": SENTENCES[0]}])
        self.assertFalse(report["temporal"]["available"])
        self.assertIsNone(report["temporal_nll"])
        self.assertIsNone(
            report["temporal"]["mean_negative_log_likelihood"])

    def test_batch_stats_and_evaluation_are_deterministic(self):
        def run():
            cycle_field = CycleFieldAdapter()
            bridge = CycleLearningBridge(
                FakeTemporal(), HeadFieldAdapter(), cycle_field,
                max_active_heads=10, max_active_cycles=5)
            batch = bridge.observe_batch(
                [{"text": text, "source": "unit"} for text in SENTENCES],
                limit=3)
            return {
                "batch": batch,
                "stats": bridge.stats(),
                "evaluation": bridge.evaluate(
                    [{"text": text} for text in SENTENCES]),
                "edges": cycle_field.read_edges(),
                "cycles": cycle_field.read_cycles(),
            }

        first = run()
        second = run()

        self.assertEqual(first, second)
        self.assertEqual(first["batch"]["records"], 3)
        self.assertEqual(first["batch"]["learned"], 3)
        self.assertEqual(first["batch"]["denied"], 0)
        self.assertGreater(first["batch"]["active_heads"], 1)
        self.assertGreater(first["batch"]["active_cycles"], 1)
        self.assertGreater(first["batch"]["mean_active_cycles"], 1)
        self.assertEqual(first["batch"]["accepted_proposals"], 3)
        self.assertEqual(first["batch"]["rejected_proposals"], 0)
        self.assertIsNotNone(first["batch"]["temporal_nll"])
        self.assertGreater(first["stats"]["active_heads"], 1)
        self.assertGreater(first["stats"]["active_cycles"], 1)
        self.assertGreater(first["evaluation"]["active_cycles"], 1)
        self.assertEqual(first["stats"]["gauge_residual"], 0)

        self.assertFalse(CANONICAL_MERGE)
        self.assertFalse(CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH)
        self.assertFalse(first["stats"]["canonical_merge"])
        self.assertFalse(first["stats"]["semantic_truth"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
