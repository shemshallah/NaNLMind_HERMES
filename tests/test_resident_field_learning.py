"""Tests for independent 256-head field learning."""
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
from nanlmind_hermes.resident.field_learning import (
    CANONICAL_MERGE,
    HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH,
    FieldLearner,
    HeadFieldAdapter,
    ResidentCellFieldAdapter,
    activation_entropy,
)
from nanlmind_hermes.temporal_lm import TemporalMirrorLM

SENTENCES = [
    "A penguin swims in cold water.",
    "A river carries water to the sea.",
    "A bird can fly over the river.",
]


class FakeTemporal:
    """Small deterministic temporal model with observe() and _prob()."""

    def __init__(self, probability: float = 0.25):
        self.probability = probability
        self.vocab: dict[str, int] = {}
        self.observed: list[str] = []

    def observe(self, text: str) -> bool:
        if str(text).strip().endswith("?"):
            return False
        self.observed.append(str(text))
        return True

    def _prob(self, context2: str | None, context1: str, word: str) -> float:
        return self.probability


class NoProbabilityTemporal:
    def __init__(self):
        self.observed: list[str] = []

    def observe(self, text: str) -> bool:
        self.observed.append(str(text))
        return True


class TamperingField(HeadFieldAdapter):
    """Stages a valid proposal, then exposes a corrupted staged copy."""

    def __init__(self):
        super().__init__()
        self.propose_calls = 0
        self.activate_calls = 0
        self.apply_calls = 0

    def propose(self, update):
        self.propose_calls += 1
        proposal = super().propose(update)
        tampered = dict(proposal)
        tampered["heads"] = {
            head: 9_999.0 for head in proposal["heads"]
        }
        return tampered

    def activate(self, head, amount=1.0):
        self.activate_calls += 1
        return super().activate(head, amount)

    def apply_proposal(self, proposal):
        self.apply_calls += 1
        return super().apply_proposal(proposal)


class FieldLearningTests(unittest.TestCase):
    def test_independent_head_activity_exceeds_one_and_is_bounded(self):
        temporal = FakeTemporal()
        field = HeadFieldAdapter()
        learner = FieldLearner(
            temporal, field, max_active_heads=16, max_couplings=24)

        result = learner.observe(SENTENCES[0])

        self.assertTrue(result["accepted"])
        self.assertTrue(result["learned"])
        self.assertEqual(result["temporal_learned"], True)
        self.assertGreater(result["active_heads"], 1)
        self.assertGreater(len(result["head_updates"]), 1)
        self.assertLessEqual(len(result["head_updates"]), 16)
        self.assertLessEqual(len(result["coupling_updates"]), 24)
        self.assertGreater(result["entropy"], 0.0)
        self.assertGreater(result["applied_heads"], 1)

        visited = {item["head"] for item in result["head_updates"]}
        deltas = {item["head"]: item["delta"] for item in result["head_updates"]}
        self.assertGreater(len(visited), 1)
        self.assertGreaterEqual(len(set(deltas.values())), 2)
        applied = [row for row in learner.ledger_rows() if row["applied"] > 0]
        self.assertGreater(len(applied), 1)
        self.assertEqual({row["head"] for row in applied}, visited)

        stats = learner.stats()
        self.assertGreater(stats["active_heads"], 1)
        self.assertGreater(stats["ledger_active_heads"], 1)
        self.assertGreater(stats["ledger_applied_heads"], 1)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertEqual(stats["accepted_proposals"], 1)
        self.assertEqual(stats["rejected_proposals"], 0)
        self.assertFalse(stats["canonical_merge"])
        self.assertFalse(stats["semantic_truth"])

        loss_rows = [row for row in learner.ledger_rows()
                     if row["loss_count"] > 0]
        self.assertTrue(loss_rows)
        for row in loss_rows:
            self.assertAlmostEqual(
                row["mean_loss"], -math.log(temporal.probability), places=6)

    def test_real_temporal_model_is_kept_and_field_learns_duplicates(self):
        temporal = TemporalMirrorLM()
        field = HeadFieldAdapter()
        learner = FieldLearner(temporal, field)

        first = learner.observe(SENTENCES[0])
        self.assertIs(learner.temporal, temporal)
        self.assertTrue(first["temporal_learned"])
        self.assertIn("penguin", temporal.vocab)
        before = field.read_all()

        duplicate = learner.observe(SENTENCES[0])
        self.assertFalse(duplicate["temporal_learned"])
        self.assertTrue(duplicate["accepted"])
        self.assertGreater(duplicate["active_heads"], 1)
        self.assertNotEqual(field.read_all(), before)
        self.assertEqual(learner.stats()["accepted_proposals"], 2)

    def test_field_coupling_grows_with_exposure(self):
        field = HeadFieldAdapter()
        learner = FieldLearner(FakeTemporal(), field)
        before = field.field_activation()

        result = learner.observe(SENTENCES[1])

        after = field.field_activation()
        self.assertTrue(result["accepted"])
        self.assertGreater(after, before)
        self.assertTrue(field.coupling)
        self.assertGreater(result["applied_coupling"], 0)
        self.assertGreater(result["field_activation"], before)
        for left, right in field.coupling:
            self.assertNotEqual(left, right)
            self.assertTrue(0 <= left < 256)
            self.assertTrue(0 <= right < 256)
        stats = learner.stats()
        self.assertGreater(stats["field_activation"], 0.0)
        self.assertGreater(stats["writes"], 0)

    def test_scratch_rejection_never_activates_the_main_field(self):
        temporal = FakeTemporal()
        field = TamperingField()
        learner = FieldLearner(temporal, field)
        before = field.read_all()
        before_activation = field.field_activation()

        result = learner.observe(SENTENCES[0])

        self.assertFalse(result["accepted"])
        self.assertFalse(result["learned"])
        self.assertEqual(result["reason"], "invariant_failure")
        self.assertFalse(result["invariants"]["staged_matches_submitted"])
        self.assertFalse(result["invariants"]["head_deltas_bounded"])
        self.assertTrue(result["invariants"]["main_field_unchanged"])
        self.assertEqual(field.propose_calls, 1)
        self.assertEqual(field.activate_calls, 0)
        self.assertEqual(field.apply_calls, 0)
        self.assertEqual(field.read_all(), before)
        self.assertEqual(field.field_activation(), before_activation)
        self.assertEqual(field.scratch, {})
        self.assertEqual(result["writes"], 0)

        stats = learner.stats()
        self.assertEqual(stats["accepted_proposals"], 0)
        self.assertEqual(stats["rejected_proposals"], 1)
        self.assertEqual(stats["writes"], 0)
        self.assertTrue(stats["last_rejection"]["invariants"][
            "staged_matches_submitted"] is False)

    def test_real_cells_keep_permanent_identity_and_bounded_writes(self):
        cells = ResidentCells.format_region(bytearray(REGION_BYTES))
        field = ResidentCellFieldAdapter(cells)
        learner = FieldLearner(
            FakeTemporal(), field, max_active_heads=12, max_couplings=12,
            max_writes_per_sentence=24)

        result = learner.observe(SENTENCES[0])

        self.assertTrue(result["accepted"])
        self.assertGreater(result["active_heads"], 1)
        self.assertLessEqual(result["writes"], 24)
        self.assertLessEqual(learner.writes, 24)
        self.assertEqual(result["applied_heads"], len(
            [value for value in field.read_all() if value > 0]))

        # Permanent head identity words are untouched for every head.
        for head in range(256):
            record = cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity, identity_address(head))

        # Pair metadata remains canonical; only additive rail values move.
        for (left, right), delta in field.coupling.items():
            self.assertGreater(delta, 0.0)
            state = cells.pair_state(left, right)
            self.assertEqual(state.canonical, canonical_address(left, right))
            self.assertEqual(state.mirror, reverse16(state.canonical))
            self.assertGreater(state.total_weight, 0)

        self.assertFalse(CANONICAL_MERGE)
        self.assertFalse(HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH)
        self.assertFalse(learner.stats()["canonical_merge"])
        self.assertFalse(hasattr(learner, "merge"))
        self.assertFalse(hasattr(field, "merge"))

    def test_evaluate_reports_independent_activity_and_temporal_nll(self):
        temporal = FakeTemporal(probability=0.2)
        field = HeadFieldAdapter()
        learner = FieldLearner(temporal, field)
        before_vector = field.read_all()
        before_stats = learner.stats()

        report = learner.evaluate([
            {"text": SENTENCES[0]},
            {"text": SENTENCES[1]},
        ])

        self.assertEqual(report["evaluated"], 2)
        self.assertGreater(report["active_heads"], 1)
        self.assertGreater(
            report["independent"]["mean_active_heads_per_record"], 1)
        self.assertGreater(report["independent"]["entropy_bits"], 0.0)
        self.assertGreater(report["independent"]["active_head_fraction"], 0.0)
        self.assertTrue(report["temporal"]["available"])
        self.assertEqual(report["temporal"]["tokens"], 17)
        self.assertAlmostEqual(
            report["temporal_nll"], -math.log(0.2), places=6)
        self.assertAlmostEqual(
            report["temporal"]["mean_negative_log_likelihood"],
            -math.log(0.2), places=6)
        self.assertFalse(report["semantic_truth"])
        self.assertFalse(report["canonical_merge"])

        # Evaluation is read-only and does not stage or apply proposals.
        self.assertEqual(field.read_all(), before_vector)
        after_stats = learner.stats()
        self.assertEqual(after_stats["sentences"], before_stats["sentences"])
        self.assertEqual(after_stats["accepted_proposals"],
                         before_stats["accepted_proposals"])
        self.assertEqual(after_stats["writes"], before_stats["writes"])

    def test_evaluate_without_probability_reports_unavailable_nll(self):
        field = HeadFieldAdapter()
        learner = FieldLearner(NoProbabilityTemporal(), field)
        report = learner.evaluate([{"text": SENTENCES[0]}])
        self.assertFalse(report["temporal"]["available"])
        self.assertIsNone(report["temporal_nll"])
        self.assertIsNone(
            report["temporal"]["mean_negative_log_likelihood"])

    def test_observe_batch_and_metrics_are_deterministic(self):
        def run():
            field = HeadFieldAdapter()
            learner = FieldLearner(FakeTemporal(), field)
            batch = learner.observe_batch(
                [{"text": text, "source": "unit"} for text in SENTENCES],
                limit=3)
            return {
                "batch": batch,
                "stats": learner.stats(),
                "evaluation": learner.evaluate(
                    [{"text": text} for text in SENTENCES]),
                "vector": field.read_all(),
                "coupling": dict(field.coupling),
            }

        first = run()
        second = run()

        self.assertEqual(first, second)
        self.assertEqual(first["batch"]["records"], 3)
        self.assertEqual(first["batch"]["learned"], 3)
        self.assertGreater(first["batch"]["active_heads"], 1)
        self.assertGreater(first["batch"]["mean_active_heads"], 1)
        self.assertGreater(first["batch"]["entropy"], 0.0)
        self.assertEqual(first["batch"]["accepted_proposals"], 3)
        self.assertEqual(first["batch"]["rejected_proposals"], 0)
        self.assertIsNotNone(first["batch"]["temporal_nll"])
        self.assertGreater(first["stats"]["active_heads"], 1)
        self.assertGreater(first["evaluation"]["active_heads"], 1)

    def test_zero_write_budget_rejects_before_staging(self):
        field = HeadFieldAdapter()
        learner = FieldLearner(
            FakeTemporal(), field, total_write_budget=0)
        result = learner.observe(SENTENCES[0])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "write_budget_exhausted")
        self.assertEqual(result["writes"], 0)
        self.assertEqual(field.read_all(), (0.0,) * 256)
        self.assertEqual(field.scratch, {})
        self.assertEqual(learner.writes, 0)
        self.assertEqual(learner.stats()["budget_rejections"], 1)

    def test_activation_entropy_is_deterministic_and_bounded(self):
        self.assertAlmostEqual(activation_entropy([1.0]), 0.0)
        self.assertAlmostEqual(
            activation_entropy([1.0, 1.0]), 1.0, places=12)
        self.assertAlmostEqual(
            activation_entropy([1.0, 1.0, 1.0, 1.0]), 2.0, places=12)
        self.assertEqual(activation_entropy([]), 0.0)
        with self.assertRaises(ValueError):
            activation_entropy([float("nan")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
