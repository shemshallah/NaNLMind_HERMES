"""Tests for full 32,385-cycle fixed-point training.

Every dense vector in this module is deliberately 32,385 entries long.  The
tests check that a 256-style window is refused, that sparse features still
produce full-width gradients, and that a root-star gauge moves pair lanes but
not cycle readouts or trained coefficients.
"""

from __future__ import annotations

import unittest

from nanlmind_hermes.resident.full_cycle_training import (
    CYCLE_BASIS,
    FULL_CYCLE_COUNT,
    FullCycleDimensionError,
    FullCycleFieldError,
    FullCycleLimitError,
    FullCycleMemory,
    FullCycleOverflowError,
    FullCycleTrainer,
    FullCycleTruncationError,
    FullCycleValidationError,
    FIXED_ONE,
    ROOT_HEAD,
    cycle_index,
    cycle_pair,
    deterministic_batches,
)


def dense_targets(offset: int = 0, step: int = 1) -> tuple[int, ...]:
    """A deterministic, non-zero 32,385-dimensional target vector."""
    return tuple((((index + offset) % 17) + 1) * (FIXED_ONE // 16) * step
                 for index in range(FULL_CYCLE_COUNT))


def dense_features(scale: int = FIXED_ONE) -> tuple[int, ...]:
    return tuple(scale for _ in range(FULL_CYCLE_COUNT))


def sparse_sample(indices=(0, 7, 255, 1000, 32384)):
    features = {}
    targets = {}
    for position, index in enumerate(indices):
        features[index] = FIXED_ONE
        targets[index] = (position + 1) * (FIXED_ONE // 2)
    return features, targets


class TruncatedMemory:
    """A hostile 256-window store that must never be accepted."""

    cycle_count = 256

    def __init__(self):
        self.generation = 0

    def read_cycle(self, index):
        return 0

    def commit_cycles(self, updates):
        self.generation += 1
        return self.generation


class FailingMemory(FullCycleMemory):
    """Reference memory with one injected partial commit failure."""

    def __init__(self):
        super().__init__()
        self.fail_next = False

    def commit_cycles(self, updates):
        if self.fail_next:
            self.fail_next = False
            normalized = self._normalize_cycle_updates(updates)
            if normalized:
                index, value = normalized[0]
                self._coefficients[index] = value
                self._generation += 1
            raise RuntimeError("injected partial commit failure")
        return super().commit_cycles(updates)


class DimensionTests(unittest.TestCase):
    def test_full_basis_is_32_385_and_indexing_round_trips(self):
        self.assertEqual(FULL_CYCLE_COUNT, 32385)
        self.assertEqual(len(CYCLE_BASIS), 32385)
        self.assertEqual(len(set(CYCLE_BASIS)), 32385)
        self.assertEqual(CYCLE_BASIS[0], (1, 2))
        self.assertEqual(CYCLE_BASIS[-1], (254, 255))
        for index in (0, 1, 253, 254, 255, 10_000, 32_384):
            pair = cycle_pair(index)
            self.assertEqual(cycle_index(*pair), index)
        with self.assertRaises(FullCycleDimensionError):
            cycle_pair(FULL_CYCLE_COUNT)
        with self.assertRaises(FullCycleValidationError):
            cycle_index(0, 1)

    def test_memory_allocates_exactly_the_full_basis(self):
        memory = FullCycleMemory()
        self.assertEqual(len(memory), FULL_CYCLE_COUNT)
        self.assertEqual(len(memory.weights), FULL_CYCLE_COUNT)
        self.assertEqual(len(memory.raw_words), FULL_CYCLE_COUNT)
        self.assertEqual(memory.dimension, FULL_CYCLE_COUNT)
        self.assertEqual(memory.validate_full_basis()["readable"],
                         FULL_CYCLE_COUNT)

    def test_256_style_memory_and_vectors_are_refused(self):
        with self.assertRaises(FullCycleTruncationError):
            FullCycleTrainer(TruncatedMemory())
        with self.assertRaises(FullCycleTruncationError):
            FullCycleMemory([0] * 256)
        trainer = FullCycleTrainer()
        with self.assertRaises(FullCycleTruncationError):
            trainer.validate_features([0] * 256)
        with self.assertRaises(FullCycleTruncationError):
            trainer.validate_targets([0] * 256)
        with self.assertRaises(FullCycleTruncationError):
            trainer.gradient([0] * 32385, [0] * 256)
        with self.assertRaises(FullCycleDimensionError):
            trainer.validate_features({FULL_CYCLE_COUNT: 1})

    def test_sparse_features_validate_against_full_basis(self):
        trainer = FullCycleTrainer()
        report = trainer.validate_features({0: 1, 32_384: -1})
        self.assertEqual(report["dimension"], FULL_CYCLE_COUNT)
        self.assertEqual(report["active_features"], 2)
        self.assertTrue(report["full_basis"])
        self.assertFalse(report["truncated"])


class GradientAndLossTests(unittest.TestCase):
    def test_gradient_has_full_shape_even_for_sparse_features(self):
        trainer = FullCycleTrainer()
        features, targets = sparse_sample()
        gradient = trainer.gradient(features, targets)
        self.assertIs(type(gradient), tuple)
        self.assertEqual(len(gradient), FULL_CYCLE_COUNT)
        non_zero = tuple(index for index, value in enumerate(gradient)
                         if value)
        self.assertEqual(non_zero, tuple(sorted(features)))
        for index in features:
            self.assertEqual(gradient[index], -targets[index])
        self.assertEqual(sum(1 for value in gradient if value), 5)

    def test_full_dense_gradient_has_32_385_entries(self):
        trainer = FullCycleTrainer()
        gradient = trainer.gradient(dense_features(), dense_targets())
        self.assertEqual(len(gradient), FULL_CYCLE_COUNT)
        self.assertTrue(all(value <= 0 for value in gradient))
        self.assertTrue(any(value < 0 for value in gradient))

    def test_loss_decreases_on_a_32_385_dimensional_target(self):
        trainer = FullCycleTrainer()
        targets = dense_targets()
        features = {index: FIXED_ONE
                    for index in range(0, FULL_CYCLE_COUNT, 257)}
        before = trainer.loss(features, targets)
        receipt = trainer.sgd_step(features, targets, FIXED_ONE // 4)
        after = trainer.loss(features, targets)
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["changed"])
        self.assertLess(after, before)
        self.assertEqual(receipt["dimension"], FULL_CYCLE_COUNT)
        self.assertFalse(receipt["truncated"])

    def test_ce_like_metric_also_decreases(self):
        trainer = FullCycleTrainer()
        features, targets = sparse_sample()
        before = trainer.ce_like_loss(features, targets)
        trainer.sgd_step(features, targets, FIXED_ONE)
        after = trainer.ce_like_loss(features, targets)
        self.assertLess(after, before)

    def test_one_step_reaches_small_targets_exactly(self):
        trainer = FullCycleTrainer()
        features, targets = sparse_sample()
        trainer.sgd_step(features, targets, FIXED_ONE)
        for index, target in targets.items():
            self.assertEqual(trainer.weight(index), target)
        self.assertEqual(trainer.loss(features, targets), 0.0)


class FullBasisCoverageTests(unittest.TestCase):
    def test_all_32_385_coefficients_are_trainable(self):
        trainer = FullCycleTrainer()
        targets = dense_targets(step=2)
        features = dense_features()
        receipt = trainer.sgd_step(features, targets, FIXED_ONE)
        self.assertEqual(receipt["weights_updated"], FULL_CYCLE_COUNT)
        self.assertEqual(receipt["active_features"], FULL_CYCLE_COUNT)
        weights = trainer.weights
        self.assertEqual(len(weights), FULL_CYCLE_COUNT)
        self.assertEqual(sum(1 for value in weights if value),
                         FULL_CYCLE_COUNT)
        coverage = trainer.validate_full_basis()
        self.assertTrue(coverage["full_basis"])
        self.assertEqual(coverage["nonzero"], FULL_CYCLE_COUNT)

    def test_freeze_and_unfreeze_cover_full_indices(self):
        trainer = FullCycleTrainer()
        features = {0: FIXED_ONE, 1: FIXED_ONE}
        targets = {0: 5 * FIXED_ONE, 1: 7 * FIXED_ONE}
        self.assertEqual(trainer.freeze(0), 1)
        self.assertTrue(trainer.is_frozen(0))
        trainer.sgd_step(features, targets, FIXED_ONE)
        self.assertEqual(trainer.weight(0), 0)
        self.assertEqual(trainer.weight(1), 7 * FIXED_ONE)
        self.assertEqual(trainer.unfreeze(0), 1)
        trainer.sgd_step(features, targets, FIXED_ONE)
        self.assertEqual(trainer.weight(0), 5 * FIXED_ONE)
        self.assertEqual(trainer.frozen, frozenset())


class GaugeTests(unittest.TestCase):
    @staticmethod
    def filled_memory():
        memory = FullCycleMemory()
        memory.commit_pairs([
            (0, 1, 10), (0, 2, 20), (1, 2, 30),
            (1, 3, -4), (2, 3, 8), (0, 3, 7),
        ])
        return memory

    def test_root_star_gauge_preserves_cycles_and_weights(self):
        memory = self.filled_memory()
        trainer = FullCycleTrainer(memory)
        features, targets = sparse_sample()
        trainer.sgd_step(features, targets, FIXED_ONE)
        before_features = trainer.cycle_features()
        before_weights = trainer.weights
        before_digest = trainer.weight_digest()
        before_root = memory.read_lane(0, 1)

        gauge = tuple(((index * 13) % 17) - 8 for index in range(256))
        receipt = trainer.apply_root_star_gauge(gauge)

        self.assertTrue(receipt["cycle_readout_unchanged"])
        self.assertTrue(receipt["weights_unchanged"])
        self.assertEqual(trainer.cycle_features(), before_features)
        self.assertEqual(trainer.weights, before_weights)
        self.assertEqual(trainer.weight_digest(), before_digest)
        self.assertEqual(memory.read_lane(0, 1) - before_root,
                         gauge[0] - gauge[1])
        self.assertEqual(trainer.cycle_readout(cycle_pair(0)[0],
                                               cycle_pair(0)[1]),
                         memory.cycle_readout(0))

    def test_gauged_and_ungauged_training_are_identical(self):
        gauge = tuple(((index * 7) % 23) - 11 for index in range(256))
        plain_memory = self.filled_memory()
        gauged_memory = self.filled_memory()
        plain = FullCycleTrainer(plain_memory)
        gauged = FullCycleTrainer(gauged_memory)
        gauged.apply_root_star_gauge(gauge)

        plain_features = plain.cycle_features()
        gauged_features = gauged.cycle_features()
        self.assertEqual(plain_features, gauged_features)

        targets = dense_targets()
        features = {index: plain_features[index]
                    for index in range(0, FULL_CYCLE_COUNT, 373)}
        plain_receipt = plain.sgd_step(features, targets, FIXED_ONE // 8)
        gauged_receipt = gauged.sgd_step(features, targets, FIXED_ONE // 8)

        self.assertEqual(plain.weights, gauged.weights)
        self.assertEqual(plain.weight_digest(), gauged.weight_digest())
        self.assertEqual(plain_receipt["loss_units_before"],
                         gauged_receipt["loss_units_before"])
        self.assertEqual(plain_receipt["loss_units_after"],
                         gauged_receipt["loss_units_after"])


class ReproducibilityTests(unittest.TestCase):
    @staticmethod
    def samples():
        result = []
        for position in range(12):
            index = (position * 2699) % FULL_CYCLE_COUNT
            result.append((
                {index: FIXED_ONE, (index + 1) % FULL_CYCLE_COUNT:
                 FIXED_ONE // 2},
                {index: (position + 1) * (FIXED_ONE // 4)},
            ))
        return result

    def test_deterministic_batches_repeat_exactly(self):
        samples = self.samples()
        first = deterministic_batches(samples, 5, seed=41)
        second = deterministic_batches(samples, 5, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(batch) for batch in first), len(samples))
        trainer = FullCycleTrainer()
        self.assertEqual(trainer.deterministic_batches(samples, 5, seed=41),
                         first)

    def test_repeated_training_runs_are_reproducible(self):
        def run():
            memory = FullCycleMemory()
            trainer = FullCycleTrainer(memory)
            batches = trainer.deterministic_batches(
                self.samples(), 4, seed=41)
            receipts = [trainer.train_batch(batch, lr=FIXED_ONE // 4)
                        for batch in batches]
            return (trainer.weight_digest(), trainer.weights,
                    trainer.stats(), receipts)

        first = run()
        second = run()
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2]["accepted_steps"],
                         second[2]["accepted_steps"])
        self.assertEqual([receipt["loss_units_after"]
                          for receipt in first[3]],
                         [receipt["loss_units_after"]
                          for receipt in second[3]])

    def test_batch_step_advances_generation_once(self):
        trainer = FullCycleTrainer()
        batch = self.samples()[:5]
        receipt = trainer.train_batch(batch, lr=FIXED_ONE // 4)
        self.assertTrue(receipt["batch"])
        self.assertEqual(receipt["generation_before"], 0)
        self.assertEqual(receipt["generation_after"], 1)
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(trainer.stats()["samples_seen"], 5)


class BoundsAndRollbackTests(unittest.TestCase):
    def test_bounded_learning_rate_and_steps(self):
        trainer = FullCycleTrainer(max_steps=2, max_steps_per_call=1)
        features, targets = sparse_sample()
        with self.assertRaises(FullCycleLimitError):
            trainer.sgd_step(features, targets, 17 * FIXED_ONE)
        with self.assertRaises(FullCycleValidationError):
            trainer.sgd_step(features, targets, 0.5)
        with self.assertRaises(FullCycleLimitError):
            trainer.sgd_step(features, targets, FIXED_ONE, steps=2)
        self.assertEqual(trainer.generation, 0)
        self.assertEqual(trainer.weight(0), 0)

    def test_excess_step_magnitude_is_denied_without_mutation(self):
        trainer = FullCycleTrainer(max_step_units=10)
        features, targets = sparse_sample()
        receipt = trainer.try_sgd_step(features, targets, FIXED_ONE)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "FullCycleOverflowError")
        self.assertEqual(trainer.generation, 0)
        self.assertEqual(sum(1 for value in trainer.weights if value), 0)

    def test_step_limit_is_enforced(self):
        trainer = FullCycleTrainer(max_steps=1)
        features, targets = sparse_sample()
        first = trainer.sgd_step(features, targets, FIXED_ONE // 8)
        self.assertTrue(first["accepted"])
        second = trainer.try_sgd_step(features, targets, FIXED_ONE // 8)
        self.assertFalse(second["accepted"])
        self.assertEqual(second["reason"], "FullCycleLimitError")

    def test_no_op_step_does_not_advance_generation(self):
        trainer = FullCycleTrainer()
        features, targets = sparse_sample()
        first = trainer.sgd_step(features, targets, FIXED_ONE)
        second = trainer.sgd_step(features, targets, FIXED_ONE)
        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(second["generation_before"],
                         second["generation_after"])
        self.assertEqual(trainer.generation, 1)
        self.assertEqual(trainer.no_op_steps, 1)

    def test_injected_commit_failure_rolls_back_full_vector(self):
        memory = FailingMemory()
        trainer = FullCycleTrainer(memory)
        features, targets = sparse_sample()
        before_weights = trainer.weights
        before_generation = trainer.generation
        memory.fail_next = True

        receipt = trainer.try_sgd_step(features, targets, FIXED_ONE)

        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "FullCycleFieldError")
        self.assertEqual(trainer.weights, before_weights)
        self.assertEqual(trainer.generation, before_generation)
        self.assertEqual(memory.weights, before_weights)
        self.assertEqual(trainer.accepted_steps, 0)
        self.assertFalse(any(trainer.weights))

    def test_trainer_snapshot_restore_round_trips(self):
        trainer = FullCycleTrainer()
        features, targets = sparse_sample()
        trainer.sgd_step(features, targets, FIXED_ONE // 2)
        snapshot = trainer.snapshot()
        before = trainer.weights
        trainer.sgd_step(features, targets, FIXED_ONE // 2)
        self.assertNotEqual(trainer.weights, before)
        trainer.restore(snapshot)
        self.assertEqual(trainer.weights, before)
        self.assertEqual(trainer.weight_digest(),
                         snapshot["weights_digest"])

    def test_raw_words_preserve_negative_payload_identity(self):
        memory = FullCycleMemory()
        memory.commit_cycles({5: -1})
        self.assertEqual(memory.read_cycle(5), -1)
        self.assertEqual(memory.read_word(5), 0xFFFFFFFF)
        self.assertEqual(memory.raw_words[5], 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
