"""Fixed-point supervised cycle-readout training tests.

The fake field below is intentionally small and auditable.  The real-cell test
uses the documented ``ResidentCells`` pair rails and verifies that both mirror
lanes are written with one coherent generation.
"""
from __future__ import annotations

import unittest

from nanlmind_hermes.resident.cells import (
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    REGION_BYTES,
    ResidentCells,
)
from nanlmind_hermes.resident.cycle_training import (
    PAYLOAD_MAX,
    CellsLaneField,
    CycleMirrorError,
    CycleObservableLaneField,
    CycleOverflowError,
    CycleTrainer,
    CycleValidationError,
    FIXED_ONE,
    IntegerLaneField,
    fixed_ratio,
    synthetic_gauge_shift_experiment,
)

try:  # The raw-u32 cycle observable is an independently developed sibling.
    from nanlmind_hermes.resident.cycle_observable import CycleField, WORD_MASK
except ImportError:  # pragma: no cover - keeps this module usable alone
    CycleField = None
    WORD_MASK = (1 << 32) - 1


class FakeCycleField:
    """Minimal raw-lane fake with explicit failure injection."""

    def __init__(self):
        self.lanes: dict[tuple[int, int], int] = {}
        self.generation = 0
        self.commit_calls = 0
        self.fail_next = False

    def read_lane(self, source: int, target: int) -> int:
        return self.lanes.get((source, target), 0)

    def commit_pairs(self, updates) -> int:
        self.commit_calls += 1
        for index, (left, right, value) in enumerate(updates):
            self.lanes[(left, right)] = value
            self.lanes[(right, left)] = -value
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("injected partial commit failure")
        self.generation += 1
        return self.generation

    def snapshot(self):
        return dict(self.lanes), self.generation

    def restore(self, snapshot) -> None:
        lanes, generation = snapshot
        self.lanes = dict(lanes)
        self.generation = generation

    def apply_gauge(self, gauge) -> None:
        updates = []
        for left in range(256):
            for right in range(left + 1, 256):
                updates.append(
                    (left, right,
                     self.lanes.get((left, right), 0)
                     + gauge[left] - gauge[right]))
        self.commit_pairs(updates)


class BrokenMirrorField:
    """Every directed read returns the same positive value."""

    generation = 0

    def read_lane(self, source: int, target: int) -> int:
        return 7

    def commit_pairs(self, updates) -> int:
        self.generation += 1
        return self.generation


def make_trainer(field=None, **options):
    field = field if field is not None else FakeCycleField()
    settings = {
        "learning_rate": FIXED_ONE,
        "normalize_gradient": True,
        "max_step_units": 1 << 26,
    }
    settings.update(options)
    return field, CycleTrainer(field, **settings)


class RawReadoutTests(unittest.TestCase):
    def test_predict_and_gradient_are_raw_integers_with_correct_sign(self):
        field, trainer = make_trainer()
        self.assertEqual(trainer.predict(2, 3), 0)
        self.assertIs(type(trainer.predict(2, 3)), int)

        target = 6 * FIXED_ONE
        gradient = trainer.gradient(2, 3, target)
        self.assertIs(type(gradient), int)
        self.assertLess(gradient, 0)
        self.assertEqual(gradient, -target)
        self.assertGreater(trainer.gradient((2, 3, -target)), 0)

        result = trainer.step((2, 3, target))
        self.assertTrue(result["accepted"])
        self.assertEqual(trainer.predict(2, 3), target)
        self.assertEqual(trainer.gradient(2, 3, target), 0)
        self.assertIs(type(trainer.loss((2, 3, target))), float)

    def test_step_decreases_loss_and_updates_three_mirror_edges(self):
        field, trainer = make_trainer()
        observation = (3, 4, 6 * FIXED_ONE)
        before = trainer.loss_units(observation)
        result = trainer.step(observation)
        after = trainer.loss_units(observation)

        self.assertTrue(result["changed"])
        self.assertEqual(result["pairs_updated"], 3)
        self.assertLess(after, before)
        self.assertEqual(after, 0)
        self.assertEqual(result["generation_before"], 0)
        self.assertEqual(result["generation_after"], 1)
        self.assertEqual(field.generation, 1)

        expected = 2 * FIXED_ONE  # 6 / 3, one distributed step
        for source, target in ((0, 3), (3, 4), (4, 0)):
            self.assertEqual(field.read_lane(source, target), expected)
            self.assertEqual(field.read_lane(target, source), -expected)
        self.assertEqual(trainer.predict(3, 4), 6 * FIXED_ONE)

    def test_default_step_matches_native_three_edge_gradient_update(self):
        field = FakeCycleField()
        trainer = CycleTrainer(field, learning_rate=fixed_ratio(1, 4))
        target = 4 * FIXED_ONE
        before = trainer.loss_units((5, 6, target))

        result = trainer.step((5, 6, target))

        self.assertTrue(result["accepted"])
        self.assertFalse(trainer.normalize_gradient)
        # gradient is -4F, so -lr*gradient is exactly +1F on each edge.
        for source, target_head in ((0, 5), (5, 6), (6, 0)):
            self.assertEqual(field.read_lane(source, target_head), FIXED_ONE)
            self.assertEqual(field.read_lane(target_head, source), -FIXED_ONE)
        self.assertEqual(trainer.predict(5, 6), 3 * FIXED_ONE)
        self.assertLess(trainer.loss_units((5, 6, target)), before)

    def test_finite_difference_check_is_exact_and_read_only(self):
        field, trainer = make_trainer()
        observation = (2, 3, 12345)
        generation = field.generation
        calls = field.commit_calls

        check = trainer.finite_difference_check(observation, delta=17)

        self.assertTrue(check["exact"])
        self.assertEqual(check["error_units"], 0)
        self.assertEqual(
            check["gradient_units"],
            (trainer.predict(2, 3) - observation[2]) * FIXED_ONE)
        self.assertEqual(check["generation"], generation)
        self.assertEqual(field.generation, generation)
        self.assertEqual(field.commit_calls, calls)


class BatchAndGaugeTests(unittest.TestCase):
    def test_batch_is_deterministic_and_generation_advances_once(self):
        first_field, first = make_trainer()
        second_field, second = make_trainer()
        batch = [
            (1, 2, 6 * FIXED_ONE),
            (3, 4, -9 * FIXED_ONE),
            (5, 6, 12 * FIXED_ONE),
        ]

        first_result = first.step(batch)
        second_result = second.step(list(reversed(batch)))

        self.assertEqual(first_field.lanes, second_field.lanes)
        self.assertEqual(first_field.generation, 1)
        self.assertEqual(second_field.generation, 1)
        self.assertEqual(first_result["generation_after"], 1)
        self.assertEqual(second_result["generation_after"], 1)
        self.assertEqual(first_result["loss_units_after"],
                         second_result["loss_units_after"])
        for left, right, target in batch:
            self.assertEqual(first.predict(left, right), target)
            self.assertEqual(second.predict(left, right), target)

    def test_unseen_gauge_preserves_cycle_prediction_exactly(self):
        field, trainer = make_trainer()
        target = 6 * FIXED_ONE
        trainer.step((3, 4, target))
        before_prediction = trainer.predict(3, 4)
        before_lane = field.read_lane(3, 4)
        other_before = trainer.predict(7, 8)

        gauge = tuple((index * 11 % 19 - 9) * FIXED_ONE
                      for index in range(256))
        field.apply_gauge(gauge)

        self.assertEqual(trainer.predict(3, 4), before_prediction)
        self.assertEqual(trainer.predict(3, 4), target)
        self.assertEqual(trainer.predict(7, 8), other_before)
        self.assertEqual(field.read_lane(3, 4) - before_lane,
                         gauge[3] - gauge[4])
        self.assertNotEqual(field.read_lane(3, 4), before_lane)

    def test_repeated_runs_are_reproducible(self):
        def run():
            field, trainer = make_trainer()
            observations = [
                (1, 2, 9 * FIXED_ONE),
                (3, 4, -12 * FIXED_ONE),
                (5, 6, 15 * FIXED_ONE),
            ]
            receipts = []
            for observation in observations:
                receipts.append(trainer.step(observation))
            return field, trainer, receipts

        first_field, first, first_receipts = run()
        second_field, second, second_receipts = run()

        self.assertEqual(first_field.lanes, second_field.lanes)
        self.assertEqual(first_field.generation, second_field.generation)
        self.assertEqual(first.stats()["accepted_steps"],
                         second.stats()["accepted_steps"])
        for first_receipt, second_receipt in zip(first_receipts,
                                                second_receipts):
            self.assertEqual(first_receipt["loss_units_before"],
                             second_receipt["loss_units_before"])
            self.assertEqual(first_receipt["loss_units_after"],
                             second_receipt["loss_units_after"])
            self.assertEqual(first_receipt["max_abs_step_units"],
                             second_receipt["max_abs_step_units"])


class DenialAndRollbackTests(unittest.TestCase):
    def test_invalid_observations_do_not_touch_the_field(self):
        field, trainer = make_trainer(max_batch_size=2)
        invalid = [
            (4, 4, 1),
            (0, 4, 1),
            (1, 1, 1),
            (1, 2, True),
            (1, 2, 3.5),
            (1, 2, 3, 4),
        ]
        for observation in invalid:
            with self.assertRaises(CycleValidationError):
                trainer.step(observation)
        with self.assertRaises(CycleOverflowError):
            trainer.step((1, 2, PAYLOAD_MAX + 1))
        with self.assertRaises(CycleValidationError):
            trainer.step([(1, 2, 3), (3, 4, 5), (5, 6, 7)])

        self.assertEqual(field.lanes, {})
        self.assertEqual(field.generation, 0)
        self.assertEqual(field.commit_calls, 0)

    def test_try_step_denial_does_not_advance_generation(self):
        field, trainer = make_trainer(max_step_units=1000)
        result = trainer.try_step((1, 2, 6 * FIXED_ONE))

        self.assertFalse(result["accepted"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["reason"], "CycleOverflowError")
        self.assertEqual(result["generation"], 0)
        self.assertEqual(field.generation, 0)
        self.assertEqual(field.lanes, {})
        self.assertEqual(trainer.rejected_steps, 1)

    def test_proposed_payload_overflow_is_denied_without_mutation(self):
        field = IntegerLaneField(payload_min=-100, payload_max=100)
        trainer = CycleTrainer(
            field,
            learning_rate=FIXED_ONE,
            normalize_gradient=True,
            max_step_units=2000,
            payload_min=-100,
            payload_max=100,
            target_min=-1000,
            target_max=1000,
        )
        # The directed triangle is (0->3)=90, (3->4)=90, (4->0)=90.
        field.commit_pairs([(0, 3, 90), (3, 4, 90), (0, 4, -90)])
        before = field.snapshot()
        before_generation = field.generation

        result = trainer.try_step((3, 4, -1000))

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "CycleOverflowError")
        self.assertEqual(field.snapshot(), before)
        self.assertEqual(field.generation, before_generation)

    def test_out_of_range_stored_lane_is_denied(self):
        field = FakeCycleField()
        field.lanes[(0, 1)] = PAYLOAD_MAX + 1
        field.lanes[(1, 0)] = -(PAYLOAD_MAX + 1)
        trainer = CycleTrainer(field)
        with self.assertRaises(CycleOverflowError):
            trainer.predict(1, 2)

    def test_field_failure_rolls_back_lanes_and_generation(self):
        field, trainer = make_trainer()
        before_lanes = dict(field.lanes)
        field.fail_next = True

        result = trainer.try_step((1, 2, 6 * FIXED_ONE))

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "CycleFieldError")
        self.assertEqual(field.lanes, before_lanes)
        self.assertEqual(field.generation, 0)
        self.assertEqual(trainer.accepted_steps, 0)

        accepted = trainer.step((1, 2, 6 * FIXED_ONE))
        self.assertTrue(accepted["accepted"])
        self.assertEqual(trainer.predict(1, 2), 6 * FIXED_ONE)

    def test_float_learning_rate_is_refused(self):
        with self.assertRaises(CycleValidationError):
            CycleTrainer(FakeCycleField(), learning_rate=0.1)
        self.assertEqual(fixed_ratio(1, 2), FIXED_ONE // 2)


class MirrorArithmeticTests(unittest.TestCase):
    def test_inconsistent_mirrors_are_rejected(self):
        trainer = CycleTrainer(BrokenMirrorField())
        with self.assertRaises(CycleMirrorError):
            trainer.predict(1, 2)

    def test_canonical_commit_writes_exact_mirror_lanes(self):
        field = IntegerLaneField()
        field.commit_pairs([(2, 5, 12345), (0, 7, -99)])
        self.assertEqual(field.read_lane(2, 5), 12345)
        self.assertEqual(field.read_lane(5, 2), -12345)
        self.assertEqual(field.read_lane(0, 7), -99)
        self.assertEqual(field.read_lane(7, 0), 99)
        self.assertEqual(field.generation, 1)

    def test_real_cells_rails_are_written_with_one_generation(self):
        cells = ResidentCells(bytearray(REGION_BYTES), initialize=True)
        trainer = CycleTrainer(cells, learning_rate=FIXED_ONE,
                               normalize_gradient=True)
        field = trainer.field

        self.assertIsInstance(field, CellsLaneField)
        self.assertEqual(trainer.predict(3, 5), 0)
        result = trainer.step((3, 5, 9 * FIXED_ONE))

        self.assertTrue(result["accepted"])
        self.assertEqual(trainer.predict(3, 5), 9 * FIXED_ONE)
        self.assertEqual(field.read_lane(3, 5), -field.read_lane(5, 3))
        self.assertEqual(field.read_lane(3, 5), 3 * FIXED_ONE)
        self.assertEqual(field.generation, 1)
        self.assertEqual(field.cycle_observable(3, 5), 9 * FIXED_ONE)

        state = cells.pair_state(3, 5)
        self.assertEqual(state.forward.weight, 3 * FIXED_ONE)
        self.assertEqual(state.reverse.weight, -3 * FIXED_ONE)
        self.assertEqual(state.forward.generation, state.reverse.generation)
        self.assertGreater(state.forward.generation, 0)
        self.assertEqual(cells.read_rail(3, 5, RAIL_S_PLUS).weight,
                         3 * FIXED_ONE)
        self.assertEqual(cells.read_rail(3, 5, RAIL_S_MINUS).weight,
                         -3 * FIXED_ONE)

        # A future ResidentCells.cycle_observable, if present, must agree with
        # the trainer's raw integer readout.
        if hasattr(cells, "cycle_observable"):
            self.assertEqual(int(cells.cycle_observable(3, 5)),
                             trainer.predict(3, 5))


class CycleObservableIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(CycleField is not None,
                         "cycle_observable.CycleField is unavailable")
    def test_cycle_trainer_uses_real_cycle_observable_field(self):
        cells = ResidentCells(bytearray(REGION_BYTES), initialize=True)
        raw_field = CycleField(cells)
        trainer = CycleTrainer(raw_field, learning_rate=FIXED_ONE,
                               normalize_gradient=True)

        self.assertIsInstance(trainer.field, CycleObservableLaneField)
        target = 9 * FIXED_ONE
        result = trainer.step((3, 5, target))

        self.assertTrue(result["accepted"])
        self.assertEqual(trainer.predict(3, 5), target)
        self.assertEqual(raw_field.get_oriented(3, 5), 3 * FIXED_ONE)
        self.assertEqual(raw_field.get_oriented(5, 3),
                         (-3 * FIXED_ONE) & WORD_MASK)
        self.assertEqual(
            (raw_field.get_oriented(3, 5)
             + raw_field.get_oriented(5, 3)) & WORD_MASK, 0)
        self.assertEqual(raw_field.cycle(3, 5), target)
        self.assertEqual(trainer.field.cycle_observable(3, 5), target)

        before = trainer.predict(3, 5)
        gauge = tuple(
            ((((index * 7 + 3) % 13) - 6) * FIXED_ONE) & WORD_MASK
            for index in range(256))
        raw_field.gauge_shift(gauge)
        self.assertEqual(trainer.predict(3, 5), before)

        # A raw CycleField write touches only one direction; the trainer must
        # refuse the now non-mirror pair rather than silently projecting it.
        raw_field.set_oriented(0, 6, 5)
        with self.assertRaises(CycleMirrorError):
            trainer.predict(6, 7)


class SyntheticExperimentTests(unittest.TestCase):
    def test_cycle_readout_is_invariant_under_unseen_gauge(self):
        result = synthetic_gauge_shift_experiment(samples=8)

        self.assertEqual(result["train_samples"], 8)
        self.assertEqual(result["cycle"]["train_exact"], 8)
        self.assertEqual(result["cycle"]["test_exact"], 8)
        self.assertTrue(result["cycle"]["predictions_unchanged"])
        self.assertEqual(result["cycle"]["test_error_units"], 0)

        self.assertEqual(result["raw_edge"]["train_exact"], 8)
        self.assertLess(result["raw_edge"]["test_exact"], 8)
        self.assertTrue(result["raw_edge"]["predictions_changed"])
        self.assertGreater(result["raw_edge"]["test_error_units"], 0)

        self.assertFalse(result["claims"]["language_model_win"])
        self.assertFalse(result["claims"]["held_out_samples"])
        self.assertTrue(result["claims"]["cycle_gauge_invariance"])
        self.assertTrue(result["claims"]["cycle_exact_reconstruction"])

    def test_experiment_is_deterministic(self):
        first = synthetic_gauge_shift_experiment(samples=6, seed=29)
        second = synthetic_gauge_shift_experiment(samples=6, seed=29)
        self.assertEqual(first["cycle"], second["cycle"])
        self.assertEqual(first["raw_edge"], second["raw_edge"])
        self.assertEqual(first["targets"], second["targets"])


if __name__ == "__main__":
    unittest.main()
