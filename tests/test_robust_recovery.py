"""Robust gauge-flat recovery tests."""
import random
import unittest

from nanlmind_hermes.robust_recovery import (
    flat_recovery_trial,
    gradient_field,
    nonflat_control,
    two_hop_median,
)


class RobustRecoveryTests(unittest.TestCase):
    def test_exact_flat_field_is_unchanged(self):
        rng = random.Random(4)
        theta = [rng.uniform(-2, 2) for _ in range(12)]
        truth = gradient_field(theta)
        recovered = two_hop_median(12, truth)
        self.assertAlmostEqual(
            sum(abs(recovered[key] - truth[key]) for key in truth), 0.0, places=10)

    def test_median_beats_mean_and_raw_under_corruption(self):
        for seed in (101, 102, 103):
            result = flat_recovery_trial(16, 0.2, seed)
            self.assertLess(result["median_mae"], result["mean_mae"])
            self.assertLess(result["median_mae"], result["raw_mae"])

    def test_nonflat_field_is_destroyed_by_median_filter(self):
        result = nonflat_control(16, 5)
        self.assertGreater(result["median_mae"], 0.2)

    def test_orientation_is_antisymmetric(self):
        values = {(0, 1): 2.0, (0, 2): 3.0, (1, 2): 1.0}
        recovered = two_hop_median(3, values)
        # The path 0 -> 1 -> 2 is 2+1=3, matching the direct edge.
        self.assertAlmostEqual(recovered[(0, 2)], 3.0, places=10)


if __name__ == "__main__":
    unittest.main()
