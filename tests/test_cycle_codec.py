"""Cycle-codec tests: gauge correction and erasure recovery."""
import math
import random
import unittest

from nanlmind_hermes.cycle_codec import (
    cycle_project,
    decode_cycle,
    encode,
    gauge_values,
    joint_decode,
    pairs,
    random_cycle_basis,
    random_linear_basis,
    robust_joint_decode,
)


def cosine(first, second):
    return (sum(a * b for a, b in zip(first, second))
            / (math.sqrt(sum(a * a for a in first)
                         * sum(b * b for b in second)) + 1e-12))


class CycleCodecTests(unittest.TestCase):
    def setUp(self):
        self.n = 16
        self.dimensions = 3
        self.columns, self.rows = random_cycle_basis(
            self.n, self.dimensions, seed=4)
        self.z = [1.0, -0.5, 0.7]
        self.cycle = encode(self.rows, self.z)
        rng = random.Random(5)
        self.gauge = gauge_values(
            self.n, [rng.uniform(-2, 2) for _ in range(self.n)])
        self.observed = [a + b for a, b in zip(self.cycle, self.gauge)]

    def test_cycle_basis_has_zero_divergence(self):
        edge_pairs = pairs(self.n)
        divergence = [0.0] * self.n
        for (first, second), row in zip(edge_pairs, self.rows):
            divergence[first] -= row[0]
            divergence[second] += row[0]
        self.assertLess(max(abs(value) for value in divergence), 1e-10)

    def test_clean_cycle_is_recovered_after_gauge_fit(self):
        mask = [True] * len(self.observed)
        recovered = decode_cycle(self.n, self.columns, self.observed, mask)
        self.assertGreater(cosine(recovered, self.z), 0.999)

    def test_erasure_recovery_is_high(self):
        rng = random.Random(7)
        mask = [rng.random() < 0.5 for _ in self.observed]
        recovered = decode_cycle(self.n, self.columns, self.observed, mask)
        self.assertGreater(cosine(recovered, self.z), 0.9)

    def test_joint_decoder_beats_nonrobust_gauge_under_spikes(self):
        rng = random.Random(9)
        observed = list(self.observed)
        for edge in range(len(observed)):
            if rng.random() < 0.2:
                observed[edge] += rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 4.0)
        mask = [True] * len(observed)
        nonrobust = decode_cycle(self.n, self.columns, observed, mask)
        robust, _potential = joint_decode(
            self.n, self.columns, observed, mask, iterations=6)
        self.assertGreater(cosine(robust, self.z), 0.7)
        self.assertGreater(cosine(robust, self.z), cosine(nonrobust, self.z))

    def test_ordinary_random_code_can_recover_too(self):
        ordinary_columns, ordinary_rows = random_linear_basis(
            self.n, self.dimensions, seed=41)
        ordinary_cycle = encode(ordinary_rows, self.z)
        observed = [a + b for a, b in zip(ordinary_cycle, self.gauge)]
        mask = [True] * len(observed)
        recovered, _potential = joint_decode(
            self.n, ordinary_columns, observed, mask, iterations=4)
        self.assertGreater(cosine(recovered, self.z), 0.9)

    def test_robust_path_is_finite(self):
        mask = [True] * len(self.observed)
        recovered, potential = robust_joint_decode(
            self.n, self.columns, self.observed, mask, iterations=2)
        self.assertEqual(len(recovered), self.dimensions)
        self.assertEqual(len(potential), self.n)
        self.assertTrue(all(math.isfinite(value) for value in recovered))


if __name__ == "__main__":
    unittest.main()
