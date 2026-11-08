"""Cycle-space tests: decomposition, fundamental fluxes, phase coherence."""
import cmath
import math
import random
import unittest

from nanlmind_hermes.cycle_space import (
    cycle_dimension,
    cycle_inner_product,
    decompose,
    divergence,
    edge_count,
    fundamental_fluxes,
    reconstruct,
    triangle_count,
    triple_coherence,
)


class CycleSpaceTests(unittest.TestCase):
    def setUp(self):
        self.n = 8
        rng = random.Random(7)
        self.values = {}
        for first in range(self.n):
            for second in range(first + 1, self.n):
                self.values[(first, second)] = complex(
                    rng.uniform(-1, 1), rng.uniform(-1, 1))

    def test_dimensions(self):
        self.assertEqual(edge_count(8), 28)
        self.assertEqual(cycle_dimension(8), 21)
        self.assertEqual(triangle_count(8), 56)
        self.assertEqual(cycle_dimension(256), 32385)
        self.assertEqual(triangle_count(256), 2763520)

    def test_gradient_cycle_split_and_reconstruction(self):
        potential, gradient, cycle = decompose(self.n, self.values)
        self.assertAlmostEqual(cycle_inner_product(gradient, cycle), 0.0, places=8)
        rebuilt = reconstruct(self.n, potential, cycle)
        for key, value in self.values.items():
            self.assertAlmostEqual(abs(rebuilt[key] - value), 0.0, places=10)
        divergence_cycle = divergence(self.n, cycle)
        self.assertLess(max(abs(value) for value in divergence_cycle), 1e-9)
        self.assertEqual(len(fundamental_fluxes(self.n, self.values)),
                         math.comb(self.n - 1, 2))

    def test_phase_sensitive_triple_coherence(self):
        self.assertAlmostEqual(triple_coherence(1, 1, 1), 1.0)
        self.assertAlmostEqual(triple_coherence(1j, 1, 1), 0.0)
        self.assertAlmostEqual(triple_coherence(cmath.exp(1j * math.pi / 3),
                                                cmath.exp(1j * math.pi / 3),
                                                cmath.exp(1j * math.pi / 3)), 1.0)


if __name__ == "__main__":
    unittest.main()
