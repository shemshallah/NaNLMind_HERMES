"""Exact pair-graph spectrum and ideal Grover checks."""
import math
import unittest

from nanlmind_hermes.pair_quantum import (
    best_grover_queries,
    grover_probability,
    johnson_degree,
    johnson_spectrum,
    pair_count,
    return_probability,
    revival_period,
)


class PairQuantumTests(unittest.TestCase):
    def test_k256_spectrum_and_degree(self):
        spectrum = johnson_spectrum(256)
        self.assertEqual(spectrum, [(508, 1), (252, 255), (-2, 32384)])
        self.assertEqual(johnson_degree(256), 508)
        self.assertEqual(sum(multiplicity for _value, multiplicity in spectrum),
                         pair_count(256))

    def test_exact_revival_at_pi(self):
        self.assertAlmostEqual(return_probability(256, revival_period(256)), 1.0,
                               places=12)
        self.assertAlmostEqual(return_probability(256, math.pi / 2),
                               0.9688735101883892, places=10)

    def test_small_johnson_spectrum(self):
        self.assertEqual(johnson_spectrum(4), [(4, 1), (0, 3), (-2, 2)])
        self.assertEqual(sum(value * multiplicity
                             for value, multiplicity in johnson_spectrum(4)), 0)

    def test_grover_optimum_matches_pair_space(self):
        queries, probability = best_grover_queries(32640)
        self.assertEqual(queries, 141)
        self.assertGreater(probability, 0.9999)
        self.assertAlmostEqual(grover_probability(32640, 141),
                               0.9999810166767039, places=12)
        self.assertEqual((32640 + 1) / 2, 16320.5)

    def test_grover_small_case_is_bounded(self):
        self.assertLessEqual(grover_probability(2, 1), 1.0)
        self.assertAlmostEqual(grover_probability(4, 1), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
