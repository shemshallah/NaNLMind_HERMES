"""Tests for classical gauge-invariant phase memory."""
import math
import random
import unittest

from nanlmind_hermes.phase_memory import (
    PhaseMemory,
    add_noise,
    complex_gauge_score,
    evaluate,
    random_key,
    real_gauge_score,
    rotate,
)


class PhaseMemoryTests(unittest.TestCase):
    def test_global_phase_cancels_for_gauge_score(self):
        rng = random.Random(7)
        pattern = random_key(32, rng)
        query = rotate(pattern, 2.1)
        self.assertAlmostEqual(complex_gauge_score(query, pattern), 32.0 ** 2, places=10)
        self.assertNotAlmostEqual(
            sum((q * p.conjugate()).real for q, p in zip(query, pattern)),
            32.0 ** 2, places=6)

    def test_real_cosine_sine_score_matches_complex(self):
        rng = random.Random(11)
        first = random_key(48, rng)
        second = random_key(48, rng)
        self.assertAlmostEqual(complex_gauge_score(first, second),
                               real_gauge_score(first, second), places=10)
        noisy = add_noise(rotate(first, 0.7), rng, 0.3)
        self.assertAlmostEqual(complex_gauge_score(noisy, second),
                               real_gauge_score(noisy, second), places=10)

    def test_gauge_retrieval_beats_naive_without_noise(self):
        result = evaluate(dimensions=64, patterns=64, queries=120, seed=3)
        self.assertEqual(result["gauge_accuracy"], 1.0)
        self.assertLess(result["naive_accuracy"], 0.9)
        self.assertEqual(result["real_channel_agreement"], 1.0)

    def test_phase_noise_eventually_breaks_recall(self):
        low = evaluate(dimensions=64, patterns=64, queries=80, seed=5,
                       noise_std=0.25)
        high = evaluate(dimensions=64, patterns=64, queries=80, seed=5,
                        noise_std=2.0)
        self.assertGreater(low["gauge_accuracy"], 0.9)
        self.assertLess(high["gauge_accuracy"], 0.5)
        self.assertGreaterEqual(low["gauge_accuracy"], high["gauge_accuracy"])

    def test_memory_store_dimension_guard(self):
        memory = PhaseMemory(8)
        memory.store(random_key(8, random.Random(1)))
        with self.assertRaises(ValueError):
            memory.store([1 + 0j] * 7)


if __name__ == "__main__":
    unittest.main()
