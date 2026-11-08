"""Gauge-aware relational memory tests."""
import random
import unittest

from nanlmind_hermes.gauge_memory import (
    gradient,
    naive_residual,
    gauge_residual,
    project_cycle,
    retrieve,
)


class GaugeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.n = 8
        rng = random.Random(11)
        raw = {(i, j): rng.uniform(-1, 1)
               for i in range(self.n) for j in range(i + 1, self.n)}
        self.pattern = project_cycle(self.n, raw)
        self.other = project_cycle(self.n, {
            key: value + rng.uniform(-0.4, 0.4)
            for key, value in raw.items()})
        self.edges = list(self.pattern)

    def test_cycle_pattern_has_zero_divergence(self):
        from nanlmind_hermes.gauge_memory import divergence
        self.assertLess(max(abs(value) for value in
                            divergence(self.n, self.pattern)), 1e-9)

    def test_gauge_contamination_is_removed_but_naive_distance_grows(self):
        rng = random.Random(3)
        nuisance = [rng.uniform(-1, 1) for _ in range(self.n)]
        query = {edge: value + gradient(self.n, nuisance)[edge]
                 for edge, value in self.pattern.items()}
        self.assertGreater(naive_residual(query, self.pattern, self.edges), 1e-3)
        self.assertLess(gauge_residual(self.n, query, self.pattern, self.edges), 1e-9)
        self.assertGreater(gauge_residual(self.n, query, self.other, self.edges),
                           1e-3)

    def test_gauge_retrieval_beats_naive_under_erasure(self):
        rng = random.Random(5)
        nuisance = [rng.uniform(-2, 2) for _ in range(self.n)]
        query = {edge: value + gradient(self.n, nuisance)[edge]
                 for edge, value in self.pattern.items()}
        rng = random.Random(7)
        observed = [edge for edge in self.edges if rng.random() > 0.7]
        patterns = [self.other, self.pattern]
        self.assertEqual(retrieve(self.n, patterns, query, observed, gauge=True), 1)
        self.assertGreater(gauge_residual(self.n, query, self.pattern, observed),
                           0.0)


if __name__ == "__main__":
    unittest.main()
