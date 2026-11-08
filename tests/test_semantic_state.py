"""Sparse semantic-state feature tests."""
import unittest

from nanlmind_hermes.semantic_state import (
    ancestor_edges,
    cycle_state,
    hash_features,
    ordinary_state,
)


class SemanticStateTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {
            1: {"offset": 1, "pointers": []},
            2: {"offset": 2, "pointers": [
                {"relation": "hypernym", "target_offset": 1}]},
            3: {"offset": 3, "pointers": [
                {"relation": "hypernym", "target_offset": 2}]},
            4: {"offset": 4, "pointers": [
                {"relation": "hypernym", "target_offset": 3}]},
        }

    def test_hash_features_are_deterministic_and_normalised(self):
        first = hash_features("a kind of plant", 16)
        second = hash_features("a kind of plant", 16)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0,
                               places=10)

    def test_ancestor_edges_are_bounded(self):
        self.assertEqual(ancestor_edges(self.nodes, 4, depth=2), [(4, 3), (3, 2)])

    def test_ordinary_state_is_nonzero(self):
        state = ordinary_state(self.nodes, 4, depth=2, dimensions=32)
        self.assertEqual(len(state), 32)
        self.assertGreater(sum(value * value for value in state), 0.9)

    def test_cycle_state_is_finite_and_nonzero(self):
        state = cycle_state(self.nodes, 4, depth=2, dimensions=32)
        self.assertEqual(len(state), 32)
        self.assertTrue(all(abs(value) < 1e-12 or value == value for value in state))
        self.assertGreater(sum(value * value for value in state), 0.0)


if __name__ == "__main__":
    unittest.main()
