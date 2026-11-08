"""Exact WordNet multi-hop traversal tests."""
import unittest

from nanlmind_hermes.wordnet_reasoning import ancestors, is_ancestor, parents


class WordNetReasoningTests(unittest.TestCase):
    def setUp(self):
        self.synsets = {
            1: {"offset": 1, "pointers": []},
            2: {"offset": 2, "pointers": [
                {"relation": "hypernym", "target_offset": 1}]},
            3: {"offset": 3, "pointers": [
                {"relation": "hypernym", "target_offset": 2}]},
            4: {"offset": 4, "pointers": [
                {"relation": "instance_hypernym", "target_offset": 3}]},
        }

    def test_parents_reads_typed_pointers(self):
        self.assertEqual(parents(self.synsets[4]), [3])

    def test_ancestors_reports_shortest_depths(self):
        result = ancestors(self.synsets, 4, max_depth=4)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[2], 2)
        self.assertEqual(result[1], 3)

    def test_direct_edge_is_not_multi_hop(self):
        self.assertTrue(is_ancestor(self.synsets, 4, 3))
        self.assertTrue(is_ancestor(self.synsets, 4, 1))
        self.assertFalse(is_ancestor(self.synsets, 4, 2, max_depth=1))


if __name__ == "__main__":
    unittest.main()
