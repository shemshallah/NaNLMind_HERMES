"""Foundational leaf algebra tests: bits, bytes, letters, words, vectors."""
import unittest

from nanlmind_hermes.leaf_algebra import (
    ByteCompletion,
    LeafAlgebra,
    build_mapping_tree,
)


class LeafAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.algebra = LeafAlgebra()

    def test_zero_and_one_are_values_and_boolean_functions(self):
        zero = self.algebra.zero
        one = self.algebra.one
        self.assertEqual(zero.kind, "BOOLEAN")
        self.assertEqual(one.value, 1)
        self.assertIn("BOOL_0", self.algebra.functions)
        self.assertIn("BOOL_15", self.algebra.functions)
        self.assertEqual(self.algebra.apply("BOOL_15", one, zero).value, 1)
        self.assertEqual(self.algebra.apply("BOOL_0", one, one).value, 0)

    def test_negate_is_a_registered_typed_function(self):
        negated = self.algebra.apply("NEGATE", self.algebra.one)
        self.assertEqual(negated.value, 0)
        self.assertEqual(negated.provenance["function"], "NEGATE")
        self.assertEqual(self.algebra.apply("AFFIRM", self.algebra.one).value, 1)

    def test_no_is_a_hyperedge_over_letters(self):
        word = self.algebra.word("NO")
        self.assertEqual(word.kind, "WORD")
        self.assertEqual(word.value, ("N", "O"))
        edge = self.algebra.hyperedges[word.provenance["hyperedge"]]
        self.assertEqual(edge["relation"], "spells")
        self.assertEqual(len(edge["members"]), 2)

    def test_vector_is_dimension_aware(self):
        vector = self.algebra.vector((1, 0, 1), dtype="BOOLEAN")
        self.assertEqual(vector.rank, 1)
        self.assertEqual(vector.shape, (3,))
        matrix = self.algebra.matrix(((1, 0), (0, 1)))
        self.assertEqual(matrix.rank, 2)
        self.assertEqual(matrix.shape, (2, 2))

    def test_mapping_tree_has_every_level(self):
        tree = build_mapping_tree(self.algebra)
        self.assertEqual(len(tree["L0_bits"]), 2)
        self.assertEqual(len(tree["L1_functions"]), 19)
        self.assertEqual(len(tree["L2_bytes"]), 256)
        self.assertEqual(len(tree["L3_letters"]), 95)
        self.assertGreaterEqual(len(tree["L4_words"]), 10)
        self.assertEqual(len(tree["L5_vectors"]), 1)

    def test_byte_layer_uses_all_256_head_coordinates(self):
        for value in range(256):
            self.algebra.byte(value)
        self.assertEqual(len(self.algebra.heads), 256)
        self.assertEqual(sorted(self.algebra.heads), list(range(256)))

    def test_heads_admit_leaves_and_ledger_transforms(self):
        self.algebra.word("YES")
        self.assertTrue(self.algebra.heads)
        self.assertTrue(any(event["event"] == "admit"
                            for event in self.algebra.ledger))
        event = self.algebra.self_modify(self.algebra.one, "test")
        self.assertEqual(event["event"], "self_modify")


class ByteCompletionTests(unittest.TestCase):
    def test_completion_learns_prefix_to_bit_counts(self):
        learner = ByteCompletion()
        learner.train(list(range(0, 192)))
        result = learner.evaluate(list(range(192, 256)))
        self.assertIn("accuracy", result)
        self.assertEqual(set(result["by_prefix"]), set(range(8)))
        self.assertGreater(result["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
