"""Typed relation path/loop tests."""
import unittest

from nanlmind_hermes.relation_operators import dagger, frobenius, unitary_error
from nanlmind_hermes.typed_cycle import TypedCycleOperators


class TypedCycleTests(unittest.TestCase):
    def setUp(self):
        self.operators = TypedCycleOperators()

    def test_path_is_unitary(self):
        path = self.operators.path(["hypernym", "part_meronym", "attribute"])
        self.assertLess(unitary_error(path), 1e-12)

    def test_reverse_path_is_dagger(self):
        relations = ["hypernym", "cause", "attribute"]
        forward = self.operators.path(relations)
        reverse = self.operators.reverse_path(relations)
        self.assertLess(frobenius([[forward[row][column] - dagger(reverse)[row][column]
                                    for column in range(2)] for row in range(2)]),
                        1e-12)

    def test_relation_order_matters(self):
        first = self.operators.path(["hypernym", "part_meronym"])
        second = self.operators.path(["part_meronym", "hypernym"])
        difference = [[first[row][column] - second[row][column]
                       for column in range(2)] for row in range(2)]
        self.assertGreater(frobenius(difference), 1e-6)

    def test_unknown_relation_is_deterministic(self):
        first = self.operators.operator("custom_relation")
        second = self.operators.operator("custom_relation")
        self.assertLess(frobenius([[first[row][column] - second[row][column]
                                    for column in range(2)] for row in range(2)]),
                        1e-12)


if __name__ == "__main__":
    unittest.main()
