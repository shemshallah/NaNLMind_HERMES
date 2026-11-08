"""Typed non-Abelian relation operator tests."""
import math
import unittest

from nanlmind_hermes.relation_operators import (
    commutator,
    dagger,
    frobenius,
    gauge_transform,
    loop_trace,
    matmul,
    su2,
    unitary_error,
)


class RelationOperatorTests(unittest.TestCase):
    def test_su2_is_unitary_and_noncommuting(self):
        x = su2((1, 0, 0), math.pi / 2)
        z = su2((0, 0, 1), math.pi / 2)
        self.assertLess(unitary_error(x), 1e-12)
        self.assertLess(unitary_error(z), 1e-12)
        self.assertGreater(frobenius(commutator(x, z)), 0.5)

    def test_closed_loop_trace_is_gauge_invariant(self):
        x = su2((1, 0, 0), math.pi / 2)
        z = su2((0, 0, 1), math.pi / 2)
        edges = [x, z, dagger(x), dagger(z)]
        baseline = loop_trace(edges)
        gauges = [su2((1, 1, 0), 0.3), su2((0, 1, 1), 0.7),
                  su2((1, 0, 1), 1.1), su2((0, 0, 1), 0.4)]
        transformed = [gauge_transform(edges[index], gauges[index],
                                       gauges[(index + 1) % 4]) for index in range(4)]
        self.assertLess(abs(loop_trace(transformed) - baseline), 1e-12)

    def test_composition_order_matters(self):
        x = su2((1, 0, 0), math.pi / 3)
        z = su2((0, 0, 1), math.pi / 5)
        self.assertGreater(frobenius([[matmul(x, z)[r][c] - matmul(z, x)[r][c]
                                       for c in range(2)] for r in range(2)]), 0.1)


if __name__ == "__main__":
    unittest.main()
