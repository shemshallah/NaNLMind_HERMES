"""Tests for SO(2)/SO(3) order and holonomy reference algebra."""
from __future__ import annotations

import math
import unittest

from nanlmind_hermes.resident.rotation_holonomy import (
    audit_order,
    commutator_norm2,
    commutator_norm3,
    compose2,
    compose3,
    conjugation_error,
    givens,
    rotation2,
    trace,
    transpose,
)


class RotationHolonomyTests(unittest.TestCase):
    def test_so2_rotations_commute(self):
        first = rotation2(0.31)
        second = rotation2(0.77)
        self.assertAlmostEqual(commutator_norm2(first, second), 0.0, places=12)
        self.assertAlmostEqual(compose2(first, second)[0][0],
                               math.cos(1.08), places=12)

    def test_so3_rotations_do_not_commute(self):
        first = givens(0, 0.31)
        second = givens(1, 0.77)
        self.assertGreater(commutator_norm3(first, second), 0.1)
        self.assertNotEqual(compose3(first, second), compose3(second, first))

    def test_trace_is_conjugation_invariant(self):
        rotation = givens(2, 0.23)
        frame = compose3(givens(0, 0.4), givens(1, -0.2))
        inverse = transpose(frame)
        self.assertLess(conjugation_error(rotation, frame, inverse), 1e-12)
        self.assertAlmostEqual(trace(rotation), 1 + 2 * math.cos(0.23), places=12)

    def test_audit_values(self):
        report = audit_order()
        self.assertEqual(report.phase_commutator, 0.0)
        self.assertGreater(report.rotation_commutator, 0.1)
        self.assertTrue(report.order_sensitive)
        self.assertFalse(report.as_dict()["canonical_merge"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
