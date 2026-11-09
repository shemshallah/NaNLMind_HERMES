"""Tests for the resident raw 0xNaN16 trig-cycle bridge."""
from __future__ import annotations

import math
import unittest

from nanlmind_hermes.resident.trig_cycle import (
    MODE_HOLONOMY,
    MODE_TRIANGLE,
    TrigCycleError,
    gauge_triangle,
    holonomy,
    kernel_status,
    run,
    triangle,
)


class ResidentTrigCycleTests(unittest.TestCase):
    def test_kernel_is_raw_and_bounded(self):
        status = kernel_status()
        self.assertTrue(status["pure_raw_words"])
        self.assertFalse(status["canonical_merge"])
        with self.assertRaises(TrigCycleError):
            run(9, [1])
        with self.assertRaises(TrigCycleError):
            run(MODE_TRIANGLE, [])

    def test_triangle_phase_and_trig(self):
        result = triangle(2000, 10000, 1808)
        self.assertTrue(result["valid"])
        self.assertEqual(result["phase"], 10192)
        angle = 2 * math.pi * (10192 / 65536.0)
        self.assertAlmostEqual(result["cos_q15"] / 32768.0,
                               math.cos(angle), places=2)
        self.assertAlmostEqual(result["sin_q15"] / 32768.0,
                               math.sin(angle), places=2)

    def test_gauge_cancels_exactly(self):
        base = triangle(2000, 10000, 1808)
        changed = gauge_triangle(2000, 10000, 1808, (7000, 9000, 11000))
        self.assertEqual(base, changed)

    def test_holonomy_loop_is_invariant(self):
        values = [1000, 2000, 3000, 4000, 5000, 6000]
        base = holonomy(values)
        self.assertEqual(base["phase"], 12000)
        # The kernel's holonomy mode is exercised as raw six-word input; the
        # local bridge does not invent an extra gauge convention.
        self.assertEqual(base, run(MODE_HOLONOMY, values))


if __name__ == "__main__":
    unittest.main()
