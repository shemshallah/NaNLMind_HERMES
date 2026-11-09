"""Raw NaN16 trigonometric cycle kernel tests.

The committed ``trig_cycle.nan`` is intentionally pure 0xNaN word lines: no
Python, no comments, no C.  These tests execute it on the included NaNVM0
boundary and compare its Q15 sin/cos readout with the mathematical reference.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import unittest

from nanlmind_hermes.nanvm import NanVM, make_word, parse_nan_image, word_payload

KERNEL = (Path(__file__).resolve().parents[1]
          / "nanlmind_hermes" / "nan" / "trig_cycle.nan")
WORD = re.compile(r"^[0-9A-F]{4}_[0-9A-F]{8}_[0-9A-F]{4}$")


def _run(image, mode: int, values: list[int]) -> list[int]:
    vm = NanVM(image)
    for offset, value in enumerate([mode] + values):
        vm.mem[0x400 + offset] = make_word(value & 0xFFFFFFFF, 15)
    vm.run(max_steps=100_000)
    return [word_payload(vm.mem[0x420 + index]) for index in range(4)]


class RawTrigCycleKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        text = KERNEL.read_text(encoding="ascii")
        cls.lines = [line.strip() for line in text.splitlines() if line.strip()]
        cls.image = parse_nan_image("\n".join(cls.lines))

    def test_image_is_pure_raw_words(self) -> None:
        self.assertTrue(self.lines)
        for line in self.lines:
            self.assertRegex(line, WORD)
        self.assertEqual(len(self.image), 513)

    def test_triangle_phase_is_exact(self) -> None:
        # e(0,a)=2000, e(a,b)=10000, e(0,b)=1808 -> 10192 in Q16.16 turns.
        out = _run(self.image, 0, [2000, 10000, 1808])
        self.assertEqual(out[0], 10192)
        self.assertEqual(out[3], 1)
        angle = 2.0 * math.pi * (10192 / 65536.0)
        self.assertAlmostEqual(out[1] / 32768.0, math.cos(angle), places=2)
        self.assertAlmostEqual(out[2] / 32768.0, math.sin(angle), places=2)

    def test_additive_vertex_gauge_cancels(self) -> None:
        values = [2000, 10000, 1808]
        base = _run(self.image, 0, values)
        g = [7000, 9000, 11000]
        # e(0,a) += g0-ga, e(a,b) += ga-gb, e(0,b) += g0-gb.
        gauged = [values[0] + g[0] - g[1],
                  values[1] + g[1] - g[2],
                  values[2] + g[0] - g[2]]
        after = _run(self.image, 0, gauged)
        self.assertEqual(base, after)

    def test_holonomy_loop_is_gauge_invariant(self) -> None:
        values = [1000, 2000, 3000, 4000, 5000, 6000]
        base = _run(self.image, 1, values)
        self.assertEqual(base[0], 12000)
        g = [7000, 9000, 11000, 13000]
        gauged = [
            values[0] + g[0] - g[1], values[1] + g[1] - g[2],
            values[2] + g[0] - g[2], values[3] + g[2] - g[3],
            values[4] + g[0] - g[3], values[5] + g[3] - g[1],
        ]
        after = _run(self.image, 1, gauged)
        self.assertEqual(base, after)


if __name__ == "__main__":
    unittest.main()
