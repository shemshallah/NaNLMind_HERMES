"""Raw 0xNaN16 trigonometric-cycle kernel bridge.

The actual executable is :mod:`nanlmind_hermes.nan.trig_cycle.nan`, a pure
0x-word image with no Python, comments or C.  This module only loads that image
into the bundled NaNVM0 boundary and exposes bounded triangle/holonomy reads.
It is deliberately not a hidden Python reimplementation of the kernel.

Kernel contract
---------------
Inputs at ``NaNVM_MEM_BASE + 0x00``::

    word 0: mode (0 = triangle, 1 = holonomy)
    words 1..6: Q16.16 edge values

Outputs at ``NaNVM_MEM_BASE + 0x20``::

    word 0: phase / loop sum in Q16.16 turns
    word 1: cos in Q15
    word 2: sin in Q15
    word 3: valid flag

The kernel performs integer Q15/Q16.16 trigonometry and gauge cancellation.
No physical quantum or unique NaN advantage is claimed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..nanvm import NanVM, make_word, parse_nan_image, word_payload

KERNEL_PATH = Path(__file__).resolve().parents[1] / "nan" / "trig_cycle.nan"
INPUT_BASE = 0x400
OUTPUT_BASE = 0x420
MODE_TRIANGLE = 0
MODE_HOLONOMY = 1
MAX_VALUES = 6
MAX_STEPS = 100_000


class TrigCycleError(ValueError):
    """Malformed kernel input or image."""


@lru_cache(maxsize=1)
def kernel_image() -> tuple[int, ...]:
    text = KERNEL_PATH.read_text(encoding="ascii")
    return tuple(parse_nan_image(text))


def _word(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrigCycleError("%s must be a Q16.16 integer word" % name)
    if value < -(1 << 31) or value > (1 << 31) - 1:
        raise TrigCycleError("%s is outside signed 32-bit Q16.16" % name)
    return value & 0xFFFFFFFF


def run(mode: int, values: Iterable[int]) -> dict[str, int | bool]:
    """Execute the raw kernel once and return its four output words."""
    if mode not in (MODE_TRIANGLE, MODE_HOLONOMY):
        raise TrigCycleError("mode must be 0 (triangle) or 1 (holonomy)")
    items = list(values)
    if not 1 <= len(items) <= MAX_VALUES:
        raise TrigCycleError("values must contain 1..%d words" % MAX_VALUES)
    vm = NanVM(kernel_image())
    vm.mem[INPUT_BASE] = make_word(mode, 15)
    for offset, value in enumerate(items):
        vm.mem[INPUT_BASE + 1 + offset] = make_word(
            _word(value, "values[%d]" % offset), 15)
    vm.run(max_steps=MAX_STEPS)
    raw = [word_payload(vm.mem[OUTPUT_BASE + index]) & 0xFFFFFFFF
           for index in range(4)]
    return {
        "phase": raw[0],
        "cos_q15": raw[1],
        "sin_q15": raw[2],
        "valid": bool(raw[3]),
        "valid_word": raw[3],
    }


def triangle(edge_0a: int, edge_ab: int, edge_0b: int) -> dict[str, int | bool]:
    """Return the gauge-invariant triangle phase and Q15 cos/sin."""
    return run(MODE_TRIANGLE, (edge_0a, edge_ab, edge_0b))


def holonomy(edges: Iterable[int]) -> dict[str, int | bool]:
    """Return the raw loop/holonomy sum for three/four-edge Q16.16 data."""
    return run(MODE_HOLONOMY, edges)


def gauge_triangle(edge_0a: int, edge_ab: int, edge_0b: int,
                   gauge: tuple[int, int, int]) -> dict[str, int | bool]:
    """Apply vertex potentials then read the same invariant triangle."""
    try:
        g0, ga, gb = gauge
    except (TypeError, ValueError) as exc:
        raise TrigCycleError("gauge must contain exactly three integers") from exc
    return triangle(edge_0a + g0 - ga, edge_ab + ga - gb,
                    edge_0b + g0 - gb)


def kernel_status() -> dict[str, object]:
    return {
        "path": str(KERNEL_PATH),
        "schema": "nanlm.nan16-trig-cycle/1",
        "mode_triangle": MODE_TRIANGLE,
        "mode_holonomy": MODE_HOLONOMY,
        "max_values": MAX_VALUES,
        "max_steps": MAX_STEPS,
        "pure_raw_words": True,
        "canonical_merge": False,
    }


__all__ = (
    "TrigCycleError", "KERNEL_PATH", "MODE_TRIANGLE", "MODE_HOLONOMY",
    "kernel_image", "run", "triangle", "holonomy", "gauge_triangle",
    "kernel_status",
)
