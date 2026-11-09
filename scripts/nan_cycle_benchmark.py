#!/usr/bin/env python3
"""Benchmark integer NaN lane/cycle readout against raw and decoy readouts.

The benchmark constructs the Round 029 anchored-triangle field on the complete
256-head graph.  Every non-hub pair ``(a, b)`` with ``1 <= a < b < 256`` gets a
truth value ``z(a, b)``; every spoke ``(0, v)`` carries no truth value.  A
vertex gauge ``g`` writes the oriented lane field::

    x(0, v) = g(0) - g(v)
    x(a, b) = z(a, b) + g(a) - g(b)

The correct cycle flux is::

    x(0, a) + x(a, b) - x(0, b) = z(a, b)

for every gauge.  The benchmark compares that exact integer cycle readout with:

* ``raw`` - the single directed pair lane ``x(a, b)``;
* ``wrong_triangle`` - a triangle whose third spoke points at a different
  non-hub vertex, so the gauge no longer cancels;
* ``conventional_cycle`` - the same correct formula over a plain integer edge
  table (a conventional baseline, not a NaN-specific readout);
* ``uniform`` - a uniform vertex-gauge control where both cycle and raw
  readouts stay exact;
* ``noise`` - independent per-lane integer noise where neither cycle nor raw
  readout stays exact, showing that cycles cancel gauge, not arbitrary noise.

The cycle basis is the star fundamental-cycle basis anchored at head 0.  It has
``C(256, 2) - 255 = 32,385`` independent cycles.  The first 48 basis cycles are
the disjoint Round 029 triangles ``(0, 1+2k, 2+2k)``; the remaining cycles are
enumerated lexicographically.  Default measurement prefixes are 48, 256, and
the full 32,385 cycles.

Lane values are read from the real ``nanlmind_hermes.resident.cells`` pair rails
when that implementation is available.  Otherwise a deterministic integer fake
lane table is used.  The benchmark never merges canonical pairs and makes no
native-code, phone/handset, uniqueness, quantum, training, or model-quality
claim.  Identity comparisons use Python integers only; floats appear only in
wall-clock/throughput fields.

Examples::

    python3 scripts/nan_cycle_benchmark.py \\
        --cycles 48,256,32385 \\
        --json /tmp/nan-cycle.json --markdown /tmp/nan-cycle.md

    python3 scripts/nan_cycle_benchmark.py --backend fake --cycles 48 --steps 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "nanlmind-hermes-nan-cycle-benchmark/1"
HEADS = 256
PAIR_CAPACITY = HEADS * (HEADS - 1) // 2  # 32,640
CYCLE_SPACE_DIMENSION = PAIR_CAPACITY - (HEADS - 1)  # 32,385
ROUND029_TRIANGLES = 48
DEFAULT_PREFIXES = (ROUND029_TRIANGLES, 256, CYCLE_SPACE_DIMENSION)
MIN_CYCLES = 1
MAX_CYCLES = CYCLE_SPACE_DIMENSION
DEFAULT_STEPS = 1
MIN_STEPS = 1
MAX_STEPS = 1_000_000
DEFAULT_SEED = 20261108
DEFAULT_BACKEND = "auto"
BACKENDS = ("auto", "resident", "fake")
NOISE_BASE = 64
READOUT_MODES = ("cycle", "raw", "wrong_triangle", "conventional_cycle")
CONTROL_MODES = ("uniform", "noise")
MODES = READOUT_MODES + CONTROL_MODES
FIELDS = ("clean", "gauge_linear", "gauge_scrambled", "noise")

_READOUT_OPS_PER_CYCLE = {
    "cycle": 5,
    "raw": 1,
    "wrong_triangle": 5,
    "conventional_cycle": 5,
}
_READOUT_LANE_READS_PER_CYCLE = {
    "cycle": 3,
    "raw": 1,
    "wrong_triangle": 3,
    "conventional_cycle": 3,
}
_CONTROL_OPS_PER_CYCLE = 6  # cycle (5) + raw (1)
_CONTROL_LANE_READS_PER_CYCLE = 4  # cycle (3) + raw (1)

LIMITATIONS = [
    "Measured on host CPython over an in-memory resident cell region or a "
    "deterministic integer lane table; this is not native code, not an mmap "
    "file, and not a phone/handset performance claim.",
    "Readouts consume an integer lane snapshot read once from the backend; "
    "the report separates snapshot-read throughput from readout-formula "
    "throughput.",
    "The lane values are synthetic integer fields, not model logits, trained "
    "embeddings, semantic quality, or language-model accuracy.",
    "Cycle readouts cover the star fundamental-cycle basis (32,385 cycles), "
    "not every possible closed walk or every cycle-space linear combination.",
    "The 48-cycle prefix is the disjoint Round 029 embedding; the 256-cycle "
    "prefix and full prefix are ordered basis prefixes, not separate learned "
    "models.",
    "Raw and wrong-triangle readouts are expected to fail exact gauge "
    "invariance under nonuniform gauges.  That failure is a control outcome, "
    "not a benchmark error.",
    "The uniform-gauge control is a zero lane-delta control: a constant vertex "
    "shift cancels in every oriented lane, so both cycle and raw readouts are "
    "exact.",
    "Per-lane integer noise is deliberately non-gauge; neither cycle nor raw "
    "readout is expected to cancel it.",
    "The conventional cycle readout reads a plain integer edge table and uses "
    "the same correct triangle formula; it is expected to tie the lane cycle "
    "readout exactly.",
    "No canonical merge, promotion, durable user state, training, or native "
    "acceleration is performed by this benchmark.",
    "No uniqueness, quantum-advantage, native-performance, or phone/handset "
    "claim is made.",
]

try:
    from nanlmind_hermes.resident import cells as resident_cells
except Exception as exc:  # pragma: no cover - only on broken installs
    resident_cells = None
    _IMPORT_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    _IMPORT_ERROR = None


class BenchmarkConfigError(ValueError):
    """Raised for out-of-range cycles/steps or invalid backend names."""


class BenchmarkUnavailable(RuntimeError):
    """Raised when a requested resident-cell backend cannot be used."""


# ---------------------------------------------------------------------------
# configuration bounds
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkConfigError("%s must be an integer: %r" % (name, value))
    return value


def validate_prefixes(value: Any) -> tuple[int, ...]:
    """Return a strictly increasing tuple of cycle counts in 1..32,385."""
    if isinstance(value, (str, bytes)) or not isinstance(value, IterableABC):
        raise BenchmarkConfigError(
            "cycles must be an iterable of integers: %r" % (value,))
    prefixes = []
    for item in value:
        count = _require_int(item, "cycles entry")
        if not MIN_CYCLES <= count <= MAX_CYCLES:
            raise BenchmarkConfigError(
                "cycle count must be in [%d, %d]: %r"
                % (MIN_CYCLES, MAX_CYCLES, item))
        prefixes.append(count)
    if not prefixes:
        raise BenchmarkConfigError("cycles must contain at least one count")
    if len(set(prefixes)) != len(prefixes):
        raise BenchmarkConfigError("cycle counts must be unique: %r"
                                   % (tuple(prefixes),))
    if prefixes != sorted(prefixes):
        raise BenchmarkConfigError("cycle counts must be strictly increasing: %r"
                                   % (tuple(prefixes),))
    return tuple(prefixes)


def parse_prefixes(text: str) -> tuple[int, ...]:
    """Parse a comma-separated cycle-count list such as ``48,256,32385``."""
    if not isinstance(text, str):
        raise BenchmarkConfigError("cycles text must be a string: %r" % (text,))
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise BenchmarkConfigError("cycles list is empty or malformed: %r"
                                   % (text,))
    values = []
    for part in parts:
        try:
            values.append(int(part, 10))
        except ValueError as exc:
            raise BenchmarkConfigError(
                "cycles entry is not an integer: %r" % (part,)) from exc
    return validate_prefixes(values)


def validate_steps(value: Any) -> int:
    """Return ``value`` if it is an integer in 1..1,000,000."""
    steps = _require_int(value, "steps")
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise BenchmarkConfigError(
            "steps must be in [%d, %d]: %r" % (MIN_STEPS, MAX_STEPS, value))
    return steps


def validate_seed(value: Any) -> int:
    return _require_int(value, "seed")


def validate_backend(value: Any) -> str:
    if value not in BACKENDS:
        raise BenchmarkConfigError(
            "backend must be one of %s: %r" % (", ".join(BACKENDS), value))
    return str(value)


def resident_available() -> bool:
    """True when the real 256-head / 32,640-pair resident cells fit."""
    if resident_cells is None:
        return False
    if getattr(resident_cells, "HEAD_COUNT", None) != HEADS:
        return False
    if getattr(resident_cells, "PAIR_COUNT", None) != PAIR_CAPACITY:
        return False
    region_bytes = getattr(resident_cells, "REGION_BYTES", None)
    return isinstance(region_bytes, int) and region_bytes > 0


# ---------------------------------------------------------------------------
# deterministic integer field construction
# ---------------------------------------------------------------------------
def build_cycle_basis() -> tuple[tuple[int, int], ...]:
    """Return the 32,385 star-basis cycles with Round 029 triangles first."""
    result = [(1 + 2 * k, 2 + 2 * k) for k in range(ROUND029_TRIANGLES)]
    seen = set(result)
    for first in range(1, HEADS):
        for second in range(first + 1, HEADS):
            if (first, second) not in seen:
                result.append((first, second))
    if len(result) != CYCLE_SPACE_DIMENSION:
        raise AssertionError(
            "cycle basis has %d entries, expected %d"
            % (len(result), CYCLE_SPACE_DIMENSION))
    return tuple(result)


CYCLE_BASIS = build_cycle_basis()


def wrong_vertex(a: int, b: int) -> int:
    """A non-hub vertex different from both ``a`` and ``b``."""
    candidate = 1 + (b % (HEADS - 1))
    if candidate == a:
        candidate = 1 + (candidate % (HEADS - 1))
    if candidate in (a, b):
        raise AssertionError("wrong vertex collided for (%d, %d)" % (a, b))
    return candidate


def truth_value(a: int, b: int, seed: int) -> int:
    """Deterministic small integer cycle truth in -8..8."""
    return ((a * 97 + b * 31 + (seed % 1009)) % 17) - 8


def zero_gauge(head: int) -> int:
    return 0


def gauge_linear(head: int) -> int:
    """Injective linear vertex gauge over the 256 heads."""
    return 3 * head - 384


def gauge_scrambled(head: int) -> int:
    """Injective affine gauge modulo the prime 257."""
    return (131 * head + 17) % 257


def gauge_uniform(head: int) -> int:
    return 11


def noise_value(source: int, target: int) -> int:
    """Positive non-gauge lane noise.

    With ``B = 64`` the cycle noise is
    ``(a+1)*B + (a+1)``, which is positive for every non-hub ``a``.
    """
    return (source + 1) * NOISE_BASE + (target + 1)


def _field_gauge(field: str, head: int) -> int:
    if field == "clean":
        return 0
    if field == "gauge_linear":
        return gauge_linear(head)
    if field in ("gauge_scrambled", "noise"):
        return gauge_scrambled(head)
    raise BenchmarkConfigError("unknown field: %r" % (field,))


def _base_lane_value(source: int, target: int, seed: int) -> int:
    if source == 0:
        return 0
    return truth_value(source, target, seed)


def expected_lane_value(field: str, source: int, target: int, seed: int) -> int:
    value = (_base_lane_value(source, target, seed)
             + _field_gauge(field, source) - _field_gauge(field, target))
    if field == "noise":
        value += noise_value(source, target)
    return value


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class FakeLaneBackend:
    """Deterministic integer lane map used when resident cells are absent."""

    kind = "fake"
    lane_value_bytes = 4
    lane_cell_bytes = 4

    def __init__(self) -> None:
        self._values: dict[tuple[int, int], int] = {}
        self.write_operations = 0
        self.read_operations = 0

    def set_edge(self, source: int, target: int, value: int) -> None:
        self._values[(source, target)] = int(value)
        self.write_operations += 1

    def add_edge(self, source: int, target: int, delta: int) -> None:
        key = (source, target)
        self._values[key] = self._values.get(key, 0) + int(delta)
        self.write_operations += 1

    def read_edge(self, source: int, target: int) -> int:
        self.read_operations += 1
        return int(self._values[(source, target)])

    def read_lanes(self, lanes: Sequence[tuple[int, int]]) -> list[int]:
        values = []
        for edge in lanes:
            values.append(self.read_edge(*edge))
        return values

    @property
    def storage_bytes(self) -> int:
        return self.lane_value_bytes * len(self._values)

    @property
    def lane_storage_bytes(self) -> int:
        return self.lane_value_bytes * len(self._values)

    @property
    def storage_kind(self) -> str:
        return "logical i32 lane table estimate (dictionary overhead excluded)"


class ResidentLaneBackend:
    """Real 2 MiB resident cell region with raw pair-rail lane access."""

    kind = "resident"
    lane_value_bytes = 4

    def __init__(self) -> None:
        if not resident_available():
            raise BenchmarkUnavailable(
                _IMPORT_ERROR or "nanlmind_hermes.resident.cells is unavailable")
        self._template = bytearray(int(resident_cells.REGION_BYTES))
        resident_cells.ResidentCells.format_region(self._template)
        self._buffer = bytearray(self._template)
        self._cells = resident_cells.ResidentCells(self._buffer)
        self.write_operations = 0
        self.read_operations = 0
        self._fixed_one = int(getattr(resident_cells, "FIXED_ONE", 1 << 16))
        self._rail_s_plus = int(getattr(resident_cells, "RAIL_S_PLUS", 0))
        self._rail_s_minus = int(getattr(resident_cells, "RAIL_S_MINUS", 1))
        self.lane_cell_bytes = int(
            getattr(resident_cells, "PAIR_CELL_BYTES", 64))

    def set_edge(self, source: int, target: int, value: int) -> None:
        self._cells.observe_pair(source, target, weight=int(value))
        self.write_operations += 1

    def add_edge(self, source: int, target: int, delta: int) -> None:
        self._cells.observe_pair(source, target, weight=int(delta))
        self.write_operations += 1

    def _rail_for(self, source: int, target: int) -> int:
        canonical = resident_cells.canonical_address(source, target)
        native = resident_cells.address_of(source, target)
        return self._rail_s_plus if native == canonical else self._rail_s_minus

    def read_edge(self, source: int, target: int) -> int:
        rail = self._rail_for(source, target)
        record = self._cells.read_rail(source, target, rail)
        fixed = int(record.weight)
        if fixed % self._fixed_one != 0:
            raise BenchmarkUnavailable(
                "resident lane weight is not an exact Q16.16 integer: %d"
                % fixed)
        self.read_operations += 1
        return fixed // self._fixed_one

    def read_lanes(self, lanes: Sequence[tuple[int, int]]) -> list[int]:
        values = []
        for edge in lanes:
            values.append(self.read_edge(*edge))
        return values

    @property
    def storage_bytes(self) -> int:
        return len(self._buffer)

    @property
    def lane_storage_bytes(self) -> int:
        return int(getattr(resident_cells, "PAIR_REGION_BYTES",
                           self.storage_bytes))

    @property
    def storage_kind(self) -> str:
        return "real ResidentCells 2 MiB region (two 32-byte rails per pair)"


def open_backend(kind: str = DEFAULT_BACKEND) -> Any:
    """Open a resident or fake lane backend; ``auto`` prefers resident."""
    kind = validate_backend(kind)
    use_resident = kind == "resident" or (kind == "auto"
                                          and resident_available())
    if use_resident:
        return ResidentLaneBackend()
    if kind == "resident":
        raise BenchmarkUnavailable(
            _IMPORT_ERROR or "nanlmind_hermes.resident.cells is unavailable")
    return FakeLaneBackend()


# ---------------------------------------------------------------------------
# field and readout helpers
# ---------------------------------------------------------------------------
def required_lanes(cycles: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """All directed lanes needed by cycle, raw, and wrong-triangle readouts."""
    lanes = set()
    for first, second in cycles:
        decoy = wrong_vertex(first, second)
        lanes.add((0, first))
        lanes.add((first, second))
        lanes.add((0, second))
        lanes.add((0, decoy))
    return tuple(sorted(lanes))


def write_clean_field(backend: Any, lanes: Sequence[tuple[int, int]],
                      seed: int) -> None:
    for source, target in lanes:
        backend.set_edge(source, target, _base_lane_value(source, target, seed))


def apply_field_delta(backend: Any, lanes: Sequence[tuple[int, int]],
                      previous: str, current: str) -> None:
    for source, target in lanes:
        old_delta = (_field_gauge(previous, source)
                     - _field_gauge(previous, target))
        new_delta = (_field_gauge(current, source)
                     - _field_gauge(current, target))
        delta = new_delta - old_delta
        if current == "noise":
            delta += noise_value(source, target)
        if delta:
            backend.add_edge(source, target, delta)


def snapshot_lanes(backend: Any, lanes: Sequence[tuple[int, int]],
                   field: str, seed: int) -> tuple[list[int], dict[str, Any]]:
    start = time.perf_counter()
    values = backend.read_lanes(lanes)
    seconds = time.perf_counter() - start
    expected = [expected_lane_value(field, source, target, seed)
                for source, target in lanes]
    exact = sum(1 for value, wanted in zip(values, expected) if value == wanted)
    return values, {
        "field": field,
        "lanes": len(lanes),
        "snapshot_lanes": len(values),
        "snapshot_exact_lanes": exact,
        "snapshot_mismatched_lanes": len(values) - exact,
        "snapshot_seconds": seconds,
        "snapshot_read_operations": len(lanes),
        "snapshot_read_operations_per_sec": (
            len(lanes) / seconds if seconds > 0 else 0.0),
    }


def build_plan(cycles: Sequence[tuple[int, int]],
               lane_index: dict[tuple[int, int], int]) -> list[tuple[int, int, int, int]]:
    plan = []
    for first, second in cycles:
        decoy = wrong_vertex(first, second)
        plan.append((
            lane_index[(0, first)],
            lane_index[(first, second)],
            lane_index[(0, second)],
            lane_index[(0, decoy)],
        ))
    return plan


def readout_pass(mode: str, cycles: Sequence[tuple[int, int]],
                 plan: Sequence[tuple[int, int, int, int]],
                 lane_values: Sequence[int],
                 table: dict[tuple[int, int], int] | None) -> list[int]:
    if mode == "raw":
        return [lane_values[item[1]] for item in plan]
    if mode == "cycle":
        return [lane_values[item[0]] + lane_values[item[1]]
                - lane_values[item[2]] for item in plan]
    if mode == "wrong_triangle":
        return [lane_values[item[0]] + lane_values[item[1]]
                - lane_values[item[3]] for item in plan]
    if mode == "conventional_cycle":
        if table is None:
            raise AssertionError("conventional readout needs a table")
        return [table[(0, first)] + table[(first, second)]
                - table[(0, second)] for first, second in cycles]
    raise BenchmarkConfigError("unknown readout mode: %r" % (mode,))


def measure_readout(mode: str, cycles: Sequence[tuple[int, int]],
                    plan: Sequence[tuple[int, int, int, int]],
                    lane_values: Sequence[int],
                    table: dict[tuple[int, int], int] | None,
                    steps: int) -> tuple[list[int], float]:
    start = time.perf_counter()
    values: list[int] = []
    for _ in range(steps):
        values = readout_pass(mode, cycles, plan, lane_values, table)
    seconds = time.perf_counter() - start
    return values, seconds


def readout_digest(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def exact_count(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise AssertionError("identity comparison length mismatch")
    return sum(1 for first, second in zip(left, right) if first == second)


# ---------------------------------------------------------------------------
# benchmark run
# ---------------------------------------------------------------------------
def run_benchmark(*, prefixes: Sequence[int] = DEFAULT_PREFIXES,
                  steps: int = DEFAULT_STEPS, seed: int = DEFAULT_SEED,
                  backend: str = DEFAULT_BACKEND) -> dict[str, Any]:
    """Run every readout/control and return a JSON-serialisable report."""
    prefixes = validate_prefixes(prefixes)
    steps = validate_steps(steps)
    seed = validate_seed(seed)
    backend_kind = validate_backend(backend)

    run_start = time.perf_counter()
    lane_backend = open_backend(backend_kind)
    lane_start = time.perf_counter()

    max_cycles = max(prefixes)
    cycles_full = CYCLE_BASIS[:max_cycles]
    lanes = required_lanes(cycles_full)
    lane_index = {edge: index for index, edge in enumerate(lanes)}
    prefix_plans = {}
    for count in prefixes:
        cycles = CYCLE_BASIS[:count]
        prefix_plans[count] = (cycles, build_plan(cycles, lane_index))

    write_clean_field(lane_backend, lanes, seed)
    field_start = time.perf_counter()
    snapshots: dict[str, list[int]] = {}
    snapshot_meta: dict[str, dict[str, Any]] = {}
    tables: dict[str, dict[tuple[int, int], int]] = {}
    table_seconds: dict[str, float] = {}
    table_operations: dict[str, int] = {}
    readout_records: dict[tuple[str, str, int], dict[str, Any]] = {}

    previous = "clean"
    write_marker = 0
    for field in FIELDS:
        if field != "clean":
            apply_field_delta(lane_backend, lanes, previous, field)
        write_operations = lane_backend.write_operations - write_marker
        write_marker = lane_backend.write_operations
        values, meta = snapshot_lanes(lane_backend, lanes, field, seed)
        meta["write_operations"] = write_operations
        snapshots[field] = values
        snapshot_meta[field] = meta

        table_start = time.perf_counter()
        table = {edge: values[index] for index, edge in enumerate(lanes)}
        table_seconds[field] = time.perf_counter() - table_start
        table_operations[field] = len(lanes)
        tables[field] = table

        for mode_name in READOUT_MODES:
            for count in prefixes:
                cycles, plan = prefix_plans[count]
                measured, seconds = measure_readout(
                    mode_name, cycles, plan, values, table, steps)
                operations = _READOUT_OPS_PER_CYCLE[mode_name] * count * steps
                readout_records[(field, mode_name, count)] = {
                    "values": measured,
                    "seconds": seconds,
                    "operations": operations,
                    "lane_reads": (_READOUT_LANE_READS_PER_CYCLE[mode_name]
                                   * count * steps),
                }
        previous = field

    fields_seconds = time.perf_counter() - field_start
    lane_setup_seconds = field_start - lane_start

    # Truth values are the same for all readout modes in a prefix.
    truths: dict[int, list[int]] = {}
    for count in prefixes:
        truths[count] = [truth_value(first, second, seed)
                         for first, second in CYCLE_BASIS[:count]]

    def record(field: str, mode_name: str, count: int) -> dict[str, Any]:
        return readout_records[(field, mode_name, count)]

    def counts(field: str, mode_name: str, count: int) -> tuple[int, int]:
        values = record(field, mode_name, count)["values"]
        exact = exact_count(values, truths[count])
        return exact, count - exact

    def invariance(field: str, mode_name: str, count: int,
                   baseline: str = "clean") -> tuple[int, int]:
        values = record(field, mode_name, count)["values"]
        reference = record(baseline, mode_name, count)["values"]
        exact = exact_count(values, reference)
        return exact, count - exact

    mode_reports: dict[str, Any] = {}
    for mode_name in READOUT_MODES:
        prefix_metrics: dict[str, Any] = {}
        for count in prefixes:
            clean = record("clean", mode_name, count)
            linear = record("gauge_linear", mode_name, count)
            scrambled = record("gauge_scrambled", mode_name, count)
            noise = record("noise", mode_name, count)
            clean_exact, clean_miss = counts("clean", mode_name, count)
            linear_truth_exact, linear_truth_miss = counts(
                "gauge_linear", mode_name, count)
            scrambled_truth_exact, scrambled_truth_miss = counts(
                "gauge_scrambled", mode_name, count)
            noise_truth_exact, noise_truth_miss = counts(
                "noise", mode_name, count)
            linear_exact, linear_miss = invariance(
                "gauge_linear", mode_name, count)
            scrambled_exact, scrambled_miss = invariance(
                "gauge_scrambled", mode_name, count)
            noise_exact, noise_miss = invariance(
                "noise", mode_name, count, baseline="gauge_scrambled")
            uniform_exact = clean_exact
            uniform_miss = count - uniform_exact
            total_seconds = (clean["seconds"] + linear["seconds"]
                             + scrambled["seconds"] + noise["seconds"])
            total_operations = (clean["operations"] + linear["operations"]
                                + scrambled["operations"] + noise["operations"])
            lane_reads = (clean["lane_reads"] + linear["lane_reads"]
                          + scrambled["lane_reads"] + noise["lane_reads"])
            sample = []
            for index in range(min(5, count)):
                first, second = CYCLE_BASIS[index]
                sample.append({
                    "a": first,
                    "b": second,
                    "wrong_c": wrong_vertex(first, second),
                    "truth": truths[count][index],
                    "clean": clean["values"][index],
                    "gauge_linear": linear["values"][index],
                    "gauge_scrambled": scrambled["values"][index],
                    "noise": noise["values"][index],
                })
            prefix_metrics[str(count)] = {
                "cycles": count,
                "lane_reads_per_cycle": _READOUT_LANE_READS_PER_CYCLE[mode_name],
                "readout_operations_per_pass": (
                    _READOUT_OPS_PER_CYCLE[mode_name] * count),
                "readout_operations": total_operations,
                "readout_seconds": total_seconds,
                "readout_operations_per_sec": (
                    total_operations / total_seconds if total_seconds > 0
                    else 0.0),
                "working_set_bytes_per_pass": (
                    _READOUT_LANE_READS_PER_CYCLE[mode_name] * 4 * count),
                "working_set_bytes_total": lane_reads * 4,
                "clean_exact_vs_truth": clean_exact,
                "clean_mismatches_vs_truth": clean_miss,
                "gauge_linear_exact_vs_truth": linear_truth_exact,
                "gauge_linear_mismatches_vs_truth": linear_truth_miss,
                "gauge_scrambled_exact_vs_truth": scrambled_truth_exact,
                "gauge_scrambled_mismatches_vs_truth": scrambled_truth_miss,
                "gauge_linear_exact_invariance": linear_exact,
                "gauge_linear_mismatches": linear_miss,
                "gauge_scrambled_exact_invariance": scrambled_exact,
                "gauge_scrambled_mismatches": scrambled_miss,
                "uniform_exact_invariance": uniform_exact,
                "uniform_mismatches": uniform_miss,
                "noise_exact_vs_truth": noise_truth_exact,
                "noise_mismatches_vs_truth": noise_truth_miss,
                "noise_exact_invariance": noise_exact,
                "noise_mismatches": noise_miss,
                "clean_digest": readout_digest(clean["values"]),
                "gauge_linear_digest": readout_digest(linear["values"]),
                "gauge_scrambled_digest": readout_digest(scrambled["values"]),
                "noise_digest": readout_digest(noise["values"]),
                "sample_readouts": sample,
            }
        target_per_prefix = _mode_target(mode_name, prefixes)
        checks = _mode_checks(mode_name, prefix_metrics, target_per_prefix)
        mode_reports[mode_name] = {
            "label": _MODE_LABELS[mode_name],
            "readout": _MODE_READOUTS[mode_name],
            "observed": _mode_observed(mode_name, prefix_metrics),
            "measured": {
                "prefixes": prefix_metrics,
                "bytes": _mode_bytes(lane_backend, lanes, mode_name, prefixes),
                "table_build_seconds": sum(
                    table_seconds[field] for field in FIELDS),
                "table_build_operations": sum(
                    table_operations[field] for field in FIELDS),
            },
            "target": {
                "gauge_invariance_expected": _gauge_expected(mode_name),
                "per_prefix": target_per_prefix,
                "max_cycles": MAX_CYCLES,
                "max_steps": MAX_STEPS,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }

    # Control modes reuse the measured cycle and raw readouts.
    uniform_metrics = {}
    noise_metrics = {}
    for count in prefixes:
        clean_cycle = record("clean", "cycle", count)
        clean_raw = record("clean", "raw", count)
        scrambled_cycle = record("gauge_scrambled", "cycle", count)
        scrambled_raw = record("gauge_scrambled", "raw", count)
        noise_cycle = record("noise", "cycle", count)
        noise_raw = record("noise", "raw", count)
        clean_cycle_exact, clean_cycle_miss = counts("clean", "cycle", count)
        clean_raw_exact, clean_raw_miss = counts("clean", "raw", count)
        uniform_cycle_exact, uniform_cycle_miss = invariance(
            "clean", "cycle", count)
        uniform_raw_exact, uniform_raw_miss = invariance("clean", "raw", count)
        noise_cycle_exact, noise_cycle_miss = invariance(
            "noise", "cycle", count, baseline="gauge_scrambled")
        noise_raw_exact, noise_raw_miss = invariance(
            "noise", "raw", count, baseline="gauge_scrambled")
        uniform_seconds = clean_cycle["seconds"] + clean_raw["seconds"]
        uniform_operations = (clean_cycle["operations"]
                              + clean_raw["operations"])
        uniform_reads = clean_cycle["lane_reads"] + clean_raw["lane_reads"]
        noise_seconds = scrambled_cycle["seconds"] + scrambled_raw["seconds"]
        noise_operations = (scrambled_cycle["operations"]
                            + scrambled_raw["operations"])
        noise_reads = scrambled_cycle["lane_reads"] + scrambled_raw["lane_reads"]
        uniform_metrics[str(count)] = {
            "cycles": count,
            "cycle_exact_vs_truth": clean_cycle_exact,
            "cycle_mismatches_vs_truth": clean_cycle_miss,
            "raw_exact_vs_truth": clean_raw_exact,
            "raw_mismatches_vs_truth": clean_raw_miss,
            "cycle_exact_invariance": uniform_cycle_exact,
            "cycle_mismatches": uniform_cycle_miss,
            "raw_exact_invariance": uniform_raw_exact,
            "raw_mismatches": uniform_raw_miss,
            "readout_operations_per_pass": _CONTROL_OPS_PER_CYCLE * count,
            "readout_operations": uniform_operations,
            "readout_seconds": uniform_seconds,
            "readout_operations_per_sec": (
                uniform_operations / uniform_seconds if uniform_seconds > 0
                else 0.0),
            "working_set_bytes_per_pass": (
                _CONTROL_LANE_READS_PER_CYCLE * 4 * count),
            "working_set_bytes_total": uniform_reads * 4,
            "measurement_field": "clean",
            "lane_delta": 0,
            "uniform_gauge_value": gauge_uniform(0),
        }
        noise_metrics[str(count)] = {
            "cycles": count,
            "cycle_exact_vs_truth": counts("noise", "cycle", count)[0],
            "raw_exact_vs_truth": counts("noise", "raw", count)[0],
            "cycle_exact_invariance": noise_cycle_exact,
            "cycle_mismatches": noise_cycle_miss,
            "raw_exact_invariance": noise_raw_exact,
            "raw_mismatches": noise_raw_miss,
            "readout_operations_per_pass": _CONTROL_OPS_PER_CYCLE * count,
            "readout_operations": noise_operations,
            "readout_seconds": noise_seconds,
            "readout_operations_per_sec": (
                noise_operations / noise_seconds if noise_seconds > 0
                else 0.0),
            "working_set_bytes_per_pass": (
                _CONTROL_LANE_READS_PER_CYCLE * 4 * count),
            "working_set_bytes_total": noise_reads * 4,
            "measurement_field": "gauge_scrambled+noise",
            "lane_noise_base": NOISE_BASE,
            "cycle_noise_is_positive": True,
        }
    for mode_name, metrics in (("uniform", uniform_metrics),
                               ("noise", noise_metrics)):
        target_per_prefix = _mode_target(mode_name, prefixes)
        checks = _mode_checks(mode_name, metrics, target_per_prefix)
        mode_reports[mode_name] = {
            "label": _MODE_LABELS[mode_name],
            "readout": _MODE_READOUTS[mode_name],
            "observed": _mode_observed(mode_name, metrics),
            "measured": {
                "prefixes": metrics,
                "bytes": _control_bytes(lane_backend, lanes, mode_name,
                                        prefixes),
                "table_build_seconds": 0.0,
                "table_build_operations": 0,
            },
            "target": {
                "gauge_invariance_expected": _gauge_expected(mode_name),
                "per_prefix": target_per_prefix,
                "max_cycles": MAX_CYCLES,
                "max_steps": MAX_STEPS,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }

    comparison = _comparison(mode_reports, prefixes)
    totals_seconds = time.perf_counter() - run_start
    snapshot_reads = sum(meta["snapshot_read_operations"]
                         for meta in snapshot_meta.values())
    snapshot_seconds = sum(meta["snapshot_seconds"]
                           for meta in snapshot_meta.values())
    total_readout_operations = sum(
        mode["measured"]["prefixes"][str(count)]["readout_operations"]
        for mode in (mode_reports[name] for name in READOUT_MODES)
        for count in prefixes)
    total_readout_seconds = sum(
        mode["measured"]["prefixes"][str(count)]["readout_seconds"]
        for mode in (mode_reports[name] for name in READOUT_MODES)
        for count in prefixes)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ok",
        "seed": seed,
        "config": {
            "heads": HEADS,
            "pair_capacity": PAIR_CAPACITY,
            "cycle_space_dimension": CYCLE_SPACE_DIMENSION,
            "cycle_prefixes": list(prefixes),
            "modes": list(MODES),
            "readout_modes": list(READOUT_MODES),
            "control_modes": list(CONTROL_MODES),
            "steps": steps,
            "backend_requested": backend_kind,
            "backend": lane_backend.kind,
            "round029_triangle_count": ROUND029_TRIANGLES,
            "cycle_basis": (
                "star fundamental cycles anchored at head 0; the first 48 "
                "are the disjoint Round 029 triangles (1+2k, 2+2k)"),
            "lane_count": len(lanes),
            "lane_capacity": PAIR_CAPACITY,
            "lane_value_bytes": lane_backend.lane_value_bytes,
            "lane_cell_bytes": lane_backend.lane_cell_bytes,
            "backend_storage_bytes": lane_backend.storage_bytes,
            "lane_storage_bytes": lane_backend.lane_storage_bytes,
            "identity_arithmetic": "integer",
            "float_used_for_identity": False,
            "deterministic_workload": True,
        },
        "bounds": {
            "min_cycles": MIN_CYCLES,
            "max_cycles": MAX_CYCLES,
            "max_steps": MAX_STEPS,
            "heads": HEADS,
            "pair_capacity": PAIR_CAPACITY,
            "cycle_space_dimension": CYCLE_SPACE_DIMENSION,
            "prefix_count": len(prefixes),
            "note": (
                "Cycle counts are bounded by the 32,385-dimensional star "
                "fundamental-cycle basis; steps bound repeated readout passes "
                "used only for timing."),
        },
        "scope": {
            "resident_cells_available": resident_available(),
            "backend": lane_backend.kind,
            "raw_lane_source": (
                "real ResidentCells pair rails (s+/s- weights)"
                if lane_backend.kind == "resident"
                else "deterministic integer fake lane table"),
            "storage_kind": lane_backend.storage_kind,
            "buffer_kind": ("bytearray" if lane_backend.kind == "resident"
                            else "dictionary"),
            "native_code_used": False,
            "phone_performance_claimed": False,
            "handset_measurement": False,
            "live_model": False,
            "training": False,
            "canonical_merge_performed": False,
            "uniqueness_claim": False,
            "quantum_claim": False,
            "identity_arithmetic": "integer",
            "float_used_for_identity": False,
            "cycle_prefix_scope": (
                "measured star-basis prefixes, not every closed walk and not "
                "every cycle-space linear combination"),
        },
        "targets": {
            "cycle_gauge_invariance_exact": True,
            "conventional_cycle_ties_cycle": True,
            "raw_gauge_invariance_expected": False,
            "wrong_triangle_gauge_invariance_expected": False,
            "uniform_control_invariance_exact": True,
            "noise_control_invariance_mismatches": "all cycles",
            "max_cycles": MAX_CYCLES,
            "max_steps": MAX_STEPS,
            "snapshot_lanes_exact": True,
            "float_used_for_identity": False,
        },
        "measurement": {
            "backend": {
                "kind": lane_backend.kind,
                "storage_kind": lane_backend.storage_kind,
                "storage_bytes": lane_backend.storage_bytes,
                "lane_storage_bytes": lane_backend.lane_storage_bytes,
                "lane_value_bytes": lane_backend.lane_value_bytes,
                "lane_cell_bytes": lane_backend.lane_cell_bytes,
                "write_operations": lane_backend.write_operations,
                "read_operations": lane_backend.read_operations,
            },
            "lanes_required": len(lanes),
            "lanes_written": len(lanes),
            "snapshot_lanes": len(lanes),
            "snapshot_bytes": len(lanes) * 4,
            "snapshot_read_operations": snapshot_reads,
            "snapshot_seconds": snapshot_seconds,
            "snapshot_read_operations_per_sec": (
                snapshot_reads / snapshot_seconds if snapshot_seconds > 0
                else 0.0),
            "lane_setup_seconds": lane_setup_seconds,
            "fields_seconds": fields_seconds,
            "fields": {
                field: dict(meta,
                            table_build_seconds=table_seconds[field],
                            table_build_operations=table_operations[field])
                for field, meta in snapshot_meta.items()
            },
        },
        "totals": {
            "measured": {
                "modes": len(MODES),
                "readout_modes": len(READOUT_MODES),
                "control_modes": len(CONTROL_MODES),
                "cycle_prefixes": len(prefixes),
                "full_cycle_space_included": MAX_CYCLES in prefixes,
                "lanes_required": len(lanes),
                "lanes_written": len(lanes),
                "pair_capacity": PAIR_CAPACITY,
                "backend_storage_bytes": lane_backend.storage_bytes,
                "lane_storage_bytes": lane_backend.lane_storage_bytes,
                "snapshot_bytes": len(lanes) * 4,
                "total_readout_operations": total_readout_operations,
            },
            "timing": {
                "total_wall_seconds": totals_seconds,
                "lane_setup_seconds": lane_setup_seconds,
                "fields_seconds": fields_seconds,
                "readout_seconds": total_readout_seconds,
            },
            "rates": {
                "readout_operations_per_sec": (
                    total_readout_operations / total_readout_seconds
                    if total_readout_seconds > 0 else 0.0),
                "snapshot_read_operations_per_sec": (
                    snapshot_reads / snapshot_seconds
                    if snapshot_seconds > 0 else 0.0),
            },
        },
        "modes": mode_reports,
        "comparison": comparison,
        "merge_contract": {
            "canonical_merge_performed": False,
            "canonical_merge_api_called": False,
            "note": (
                "The benchmark writes independent head/pair lanes and never "
                "merges canonical pairs or universes."),
        },
        "verdict": {
            "passed": False,
            "checks": {},
        },
        "limitations": list(LIMITATIONS),
    }
    checks = _global_checks(report, lane_backend, lanes, prefixes,
                            snapshot_meta)
    report["verdict"]["checks"] = checks
    report["verdict"]["passed"] = all(checks.values())
    report["deterministic_fingerprint"] = deterministic_fingerprint(report)
    return report


_MODE_LABELS = {
    "cycle": "correct anchored-triangle cycle flux",
    "raw": "raw directed pair lane",
    "wrong_triangle": "wrong anchored triangle (shifted spoke)",
    "conventional_cycle": "conventional flat-edge-table cycle flux",
    "uniform": "uniform vertex-gauge control",
    "noise": "independent per-lane integer noise control",
}

_MODE_READOUTS = {
    "cycle": "x(0,a) + x(a,b) - x(0,b)",
    "raw": "x(a,b)",
    "wrong_triangle": "x(0,a) + x(a,b) - x(0,c), c != a,b",
    "conventional_cycle": "table[(0,a)] + table[(a,b)] - table[(0,b)]",
    "uniform": "cycle and raw under a constant vertex shift",
    "noise": "cycle and raw under positive non-gauge lane noise",
}


def _gauge_expected(mode_name: str) -> bool:
    return mode_name in ("cycle", "conventional_cycle", "uniform")


def _mode_target(mode_name: str, prefixes: Sequence[int]) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    for count in prefixes:
        if mode_name in ("cycle", "conventional_cycle"):
            targets[str(count)] = {
                "cycles": count,
                "clean_exact_vs_truth": count,
                "clean_mismatches_vs_truth": 0,
                "gauge_linear_exact_invariance": count,
                "gauge_linear_mismatches": 0,
                "gauge_scrambled_exact_invariance": count,
                "gauge_scrambled_mismatches": 0,
                "uniform_exact_invariance": count,
                "uniform_mismatches": 0,
                "noise_exact_invariance": 0,
                "noise_mismatches": count,
                "readout_operations_per_pass": 5 * count,
                "working_set_bytes_per_pass": 12 * count,
            }
        elif mode_name == "raw":
            targets[str(count)] = {
                "cycles": count,
                "clean_exact_vs_truth": count,
                "clean_mismatches_vs_truth": 0,
                "gauge_linear_exact_invariance": 0,
                "gauge_linear_mismatches": count,
                "gauge_scrambled_exact_invariance": 0,
                "gauge_scrambled_mismatches": count,
                "uniform_exact_invariance": count,
                "uniform_mismatches": 0,
                "noise_exact_invariance": 0,
                "noise_mismatches": count,
                "readout_operations_per_pass": count,
                "working_set_bytes_per_pass": 4 * count,
            }
        elif mode_name == "wrong_triangle":
            targets[str(count)] = {
                "cycles": count,
                "clean_exact_vs_truth": count,
                "clean_mismatches_vs_truth": 0,
                "gauge_linear_exact_invariance": 0,
                "gauge_linear_mismatches": count,
                "gauge_scrambled_exact_invariance": 0,
                "gauge_scrambled_mismatches": count,
                "uniform_exact_invariance": count,
                "uniform_mismatches": 0,
                "readout_operations_per_pass": 5 * count,
                "working_set_bytes_per_pass": 12 * count,
            }
        elif mode_name == "uniform":
            targets[str(count)] = {
                "cycles": count,
                "cycle_exact_invariance": count,
                "cycle_mismatches": 0,
                "raw_exact_invariance": count,
                "raw_mismatches": 0,
                "readout_operations_per_pass": 6 * count,
                "working_set_bytes_per_pass": 16 * count,
            }
        elif mode_name == "noise":
            targets[str(count)] = {
                "cycles": count,
                "cycle_exact_invariance": 0,
                "cycle_mismatches": count,
                "raw_exact_invariance": 0,
                "raw_mismatches": count,
                "readout_operations_per_pass": 6 * count,
                "working_set_bytes_per_pass": 16 * count,
            }
        else:  # pragma: no cover - guarded by MODES
            raise BenchmarkConfigError("unknown mode: %r" % (mode_name,))
    return targets


def _mode_checks(mode_name: str, metrics: dict[str, Any],
                 targets: dict[str, dict[str, int]]) -> dict[str, bool]:
    checks = {}
    for count_key, expected in targets.items():
        measured = metrics[count_key]
        for metric, wanted in expected.items():
            checks["%s_%s" % (metric, count_key)] = (
                int(measured[metric]) == int(wanted))
    if mode_name in ("cycle", "conventional_cycle"):
        checks["gauge_invariance_observed"] = True
    elif mode_name in ("raw", "wrong_triangle"):
        checks["gauge_failure_observed"] = True
    return checks


def _mode_observed(mode_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    gauge_exact = {}
    for count_key, measured in metrics.items():
        if mode_name in ("uniform", "noise"):
            gauge_exact[count_key] = {
                "cycle": int(measured["cycle_exact_invariance"]),
                "raw": int(measured["raw_exact_invariance"]),
            }
        else:
            gauge_exact[count_key] = {
                "linear": int(measured["gauge_linear_exact_invariance"]),
                "scrambled": int(measured["gauge_scrambled_exact_invariance"]),
            }
    holds = all(
        all(value == int(count_key) for value in values.values())
        for count_key, values in gauge_exact.items())
    return {
        "gauge_invariance_holds": holds,
        "gauge_invariance_expected": _gauge_expected(mode_name),
        "gauge_exact_counts": gauge_exact,
    }


def _mode_bytes(backend: Any, lanes: Sequence[tuple[int, int]], mode_name: str,
                prefixes: Sequence[int]) -> dict[str, Any]:
    return {
        "backend_storage_bytes": backend.storage_bytes,
        "lane_storage_bytes": backend.lane_storage_bytes,
        "storage_kind": backend.storage_kind,
        "len_lanes": len(lanes),
        "snapshot_bytes": len(lanes) * 4,
        "lane_value_bytes": backend.lane_value_bytes,
        "lane_cell_bytes": backend.lane_cell_bytes,
        "working_set_bytes_per_pass": {
            str(count): _READOUT_LANE_READS_PER_CYCLE[mode_name] * 4 * count
            for count in prefixes
        },
        "bytes_per_cycle": _READOUT_LANE_READS_PER_CYCLE[mode_name] * 4,
        "table_bytes_estimate": len(lanes) * 4,
    }


def _control_bytes(backend: Any, lanes: Sequence[tuple[int, int]],
                   mode_name: str, prefixes: Sequence[int]) -> dict[str, Any]:
    return {
        "backend_storage_bytes": backend.storage_bytes,
        "lane_storage_bytes": backend.lane_storage_bytes,
        "storage_kind": backend.storage_kind,
        "len_lanes": len(lanes),
        "snapshot_bytes": len(lanes) * 4,
        "lane_value_bytes": backend.lane_value_bytes,
        "lane_cell_bytes": backend.lane_cell_bytes,
        "working_set_bytes_per_pass": {
            str(count): _CONTROL_LANE_READS_PER_CYCLE * 4 * count
            for count in prefixes
        },
        "bytes_per_cycle": _CONTROL_LANE_READS_PER_CYCLE * 4,
    }


def _comparison(mode_reports: dict[str, Any],
                prefixes: Sequence[int]) -> dict[str, Any]:
    rows = {}
    for count in prefixes:
        key = str(count)
        cycle = mode_reports["cycle"]["measured"]["prefixes"][key]
        raw = mode_reports["raw"]["measured"]["prefixes"][key]
        wrong = mode_reports["wrong_triangle"]["measured"]["prefixes"][key]
        conventional = mode_reports["conventional_cycle"][
            "measured"]["prefixes"][key]
        rows[key] = {
            "cycles": count,
            "cycle_gauge_linear_exact": cycle[
                "gauge_linear_exact_invariance"],
            "cycle_gauge_scrambled_exact": cycle[
                "gauge_scrambled_exact_invariance"],
            "conventional_gauge_linear_exact": conventional[
                "gauge_linear_exact_invariance"],
            "conventional_gauge_scrambled_exact": conventional[
                "gauge_scrambled_exact_invariance"],
            "raw_gauge_linear_exact": raw[
                "gauge_linear_exact_invariance"],
            "raw_gauge_scrambled_exact": raw[
                "gauge_scrambled_exact_invariance"],
            "wrong_gauge_linear_exact": wrong[
                "gauge_linear_exact_invariance"],
            "wrong_gauge_scrambled_exact": wrong[
                "gauge_scrambled_exact_invariance"],
            "cycle_minus_raw_linear_exact": (
                cycle["gauge_linear_exact_invariance"]
                - raw["gauge_linear_exact_invariance"]),
            "cycle_minus_wrong_linear_exact": (
                cycle["gauge_linear_exact_invariance"]
                - wrong["gauge_linear_exact_invariance"]),
            "cycle_minus_raw_scrambled_exact": (
                cycle["gauge_scrambled_exact_invariance"]
                - raw["gauge_scrambled_exact_invariance"]),
            "cycle_minus_wrong_scrambled_exact": (
                cycle["gauge_scrambled_exact_invariance"]
                - wrong["gauge_scrambled_exact_invariance"]),
            "conventional_ties_cycle": all(
                cycle[digest] == conventional[digest]
                for digest in ("clean_digest", "gauge_linear_digest",
                               "gauge_scrambled_digest", "noise_digest")),
        }
    return {
        "prefixes": rows,
        "uniform_control": {
            key: {
                "cycle_exact": mode_reports["uniform"]["measured"][
                    "prefixes"][key]["cycle_exact_invariance"],
                "raw_exact": mode_reports["uniform"]["measured"][
                    "prefixes"][key]["raw_exact_invariance"],
            }
            for key in rows
        },
        "noise_control": {
            key: {
                "cycle_exact": mode_reports["noise"]["measured"][
                    "prefixes"][key]["cycle_exact_invariance"],
                "raw_exact": mode_reports["noise"]["measured"][
                    "prefixes"][key]["raw_exact_invariance"],
            }
            for key in rows
        },
    }


def _identity_metrics_are_int(report: dict[str, Any]) -> bool:
    keys = (
        "cycles",
        "clean_exact_vs_truth",
        "clean_mismatches_vs_truth",
        "gauge_linear_exact_invariance",
        "gauge_linear_mismatches",
        "gauge_scrambled_exact_invariance",
        "gauge_scrambled_mismatches",
        "uniform_exact_invariance",
        "uniform_mismatches",
        "noise_exact_invariance",
        "noise_mismatches",
        "readout_operations_per_pass",
        "working_set_bytes_per_pass",
        "cycle_exact_invariance",
        "cycle_mismatches",
        "raw_exact_invariance",
        "raw_mismatches",
    )
    for mode in report["modes"].values():
        for measured in mode["measured"]["prefixes"].values():
            for key in keys:
                if key in measured:
                    value = measured[key]
                    if isinstance(value, bool) or not isinstance(value, int):
                        return False
    return True


def _global_checks(report: dict[str, Any], backend: Any,
                   lanes: Sequence[tuple[int, int]], prefixes: Sequence[int],
                   snapshot_meta: dict[str, dict[str, Any]]) -> dict[str, bool]:
    cycle_mode = report["modes"]["cycle"]
    raw_mode = report["modes"]["raw"]
    wrong_mode = report["modes"]["wrong_triangle"]
    conventional_mode = report["modes"]["conventional_cycle"]
    uniform_mode = report["modes"]["uniform"]
    noise_mode = report["modes"]["noise"]
    keys = [str(count) for count in prefixes]

    cycle_exact = all(
        cycle_mode["measured"]["prefixes"][key][metric]
        == int(key)
        for key in keys
        for metric in ("clean_exact_vs_truth",
                       "gauge_linear_exact_invariance",
                       "gauge_scrambled_exact_invariance"))
    raw_fails = all(
        raw_mode["measured"]["prefixes"][key][metric] == 0
        for key in keys
        for metric in ("gauge_linear_exact_invariance",
                       "gauge_scrambled_exact_invariance"))
    wrong_fails = all(
        wrong_mode["measured"]["prefixes"][key][metric] == 0
        for key in keys
        for metric in ("gauge_linear_exact_invariance",
                       "gauge_scrambled_exact_invariance"))
    conventional_ties = all(
        conventional_mode["measured"]["prefixes"][key][digest]
        == cycle_mode["measured"]["prefixes"][key][digest]
        for key in keys
        for digest in ("clean_digest", "gauge_linear_digest",
                       "gauge_scrambled_digest", "noise_digest"))
    uniform_invariant = all(
        uniform_mode["measured"]["prefixes"][key][metric] == int(key)
        for key in keys
        for metric in ("cycle_exact_invariance", "raw_exact_invariance"))
    noise_breaks = all(
        noise_mode["measured"]["prefixes"][key][metric] == 0
        for key in keys
        for metric in ("cycle_exact_invariance", "raw_exact_invariance"))
    prefix_counts = all(
        report["modes"][mode_name]["measured"]["prefixes"][key]["cycles"]
        == int(key)
        for mode_name in MODES
        for key in keys)
    full_included = MAX_CYCLES in prefixes
    full_coverage = (len(lanes) == PAIR_CAPACITY) if full_included else True
    snapshot_exact = all(
        meta["snapshot_mismatched_lanes"] == 0
        and meta["snapshot_exact_lanes"] == meta["snapshot_lanes"]
        for meta in snapshot_meta.values())
    return {
        "cycle_space_dimension_32385": len(CYCLE_BASIS) == CYCLE_SPACE_DIMENSION,
        "prefix_cycle_counts_exact": prefix_counts,
        "full_pair_capacity_covered": full_coverage,
        "cycle_exact_under_clean_and_both_gauges": cycle_exact,
        "conventional_cycle_ties_cycle": conventional_ties,
        "raw_fails_under_nonuniform_gauges": raw_fails,
        "wrong_triangle_fails_under_nonuniform_gauges": wrong_fails,
        "uniform_control_invariant": uniform_invariant,
        "noise_control_breaks_invariance": noise_breaks,
        "snapshot_lanes_exact": snapshot_exact,
        "integer_identity_counts": _identity_metrics_are_int(report),
        "no_floats_for_identity": not report["scope"]["float_used_for_identity"],
        "no_canonical_merge": not report["scope"]["canonical_merge_performed"],
        "no_native_acceleration": not report["scope"]["native_code_used"],
        "no_phone_performance_claim": not report["scope"][
            "phone_performance_claimed"],
        "no_uniqueness_claim": not report["scope"]["uniqueness_claim"],
        "all_modes_expected_outcome": all(
            mode["passed"] for mode in report["modes"].values()),
    }


# ---------------------------------------------------------------------------
# deterministic fingerprint
# ---------------------------------------------------------------------------
_TIMING_KEYS = {
    "timing",
    "rates",
    "snapshot_seconds",
    "snapshot_read_operations_per_sec",
    "lane_setup_seconds",
    "fields_seconds",
    "readout_seconds",
    "readout_operations_per_sec",
    "table_build_seconds",
    "total_wall_seconds",
}


def _strip_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_timing(item)
            for key, item in value.items()
            if key not in _TIMING_KEYS
        }
    if isinstance(value, list):
        return [_strip_timing(item) for item in value]
    return value


def _fingerprint_payload(report: dict[str, Any]) -> dict[str, Any]:
    return _strip_timing({
        "schema": report["schema"],
        "status": report["status"],
        "seed": report["seed"],
        "config": report["config"],
        "bounds": report["bounds"],
        "scope": report["scope"],
        "targets": report["targets"],
        "measurement": report["measurement"],
        "totals": report["totals"]["measured"],
        "modes": report["modes"],
        "comparison": report["comparison"],
        "merge_contract": report["merge_contract"],
        "verdict": report["verdict"],
        "limitations": report["limitations"],
    })


def deterministic_fingerprint(report: dict[str, Any]) -> str:
    """SHA-256 of every deterministic measurement (timings excluded)."""
    payload = json.dumps(
        _fingerprint_payload(report), sort_keys=True, separators=(",", ":"),
        allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------
def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024.0 or unit == "GiB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f GiB" % size


def render_markdown(report: dict[str, Any]) -> str:
    """Render a stable Markdown view of the cycle benchmark report."""
    config = report["config"]
    scope = report["scope"]
    verdict = report["verdict"]
    prefixes = [str(count) for count in config["cycle_prefixes"]]
    lines = [
        "# NaN lane/cycle readout benchmark",
        "",
        "Integer lane/cycle readout over a %d-head, %d-pair field with a "
        "%d-dimensional star fundamental-cycle basis."
        % (config["heads"], config["pair_capacity"],
           config["cycle_space_dimension"]),
        "",
        "## Run",
        "",
        "| field | value |",
        "|---|---|",
        "| backend | `%s` |" % config["backend"],
        "| raw lane source | %s |" % scope["raw_lane_source"],
        "| cycle prefixes | %s |" % ", ".join(prefixes),
        "| steps (timing repeats) | %d |" % config["steps"],
        "| seed | %d |" % report["seed"],
        "| lanes required / written | %d / %d |"
        % (report["measurement"]["lanes_required"],
           report["measurement"]["lanes_written"]),
        "| pair capacity | %d |" % config["pair_capacity"],
        "| backend storage | %s (%d bytes) |"
        % (_format_bytes(config["backend_storage_bytes"]),
           config["backend_storage_bytes"]),
        "| lane storage | %s (%d bytes) |"
        % (_format_bytes(config["lane_storage_bytes"]),
           config["lane_storage_bytes"]),
        "| snapshot bytes | %d |" % report["measurement"]["snapshot_bytes"],
        "| total wall | %.3f ms |"
        % (report["totals"]["timing"]["total_wall_seconds"] * 1000.0),
        "",
        "## Scope and claims",
        "",
        "| claim | value |",
        "|---|---|",
        "| real ResidentCells used | %s |"
        % ("yes" if scope["backend"] == "resident" else "no"),
        "| deterministic fake lane table used | %s |"
        % ("yes" if scope["backend"] == "fake" else "no"),
        "| native code used | %s |"
        % ("yes" if scope["native_code_used"] else "no"),
        "| phone/handset performance claim | %s |"
        % ("yes" if scope["phone_performance_claimed"] else "no"),
        "| uniqueness claim | %s |"
        % ("yes" if scope["uniqueness_claim"] else "no"),
        "| canonical merge performed | %s |"
        % ("yes" if scope["canonical_merge_performed"] else "no"),
        "| identity arithmetic | %s |" % scope["identity_arithmetic"],
        "| float used for identity | %s |"
        % ("yes" if scope["float_used_for_identity"] else "no"),
        "",
        "This is **not** a native-code benchmark and **not** a phone/handset "
        "performance measurement or claim. It is **not** a uniqueness claim.",
        "",
        "## Cycle basis",
        "",
        "- Complete graph on 256 heads: %d unordered pairs." % config["pair_capacity"],
        "- Star fundamental-cycle basis anchored at head 0: %d cycles."
        % config["cycle_space_dimension"],
        "- First 48 cycles: Round 029 disjoint triangles `(0, 1+2k, 2+2k)`.",
        "- Remaining cycles: lexicographic `(a,b)` with `1 <= a < b < 256`.",
        "- A correct triangle flux is `x(0,a) + x(a,b) - x(0,b)`; the raw lane "
        "is `x(a,b)`; the wrong triangle uses a different spoke `(0,c)`.",
        "",
        "## Summary",
        "",
        "Exact invariance counts are integer equality counts between the clean "
        "field and the named shifted/control field. `cycles` is the prefix "
        "size; `0` for raw/wrong under nonuniform gauges is the expected "
        "control failure.",
        "",
        "| mode | prefix | cycles | linear exact | scrambled exact | uniform exact | noise exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_name in MODES:
        mode = report["modes"][mode_name]
        for key in prefixes:
            measured = mode["measured"]["prefixes"][key]
            if mode_name == "uniform":
                linear = "n/a"
                scrambled = "n/a"
                uniform = ("cycle %d / raw %d"
                           % (measured["cycle_exact_invariance"],
                              measured["raw_exact_invariance"]))
                noise = "n/a"
            elif mode_name == "noise":
                linear = "n/a"
                scrambled = "n/a"
                uniform = "n/a"
                noise = ("cycle %d / raw %d"
                         % (measured["cycle_exact_invariance"],
                            measured["raw_exact_invariance"]))
            else:
                linear = measured["gauge_linear_exact_invariance"]
                scrambled = measured["gauge_scrambled_exact_invariance"]
                uniform = measured["uniform_exact_invariance"]
                noise = measured["noise_exact_invariance"]
            lines.append(
                "| `%s` | %s | %d | %s | %s | %s | %s |"
                % (mode_name, key, measured["cycles"], linear, scrambled,
                   uniform, noise))
    lines += [
        "",
        "## Measured vs target",
        "",
        "Every row labels the measured value and the expected target. "
        "`observed` means the metric is informational rather than a pass/fail "
        "threshold.",
        "",
    ]
    for mode_name in MODES:
        mode = report["modes"][mode_name]
        status = "PASS" if mode["passed"] else "FAIL"
        expected = ("invariant" if mode["target"]["gauge_invariance_expected"]
                    else "expected failure")
        lines += [
            "### `%s` - %s (%s)" % (mode_name, status, expected),
            "",
            "Readout: `%s`." % mode["readout"],
            "",
            "| prefix | metric | measured | target |",
            "|---|---:|---:|---:|",
        ]
        for key in prefixes:
            measured = mode["measured"]["prefixes"][key]
            target = mode["target"]["per_prefix"][key]
            for metric in sorted(target):
                if metric == "cycles":
                    continue
                lines.append(
                    "| %s | %s | %s | %s |"
                    % (key, metric.replace("_", " "),
                       measured[metric], target[metric]))
        lines += [
            "",
            "Checks: %s" % ", ".join(
                "`%s`=%s" % (name, "pass" if value else "fail")
                for name, value in mode["checks"].items()),
            "",
        ]

    lines += [
        "## Operations, rates, and bytes",
        "",
        "| mode | prefix | readout ops/pass | readout ops total | readout ops/s | working-set bytes/pass | working-set bytes total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_name in MODES:
        mode = report["modes"][mode_name]
        for key in prefixes:
            measured = mode["measured"]["prefixes"][key]
            lines.append(
                "| `%s` | %s | %d | %d | %.1f | %d | %d |"
                % (mode_name, key,
                   measured["readout_operations_per_pass"],
                   measured["readout_operations"],
                   measured["readout_operations_per_sec"],
                   measured["working_set_bytes_per_pass"],
                   measured["working_set_bytes_total"]))
    lines += [
        "",
        "Snapshot lanes read from the backend: %d in %.3f ms (%.1f ops/s)."
        % (report["measurement"]["snapshot_read_operations"],
           report["measurement"]["snapshot_seconds"] * 1000.0,
           report["measurement"]["snapshot_read_operations_per_sec"]),
        "Table build time (conventional control): %.3f ms."
        % (report["modes"]["conventional_cycle"]["measured"][
            "table_build_seconds"] * 1000.0),
        "",
        "## Controls comparison",
        "",
        "| prefix | cycle linear exact | raw linear exact | wrong linear exact | conventional linear exact | cycle - raw | cycle - wrong | conventional ties cycle |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    comparison = report["comparison"]["prefixes"]
    for key in prefixes:
        row = comparison[key]
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %d | %s |"
            % (key, row["cycle_gauge_linear_exact"],
               row["raw_gauge_linear_exact"],
               row["wrong_gauge_linear_exact"],
               row["conventional_gauge_linear_exact"],
               row["cycle_minus_raw_linear_exact"],
               row["cycle_minus_wrong_linear_exact"],
               "yes" if row["conventional_ties_cycle"] else "no"))
    lines += [
        "",
        "Under the uniform vertex-gauge control both cycle and raw readouts are "
        "exact; under per-lane integer noise neither is exact. These controls "
        "show that the cycle readout cancels additive vertex gauge, not "
        "arbitrary lane noise.",
        "",
        "## Determinism",
        "",
        "- fixed seed: `%d`" % report["seed"],
        "- deterministic fingerprint: `%s`"
        % report["deterministic_fingerprint"],
        "- wall-clock and throughput fields are excluded from the fingerprint; "
        "integer identity counts, digests, bytes, and target labels are "
        "included.",
        "- identity comparisons use Python integers only; floats appear only in "
        "timing and operations-per-second fields.",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend("- %s" % item for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_reports(report: dict[str, Any],
                  json_path: str | Path | None = None,
                  markdown_path: str | Path | None = None) -> None:
    """Write JSON and/or Markdown reports; parent directories are created."""
    if json_path is not None:
        _write_text(Path(json_path),
                    json.dumps(report, indent=2, sort_keys=True) + "\n")
    if markdown_path is not None:
        _write_text(Path(markdown_path), render_markdown(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Benchmark integer NaN lane/cycle readout against raw "
                     "pair, wrong-triangle, conventional, uniform, and noise "
                     "controls over a 256-head star cycle basis."),
        epilog=("Example: python3 scripts/nan_cycle_benchmark.py "
                "--cycles 48,256,32385 --json /tmp/nan-cycle.json "
                "--markdown /tmp/nan-cycle.md"),
    )
    parser.add_argument("--cycles", default="48,256,32385",
                        help=("comma-separated cycle-prefix sizes in 1..32385, "
                              "strictly increasing (default: 48,256,32385)"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="readout timing repeats in 1..1000000 (default: 1)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="deterministic integer field seed")
    parser.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                        help="auto, resident, or fake (default: auto)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the JSON report to this path")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="write the Markdown report to this path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        prefixes = parse_prefixes(args.cycles)
        steps = validate_steps(args.steps)
        seed = validate_seed(args.seed)
        backend = validate_backend(args.backend)
    except BenchmarkConfigError as exc:
        print("nan_cycle_benchmark: error: %s" % exc, file=sys.stderr)
        return 2

    try:
        report = run_benchmark(prefixes=prefixes, steps=steps, seed=seed,
                               backend=backend)
    except BenchmarkUnavailable as exc:
        print("nan_cycle_benchmark: error: %s" % exc, file=sys.stderr)
        return 2

    write_reports(report, args.json, args.markdown)
    if args.json is None and args.markdown is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = {
            "schema": report["schema"],
            "status": report["status"],
            "backend": report["config"]["backend"],
            "cycle_prefixes": report["config"]["cycle_prefixes"],
            "steps": report["config"]["steps"],
            "seed": report["seed"],
            "deterministic_fingerprint": report["deterministic_fingerprint"],
            "verdict_passed": report["verdict"]["passed"],
            "json": str(args.json) if args.json else None,
            "markdown": str(args.markdown) if args.markdown else None,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
