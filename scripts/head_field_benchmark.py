#!/usr/bin/env python3
"""Benchmark the real 256-head resident cell field.

This script drives ``nanlmind_hermes.resident.cells.ResidentCells`` over a
fresh 2 MiB ``bytearray``.  It measures four deterministic activation modes:

* ``independent`` - every requested head receives its own per-head updates;
* ``coupled`` - heads receive updates while their pair cells are observed in
  both directions, so the field is driven by actual pair coupling;
* ``single_head`` - control: only head 0 is activated;
* ``uniform`` - control: every head receives the same activation update.

For every mode it records per-head update operations, pair-coupling update
operations, wall time, update rate, allocated/region bytes, head-activation
entropy, active-head count, the full active-head distribution, and address
collision / canonical aliasing rates.

The measurements are host CPython measurements of a synthetic activation
workload on a real resident-cell buffer.  They are not a native-code
benchmark, not a phone/handset performance claim, and do not perform a
canonical merge or any model training/inference.

Examples::

    python3 scripts/head_field_benchmark.py --heads 256 --steps 64 \\
        --json /tmp/head-field.json --markdown /tmp/head-field.md

    python3 scripts/head_field_benchmark.py --heads 16 --steps 8 --seed 7

If the resident cells module is unavailable the script writes a deterministic
``status: skipped`` report instead of failing with an import error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "nanlmind-hermes-head-field-benchmark/1"
DEFAULT_HEADS = 256
MAX_HEADS = 256
DEFAULT_STEPS = 64
MAX_STEPS = 1_000_000
DEFAULT_SEED = 20261108
BUFFER_BYTES = 2 * 1024 * 1024
MIB = 1024 * 1024
ACTIVATION_THRESHOLD = 0.0
MODES = ("independent", "coupled", "single_head", "uniform")

LIMITATIONS = [
    "Measured on host CPython over a 2 MiB in-memory bytearray, not native "
    "code, not an mmap file, and not a phone/handset performance claim.",
    "The workload is deterministic synthetic activation, not model inference, "
    "training, semantic quality, or learned behaviour.",
    "Active heads and entropy describe the injected activation distribution, "
    "not semantic capacity or information created by the cell field.",
    "Canonical pair addresses intentionally share one mirror cell; the s+ and "
    "s- rails preserve direction, so zero rail-key collisions is expected.",
    "Very large --steps values can saturate the fixed-point Q16.16 pair "
    "accumulators; generation counts remain exact while amplitude/weight "
    "readouts then reflect saturation.",
    "No canonical merge, promotion, durable user state, or native acceleration "
    "is performed by this benchmark.",
]

try:
    from nanlmind_hermes.resident import cells as resident_cells
except Exception as exc:  # pragma: no cover - exercised only on broken installs
    resident_cells = None
    _IMPORT_ERROR = "%s: %s" % (type(exc).__name__, exc)
else:
    _IMPORT_ERROR = None


class BenchmarkConfigError(ValueError):
    """Raised for out-of-range heads/steps or non-integer arguments."""


class BenchmarkUnavailable(RuntimeError):
    """Raised when the resident cells implementation cannot be used."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkConfigError("%s must be an integer: %r" % (name, value))
    return value


def validate_heads(value: Any) -> int:
    """Return ``value`` if it is an integer in 1..256, else raise."""
    heads = _require_int(value, "heads")
    if not 1 <= heads <= MAX_HEADS:
        raise BenchmarkConfigError(
            "heads must be in [1, %d]: %r" % (MAX_HEADS, value))
    return heads


def validate_steps(value: Any) -> int:
    """Return ``value`` if it is an integer in 1..1000000, else raise."""
    steps = _require_int(value, "steps")
    if not 1 <= steps <= MAX_STEPS:
        raise BenchmarkConfigError(
            "steps must be in [1, %d]: %r" % (MAX_STEPS, value))
    return steps


def validate_seed(value: Any) -> int:
    return _require_int(value, "seed")


def cells_available() -> bool:
    """True when a usable ResidentCells region fits in the 2 MiB buffer."""
    if resident_cells is None:
        return False
    cls = getattr(resident_cells, "ResidentCells", None)
    if cls is None:
        return False
    head_count = getattr(resident_cells, "HEAD_COUNT", None)
    if not isinstance(head_count, int) or head_count < MAX_HEADS:
        return False
    region_bytes = getattr(resident_cells, "REGION_BYTES", None)
    if not isinstance(region_bytes, int) or region_bytes > BUFFER_BYTES:
        return False
    return True


def _layout() -> dict[str, int]:
    if resident_cells is None:
        return {
            "head_capacity": 0,
            "pair_capacity": 0,
            "region_bytes": 0,
            "head_region_bytes": 0,
            "pair_region_bytes": 0,
            "head_record_bytes": 0,
            "rail_record_bytes": 0,
            "pair_cell_bytes": 0,
        }
    return {
        "head_capacity": int(getattr(resident_cells, "HEAD_COUNT", 0)),
        "pair_capacity": int(getattr(resident_cells, "PAIR_COUNT", 0)),
        "region_bytes": int(getattr(resident_cells, "REGION_BYTES", 0)),
        "head_region_bytes": int(getattr(resident_cells, "HEAD_REGION_BYTES", 0)),
        "pair_region_bytes": int(getattr(resident_cells, "PAIR_REGION_BYTES", 0)),
        "head_record_bytes": int(getattr(resident_cells, "HEAD_RECORD_BYTES", 0)),
        "rail_record_bytes": int(getattr(resident_cells, "RAIL_RECORD_BYTES", 0)),
        "pair_cell_bytes": int(getattr(resident_cells, "PAIR_CELL_BYTES", 0)),
    }


# ---------------------------------------------------------------------------
# deterministic workload helpers
# ---------------------------------------------------------------------------
def _formatted_template() -> bytearray:
    """Format one pristine 2 MiB ResidentCells region for copying."""
    if not cells_available():
        raise BenchmarkUnavailable(
            _IMPORT_ERROR or "nanlmind_hermes.resident.cells is unavailable")
    try:
        template = bytearray(BUFFER_BYTES)
        resident_cells.ResidentCells.format_region(template)
    except Exception as exc:
        raise BenchmarkUnavailable(
            "failed to format ResidentCells region: %s: %s"
            % (type(exc).__name__, exc)) from exc
    return template


def _head_profile(heads: int) -> list[float]:
    """Distinct per-head base amplitudes used by independent/coupled modes."""
    return [0.5 + 0.1 * (head % 16) + 0.0001 * head for head in range(heads)]


def _fixed_one() -> int:
    return int(getattr(resident_cells, "FIXED_ONE", 1 << 16))


def _percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _quantiles(values: Sequence[int], fixed_one: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": _percentile(values, 0.0) / fixed_one,
        "p25": _percentile(values, 25.0) / fixed_one,
        "p50": _percentile(values, 50.0) / fixed_one,
        "p75": _percentile(values, 75.0) / fixed_one,
        "p90": _percentile(values, 90.0) / fixed_one,
        "p95": _percentile(values, 95.0) / fixed_one,
        "p99": _percentile(values, 99.0) / fixed_one,
        "max": _percentile(values, 100.0) / fixed_one,
        "mean": (sum(values) / len(values)) / fixed_one,
    }


_ACTIVATION_BUCKETS = (
    ("inactive", 0.0, 0.0),
    ("(0, 0.1%]", 0.0, 0.001),
    ("(0.1%, 0.25%]", 0.001, 0.0025),
    ("(0.25%, 0.5%]", 0.0025, 0.005),
    ("(0.5%, 1%]", 0.005, 0.01),
    ("(1%, 2%]", 0.01, 0.02),
    ("(2%, 5%]", 0.02, 0.05),
    ("(5%, 10%]", 0.05, 0.10),
    ("> 10%", 0.10, 1.0),
)


def _activation_histogram(scores: Sequence[int]) -> list[dict[str, Any]]:
    """Count heads in activation-share buckets; counts sum to all heads."""
    total = sum(score for score in scores if score > 0)
    counts = [0] * len(_ACTIVATION_BUCKETS)
    for score in scores:
        if score <= 0 or total <= 0:
            counts[0] += 1
            continue
        share = score / total
        for index, (_label, _low, high) in enumerate(
                _ACTIVATION_BUCKETS[1:], start=1):
            if share <= high:
                counts[index] += 1
                break
        else:
            counts[-1] += 1
    head_count = len(scores)
    return [
        {
            "bucket": _label,
            "count": count,
            "fraction_of_heads": (count / head_count) if head_count else 0.0,
        }
        for (_label, _low, _high), count in zip(_ACTIVATION_BUCKETS, counts)
    ]


def _field_metrics(scores: Sequence[int]) -> dict[str, Any]:
    """Entropy, active count and full distribution of head activations."""
    fixed_one = _fixed_one()
    active_scores = [score for score in scores if score > ACTIVATION_THRESHOLD]
    active_ids = [head for head, score in enumerate(scores)
                  if score > ACTIVATION_THRESHOLD]
    total = sum(active_scores)
    shares = [score / total for score in active_scores] if total > 0 else []
    entropy = -sum(share * math.log2(share) for share in shares if share > 0)
    if entropy == 0.0:
        entropy = 0.0
    normalized = (entropy / math.log2(len(active_scores))
                  if len(active_scores) > 1 else 0.0)
    ranked = sorted(
        ((score, head) for head, score in enumerate(scores) if score > 0),
        key=lambda item: (-item[0], item[1]))
    top_heads = [
        {
            "head": head,
            "activation_value": score / fixed_one,
            "share": (score / total) if total > 0 else 0.0,
        }
        for score, head in ranked[:5]
    ]
    top_share = max(shares) if shares else 0.0
    return {
        "activation_threshold": ACTIVATION_THRESHOLD,
        "active_head_count": len(active_scores),
        "active_head_ids": active_ids,
        "updated_head_count": sum(1 for score in scores if score != 0),
        "field_entropy_bits": entropy,
        "field_entropy_normalized": normalized,
        "effective_active_heads": 2.0 ** entropy,
        "distinct_activation_levels": len(set(active_scores)),
        "top_head_share": top_share,
        "top_heads": top_heads,
        "activation_share_histogram": _activation_histogram(scores),
        "activation_quantiles": {
            "all_heads": _quantiles(scores, fixed_one),
            "active_heads": _quantiles(active_scores, fixed_one),
        },
    }


def _pair_metrics(cells, cell_keys: dict[int, tuple[int, int]],
                  pair_updates: int) -> dict[str, Any] | None:
    """Read back touched pair cells so coupling is verified, not assumed."""
    if not cell_keys:
        return None
    fixed_one = _fixed_one()
    amplitude_total = 0
    absolute_weight_total = 0
    generation_total = 0
    max_amplitude = 0
    for index in sorted(cell_keys):
        first, second = cell_keys[index]
        state = cells.pair_state(first, second)
        amplitude = int(state.total_amplitude)
        amplitude_total += amplitude
        max_amplitude = max(max_amplitude, amplitude)
        absolute_weight_total += (abs(int(state.s_plus.weight))
                                  + abs(int(state.s_minus.weight)))
        generation_total += int(state.s_plus.generation)
    return {
        "unique_cells": len(cell_keys),
        "total_amplitude": amplitude_total,
        "total_amplitude_value": amplitude_total / fixed_one,
        "max_cell_amplitude": max_amplitude,
        "max_cell_amplitude_value": max_amplitude / fixed_one,
        "total_absolute_weight": absolute_weight_total,
        "total_absolute_weight_value": absolute_weight_total / fixed_one,
        "generation_sum": generation_total,
        "generation_matches_updates": generation_total == pair_updates,
    }


def _addressing_metrics(cells, observed_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    """Collision and canonical-aliasing rates over distinct directed keys."""
    pair_capacity = int(getattr(cells, "pair_count", 0))
    if not observed_pairs:
        return {
            "directed_pair_keys": 0,
            "unique_directed_addresses": 0,
            "address_collisions": 0,
            "address_collision_rate": 0.0,
            "unique_canonical_cells": 0,
            "canonical_aliasing_rate": None,
            "unique_rail_keys": 0,
            "rail_key_collisions": 0,
            "rail_collision_rate": 0.0,
            "pair_cell_capacity": pair_capacity,
            "canonical_pair_coverage": 0.0,
            "semantics": (
                "Distinct directed pair keys only; repeated observations are "
                "accumulated updates, not collisions. Canonical mirror sharing "
                "is intentional and the s+/s- rails preserve orientation."),
        }

    directed = sorted(observed_pairs)
    addresses: list[int] = []
    rail_keys: set[tuple[int, int]] = set()
    canonical_cells: set[int] = set()
    rail_collisions = 0
    for first, second in directed:
        address = resident_cells.address_of(first, second)
        addresses.append(address)
        cell_index = cells.pair_index(first, second)
        canonical_cells.add(cell_index)
        canonical = resident_cells.canonical_address(first, second)
        rail = (resident_cells.RAIL_S_PLUS
                if address == canonical else resident_cells.RAIL_S_MINUS)
        key = (cell_index, rail)
        if key in rail_keys:
            rail_collisions += 1
        rail_keys.add(key)

    unique_addresses = len(set(addresses))
    directed_count = len(directed)
    return {
        "directed_pair_keys": directed_count,
        "unique_directed_addresses": unique_addresses,
        "address_collisions": directed_count - unique_addresses,
        "address_collision_rate": ((directed_count - unique_addresses)
                                   / directed_count),
        "unique_canonical_cells": len(canonical_cells),
        "canonical_aliasing_rate": 1.0 - (len(canonical_cells) / directed_count),
        "unique_rail_keys": len(rail_keys),
        "rail_key_collisions": rail_collisions,
        "rail_collision_rate": rail_collisions / directed_count,
        "pair_cell_capacity": pair_capacity,
        "canonical_pair_coverage": (len(canonical_cells) / pair_capacity
                                    if pair_capacity else 0.0),
        "semantics": (
            "Distinct directed pair keys only; repeated observations are "
            "accumulated updates, not collisions. Canonical mirror sharing is "
            "intentional and the s+/s- rails preserve orientation."),
    }


# ---------------------------------------------------------------------------
# one mode
# ---------------------------------------------------------------------------
def _expected_updates(mode: str, heads: int, steps: int) -> tuple[int, int]:
    if mode == "single_head":
        return steps, 0
    if mode == "coupled":
        if heads == 1:
            return steps, 0
        edges = 1 if heads == 2 else heads
        return heads * steps, 2 * edges * steps
    return heads * steps, 0


def _run_mode(mode: str, heads: int, steps: int, seed: int,
              template: bytearray) -> dict[str, Any]:
    start = time.perf_counter()
    buffer = bytearray(template)
    cells = resident_cells.ResidentCells(buffer)
    setup_seconds = time.perf_counter() - start

    observed_pairs: set[tuple[int, int]] = set()
    cell_keys: dict[int, tuple[int, int]] = {}
    profile = _head_profile(heads)
    head_updates = 0
    pair_updates = 0

    update_start = time.perf_counter()
    if mode == "independent":
        rng = random.Random(seed ^ 0x1DEA5EED)
        for _step in range(steps):
            for head in range(heads):
                delta = profile[head] * (0.9 + 0.2 * rng.random())
                cells.update_head(head, score_delta=delta, state_delta=1)
                head_updates += 1

    elif mode == "coupled":
        rng = random.Random(seed ^ 0xC0FFEE)
        if heads > 1:
            edges = ([(0, 1)] if heads == 2
                     else [(head, (head + 1) % heads)
                           for head in range(heads)])
            for first, second in edges:
                cell_keys[cells.pair_index(first, second)] = (first, second)
        for step in range(steps):
            coupling = [0.0] * heads
            if heads > 1:
                for first, second in edges:
                    weight = 0.25 + 0.75 * rng.random()
                    phase = ((first * 13 + step * 7) % 360) / 360.0
                    cells.observe_pair(first, second, weight=weight, phase=phase)
                    cells.observe_pair(second, first, weight=weight, phase=phase)
                    observed_pairs.add((first, second))
                    observed_pairs.add((second, first))
                    coupling[first] += weight
                    coupling[second] += weight
                    pair_updates += 2
            for head in range(heads):
                delta = (0.25 * profile[head] + 0.1 * coupling[head]
                         + 0.001 * step)
                cells.update_head(head, score_delta=delta, state_delta=1)
                head_updates += 1

    elif mode == "single_head":
        for _step in range(steps):
            cells.update_head(0, score_delta=0.75, state_delta=1)
            head_updates += 1

    elif mode == "uniform":
        for _step in range(steps):
            for head in range(heads):
                cells.update_head(head, score_delta=0.5, state_delta=1)
                head_updates += 1

    else:  # pragma: no cover - guarded by the caller
        raise BenchmarkConfigError("unknown mode: %r" % (mode,))
    update_seconds = time.perf_counter() - update_start

    analysis_start = time.perf_counter()
    records = [cells.read_head(head) for head in range(heads)]
    scores = [int(record.score) for record in records]
    generations = [int(record.generation) for record in records]
    head_generation_sum = sum(generations)
    field = _field_metrics(scores)
    field["updated_head_count"] = sum(1 for value in generations if value > 0)
    pair_field = _pair_metrics(cells, cell_keys, pair_updates)
    addressing = _addressing_metrics(cells, observed_pairs)

    expected_heads, expected_pairs = _expected_updates(mode, heads, steps)
    layout = _layout()
    memory = {
        "buffer_bytes": len(buffer),
        "buffer_mib": len(buffer) / MIB,
        "region_bytes": int(cells.region_bytes),
        "region_mib": int(cells.region_bytes) / MIB,
        "head_region_bytes": layout["head_region_bytes"],
        "pair_region_bytes": layout["pair_region_bytes"],
        "head_record_bytes": layout["head_record_bytes"],
        "rail_record_bytes": layout["rail_record_bytes"],
        "pair_cell_bytes": layout["pair_cell_bytes"],
        "head_capacity": layout["head_capacity"],
        "pair_capacity": layout["pair_capacity"],
        "bytes_per_head": (int(cells.region_bytes) / layout["head_capacity"]
                           if layout["head_capacity"] else None),
        "region_utilization": (int(cells.region_bytes) / len(buffer)
                               if buffer else 0.0),
    }
    total_update_ops = head_updates + pair_updates
    analysis_seconds = time.perf_counter() - analysis_start
    total_wall_seconds = time.perf_counter() - start
    updates = {
        "per_head": head_updates,
        "pair_coupling": pair_updates,
        "total": total_update_ops,
        "expected_per_head": expected_heads,
        "expected_pair_coupling": expected_pairs,
        "per_head_matches_expected": head_updates == expected_heads,
        "pair_coupling_matches_expected": pair_updates == expected_pairs,
        "head_generation_sum": head_generation_sum,
        "head_generations_match_updates": head_generation_sum == head_updates,
        "pair_generation_sum": (pair_field["generation_sum"]
                                if pair_field else 0),
        "pair_generations_match_updates": (
            pair_field["generation_matches_updates"] if pair_field else True),
    }
    rates = {
        "update_operations_per_sec": (total_update_ops / update_seconds
                                      if update_seconds > 0 else 0.0),
        "head_updates_per_sec": (head_updates / update_seconds
                                 if update_seconds > 0 else 0.0),
        "pair_updates_per_sec": (pair_updates / update_seconds
                                 if update_seconds > 0 else 0.0),
        "end_to_end_operations_per_sec": (total_update_ops / total_wall_seconds
                                          if total_wall_seconds > 0 else 0.0),
    }
    measured = {
        "updates": updates,
        "timing": {
            "setup_seconds": setup_seconds,
            "update_seconds": update_seconds,
            "analysis_seconds": analysis_seconds,
            "total_wall_seconds": total_wall_seconds,
        },
        "rates": rates,
        "memory_bytes": len(buffer),
        "memory": memory,
        "field": field,
        "addressing": addressing,
        "pair_field": pair_field,
    }

    target = _mode_target(mode, heads, steps)
    checks = _mode_checks(mode, heads, measured, target)
    return {
        "label": _MODE_LABELS[mode],
        "measured": measured,
        "target": target,
        "checks": checks,
        "passed": all(checks.values()),
    }


_MODE_LABELS = {
    "independent": "independent per-head activation",
    "coupled": "ring-coupled pair-field activation",
    "single_head": "single-head control",
    "uniform": "uniform-field control",
}


def _mode_target(mode: str, heads: int, steps: int) -> dict[str, Any]:
    target: dict[str, Any] = {
        "memory_bytes_max": BUFFER_BYTES,
        "address_collision_rate_max": 0.0,
        "rail_collision_rate_max": 0.0,
    }
    if mode == "independent":
        target.update({
            "active_head_count_min": min(heads, 2),
            "active_head_count_ideal": heads,
            "distinct_activation_levels_min": min(heads, 2),
            "per_head_updates_exact": heads * steps,
            "pair_coupling_updates_exact": 0,
        })
    elif mode == "coupled":
        _heads_expected, pairs_expected = _expected_updates(mode, heads, steps)
        target.update({
            "active_head_count_min": min(heads, 2),
            "active_head_count_ideal": heads,
            "distinct_activation_levels_min": min(heads, 2),
            "per_head_updates_exact": heads * steps,
            "pair_coupling_updates_exact": pairs_expected,
        })
    elif mode == "single_head":
        target.update({
            "active_head_count_exact": 1,
            "distinct_activation_levels_exact": 1,
            "per_head_updates_exact": steps,
            "pair_coupling_updates_exact": 0,
        })
    elif mode == "uniform":
        target.update({
            "active_head_count_exact": heads,
            "distinct_activation_levels_exact": 1,
            "field_entropy_normalized_min": (0.999999 if heads > 1 else 0.0),
            "per_head_updates_exact": heads * steps,
            "pair_coupling_updates_exact": 0,
        })
    return target


def _mode_checks(mode: str, heads: int, measured: dict[str, Any],
                 target: dict[str, Any]) -> dict[str, bool]:
    updates = measured["updates"]
    field = measured["field"]
    addressing = measured["addressing"]
    pair_field = measured["pair_field"]
    checks = {
        "per_head_updates_match": updates["per_head_matches_expected"],
        "pair_coupling_updates_match": updates["pair_coupling_matches_expected"],
        "head_generations_match": updates["head_generations_match_updates"],
        "pair_generations_match": updates["pair_generations_match_updates"],
        "memory_bytes_within_buffer": measured["memory_bytes"] <= BUFFER_BYTES,
        "address_collisions_zero": int(addressing["address_collisions"]) == 0,
        "rail_key_collisions_zero": int(addressing["rail_key_collisions"]) == 0,
    }
    if mode == "independent":
        checks.update({
            "independent_multi_head": (field["active_head_count"] > 1
                                       if heads > 1 else True),
            "independent_distinct_levels": (
                field["distinct_activation_levels"]
                >= target["distinct_activation_levels_min"]),
            "independent_all_requested_heads_active": (
                field["active_head_count"] == heads),
        })
    elif mode == "coupled":
        checks.update({
            "coupled_multi_head": (field["active_head_count"] > 1
                                   if heads > 1 else True),
            "coupled_pair_updates_present": (updates["pair_coupling"] > 0
                                             if heads > 1 else True),
            "coupled_pair_field_present": (pair_field is not None
                                           if heads > 1 else True),
            "coupled_all_requested_heads_active": (
                field["active_head_count"] == heads),
        })
    elif mode == "single_head":
        checks.update({
            "single_head_exactly_one_active": field["active_head_count"] == 1,
            "single_head_is_head_zero": field["active_head_ids"] == [0],
            "single_head_one_distinct_level": (
                field["distinct_activation_levels"] == 1),
            "single_head_histogram_total": (
                sum(item["count"] for item
                    in field["activation_share_histogram"]) == heads),
        })
    elif mode == "uniform":
        checks.update({
            "uniform_all_requested_heads_active": (
                field["active_head_count"] == heads),
            "uniform_one_distinct_level": (
                field["distinct_activation_levels"] == 1),
            "uniform_entropy_near_max": (
                field["field_entropy_normalized"]
                >= target["field_entropy_normalized_min"]),
        })
    return checks


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def _comparison(modes: dict[str, Any]) -> dict[str, Any]:
    independent = modes["independent"]["measured"]["field"]
    coupled = modes["coupled"]["measured"]
    single = modes["single_head"]["measured"]["field"]
    uniform = modes["uniform"]["measured"]["field"]
    return {
        "independent_vs_single_head": {
            "independent_active_heads": independent["active_head_count"],
            "single_head_active_heads": single["active_head_count"],
            "extra_active_heads": (independent["active_head_count"]
                                   - single["active_head_count"]),
            "independent_distinct_levels": independent["distinct_activation_levels"],
            "single_head_distinct_levels": single["distinct_activation_levels"],
        },
        "independent_vs_uniform": {
            "independent_entropy_bits": independent["field_entropy_bits"],
            "uniform_entropy_bits": uniform["field_entropy_bits"],
            "independent_distinct_levels": independent["distinct_activation_levels"],
            "uniform_distinct_levels": uniform["distinct_activation_levels"],
            "independent_top_head_share": independent["top_head_share"],
            "uniform_top_head_share": uniform["top_head_share"],
        },
        "coupled_field": {
            "pair_coupling_updates": coupled["updates"]["pair_coupling"],
            "unique_cells": (coupled["pair_field"]["unique_cells"]
                             if coupled["pair_field"] else 0),
            "active_heads": coupled["field"]["active_head_count"],
            "canonical_aliasing_rate": coupled["addressing"]["canonical_aliasing_rate"],
            "rail_collision_rate": coupled["addressing"]["rail_collision_rate"],
        },
    }


def _skipped_report(heads: int, steps: int, seed: int,
                    reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "skipped",
        "seed": seed,
        "config": {
            "heads": heads,
            "steps": steps,
            "modes": list(MODES),
            "buffer_bytes": BUFFER_BYTES,
            "deterministic_workload": True,
        },
        "scope": {
            "resident_cells_available": False,
            "buffer_kind": "bytearray",
            "native_code_used": False,
            "phone_performance_claimed": False,
            "handset_measurement": False,
            "canonical_merge_performed": False,
        },
        "targets": {},
        "modes": {},
        "comparison": {},
        "merge_contract": {
            "canonical_merge_performed": False,
            "canonical_merge_api_called": False,
            "note": "No merge is attempted when the resident cells are absent.",
        },
        "verdict": {
            "passed": False,
            "skipped": True,
            "reason": reason,
            "checks": {},
        },
        "limitations": list(LIMITATIONS),
    }


def _fingerprint_payload(report: dict[str, Any]) -> dict[str, Any]:
    modes = {}
    for name, mode in report["modes"].items():
        measured = mode["measured"]
        modes[name] = {
            "target": mode["target"],
            "checks": mode["checks"],
            "passed": mode["passed"],
            "updates": measured["updates"],
            "memory_bytes": measured["memory_bytes"],
            "memory": measured["memory"],
            "field": measured["field"],
            "addressing": measured["addressing"],
            "pair_field": measured["pair_field"],
        }
    return {
        "schema": report["schema"],
        "status": report["status"],
        "seed": report["seed"],
        "config": report["config"],
        "scope": report["scope"],
        "targets": report["targets"],
        "totals": report["totals"]["measured"],
        "modes": modes,
        "comparison": report["comparison"],
        "verdict": report["verdict"],
        "merge_contract": report["merge_contract"],
        "limitations": report["limitations"],
    }


def deterministic_fingerprint(report: dict[str, Any]) -> str:
    """SHA-256 of every deterministic measurement (timings excluded)."""
    payload = json.dumps(
        _fingerprint_payload(report), sort_keys=True, separators=(",", ":"),
        allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_benchmark(heads: int = DEFAULT_HEADS, steps: int = DEFAULT_STEPS,
                  *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run all four modes and return a JSON-serialisable report."""
    heads = validate_heads(heads)
    steps = validate_steps(steps)
    seed = validate_seed(seed)
    if not cells_available():
        raise BenchmarkUnavailable(
            _IMPORT_ERROR or "nanlmind_hermes.resident.cells is unavailable")

    benchmark_start = time.perf_counter()
    template = _formatted_template()
    template_seconds = time.perf_counter() - benchmark_start
    modes_start = time.perf_counter()
    modes = {
        mode: _run_mode(mode, heads, steps, seed, template)
        for mode in MODES
    }
    modes_wall_seconds = time.perf_counter() - modes_start
    total_wall_seconds = time.perf_counter() - benchmark_start
    total_update_operations = sum(
        mode["measured"]["updates"]["total"] for mode in modes.values())
    checks = {
        "independent_multi_head": modes["independent"]["checks"][
            "independent_multi_head"],
        "independent_distinct_levels": modes["independent"]["checks"][
            "independent_distinct_levels"],
        "coupled_multi_head": modes["coupled"]["checks"]["coupled_multi_head"],
        "coupled_pair_updates_present": modes["coupled"]["checks"][
            "coupled_pair_updates_present"],
        "single_head_exactly_one": modes["single_head"]["checks"][
            "single_head_exactly_one_active"],
        "uniform_all_active": modes["uniform"]["checks"][
            "uniform_all_requested_heads_active"],
        "uniform_one_distinct_level": modes["uniform"]["checks"][
            "uniform_one_distinct_level"],
        "address_collisions_zero": all(
            mode["checks"]["address_collisions_zero"]
            for mode in modes.values()),
        "rail_key_collisions_zero": all(
            mode["checks"]["rail_key_collisions_zero"]
            for mode in modes.values()),
        "memory_within_2mib": all(
            mode["checks"]["memory_bytes_within_buffer"]
            for mode in modes.values()),
        "all_modes_passed": all(mode["passed"] for mode in modes.values()),
        "no_canonical_merge": True,
        "no_native_acceleration": True,
        "no_phone_performance_claim": True,
    }
    layout = _layout()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ok",
        "seed": seed,
        "config": {
            "heads": heads,
            "steps": steps,
            "modes": list(MODES),
            "buffer_bytes": BUFFER_BYTES,
            "buffer_mib": BUFFER_BYTES / MIB,
            "head_capacity": layout["head_capacity"],
            "pair_capacity": layout["pair_capacity"],
            "region_bytes": layout["region_bytes"],
            "activation_threshold": ACTIVATION_THRESHOLD,
            "deterministic_workload": True,
        },
        "scope": {
            "resident_cells_available": True,
            "resident_cells_module": "nanlmind_hermes.resident.cells",
            "buffer_kind": "bytearray",
            "buffer_bytes": BUFFER_BYTES,
            "native_code_used": False,
            "phone_performance_claimed": False,
            "handset_measurement": False,
            "live_model": False,
            "training": False,
            "canonical_merge_performed": False,
        },
        "targets": {
            "independent_active_heads_min": min(heads, 2),
            "independent_active_heads_ideal": heads,
            "independent_distinct_levels_min": min(heads, 2),
            "coupled_active_heads_min": min(heads, 2),
            "single_head_active_heads_exact": 1,
            "uniform_active_heads_exact": heads,
            "address_collision_rate_max": 0.0,
            "rail_collision_rate_max": 0.0,
            "memory_bytes_max": BUFFER_BYTES,
        },
        "totals": {
            "measured": {
                "total_update_operations": total_update_operations,
                "memory_bytes": BUFFER_BYTES,
                "modes": len(MODES),
            },
            "timing": {
                "template_setup_seconds": template_seconds,
                "modes_wall_seconds": modes_wall_seconds,
                "total_wall_seconds": total_wall_seconds,
            },
            "rates": {
                "update_operations_per_sec": (
                    total_update_operations / total_wall_seconds
                    if total_wall_seconds > 0 else 0.0),
                "end_to_end_operations_per_sec": (
                    total_update_operations / total_wall_seconds
                    if total_wall_seconds > 0 else 0.0),
            },
        },
        "modes": modes,
        "comparison": _comparison(modes),
        "merge_contract": {
            "canonical_merge_performed": False,
            "canonical_merge_api_called": False,
            "note": (
                "The benchmark only updates pre-formatted head and pair cells; "
                "it never merges canonical pair addresses or universes."),
        },
        "verdict": {
            "passed": all(checks.values()),
            "skipped": False,
            "checks": checks,
        },
        "limitations": list(LIMITATIONS),
    }
    report["deterministic_fingerprint"] = deterministic_fingerprint(report)
    return report


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


def _format_float(value: float | None, places: int = 6) -> str:
    return "n/a" if value is None else ("%%.%df" % places) % value


def _target_text(target: dict[str, Any], *keys: str) -> str:
    parts = []
    for key in keys:
        if key not in target:
            continue
        label = key.replace("_", " ")
        parts.append("%s %s" % (label, target[key]))
    return "; ".join(parts) if parts else "observed"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a stable Markdown view of the benchmark report."""
    if report.get("status") == "skipped":
        return "\n".join([
            "# Head-field activation benchmark",
            "",
            "**SKIPPED**: %s" % report["verdict"]["reason"],
            "",
            "The real `nanlmind_hermes.resident.cells.ResidentCells` region was "
            "not available, so no measurement or performance claim is made.",
            "",
        ])

    config = report["config"]
    scope = report["scope"]
    verdict = report["verdict"]
    lines = [
        "# Head-field activation benchmark",
        "",
        "Real `nanlmind_hermes.resident.cells.ResidentCells` region over a "
        "fresh 2 MiB `bytearray`. The workload is deterministic synthetic "
        "activation, not a live model.",
        "",
        "## Run",
        "",
        "| field | value |",
        "|---|---|",
        "| heads requested / capacity | %d / %d |"
        % (config["heads"], config["head_capacity"]),
        "| steps | %d |" % config["steps"],
        "| seed | %d |" % report["seed"],
        "| buffer | %s (%d bytes) |"
        % (_format_bytes(config["buffer_bytes"]), config["buffer_bytes"]),
        "| resident region | %d bytes |" % config["region_bytes"],
        "| modes | %s |" % ", ".join("`%s`" % mode for mode in config["modes"]),
        "| total update operations | %d |"
        % report["totals"]["measured"]["total_update_operations"],
        "| total wall (region formatting + modes) | %.3f ms |"
        % (report["totals"]["timing"]["total_wall_seconds"] * 1000.0),
        "",
        "## Scope and claims",
        "",
        "| claim | value |",
        "|---|---|",
        "| real ResidentCells used | %s |"
        % ("yes" if scope["resident_cells_available"] else "no"),
        "| native code used | %s |" % ("yes" if scope["native_code_used"] else "no"),
        "| phone/handset performance claim | %s |"
        % ("yes" if scope["phone_performance_claimed"] else "no"),
        "| canonical merge performed | %s |"
        % ("yes" if scope["canonical_merge_performed"] else "no"),
        "| live model / training | no |",
        "",
        "This is **not** a native-code benchmark and **not** a phone/handset "
        "performance measurement or claim.",
        "",
        "## Mode summary",
        "",
        "| mode | active heads | updated heads | entropy bits | normalized entropy | distinct levels | effective heads | top share | per-head ops | pair ops | update ops/s | wall ms | memory bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode_name in MODES:
        mode = report["modes"][mode_name]
        measured = mode["measured"]
        field = measured["field"]
        updates = measured["updates"]
        timing = measured["timing"]
        rates = measured["rates"]
        lines.append(
            "| `%s` | %d | %d | %.6f | %.6f | %d | %.3f | %.6f | %d | %d | %.1f | %.3f | %d |"
            % (mode_name, field["active_head_count"],
               field["updated_head_count"], field["field_entropy_bits"],
               field["field_entropy_normalized"],
               field["distinct_activation_levels"],
               field["effective_active_heads"], field["top_head_share"],
               updates["per_head"], updates["pair_coupling"],
               rates["update_operations_per_sec"],
               timing["total_wall_seconds"] * 1000.0,
               measured["memory_bytes"]))

    lines += [
        "",
        "## Active-head distribution",
        "",
        "Counts are over all requested heads; buckets are each head's share of "
        "the total positive activation. The distribution is reported per mode, "
        "not as one aggregate.",
        "",
        "| activation share bucket | %s |"
        % " | ".join("`%s`" % mode for mode in MODES),
        "|---|---:|---:|---:|---:|",
    ]
    for bucket_index, bucket in enumerate(_ACTIVATION_BUCKETS):
        row = [bucket[0]]
        for mode_name in MODES:
            histogram = report["modes"][mode_name]["measured"]["field"][
                "activation_share_histogram"]
            row.append(str(histogram[bucket_index]["count"]))
        lines.append("| %s | %s |" % (row[0], " | ".join(row[1:])))

    lines += [
        "",
        "## Measured vs target",
        "",
        "Every target is labelled explicitly; `observed` means the metric is "
        "informational rather than a pass/fail threshold.",
        "",
    ]
    for mode_name in MODES:
        mode = report["modes"][mode_name]
        measured = mode["measured"]
        field = measured["field"]
        updates = measured["updates"]
        addressing = measured["addressing"]
        target = mode["target"]
        lines += [
            "### `%s` - %s" % (mode_name, "PASS" if mode["passed"] else "FAIL"),
            "",
            "| metric | measured | target |",
            "|---|---:|---|",
            "| active head count | %d | %s |"
            % (field["active_head_count"],
               _target_text(target, "active_head_count_exact",
                            "active_head_count_min",
                            "active_head_count_ideal")),
            "| distinct activation levels | %d | %s |"
            % (field["distinct_activation_levels"],
               _target_text(target, "distinct_activation_levels_exact",
                            "distinct_activation_levels_min")),
            "| field entropy bits | %.6f | observed |"
            % field["field_entropy_bits"],
            "| normalized entropy | %.6f | %s |"
            % (field["field_entropy_normalized"],
               _target_text(target, "field_entropy_normalized_min")),
            "| top head share | %.6f | observed |" % field["top_head_share"],
            "| per-head update ops | %d | %s |"
            % (updates["per_head"],
               _target_text(target, "per_head_updates_exact")),
            "| pair coupling update ops | %d | %s |"
            % (updates["pair_coupling"],
               _target_text(target, "pair_coupling_updates_exact")),
            "| address collision rate | %.6f | %s |"
            % (addressing["address_collision_rate"],
               _target_text(target, "address_collision_rate_max")),
            "| rail-key collision rate | %.6f | %s |"
            % (addressing["rail_collision_rate"],
               _target_text(target, "rail_collision_rate_max")),
            "| canonical aliasing rate | %s | observed (intentional mirror sharing) |"
            % _format_float(addressing["canonical_aliasing_rate"]),
            "| memory bytes | %d | %s |"
            % (measured["memory_bytes"],
               _target_text(target, "memory_bytes_max")),
            "",
            "Checks: %s" % ", ".join(
                "`%s`=%s" % (key, "pass" if value else "fail")
                for key, value in mode["checks"].items()),
            "",
        ]

    lines += [
        "## Controls comparison",
        "",
        "| comparison | value |",
        "|---|---:|",
    ]
    comparison = report["comparison"]
    independent_single = comparison["independent_vs_single_head"]
    independent_uniform = comparison["independent_vs_uniform"]
    coupled = comparison["coupled_field"]
    lines += [
        "| independent active heads | %d |"
        % independent_single["independent_active_heads"],
        "| single-head control active heads | %d |"
        % independent_single["single_head_active_heads"],
        "| extra active heads vs single-head control | %d |"
        % independent_single["extra_active_heads"],
        "| independent distinct activation levels | %d |"
        % independent_single["independent_distinct_levels"],
        "| single-head distinct levels | %d |"
        % independent_single["single_head_distinct_levels"],
        "| independent entropy bits | %.6f |"
        % independent_uniform["independent_entropy_bits"],
        "| uniform entropy bits | %.6f |"
        % independent_uniform["uniform_entropy_bits"],
        "| independent top head share | %.6f |"
        % independent_uniform["independent_top_head_share"],
        "| uniform top head share | %.6f |"
        % independent_uniform["uniform_top_head_share"],
        "| coupled pair updates | %d |" % coupled["pair_coupling_updates"],
        "| coupled unique pair cells | %d |" % coupled["unique_cells"],
        "| coupled active heads | %d |" % coupled["active_heads"],
        "| coupled canonical aliasing rate | %s |"
        % _format_float(coupled["canonical_aliasing_rate"]),
        "| coupled rail collision rate | %s |"
        % _format_float(coupled["rail_collision_rate"]),
        "",
        "## Pair addressing",
        "",
        "| metric | %s |" % " | ".join("`%s`" % mode for mode in MODES),
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
            ("directed pair keys", "directed_pair_keys"),
            ("unique canonical cells", "unique_canonical_cells"),
            ("address collisions", "address_collisions"),
            ("address collision rate", "address_collision_rate"),
            ("rail-key collisions", "rail_key_collisions"),
            ("rail collision rate", "rail_collision_rate"),
            ("canonical aliasing rate", "canonical_aliasing_rate"),
            ("canonical pair coverage", "canonical_pair_coverage")):
        row = [label]
        for mode_name in MODES:
            value = report["modes"][mode_name]["measured"]["addressing"][key]
            row.append(_format_float(value) if isinstance(value, float)
                       else ("n/a" if value is None else str(value)))
        lines.append("| %s | %s |" % (row[0], " | ".join(row[1:])))

    lines += [
        "",
        "Address collisions are checked over distinct directed pair keys; "
        "repeated observations are accumulated updates, not collisions. "
        "Canonical mirror aliasing is intentional: the two rails preserve the "
        "two directions.",
        "",
        "## Determinism",
        "",
        "- fixed seed: `%d`" % report["seed"],
        "- deterministic fingerprint: `%s`"
        % report["deterministic_fingerprint"],
        "- wall-clock and throughput fields are excluded from the fingerprint; "
        "structural and field measurements are included.",
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
        description=("Benchmark independent and coupled activation of the real "
                     "resident 256-head cell field on a 2 MiB bytearray."),
        epilog=("Example: python3 scripts/head_field_benchmark.py --heads 256 "
                "--steps 64 --json /tmp/head-field.json "
                "--markdown /tmp/head-field.md"),
    )
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS,
                        help="heads to activate in 1..256 (default: 256)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="deterministic steps in 1..1000000 (default: 64)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="deterministic workload seed")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the JSON report to this path")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="write the Markdown report to this path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        heads = validate_heads(args.heads)
        steps = validate_steps(args.steps)
        seed = validate_seed(args.seed)
    except BenchmarkConfigError as exc:
        print("head_field_benchmark: error: %s" % exc, file=sys.stderr)
        return 2

    if not cells_available():
        report = _skipped_report(
            heads, steps, seed,
            _IMPORT_ERROR or "nanlmind_hermes.resident.cells is unavailable")
        write_reports(report, args.json, args.markdown)
        if args.json is None and args.markdown is None:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(json.dumps({
                "schema": report["schema"],
                "status": report["status"],
                "reason": report["verdict"]["reason"],
                "json": str(args.json) if args.json else None,
                "markdown": str(args.markdown) if args.markdown else None,
            }, indent=2, sort_keys=True))
        return 0

    try:
        report = run_benchmark(heads, steps, seed=seed)
    except BenchmarkUnavailable as exc:
        report = _skipped_report(heads, steps, seed, str(exc))
        write_reports(report, args.json, args.markdown)
        print("head_field_benchmark: skipped: %s" % exc, file=sys.stderr)
        return 0

    write_reports(report, args.json, args.markdown)
    if args.json is None and args.markdown is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = {
            "schema": report["schema"],
            "status": report["status"],
            "heads": report["config"]["heads"],
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
