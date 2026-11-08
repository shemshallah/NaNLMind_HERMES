"""Robust recovery of gauge-flat relational fields from corrupted edges.

For a zero-curvature field x[i,j] = theta[j] - theta[i], every two-hop path
i -> k -> j gives the same value. A median over those paths rejects a minority
of independently corrupted edges without knowing which edges are bad. This is
classical robust estimation, not a decoder for the independent cycle space.

The pure-Python implementation is intentionally dependency-free; scripts may use
NumPy for the full 256-node audit.
"""
from __future__ import annotations

import random

Edge = tuple[int, int]


def _ordered(edge: Edge) -> Edge:
    first, second = edge
    return (first, second) if first < second else (second, first)


def _oriented(values: dict[Edge, float], first: int, second: int) -> float:
    if first == second:
        return 0.0
    if first < second:
        return values.get((first, second), 0.0)
    return -values.get((second, first), 0.0)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def two_hop_median(n: int, values: dict[Edge, float]) -> dict[Edge, float]:
    """Median of all paths i -> k -> j, excluding the direct i -> j path."""
    result = {}
    for first in range(n):
        for second in range(first + 1, n):
            paths = [_oriented(values, first, middle)
                     + _oriented(values, middle, second)
                     for middle in range(n)
                     if middle != first and middle != second]
            result[(first, second)] = median(paths)
    return result


def two_hop_mean(n: int, values: dict[Edge, float]) -> dict[Edge, float]:
    """Non-robust mean of two-hop paths; rejects no outliers."""
    result = {}
    for first in range(n):
        for second in range(first + 1, n):
            paths = [_oriented(values, first, middle)
                     + _oriented(values, middle, second)
                     for middle in range(n)
                     if middle != first and middle != second]
            result[(first, second)] = sum(paths) / len(paths)
    return result


def gradient_field(theta: list[float]) -> dict[Edge, float]:
    return {(first, second): theta[second] - theta[first]
            for first in range(len(theta))
            for second in range(first + 1, len(theta))}


def corrupt(values: dict[Edge, float], fraction: float,
            rng: random.Random, low: float = 1.0,
            high: float = 4.0) -> tuple[dict[Edge, float], set[Edge]]:
    """Corrupt independently selected edges with signed bounded outliers."""
    result = dict(values)
    edges = list(result)
    count = round(len(edges) * fraction)
    selected = set(rng.sample(edges, count))
    for edge in selected:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        result[edge] += sign * rng.uniform(low, high)
    return result, selected


def mean_absolute_error(first: dict[Edge, float],
                        second: dict[Edge, float]) -> float:
    return sum(abs(first.get(edge, 0.0) - second.get(edge, 0.0))
               for edge in second) / len(second)


def flat_recovery_trial(n: int, fraction: float, seed: int) -> dict:
    rng = random.Random(seed)
    theta = [rng.uniform(-2.0, 2.0) for _ in range(n)]
    truth = gradient_field(theta)
    observed, _corrupted = corrupt(truth, fraction, rng)
    median_field = two_hop_median(n, observed)
    mean_field = two_hop_mean(n, observed)
    return {
        "n": n,
        "fraction": fraction,
        "raw_mae": mean_absolute_error(observed, truth),
        "mean_mae": mean_absolute_error(mean_field, truth),
        "median_mae": mean_absolute_error(median_field, truth),
    }


def nonflat_control(n: int, seed: int) -> dict:
    """A genuinely non-flat field is not recovered by the median filter."""
    rng = random.Random(seed)
    values = {}
    for first in range(n):
        for second in range(first + 1, n):
            values[(first, second)] = rng.gauss(0.0, 1.0)
    median_field = two_hop_median(n, values)
    return {
        "n": n,
        "original_std": (sum(value * value for value in values.values())
                         / len(values)) ** 0.5,
        "median_mae": mean_absolute_error(median_field, values),
    }
