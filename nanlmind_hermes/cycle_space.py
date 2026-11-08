"""Cycle space over the complete 256-node pair graph.

Edges are unordered pairs (i,j), i<j. The oriented incidence decomposition is
im(G^T) (vertex gradients, dimension n-1) plus ker(B) (cycle space, dimension
C(n,2)-(n-1)). The star spanning tree rooted at 0 gives C(n-1,2) independent
fundamental triangle fluxes. Values may be complex; a phase-sensitive triple
observable measures triangle coherence.

This is classical linear algebra, not quantum hardware.
"""
from __future__ import annotations

import math


def edge_count(n: int) -> int:
    return n * (n - 1) // 2


def cycle_dimension(n: int) -> int:
    return edge_count(n) - (n - 1)


def triangle_count(n: int) -> int:
    return n * (n - 1) * (n - 2) // 6


def decompose(n: int, values: dict[tuple[int, int], complex]):
    """Orthogonal split into a vertex gradient and a divergence-free cycle."""
    potential = [value / n for value in divergence(n, values)]
    gradient = {}
    cycle = {}
    for first in range(n):
        for second in range(first + 1, n):
            key = (first, second)
            value = values.get(key, 0j)
            grad = potential[second] - potential[first]
            gradient[key] = grad
            cycle[key] = value - grad
    return potential, gradient, cycle


def star_decompose(n: int, values: dict[tuple[int, int], complex]):
    """Star-gauge split used for the independent fundamental triangles."""
    potential = [0j] * n
    for head in range(1, n):
        potential[head] = values.get((0, head), 0j)
    gradient = {}
    cycle = {}
    for first in range(n):
        for second in range(first + 1, n):
            key = (first, second)
            value = values.get(key, 0j)
            grad = potential[second] - potential[first]
            gradient[key] = grad
            cycle[key] = value - grad
    return potential, gradient, cycle


def reconstruct(n: int, potential: list[complex],
                cycle: dict[tuple[int, int], complex]) -> dict[tuple[int, int], complex]:
    values = {}
    for first in range(n):
        for second in range(first + 1, n):
            values[(first, second)] = (potential[second] - potential[first]
                                       + cycle.get((first, second), 0j))
    return values


def fundamental_fluxes(n: int, values: dict[tuple[int, int], complex]) -> dict:
    """Independent triangle fluxes for 0 < i < j, using the star tree."""
    _potential, _gradient, cycle = star_decompose(n, values)
    return {(first, second): cycle[(first, second)]
            for first in range(1, n) for second in range(first + 1, n)
            if (first, second) in cycle}


def divergence(n: int, values: dict[tuple[int, int], complex]) -> list[complex]:
    result = [0j] * n
    for (first, second), value in values.items():
        result[first] -= value
        result[second] += value
    return result


def triangle_flux(values: dict[tuple[int, int], complex],
                  first: int, second: int, third: int) -> complex:
    ordered = sorted((first, second, third))
    a, b, c = ordered
    return (values.get((a, b), 0j) + values.get((b, c), 0j)
            - values.get((a, c), 0j))


def triple_coherence(z1: complex, z2: complex, z3: complex) -> float:
    """Phase-sensitive triple observable in [0,1]; magnitudes alone discard it."""
    denominator = abs(z1) * abs(z2) * abs(z3)
    if denominator == 0:
        return 0.0
    return abs((z1 * z2 * z3).real) / denominator


def cycle_inner_product(first: dict, second: dict) -> float:
    return sum((first[key] * second[key].conjugate()).real
               for key in first if key in second)
