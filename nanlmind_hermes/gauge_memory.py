"""Gauge-aware relational memory over the complete pair graph.

Patterns live in the cycle space (divergence-free edge fields). A query may be
contaminated by an arbitrary vertex-gauge field u_j-u_i plus edge noise. The
gauge-aware matcher solves for the nuisance vertex potential and scores the
residual; the naive matcher compares raw edge values. This is the strong
baseline from the NaN cycle-memory experiments: it uses only the observed
edges and does not create information.
"""
from __future__ import annotations


def divergence(n: int, values: dict[tuple[int, int], float]) -> list[float]:
    result = [0.0] * n
    for (first, second), value in values.items():
        result[first] -= value
        result[second] += value
    return result


def project_cycle(n: int, values: dict[tuple[int, int], float]) -> dict:
    div = divergence(n, values)
    potential = [value / n for value in div]
    return {(first, second): value - (potential[second] - potential[first])
            for (first, second), value in values.items()}


def gradient(n: int, potential: list[float]) -> dict:
    return {(first, second): potential[second] - potential[first]
            for first in range(n) for second in range(first + 1, n)}


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            continue
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def gauge_residual(n: int, query: dict, prototype: dict,
                   observed: list[tuple[int, int]], ridge: float = 1e-7) -> float:
    """Residual after fitting a vertex potential to the observed differences."""
    differences = {edge: query[edge] - prototype.get(edge, 0.0) for edge in observed}
    size = n - 1
    normal = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    index = {head: head - 1 for head in range(1, n)}

    def add(first: int, second: int, value: float) -> None:
        if first > 0:
            normal[first - 1][first - 1] += value
        if second > 0:
            normal[second - 1][second - 1] += value
        if first > 0 and second > 0:
            normal[first - 1][second - 1] -= value
            normal[second - 1][first - 1] -= value

    for (first, second), difference in differences.items():
        add(first, second, 1.0)
        if second > 0:
            rhs[second - 1] += difference
        if first > 0:
            rhs[first - 1] -= difference
    for head in range(size):
        normal[head][head] += ridge
    potential = _solve(normal, rhs)
    full = [0.0] + potential
    residual = 0.0
    for (first, second), difference in differences.items():
        residual += (difference - (full[second] - full[first])) ** 2
    return residual


def naive_residual(query: dict, prototype: dict,
                   observed: list[tuple[int, int]]) -> float:
    return sum((query[edge] - prototype.get(edge, 0.0)) ** 2
               for edge in observed)


def retrieve(n: int, patterns: list[dict], query: dict,
             observed: list[tuple[int, int]], gauge: bool = True) -> int:
    residuals = []
    for pattern in patterns:
        if gauge:
            residuals.append(gauge_residual(n, query, pattern, observed))
        else:
            residuals.append(naive_residual(query, pattern, observed))
    return min(range(len(residuals)), key=lambda index: residuals[index])
