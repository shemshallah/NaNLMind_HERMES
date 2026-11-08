"""Gauge-corrected coding of low-dimensional semantic cycle fields.

A semantic vector z is encoded as c = Q z, where the columns of Q are an
orthonormal basis of divergence-free edge fields on K_n. A vertex gauge
g[i,j] = phi[j] - phi[i] is added, edges may be erased, and some observations
may be outliers. The decoder fits the nuisance gauge, subtracts it, and projects
back through Q.

This is a classical linear code. It does not add semantic information, does not
use IEEE NaN arithmetic as a resource, and is not a quantum code.
"""
from __future__ import annotations

import math
import random


def pairs(n: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(n)
            for second in range(first + 1, n)]


def cycle_project(n: int, rows: list[list[float]]) -> list[list[float]]:
    """Remove the vertex-gradient component from each edge row."""
    edge_pairs = pairs(n)
    dimensions = len(rows[0]) if rows else 0
    divergence = [[0.0] * dimensions for _ in range(n)]
    for (first, second), row in zip(edge_pairs, rows):
        for column in range(dimensions):
            divergence[first][column] -= row[column]
            divergence[second][column] += row[column]
    potential = [[value / n for value in row] for row in divergence]
    projected = []
    for (first, second), row in zip(edge_pairs, rows):
        projected.append([
            row[column] - (potential[second][column] - potential[first][column])
            for column in range(dimensions)
        ])
    return projected


def _dot(first: list[float], second: list[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _orthonormal_columns(n: int, dimensions: int, rows: list[list[float]]):
    edge_pairs = pairs(n)
    columns = [[rows[edge][column] for edge in range(len(edge_pairs))]
               for column in range(dimensions)]
    orthonormal = []
    for vector in columns:
        for previous in orthonormal:
            projection = _dot(vector, previous)
            vector = [value - projection * basis for value, basis
                      in zip(vector, previous)]
        norm = math.sqrt(_dot(vector, vector))
        if norm < 1e-12:
            raise ValueError("degenerate random code basis")
        orthonormal.append([value / norm for value in vector])
    row_view = [[orthonormal[column][edge] for column in range(len(orthonormal))]
                for edge in range(len(edge_pairs))]
    return orthonormal, row_view


def random_cycle_basis(n: int, dimensions: int, seed: int = 0):
    """Return orthonormal cycle columns and the row view of the same basis."""
    rng = random.Random(seed)
    edge_pairs = pairs(n)
    rows = [[rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
            for _ in edge_pairs]
    rows = cycle_project(n, rows)
    return _orthonormal_columns(n, dimensions, rows)


def random_linear_basis(n: int, dimensions: int, seed: int = 0):
    """Capacity-matched ordinary random linear code, not divergence-free."""
    rng = random.Random(seed)
    rows = [[rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
            for _ in pairs(n)]
    return _orthonormal_columns(n, dimensions, rows)


def encode(row_view: list[list[float]], z: list[float]) -> list[float]:
    return [sum(row[column] * z[column] for column in range(len(z)))
            for row in row_view]


def gauge_values(n: int, potential: list[float]) -> list[float]:
    return [potential[second] - potential[first] for first, second in pairs(n)]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size),
                    key=lambda row: abs(augmented[row][column]))
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


def estimate_gauge(n: int, observed: list[float], mask: list[bool],
                   weights: list[float] | None = None,
                   ridge: float = 1e-8) -> list[float]:
    """Weighted least-squares vertex potential from observed edge differences."""
    matrix = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for edge, (first, second) in enumerate(pairs(n)):
        if not mask[edge]:
            continue
        weight = 1.0 if weights is None else weights[edge]
        if weight == 0.0:
            continue
        value = observed[edge]
        matrix[first][first] += weight
        matrix[second][second] += weight
        matrix[first][second] -= weight
        matrix[second][first] -= weight
        rhs[first] -= weight * value
        rhs[second] += weight * value
    for index in range(n):
        matrix[index][index] += ridge
    return _solve(matrix, rhs)


def decode_cycle(n: int, columns: list[list[float]], observed: list[float],
                 mask: list[bool], weights: list[float] | None = None) -> list[float]:
    potential = estimate_gauge(n, observed, mask, weights)
    count = sum(mask)
    if count == 0:
        return [0.0] * len(columns)
    residual = []
    for edge, (first, second) in enumerate(pairs(n)):
        if mask[edge]:
            residual.append(observed[edge] - (potential[second] - potential[first]))
        else:
            residual.append(0.0)
    scale = count / len(observed)
    return [sum(column[edge] * residual[edge] for edge in range(len(residual)))
            / scale for column in columns]


def joint_decode(n: int, columns: list[list[float]],
                 observed: list[float], mask: list[bool],
                 iterations: int = 5, huber: float = 1.0,
                 ridge: float = 1e-8):
    """Robust joint least squares for cycle coefficients and vertex gauge.

    Model: y[i,j] = (Q z)[i,j] + phi[j] - phi[i] + outlier.
    The cycle basis and incidence columns are orthogonal for a complete graph,
    so the clean system is well posed after pinning the constant gauge mode.
    """
    edge_pairs = pairs(n)
    dimensions = len(columns)
    size = dimensions + n
    weights = [1.0] * len(observed)
    params = [0.0] * size
    for _ in range(iterations):
        normal = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size
        for edge, (first, second) in enumerate(edge_pairs):
            if not mask[edge]:
                continue
            weight = weights[edge]
            if weight == 0.0:
                continue
            row = [columns[column][edge] for column in range(dimensions)]
            gauge_row = [0.0] * n
            gauge_row[second] = 1.0
            gauge_row[first] = -1.0
            row.extend(gauge_row)
            value = observed[edge]
            for first_index in range(size):
                if row[first_index] == 0.0:
                    continue
                rhs[first_index] += weight * row[first_index] * value
                for second_index in range(size):
                    if row[second_index] != 0.0:
                        normal[first_index][second_index] += (
                            weight * row[first_index] * row[second_index])
        for index in range(size):
            normal[index][index] += ridge
        params = _solve(normal, rhs)
        squared = []
        for edge, (first, second) in enumerate(edge_pairs):
            if not mask[edge]:
                squared.append(0.0)
                continue
            prediction = sum(columns[column][edge] * params[column]
                             for column in range(dimensions))
            prediction += params[dimensions + second] - params[dimensions + first]
            error = abs(observed[edge] - prediction)
            squared.append(1.0 if error <= huber else huber / error)
        weights = squared
    return params[:dimensions], params[dimensions:]


def robust_joint_decode(n: int, columns: list[list[float]],
                        observed: list[float], mask: list[bool],
                        iterations: int = 5, huber: float = 1.0):
    """Alternate gauge fitting and cycle projection with Huber reweighting."""
    edge_pairs = pairs(n)
    row_view = [[columns[column][edge] for column in range(len(columns))]
                for edge in range(len(edge_pairs))]
    weights = [1.0] * len(observed)
    potential = estimate_gauge(n, observed, mask, weights)
    for _ in range(iterations):
        count = sum(mask)
        if count == 0:
            break
        scale = count / len(observed)
        residual = []
        for edge, (first, second) in enumerate(edge_pairs):
            residual.append(observed[edge] - (potential[second] - potential[first])
                            if mask[edge] else 0.0)
        z = [sum(column[edge] * residual[edge]
                 for edge in range(len(residual))) / scale
             for column in columns]
        cycle = encode(row_view, z)
        target = [observed[edge] - cycle[edge] for edge in range(len(observed))]
        for _inner in range(3):
            potential = estimate_gauge(n, target, mask, weights)
            squared = []
            for edge, (first, second) in enumerate(edge_pairs):
                if not mask[edge]:
                    squared.append(0.0)
                    continue
                error = abs(target[edge] - (potential[second] - potential[first]))
                squared.append(1.0 if error <= huber else huber / error)
            weights = squared
    count = sum(mask)
    if count == 0:
        return [0.0] * len(columns), potential
    scale = count / len(observed)
    residual = []
    for edge, (first, second) in enumerate(edge_pairs):
        residual.append(observed[edge] - (potential[second] - potential[first])
                        if mask[edge] else 0.0)
    z = [sum(column[edge] * residual[edge]
             for edge in range(len(residual))) * len(observed) / count
         for column in columns]
    return z, potential
