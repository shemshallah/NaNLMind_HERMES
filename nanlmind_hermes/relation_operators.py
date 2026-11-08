"""Typed non-Abelian relation operators for directed WordNet relations.

Each directed relationship carries a 2x2 unitary matrix. Composition depends on
order; a closed loop trace is invariant under a consistent vertex-gauge change.
This is classical matrix algebra, not a claim of physical gauge fields.
"""
from __future__ import annotations

import cmath
import math

Matrix = list[list[complex]]


def identity() -> Matrix:
    return [[1 + 0j, 0j], [0j, 1 + 0j]]


def su2(axis: tuple[float, float, float], angle: float) -> Matrix:
    norm = math.sqrt(sum(value * value for value in axis)) or 1.0
    x, y, z = (value / norm for value in axis)
    cosine = math.cos(angle / 2)
    sine = math.sin(angle / 2)
    return [[cosine - 1j * sine * z, -1j * sine * (x - 1j * y)],
            [-1j * sine * (x + 1j * y), cosine + 1j * sine * z]]


def matmul(first: Matrix, second: Matrix) -> Matrix:
    return [[sum(first[row][k] * second[k][column] for k in range(2))
             for column in range(2)] for row in range(2)]


def dagger(matrix: Matrix) -> Matrix:
    return [[matrix[column][row].conjugate() for column in range(2)]
            for row in range(2)]


def commutator(first: Matrix, second: Matrix) -> Matrix:
    left = matmul(first, second)
    right = matmul(second, first)
    return [[left[row][column] - right[row][column] for column in range(2)]
            for row in range(2)]


def frobenius(matrix: Matrix) -> float:
    return math.sqrt(sum(abs(value) ** 2 for row in matrix for value in row))


def unitary_error(matrix: Matrix) -> float:
    product = matmul(dagger(matrix), matrix)
    return frobenius([[product[row][column] - identity()[row][column]
                       for column in range(2)] for row in range(2)])


def gauge_transform(edge: Matrix, source_gauge: Matrix,
                    target_gauge: Matrix) -> Matrix:
    return matmul(matmul(target_gauge, edge), dagger(source_gauge))


def loop_product(edges: list[Matrix]) -> Matrix:
    """Product around a loop listed in traversal order (multiply in reverse)."""
    product = identity()
    for edge in reversed(edges):
        product = matmul(product, edge)
    return product


def loop_trace(edges: list[Matrix]) -> complex:
    product = loop_product(edges)
    return product[0][0] + product[1][1]
