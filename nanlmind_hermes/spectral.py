"""Spectral wave operator over the 256x256 address space.

A[m][n] is the ordered amplitude matrix. The symmetric part is the 32,640
unordered pair cells plus 256 fixed points; the antisymmetric part is the
directed s+/s- rails; the diagonal is the head/polarity self term. Head 0 is
the void/NaNNet anchor. A centrosymmetric matrix is folded into two 128x128
blocks, matching the native mirror/palindrome structure.

This is a mathematical operator, not a quantum claim. "Tunneling" here means
spectral leakage: amplitude of an eigenmode inside the void barrier.
"""
from __future__ import annotations

import math

HEADS = 256
SIDE = 16
CENTER = 7.5
BARRIER_RADIUS = 5.5


def zeros(n: int = HEADS) -> list[list[float]]:
    return [[0.0] * n for _ in range(n)]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[index] * vector[index] for index in range(len(vector)))
            for row in matrix]


def dot(first: list[float], second: list[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def norm(vector: list[float]) -> float:
    return math.sqrt(max(0.0, dot(vector, vector)))


def normalize(vector: list[float]) -> list[float]:
    length = norm(vector)
    return [value / length for value in vector] if length else list(vector)


def power_iteration(matrix: list[list[float]], k: int = 8,
                    iterations: int = 160) -> tuple[list[float], list[list[float]]]:
    """Top-k eigenpairs of a symmetric positive-semidefinite matrix."""
    n = len(matrix)
    values: list[float] = []
    vectors: list[list[float]] = []
    working = [row[:] for row in matrix]
    for index in range(min(k, n)):
        vector = [0.0] * n
        vector[index % n] = 1.0
        vector = normalize([value + 1.0 / math.sqrt(n) for value in vector])
        for _ in range(iterations):
            vector = normalize(matvec(working, vector))
        value = dot(vector, matvec(working, vector))
        values.append(value)
        vectors.append(vector)
        for row in range(n):
            for column in range(n):
                working[row][column] -= value * vector[row] * vector[column]
    return values, vectors


def reverse_permutation(n: int) -> list[int]:
    return [n - 1 - index for index in range(n)]


def native_mirror(n: int = 256) -> list[int]:
    """Native head mirror: reverse the 8-bit head address."""
    return [int(format(index & 0xFF, "08b")[::-1], 2) for index in range(n)]


def complement_mirror(n: int = 256) -> list[int]:
    return [n - 1 - index for index in range(n)]


def j_closure(heads: set[int], permutation: list[int]) -> set[int]:
    result = set(heads)
    for head in list(heads):
        result.add(permutation[head])
    return result


def fiedler_partition(matrix: list[list[float]]) -> dict:
    """Spectral cut: second Laplacian eigenvector partitions the heads."""
    laplacian = graph_laplacian(matrix)
    values, vectors = smallest_eigen(laplacian, k=2)
    vector = vectors[1] if len(vectors) > 1 else vectors[0]
    positive = {head for head, value in enumerate(vector) if value >= 0}
    negative = set(range(len(vector))) - positive
    inner = negative if len(negative) <= len(positive) else positive
    boundary = set()
    threshold = 1e-12
    for head in range(len(matrix)):
        for other in range(head + 1, len(matrix)):
            if matrix[head][other] > threshold and                     (head in positive) != (other in positive):
                boundary.add(head)
                boundary.add(other)
    return {"positive": sorted(positive), "negative": sorted(negative),
            "inner": sorted(inner), "boundary": sorted(boundary),
            "values": [round(value, 6) for value in values],
            "vector": vector}


def normalized_laplacian(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    degrees = [sum(max(0.0, matrix[row][column]) for column in range(n))
               for row in range(n)]
    laplacian = zeros(n)
    for row in range(n):
        for column in range(n):
            if row == column:
                laplacian[row][column] = 1.0 if degrees[row] > 0 else 1.0
            elif degrees[row] > 0 and degrees[column] > 0:
                laplacian[row][column] = -matrix[row][column] / math.sqrt(
                    degrees[row] * degrees[column])
    return laplacian


def cut_potential(matrix: list[list[float]], inner: set[int],
                  lam: float = 1.0) -> list[list[float]]:
    result = [row[:] for row in matrix]
    for head in inner:
        result[head][head] += lam
    return result


def apply_reversal(matrix: list[list[float]], n: int) -> list[list[float]]:
    """J A J for the reversal permutation J."""
    reversed_rows = [row[::-1] for row in matrix[::-1]]
    return reversed_rows


def centrosymmetrize(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    mirrored = apply_reversal(matrix, n)
    return [[0.5 * (matrix[r][c] + mirrored[r][c]) for c in range(n)]
            for r in range(n)]


def mirror_fold(matrix: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Fold a centrosymmetric 256x256 matrix into P = B + C J, M = B - C J."""
    n = len(matrix)
    half = n // 2
    B = [row[:half] for row in matrix[:half]]
    C = [row[half:] for row in matrix[:half]]
    CJ = [row[::-1] for row in C]
    P = [[B[r][c] + CJ[r][c] for c in range(half)] for r in range(half)]
    M = [[B[r][c] - CJ[r][c] for c in range(half)] for r in range(half)]
    return P, M


def graph_laplacian(matrix: list[list[float]]) -> list[list[float]]:
    """L = D - A for a nonnegative symmetric coupling matrix."""
    n = len(matrix)
    laplacian = zeros(n)
    for row in range(n):
        degree = sum(max(0.0, matrix[row][column]) for column in range(n)
                     if column != row)
        for column in range(n):
            laplacian[row][column] = -max(0.0, matrix[row][column]) \
                if row != column else degree
    return laplacian


def smallest_eigen(matrix: list[list[float]], k: int = 4,
                   iterations: int = 80) -> tuple[list[float], list[list[float]]]:
    """Smallest eigenpairs via a positive shift and power iteration."""
    n = len(matrix)
    shift = max(sum(abs(value) for value in row) for row in matrix) + 1.0
    shifted = [[((shift if row == column else 0.0) - matrix[row][column])
                for column in range(n)] for row in range(n)]
    values, vectors = power_iteration(shifted, k=k, iterations=iterations)
    return [shift - value for value in values], vectors


def void_potential(matrix: list[list[float]], lam: float = 1.0) -> list[list[float]]:
    """Add the inverse-square void/centrifugal term to the diagonal."""
    result = [row[:] for row in matrix]
    for head in range(len(result)):
        x = (head % SIDE) - CENTER
        y = (head // SIDE) - CENTER
        radius2 = x * x + y * y
        result[head][head] += lam / (radius2 + 0.5)
    return result


def head_region(head: int) -> str:
    x = (head % SIDE) - CENTER
    y = (head // SIDE) - CENTER
    radius = math.sqrt(x * x + y * y)
    if radius < BARRIER_RADIUS:
        return "inner"
    if radius < BARRIER_RADIUS + 1.5:
        return "ring"
    return "outer"


def mode_regions(vector: list[float]) -> dict:
    mass = {"inner": 0.0, "ring": 0.0, "outer": 0.0}
    for head, value in enumerate(vector):
        mass[head_region(head)] += value * value
    total = sum(mass.values()) or 1.0
    return {region: round(value / total, 6) for region, value in mass.items()}


class WaveMatrix:
    def __init__(self, matrix: list[list[float]] | None = None):
        self.matrix = matrix or zeros()

    @classmethod
    def from_head_population(cls, population) -> "WaveMatrix":
        """Cosine Gram matrix of head letter distributions (a PSD A)."""
        vectors = []
        for head in range(population.head_count):
            counts = population.heads[head]
            total = sum(counts.values())
            if total:
                vectors.append([counts.get(letter, 0) / total
                                for letter in "abcdefghijklmnopqrstuvwxyz$"])
            else:
                vectors.append([0.0] * 27)
        n = len(vectors)
        matrix = zeros(n)
        for i in range(n):
            for j in range(i, n):
                value = dot(vectors[i], vectors[j])
                matrix[i][j] = value
                matrix[j][i] = value
        return cls(matrix)

    def symmetrize(self) -> "WaveMatrix":
        n = len(self.matrix)
        matrix = [[0.5 * (self.matrix[r][c] + self.matrix[c][r])
                   for c in range(n)] for r in range(n)]
        return WaveMatrix(matrix)

    def antisymmetric(self) -> list[list[float]]:
        n = len(self.matrix)
        return [[0.5 * (self.matrix[r][c] - self.matrix[c][r])
                 for c in range(n)] for r in range(n)]

    def centrosymmetrize(self) -> "WaveMatrix":
        return WaveMatrix(centrosymmetrize(self.matrix))

    def apply_void(self, lam: float = 1.0) -> "WaveMatrix":
        return WaveMatrix(void_potential(self.matrix, lam))

    def fold(self) -> tuple[WaveMatrix, WaveMatrix]:
        P, M = mirror_fold(self.matrix)
        return WaveMatrix(P), WaveMatrix(M)

    def eigen(self, k: int = 8) -> tuple[list[float], list[list[float]]]:
        return power_iteration(self.matrix, k=k)

    def centrosymmetry_residual(self) -> float:
        mirrored = apply_reversal(self.matrix, len(self.matrix))
        return max(abs(self.matrix[r][c] - mirrored[r][c])
                   for r in range(len(self.matrix))
                   for c in range(len(self.matrix)))

    def symmetry_residual(self) -> float:
        n = len(self.matrix)
        return max(abs(self.matrix[r][c] - self.matrix[c][r])
                   for r in range(n) for c in range(n))

    def status(self) -> dict:
        return {
            "size": len(self.matrix),
            "symmetric": self.symmetry_residual() < 1e-9,
            "centrosymmetric": self.centrosymmetry_residual() < 1e-9,
            "void_head": 0,
            "barrier_radius": BARRIER_RADIUS,
        }
