"""Atomic word-leaf writes into the wave coupling matrix.

A letter is a packet vector u supported on the barrier ring (no void/inner
support). A word is an ordered set of adjacent letter packets. The atomic
update adds symmetric association and directed-order couplings, projects out
the void (inner) subspace, and restores centrosymmetry before adding to A.

Structural constraints:
- A is centrosymmetric: J A J = A.
- The inner/void set is J-invariant, so the diagonal projector P_out commutes
  with J and P_out A P_out preserves centrosymmetry.
- The directed part D is antisymmetric; the symmetric part S is symmetric.
"""
from __future__ import annotations

PAIR_COUNT = 256 * 255 // 2


def zeros(n: int) -> list[list[float]]:
    return [[0.0] * n for _ in range(n)]


def outer(first: list[float], second: list[float]) -> list[list[float]]:
    return [[a * b for b in second] for a in first]


def add(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    n = len(first)
    return [[first[r][c] + second[r][c] for c in range(n)] for r in range(n)]


def scale(matrix: list[list[float]], value: float) -> list[list[float]]:
    n = len(matrix)
    return [[matrix[r][c] * value for c in range(n)] for r in range(n)]


def apply_j(matrix: list[list[float]], permutation: list[int]) -> list[list[float]]:
    """J A J for a permutation J."""
    n = len(matrix)
    return [[matrix[permutation[r]][permutation[c]] for c in range(n)]
            for r in range(n)]


def centrosymmetrize(matrix: list[list[float]], permutation: list[int]) -> list[list[float]]:
    mirrored = apply_j(matrix, permutation)
    n = len(matrix)
    return [[0.5 * (matrix[r][c] + mirrored[r][c]) for c in range(n)]
            for r in range(n)]


def project_out(matrix: list[list[float]], inner: set[int]) -> list[list[float]]:
    n = len(matrix)
    keep = [0.0 if head in inner else 1.0 for head in range(n)]
    return [[matrix[r][c] * keep[r] * keep[c] for c in range(n)]
            for r in range(n)]


def rank_of_pair(i: int, j: int) -> int:
    if not 0 <= i < j <= 255:
        raise ValueError("pair requires 0 <= i < j <= 255")
    return i * (511 - i) // 2 + (j - i - 1)


def pack_state(rank: int, amplitude: int, phase: int) -> int:
    """32-bit NaN16 payload: 15-bit pair rank, 8-bit amplitude, 9-bit phase."""
    if not 0 <= rank < PAIR_COUNT:
        raise ValueError("rank must be 0..32639")
    return ((rank & 0x7FFF) << 17) | ((amplitude & 0xFF) << 9) | (phase & 0x1FF)


def unpack_state(payload: int) -> dict:
    return {"rank": (payload >> 17) & 0x7FFF,
            "amplitude": (payload >> 9) & 0xFF,
            "phase": payload & 0x1FF}


class WordLeafWriter:
    def __init__(self, n: int = 256, inner: set[int] | None = None,
                 permutation: list[int] | None = None):
        self.n = n
        self.inner = inner if inner is not None else {0, 255}
        self.permutation = permutation or [n - 1 - index for index in range(n)]

    def letter_packet(self, heads: list[int], amplitudes: list[float]) -> list[float]:
        vector = [0.0] * self.n
        for head, amplitude in zip(heads, amplitudes):
            if head in self.inner:
                raise ValueError("letter packet cannot occupy the void")
            vector[head % self.n] += amplitude
        return vector

    def _clean(self, delta: list[list[float]]) -> list[list[float]]:
        delta = project_out(delta, self.inner)
        return centrosymmetrize(delta, self.permutation)

    def write_letter(self, matrix: list[list[float]], vector: list[float],
                     alpha: float = 1.0) -> tuple[list[list[float]], dict]:
        delta = self._clean(scale(outer(vector, vector), alpha))
        return add(matrix, delta), {"kind": "letter", "alpha": alpha,
                                    "support": [i for i, v in enumerate(vector) if v]}

    def write_word(self, matrix: list[list[float]], letters: list[list[float]],
                   beta: float = 0.5, gamma: float = 0.25) -> tuple[list[list[float]], dict]:
        symmetric = zeros(self.n)
        directed = zeros(self.n)
        couplings = []
        for left, right in zip(letters, letters[1:]):
            association = outer(left, right)
            reverse = outer(right, left)
            symmetric = add(symmetric, scale(add(association, reverse), beta))
            directed = add(directed, scale(
                [[association[r][c] - reverse[r][c] for c in range(self.n)]
                 for r in range(self.n)], gamma))
            couplings.append({
                "left_support": [i for i, v in enumerate(left) if v],
                "right_support": [i for i, v in enumerate(right) if v],
            })
        delta = self._clean(add(symmetric, directed))
        record = {
            "kind": "word",
            "letters": len(letters),
            "couplings": couplings,
            "beta": beta,
            "gamma": gamma,
            "centrosymmetric": self._residual(delta) < 1e-9,
            "void_zero": all(abs(delta[head][c]) < 1e-12
                             for head in self.inner for c in range(self.n)),
        }
        return add(matrix, delta), record

    def _residual(self, delta: list[list[float]]) -> float:
        mirrored = apply_j(delta, self.permutation)
        return max(abs(delta[r][c] - mirrored[r][c])
                   for r in range(self.n) for c in range(self.n))
