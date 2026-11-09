"""Small exact-interface SO(2)/SO(3) rotation and holonomy utilities.

Round 038's useful mathematical distinction is that commutative 2D phase sums
cannot represent order, while noncommuting 3D rotations can.  This module is a
dependency-free reference for those operations and for the conjugation-
invariant trace of a closed loop.

It is not a trained neural model and does not claim a geometric advantage.  It
provides the stable algebra that the external PyTorch experiments used, so the
resident layer can validate order/holonomy claims without importing torch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

Matrix = tuple[tuple[float, ...], ...]


class RotationError(ValueError):
    """Malformed matrix, angle or axis."""


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RotationError("%s must be a number" % name) from exc
    if not math.isfinite(result):
        raise RotationError("%s must be finite" % name)
    return result


def rotation2(angle: float) -> Matrix:
    theta = _finite(angle, "angle")
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s), (s, c))


def compose2(first: Matrix, second: Matrix) -> Matrix:
    _check_matrix(first, 2)
    _check_matrix(second, 2)
    return ((
        first[0][0] * second[0][0] + first[0][1] * second[1][0],
        first[0][0] * second[0][1] + first[0][1] * second[1][1],
    ), (
        first[1][0] * second[0][0] + first[1][1] * second[1][0],
        first[1][0] * second[0][1] + first[1][1] * second[1][1],
    ))


def commutator_norm2(first: Matrix, second: Matrix) -> float:
    """Frobenius norm of ``AB - BA``; zero for all SO(2) rotations."""
    left = compose2(first, second)
    right = compose2(second, first)
    return math.sqrt(sum((left[r][c] - right[r][c]) ** 2
                         for r in range(2) for c in range(2)))


def givens(axis: int, angle: float) -> Matrix:
    """Right-handed 3D Givens rotation around ``axis`` 0, 1 or 2."""
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise RotationError("axis must be 0, 1 or 2")
    theta = _finite(angle, "angle")
    c, s = math.cos(theta), math.sin(theta)
    rows = [[1.0 if r == k else 0.0 for k in range(3)] for r in range(3)]
    first, second = [(1, 2), (2, 0), (0, 1)][axis]
    rows[first][first] = c
    rows[first][second] = -s
    rows[second][first] = s
    rows[second][second] = c
    return tuple(tuple(row) for row in rows)


def compose3(first: Matrix, second: Matrix) -> Matrix:
    _check_matrix(first, 3)
    _check_matrix(second, 3)
    return tuple(
        tuple(sum(first[r][k] * second[k][c] for k in range(3))
              for c in range(3))
        for r in range(3)
    )


def ordered_product(rotations: Sequence[Matrix]) -> Matrix:
    result: Matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for rotation in rotations:
        result = compose3(result, rotation)
    return result


def commutator_norm3(first: Matrix, second: Matrix) -> float:
    left = compose3(first, second)
    right = compose3(second, first)
    return math.sqrt(sum((left[r][c] - right[r][c]) ** 2
                         for r in range(3) for c in range(3)))


def trace(matrix: Matrix) -> float:
    n = len(matrix)
    _check_matrix(matrix, n)
    return sum(matrix[index][index] for index in range(n))


def conjugation_error(matrix: Matrix, frame: Matrix,
                      inverse: Matrix) -> float:
    """Compare ``trace(M)`` with ``trace(F M F^-1)``."""
    _check_matrix(matrix, 3)
    _check_matrix(frame, 3)
    _check_matrix(inverse, 3)
    transformed = compose3(compose3(frame, matrix), inverse)
    return abs(trace(matrix) - trace(transformed))


def closed_loop_holonomy(first: Matrix, second: Matrix,
                         third: Matrix, inverse_third: Matrix) -> Matrix:
    """Return ``A B C C^-1`` as an explicitly constructed closed loop."""
    return ordered_product((first, second, third, inverse_third))


@dataclass(frozen=True)
class OrderReport:
    phase_commutator: float
    rotation_commutator: float
    holonomy_trace_error: float
    order_sensitive: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "so2_commutator_norm": self.phase_commutator,
            "so3_commutator_norm": self.rotation_commutator,
            "holonomy_trace_error": self.holonomy_trace_error,
            "order_sensitive": self.order_sensitive,
            "canonical_merge": False,
        }


def audit_order(angle_a: float = 0.31, angle_b: float = 0.77) -> OrderReport:
    """Produce the Round 038 order/commutator audit values."""
    p1, p2 = rotation2(angle_a), rotation2(angle_b)
    r1, r2 = givens(0, angle_a), givens(1, angle_b)
    inverse = givens(1, -angle_b)
    loop = closed_loop_holonomy(r1, r2, r1, inverse)
    # The explicit loop is not identity; its trace is the invariant readout.
    error = conjugation_error(loop, r1, transpose(r1))
    return OrderReport(commutator_norm2(p1, p2),
                       commutator_norm3(r1, r2), error,
                       commutator_norm3(r1, r2) > 1e-9)


def transpose(matrix: Matrix) -> Matrix:
    _check_matrix(matrix, len(matrix))
    return tuple(tuple(matrix[c][r] for c in range(len(matrix)))
                 for r in range(len(matrix)))


def _check_matrix(matrix: Matrix, size: int) -> None:
    if not isinstance(matrix, (tuple, list)) or len(matrix) != size:
        raise RotationError("matrix must be %dx%d" % (size, size))
    for row in matrix:
        if not isinstance(row, (tuple, list)) or len(row) != size:
            raise RotationError("matrix must be %dx%d" % (size, size))
        for value in row:
            _finite(value, "matrix value")


__all__ = (
    "RotationError", "OrderReport", "rotation2", "compose2",
    "commutator_norm2", "givens", "compose3", "ordered_product",
    "commutator_norm3", "trace", "transpose", "conjugation_error",
    "closed_loop_holonomy", "audit_order",
)
