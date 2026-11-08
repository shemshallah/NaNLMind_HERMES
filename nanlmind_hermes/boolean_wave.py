"""Boolean-to-wave embedding.

The 16 Boolean functions are embedded as multilinear extensions on amplitudes
in [0, 1]. They are exact on {0, 1} (the Boolean truth table) and smooth fuzzy
logic on [0, 1]; they are NOT a Boolean algebra on complex amplitudes.

AND  = a b
OR   = a + b - a b
XOR  = a + b - 2 a b
NOT  = 1 - a

Phase is a separate temporal channel. Claims that OR/XOR/NOT "preserve phase"
are false for complex amplitudes; magnitude logic and phase dynamics must be
kept distinct.
"""
from __future__ import annotations

BOOL_MASKS = tuple(range(16))


def table(mask: int) -> tuple[int, int, int, int]:
    """Truth table indexed by k = 2a+b."""
    return tuple((mask >> k) & 1 for k in range(4))


def boolean(mask: int, a: int, b: int) -> int:
    return (mask >> ((a << 1) | b)) & 1


def multilinear(mask: int, a: float, b: float) -> float:
    """Exact on {0,1}; a fuzzy-logic extension on [0,1]."""
    f00, f01, f10, f11 = table(mask)
    return (f00 * (1 - a) * (1 - b) + f01 * (1 - a) * b
            + f10 * a * (1 - b) + f11 * a * b)


def negate(a: float, barrier: float = 1.0) -> float:
    return barrier - a


def polarity(coupling: float) -> dict:
    return {
        "yes": coupling > 0,
        "no": coupling < 0,
        "magnitude": abs(coupling),
        "barrier": abs(coupling),
    }


def verify_exact() -> dict:
    """Check all 16 functions on all four Boolean inputs."""
    correct = total = 0
    for mask in BOOL_MASKS:
        for a in (0, 1):
            for b in (0, 1):
                total += 1
                correct += abs(multilinear(mask, a, b) - boolean(mask, a, b)) < 1e-12
    return {"correct": correct, "total": total, "exact": correct == total}
