"""Exact mathematics of the K256 pair-address graph and ideal Grover search.

The pair-address graph is the Johnson graph J(n,2): vertices are unordered
pairs, adjacent when they share an endpoint. Its adjacency spectrum is known
exactly, so no 32,640 x 32,640 matrix is materialized. These are classical
calculations; no physical quantum system is involved.
"""
from __future__ import annotations

import math


def pair_count(n: int) -> int:
    return n * (n - 1) // 2


def johnson_spectrum(n: int):
    """Return (eigenvalue, multiplicity) for J(n,2)."""
    if n < 3:
        raise ValueError("Johnson graph J(n,2) needs n >= 3")
    return [
        (2 * n - 4, 1),
        (n - 4, n - 1),
        (-2, pair_count(n) - n),
    ]


def johnson_degree(n: int) -> int:
    return 2 * (n - 2)


def revival_period(n: int) -> float:
    """Period of exp(-i A t) for the integer-spectrum Johnson graph."""
    return math.pi if n % 2 == 0 else 2.0 * math.pi


def return_probability(n: int, time: float) -> float:
    """Probability of returning to the initial pair address at time t."""
    total = pair_count(n)
    terms = 0j
    for eigenvalue, multiplicity in johnson_spectrum(n):
        terms += multiplicity * complex(math.cos(eigenvalue * time),
                                        -math.sin(eigenvalue * time))
    return abs(terms / total) ** 2


def grover_probability(valid_states: int, queries: int) -> float:
    """Ideal Grover success probability for a uniform marked state."""
    if valid_states < 1:
        raise ValueError("valid_states must be positive")
    angle = math.asin(1.0 / math.sqrt(valid_states))
    return math.sin((2 * queries + 1) * angle) ** 2


def best_grover_queries(valid_states: int):
    """Best ideal query count and its success probability near the optimum."""
    centre = max(0, int(round(math.pi / 4.0 * math.sqrt(valid_states) - 0.5)))
    candidates = range(max(0, centre - 3), centre + 4)
    scored = [(grover_probability(valid_states, queries), queries)
              for queries in candidates]
    probability, queries = max(scored)
    return queries, probability
