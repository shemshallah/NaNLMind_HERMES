"""Gauge-invariant phase memory without quantum claims.

A memory key is a unit phasor per channel. A query may carry an arbitrary global
phase and independent per-channel phase noise. The invariant score is

    |sum_j conj(q_j) p_j|^2

which cancels a global phase but not independent noise. The same score is
exactly reproducible with real cosine/sine channels:

    (sum_j cos(q_j-p_j))^2 + (sum_j sin(q_j-p_j))^2.

The naive phase-sensitive score is Re sum_j conj(q_j) p_j. This is classical
signal processing; it does not establish quantum behavior or a quantum speedup.
"""
from __future__ import annotations

import cmath
import math
import random

Phases = list[complex]


def random_key(dimensions: int, rng: random.Random) -> Phases:
    return [cmath.exp(1j * rng.uniform(0.0, 2.0 * math.pi))
            for _ in range(dimensions)]


def rotate(key: Phases, phase: float) -> Phases:
    factor = cmath.exp(1j * phase)
    return [value * factor for value in key]


def add_noise(key: Phases, rng: random.Random, std: float) -> Phases:
    if std <= 0:
        return list(key)
    result = []
    for value in key:
        result.append(value * cmath.exp(1j * rng.gauss(0.0, std)))
    return [value / abs(value) for value in result]


def complex_gauge_score(query: Phases, prototype: Phases) -> float:
    total = sum(q * p.conjugate() for q, p in zip(query, prototype))
    return abs(total) ** 2


def complex_naive_score(query: Phases, prototype: Phases) -> float:
    return sum((q * p.conjugate()).real for q, p in zip(query, prototype))


def real_channels(key: Phases) -> list[tuple[float, float]]:
    return [(value.real, value.imag) for value in key]


def real_gauge_score(query: Phases, prototype: Phases) -> float:
    cosine = 0.0
    sine = 0.0
    for q, p in zip(real_channels(query), real_channels(prototype)):
        # q = (qc, qs), p = (pc, ps); cos/sin of phase difference.
        cosine += q[0] * p[0] + q[1] * p[1]
        sine += q[1] * p[0] - q[0] * p[1]
    return cosine * cosine + sine * sine


def real_naive_score(query: Phases, prototype: Phases) -> float:
    return sum(q[0] * p[0] + q[1] * p[1]
               for q, p in zip(real_channels(query), real_channels(prototype)))


class PhaseMemory:
    def __init__(self, dimensions: int, seed: int = 0):
        self.dimensions = dimensions
        self.rng = random.Random(seed)
        self.patterns: list[Phases] = []

    def store(self, key: Phases) -> int:
        if len(key) != self.dimensions:
            raise ValueError("key dimension mismatch")
        self.patterns.append(list(key))
        return len(self.patterns) - 1

    def scores(self, query: Phases, gauge: bool = True) -> list[float]:
        if gauge:
            return [complex_gauge_score(query, pattern) for pattern in self.patterns]
        return [complex_naive_score(query, pattern) for pattern in self.patterns]

    def retrieve(self, query: Phases, gauge: bool = True) -> int:
        scores = self.scores(query, gauge)
        return max(range(len(scores)), key=lambda index: scores[index])


def evaluate(dimensions: int, patterns: int, queries: int, seed: int = 0,
             noise_std: float = 0.0):
    rng = random.Random(seed)
    memory = PhaseMemory(dimensions, seed + 1)
    for _ in range(patterns):
        memory.store(random_key(dimensions, rng))
    correct_gauge = 0
    correct_naive = 0
    real_agreements = 0
    for _ in range(queries):
        index = rng.randrange(patterns)
        query = rotate(memory.patterns[index], rng.uniform(0.0, 2.0 * math.pi))
        query = add_noise(query, rng, noise_std)
        correct_gauge += memory.retrieve(query, gauge=True) == index
        correct_naive += memory.retrieve(query, gauge=False) == index
        complex_scores = memory.scores(query, gauge=True)
        real_scores = [real_gauge_score(query, pattern) for pattern in memory.patterns]
        real_agreements += (max(range(patterns), key=lambda i: real_scores[i])
                            == max(range(patterns), key=lambda i: complex_scores[i]))
    return {
        "dimensions": dimensions,
        "patterns": patterns,
        "queries": queries,
        "noise_std": noise_std,
        "gauge_accuracy": correct_gauge / queries,
        "naive_accuracy": correct_naive / queries,
        "real_channel_agreement": real_agreements / queries,
    }
