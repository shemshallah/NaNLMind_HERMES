"""Corrected temporal spine: true rank involution and local coupling updates.

The true mirror is rank r <-> 32639-r, not complement 255-i. For each rank:
B = past signal (s- rail), F = future signal (s+ rail), A = coupling.
Local update: eps = F - A*B; A <- A - eta*eps*conj(B); mirror A <- conj(A).
"""
from __future__ import annotations

PAIR_COUNT = 256 * 255 // 2  # 32,640


def rank_of_pair(i: int, j: int) -> int:
    if not 0 <= i < j <= 255:
        raise ValueError("pair requires 0 <= i < j <= 255")
    return i * (511 - i) // 2 + (j - i - 1)


def pair_at(rank: int) -> tuple[int, int]:
    if not 0 <= rank < PAIR_COUNT:
        raise ValueError("rank must be 0..32639")
    remaining = rank
    for i in range(256):
        width = 255 - i
        if remaining < width:
            return i, i + 1 + remaining
        remaining -= width
    raise ValueError("rank out of range")


def mirror_rank(rank: int) -> int:
    return PAIR_COUNT - 1 - rank


def mirror_pair(rank: int) -> tuple[int, int]:
    return pair_at(mirror_rank(rank))


class Spine:
    """A sampled set of rank addresses with B/F signals and complex couplings."""

    def __init__(self, ranks: list[int] | None = None):
        self.ranks = ranks or list(range(0, 128))
        for rank in self.ranks:
            if not 0 <= rank < PAIR_COUNT:
                raise ValueError("rank out of range")
        self.couplings: dict[int, complex] = {rank: 0j for rank in self.ranks}
        self.past: dict[int, complex] = {}
        self.future: dict[int, complex] = {}

    def load(self, past: dict[int, complex], future: dict[int, complex]) -> None:
        self.past = dict(past)
        self.future = dict(future)

    def error(self) -> float:
        total = 0.0
        count = 0
        for rank in self.ranks:
            if rank in self.past and rank in self.future:
                total += abs(self.future[rank] - self.couplings[rank] * self.past[rank])
                count += 1
        return total / count if count else 0.0

    def train(self, epochs: int = 100, eta: float = 0.05) -> list[float]:
        history = []
        for _ in range(epochs):
            for rank in self.ranks:
                if rank not in self.past or rank not in self.future:
                    continue
                b = self.past[rank]
                error = self.future[rank] - self.couplings[rank] * b
                self.couplings[rank] += (eta * error * b.conjugate()
                                         / (abs(b) ** 2 + 1e-9))
                mirror = mirror_rank(rank)
                if mirror in self.couplings:
                    self.couplings[mirror] = self.couplings[rank].conjugate()
            history.append(self.error())
        return history

    def mirror_residual(self) -> float:
        worst = 0.0
        for rank, value in self.couplings.items():
            mirror = mirror_rank(rank)
            if mirror in self.couplings:
                worst = max(worst, abs(self.couplings[mirror] - value.conjugate()))
        return worst

    def status(self) -> dict:
        return {
            "ranks": len(self.ranks),
            "pairs": [pair_at(rank) for rank in self.ranks[:4]],
            "mirrors": [mirror_rank(rank) for rank in self.ranks[:4]],
            "mirror_residual": self.mirror_residual(),
            "error": self.error(),
        }
