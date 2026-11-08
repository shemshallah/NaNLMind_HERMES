"""Persistent gauge-preserving cycle state.

The state stores complex edge values. Updates are projected onto the
divergence-free cycle subspace after every step, so adding an arbitrary vertex
potential cannot change the stored relational content. This is classical linear
algebra; it does not claim a quantum state.
"""
from __future__ import annotations

from nanlmind_hermes.gauge_memory import project_cycle


def _normalise(edge):
    first, second = edge
    return (first, second) if first < second else (second, first)


class CycleState:
    def __init__(self, nodes: int, decay: float = 1.0):
        if nodes < 2:
            raise ValueError("need at least two nodes")
        self.nodes = nodes
        self.decay = decay
        self.values: dict[tuple[int, int], complex] = {}

    def observe(self, observations: dict, rate: float = 0.5) -> None:
        """Blend observations, then remove the vertex-gradient component."""
        merged = {edge: value for edge, value in self.values.items()}
        for edge, value in observations.items():
            key = _normalise(edge)
            current = merged.get(key, 0j)
            merged[key] = self.decay * current + rate * complex(value)
        self.values = project_cycle(self.nodes, merged)

    def project(self) -> None:
        self.values = project_cycle(self.nodes, self.values)

    def apply_diagonal(self, strengths: dict[tuple[int, int], complex]) -> None:
        """Apply P_cycle D P_cycle to the stored cycle state.

        This is a classical projected operator. It is the state-update form of
        the nonuniform Hamiltonian proposed in the pair-space experiments, not
        a physical quantum evolution.
        """
        product = {}
        for edge, value in self.values.items():
            strength = strengths.get(edge, strengths.get((edge[1], edge[0]), 1.0))
            product[edge] = strength * value
        self.values = project_cycle(self.nodes, product)

    def with_gauge(self, potential: list[complex]) -> "CycleState":
        """Return the same cycle content represented in another vertex gauge."""
        if len(potential) != self.nodes:
            raise ValueError("gauge dimension mismatch")
        shifted = {}
        for (first, second), value in self.values.items():
            shifted[(first, second)] = value + potential[second] - potential[first]
        result = CycleState(self.nodes, self.decay)
        result.values = project_cycle(self.nodes, shifted)
        return result

    def snapshot(self) -> dict:
        return {"nodes": self.nodes, "decay": self.decay,
                "values": dict(self.values)}

    def restore(self, snapshot: dict) -> None:
        if snapshot["nodes"] != self.nodes:
            raise ValueError("snapshot node mismatch")
        self.decay = snapshot["decay"]
        self.values = dict(snapshot["values"])

    def energy(self) -> float:
        return sum(abs(value) ** 2 for value in self.values.values())
