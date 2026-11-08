"""Sparse state features for a definition-conditioned answer decoder.

The ordinary state is a hashed ancestry memory. The cycle state projects the
same typed ancestry edges onto the divergence-free K256 cycle basis and then
uses a deterministic sparse projection. Both are classical features, not a
trained language state yet.
"""
from __future__ import annotations

import hashlib

from nanlmind_hermes.gauge_memory import project_cycle
from nanlmind_hermes.wordnet_reasoning import parents


def _bucket(value, dimensions: int) -> int:
    digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimensions


def hash_features(text: str, dimensions: int = 64) -> list[float]:
    values = [0.0] * dimensions
    for token in str(text).casefold().split():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if (value >> 8) & 1 else -1.0
        values[value % dimensions] += sign
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values


def ancestor_edges(nodes: dict, start: int, depth: int = 3) -> list[tuple[int, int]]:
    """Return (child, parent) edges on the ancestry frontier."""
    edges = []
    frontier = [start]
    seen = {start}
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for parent in parents(nodes.get(node, {})):
                if parent in seen or parent not in nodes:
                    continue
                seen.add(parent)
                edges.append((node, parent))
                next_frontier.append(parent)
        frontier = next_frontier
    return edges


def ordinary_state(nodes: dict, start: int, depth: int = 3,
                   dimensions: int = 256) -> list[float]:
    values = [0.0] * dimensions
    edges = ancestor_edges(nodes, start, depth)
    for child, parent in edges:
        values[_bucket(child, dimensions)] += 1.0
        values[_bucket(parent, dimensions)] += 0.5
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values


def cycle_state(nodes: dict, start: int, depth: int = 3,
                dimensions: int = 256, vertices: int = 256) -> list[float]:
    edge_values = {}
    for child, parent in ancestor_edges(nodes, start, depth):
        first = _bucket(child, vertices)
        second = _bucket(parent, vertices)
        if first == second:
            continue
        key = (first, second) if first < second else (second, first)
        edge_values[key] = edge_values.get(key, 0.0) + 1.0
    cycle = project_cycle(vertices, edge_values)
    values = [0.0] * dimensions
    for (first, second), value in cycle.items():
        digest = hashlib.blake2b(f"{first}:{second}".encode(),
                                 digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dimensions
        sign = 1.0 if (int.from_bytes(digest, "little") >> 8) & 1 else -1.0
        values[index] += sign * value
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values] if norm else values
