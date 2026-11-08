"""Exact multi-hop traversal over the typed WordNet graph.

This is symbolic graph composition with the correct source synset supplied. It
is not natural-language understanding and does not learn definitions.
"""
from __future__ import annotations


def parents(record: dict) -> list[int]:
    parent_offsets = list(record.get("hypernyms", []))
    for pointer in record.get("pointers", []):
        if pointer.get("relation") in {"hypernym", "instance_hypernym"}:
            parent_offsets.append(pointer.get("target_offset"))
    return [offset for offset in parent_offsets if offset is not None]


def ancestors(synsets: dict, start: int, max_depth: int = 6) -> dict[int, int]:
    """Return ancestor offset -> shortest hop depth within max_depth."""
    found: dict[int, int] = {}
    frontier = [start]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier = []
        for node in frontier:
            record = synsets.get(node, {})
            for parent in parents(record):
                if parent != start and parent not in found:
                    found[parent] = depth
                    next_frontier.append(parent)
        frontier = next_frontier
    return found


def is_ancestor(synsets: dict, start: int, target: int,
                max_depth: int = 6) -> bool:
    return target in ancestors(synsets, start, max_depth)
