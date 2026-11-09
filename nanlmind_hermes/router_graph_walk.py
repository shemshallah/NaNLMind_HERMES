"""Bounded, read-only navigation of a fully replayed HERMES graph snapshot.

This derives ephemeral candidate paths from the *same* page batch independently
checked against the HERMES chart and complete NaNROUTER history. A path is a
route through unproven claims, not a logical implication, MTH proof or a model
update. No second graph store or chart writes are performed.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hmac

from .mirror_topology import CANONICAL_CELLS
from .router_graph_history import RouterGraphHistoryVerifier
from .router_graph_page import RouterGraphPageError, RouterGraphPageVerifier
from .universe_root import read_universe_root


class VerifiedGraphWalkError(RouterGraphPageError):
    pass


def _node(value):
    if type(value) is not int or not 0 <= value < CANONICAL_CELLS:
        raise VerifiedGraphWalkError("invalid canonical head/MP graph node")
    return value


def verified_candidate_walk(
        verifier: RouterGraphHistoryVerifier, state_pages: list, *,
        start: int, target: int | None = None,
        layer: int | None = None, relation: str | None = None,
        max_hops: int = 4, max_nodes: int = 256,
        max_edges: int = 1024) -> dict:
    """Return deterministic shortest candidate route and competing claims.

    The complete state is rehashed rather than trusting objects that a caller
    might have mutated *after* the original state/history verification.
    Direction is src -> dst; no 'isa' transitivity or other semantic proof is
    implied. Hard bounds fail closed rather than returning partial success.
    """
    if (type(verifier) is not RouterGraphHistoryVerifier
            or not verifier.status()["history_verified"]
            or verifier.status()["failed"]):
        raise VerifiedGraphWalkError("complete independent history replay required")
    if (type(state_pages) is not list or not 1 <= len(state_pages) <= 128
            or type(max_hops) is not int or not 0 <= max_hops <= 8
            or type(max_nodes) is not int or not 1 <= max_nodes <= 1024
            or type(max_edges) is not int or not 1 <= max_edges <= 12800
            or layer is not None and (type(layer) is not int or not 0 <= layer <= 9)
            or relation is not None and
                (type(relation) is not str or relation not in (
                    "isa", "part_of", "defines", "supports", "refutes",
                    "equivalent", "transforms", "causes", "depends_on",
                    "similar", "contradicts", "observes", "routes_to"))):
        raise VerifiedGraphWalkError("invalid bounded graph query")
    _node(start)
    if target is not None:
        _node(target)

    state = RouterGraphPageVerifier(verifier.state.chart)
    entries = []
    for page in state_pages:
        state.add_page(page)
        entries.extend(page["entries"])
    if (not state.status()["state_verified"]
            or not hmac.compare_digest(
                state._commitment, verifier.state._commitment)):
        raise VerifiedGraphWalkError("query state differs from verified history")

    # The chart must still name the same physical root and logical epoch.
    chart = verifier.state.chart
    root = read_universe_root(chart)
    if (not root.get("mounted") or chart.root_hash != state.parent
            or chart.snapshot_generation != state.generation):
        raise VerifiedGraphWalkError("resident chart changed during query")

    candidate = [
        entry for entry in entries
        if entry["status"] == "candidate"
        and (layer is None or entry["layer"] == layer)
    ]
    active = [
        {key: entry[key] for key in (
            "edge_id", "src", "dst", "layer", "relation",
            "evidence_id", "evidence_version", "evidence_digest", "generation")}
        for entry in candidate
        if relation is None or entry["relation"] == relation
    ]
    if len(active) > max_edges:
        raise VerifiedGraphWalkError("query exceeds complete edge bound")

    active.sort(key=lambda e: (
        e["src"], e["dst"], e["layer"], e["relation"], e["edge_id"]))
    adjacency = defaultdict(list)
    for entry in active:
        adjacency[entry["src"]].append(entry)

    # A relation filter controls navigation, not truth visibility. In
    # particular filtering to 'supports' must never erase a simultaneous
    # 'refutes' candidate from the contested-claim report.
    competing = defaultdict(set)
    for entry in candidate:
        competing[(entry["src"], entry["dst"], entry["layer"])].add(
            entry["relation"])
    contested = [
        {"src": src, "dst": dst, "layer": level,
         "relations": sorted(kinds)}
        for (src, dst, level), kinds in sorted(competing.items())
        if "supports" in kinds and "refutes" in kinds
    ]
    if len(contested) > max_edges:
        raise VerifiedGraphWalkError("query exceeds contested-claim bound")

    # A visit is a canonical head/MP cell, never an independent universe.
    discovered = {start: (0, None)}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        distance = discovered[node][0]
        if distance >= max_hops:
            continue
        for entry in adjacency.get(node, ()):
            nxt = entry["dst"]
            if nxt in discovered:
                continue
            if len(discovered) == max_nodes:
                raise VerifiedGraphWalkError("query exceeds complete node bound")
            discovered[nxt] = (distance + 1, entry)
            queue.append(nxt)

    def path_to(node):
        output = []
        while node != start:
            edge = discovered[node][1]
            output.append(edge)
            node = edge["src"]
        return list(reversed(output))

    paths = ([{"target": target, "edges": path_to(target)}]
             if target in discovered else []) if target is not None else [
                 {"target": node, "edges": path_to(node)}
                 for node in sorted(discovered) if node != start
             ]
    return {
        "schema": "nanlm.router-verified-candidate-walk/1",
        "graph_commitment_sha256": state._commitment,
        "history_sha256": state._header["history_sha256"],
        "start": start, "target": target, "layer": layer,
        "relation": relation, "max_hops": max_hops,
        "reached_nodes": len(discovered), "matched_candidate_edges": len(active),
        "paths": paths, "contested_pairs": contested,
        "state_verified": True, "history_verified": True,
        "sender_authenticated": False, "evidence_integrity_verified": False,
        "native_mth_executed": False, "learning_admitted": False,
        "mathematical_truth_promoted": False,
    }


__all__ = ("VerifiedGraphWalkError", "verified_candidate_walk")
