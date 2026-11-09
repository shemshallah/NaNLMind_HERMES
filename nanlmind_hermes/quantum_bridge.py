"""Read-only bridge across the quantum-inspired / gauge subsystems.

This module composes existing classical implementations into one bounded,
provenance-bound summary.  It does not write canonical state, does not call
NANNET, the router, or CLOCKED, and does not claim a quantum speedup or proof.

Everything emitted by :class:`QuantumMemoryBridge` is local and derived from
the supplied memory rows / concept graph.  Unsupported inputs fail closed with
an explicit ``available``/``reason`` pair instead of inventing measurements.
"""
from __future__ import annotations

import cmath
import copy
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import (
    cycle_space,
    gauge_memory,
    pair_quantum,
    phase_memory,
    robust_recovery,
    semantic_state,
    spectral,
)
from .concept_matrix import ConceptMatrix
from .cycle_codec import (
    decode_cycle,
    encode,
    estimate_gauge,
    gauge_values,
    pairs as codec_pairs,
    random_cycle_basis,
)
from .cycle_state import CycleState
from .relation_operators import matmul
from .typed_cycle import TypedCycleOperators

SCHEMA = "nanlmind-hermes.quantum-memory-bridge/1"
INPUT_SCHEMA = "nanlmind-hermes.quantum-memory-bridge-input/1"
LOCALITY = "local/derived"
MAX_LABEL_CHARS = 96
SECTION_ORDER = (
    "gauge_cycle",
    "phase_coherence",
    "pair_spectral",
    "robust_recovery",
    "concept_graph",
    "semantic_state",
)
CLAIMS = {
    "quantum_speedup": False,
    "quantum_proof": False,
    "physical_quantum_system": False,
}
MODULES_REUSED = (
    "concept_matrix",
    "cycle_codec",
    "cycle_space",
    "cycle_state",
    "gauge_memory",
    "pair_quantum",
    "phase_memory",
    "relation_operators",
    "robust_recovery",
    "semantic_state",
    "spectral",
    "typed_cycle",
)
UNWIRED = (
    "chat handler: tools.chat / nanlmind_hermes.__init__.handle_chat do not call this bridge",
    "learning status: nanlm_learning_status does not emit these sections",
    "tool exposure: plugin.yaml and the plugin tool schema are unchanged",
    "durable provenance: the caller must persist input_sha256 and the output summary",
    "joint semantic decoder: scripts/joint_semantic_decoder.py is a training script, "
    "not an importable decoder API; no weights are loaded or fabricated",
    "NANNET / router / CLOCKED: never called by this module",
)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def _stable_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<nonfinite>"
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"<type>": _stable_type_name(value)}


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip().casefold()
    elif isinstance(value, (int, float, bool)):
        text = str(value).strip().casefold()
    else:
        return ""
    if len(text) <= MAX_LABEL_CHARS:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text[:MAX_LABEL_CHARS - 17] + "#" + digest


def _number(value: Any, default: float = 1.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _weight(value: Any, default: float = 1.0) -> float:
    return max(0.0, min(1000.0, _number(value, default)))


def _rounded(value: Any, places: int = 12) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = round(number, places)
    return 0.0 if rounded == 0.0 else rounded


def _norm(values: Iterable[complex]) -> float:
    return math.sqrt(sum(abs(value) ** 2 for value in values))


def _predicate_phase(predicate: str) -> float:
    """Stable label-encoding phase.

    This is *not* a measured physical phase.  It maps a predicate label to an
    angle so the existing phase/coherence APIs can be exercised deterministically.
    """
    digest = hashlib.sha256(predicate.encode("utf-8")).digest()
    return 2.0 * math.pi * int.from_bytes(digest[:8], "big") / 2.0 ** 64


def _bounded_items(value: Any, limit: int):
    """Return (items, received, received_is_lower_bound, truncated, reason)."""
    if value is None:
        return [], 0, False, False, None
    if isinstance(value, (str, bytes, Mapping)):
        return [], 0, False, False, "expected-an-iterable-of-items"
    if isinstance(value, (set, frozenset)):
        return [], 0, False, False, "unordered-input-not-supported"
    if isinstance(value, Sequence):
        total = len(value)
        return list(value[:limit]), total, False, total > limit, None
    try:
        iterator = iter(value)
    except TypeError:
        return [], 0, False, False, "not-iterable"
    items = []
    for _ in range(limit):
        try:
            items.append(next(iterator))
        except StopIteration:
            return items, len(items), False, False, None
    try:
        next(iterator)
    except StopIteration:
        return items, len(items), False, False, None
    return items, len(items) + 1, True, True, None


def _first_present(mapping: Mapping, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


@dataclass(frozen=True)
class _Relation:
    subject: str
    predicate: str
    object: str
    kind: str
    weight: float
    source: str
    group: str
    row_id: str


def _relation_key(relation: _Relation) -> tuple:
    return (
        relation.source,
        relation.row_id,
        relation.subject,
        relation.predicate,
        relation.object,
        relation.group,
        relation.kind,
        relation.weight,
    )


def _group_label(item: Mapping, fallback: str) -> str:
    provenance = item.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("resolution", "source", "origin"):
            value = _label(provenance.get(key))
            if value:
                return f"provenance:{key}:{value}"
    kind = _label(item.get("kind")) or fallback
    return f"kind:{kind}"


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class QuantumMemoryBridge:
    """Bounded, read-only summary over local gauge/cycle/phase subsystems."""

    def __init__(
        self,
        memory_rows: Iterable[Mapping] | None = None,
        concept_graph: Any | None = None,
        *,
        max_rows: int = 64,
        max_graph_edges: int = 64,
        max_vertices: int = 12,
        max_triangles: int = 8,
        max_channels: int = 8,
        max_patterns: int = 8,
        seed: int = 0,
        semantic_nodes: Mapping | None = None,
        semantic_start: Any | None = None,
        semantic_depth: int = 3,
        semantic_dimensions: int = 32,
    ):
        self.max_rows = self._int_bound("max_rows", max_rows, 1, 256)
        self.max_graph_edges = self._int_bound(
            "max_graph_edges", max_graph_edges, 1, 256)
        self.max_vertices = self._int_bound("max_vertices", max_vertices, 2, 16)
        self.max_triangles = self._int_bound(
            "max_triangles", max_triangles, 1, 32)
        self.max_channels = self._int_bound(
            "max_channels", max_channels, 1, 16)
        self.max_patterns = self._int_bound(
            "max_patterns", max_patterns, 2, 16)
        self.seed = self._int_bound("seed", seed, 0, 2 ** 31 - 1)
        self.semantic_depth = self._int_bound(
            "semantic_depth", semantic_depth, 1, 8)
        self.semantic_dimensions = self._int_bound(
            "semantic_dimensions", semantic_dimensions, 8, 256)
        self.semantic_nodes = semantic_nodes
        self.semantic_start = semantic_start

        self._bounds = {
            "max_rows": self.max_rows,
            "max_graph_edges": self.max_graph_edges,
            "max_vertices": self.max_vertices,
            "max_triangles": self.max_triangles,
            "max_channels": self.max_channels,
            "max_patterns": self.max_patterns,
            "seed": self.seed,
            "semantic_depth": self.semantic_depth,
            "semantic_dimensions": self.semantic_dimensions,
        }
        self._input_reasons: list[str] = []
        self._skip_counts: Counter[str] = Counter()
        self._row_entries: list[dict] = []
        self._row_relations: list[_Relation] = []

        rows, received, lower_bound, truncated, reason = _bounded_items(
            memory_rows, self.max_rows)
        self._rows_received = received
        self._rows_received_lower_bound = lower_bound
        self._rows_truncated = truncated
        self._raw_rows_digest = _digest(
            sorted(_canonical(row) for row in rows))
        if reason:
            self._input_reasons.append(f"memory_rows:{reason}")
        for row in rows:
            relation, skip = self._normalise_row(row)
            if relation is None:
                self._skip_counts[skip] += 1
                continue
            self._row_relations.append(relation)
            self._row_entries.append({
                "row_id": relation.row_id,
                "kind": relation.kind,
                "group": relation.group,
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "source": relation.source,
            })
        self._row_entries.sort(
            key=lambda entry: (
                entry["row_id"],
                entry["subject"],
                entry["predicate"],
                entry["object"],
            ))

        self._graph_relations, self._graph_meta = self._normalise_concept_graph(
            concept_graph)
        self._semantic_meta = self._normalise_semantic()
        self._relations = sorted(
            self._row_relations + self._graph_relations, key=_relation_key)
        self._input_sha256 = self._compute_input_digest()
        self._payload = self._build_payload()

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _int_bound(name: str, value: Any, low: int, high: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be in [{low},{high}]")
        return value

    # -- memory rows --------------------------------------------------------

    def _normalise_row(self, row: Any):
        if not isinstance(row, Mapping):
            return None, "row-is-not-a-mapping"
        item = row.get("item")
        if not isinstance(item, Mapping):
            item = row
        subject = _label(item.get("subject"))
        predicate = _label(item.get("predicate"))
        obj = _label(item.get("object"))
        if not subject or not predicate or not obj:
            return None, "missing-subject-predicate-object"
        if subject == obj:
            return None, "self-loop"
        kind = _label(item.get("kind")) or _label(row.get("kind")) or "relation"
        raw_confidence = item.get("confidence", row.get("confidence", 1.0))
        weight = _weight(raw_confidence, 1.0)
        if weight <= 0.0:
            return None, "non-positive-weight"
        provenance = item.get("provenance", row.get("provenance"))
        group = _group_label(
            {"kind": kind, "provenance": provenance}, "relation")
        return _Relation(
            subject=subject,
            predicate=predicate,
            object=obj,
            kind=kind,
            weight=weight,
            source="memory_row",
            group=group,
            row_id=_digest(item),
        ), None

    # -- concept graph ------------------------------------------------------

    def _normalise_concept_graph(self, graph: Any):
        meta = {
            "available": False,
            "reason": "no concept graph supplied",
            "source": "none",
            "nodes": 0,
            "edges_received": 0,
            "edges_used": 0,
            "truncated": False,
            "statements": None,
            "top_concepts": [],
            "top_pairs": [],
            "roles": [],
            "digest": _digest(None),
        }
        if graph is None:
            return [], meta
        try:
            if isinstance(graph, ConceptMatrix):
                return self._concept_matrix_relations(graph, meta)
            if isinstance(graph, Mapping):
                edges = _first_present(
                    graph, ("edges", "relations", "pairs"))
                nodes = graph.get("nodes")
                if edges is None:
                    meta["source"] = "mapping"
                    meta["reason"] = (
                        "concept_graph mapping must contain an "
                        "edges/relations/pairs iterable")
                    meta["digest"] = _digest({
                        "source": "mapping",
                        "keys": sorted(str(key) for key in graph.keys()),
                    })
                    return [], meta
                return self._edge_list_relations(
                    edges, "mapping", nodes, meta)
            if isinstance(graph, Iterable) and not isinstance(
                    graph, (str, bytes)):
                return self._edge_list_relations(
                    graph, "iterable", None, meta)
            meta["reason"] = (
                "concept_graph must be a ConceptMatrix, a mapping with "
                "edges/relations/pairs, or an iterable of edges")
            meta["digest"] = _digest({
                "source": "unsupported",
                "type": _stable_type_name(graph),
            })
            return [], meta
        except Exception as exc:  # fail closed on malformed graph objects
            meta["available"] = False
            meta["reason"] = f"concept_graph rejected: {type(exc).__name__}"
            meta["digest"] = _digest({
                "source": "rejected",
                "type": _stable_type_name(graph),
            })
            return [], meta

    def _concept_matrix_relations(self, matrix: ConceptMatrix, meta: dict):
        meta.update({
            "available": True,
            "reason": "",
            "source": "ConceptMatrix",
            "nodes": len(matrix.concepts),
            "statements": int(matrix.statements),
        })
        if not matrix.concepts and not matrix.pairs:
            meta["available"] = False
            meta["reason"] = "concept graph contained no nodes or edges"
        try:
            saturation = matrix.saturation()
            meta["top_concepts"] = [
                {"concept": _label(row["concept"]),
                 "degree": int(row["degree"])}
                for row in saturation.get("top_concepts", [])[:8]
            ]
            meta["top_pairs"] = [
                {
                    "pair": [_label(part) for part in row["pair"]],
                    "weight": _rounded(row["weight"]),
                }
                for row in saturation.get("top_pairs", [])[:8]
            ]
            meta["roles"] = [
                {"role": _label(role), "count": int(count)}
                for role, count in sorted(
                    saturation.get("roles", {}).items())[:16]
            ]
        except Exception:
            meta["top_concepts"] = []
            meta["top_pairs"] = []
            meta["roles"] = []

        pairs = matrix.pairs
        edges_received = 0
        for cell in pairs.values():
            roles = cell.get("roles") if isinstance(cell, Mapping) else None
            edges_received += max(1, len(roles)) if isinstance(roles, Mapping) else 1
        meta["edges_received"] = edges_received

        relations: list[_Relation] = []
        for key in sorted(pairs):
            cell = pairs[key]
            if not isinstance(cell, Mapping):
                continue
            parts = str(key).split("\u0000")
            if len(parts) != 2:
                continue
            first, second = _label(parts[0]), _label(parts[1])
            if not first or not second or first == second:
                continue
            raw_weight = cell.get("weight", cell.get("count", 1.0))
            weight = _weight(raw_weight, 1.0)
            if weight <= 0.0:
                weight = max(
                    1.0, _weight(cell.get("count", 1.0), 1.0))
            roles = cell.get("roles") if isinstance(cell.get("roles"), Mapping) else {}
            role_items = [
                (str(role), int(count))
                for role, count in sorted(roles.items())
                if int(count) > 0
            ]
            if not role_items:
                role_items = [("related_to", 1)]
            share = weight / max(1, len(role_items))
            for role, _count in role_items:
                if len(relations) >= self.max_graph_edges:
                    break
                relations.append(_Relation(
                    subject=first,
                    predicate=_label(role) or "related_to",
                    object=second,
                    kind="concept",
                    weight=max(share, 0.0),
                    source="concept_matrix",
                    group="concept_matrix",
                    row_id=_digest({
                        "concept_matrix": key,
                        "role": role,
                        "cell": cell,
                    }),
                ))
        meta["edges_used"] = len(relations)
        meta["truncated"] = edges_received > len(relations)
        meta["digest"] = _digest({
            "pairs": matrix.pairs,
            "concepts": matrix.concepts,
            "statements": matrix.statements,
        })
        return relations, meta

    def _edge_list_relations(self, edges: Any, source: str,
                             nodes: Any, meta: dict):
        if isinstance(nodes, Mapping):
            explicit_nodes = {
                _label(key) for key in nodes.keys() if _label(key)
            }
        elif nodes is None or isinstance(nodes, (str, bytes)):
            explicit_nodes = set()
        elif isinstance(nodes, Iterable):
            try:
                explicit_nodes = {
                    _label(node) for node in nodes if _label(node)
                }
            except Exception:
                explicit_nodes = set()
        else:
            explicit_nodes = set()

        items, received, lower_bound, truncated, reason = _bounded_items(
            edges, self.max_graph_edges)
        if reason:
            meta.update({
                "available": False,
                "reason": f"concept_graph edges:{reason}",
                "source": source,
                "digest": _digest({"source": source, "reason": reason}),
            })
            return [], meta

        relations: list[_Relation] = []
        for item in items:
            relation, skip = self._normalise_graph_edge(item, source)
            if relation is None:
                self._skip_counts[f"graph:{skip}"] += 1
                continue
            relations.append(relation)

        node_count, top_concepts, top_pairs = self._summarise_graph(
            relations, explicit_nodes)
        role_counts = Counter(relation.predicate for relation in relations)
        meta.update({
            "available": True,
            "reason": "" if node_count or relations else (
                "concept graph contained no nodes or edges"),
            "source": source,
            "nodes": node_count,
            "edges_received": received,
            "edges_used": len(relations),
            "truncated": truncated,
            "top_concepts": top_concepts,
            "top_pairs": top_pairs,
            "roles": [
                {"role": role, "count": count}
                for role, count in sorted(role_counts.items())[:16]
            ],
            "digest": _digest({
                "source": source,
                "nodes": sorted(explicit_nodes),
                "edges": [
                    [relation.subject, relation.predicate, relation.object,
                     _rounded(relation.weight)]
                    for relation in sorted(relations, key=_relation_key)
                ],
                "received_lower_bound": lower_bound,
            }),
        })
        if not node_count and not relations:
            meta["available"] = False
        return relations, meta

    def _normalise_graph_edge(self, item: Any, source: str):
        if isinstance(item, Mapping):
            subject = _label(_first_present(
                item, ("subject", "s", "source", "from")))
            obj = _label(_first_present(
                item, ("object", "o", "target", "to")))
            predicate = _label(_first_present(
                item, ("predicate", "relation", "role"))) or "related_to"
            kind = _label(item.get("kind")) or "concept"
            raw_weight = _first_present(item, ("confidence", "weight"))
            weight = _weight(raw_weight, 1.0)
            provenance = item.get("provenance")
            group = _group_label(
                {"kind": kind, "provenance": provenance}, "concept_graph")
            row_id = _digest(item)
        elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes)):
            parts = list(item)
            if len(parts) == 2:
                subject = _label(parts[0])
                predicate = "related_to"
                obj = _label(parts[1])
            elif len(parts) >= 3:
                subject = _label(parts[0])
                predicate = _label(parts[1]) or "related_to"
                obj = _label(parts[2])
            else:
                return None, "edge-sequence-too-short"
            kind = "concept"
            weight = 1.0
            group = "concept_graph"
            row_id = _digest(parts)
        else:
            return None, "edge-is-not-a-mapping-or-sequence"

        if not subject or not obj:
            return None, "missing-edge-endpoint"
        if subject == obj:
            return None, "self-loop"
        if weight <= 0.0:
            return None, "non-positive-weight"
        return _Relation(
            subject=subject,
            predicate=predicate,
            object=obj,
            kind=kind,
            weight=weight,
            source=source,
            group=group,
            row_id=row_id,
        ), None

    @staticmethod
    def _summarise_graph(relations: list[_Relation], explicit_nodes: set[str]):
        concept_counts: Counter[str] = Counter()
        pair_weights: Counter[tuple[str, str]] = Counter()
        for relation in relations:
            concept_counts[relation.subject] += 1
            concept_counts[relation.object] += 1
            pair = tuple(sorted((relation.subject, relation.object)))
            pair_weights[pair] += relation.weight
        top_concepts = [
            {"concept": concept, "count": count}
            for concept, count in sorted(
                concept_counts.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        ]
        top_pairs = [
            {
                "pair": [pair[0], pair[1]],
                "weight": _rounded(weight),
            }
            for pair, weight in sorted(
                pair_weights.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        ]
        nodes = set(explicit_nodes)
        nodes.update(concept_counts)
        return len(nodes), top_concepts, top_pairs

    # -- semantic state -----------------------------------------------------

    def _normalise_semantic(self) -> dict:
        meta = {
            "available": False,
            "reason": (
                "semantic_state requires an explicit WordNet-shaped nodes "
                "mapping and start offset; flat memory rows do not encode "
                "ancestry"),
            "nodes": 0,
            "start": _json_safe(self.semantic_start),
            "depth": self.semantic_depth,
            "dimensions": self.semantic_dimensions,
            "ancestor_edges": 0,
            "ordinary_norm": None,
            "cycle_norm": None,
            "ordinary_sha256": None,
            "cycle_sha256": None,
            "digest": _digest(None),
        }
        if self.semantic_nodes is None:
            return meta
        if not isinstance(self.semantic_nodes, Mapping):
            meta["reason"] = "semantic_nodes must be a mapping"
            meta["digest"] = _digest({
                "source": "unsupported",
                "type": _stable_type_name(self.semantic_nodes),
            })
            return meta
        try:
            start_present = (
                self.semantic_start in self.semantic_nodes
                if self.semantic_start is not None else False)
        except TypeError:
            start_present = False
        if not start_present:
            meta["reason"] = (
                "semantic_start must identify a node in semantic_nodes")
            meta["digest"] = _digest({
                "nodes": sorted(str(key) for key in self.semantic_nodes.keys()),
                "start": self.semantic_start,
            })
            return meta
        try:
            nodes = self.semantic_nodes
            start = self.semantic_start
            edges = semantic_state.ancestor_edges(
                nodes, start, self.semantic_depth)
            ordinary = semantic_state.ordinary_state(
                nodes, start, self.semantic_depth,
                self.semantic_dimensions)
            cycle = semantic_state.cycle_state(
                nodes, start, self.semantic_depth,
                dimensions=self.semantic_dimensions,
                vertices=self.max_vertices)
        except Exception as exc:
            meta["reason"] = (
                f"semantic_state computation failed: {type(exc).__name__}")
            meta["digest"] = _digest({
                "nodes": sorted(str(key) for key in self.semantic_nodes.keys()),
                "start": self.semantic_start,
                "depth": self.semantic_depth,
            })
            return meta
        meta.update({
            "available": True,
            "reason": "",
            "nodes": len(self.semantic_nodes),
            "ancestor_edges": len(edges),
            "ordinary_norm": _rounded(
                math.sqrt(sum(value * value for value in ordinary))),
            "cycle_norm": _rounded(
                math.sqrt(sum(value * value for value in cycle))),
            "ordinary_sha256": _digest(
                [_rounded(value) for value in ordinary]),
            "cycle_sha256": _digest(
                [_rounded(value) for value in cycle]),
            "digest": _digest({
                "nodes": self.semantic_nodes,
                "start": start,
                "depth": self.semantic_depth,
                "dimensions": self.semantic_dimensions,
            }),
        })
        return meta

    # -- field construction -------------------------------------------------

    def _build_fields(self):
        counts: Counter[str] = Counter()
        for relation in self._relations:
            counts[relation.subject] += 1
            counts[relation.object] += 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        vertices = tuple(sorted(
            label for label, _count in ordered[:self.max_vertices]))
        index = {label: position for position, label in enumerate(vertices)}

        scalar: defaultdict[tuple[int, int], float] = defaultdict(float)
        complex_edges: defaultdict[tuple[int, int], complex] = defaultdict(complex)
        pair_predicates: defaultdict[tuple[int, int], list[_Relation]] = defaultdict(list)
        dropped = 0
        for relation in self._relations:
            first = index.get(relation.subject)
            second = index.get(relation.object)
            if first is None or second is None:
                dropped += 1
                continue
            if first == second:
                dropped += 1
                continue
            key = (first, second) if first < second else (second, first)
            scalar[key] += relation.weight
            complex_edges[key] += (
                relation.weight
                * cmath.exp(1j * _predicate_phase(relation.predicate)))
            pair_predicates[key].append(relation)

        scalar = {
            key: value for key, value in sorted(scalar.items()) if value > 0.0
        }
        complex_edges = {
            key: complex_edges[key]
            for key in sorted(complex_edges)
            if key in scalar
        }
        pair_predicates = {
            key: value for key, value in pair_predicates.items()
            if key in scalar
        }
        return (
            len(vertices), list(vertices), index, dict(scalar),
            dict(complex_edges), dict(pair_predicates), dropped,
        )

    def _build_groups(self):
        grouped: defaultdict[str, list[_Relation]] = defaultdict(list)
        for relation in self._relations:
            grouped[relation.group].append(relation)
        return [
            (label, sorted(relations, key=_relation_key))
            for label, relations in sorted(grouped.items())
        ]

    @staticmethod
    def _aggregate(relations: Iterable[_Relation], index: dict[str, int],
                   complex_mode: bool = False):
        values: defaultdict[tuple[int, int], Any] = defaultdict(
            complex if complex_mode else float)
        for relation in relations:
            first = index.get(relation.subject)
            second = index.get(relation.object)
            if first is None or second is None or first == second:
                continue
            key = (first, second) if first < second else (second, first)
            if complex_mode:
                values[key] += (
                    relation.weight
                    * cmath.exp(1j * _predicate_phase(relation.predicate)))
            else:
                values[key] += relation.weight
        if complex_mode:
            return {
                key: value for key, value in sorted(values.items())
                if abs(value) > 0.0
            }
        return {
            key: value for key, value in sorted(values.items())
            if value > 0.0
        }

    def _selected_groups(self, groups, index: dict[str, int]):
        usable = []
        for label, relations in groups:
            usable_relations = [
                relation for relation in relations
                if relation.subject in index
                and relation.object in index
                and relation.subject != relation.object
            ]
            if usable_relations:
                usable.append((label, usable_relations))
        usable.sort(key=lambda item: (-len(item[1]), item[0]))
        return usable[:self.max_patterns], len(usable) > self.max_patterns

    @staticmethod
    def _best_predicate(relations: list[_Relation]) -> str | None:
        totals: Counter[str] = Counter()
        for relation in relations:
            totals[relation.predicate] += relation.weight
        if not totals:
            return None
        return sorted(totals.items(), key=lambda item: (-item[1], item[0]))[0][0]

    # -- sections -----------------------------------------------------------

    def _gauge_section(self, n, vertices, index, scalar, complex_edges,
                       pair_predicates, groups):
        section = {
            "available": False,
            "reason": "",
            "graph": self._mark({
                "available": True,
                "reason": "",
                "vertex_count": n,
                "vertices": list(vertices),
                "observed_edges": len(scalar),
                "complete_graph_edges": cycle_space.edge_count(n) if n else 0,
                "cycle_rank": cycle_space.cycle_dimension(n) if n else 0,
                "fundamental_cycle_count": (
                    (n - 1) * (n - 2) // 2 if n >= 3 else 0),
                "triangle_count": cycle_space.triangle_count(n) if n else 0,
                "complete_graph_scope": (
                    "cycle_rank, fundamental_cycle_count and triangle_count "
                    "describe the complete graph over the selected vertices; "
                    "the computed complete-field invariant is in invariants"),
                "observed_fraction": _rounded(
                    len(scalar) / cycle_space.edge_count(n)
                    if n >= 2 and cycle_space.edge_count(n) else None),
            }),
            "invariants": {
                "available": False,
                "reason": "",
            },
            "gauge_fixed_retrieval": {
                "available": False,
                "reason": "",
            },
            "codec": {
                "available": False,
                "reason": "",
            },
            "cycle_state": {
                "available": False,
                "reason": "",
            },
        }
        if n < 3:
            section["reason"] = (
                "at least three concept vertices with usable relations are "
                "required for cycle-space invariants")
            return self._mark(section)
        if not scalar:
            section["reason"] = "no usable relation edges"
            return self._mark(section)
        section["available"] = True

        complete = len(scalar) == cycle_space.edge_count(n)
        missing = cycle_space.edge_count(n) - len(scalar)
        if complete:
            potential, gradient, cycle = cycle_space.decompose(
                n, complex_edges)
            rebuilt = cycle_space.reconstruct(n, potential, cycle)
            reconstruction_error = max(
                abs(rebuilt[edge] - complex_edges[edge])
                for edge in complex_edges)
            fundamental = cycle_space.fundamental_fluxes(n, complex_edges)
            cycle_divergence = cycle_space.divergence(n, cycle)
            section["invariants"] = self._mark({
                "available": True,
                "reason": "",
                "cycle_rank": cycle_space.cycle_dimension(n),
                "fundamental_cycle_count": len(fundamental),
                "triangle_count": cycle_space.triangle_count(n),
                "fundamental_flux_norm": _rounded(
                    _norm(fundamental.values())),
                "gradient_norm": _rounded(_norm(gradient.values())),
                "cycle_norm": _rounded(_norm(cycle.values())),
                "max_cycle_divergence": _rounded(
                    max((abs(value) for value in cycle_divergence), default=0.0)),
                "gradient_cycle_inner_product": _rounded(
                    cycle_space.cycle_inner_product(gradient, cycle)),
                "reconstruction_error": _rounded(reconstruction_error),
            })
        else:
            section["invariants"] = self._mark({
                "available": False,
                "reason": (
                    f"complete pair field required: {missing} of "
                    f"{cycle_space.edge_count(n)} pairs missing"),
            })

        # Gauge-fixed retrieval over provenance groups.
        selected, groups_truncated = self._selected_groups(groups, index)
        if len(selected) < 2:
            section["gauge_fixed_retrieval"] = self._mark({
                "available": False,
                "reason": (
                    "gauge-fixed retrieval needs at least two provenance "
                    "groups with usable relations"),
            })
        else:
            patterns = [
                gauge_memory.project_cycle(
                    n, self._aggregate(relations, index))
                for _label, relations in selected
            ]
            query = gauge_memory.project_cycle(n, scalar)
            observed = sorted(set.intersection(
                *(set(pattern) for pattern in patterns)))
            if len(observed) < 3:
                section["gauge_fixed_retrieval"] = self._mark({
                    "available": False,
                    "reason": (
                        "gauge-fixed retrieval needs at least three shared "
                        "observed pairs across provenance groups"),
                    "shared_observed_pairs": len(observed),
                    "pattern_count": len(patterns),
                })
            else:
                residuals = [
                    gauge_memory.gauge_residual(
                        n, query, pattern, observed)
                    for pattern in patterns
                ]
                naive_residuals = [
                    gauge_memory.naive_residual(query, pattern, observed)
                    for pattern in patterns
                ]
                best = gauge_memory.retrieve(
                    n, patterns, query, observed, gauge=True)
                best_naive = gauge_memory.retrieve(
                    n, patterns, query, observed, gauge=False)
                ordered_residuals = sorted(residuals)
                margin = (
                    ordered_residuals[1] - ordered_residuals[0]
                    if len(ordered_residuals) > 1 else 0.0)
                section["gauge_fixed_retrieval"] = self._mark({
                    "available": True,
                    "reason": "",
                    "pattern_count": len(patterns),
                    "groups_truncated": groups_truncated,
                    "shared_observed_pairs": len(observed),
                    "best_index": best,
                    "best_group": selected[best][0],
                    "gauge_residual": _rounded(residuals[best]),
                    "gauge_margin": _rounded(margin),
                    "naive_best_index": best_naive,
                    "naive_best_group": selected[best_naive][0],
                    "naive_residual": _rounded(naive_residuals[best_naive]),
                    "gauge_naive_agree": best == best_naive,
                    "group_residuals": [
                        {
                            "group": label,
                            "gauge_residual": _rounded(residual),
                            "naive_residual": _rounded(naive),
                        }
                        for (label, _relations), residual, naive
                        in zip(selected, residuals, naive_residuals)
                    ],
                })

        # Gauge-corrected cycle-codec reconstruction on a complete field.
        if complete:
            try:
                edge_pairs = codec_pairs(n)
                observed_values = [scalar[edge] for edge in edge_pairs]
                mask = [True] * len(observed_values)
                dimensions = min(
                    self.max_channels, cycle_space.cycle_dimension(n))
                if dimensions < 1:
                    raise ValueError("empty cycle basis")
                columns, row_view = random_cycle_basis(
                    n, dimensions, seed=self.seed)
                decoded = decode_cycle(
                    n, columns, observed_values, mask)
                potential = estimate_gauge(n, observed_values, mask)
                predicted_cycle = encode(row_view, decoded)
                predicted_gauge = gauge_values(n, potential)
                error = max(
                    abs(observed_values[position]
                        - (predicted_cycle[position]
                           + predicted_gauge[position]))
                    for position in range(len(observed_values)))
                section["codec"] = self._mark({
                    "available": True,
                    "reason": "",
                    "basis_dimensions": dimensions,
                    "basis_seed": self.seed,
                    "reconstruction_error": _rounded(error),
                    "coefficient_norm": _rounded(
                        math.sqrt(sum(value * value for value in decoded))),
                })
            except Exception as exc:
                section["codec"] = self._mark({
                    "available": False,
                    "reason": f"cycle codec failed: {type(exc).__name__}",
                })
        else:
            section["codec"] = self._mark({
                "available": False,
                "reason": (
                    "cycle codec requires a complete observed pair field"),
            })

        # Local ephemeral cycle state.  Nothing is persisted.
        if n >= 2 and complex_edges:
            state = CycleState(n)
            state.observe(complex_edges, rate=1.0)
            snapshot = state.snapshot()
            divergence = gauge_memory.divergence(n, state.values)
            potential = [
                complex(0.125 * (position + 1), -0.03125 * (position + 1))
                for position in range(n)
            ]
            shifted = state.with_gauge(potential)
            gauge_shift_residual = max(
                (abs(state.values[edge] - shifted.values.get(edge, 0j))
                 for edge in state.values),
                default=0.0)
            section["cycle_state"] = self._mark({
                "available": True,
                "reason": "",
                "nodes": n,
                "stored_edges": len(snapshot.get("values", {})),
                "energy": _rounded(state.energy()),
                "max_divergence": _rounded(
                    max((abs(value) for value in divergence), default=0.0)),
                "gauge_shift_residual": _rounded(gauge_shift_residual),
            })
        else:
            section["cycle_state"] = self._mark({
                "available": False,
                "reason": "no complex edge field for an ephemeral cycle state",
            })
        return self._mark(section)

    def _phase_section(self, n, index, complex_edges, pair_predicates,
                       groups):
        section = {
            "available": False,
            "reason": "",
            "triangles_used": 0,
            "phase_encoding": (
                "sha256(predicate)[:8] -> angle in [0,2pi); label encoding, "
                "not a measured physical phase"),
            "mean_triple_coherence": None,
            "min_triple_coherence": None,
            "max_triple_coherence": None,
            "fraction_triple_coherence_above_half": None,
            "mean_abs_triangle_flux": None,
            "phase_locking_order": None,
            "phase_memory": {
                "available": False,
                "reason": "",
            },
            "typed_loops": {
                "available": False,
                "reason": "",
            },
        }
        if n < 3 or not complex_edges:
            section["reason"] = (
                "at least three concept vertices and observed complex edges "
                "are required for phase/coherence features")
            return self._mark(section)

        triangles = list(itertools.combinations(range(n), 3))[
            :self.max_triangles]
        used = []
        for first, second, third in triangles:
            z1 = complex_edges.get((first, second))
            z2 = complex_edges.get((second, third))
            z3 = complex_edges.get((first, third))
            if z1 is None or z2 is None or z3 is None:
                continue
            if abs(z1 * z2 * z3) <= 1e-15:
                continue
            flux = cycle_space.triangle_flux(
                complex_edges, first, second, third)
            coherence = min(
                1.0, max(0.0,
                         cycle_space.triple_coherence(z1, z2, z3)))
            used.append(((first, second, third), flux, coherence))
        if not used:
            section["reason"] = (
                "no complete triangle of observed complex edges")
            return self._mark(section)

        coherences = [coherence for _triangle, _flux, coherence in used]
        fluxes = [flux for _triangle, flux, _coherence in used]
        unit_fluxes = [
            flux / abs(flux) for flux in fluxes if abs(flux) > 1e-15]
        locking_score = (
            phase_memory.complex_gauge_score(
                unit_fluxes, [1 + 0j] * len(unit_fluxes))
            if unit_fluxes else 0.0)
        section.update({
            "available": True,
            "reason": "",
            "triangles_used": len(used),
            "mean_triple_coherence": _rounded(
                sum(coherences) / len(coherences)),
            "min_triple_coherence": _rounded(min(coherences)),
            "max_triple_coherence": _rounded(max(coherences)),
            "fraction_triple_coherence_above_half": _rounded(
                sum(1 for value in coherences if value > 0.5)
                / len(coherences)),
            "mean_abs_triangle_flux": _rounded(
                sum(abs(flux) for flux in fluxes) / len(fluxes)),
            "phase_locking_order": _rounded(
                min(1.0, math.sqrt(max(0.0, locking_score))
                    / len(unit_fluxes))
                if unit_fluxes else None),
        })

        # Phase-memory readout over triangle-flux channels common to groups.
        selected, groups_truncated = self._selected_groups(groups, index)
        group_fluxes = []
        for label, relations in selected:
            field = self._aggregate(relations, index, complex_mode=True)
            per_triangle = {}
            for triangle in (item[0] for item in used):
                flux = cycle_space.triangle_flux(field, *triangle)
                if abs(flux) > 1e-15:
                    per_triangle[triangle] = flux
            group_fluxes.append((label, per_triangle))
        common = set()
        if group_fluxes:
            common = set(group_fluxes[0][1])
            for _label, per_triangle in group_fluxes[1:]:
                common &= set(per_triangle)
        channels = sorted(common)[:self.max_channels]
        if len(group_fluxes) < 2 or len(channels) < 2:
            section["phase_memory"] = self._mark({
                "available": False,
                "reason": (
                    "phase memory needs at least two provenance groups and "
                    "two triangle-flux channels common to all groups"),
                "pattern_count": len(group_fluxes),
                "shared_channels": len(channels),
            })
        else:
            dimensions = len(channels)
            patterns = []
            for _label, per_triangle in group_fluxes:
                patterns.append([
                    per_triangle[triangle] / abs(per_triangle[triangle])
                    for triangle in channels
                ])
            query = []
            for triangle in channels:
                flux = cycle_space.triangle_flux(
                    complex_edges, *triangle)
                if abs(flux) <= 1e-15:
                    query = []
                    break
                query.append(flux / abs(flux))
            if not query:
                section["phase_memory"] = self._mark({
                    "available": False,
                    "reason": (
                        "aggregate triangle flux vanished on a shared "
                        "channel, so no unit phase query can be formed"),
                    "pattern_count": len(group_fluxes),
                    "shared_channels": len(channels),
                })
            else:
                memory = phase_memory.PhaseMemory(
                    dimensions, seed=self.seed)
                for pattern in patterns:
                    memory.store(pattern)
                gauge_scores = memory.scores(query, gauge=True)
                naive_scores = memory.scores(query, gauge=False)
                best = memory.retrieve(query, gauge=True)
                naive_best = memory.retrieve(query, gauge=False)
                rotated = phase_memory.rotate(query, math.pi / 3.0)
                rotated_scores = memory.scores(rotated, gauge=True)
                global_phase_delta = max(
                    abs(left - right)
                    for left, right in zip(gauge_scores, rotated_scores))
                real_scores = [
                    phase_memory.real_gauge_score(query, pattern)
                    for pattern in patterns
                ]
                real_best = max(
                    range(len(real_scores)),
                    key=lambda position: real_scores[position])
                section["phase_memory"] = self._mark({
                    "available": True,
                    "reason": "",
                    "pattern_count": len(patterns),
                    "groups_truncated": groups_truncated,
                    "channels": [list(triangle) for triangle in channels],
                    "best_index": best,
                    "best_group": group_fluxes[best][0],
                    "gauge_score": _rounded(gauge_scores[best]),
                    "naive_best_index": naive_best,
                    "naive_best_group": group_fluxes[naive_best][0],
                    "naive_score": _rounded(naive_scores[naive_best]),
                    "global_phase_score_delta": _rounded(global_phase_delta),
                    "real_channel_agreement": real_best == best,
                    "group_scores": [
                        {
                            "group": group_fluxes[position][0],
                            "gauge_score": _rounded(gauge_scores[position]),
                            "naive_score": _rounded(naive_scores[position]),
                        }
                        for position in range(len(patterns))
                    ],
                })

        # Typed relation-operator loop diagnostic, bounded to the same triangles.
        operators = TypedCycleOperators()
        trace_norms = []
        loops_used = 0
        for triangle in (item[0] for item in used):
            first, second, third = triangle
            p_ab = self._best_predicate(pair_predicates.get(
                (first, second), []))
            p_bc = self._best_predicate(pair_predicates.get(
                (second, third), []))
            p_ac = self._best_predicate(pair_predicates.get(
                (first, third), []))
            if not p_ab or not p_bc or not p_ac:
                continue
            path = operators.path([p_ab, p_bc])
            closing = operators.reverse_path([p_ac])
            loop = matmul(path, closing)
            trace = loop[0][0] + loop[1][1]
            trace_norms.append(
                min(1.0, max(0.0, abs(trace) / 2.0)))
            loops_used += 1
        if loops_used:
            section["typed_loops"] = self._mark({
                "available": True,
                "reason": "",
                "loops_used": loops_used,
                "mean_trace_norm": _rounded(
                    sum(trace_norms) / len(trace_norms)),
                "min_trace_norm": _rounded(min(trace_norms)),
                "max_trace_norm": _rounded(max(trace_norms)),
                "note": (
                    "default SU(2) typed relation operators; diagnostic only, "
                    "not learned loop dynamics"),
            })
        else:
            section["typed_loops"] = self._mark({
                "available": False,
                "reason": (
                    "no triangle with a representative predicate on every "
                    "edge"),
            })
        return self._mark(section)

    def _pair_spectral_section(self, n, scalar):
        section = {
            "available": False,
            "reason": "",
            "pair_space": {
                "available": False,
                "reason": "",
            },
            "spectral": {
                "available": False,
                "reason": "",
            },
        }
        if n < 3:
            section["reason"] = (
                "at least three concept vertices are required for pair-space "
                "and spectral features")
            return self._mark(section)

        pair_count = pair_quantum.pair_count(n)
        queries, probability = pair_quantum.best_grover_queries(pair_count)
        period = pair_quantum.revival_period(n)
        section["pair_space"] = self._mark({
            "available": True,
            "reason": "",
            "vertex_count": n,
            "pair_count": pair_count,
            "johnson_degree": pair_quantum.johnson_degree(n),
            "johnson_spectrum": [
                {"eigenvalue": eigenvalue, "multiplicity": multiplicity}
                for eigenvalue, multiplicity in pair_quantum.johnson_spectrum(n)
            ],
            "revival_period": _rounded(period),
            "return_probability_at_revival": _rounded(
                pair_quantum.return_probability(n, period)),
            "ideal_grover_queries": queries,
            "ideal_grover_probability": _rounded(probability),
            "note": (
                "exact Johnson-graph arithmetic for the derived bounded "
                "subgraph; ideal mathematical control only, no speedup claim"),
        })

        if len(scalar) < 2:
            section["spectral"] = self._mark({
                "available": False,
                "reason": (
                    "at least two observed pair weights are required for a "
                    "Fiedler split"),
            })
        else:
            matrix = [[0.0] * n for _ in range(n)]
            for (first, second), value in scalar.items():
                matrix[first][second] = value
                matrix[second][first] = value
            status = spectral.WaveMatrix(matrix).status()
            fiedler = spectral.fiedler_partition(matrix)
            values = fiedler.get("values", [])
            section["spectral"] = self._mark({
                "available": True,
                "reason": "",
                "matrix_size": n,
                "observed_pair_weights": len(scalar),
                "symmetric": bool(status.get("symmetric")),
                "centrosymmetric": bool(status.get("centrosymmetric")),
                "fiedler_values": [
                    _rounded(value) for value in values],
                "fiedler_gap": _rounded(
                    values[1] - values[0]) if len(values) > 1 else None,
                "positive_count": len(fiedler.get("positive", [])),
                "negative_count": len(fiedler.get("negative", [])),
                "inner_count": len(fiedler.get("inner", [])),
                "boundary_count": len(fiedler.get("boundary", [])),
                "note": (
                    "zero matrix entries mean no observed relation; the "
                    "Laplacian/Fiedler split is a bounded classical diagnostic"),
            })

        section["available"] = (
            section["pair_space"]["available"]
            or section["spectral"]["available"])
        if not section["available"]:
            section["reason"] = (
                "pair-space and spectral features are both unavailable")
        return self._mark(section)

    def _robust_section(self, n, scalar):
        section = {
            "available": False,
            "reason": "",
            "gauge_flat": None,
            "flatness_ratio": None,
            "observed_cycle_norm": None,
            "observed_gradient_norm": None,
            "raw_median_mae": None,
            "raw_mean_mae": None,
            "median_mae_to_projected_gradient": None,
            "mean_mae_to_projected_gradient": None,
            "median_mean_disagreement": None,
            "note": (
                "the projected gradient is a Hodge reference computed from "
                "this same input, not ground truth; no recovery accuracy is "
                "claimed"),
        }
        if n < 3:
            section["reason"] = (
                "at least three concept vertices are required for two-hop "
                "robust-recovery checks")
            return self._mark(section)
        total = cycle_space.edge_count(n)
        missing = total - len(scalar)
        if missing:
            section["reason"] = (
                f"robust recovery requires a complete observed pair field: "
                f"{missing} of {total} pairs missing")
            section["missing_pairs"] = missing
            return self._mark(section)

        values = dict(scalar)
        cycle = gauge_memory.project_cycle(n, values)
        projected_gradient = {
            edge: values[edge] - cycle[edge] for edge in values
        }
        median_field = robust_recovery.two_hop_median(n, values)
        mean_field = robust_recovery.two_hop_mean(n, values)
        cycle_norm = _norm(cycle.values())
        gradient_norm = _norm(projected_gradient.values())
        section.update({
            "available": True,
            "reason": "",
            "gauge_flat": bool(
                cycle_norm <= 1e-9 * max(1.0, gradient_norm)),
            "flatness_ratio": _rounded(
                cycle_norm / max(1e-12, gradient_norm + cycle_norm)),
            "observed_cycle_norm": _rounded(cycle_norm),
            "observed_gradient_norm": _rounded(gradient_norm),
            "raw_median_mae": _rounded(
                robust_recovery.mean_absolute_error(median_field, values)),
            "raw_mean_mae": _rounded(
                robust_recovery.mean_absolute_error(mean_field, values)),
            "median_mae_to_projected_gradient": _rounded(
                robust_recovery.mean_absolute_error(
                    median_field, projected_gradient)),
            "mean_mae_to_projected_gradient": _rounded(
                robust_recovery.mean_absolute_error(
                    mean_field, projected_gradient)),
            "median_mean_disagreement": _rounded(
                robust_recovery.mean_absolute_error(
                    median_field, mean_field)),
            "pairs_used": len(values),
        })
        return self._mark(section)

    def _concept_section(self):
        meta = self._graph_meta
        section = {
            "available": bool(meta.get("available")),
            "reason": meta.get("reason", ""),
            "source": meta.get("source", "none"),
            "nodes": meta.get("nodes", 0),
            "edges_received": meta.get("edges_received", 0),
            "edges_used": meta.get("edges_used", 0),
            "truncated": bool(meta.get("truncated")),
            "statements": meta.get("statements"),
            "top_concepts": list(meta.get("top_concepts", [])),
            "top_pairs": list(meta.get("top_pairs", [])),
            "roles": list(meta.get("roles", [])),
            "graph_sha256": meta.get("digest"),
        }
        return self._mark(section)

    def _semantic_section(self):
        meta = self._semantic_meta
        section = {
            "available": bool(meta.get("available")),
            "reason": meta.get("reason", ""),
            "nodes": meta.get("nodes", 0),
            "start": meta.get("start"),
            "depth": meta.get("depth"),
            "dimensions": meta.get("dimensions"),
            "ancestor_edges": meta.get("ancestor_edges", 0),
            "ordinary_norm": meta.get("ordinary_norm"),
            "cycle_norm": meta.get("cycle_norm"),
            "ordinary_sha256": meta.get("ordinary_sha256"),
            "cycle_sha256": meta.get("cycle_sha256"),
            "note": (
                "classical hashed ancestry features; the full vectors are not "
                "emitted and this is not a trained language state"),
        }
        return self._mark(section)

    # -- provenance / payload ----------------------------------------------

    def _compute_input_digest(self) -> str:
        records = [
            [
                relation.subject,
                relation.predicate,
                relation.object,
                relation.kind,
                _rounded(relation.weight),
                relation.source,
                relation.group,
                relation.row_id,
            ]
            for relation in self._relations
        ]
        records.sort(key=_canonical)
        return _digest({
            "schema": INPUT_SCHEMA,
            "bounds": self._bounds,
            "raw_rows_sha256": self._raw_rows_digest,
            "relations": records,
            "skip_counts": dict(sorted(self._skip_counts.items())),
            "graph_sha256": self._graph_meta.get("digest"),
            "graph_available": self._graph_meta.get("available"),
            "semantic_sha256": self._semantic_meta.get("digest"),
            "semantic_available": self._semantic_meta.get("available"),
            "input_reasons": sorted(self._input_reasons),
        })

    def _mark(self, block: dict) -> dict:
        block["locality"] = LOCALITY
        block["derived"] = True
        block["input_sha256"] = self._input_sha256
        return block

    def _build_payload(self) -> dict:
        (
            n, vertices, index, scalar, complex_edges, pair_predicates,
            dropped,
        ) = self._build_fields()
        groups = self._build_groups()

        gauge = self._gauge_section(
            n, vertices, index, scalar, complex_edges, pair_predicates, groups)
        phase = self._phase_section(
            n, index, complex_edges, pair_predicates, groups)
        pair_spectral = self._pair_spectral_section(n, scalar)
        robust = self._robust_section(n, scalar)
        concept = self._concept_section()
        semantic = self._semantic_section()

        provenance = {
            "available": True,
            "reason": "",
            "locality": LOCALITY,
            "derived": True,
            "input_sha256": self._input_sha256,
            "input_schema": INPUT_SCHEMA,
            "bounds": dict(sorted(self._bounds.items())),
            "counts": {
                "rows_received": self._rows_received,
                "rows_received_is_lower_bound": (
                    self._rows_received_lower_bound),
                "rows_used": len(self._row_relations),
                "rows_skipped": sum(self._skip_counts.values()),
                "rows_truncated": self._rows_truncated,
                "graph_edges_received": self._graph_meta.get(
                    "edges_received", 0),
                "graph_edges_used": self._graph_meta.get("edges_used", 0),
                "relations_used": len(self._relations),
                "vertices": n,
                "edges": len(scalar),
                "complete_graph_edges": (
                    cycle_space.edge_count(n) if n else 0),
                "groups": len(groups),
                "dropped_relations": dropped,
            },
            "rows": copy.deepcopy(self._row_entries[:self.max_rows]),
            "skipped_rows": [
                {"reason": reason, "count": count}
                for reason, count in sorted(self._skip_counts.items())
            ],
            "input_reasons": sorted(self._input_reasons),
            "graph": {
                "source": self._graph_meta.get("source", "none"),
                "sha256": self._graph_meta.get("digest"),
                "available": bool(self._graph_meta.get("available")),
            },
            "semantic": {
                "sha256": self._semantic_meta.get("digest"),
                "available": bool(self._semantic_meta.get("available")),
                "start": _json_safe(self.semantic_start),
                "depth": self.semantic_depth,
                "dimensions": self.semantic_dimensions,
            },
            "derivation": (
                "deterministic local projection over at most "
                f"{self.max_rows} memory rows, "
                f"{self.max_graph_edges} concept-graph edges, "
                f"{self.max_vertices} vertices, "
                f"{self.max_triangles} triangles, "
                f"{self.max_channels} channels and "
                f"{self.max_patterns} provenance patterns"),
        }

        integration = {
            "available": True,
            "reason": "",
            "locality": LOCALITY,
            "derived": True,
            "local_only": True,
            "canonical_state": "read-only; no writes",
            "external_calls": [],
            "modules_reused": list(MODULES_REUSED),
            "unwired": list(UNWIRED),
            "claims": dict(CLAIMS),
        }

        return {
            "schema": SCHEMA,
            "locality": LOCALITY,
            "canonical_state_modified": False,
            "claims": dict(CLAIMS),
            "provenance": provenance,
            "integration": integration,
            "sections": {
                "gauge_cycle": gauge,
                "phase_coherence": phase,
                "pair_spectral": pair_spectral,
                "robust_recovery": robust,
                "concept_graph": concept,
                "semantic_state": semantic,
            },
        }

    # -- public read API ----------------------------------------------------

    def to_dict(self) -> dict:
        """Return a fresh, deterministic copy of the full summary."""
        return copy.deepcopy(self._payload)

    def summary(self) -> dict:
        """Return a small deterministic digest of the full summary."""
        sections = []
        for name in SECTION_ORDER:
            section = self._payload["sections"][name]
            sections.append({
                "section": name,
                "available": bool(section["available"]),
                "reason": section["reason"],
            })
        gauge = self._payload["sections"]["gauge_cycle"]
        phase = self._payload["sections"]["phase_coherence"]
        pair = self._payload["sections"]["pair_spectral"]
        robust = self._payload["sections"]["robust_recovery"]
        headline = {
            "cycle_dimension": (
                gauge["invariants"].get("cycle_rank")
                if gauge["invariants"].get("available") else None),
            "fundamental_cycle_count": (
                gauge["invariants"].get("fundamental_cycle_count")
                if gauge["invariants"].get("available") else None),
            "gauge_retrieval_group": (
                gauge["gauge_fixed_retrieval"].get("best_group")
                if gauge["gauge_fixed_retrieval"].get("available") else None),
            "mean_triple_coherence": (
                phase.get("mean_triple_coherence")
                if phase["available"] else None),
            "fiedler_values": (
                pair["spectral"].get("fiedler_values")
                if pair["spectral"].get("available") else None),
            "robust_gauge_flat": (
                robust.get("gauge_flat") if robust["available"] else None),
        }
        return {
            "schema": SCHEMA,
            "locality": LOCALITY,
            "input_sha256": self._input_sha256,
            "counts": copy.deepcopy(
                self._payload["provenance"]["counts"]),
            "sections": sections,
            "headline": copy.deepcopy(headline),
            "canonical_state_modified": False,
            "claims": dict(CLAIMS),
        }
