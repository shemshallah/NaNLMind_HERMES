"""All 32,385 root cycles as first-class typed hypergraph members.

This module is the full-space companion to
:mod:`nanlmind_hermes.resident.cycle_hypergraph`.  The earlier layer attaches
individual cycle observables; this layer attaches the *complete* root cycle
space ``(root, i, j)`` for every distinct ``(i, j)`` pair and gives those
cycles typed members, concept membership and recursive parent hyperedges.

Hard properties
---------------
* There are ``C(nodes - 1, 2)`` root cycles.  With the default 256 heads and
  root ``0`` this is exactly ``32,385`` (never ``256``).
* A cycle is identified by a deterministic content address.  No Python
  dictionary of all cycles is authoritative; lookups use the resident
  hyperedge index.
* A cycle hyperedge stores references only: three 16-bit pair addresses, three
  head indices, one self cycle reference, one raw integer value and one payload
  digest.  Pair cells and the 32,640-entry pair graph are never duplicated.
* Concepts are resident ``concept`` leaves; links and parent compositions are
  append-only typed hyperedges.  A parent edge is a reference, not a second
  copy of its children.
* Traversal, snapshots and audits are explicitly bounded.  The audit default
  scans the complete resident hyperedge table, so the 32,385-cycle space is not
  silently truncated at 256.

No canonical merge is performed.  Composition returns a parent-edge receipt
and never creates a merged cycle record or rewrites a source cycle.
"""
from __future__ import annotations

import bisect
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Iterable, Iterator

from .cells import canonical_address, mirror_address, reverse16
from .leaves import (
    HEADER_BYTES as LEAF_HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    RECORD_BYTES as LEAF_RECORD_BYTES,
    LeafStore,
)
from .relations import (
    RelationBoundsError,
    RelationConflictError,
    RelationError,
    RelationStore,
)

SCHEMA = "nanlm.resident-full-cycle-hypergraph/1"

DEFAULT_NODES = 256
MAX_NODES = 256
DEFAULT_ROOT = 0
DEFAULT_EXTRA_EDGES = 8192
DEFAULT_CONCEPT_CAPACITY = 4096
DEFAULT_SNAPSHOT_CYCLES = 32385
DEFAULT_NEIGHBOR_LIMIT = 64
DEFAULT_LINK_LIMIT = 64
MAX_NEIGHBOR_HOPS = 8
MAX_NEIGHBOR_LIMIT = 512
MAX_SNAPSHOT_LIMIT = 32385
MAX_AUDIT_LIMIT = 1_000_000
MAX_PROBLEMS = 256
MAX_PARENT_DEPTH = 32

VALUE_BYTES = 32
VALUE_MIN = -(1 << (VALUE_BYTES * 8 - 1))
VALUE_MAX = (1 << (VALUE_BYTES * 8 - 1)) - 1
ZERO_ID = b"\x00" * 32
LEAF_ID_BYTES = 16

# Relation-store built-ins: 1=leaf, 2=relation, 3=pair, 4=mirror.
# The values 5..11 preserve the existing cycle_hypergraph member codes where
# they overlap and extend the vocabulary for concepts and parent edges.
MEMBER_LEAF = 1
MEMBER_PAIR = 3
MEMBER_CYCLE = 5
MEMBER_VALUE = 6
MEMBER_HEAD = 7
MEMBER_VERTEX = MEMBER_HEAD
MEMBER_PAYLOAD = 8
MEMBER_CONCEPT = 9
MEMBER_PARENT = 10
MEMBER_HYPEREDGE = 11

MEMBER_NAMES = {
    MEMBER_LEAF: "leaf",
    MEMBER_PAIR: "pair",
    MEMBER_CYCLE: "cycle",
    MEMBER_VALUE: "value",
    MEMBER_HEAD: "head",
    MEMBER_PAYLOAD: "payload",
    MEMBER_CONCEPT: "concept",
    MEMBER_PARENT: "parent",
    MEMBER_HYPEREDGE: "hyperedge",
}
MEMBER_CODES = {name: code for code, name in MEMBER_NAMES.items()}
MEMBERSHIP_KINDS = {
    MEMBER_CYCLE, MEMBER_CONCEPT, MEMBER_PARENT, MEMBER_HYPEREDGE, MEMBER_HEAD,
}

CYCLE_PREDICATE = hashlib.sha256(b"NLMR-FULL-CYCLE-PREDICATE\0").digest()
CONCEPT_PREDICATE = hashlib.sha256(b"NLMR-FULL-CONCEPT-PREDICATE\0").digest()
LINK_PREDICATE = hashlib.sha256(b"NLMR-FULL-LINK-PREDICATE\0").digest()
COMPOSE_PREDICATE = hashlib.sha256(
    b"NLMR-FULL-COMPOSE-PREDICATE\0").digest()

FULL_CYCLE_COUNT = (DEFAULT_NODES - 1) * (DEFAULT_NODES - 2) // 2  # 32,385
ROOT = DEFAULT_ROOT
ROOT_HEAD = DEFAULT_ROOT
CYCLE_HEAD_COUNT = DEFAULT_NODES - 1
CYCLE_COUNT = FULL_CYCLE_COUNT
DIMENSION = FULL_CYCLE_COUNT
BASIS_SIZE = FULL_CYCLE_COUNT
ROOT_CYCLE_COUNT = FULL_CYCLE_COUNT
full_cycle_count = FULL_CYCLE_COUNT


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class FullCycleHypergraphError(ValueError):
    """Base error for full-cycle hypergraph operations."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


class FullCycleBoundsError(FullCycleHypergraphError):
    """A capacity, traversal or numeric bound was exceeded."""


class FullCycleProvenanceError(FullCycleHypergraphError):
    """A cycle, concept or edge was supplied without provenance."""


class FullCycleConflictError(FullCycleHypergraphError):
    """An immutable identity already names different content."""


class FullCycleCorruptionError(FullCycleHypergraphError):
    """Persisted links do not agree with their deterministic identity."""


class FullCycleNotFound(FullCycleHypergraphError, KeyError):
    """A requested cycle, concept or parent edge does not exist."""


class FullCycleDenied(FullCycleHypergraphError):
    """A composition or traversal is structurally not allowed."""


class FullCycleParentCycleError(FullCycleDenied):
    """A parent link would introduce a self/indirect parent cycle."""


# Common compatibility spellings used by earlier resident cycle layers.
FullCycleError = FullCycleHypergraphError
FullCycleRangeError = FullCycleBoundsError
CycleHypergraphError = FullCycleHypergraphError
CycleBoundsError = FullCycleBoundsError
CycleProvenanceError = FullCycleProvenanceError
CycleConflictError = FullCycleConflictError
CycleCorruptionError = FullCycleCorruptionError
CycleNotFound = FullCycleNotFound
CycleDenied = FullCycleDenied


# ---------------------------------------------------------------------------
# small value helpers
# ---------------------------------------------------------------------------
class TypedMembers(list):
    """Bounded list of typed members with a mapping-style metadata view."""

    def __init__(self, members: Iterable[Mapping[str, Any]] = (),
                 **metadata: Any):
        super().__init__(members)
        for name, value in metadata.items():
            setattr(self, name, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("members", "items"):
                return list(self)
            return getattr(self, key)
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key in ("members", "items"):
            return list(self)
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "owner": getattr(self, "owner", None),
            "kind": getattr(self, "kind", None),
            "ref": getattr(self, "ref", None),
            "cycle": getattr(self, "cycle", None),
            "cycle_id": getattr(self, "cycle_id", None),
            "concept_id": getattr(self, "concept_id", None),
            "hyperedge_id": getattr(self, "hyperedge_id", None),
            "source_digest": getattr(self, "source_digest", None),
            "value": getattr(self, "value", None),
            "payload_digest": getattr(self, "payload_digest", None),
            "leaf_id": getattr(self, "leaf_id", None),
            "pair_refs": getattr(self, "pair_refs", ()),
            "heads": getattr(self, "heads", ()),
            "cycle_ids": getattr(self, "cycle_ids", ()),
            "concept_ids": getattr(self, "concept_ids", ()),
            "parent_ids": getattr(self, "parent_ids", ()),
            "members": list(self),
            "missing": getattr(self, "missing", False),
            "bounded": True,
        }


class FullCycleNeighbors(list):
    """Bounded cycle-neighbour list with traversal evidence."""

    def __init__(self, cycles: Iterable[tuple[int, ...]] = (),
                 **metadata: Any):
        super().__init__(cycles)
        for name, value in metadata.items():
            setattr(self, name, value)

    @property
    def neighbors(self) -> list[tuple[int, ...]]:
        return list(self)

    @property
    def cycles(self) -> list[tuple[int, ...]]:
        return list(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("neighbors", "cycles"):
                return list(self)
            return getattr(self, key)
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key in ("neighbors", "cycles"):
            return list(self)
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": getattr(self, "cycle", None),
            "cycle_id": getattr(self, "cycle_id", None),
            "hops": getattr(self, "hops", None),
            "limit": getattr(self, "limit", None),
            "neighbors": list(self),
            "cycles": list(self),
            "cycle_ids": getattr(self, "cycle_ids", ()),
            "concept_ids": getattr(self, "concept_ids", ()),
            "parent_ids": getattr(self, "parent_ids", ()),
            "edges_visited": getattr(self, "edges_visited", 0),
            "members_visited": getattr(self, "members_visited", 0),
            "truncated": getattr(self, "truncated", False),
            "reason": getattr(self, "reason", None),
            "missing": getattr(self, "missing", False),
            "bounded": True,
        }


class Composition(dict):
    """Parent-composition receipt usable as a mapping or raw integer.

    ``int(receipt)`` and comparisons with an integer expose the summed raw
    cycle value.  Mapping access exposes the stored parent-edge identity and
    the typed members.  The object never represents a merged cycle record.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __int__(self) -> int:
        return int(self.get("value", 0) or 0)

    def as_dict(self) -> dict[str, Any]:
        return dict(self)

    def __eq__(self, other):
        if isinstance(other, Composition):
            return dict(self) == dict(other)
        if isinstance(other, bool):
            return False
        if isinstance(other, int):
            return int(self) == other
        if isinstance(other, (tuple, list)):
            return tuple(self.get("cycles", ())) == tuple(other)
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    __hash__ = None


def _field(record: Any, name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _field_any(record: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = _field(record, name, None)
        if value is not None:
            return value
    return default


def _host_attr(host: Any, name: str, default: Any = None) -> Any:
    if host is None:
        return default
    if isinstance(host, Mapping):
        return host.get(name, default)
    return getattr(host, name, default)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _check_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleHypergraphError(
            "%s must be an integer" % label, code="invalid_%s" % label)
    return value


def _check_bounded_int(value: Any, label: str, minimum: int = 0,
                       maximum: int | None = None) -> int:
    number = _check_int(value, label)
    if number < minimum:
        raise FullCycleBoundsError(
            "%s must be at least %d" % (label, minimum),
            code="%s_out_of_bounds" % label, details={label: number})
    if maximum is not None and number > maximum:
        raise FullCycleBoundsError(
            "%s must be at most %d" % (label, maximum),
            code="%s_too_large" % label,
            details={label: number, "maximum": maximum})
    return number


def _bytes32(value: Any, *, signed: bool = False,
             field: str = "id") -> bytes:
    """Normalize an id or value to exactly 32 big-endian bytes."""
    if isinstance(value, bool):
        raise FullCycleHypergraphError(
            "%s must not be a boolean" % field, code="invalid_%s" % field)
    if isinstance(value, int):
        try:
            return value.to_bytes(VALUE_BYTES, "big", signed=signed)
        except OverflowError as exc:
            raise FullCycleBoundsError(
                "%s does not fit in %d bytes" % (field, VALUE_BYTES),
                code="%s_out_of_bounds" % field,
                details={field: value}) from exc
    if isinstance(value, str):
        text = value.strip()
        if len(text) == VALUE_BYTES * 2:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        return hashlib.sha256(text.encode("utf-8")).digest()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise FullCycleHypergraphError(
                "%s must not be empty" % field, code="empty_%s" % field)
        if len(raw) == VALUE_BYTES:
            return raw
        if len(raw) < VALUE_BYTES:
            return raw.rjust(VALUE_BYTES, b"\x00")
        return hashlib.sha256(raw).digest()
    raise FullCycleHypergraphError(
        "%s must be an integer, string or bytes" % field,
        code="invalid_%s" % field)


def _decode_value(raw: bytes) -> int:
    return int.from_bytes(raw, "big", signed=True)


def _source_digest(value: Any) -> bytes:
    if value is None:
        raise FullCycleProvenanceError(
            "provenance source is required", code="source_required")
    if isinstance(value, bool):
        raise FullCycleProvenanceError(
            "provenance source must not be a boolean", code="invalid_source")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise FullCycleProvenanceError(
                "provenance source must not be empty", code="empty_source")
        return raw if len(raw) == VALUE_BYTES else hashlib.sha256(raw).digest()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise FullCycleProvenanceError(
                "provenance source must not be empty", code="empty_source")
        if len(text) == VALUE_BYTES * 2:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        return hashlib.sha256(text.encode("utf-8")).digest()
    if isinstance(value, int):
        return _bytes32(value, field="source")
    if isinstance(value, (Mapping, Sequence)):
        try:
            return hashlib.sha256(
                b"NLMR-FULL-SOURCE\0" + _canonical(value)).digest()
        except (TypeError, ValueError) as exc:
            raise FullCycleProvenanceError(
                "provenance source is not serializable",
                code="invalid_source") from exc
    raise FullCycleProvenanceError(
        "provenance source must be text, bytes, a mapping or an integer",
        code="invalid_source")


def _payload_value_and_digest(payload: Any) -> tuple[int, bytes, bytes]:
    """Return ``(raw_int_value, canonical_bytes, digest)`` for a payload."""
    if isinstance(payload, bool):
        raise FullCycleHypergraphError(
            "semantic payload value must be a raw integer, not a boolean",
            code="invalid_payload")
    if payload is None:
        value = 0
        canonical = _canonical({"value": 0})
    elif isinstance(payload, int):
        value = payload
        canonical = _canonical({"value": value})
    elif isinstance(payload, Mapping):
        record = dict(payload)
        raw_value = record.get("value", record.get("cycle_value", 0))
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise FullCycleHypergraphError(
                "semantic payload 'value' must be a raw integer",
                code="invalid_payload_value", details={"value": raw_value})
        value = raw_value
        canonical = _canonical(record)
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        value = 0
        canonical = b"NLMR-FULL-CYCLE-PAYLOAD\0" + bytes(payload)
    elif isinstance(payload, str):
        value = 0
        canonical = b"NLMR-FULL-CYCLE-PAYLOAD\0" + payload.encode("utf-8")
    else:
        raise FullCycleHypergraphError(
            "semantic payload must be a raw integer, mapping, string or bytes",
            code="invalid_payload")
    if value < VALUE_MIN or value > VALUE_MAX:
        raise FullCycleBoundsError(
            "cycle value does not fit in a signed %d-byte integer"
            % VALUE_BYTES, code="value_out_of_bounds",
            details={"value": value})
    digest = hashlib.sha256(
        b"NLMR-FULL-CYCLE-PAYLOAD\0" + canonical).digest()
    return value, canonical, digest


# ---------------------------------------------------------------------------
# cycle identity and exhaustive mapping
# ---------------------------------------------------------------------------
def root_cycle_count(nodes: int = DEFAULT_NODES) -> int:
    """Number of root cycles: ``C(nodes - 1, 2)``."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) \
            or nodes < 3 or nodes > MAX_NODES:
        raise FullCycleBoundsError(
            "nodes must be an integer in [3, %d]" % MAX_NODES,
            code="invalid_nodes", details={"nodes": nodes})
    return (nodes - 1) * (nodes - 2) // 2


def _check_nodes(nodes: Any) -> int:
    if isinstance(nodes, bool) or not isinstance(nodes, int) \
            or nodes < 3 or nodes > MAX_NODES:
        raise FullCycleBoundsError(
            "nodes must be an integer in [3, %d]" % MAX_NODES,
            code="invalid_nodes", details={"nodes": nodes})
    return nodes


def _check_root(root: Any, nodes: int) -> int:
    value = _check_int(root, "root")
    if value < 0 or value >= nodes:
        raise FullCycleBoundsError(
            "root must be in [0, %d)" % nodes, code="root_out_of_bounds",
            details={"root": value, "nodes": nodes})
    return value


def _parse_cycle_text(text: str) -> tuple[int, ...]:
    body = text.strip()
    for prefix in ("cycle:", "triangle:", "hyperedge:", "root-cycle:"):
        if body.lower().startswith(prefix):
            body = body[len(prefix):]
            break
    parts = body.replace(",", ":").replace("/", ":").split(":")
    parts = [part for part in parts if part != ""]
    if not parts:
        raise FullCycleHypergraphError("cycle id is empty",
                                       code="invalid_cycle")
    try:
        return tuple(int(part, 10) for part in parts)
    except ValueError as exc:
        raise FullCycleHypergraphError(
            "cycle id must contain integers", code="invalid_cycle",
            details={"cycle": text}) from exc


def normalize_cycle(cycle: Any, nodes: int = DEFAULT_NODES,
                    root: int = DEFAULT_ROOT) -> tuple[int, ...]:
    """Return the canonical, mirror-safe cycle id.

    Two-element ``(i, j)`` ids are interpreted as ``(root, i, j)``.  Three
    element ids are sorted.  All vertices must be distinct and in range.
    """
    nodes = _check_nodes(nodes)
    root = _check_root(root, nodes)
    if isinstance(cycle, Mapping):
        if "cycle" in cycle:
            cycle = cycle["cycle"]
        elif "vertices" in cycle:
            cycle = cycle["vertices"]
        elif "i" in cycle and "j" in cycle:
            cycle = (root, cycle["i"], cycle["j"])
        elif "id" in cycle:
            cycle = cycle["id"]
        else:
            raise FullCycleHypergraphError(
                "cycle mapping needs cycle, vertices, id or i/j",
                code="invalid_cycle")
    if isinstance(cycle, str):
        cycle = _parse_cycle_text(cycle)
    if isinstance(cycle, Composition):
        cycle = cycle.get("cycles", (cycle.get("cycle"),))[0] \
            if cycle.get("cycles") else cycle.get("cycle")
    if isinstance(cycle, (tuple, list)):
        values = tuple(cycle)
    else:
        raise FullCycleHypergraphError(
            "cycle must be a tuple, list, string or mapping",
            code="invalid_cycle")
    if len(values) == 2:
        values = (root,) + values
    if len(values) != 3:
        raise FullCycleHypergraphError(
            "cycle id must contain exactly three vertices",
            code="invalid_cycle", details={"cycle": values})
    vertices = tuple(_check_int(value, "cycle vertex") for value in values)
    if len(set(vertices)) != 3:
        raise FullCycleHypergraphError(
            "cycle vertices must be distinct", code="invalid_cycle",
            details={"cycle": vertices})
    if any(vertex < 0 or vertex >= nodes for vertex in vertices):
        raise FullCycleBoundsError(
            "cycle vertices must be in [0, %d)" % nodes,
            code="cycle_vertex_out_of_bounds",
            details={"cycle": vertices, "nodes": nodes})
    return tuple(sorted(vertices))


normalize_root_cycle = normalize_cycle


def _cycle_key(cycle: tuple[int, ...]) -> bytes:
    return (":".join(str(vertex) for vertex in cycle)).encode("ascii")


def cycle_id(cycle: Any, nodes: int = DEFAULT_NODES,
             root: int = DEFAULT_ROOT) -> str:
    return ":".join(str(vertex) for vertex in
                    normalize_cycle(cycle, nodes, root))


def cycle_leaf_id(cycle: Any, nodes: int = DEFAULT_NODES,
                  root: int = DEFAULT_ROOT) -> bytes:
    """Deterministic 128-bit leaf id (compatibility with cycle_hypergraph)."""
    canonical = normalize_cycle(cycle, nodes, root)
    return hashlib.sha256(
        b"NLMR-CYCLE-LEAF\0" + _cycle_key(canonical)).digest()[:LEAF_ID_BYTES]


def cycle_hyperedge_id(cycle: Any, nodes: int = DEFAULT_NODES,
                       root: int = DEFAULT_ROOT) -> bytes:
    """Deterministic 256-bit cycle hyperedge / cycle id."""
    canonical = normalize_cycle(cycle, nodes, root)
    return hashlib.sha256(
        b"NLMR-CYCLE-EDGE\0" + _cycle_key(canonical)).digest()


def cycle_relation_id(cycle: Any, nodes: int = DEFAULT_NODES,
                      root: int = DEFAULT_ROOT) -> bytes:
    canonical = normalize_cycle(cycle, nodes, root)
    return hashlib.sha256(
        b"NLMR-CYCLE-REL\0" + _cycle_key(canonical)).digest()


def cycle_edges(cycle: Any, nodes: int = DEFAULT_NODES,
                root: int = DEFAULT_ROOT
                ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a, b, c = normalize_cycle(cycle, nodes, root)
    if root in (a, b, c):
        if a == root:
            return ((root, b), (b, c), (root, c))
        if b == root:
            return ((root, a), (a, c), (root, c))
        return ((root, a), (a, b), (root, b))
    return tuple(sorted(((a, b), (b, c), (a, c))))  # type: ignore[return-value]


def pair_reference(a: int, b: int) -> int:
    """Mirror-safe canonical 16-bit pair address."""
    return canonical_address(int(a), int(b))


def _pair_vertices(reference: int) -> tuple[int, int]:
    address = int(reference) & 0xFFFF
    first = address >> 8
    second = int(format(address & 0xFF, "08b")[::-1], 2)
    return (first, second) if first <= second else (second, first)


@lru_cache(maxsize=16)
def _cycle_table(nodes: int, root: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    heads = tuple(head for head in range(nodes) if head != root)
    starts = []
    cursor = 0
    count = len(heads)
    for position in range(count):
        starts.append(cursor)
        cursor += count - position - 1
    return heads, tuple(starts)


def _validate_root_pair(i: Any, j: Any, nodes: int,
                        root: int) -> tuple[int, int]:
    nodes = _check_nodes(nodes)
    root = _check_root(root, nodes)
    first = _check_int(i, "i")
    second = _check_int(j, "j")
    if first == second:
        raise FullCycleHypergraphError(
            "cycle endpoints must be distinct", code="invalid_cycle",
            details={"i": first, "j": second})
    for value in (first, second):
        if value < 0 or value >= nodes:
            raise FullCycleBoundsError(
                "cycle vertices must be in [0, %d)" % nodes,
                code="cycle_vertex_out_of_bounds",
                details={"i": first, "j": second, "nodes": nodes})
        if value == root:
            raise FullCycleHypergraphError(
                "cycle endpoints must differ from the root",
                code="invalid_cycle", details={"root": root, "i": first,
                                               "j": second})
    if first > second:
        first, second = second, first
    return first, second


def cycle_index(i: int, j: int, nodes: int = DEFAULT_NODES,
                root: int = DEFAULT_ROOT) -> int:
    """Dense index in ``[0, full_cycle_count)`` for a root pair."""
    first, second = _validate_root_pair(i, j, nodes, root)
    heads, starts = _cycle_table(nodes, root)
    left = heads.index(first)
    right = heads.index(second)
    if left > right:
        left, right = right, left
    return int(starts[left] + (right - left - 1))


def pair_from_index(index: int, nodes: int = DEFAULT_NODES,
                    root: int = DEFAULT_ROOT) -> tuple[int, int]:
    """Inverse of :func:`cycle_index`, returning the non-root pair."""
    nodes = _check_nodes(nodes)
    root = _check_root(root, nodes)
    full = root_cycle_count(nodes)
    position = _check_bounded_int(index, "index", 0, full - 1)
    heads, starts = _cycle_table(nodes, root)
    left = bisect.bisect_right(starts, position) - 1
    right = left + (position - starts[left]) + 1
    return heads[left], heads[right]


def cycle_from_index(index: int, nodes: int = DEFAULT_NODES,
                     root: int = DEFAULT_ROOT) -> tuple[int, ...]:
    first, second = pair_from_index(index, nodes, root)
    return tuple(sorted((root, first, second)))


cycle_at = cycle_from_index
cycle_pair = pair_from_index
pair_of_index = pair_from_index
index_of = cycle_index
pair_to_index = cycle_index


def cycle_row_start(head: int, nodes: int = DEFAULT_NODES,
                    root: int = DEFAULT_ROOT) -> int:
    """First dense index of the row beginning at ``head``."""
    nodes = _check_nodes(nodes)
    root = _check_root(root, nodes)
    value = _check_int(head, "head")
    if value == root or value < 0 or value >= nodes:
        raise FullCycleBoundsError(
            "cycle row head must be a non-root head",
            code="invalid_head", details={"head": value, "root": root})
    heads, starts = _cycle_table(nodes, root)
    position = heads.index(value)
    if position == len(heads) - 1:
        return root_cycle_count(nodes)
    return int(starts[position])


def cycle_id_for_index(index: int, nodes: int = DEFAULT_NODES,
                       root: int = DEFAULT_ROOT) -> str:
    return ":".join(str(vertex) for vertex in
                    cycle_from_index(index, nodes, root))


def iter_root_pairs(nodes: int = DEFAULT_NODES, root: int = DEFAULT_ROOT,
                    limit: int | None = None) -> Iterator[tuple[int, int]]:
    nodes = _check_nodes(nodes)
    root = _check_root(root, nodes)
    full = root_cycle_count(nodes)
    if limit is None:
        limit = full
    limit = _check_bounded_int(limit, "limit", 0, full)
    heads = tuple(head for head in range(nodes) if head != root)
    emitted = 0
    for left in range(len(heads)):
        for right in range(left + 1, len(heads)):
            if emitted >= limit:
                return
            yield heads[left], heads[right]
            emitted += 1


iter_cycle_pairs = iter_root_pairs


def iter_root_cycles(nodes: int = DEFAULT_NODES, root: int = DEFAULT_ROOT,
                     limit: int | None = None) -> Iterator[tuple[int, ...]]:
    for first, second in iter_root_pairs(nodes, root, limit):
        yield tuple(sorted((root, first, second)))


iter_cycles = iter_root_cycles


def root_cycle_pairs(nodes: int = DEFAULT_NODES, root: int = DEFAULT_ROOT,
                     limit: int | None = None) -> tuple[tuple[int, int], ...]:
    return tuple(iter_root_pairs(nodes, root, limit))


def full_cycle_ids(nodes: int = DEFAULT_NODES,
                   root: int = DEFAULT_ROOT) -> tuple[str, ...]:
    """All deterministic cycle ids, useful for exhaustive verification."""
    return tuple(cycle_id_for_index(index, nodes, root)
                 for index in range(root_cycle_count(nodes)))


# ---------------------------------------------------------------------------
# concept identity
# ---------------------------------------------------------------------------
def concept_leaf_id(name: Any) -> bytes:
    """Stable 128-bit concept leaf id.

    Text names use the documented ``SHA-256("concept\\0" + casefold(term))``
    prefix.  A 16-byte value or 32-character hex string is accepted verbatim.
    """
    if isinstance(name, (bytes, bytearray, memoryview)):
        raw = bytes(name)
        if len(raw) == LEAF_ID_BYTES:
            return raw
        if len(raw) == VALUE_BYTES:
            return raw[-LEAF_ID_BYTES:]
        if not raw:
            raise FullCycleHypergraphError(
                "concept id must not be empty", code="invalid_concept")
        return hashlib.sha256(b"concept\0" + raw).digest()[:LEAF_ID_BYTES]
    if isinstance(name, str):
        text = name.strip()
        if not text:
            raise FullCycleHypergraphError(
                "concept name must not be empty", code="invalid_concept")
        if len(text) == LEAF_ID_BYTES * 2:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        if len(text) == VALUE_BYTES * 2:
            try:
                return bytes.fromhex(text)[-LEAF_ID_BYTES:]
            except ValueError:
                pass
        return hashlib.sha256(
            b"concept\0" + text.casefold().encode("utf-8")
        ).digest()[:LEAF_ID_BYTES]
    raise FullCycleHypergraphError(
        "concept name must be text or 16 bytes", code="invalid_concept")


def concept_hyperedge_id(concept: Any) -> bytes:
    return hashlib.sha256(
        b"NLMR-FULL-CONCEPT-EDGE\0" + concept_leaf_id(concept)).digest()


def _pad_leaf_id(leaf_id: bytes) -> bytes:
    return bytes(leaf_id).rjust(VALUE_BYTES, b"\x00")


def _leaf_id_from_member(raw: bytes) -> bytes:
    raw = bytes(raw)
    if len(raw) >= LEAF_ID_BYTES:
        return raw[-LEAF_ID_BYTES:]
    return raw.rjust(LEAF_ID_BYTES, b"\x00")


def _edge_identity(kind: str, *, owner: bytes | None,
                   parent: bytes | None,
                   members: Iterable[tuple[int, bytes]]) -> bytes:
    hasher = hashlib.sha256(b"NLMR-FULL-EDGE\0" + kind.encode("utf-8")
                            + b"\0")
    if owner is not None:
        hasher.update(b"O" + bytes(owner))
    if parent is not None:
        hasher.update(b"P" + bytes(parent))
    for code, identifier in sorted(members):
        hasher.update(bytes((int(code),)))
        hasher.update(bytes(identifier))
    return hasher.digest()


def _compose_payload(cycle_ids: Iterable[bytes], values: Iterable[int],
                     concept_ids: Iterable[bytes],
                     parent_id: bytes | None) -> bytes:
    hasher = hashlib.sha256(b"NLMR-FULL-COMPOSE-PAYLOAD\0")
    # Sort cycles together with their values so argument order cannot change
    # the composition identity.
    paired = sorted((bytes(identifier), int(value))
                    for identifier, value in zip(cycle_ids, values))
    for identifier, value in paired:
        hasher.update(identifier)
        hasher.update(bytes(_bytes32(value, signed=True)))
    for identifier in sorted(bytes(item) for item in concept_ids):
        hasher.update(identifier)
    if parent_id is not None:
        hasher.update(bytes(parent_id))
    return hasher.digest()


# ---------------------------------------------------------------------------
# store adapters
# ---------------------------------------------------------------------------
def _leaf_buffer(capacity: int) -> bytearray:
    index_capacity = _next_power_of_two(max(16, capacity * 2))
    return bytearray(
        LEAF_HEADER_BYTES + capacity * LEAF_RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)


def _default_stores(nodes: int = DEFAULT_NODES,
                    extra_edges: int = DEFAULT_EXTRA_EDGES,
                    concept_capacity: int = DEFAULT_CONCEPT_CAPACITY,
                    capacity: int | None = None):
    full = root_cycle_count(nodes)
    extra = max(64, min(int(extra_edges), max(64, full)))
    edge_capacity = full + extra if capacity is None else int(capacity)
    edge_capacity = max(1, edge_capacity)
    relation_capacity = edge_capacity + extra
    # Cycle relations add one unique subject and (usually) one unique object
    # key.  Two chain slots per relation leave headroom for concept/link edges
    # while keeping the default resident section substantially smaller.
    chain_capacity = _next_power_of_two(max(16, relation_capacity * 2 + 32))
    cycle_edges = min(full, edge_capacity)
    arena_bytes = _align8(cycle_edges * 9 * (VALUE_BYTES + 1)
                          + extra * 16 * (VALUE_BYTES + 1))
    relations = RelationStore.allocate(
        relation_capacity=relation_capacity,
        hyperedge_capacity=edge_capacity,
        chain_capacity=chain_capacity,
        arena_bytes=arena_bytes)
    leaves = LeafStore.create(
        _leaf_buffer(max(16, int(concept_capacity))),
        capacity=max(16, int(concept_capacity)))
    return relations, leaves, None


def _resolve_stores(universe_or_stores: Any = None, *,
                    relations: Any = None, leaves: Any = None,
                    cells: Any = None):
    """Resolve a universe, mapping, tuple/list or direct stores."""
    host = None
    source = universe_or_stores
    if source is None:
        pass
    elif isinstance(source, Mapping) and (
            "relations" in source or "leaves" in source or "cells" in source):
        relations = relations if relations is not None else source.get(
            "relations")
        leaves = leaves if leaves is not None else source.get("leaves")
        cells = cells if cells is not None else source.get("cells")
        host = source.get("host")
    elif isinstance(source, (tuple, list)):
        values = list(source)
        if len(values) >= 1 and relations is None:
            relations = values[0]
        if len(values) >= 2 and leaves is None:
            leaves = values[1]
        if len(values) >= 3 and cells is None:
            cells = values[2]
    elif _host_attr(source, "relations", None) is not None:
        host = source
        relations = relations if relations is not None else source.relations
        leaves = leaves if leaves is not None else _host_attr(source, "leaves")
        cells = cells if cells is not None else _host_attr(source, "cells")
    elif relations is None and (
            callable(_host_attr(source, "get_hyperedge"))
            or callable(_host_attr(source, "append_relation"))
            or callable(_host_attr(source, "append_hyperedge"))):
        relations = source
    elif leaves is None and callable(_host_attr(source, "insert")):
        leaves = source
    elif cells is None and (
            callable(_host_attr(source, "read_rail"))
            or callable(_host_attr(source, "read_raw_word"))
            or callable(_host_attr(source, "pair_index"))):
        cells = source
    if relations is None or leaves is None:
        default_relations, default_leaves, _ = _default_stores()
        if relations is None:
            relations = default_relations
        if leaves is None:
            leaves = default_leaves
    return relations, leaves, cells, host


def _member_pairs(view: Any) -> list[tuple[int, bytes]]:
    """Normalize a hyperedge view into ``(kind_code, id_bytes)`` pairs."""
    members = _field_any(view, ("members", "member_ids"))
    kinds = _field_any(view, ("member_kinds",))
    out: list[tuple[int, bytes]] = []
    if members is None:
        return out
    for index, member in enumerate(members):
        if isinstance(member, Mapping):
            code = member.get("kind_code")
            if code is None:
                name = member.get("kind")
                if isinstance(name, str):
                    if name.startswith("kind:"):
                        try:
                            code = int(name[5:])
                        except ValueError:
                            code = None
                    elif name in MEMBER_CODES:
                        code = MEMBER_CODES[name]
            identifier = member.get("id", member.get("member_id"))
            if identifier is None and "id_int" in member:
                identifier = member["id_int"]
        elif isinstance(member, (tuple, list)) and len(member) == 2:
            kind, identifier = member
            if isinstance(kind, int) and not isinstance(kind, bool):
                code = kind
            else:
                code = MEMBER_CODES.get(str(kind), MEMBER_LEAF)
        else:
            if kinds is not None and index < len(kinds):
                kind = kinds[index]
                if isinstance(kind, int):
                    code = kind
                else:
                    name = str(kind)
                    if name.startswith("kind:"):
                        try:
                            code = int(name[5:])
                        except ValueError:
                            code = MEMBER_LEAF
                    else:
                        code = MEMBER_CODES.get(name, MEMBER_LEAF)
            else:
                code = MEMBER_LEAF
            identifier = member
        if isinstance(code, bool) or not isinstance(code, int):
            code = MEMBER_LEAF
        out.append((code, _bytes32(identifier, field="member id")))
    return out


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------
class FullCycleHypergraph:
    """Bounded, resident, recursive typed hypergraph over all root cycles."""

    full_cycle_count = FULL_CYCLE_COUNT
    cycle_capacity = FULL_CYCLE_COUNT
    FULL_CYCLE_COUNT = FULL_CYCLE_COUNT
    CYCLE_COUNT = FULL_CYCLE_COUNT
    DIMENSION = FULL_CYCLE_COUNT
    BASIS_SIZE = FULL_CYCLE_COUNT
    SCHEMA = SCHEMA

    def __init__(self, universe_or_stores: Any = None, *,
                 relations: Any = None, leaves: Any = None, cells: Any = None,
                 nodes: int = DEFAULT_NODES, root: int = DEFAULT_ROOT,
                 capacity: int | None = None,
                 concept_capacity: int = DEFAULT_CONCEPT_CAPACITY,
                 extra_edges: int = DEFAULT_EXTRA_EDGES,
                 commit: Any = None, flush: Any = None,
                 scan_limit: int = 0):
        self.nodes = _check_nodes(nodes)
        self.root = _check_root(root, self.nodes)
        full = root_cycle_count(self.nodes)
        self.full_cycle_count = full
        self.cycle_capacity = full
        self.dimension = full
        self.basis_size = full
        self.root_cycle_count = full
        self.extra_edges = max(64, min(int(extra_edges),
                                       max(64, full)))
        self.scan_limit = _check_bounded_int(scan_limit, "scan_limit", 0,
                                             MAX_AUDIT_LIMIT)
        # A bounded fallback scan is intentionally opt-in: the default add
        # path is O(1) and never scans the growing hyperedge table.
        if capacity is not None:
            capacity = _check_bounded_int(capacity, "capacity", 1)
        else:
            capacity = full + self.extra_edges
        self.capacity = capacity
        if relations is None and leaves is None and universe_or_stores is None:
            relations, leaves, cells = _default_stores(
                self.nodes, self.extra_edges, concept_capacity,
                capacity=capacity)
        self.relations, self.leaves, self.cells, self.host = _resolve_stores(
            universe_or_stores, relations=relations, leaves=leaves,
            cells=cells)
        self._commit = commit if commit is not None else (
            _host_attr(self.host, "_commit") or _host_attr(self.host, "commit"))
        self._flush = flush if flush is not None else _host_attr(
            self.host, "flush")

    # -- metadata -------------------------------------------------------
    @property
    def canonical_merge(self) -> bool:
        """Canonical merge is never performed by this layer."""
        return False

    @property
    def head_count(self) -> int:
        return self.nodes

    @property
    def pair_count(self) -> int:
        return self.nodes * (self.nodes - 1) // 2

    def cycle_id(self, i: int, j: int) -> str:
        """Canonical textual id for one root cycle."""
        return cycle_id((i, j), self.nodes, self.root)

    def cycle_ref(self, i: int, j: int) -> dict[str, Any]:
        """Small deterministic cycle reference for use in parent edges."""
        cycle = normalize_cycle((i, j), self.nodes, self.root)
        edge_id = cycle_hyperedge_id(cycle, self.nodes, self.root)
        non_root = tuple(vertex for vertex in cycle if vertex != self.root)
        return {
            "kind": "cycle", "kind_code": MEMBER_CYCLE,
            "id": edge_id.hex(), "hyperedge_id": edge_id.hex(),
            "cycle": cycle, "cycle_id": ":".join(str(v) for v in cycle),
            "cycle_index": cycle_index(non_root[0], non_root[1],
                                       self.nodes, self.root),
        }

    cycle_reference = cycle_ref

    def cycle_index(self, i: int, j: int) -> int:
        return cycle_index(i, j, self.nodes, self.root)

    def cycle_pair(self, index: int) -> tuple[int, int]:
        return pair_from_index(index, self.nodes, self.root)

    pair_of_index = cycle_pair

    def cycle_hyperedge_id(self, i: int, j: int) -> bytes:
        return cycle_hyperedge_id((i, j), self.nodes, self.root)

    def cycle_ids(self, limit: int | None = None) -> tuple[str, ...]:
        return tuple(cycle_id_for_index(index, self.nodes, self.root)
                     for index in range(
                         self.full_cycle_count if limit is None else
                         _check_bounded_int(limit, "limit", 0,
                                            self.full_cycle_count)))

    def iter_cycles(self, limit: int | None = None
                    ) -> Iterator[tuple[int, ...]]:
        yield from iter_root_cycles(self.nodes, self.root, limit)

    def iter_cycle_pairs(self, limit: int | None = None
                         ) -> Iterator[tuple[int, int]]:
        yield from iter_root_pairs(self.nodes, self.root, limit)

    def attached_cycle_count(self) -> int:
        return self._scan_cycles(validate=False)["attached_cycle_count"]

    cycle_count = attached_cycle_count

    @property
    def generation(self) -> int | None:
        status = self._relation_status()
        for owner in (self.host, self.relations):
            value = _host_attr(owner, "generation", None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = status.get("generation")
        return value if isinstance(value, int) else None

    def __len__(self) -> int:
        return self._scan_cycles(validate=False)["attached_cycle_count"]

    def _relation_status(self) -> dict[str, Any]:
        status = _host_attr(self.relations, "status", None)
        if callable(status):
            try:
                value = status()
            except Exception:
                return {}
            return dict(value) if isinstance(value, Mapping) else {}
        if isinstance(status, Mapping):
            return dict(status)
        return {}

    def _hyperedge_count(self) -> int | None:
        status = self._relation_status()
        for key in ("hyperedges", "hyperedge_count"):
            value = status.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = _host_attr(self.relations, "hyperedge_count", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        records = _host_attr(self.relations, "hyperedges", None)
        if records is None:
            records = _host_attr(self.relations, "hyperedge_records", None)
        if isinstance(records, Mapping) or isinstance(records, (list, tuple)):
            return len(records)
        return None

    def _leaf_count(self) -> int | None:
        stats = _host_attr(self.leaves, "stats", None)
        if callable(stats):
            try:
                value = stats()
                if isinstance(value, Mapping) and isinstance(
                        value.get("count"), int):
                    return int(value["count"])
            except Exception:
                pass
        count = _host_attr(self.leaves, "count", None)
        if isinstance(count, int) and not isinstance(count, bool):
            return count
        try:
            return len(self.leaves)
        except Exception:
            return None

    # -- hyperedge access ----------------------------------------------
    def _get_hyperedge(self, identifier: Any) -> Any:
        get = _host_attr(self.relations, "get_hyperedge", None)
        if callable(get):
            keys = [identifier]
            if isinstance(identifier, (bytes, bytearray, memoryview)):
                keys.append(bytes(identifier).hex())
            for key in keys:
                try:
                    value = get(key)
                except Exception:
                    value = None
                if value is not None:
                    return value
        records = _host_attr(self.relations, "hyperedges", None)
        if isinstance(records, Mapping):
            for key in (identifier,
                        bytes(identifier).hex()
                        if isinstance(identifier,
                                      (bytes, bytearray, memoryview))
                        else identifier):
                if key in records:
                    return records[key]
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            if isinstance(identifier, int) and not isinstance(identifier, bool):
                if 0 <= identifier < len(records):
                    return records[identifier]
            for record in records:
                stored = _field(record, "hyperedge_id")
                if stored == identifier:
                    return record
                if isinstance(identifier, (bytes, bytearray, memoryview)) \
                        and stored == bytes(identifier).hex():
                    return record
        return None

    def _iter_hyperedges(self, limit: int | None = None) -> Iterator[Any]:
        if limit is None:
            limit = max(1, self._hyperedge_count() or MAX_AUDIT_LIMIT)
        limit = max(0, int(limit))
        iterator = _host_attr(self.relations, "iter_hyperedges", None)
        if callable(iterator):
            for index, item in enumerate(iterator()):
                if index >= limit:
                    break
                yield item
            return
        records = _host_attr(self.relations, "hyperedges", None)
        if records is None:
            records = _host_attr(self.relations, "hyperedge_records", None)
        if isinstance(records, Mapping):
            for index, item in enumerate(records.values()):
                if index >= limit:
                    break
                yield item
            return
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            for index, item in enumerate(records):
                if index >= limit:
                    break
                yield item
            return
        count = self._hyperedge_count()
        get = _host_attr(self.relations, "get_hyperedge", None)
        if callable(get) and count is not None:
            for index in range(min(count, limit)):
                item = self._get_hyperedge(index)
                if item is not None:
                    yield item

    def _find_cycle_view(self, cycle: tuple[int, ...]) -> Any | None:
        direct = self._get_hyperedge(cycle_hyperedge_id(
            cycle, self.nodes, self.root))
        if direct is not None:
            parsed = self._parse_view(direct)
            if parsed["kind"] == "cycle" and parsed["cycle"] == cycle:
                return direct
        if self.scan_limit:
            for view in self._iter_hyperedges(self.scan_limit):
                parsed = self._parse_view(view)
                if parsed["kind"] == "cycle" and parsed["cycle"] == cycle:
                    return view
        return None

    def _cycle_exists(self, cycle: tuple[int, ...]) -> bool:
        direct = self._get_hyperedge(cycle_hyperedge_id(
            cycle, self.nodes, self.root))
        if direct is not None:
            parsed = self._parse_view(direct)
            return parsed["kind"] == "cycle" and parsed["cycle"] == cycle
        return False

    # -- leaves ---------------------------------------------------------
    def _find_leaf(self, leaf_id: bytes) -> Any | None:
        find = _host_attr(self.leaves, "find", None)
        if callable(find):
            for key in (leaf_id, bytes(leaf_id).hex()):
                try:
                    record = find(key)
                except Exception:
                    record = None
                if record is not None:
                    return record
        get = _host_attr(self.leaves, "get", None)
        if callable(get):
            for key in (leaf_id, bytes(leaf_id).hex()):
                try:
                    return get(key)
                except Exception:
                    continue
        return None

    def _insert_concept_leaf(self, leaf_id: bytes, payload_digest: bytes,
                             canonical_payload: bytes,
                             parent: bytes | None) -> Any:
        insert = _host_attr(self.leaves, "insert", None)
        if not callable(insert):
            raise FullCycleHypergraphError(
                "leaf store does not support insert",
                code="unsupported_store")
        kwargs: dict[str, Any] = {
            "leaf_id": leaf_id,
            "payload_digest": payload_digest,
            "content": canonical_payload,
        }
        if parent is not None:
            kwargs["parent"] = parent
        try:
            return insert("concept", **kwargs)
        except TypeError:
            kwargs.pop("content", None)
            try:
                return insert("concept", **kwargs)
            except TypeError:
                kwargs.pop("parent", None)
                return insert("concept", **kwargs)

    def _leaf_record(self, leaf_id: bytes) -> Any | None:
        record = self._find_leaf(leaf_id)
        if record is not None:
            return record
        records = _host_attr(self.leaves, "records", None)
        if callable(records):
            for item in records():
                if _field(item, "leaf_id") == leaf_id:
                    return item
        return None

    def _leaf_parent_id(self, record: Any) -> bytes | None:
        parent = _field(record, "parent")
        if parent is None:
            return None
        leaf_at = _host_attr(self.leaves, "leaf_at", None)
        if callable(leaf_at):
            try:
                parent_record = leaf_at(int(parent))
            except Exception:
                parent_record = None
            if parent_record is not None:
                return bytes(_field(parent_record, "leaf_id"))
        return None

    def _concept_ancestors(self, concept: bytes, max_depth: int = MAX_PARENT_DEPTH
                           ) -> list[bytes]:
        """Follow resident leaf parent links, bounded and cycle-safe."""
        ancestors: list[bytes] = []
        current = bytes(concept)
        seen: set[bytes] = set()
        for _ in range(max_depth + 1):
            if current in seen:
                raise FullCycleParentCycleError(
                    "concept parent chain contains a cycle",
                    code="parent_cycle", details={
                        "concept_id": current.hex()})
            seen.add(current)
            record = self._leaf_record(current)
            if record is None:
                break
            parent = self._leaf_parent_id(record)
            if parent is None:
                break
            if len(ancestors) >= max_depth:
                raise FullCycleParentCycleError(
                    "concept parent depth exceeds %d" % max_depth,
                    code="parent_depth_exceeded",
                    details={"concept_id": concept.hex()})
            ancestors.append(parent)
            current = parent
        return ancestors

    # -- parse views ----------------------------------------------------
    def _parse_view(self, view: Any) -> dict[str, Any]:
        pairs = _member_pairs(view)
        edge_id_raw = _field_any(view, ("hyperedge_id", "id"))
        edge_id = _bytes32(edge_id_raw, field="hyperedge_id") \
            if edge_id_raw is not None else None
        source_raw = _field_any(view, ("source_digest", "source"))
        source = _bytes32(source_raw, field="source_digest") \
            if source_raw is not None else None
        generation = _field(view, "generation")
        typed: list[dict[str, Any]] = []
        pair_refs: list[int] = []
        heads: list[int] = []
        cycle_ids: list[bytes] = []
        concept_ids: list[bytes] = []
        parent_ids: list[bytes] = []
        hyperedge_ids: list[bytes] = []
        value = None
        payload_digest = None
        duplicate_member_count = 0
        seen_members: set[tuple[int, bytes]] = set()
        for position, (code, raw) in enumerate(pairs):
            if (code, raw) in seen_members:
                duplicate_member_count += 1
            seen_members.add((code, raw))
            if code == MEMBER_PAIR:
                reference = int.from_bytes(raw, "big")
                first, second = _pair_vertices(reference)
                pair_refs.append(reference)
                typed.append({
                    "index": position, "kind": "pair", "kind_code": code,
                    "id": raw.hex(), "id_int": reference,
                    "address": reference,
                    "canonical_address": reference,
                    "mirror": mirror_address(reference),
                    "pair": (first, second) if first != second else None,
                })
            elif code == MEMBER_HEAD:
                head = int.from_bytes(raw, "big")
                heads.append(head)
                typed.append({
                    "index": position, "kind": "head", "kind_code": code,
                    "id": raw.hex(), "head": head, "vertex": head,
                })
            elif code == MEMBER_CYCLE:
                identifier = bytes(raw)
                cycle_ids.append(identifier)
                # The cycle's own self-member must not recurse back into
                # _parse_view; resolve it after the head members are known.
                self_member = edge_id is not None and identifier == edge_id
                resolved = None if (self_member or edge_id is None) else \
                    self._resolve_cycle_id(identifier)
                typed.append({
                    "index": position, "kind": "cycle", "kind_code": code,
                    "id": identifier.hex(),
                    "cycle": None if resolved is None else resolved[0],
                    "cycle_id": None if resolved is None else resolved[1],
                    "resolved": self_member or resolved is not None,
                    "self": self_member,
                })
            elif code == MEMBER_CONCEPT:
                identifier = _leaf_id_from_member(raw)
                concept_ids.append(identifier)
                typed.append({
                    "index": position, "kind": "concept", "kind_code": code,
                    "id": identifier.hex(), "concept_id": identifier.hex(),
                    "leaf_id": identifier.hex(),
                })
            elif code == MEMBER_PARENT:
                identifier = bytes(raw)
                parent_ids.append(identifier)
                typed.append({
                    "index": position, "kind": "parent", "kind_code": code,
                    "id": identifier.hex(), "parent_id": identifier.hex(),
                })
            elif code == MEMBER_HYPEREDGE:
                identifier = bytes(raw)
                hyperedge_ids.append(identifier)
                typed.append({
                    "index": position, "kind": "hyperedge", "kind_code": code,
                    "id": identifier.hex(), "hyperedge_id": identifier.hex(),
                })
            elif code == MEMBER_VALUE:
                value = _decode_value(raw)
                typed.append({
                    "index": position, "kind": "value", "kind_code": code,
                    "id": raw.hex(), "value": value,
                })
            elif code == MEMBER_PAYLOAD:
                payload_digest = bytes(raw)
                typed.append({
                    "index": position, "kind": "payload", "kind_code": code,
                    "id": raw.hex(), "digest": raw.hex(),
                })
            else:
                typed.append({
                    "index": position, "kind": "foreign", "kind_code": code,
                    "id": raw.hex(),
                })
        cycle = None
        if len(heads) == 3:
            try:
                cycle = normalize_cycle(tuple(heads), self.nodes, self.root)
            except FullCycleHypergraphError:
                cycle = None
        kind = "unknown"
        if len(pair_refs) == 3 and len(heads) == 3:
            kind = "cycle"
        elif concept_ids and (cycle_ids or parent_ids or hyperedge_ids):
            kind = "link"
        elif parent_ids or cycle_ids:
            kind = "compose"
        elif concept_ids:
            # A concept's own metadata edge has a deterministic id.  A
            # concept-to-concept membership edge is a link, not a duplicate
            # concept record.
            if edge_id is not None and concept_ids \
                    and edge_id == concept_hyperedge_id(concept_ids[0]):
                kind = "concept"
            else:
                kind = "link"
        if kind == "cycle" and cycle is not None:
            cycle_text = ":".join(str(vertex) for vertex in cycle)
            for member in typed:
                if member.get("kind") == "cycle" and member.get("self"):
                    member["cycle"] = cycle
                    member["cycle_id"] = cycle_text
                    member["resolved"] = True
        return {
            "hyperedge_id": edge_id,
            "hyperedge_id_hex": None if edge_id is None else edge_id.hex(),
            "source_digest": source,
            "generation": generation,
            "members": typed,
            "member_pairs": pairs,
            "pair_refs": tuple(pair_refs),
            "heads": tuple(heads),
            "cycle_ids": tuple(cycle_ids),
            "concept_ids": tuple(concept_ids),
            "parent_ids": tuple(parent_ids),
            "hyperedge_ids": tuple(hyperedge_ids),
            "value": value,
            "payload_digest": payload_digest,
            "cycle": cycle,
            "cycle_id": None if cycle is None else ":".join(
                str(vertex) for vertex in cycle),
            "kind": kind,
            "duplicate_member_count": duplicate_member_count,
        }

    def _resolve_cycle_id(self, identifier: bytes
                          ) -> tuple[tuple[int, ...], str] | None:
        view = self._get_hyperedge(identifier)
        if view is None:
            return None
        parsed = self._parse_view(view)
        if parsed["kind"] != "cycle" or parsed["cycle"] is None:
            return None
        return parsed["cycle"], parsed["cycle_id"]

    # -- capacity, append and commit ------------------------------------
    def _check_capacity(self, *, hyperedges: int = 1, relations: int = 1,
                        members: int = 0, leaf: bool = False,
                        chains: int | None = None) -> None:
        status = self._relation_status()
        if chains is None:
            chains = relations * 2
        chain_used = status.get("chain_used")
        chain_capacity = status.get("chain_capacity")
        if isinstance(chain_used, int) and isinstance(chain_capacity, int) \
                and chain_used + chains > chain_capacity:
            raise FullCycleBoundsError(
                "relation chain table capacity exhausted",
                code="chain_capacity",
                details={"used": chain_used, "required": chains,
                         "capacity": chain_capacity})
        hyperedge_count = status.get("hyperedges")
        hyperedge_capacity = status.get("hyperedge_capacity")
        if isinstance(hyperedge_count, int) and isinstance(
                hyperedge_capacity, int) \
                and hyperedge_count + hyperedges > hyperedge_capacity:
            raise FullCycleBoundsError(
                "hyperedge capacity exhausted", code="hyperedge_capacity",
                details={"count": hyperedge_count,
                         "required": hyperedges,
                         "capacity": hyperedge_capacity})
        relation_count = status.get("relations")
        relation_capacity = status.get("relation_capacity")
        if isinstance(relation_count, int) and isinstance(
                relation_capacity, int) \
                and relation_count + relations > relation_capacity:
            raise FullCycleBoundsError(
                "relation capacity exhausted", code="relation_capacity",
                details={"count": relation_count,
                         "required": relations,
                         "capacity": relation_capacity})
        arena_used = status.get("arena_used")
        arena_bytes = status.get("arena_bytes")
        if members and isinstance(arena_used, int) and isinstance(
                arena_bytes, int):
            needed = _align8(members * (VALUE_BYTES + 1))
            if arena_used + needed > arena_bytes:
                raise FullCycleBoundsError(
                    "hyperedge member arena exhausted",
                    code="arena_capacity",
                    details={"used": arena_used, "required": needed,
                             "capacity": arena_bytes})
        if leaf:
            leaf_capacity = _host_attr(self.leaves, "capacity", None)
            leaf_count = _host_attr(self.leaves, "count", None)
            if not isinstance(leaf_count, int):
                try:
                    leaf_count = len(self.leaves)
                except Exception:
                    leaf_count = None
            if isinstance(leaf_capacity, int) and isinstance(leaf_count, int) \
                    and leaf_count >= leaf_capacity:
                raise FullCycleBoundsError(
                    "concept leaf capacity exhausted", code="leaf_capacity",
                    details={"count": leaf_count,
                             "capacity": leaf_capacity})

    def _append_relation_edge(self, *, hyperedge_id: bytes,
                              relation_id: bytes, subject: bytes,
                              predicate: bytes, object_id: bytes,
                              source: bytes,
                              members: list[tuple[int, bytes]]) -> dict[str, Any]:
        append_edge = _host_attr(self.relations, "append_hyperedge", None)
        append_relation = _host_attr(self.relations, "append_relation", None)
        if callable(append_relation):
            try:
                result = append_relation(
                    subject=subject, predicate=predicate, object=object_id,
                    polarity="supports", confidence=0.5,
                    source_digest=source, relation_id=relation_id,
                    hyperedge_id=hyperedge_id, hyperedge_members=members)
            except TypeError:
                result = append_relation(
                    subject=subject, predicate=predicate, object=object_id,
                    source_digest=source, relation_id=relation_id,
                    hyperedge_id=hyperedge_id, hyperedge_members=members)
            return dict(result) if isinstance(result, Mapping) else {}
        if callable(append_edge):
            try:
                result = append_edge(
                    hyperedge_id=hyperedge_id, source_digest=source,
                    members=members)
            except TypeError:
                result = append_edge(hyperedge_id, source, members)
            return dict(result) if isinstance(result, Mapping) else {}
        raise FullCycleHypergraphError(
            "relation store does not support hyperedge append",
            code="unsupported_store")

    def _append_incidence(self, *, hyperedge_id: bytes, subject: bytes,
                          predicate: bytes, target: bytes, source: bytes,
                          position: int) -> None:
        append_relation = _host_attr(self.relations, "append_relation", None)
        if not callable(append_relation):
            return
        relation_id = hashlib.sha256(
            b"NLMR-FULL-INCIDENCE\0" + hyperedge_id + target
            + position.to_bytes(4, "big")).digest()
        try:
            append_relation(
                subject=subject, predicate=predicate, object=target,
                source_digest=source, relation_id=relation_id,
                hyperedge_id=hyperedge_id)
        except (RelationConflictError, RelationBoundsError, RelationError) as exc:
            raise FullCycleCorruptionError(
                "incidence relation could not be appended: %s" % exc,
                code="incidence_append_failed",
                details={"hyperedge_id": hyperedge_id.hex()}) from exc

    def _append_edge(self, *, edge_id: bytes, kind: str,
                     members: list[tuple[int, bytes]], source: bytes,
                     subject: bytes, predicate: bytes, object_id: bytes,
                     relation_id: bytes,
                     incidence_targets: Iterable[bytes] = (),
                     check_capacity: bool = True,
                     readback: bool = True,
                     commit: bool = True) -> dict[str, Any]:
        targets = list(dict.fromkeys(bytes(item) for item in incidence_targets
                                     if item is not None))
        extra = len([target for target in targets if target != object_id])
        if check_capacity:
            self._check_capacity(hyperedges=1, relations=1 + extra,
                                 members=len(members))
        try:
            result = self._append_relation_edge(
                hyperedge_id=edge_id, relation_id=relation_id,
                subject=subject, predicate=predicate, object_id=object_id,
                source=source, members=members)
        except FullCycleHypergraphError:
            raise
        except RelationConflictError as exc:
            raise FullCycleConflictError(
                "hyperedge id already names different members",
                code="hyperedge_conflict",
                details={"hyperedge_id": edge_id.hex()}) from exc
        except RelationBoundsError as exc:
            raise FullCycleBoundsError(
                "hyperedge capacity exhausted: %s" % exc,
                code="hyperedge_capacity",
                details={"hyperedge_id": edge_id.hex()}) from exc
        except RelationError as exc:
            raise FullCycleConflictError(
                "hyperedge append failed: %s" % exc,
                code="hyperedge_conflict",
                details={"hyperedge_id": edge_id.hex()}) from exc
        except Exception as exc:
            raise FullCycleHypergraphError(
                "hyperedge could not be appended: %s" % exc,
                code="hyperedge_append_failed",
                details={"hyperedge_id": edge_id.hex()}) from exc
        position = 0
        for target in targets:
            if target == object_id:
                continue
            self._append_incidence(
                hyperedge_id=edge_id, subject=subject,
                predicate=predicate, target=target, source=source,
                position=position)
            position += 1
        if readback:
            view = self._get_hyperedge(edge_id)
            if view is None:
                raise FullCycleCorruptionError(
                    "hyperedge was not persisted", code="missing_hyperedge",
                    details={"hyperedge_id": edge_id.hex(), "kind": kind})
        if commit:
            self._maybe_commit(kind, edge_id.hex())
        return result

    def _maybe_commit(self, kind: str, identity: str) -> None:
        callback = self._commit
        if callback is None:
            return
        try:
            signature = inspect.signature(callback)
            positional = [
                parameter for parameter in signature.parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD)]
            varargs = any(parameter.kind == parameter.VAR_POSITIONAL
                          for parameter in signature.parameters.values())
            if varargs or len(positional) >= 2:
                callback(kind, identity)
            elif len(positional) == 1:
                callback(kind)
            else:
                callback()
        except (TypeError, ValueError):
            try:
                callback(kind, identity)
            except TypeError:
                callback()
        flush = self._flush
        if flush is not None and flush is not callback:
            try:
                flush()
            except Exception:
                pass

    # -- cycle receipts -------------------------------------------------
    def _cycle_receipt(self, parsed: dict[str, Any], *, status: str,
                       created: bool, duplicate: bool,
                       source_mismatch: bool = False) -> dict[str, Any]:
        cycle = parsed["cycle"]
        non_root = () if cycle is None else tuple(
            vertex for vertex in cycle if vertex != self.root)
        first = non_root[0] if len(non_root) == 2 else None
        second = non_root[1] if len(non_root) == 2 else None
        return {
            "status": status,
            "created": created,
            "duplicate": duplicate,
            "conflict": False,
            "kind": "cycle",
            "cycle": cycle,
            "cycle_id": parsed["cycle_id"],
            "i": first,
            "j": second,
            "cycle_index": None if first is None else cycle_index(
                first, second, self.nodes, self.root),
            "value": parsed["value"],
            "source_digest": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "payload_digest": None if parsed["payload_digest"] is None
            else parsed["payload_digest"].hex(),
            "hyperedge_id": parsed["hyperedge_id_hex"],
            "leaf_id": None if cycle is None else cycle_leaf_id(
                cycle, self.nodes, self.root).hex(),
            "pair_refs": parsed["pair_refs"],
            "heads": parsed["heads"],
            "members": parsed["members"],
            "source_mismatch": source_mismatch,
            "full_cycle_count": self.full_cycle_count,
            "canonical_merge": False,
            "generation": self.generation,
        }

    def _build_cycle_members(self, cycle: tuple[int, ...], edge_id: bytes,
                             cycle_value: int,
                             payload_digest: bytes
                             ) -> list[tuple[int, bytes]]:
        members: list[tuple[int, bytes]] = [(MEMBER_CYCLE, edge_id)]
        for edge in cycle_edges(cycle, self.nodes, self.root):
            members.append((MEMBER_PAIR, _bytes32(
                pair_reference(*edge), field="pair reference")))
        for vertex in cycle:
            members.append((MEMBER_HEAD, _bytes32(vertex, field="vertex")))
        members.append((MEMBER_VALUE, _bytes32(
            cycle_value, signed=True, field="cycle value")))
        members.append((MEMBER_PAYLOAD, payload_digest))
        return members

    def _add_cycle_bulk(self, first: int, second: int, cycle_value: int,
                        payload_digest: bytes, source_bytes: bytes
                        ) -> str:
        """Fast idempotent append used by :meth:`add_all_cycles`.

        It deliberately skips the per-cycle receipt/read-back parse.  The
        full-space audit verifies every persisted record afterwards.
        """
        canonical = tuple(sorted((self.root, first, second)))
        edge_id = cycle_hyperedge_id(canonical, self.nodes, self.root)
        relation_id = cycle_relation_id(canonical, self.nodes, self.root)
        existing = self._get_hyperedge(edge_id)
        if existing is not None:
            parsed = self._parse_view(existing)
            if parsed["kind"] != "cycle" or parsed["cycle"] != canonical:
                raise FullCycleConflictError(
                    "cycle id already names a non-cycle hyperedge",
                    code="cycle_hyperedge_conflict",
                    details={"cycle": ":".join(str(v) for v in canonical)})
            if parsed["value"] != cycle_value \
                    or parsed["payload_digest"] != payload_digest:
                raise FullCycleConflictError(
                    "cycle %s already names different immutable content"
                    % parsed["cycle_id"], code="cycle_conflict",
                    details={"cycle": parsed["cycle_id"]})
            return "duplicate"
        members = self._build_cycle_members(
            canonical, edge_id, cycle_value, payload_digest)
        self._append_edge(
            edge_id=edge_id, kind="cycle", members=members,
            source=source_bytes, subject=edge_id, predicate=CYCLE_PREDICATE,
            object_id=payload_digest, relation_id=relation_id,
            check_capacity=False, readback=False, commit=False)
        return "appended"

    # -- public cycle operations ---------------------------------------
    def add_cycle(self, i: Any = None, j: Any = None,
                  semantic_payload: Any = None, source: Any = None, *,
                  cycle: Any = None, value: Any = None, payload: Any = None,
                  source_digest: Any = None) -> dict[str, Any]:
        """Attach one root cycle ``(root, i, j)``.

        The canonical cycle id is deterministic and mirror-safe.  Repeating the
        same cycle with the same value and payload is idempotent.  A different
        value or payload is a conflict and is rejected.  A different provenance
        is reported as a duplicate with ``source_mismatch=True``; the original
        source is never overwritten.
        """
        if cycle is None and i is not None and not isinstance(i, int):
            # Accept both the documented (i, j, payload, source) form and the
            # cycle-first convenience form (cycle, payload, source).
            if j is None:
                cycle = i
            elif semantic_payload is None:
                cycle = i
                semantic_payload = j
            else:
                cycle = i
                semantic_payload = j
                if source is None:
                    source = semantic_payload
            i = None
            j = None
        if cycle is not None:
            canonical = normalize_cycle(cycle, self.nodes, self.root)
        else:
            if i is None or j is None:
                raise FullCycleHypergraphError(
                    "add_cycle needs i/j, a cycle tuple or a cycle id",
                    code="invalid_cycle")
            first, second = _validate_root_pair(i, j, self.nodes, self.root)
            canonical = tuple(sorted((self.root, first, second)))
        if canonical[0] != self.root and self.root not in canonical:
            raise FullCycleHypergraphError(
                "a full cycle must contain the configured root",
                code="invalid_cycle", details={"cycle": canonical,
                                               "root": self.root})
        if self.root not in canonical:
            raise FullCycleHypergraphError(
                "a full cycle must contain the configured root",
                code="invalid_cycle", details={"cycle": canonical,
                                               "root": self.root})
        non_root = tuple(vertex for vertex in canonical if vertex != self.root)
        if len(non_root) != 2:
            raise FullCycleHypergraphError(
                "a root cycle needs exactly two non-root heads",
                code="invalid_cycle", details={"cycle": canonical})
        if payload is not None:
            semantic_payload = payload
        elif value is not None and semantic_payload is None:
            semantic_payload = value
        if semantic_payload is None:
            semantic_payload = 0
        if source is None and source_digest is not None:
            source = source_digest
        source_bytes = _source_digest(source)
        cycle_value, _canonical_payload, payload_digest = \
            _payload_value_and_digest(semantic_payload)
        edge_id = cycle_hyperedge_id(canonical, self.nodes, self.root)
        relation_id = cycle_relation_id(canonical, self.nodes, self.root)
        existing = self._get_hyperedge(edge_id)
        if existing is not None:
            parsed = self._parse_view(existing)
            if parsed["kind"] != "cycle" or parsed["cycle"] != canonical:
                raise FullCycleConflictError(
                    "cycle id already names a non-cycle hyperedge",
                    code="cycle_hyperedge_conflict",
                    details={"cycle": ":".join(str(v) for v in canonical)})
            differences = []
            if parsed["value"] != cycle_value:
                differences.append("value")
            if parsed["payload_digest"] != payload_digest:
                differences.append("payload")
            if differences:
                raise FullCycleConflictError(
                    "cycle %s already names different immutable content"
                    % parsed["cycle_id"], code="cycle_conflict",
                    details={"cycle": parsed["cycle_id"],
                             "fields": differences})
            return self._cycle_receipt(
                parsed, status="duplicate", created=False, duplicate=True,
                source_mismatch=parsed["source_digest"] != source_bytes)
        members = self._build_cycle_members(
            canonical, edge_id, cycle_value, payload_digest)
        self._append_edge(
            edge_id=edge_id, kind="cycle", members=members,
            source=source_bytes, subject=edge_id, predicate=CYCLE_PREDICATE,
            object_id=payload_digest, relation_id=relation_id)
        view = self._get_hyperedge(edge_id)
        if view is None:
            raise FullCycleCorruptionError(
                "cycle hyperedge was not persisted",
                code="missing_cycle_hyperedge",
                details={"cycle": ":".join(str(v) for v in canonical)})
        parsed = self._parse_view(view)
        if parsed["kind"] != "cycle" or parsed["cycle"] != canonical \
                or parsed["value"] != cycle_value \
                or parsed["payload_digest"] != payload_digest \
                or parsed["source_digest"] != source_bytes:
            raise FullCycleCorruptionError(
                "persisted cycle hyperedge does not match the receipt",
                code="cycle_readback_mismatch",
                details={"cycle": ":".join(str(v) for v in canonical)})
        return self._cycle_receipt(parsed, status="appended", created=True,
                                   duplicate=False)

    add = add_cycle
    attach_cycle = add_cycle

    def add_all_cycles(self, source: Any = None, *,
                       payload_factory: Any = None,
                       limit: int | None = None,
                       source_digest: Any = None,
                       stop_on_error: bool = True) -> dict[str, Any]:
        """Attach the complete root cycle space (32,385 by default).

        ``limit`` bounds the work; it defaults to the full combinatorial count,
        never to 256.  ``payload_factory`` may be callable with either
        ``(index, i, j)``, ``(i, j)``, ``(index,)`` or ``()``.
        """
        if source is None and source_digest is not None:
            source = source_digest
        if limit is None:
            requested = self.full_cycle_count
        else:
            requested = _check_bounded_int(limit, "limit", 0,
                                           self.full_cycle_count)
        source_bytes = _source_digest(source)
        # Reserve the bounded batch before writing anything.  Existing cycles
        # are excluded so a repeated idempotent attach does not demand a
        # second full table.
        existing_scan = self._scan_cycles(validate=False)
        already_attached = min(existing_scan["attached_cycle_count"],
                               requested)
        needed = requested - already_attached
        if needed:
            self._check_capacity(
                hyperedges=needed, relations=needed, members=needed * 9)
        added = duplicates = errors = 0
        problems: list[dict[str, Any]] = []
        for index in range(requested):
            first, second = pair_from_index(index, self.nodes, self.root)
            if payload_factory is None:
                semantic_payload: Any = {"value": index}
            elif callable(payload_factory):
                semantic_payload = None
                for args in ((index, first, second), (first, second),
                             (index,), ()):
                    try:
                        semantic_payload = payload_factory(*args)
                        break
                    except TypeError:
                        continue
            else:
                semantic_payload = payload_factory
            try:
                cycle_value, _canonical_payload, payload_digest = \
                    _payload_value_and_digest(semantic_payload)
                status = self._add_cycle_bulk(
                    first, second, cycle_value, payload_digest, source_bytes)
            except FullCycleHypergraphError as exc:
                errors += 1
                if len(problems) < MAX_PROBLEMS:
                    problems.append(exc.to_dict())
                if stop_on_error:
                    raise
                continue
            if status == "appended":
                added += 1
            else:
                duplicates += 1
        if added:
            self._maybe_commit("cycles", "full-cycle-space")
        scan = self._scan_cycles(validate=False)
        attached = scan["attached_cycle_count"]
        return {
            "schema": SCHEMA,
            "status": "complete" if attached == self.full_cycle_count
            and errors == 0 else "partial",
            "requested": requested,
            "added": added,
            "duplicates": duplicates,
            "errors": errors,
            "problems": problems,
            "attached_cycle_count": attached,
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "remaining_cycle_count": self.full_cycle_count - attached,
            "complete": attached == self.full_cycle_count,
            "truncated": requested < self.full_cycle_count,
            "bounded": True,
            "canonical_merge": False,
            "generation": self.generation,
        }

    add_all = add_all_cycles
    attach_all = add_all_cycles
    attach_full_cycle_space = add_all_cycles
    populate = add_all_cycles

    # -- concepts -------------------------------------------------------
    def _resolve_concept(self, concept: Any, *, required: bool = True
                         ) -> bytes | None:
        if isinstance(concept, Mapping):
            concept = concept.get("concept", concept.get("name",
                                   concept.get("term", concept.get("id"))))
        if concept is None:
            if required:
                raise FullCycleHypergraphError(
                    "concept reference is required", code="invalid_concept")
            return None
        try:
            leaf_id = concept_leaf_id(concept)
        except FullCycleHypergraphError:
            if required:
                raise
            return None
        if required and self._find_leaf(leaf_id) is None:
            raise FullCycleNotFound(
                "concept %s does not exist" % leaf_id.hex(),
                code="concept_not_found",
                details={"concept_id": leaf_id.hex()})
        return leaf_id

    def add_concept(self, name: Any = None, payload: Any = None,
                    source: Any = None, *, parent: Any = None,
                    concept: Any = None, source_digest: Any = None,
                    payload_digest: Any = None) -> dict[str, Any]:
        """Create a resident concept leaf and its typed concept hyperedge."""
        if concept is not None and name is None:
            name = concept
        if name is None:
            raise FullCycleHypergraphError(
                "concept name is required", code="invalid_concept")
        leaf_id = concept_leaf_id(name)
        if source is None and source_digest is not None:
            source = source_digest
        source_bytes = _source_digest(source)
        if parent is not None:
            try:
                if concept_leaf_id(parent) == leaf_id:
                    raise FullCycleParentCycleError(
                        "a concept cannot be its own parent",
                        code="self_parent",
                        details={"concept_id": leaf_id.hex()})
            except FullCycleHypergraphError as exc:
                if isinstance(exc, FullCycleParentCycleError):
                    raise
        parent_id = None if parent is None else self._resolve_concept(
            parent, required=True)
        if parent_id == leaf_id:
            raise FullCycleParentCycleError(
                "a concept cannot be its own parent", code="self_parent",
                details={"concept_id": leaf_id.hex()})
        if parent_id is not None:
            ancestors = self._concept_ancestors(parent_id)
            if leaf_id in ancestors:
                raise FullCycleParentCycleError(
                    "concept parent link would create a cycle",
                    code="parent_cycle",
                    details={"concept_id": leaf_id.hex()})
            if len(ancestors) >= MAX_PARENT_DEPTH:
                raise FullCycleParentCycleError(
                    "concept parent depth exceeds %d" % MAX_PARENT_DEPTH,
                    code="parent_depth_exceeded",
                    details={"concept_id": leaf_id.hex()})
        if payload_digest is None:
            if payload is None:
                canonical_payload = b""
            elif isinstance(payload, (bytes, bytearray, memoryview)):
                canonical_payload = bytes(payload)
            elif isinstance(payload, str):
                canonical_payload = payload.encode("utf-8")
            else:
                canonical_payload = _canonical(payload)
            digest = hashlib.sha256(
                b"NLMR-FULL-CONCEPT-PAYLOAD\0" + canonical_payload).digest()
        else:
            digest = _bytes32(payload_digest, field="payload_digest")
            canonical_payload = b""
        existing_leaf = self._find_leaf(leaf_id)
        if existing_leaf is not None:
            existing_digest = _field_any(existing_leaf,
                                         ("payload_digest", "digest"))
            if isinstance(existing_digest, str):
                try:
                    existing_digest = bytes.fromhex(existing_digest)
                except ValueError:
                    existing_digest = None
            if existing_digest is not None and bytes(existing_digest) != digest:
                raise FullCycleConflictError(
                    "concept %s already names a different payload"
                    % leaf_id.hex(), code="concept_conflict",
                    details={"concept_id": leaf_id.hex()})
            existing_parent = self._leaf_parent_id(existing_leaf)
            if parent_id is not None and existing_parent is not None \
                    and existing_parent != parent_id:
                raise FullCycleConflictError(
                    "concept %s already has a different parent"
                    % leaf_id.hex(), code="concept_parent_conflict",
                    details={"concept_id": leaf_id.hex()})
        edge_id = concept_hyperedge_id(leaf_id)
        members: list[tuple[int, bytes]] = [
            (MEMBER_CONCEPT, _pad_leaf_id(leaf_id)),
        ]
        if parent_id is not None:
            members.append((MEMBER_CONCEPT, _pad_leaf_id(parent_id)))
        existing = self._get_hyperedge(edge_id)
        if existing is not None:
            parsed = self._parse_view(existing)
            actual = sorted(parsed["member_pairs"])
            if actual != sorted(members):
                raise FullCycleConflictError(
                    "concept edge already names different members",
                    code="concept_edge_conflict",
                    details={"concept_id": leaf_id.hex()})
            if existing_leaf is None:
                self._check_capacity(leaf=True)
                self._insert_concept_leaf(
                    leaf_id, digest, canonical_payload, parent_id)
            receipt = self._concept_receipt(
                leaf_id, edge_id, parent_id, parsed,
                status="duplicate", created=False, duplicate=True,
                source_mismatch=parsed["source_digest"] != source_bytes)
            return receipt
        self._check_capacity(hyperedges=1, relations=1,
                             members=len(members), leaf=existing_leaf is None)
        if existing_leaf is None:
            try:
                self._insert_concept_leaf(
                    leaf_id, digest, canonical_payload, parent_id)
            except FullCycleHypergraphError:
                raise
            except Exception as exc:
                raise FullCycleBoundsError(
                    "concept leaf could not be appended: %s" % exc,
                    code="leaf_capacity",
                    details={"concept_id": leaf_id.hex()}) from exc
        relation_id = hashlib.sha256(
            b"NLMR-FULL-CONCEPT-REL\0" + edge_id).digest()
        self._append_edge(
            edge_id=edge_id, kind="concept", members=members,
            source=source_bytes, subject=_pad_leaf_id(leaf_id),
            predicate=CONCEPT_PREDICATE,
            object_id=_pad_leaf_id(parent_id or leaf_id),
            relation_id=relation_id)
        view = self._get_hyperedge(edge_id)
        if view is None:
            raise FullCycleCorruptionError(
                "concept hyperedge was not persisted",
                code="missing_concept_hyperedge",
                details={"concept_id": leaf_id.hex()})
        parsed = self._parse_view(view)
        if parsed["source_digest"] != source_bytes \
                or sorted(parsed["member_pairs"]) != sorted(members):
            raise FullCycleCorruptionError(
                "persisted concept edge does not match the receipt",
                code="concept_readback_mismatch",
                details={"concept_id": leaf_id.hex()})
        return self._concept_receipt(
            leaf_id, edge_id, parent_id, parsed, status="appended",
            created=True, duplicate=False)

    def _concept_receipt(self, leaf_id: bytes, edge_id: bytes,
                         parent_id: bytes | None, parsed: dict[str, Any], *,
                         status: str, created: bool, duplicate: bool,
                         source_mismatch: bool = False) -> dict[str, Any]:
        leaf = self._find_leaf(leaf_id)
        digest = _field_any(leaf, ("payload_digest", "digest"))
        if isinstance(digest, (bytes, bytearray, memoryview)):
            digest = bytes(digest).hex()
        return {
            "status": status,
            "created": created,
            "duplicate": duplicate,
            "conflict": False,
            "kind": "concept",
            "concept_id": leaf_id.hex(),
            "leaf_id": leaf_id.hex(),
            "payload_digest": digest,
            "parent_concept_id": None if parent_id is None
            else parent_id.hex(),
            "hyperedge_id": edge_id.hex(),
            "source_digest": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "members": parsed["members"],
            "source_mismatch": source_mismatch,
            "full_cycle_count": self.full_cycle_count,
            "canonical_merge": False,
            "generation": self.generation,
        }

    # -- member normalization for link/compose -------------------------
    def _normalize_member_spec(self, spec: Any, *, require_existing: bool = True
                               ) -> dict[str, Any]:
        kind = None
        reference = spec
        if isinstance(spec, Mapping):
            kind = spec.get("kind")
            reference = spec.get("ref", spec.get(
                "cycle", spec.get("concept", spec.get(
                    "id", spec.get("hyperedge_id",
                                   spec.get("head", spec.get("vertex")))))))
            if kind is None:
                if "cycle" in spec:
                    kind = "cycle"
                elif "concept" in spec:
                    kind = "concept"
                elif "parent_id" in spec or "parent" in spec:
                    kind = "parent"
                    reference = spec.get("parent_id", spec.get("parent"))
                elif "hyperedge_id" in spec:
                    kind = "hyperedge"
        if kind == "pair":
            if isinstance(reference, (tuple, list)) and len(reference) == 2:
                first, second = reference
                return {
                    "kind": "pair", "kind_code": MEMBER_PAIR,
                    "id": _bytes32(pair_reference(first, second),
                                   field="pair reference"),
                    "pair": (min(int(first), int(second)),
                             max(int(first), int(second))),
                }
            raise FullCycleHypergraphError(
                "pair member needs two heads", code="invalid_member")
        if kind in ("head", "vertex") and not isinstance(
                spec, (tuple, list)):
            head = _check_bounded_int(reference, "head", 0, self.nodes - 1)
            return {
                "kind": "head", "kind_code": MEMBER_HEAD,
                "id": _bytes32(head, field="head"), "head": head,
            }
        if isinstance(spec, (tuple, list)) and spec \
                and isinstance(spec[0], str):
            kind = spec[0].lower()
            if kind in ("cycle", "triangle"):
                reference = tuple(spec[1:])
                kind = "cycle"
            elif kind == "pair":
                if len(spec) != 3:
                    raise FullCycleHypergraphError(
                        "pair member needs (pair, i, j)",
                        code="invalid_member")
                return {
                    "kind": "pair",
                    "kind_code": MEMBER_PAIR,
                    "id": _bytes32(pair_reference(spec[1], spec[2]),
                                   field="pair reference"),
                    "pair": (min(int(spec[1]), int(spec[2])),
                             max(int(spec[1]), int(spec[2]))),
                }
            elif kind in ("head", "vertex"):
                if len(spec) != 2:
                    raise FullCycleHypergraphError(
                        "head member needs (head, value)",
                        code="invalid_member")
                head = _check_bounded_int(spec[1], "head", 0, self.nodes - 1)
                return {
                    "kind": "head", "kind_code": MEMBER_HEAD,
                    "id": _bytes32(head, field="head"), "head": head,
                }
            elif kind in ("concept", "term"):
                reference = spec[1] if len(spec) > 1 else None
                kind = "concept"
            elif kind in ("parent", "hyperedge"):
                reference = spec[1] if len(spec) > 1 else None
                kind = "parent" if kind == "parent" else "hyperedge"
            else:
                raise FullCycleHypergraphError(
                    "unknown member kind %r" % spec[0], code="invalid_member")
        if kind is None and isinstance(reference, str):
            text = reference.strip()
            if len(text) == VALUE_BYTES * 2:
                try:
                    raw = bytes.fromhex(text)
                except ValueError:
                    raw = None
                if raw is not None and self._get_hyperedge(raw) is not None:
                    return {
                        "kind": "parent", "kind_code": MEMBER_PARENT,
                        "id": raw, "hyperedge_id": raw,
                    }
            if len(text) == LEAF_ID_BYTES * 2 and self._find_leaf(
                    bytes.fromhex(text)) is not None:
                concept_id = bytes.fromhex(text)
                return {
                    "kind": "concept", "kind_code": MEMBER_CONCEPT,
                    "id": _pad_leaf_id(concept_id),
                    "concept_id": concept_id, "leaf_id": concept_id,
                }
        if kind in (None, "cycle", "triangle"):
            try:
                cycle = normalize_cycle(reference, self.nodes, self.root)
            except FullCycleHypergraphError:
                if kind is not None:
                    raise
                cycle = None
            if cycle is not None:
                view = self._find_cycle_view(cycle)
                if view is None and require_existing:
                    raise FullCycleNotFound(
                        "cycle %s does not exist" % cycle_id(
                            cycle, self.nodes, self.root),
                        code="cycle_not_found",
                        details={"cycle": ":".join(str(v) for v in cycle)})
                identifier = cycle_hyperedge_id(cycle, self.nodes, self.root)
                return {
                    "kind": "cycle", "kind_code": MEMBER_CYCLE,
                    "id": identifier, "cycle": cycle,
                    "cycle_id": ":".join(str(v) for v in cycle),
                    "exists": view is not None,
                }
            if kind is not None:
                raise FullCycleHypergraphError(
                    "invalid cycle member", code="invalid_member")
        if kind in (None, "concept", "term"):
            if reference is None:
                raise FullCycleHypergraphError(
                    "concept member is required", code="invalid_member")
            concept_id = self._resolve_concept(
                reference, required=require_existing)
            if concept_id is None:
                raise FullCycleHypergraphError(
                    "concept member is required", code="invalid_member")
            return {
                "kind": "concept", "kind_code": MEMBER_CONCEPT,
                "id": _pad_leaf_id(concept_id), "concept_id": concept_id,
                "leaf_id": concept_id,
            }
        if kind in ("parent", "hyperedge"):
            if reference is None:
                raise FullCycleHypergraphError(
                    "parent hyperedge reference is required",
                    code="invalid_member")
            identifier = _bytes32(reference, field="parent")
            view = self._get_hyperedge(identifier)
            if view is None and require_existing:
                raise FullCycleNotFound(
                    "parent hyperedge %s does not exist" % identifier.hex(),
                    code="parent_not_found",
                    details={"hyperedge_id": identifier.hex()})
            code = MEMBER_PARENT if kind == "parent" else MEMBER_HYPEREDGE
            return {
                "kind": kind, "kind_code": code, "id": identifier,
                "hyperedge_id": identifier,
            }
        raise FullCycleHypergraphError(
            "member kind could not be resolved", code="invalid_member",
            details={"member": repr(spec)})

    def _normalize_members(self, members: Any, *,
                           require_existing: bool = True
                           ) -> tuple[list[dict[str, Any]], int]:
        if members is None:
            return [], 0
        if isinstance(members, (str, bytes, bytearray, Mapping)):
            items = [members]
        elif isinstance(members, (tuple, list)) and members \
                and isinstance(members[0], int) \
                and len(members) in (2, 3):
            items = [members]
        else:
            try:
                items = list(members)
            except TypeError:
                items = [members]
        resolved: list[dict[str, Any]] = []
        seen: set[tuple[int, bytes]] = set()
        duplicates = 0
        for item in items:
            member = self._normalize_member_spec(
                item, require_existing=require_existing)
            key = (member["kind_code"], bytes(member["id"]))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            resolved.append(member)
        return resolved, duplicates

    # -- link -----------------------------------------------------------
    def _resolve_link_owner(self, parent: Any, concept: Any = None
                            ) -> dict[str, Any]:
        if concept is not None and parent is None:
            parent = concept
        if parent is None:
            raise FullCycleHypergraphError(
                "link needs a concept or parent hyperedge",
                code="invalid_parent")
        if isinstance(parent, Mapping):
            if "concept" in parent:
                parent = parent["concept"]
            elif "hyperedge_id" in parent:
                parent = parent["hyperedge_id"]
            elif "id" in parent:
                parent = parent["id"]
        if isinstance(parent, str):
            text = parent.strip()
            if len(text) in (VALUE_BYTES * 2, LEAF_ID_BYTES * 2):
                try:
                    raw = bytes.fromhex(text)
                except ValueError:
                    raw = None
                if raw is not None and len(raw) == VALUE_BYTES:
                    view = self._get_hyperedge(raw)
                    if view is not None:
                        parsed = self._parse_view(view)
                        if parsed["kind"] == "concept" and parsed["concept_ids"]:
                            concept_id = parsed["concept_ids"][0]
                            return {
                                "kind": "concept", "concept_id": concept_id,
                                "owner_id": _pad_leaf_id(concept_id),
                                "hyperedge_id": concept_hyperedge_id(
                                    concept_id),
                            }
                        if parsed["kind"] == "cycle":
                            raise FullCycleDenied(
                                "a cycle cannot own concept/parent links; "
                                "use a concept or parent hyperedge",
                                code="cycle_parent_denied")
                        return {
                            "kind": "hyperedge", "hierarchy_id": raw,
                            "owner_id": raw, "hyperedge_id": raw,
                        }
            # Fall through to concept-name lookup for ordinary text.
        if isinstance(parent, (tuple, list)) and parent \
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in parent):
            raise FullCycleDenied(
                "a cycle cannot own concept/parent links; use a concept as "
                "the parent", code="cycle_parent_denied")
        try:
            concept_id = self._resolve_concept(parent, required=True)
        except FullCycleNotFound:
            raw = _bytes32(parent, field="parent")
            view = self._get_hyperedge(raw)
            if view is None:
                raise
            return {
                "kind": "hyperedge", "hierarchy_id": raw,
                "owner_id": raw, "hyperedge_id": raw,
            }
        return {
            "kind": "concept", "concept_id": concept_id,
            "owner_id": _pad_leaf_id(concept_id),
            "hyperedge_id": concept_hyperedge_id(concept_id),
        }

    def _parent_reaches(self, start: bytes, target: bytes,
                        max_depth: int = MAX_PARENT_DEPTH) -> bool:
        if start == target:
            return True
        current = bytes(start)
        seen: set[bytes] = set()
        for _ in range(max_depth + 1):
            if current in seen:
                return True
            seen.add(current)
            view = self._get_hyperedge(current)
            if view is None:
                return False
            parsed = self._parse_view(view)
            parents = parsed["parent_ids"]
            if not parents:
                return False
            current = parents[0]
            if current == target:
                return True
        return True

    def link(self, parent: Any = None, members: Any = None,
             source: Any = None, *, concept: Any = None, cycles: Any = None,
             parent_edge: Any = None, source_digest: Any = None,
             require_existing: bool = True) -> dict[str, Any]:
        """Link cycles, concepts or parent edges into a concept/parent edge."""
        owner = self._resolve_link_owner(parent, concept)
        if source is None and source_digest is not None:
            source = source_digest
        source_bytes = _source_digest(source)
        if members is None:
            members = cycles
        resolved, duplicates = self._normalize_members(
            members, require_existing=require_existing)
        if not resolved:
            raise FullCycleHypergraphError(
                "link needs at least one member", code="invalid_member")
        parent_id = None
        if parent_edge is not None:
            parent_id = _bytes32(parent_edge, field="parent_edge")
            view = self._get_hyperedge(parent_id)
            if view is None and require_existing:
                raise FullCycleNotFound(
                    "parent edge %s does not exist" % parent_id.hex(),
                    code="parent_not_found",
                    details={"parent_id": parent_id.hex()})
            if view is not None and self._parse_view(view)["kind"] == "cycle":
                raise FullCycleDenied(
                    "a cycle cannot be used as a parent hyperedge",
                    code="cycle_parent_denied",
                    details={"parent_id": parent_id.hex()})
            if parent_id == owner["hyperedge_id"]:
                raise FullCycleParentCycleError(
                    "an edge cannot be its own parent", code="self_parent",
                    details={"hyperedge_id": parent_id.hex()})
            if self._parent_reaches(parent_id, owner["hyperedge_id"]):
                raise FullCycleParentCycleError(
                    "parent link would create a parent cycle",
                    code="parent_cycle",
                    details={"parent_id": parent_id.hex(),
                             "owner_id": owner["hyperedge_id"].hex()})
        edge_members: list[tuple[int, bytes]] = []
        if owner["kind"] == "concept":
            edge_members.append((MEMBER_CONCEPT,
                                 _pad_leaf_id(owner["concept_id"])))
        else:
            edge_members.append((MEMBER_HYPEREDGE, owner["hyperedge_id"]))
        concept_members = []
        for member in resolved:
            edge_members.append((member["kind_code"], bytes(member["id"])))
            if member["kind"] == "concept":
                concept_members.append(member["concept_id"])
        if parent_id is not None:
            if any(code == MEMBER_PARENT and identifier == parent_id
                   for code, identifier in edge_members):
                raise FullCycleParentCycleError(
                    "a parent edge cannot also be its own child member",
                    code="self_parent",
                    details={"parent_id": parent_id.hex()})
            edge_members.append((MEMBER_PARENT, parent_id))
        edge_members, extra_duplicates = self._dedupe_member_pairs(
            edge_members)
        duplicates += extra_duplicates
        for concept_id in concept_members:
            if concept_id == owner.get("concept_id"):
                raise FullCycleParentCycleError(
                    "a concept cannot contain itself", code="self_parent",
                    details={"concept_id": concept_id.hex()})
            if owner["kind"] == "concept" and concept_id in \
                    self._concept_ancestors(owner["concept_id"]):
                # Member is an ancestor of the owner; membership is harmless
                # but a concept parent chain would become cyclic if promoted.
                raise FullCycleParentCycleError(
                    "concept membership would close a parent chain",
                    code="parent_cycle",
                    details={"concept_id": concept_id.hex()})
        if owner["kind"] == "hyperedge":
            for member in resolved:
                if member["kind"] not in ("parent", "hyperedge"):
                    continue
                identifier = bytes(member["id"])
                if identifier == owner["hyperedge_id"]:
                    raise FullCycleParentCycleError(
                        "a parent edge cannot contain itself",
                        code="self_parent",
                        details={"hyperedge_id": identifier.hex()})
                if self._parent_reaches(identifier, owner["hyperedge_id"]):
                    raise FullCycleParentCycleError(
                        "parent membership would close a parent chain",
                        code="parent_cycle",
                        details={"hyperedge_id": identifier.hex()})
        member_pairs = self._member_pairs_for_edge(edge_members)
        edge_id = _edge_identity(
            "link", owner=owner["owner_id"], parent=parent_id,
            members=member_pairs)
        existing = self._get_hyperedge(edge_id)
        if existing is not None:
            parsed = self._parse_view(existing)
            if sorted(parsed["member_pairs"]) != sorted(edge_members):
                raise FullCycleConflictError(
                    "link id already names different members",
                    code="link_conflict",
                    details={"hyperedge_id": edge_id.hex()})
            return self._link_receipt(
                owner, edge_id, parent_id, resolved, parsed, duplicates,
                status="duplicate", created=False, duplicate=True,
                source_mismatch=parsed["source_digest"] != source_bytes)
        relation_id = hashlib.sha256(
            b"NLMR-FULL-LINK-REL\0" + edge_id).digest()
        first_target = bytes(resolved[0]["id"])
        incidence = [bytes(member["id"]) for member in resolved
                     if member["kind_code"] in MEMBERSHIP_KINDS]
        if parent_id is not None:
            incidence.append(parent_id)
        self._append_edge(
            edge_id=edge_id, kind="link", members=edge_members,
            source=source_bytes, subject=owner["owner_id"],
            predicate=LINK_PREDICATE, object_id=first_target,
            relation_id=relation_id, incidence_targets=incidence)
        view = self._get_hyperedge(edge_id)
        if view is None:
            raise FullCycleCorruptionError(
                "link hyperedge was not persisted",
                code="missing_link_hyperedge",
                details={"hyperedge_id": edge_id.hex()})
        parsed = self._parse_view(view)
        if parsed["source_digest"] != source_bytes \
                or sorted(parsed["member_pairs"]) != sorted(edge_members):
            raise FullCycleCorruptionError(
                "persisted link does not match the receipt",
                code="link_readback_mismatch",
                details={"hyperedge_id": edge_id.hex()})
        return self._link_receipt(
            owner, edge_id, parent_id, resolved, parsed, duplicates,
            status="appended", created=True, duplicate=False)

    @staticmethod
    def _member_pairs_for_edge(members: Iterable[tuple[int, bytes]]
                               ) -> list[tuple[int, bytes]]:
        return [(int(code), bytes(identifier)) for code, identifier in members]

    @staticmethod
    def _dedupe_member_pairs(members: Iterable[tuple[int, bytes]]
                             ) -> tuple[list[tuple[int, bytes]], int]:
        out: list[tuple[int, bytes]] = []
        seen: set[tuple[int, bytes]] = set()
        duplicates = 0
        for code, identifier in members:
            key = (int(code), bytes(identifier))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            out.append(key)
        return out, duplicates

    def _link_receipt(self, owner: dict[str, Any], edge_id: bytes,
                      parent_id: bytes | None, resolved: list[dict[str, Any]],
                      parsed: dict[str, Any], duplicates: int, *,
                      status: str, created: bool, duplicate: bool,
                      source_mismatch: bool = False) -> dict[str, Any]:
        cycle_ids = [member["cycle_id"] for member in resolved
                     if member["kind"] == "cycle"]
        concept_ids = [member["concept_id"].hex() for member in resolved
                       if member["kind"] == "concept"]
        return {
            "status": status,
            "created": created,
            "duplicate": duplicate,
            "conflict": False,
            "kind": "link",
            "owner_kind": owner["kind"],
            "owner_id": (owner.get("concept_id").hex()
                         if owner["kind"] == "concept"
                         else owner["hyperedge_id"].hex()),
            "concept_id": (owner.get("concept_id").hex()
                           if owner["kind"] == "concept" else None),
            "hyperedge_id": edge_id.hex(),
            "parent_id": None if parent_id is None else parent_id.hex(),
            "cycle_ids": cycle_ids,
            "cycles": [member["cycle"] for member in resolved
                       if member["kind"] == "cycle"],
            "concept_ids": concept_ids,
            "duplicates_removed": duplicates,
            "source_digest": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "members": parsed["members"],
            "source_mismatch": source_mismatch,
            "full_cycle_count": self.full_cycle_count,
            "canonical_merge": False,
            "generation": self.generation,
        }

    # -- compose --------------------------------------------------------
    def compose(self, *cycle_args: Any, source: Any = None,
                concept: Any = None, parent: Any = None,
                cycles: Any = None, require_shared: bool = False,
                source_digest: Any = None) -> Composition:
        """Compose existing cycles into a recursive parent hyperedge.

        The source cycles are untouched.  The returned object is a parent-edge
        receipt; it is not a canonical merged cycle.  A repeated composition
        with identical members is idempotent.
        """
        if source is None and cycle_args \
                and isinstance(cycle_args[-1],
                               (str, bytes, bytearray, memoryview)):
            possible_source = cycle_args[-1]
            try:
                normalize_cycle(possible_source, self.nodes, self.root)
                is_cycle = True
            except FullCycleHypergraphError:
                is_cycle = False
            if not is_cycle:
                source = possible_source
                cycle_args = cycle_args[:-1]
        if source is None and source_digest is not None:
            source = source_digest
        source_bytes = _source_digest(source)
        if cycles is not None:
            if isinstance(cycles, (tuple, list)) and cycles \
                    and all(isinstance(value, int)
                            and not isinstance(value, bool)
                            for value in cycles):
                raw_items = [cycles]
            else:
                raw_items = list(cycles) if not isinstance(
                    cycles, (str, bytes, Mapping)) else [cycles]
        elif len(cycle_args) == 1 and isinstance(
                cycle_args[0], (list, tuple)) \
                and not all(isinstance(value, int)
                            and not isinstance(value, bool)
                            for value in cycle_args[0]):
            raw_items = list(cycle_args[0])
        elif len(cycle_args) == 2 and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in cycle_args):
            raw_items = [cycle_args]
        else:
            raw_items = list(cycle_args)
        if not raw_items:
            raise FullCycleHypergraphError(
                "compose needs at least one cycle", code="invalid_compose")
        resolved_cycles: list[dict[str, Any]] = []
        seen: set[bytes] = set()
        duplicates = 0
        for item in raw_items:
            member = self._normalize_member_spec(item, require_existing=True)
            if member["kind"] != "cycle":
                raise FullCycleHypergraphError(
                    "compose members must be cycles", code="invalid_compose",
                    details={"kind": member["kind"]})
            identifier = bytes(member["id"])
            if identifier in seen:
                duplicates += 1
                continue
            seen.add(identifier)
            view = self._find_cycle_view(member["cycle"])
            parsed = self._parse_view(view) if view is not None else None
            if parsed is None:
                raise FullCycleNotFound(
                    "cycle %s does not exist" % member["cycle_id"],
                    code="cycle_not_found",
                    details={"cycle": member["cycle_id"]})
            resolved_cycles.append({
                "cycle": member["cycle"], "cycle_id": member["cycle_id"],
                "id": identifier, "value": parsed["value"] or 0,
                "pair_refs": parsed["pair_refs"],
            })
        if require_shared and len(resolved_cycles) > 1:
            shared = set(resolved_cycles[0]["pair_refs"])
            for item in resolved_cycles[1:]:
                shared &= set(item["pair_refs"])
            if not shared:
                raise FullCycleDenied(
                    "cycles do not share an incidence edge",
                    code="not_composable",
                    details={"cycles": [item["cycle_id"]
                                        for item in resolved_cycles]})
        concept_ids: list[bytes] = []
        if concept is not None:
            if isinstance(concept, (list, tuple)) and not isinstance(
                    concept, str):
                for item in concept:
                    resolved = self._resolve_concept(item, required=True)
                    if resolved not in concept_ids:
                        concept_ids.append(resolved)
            else:
                concept_ids.append(self._resolve_concept(
                    concept, required=True))
        parent_id = None
        if parent is not None:
            if isinstance(parent, Mapping) and "hyperedge_id" in parent:
                parent = parent["hyperedge_id"]
            parent_id = _bytes32(parent, field="parent")
            parent_view = self._get_hyperedge(parent_id)
            if parent_view is None:
                raise FullCycleNotFound(
                    "parent edge %s does not exist" % parent_id.hex(),
                    code="parent_not_found",
                    details={"parent_id": parent_id.hex()})
            if self._parse_view(parent_view)["kind"] == "cycle":
                raise FullCycleDenied(
                    "a cycle cannot be used as a parent hyperedge",
                    code="cycle_parent_denied",
                    details={"parent_id": parent_id.hex()})
        cycle_ids = [item["id"] for item in resolved_cycles]
        values = [int(item["value"]) for item in resolved_cycles]
        payload_digest = _compose_payload(
            cycle_ids, values, concept_ids, parent_id)
        if len(resolved_cycles) == 1 and concept is None and parent is None:
            item = resolved_cycles[0]
            return Composition({
                "status": "idempotent", "created": False,
                "duplicate": False, "conflict": False, "kind": "compose",
                "cycle": item["cycle"], "cycles": [item["cycle"]],
                "cycle_ids": [item["cycle_id"]],
                "hyperedge_id": item["id"].hex(),
                "parent_id": None, "value": item["value"],
                "shared": (), "members": [], "stored": False,
                "duplicates_removed": duplicates,
                "full_cycle_count": self.full_cycle_count,
                "canonical_merge": False,
                "source_digest": source_bytes.hex(),
            })
        edge_members: list[tuple[int, bytes]] = []
        if parent_id is not None:
            edge_members.append((MEMBER_PARENT, parent_id))
        edge_members.extend(
            (MEMBER_CYCLE, identifier) for identifier in cycle_ids)
        for concept_id in concept_ids:
            edge_members.append((MEMBER_CONCEPT, _pad_leaf_id(concept_id)))
        edge_members.append((MEMBER_VALUE, _bytes32(
            sum(values), signed=True, field="value")))
        edge_members.append((MEMBER_PAYLOAD, payload_digest))
        edge_members, extra_duplicates = self._dedupe_member_pairs(
            edge_members)
        duplicates += extra_duplicates
        member_pairs = self._member_pairs_for_edge(edge_members)
        edge_id = _edge_identity(
            "compose", owner=parent_id, parent=parent_id,
            members=member_pairs)
        existing = self._get_hyperedge(edge_id)
        if existing is not None:
            parsed = self._parse_view(existing)
            if sorted(parsed["member_pairs"]) != sorted(edge_members):
                raise FullCycleConflictError(
                    "composition id already names different members",
                    code="compose_conflict",
                    details={"hyperedge_id": edge_id.hex()})
            return self._composition_receipt(
                edge_id, parent_id, resolved_cycles, concept_ids, parsed,
                duplicates, status="duplicate", created=False, duplicate=True,
                source_mismatch=parsed["source_digest"] != source_bytes)
        relation_id = hashlib.sha256(
            b"NLMR-FULL-COMPOSE-REL\0" + edge_id).digest()
        subject = parent_id if parent_id is not None else edge_id
        incidence = list(cycle_ids) + concept_ids
        if parent_id is not None:
            incidence.append(parent_id)
        self._append_edge(
            edge_id=edge_id, kind="compose", members=edge_members,
            source=source_bytes, subject=subject,
            predicate=COMPOSE_PREDICATE, object_id=cycle_ids[0],
            relation_id=relation_id, incidence_targets=incidence)
        view = self._get_hyperedge(edge_id)
        if view is None:
            raise FullCycleCorruptionError(
                "composition hyperedge was not persisted",
                code="missing_compose_hyperedge",
                details={"hyperedge_id": edge_id.hex()})
        parsed = self._parse_view(view)
        if parsed["source_digest"] != source_bytes \
                or sorted(parsed["member_pairs"]) != sorted(edge_members):
            raise FullCycleCorruptionError(
                "persisted composition does not match the receipt",
                code="compose_readback_mismatch",
                details={"hyperedge_id": edge_id.hex()})
        return self._composition_receipt(
            edge_id, parent_id, resolved_cycles, concept_ids, parsed,
            duplicates, status="composed", created=True, duplicate=False)

    compose_cycles = compose

    def _composition_receipt(
            self, edge_id: bytes, parent_id: bytes | None,
            resolved_cycles: list[dict[str, Any]], concept_ids: list[bytes],
            parsed: dict[str, Any], duplicates: int, *, status: str,
            created: bool, duplicate: bool,
            source_mismatch: bool = False) -> Composition:
        shared: set[int] = set()
        if resolved_cycles:
            shared = set(resolved_cycles[0]["pair_refs"])
            for item in resolved_cycles[1:]:
                shared &= set(item["pair_refs"])
        values = [int(item["value"]) for item in resolved_cycles]
        return Composition({
            "status": status,
            "created": created,
            "duplicate": duplicate,
            "conflict": False,
            "kind": "compose",
            "cycle": tuple(sorted({
                vertex for item in resolved_cycles
                for vertex in item["cycle"]})),
            "cycles": [item["cycle"] for item in resolved_cycles],
            "cycle_ids": [item["cycle_id"] for item in resolved_cycles],
            "concept_ids": [concept_id.hex() for concept_id in concept_ids],
            "hyperedge_id": edge_id.hex(),
            "parent_id": None if parent_id is None else parent_id.hex(),
            "value": sum(values),
            "shared": tuple(sorted(shared)),
            "duplicates_removed": duplicates,
            "members": parsed["members"],
            "stored": True,
            "source_digest": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "source_mismatch": source_mismatch,
            "full_cycle_count": self.full_cycle_count,
            "canonical_merge": False,
            "generation": self.generation,
        })

    # -- members --------------------------------------------------------
    def _concept_edges_for_owner(self, concept_id: bytes,
                                 limit: int = DEFAULT_LINK_LIMIT
                                 ) -> tuple[list[bytes], bool]:
        owner = _pad_leaf_id(concept_id)
        edges, truncated = self._chain_edges(owner, "subject", limit)
        concept_edge = concept_hyperedge_id(concept_id)
        return [edge for edge in edges if edge != concept_edge], truncated

    def _owner_edges_for_edge(self, edge_id: bytes,
                              limit: int = DEFAULT_LINK_LIMIT
                              ) -> tuple[list[bytes], bool]:
        edges, truncated = self._chain_edges(edge_id, "subject", limit)
        return [edge for edge in edges if edge != edge_id], truncated

    def _chain_edges(self, key: bytes, role: str,
                     limit: int = DEFAULT_LINK_LIMIT
                     ) -> tuple[list[bytes], bool]:
        limit = _check_bounded_int(limit, "limit", 1, MAX_NEIGHBOR_LIMIT)
        chain = _host_attr(self.relations, "chain_walk", None)
        if callable(chain):
            try:
                result = chain(bytes(key), role=role, limit=limit)
            except Exception:
                result = None
            if isinstance(result, Mapping):
                edges: list[bytes] = []
                for relation in result.get("relations", ())[:limit]:
                    identifier = _field(relation, "hyperedge_id")
                    if identifier:
                        edges.append(_bytes32(identifier,
                                              field="hyperedge_id"))
                return list(dict.fromkeys(edges)), bool(
                    result.get("truncated", False))
        # Bounded fallback for duck-typed stores without chain indexes.
        total_hint = self._hyperedge_count()
        scan_limit = self.scan_limit or min(
            4096, total_hint if total_hint is not None else 4096)
        edges = []
        parsed_count = 0
        target_leaf = bytes(key)[-LEAF_ID_BYTES:] \
            if len(key) == VALUE_BYTES else bytes(key)
        target_hex = target_leaf.hex()
        target_full_hex = bytes(key).hex()
        for view in self._iter_hyperedges(scan_limit):
            parsed_count += 1
            parsed = self._parse_view(view)
            if role == "subject":
                matched = any(
                    member["kind"] == "concept"
                    and member.get("concept_id") == target_hex
                    for member in parsed["members"]) or any(
                    member["kind"] in ("hyperedge", "parent")
                    and member["id"] == target_full_hex
                    for member in parsed["members"])
                if parsed["hyperedge_id"] is not None \
                        and (matched or parsed["hyperedge_id"] == key):
                    edges.append(parsed["hyperedge_id"])
            elif role == "object":
                matched = any(
                    (member["kind"] in
                     ("cycle", "parent", "hyperedge")
                     and member["id"] == target_full_hex)
                    or (member["kind"] == "concept"
                        and member.get("concept_id") == target_hex)
                    for member in parsed["members"])
                if matched and parsed["hyperedge_id"] is not None:
                    edges.append(parsed["hyperedge_id"])
            if len(edges) >= limit:
                return list(dict.fromkeys(edges))[:limit], True
        total = self._hyperedge_count()
        truncated = total is not None and parsed_count < total
        return list(dict.fromkeys(edges))[:limit], truncated

    def members(self, ref: Any = None, *, include_parents: bool = False,
                limit: int = DEFAULT_LINK_LIMIT, required: bool = False,
                cycle: Any = None, concept: Any = None) -> TypedMembers:
        """Return bounded typed members for a cycle, concept or edge."""
        limit = _check_bounded_int(limit, "limit", 1, MAX_NEIGHBOR_LIMIT)
        if cycle is not None:
            ref = cycle
        if concept is not None:
            ref = concept
        if ref is None:
            raise FullCycleHypergraphError(
                "a cycle, concept or hyperedge reference is required",
                code="invalid_ref")
        # Direct cycle tuple / pair.
        cycle_ref = None
        concept_ref = None
        edge_ref = None
        if isinstance(ref, (tuple, list)) and ref \
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in ref):
            cycle_ref = normalize_cycle(ref, self.nodes, self.root)
        elif isinstance(ref, Mapping):
            if "cycle" in ref:
                cycle_ref = normalize_cycle(ref["cycle"], self.nodes, self.root)
            elif "i" in ref and "j" in ref:
                cycle_ref = normalize_cycle((ref["i"], ref["j"]),
                                            self.nodes, self.root)
            elif "concept" in ref:
                concept_ref = self._resolve_concept(
                    ref["concept"], required=required)
            elif "hyperedge_id" in ref:
                edge_ref = _bytes32(ref["hyperedge_id"], field="hyperedge_id")
            elif "id" in ref:
                edge_ref = _bytes32(ref["id"], field="id")
        elif isinstance(ref, str):
            text = ref.strip()
            parsed_text = text
            for prefix in ("cycle:", "triangle:", "root-cycle:"):
                if parsed_text.lower().startswith(prefix):
                    parsed_text = parsed_text[len(prefix):]
                    break
            if ":" in parsed_text or "," in parsed_text:
                try:
                    cycle_ref = normalize_cycle(parsed_text, self.nodes,
                                                self.root)
                except FullCycleHypergraphError:
                    cycle_ref = None
            if cycle_ref is None and len(text) == VALUE_BYTES * 2:
                try:
                    raw = bytes.fromhex(text)
                except ValueError:
                    raw = None
                if raw is not None:
                    view = self._get_hyperedge(raw)
                    if view is not None:
                        edge_ref = raw
            if cycle_ref is None and edge_ref is None:
                concept_ref = self._resolve_concept(
                    text, required=required)
        elif isinstance(ref, (bytes, bytearray, memoryview)):
            raw = bytes(ref)
            if len(raw) == LEAF_ID_BYTES:
                concept_ref = raw
                if required and self._find_leaf(raw) is None:
                    raise FullCycleNotFound(
                        "concept %s does not exist" % raw.hex(),
                        code="concept_not_found")
            elif len(raw) == VALUE_BYTES:
                view = self._get_hyperedge(raw)
                if view is not None:
                    edge_ref = raw
                else:
                    concept_ref = raw[-LEAF_ID_BYTES:] \
                        if self._find_leaf(raw[-LEAF_ID_BYTES:]) else None
        else:
            try:
                concept_ref = self._resolve_concept(ref, required=required)
            except FullCycleHypergraphError:
                concept_ref = None
        if cycle_ref is not None:
            view = self._find_cycle_view(cycle_ref)
            if view is None:
                if required:
                    raise FullCycleNotFound(
                        "cycle %s does not exist" % cycle_id(
                            cycle_ref, self.nodes, self.root),
                        code="cycle_not_found")
                return TypedMembers(
                    (), owner="cycle", kind="cycle", ref=cycle_ref,
                    cycle=cycle_ref,
                    cycle_id=":".join(str(v) for v in cycle_ref),
                    missing=True, bounded=True)
            parsed = self._parse_view(view)
            non_root = tuple(vertex for vertex in cycle_ref
                             if vertex != self.root)
            out = TypedMembers(
                parsed["members"], owner="cycle", kind="cycle",
                ref=parsed["hyperedge_id_hex"], cycle=cycle_ref,
                cycle_id=parsed["cycle_id"],
                cycle_index=(cycle_index(non_root[0], non_root[1],
                                         self.nodes, self.root)
                             if len(non_root) == 2 else None),
                leaf_id=cycle_leaf_id(cycle_ref, self.nodes,
                                      self.root).hex(),
                hyperedge_id=parsed["hyperedge_id_hex"],
                source_digest=None if parsed["source_digest"] is None
                else parsed["source_digest"].hex(),
                value=parsed["value"],
                payload_digest=None if parsed["payload_digest"] is None
                else parsed["payload_digest"].hex(),
                pair_refs=parsed["pair_refs"], heads=parsed["heads"],
                cycle_ids=parsed["cycle_ids"],
                concept_ids=[item.hex() for item in parsed["concept_ids"]],
                parent_ids=[item.hex() for item in parsed["parent_ids"]],
                missing=False, bounded=True)
            # Reverse concept/parent incidence is derived from the resident
            # object chain, not stored twice on the cycle record.
            linked_edges, linked_truncated = self._linked_edges_for_target(
                parsed["hyperedge_id"], limit)
            concept_ids = list(out.concept_ids)
            parent_ids = list(out.parent_ids)
            for link_id in dict.fromkeys(linked_edges):
                link_view = self._get_hyperedge(link_id)
                if link_view is None:
                    continue
                link = self._parse_view(link_view)
                for member in link["members"]:
                    if member["kind"] == "concept":
                        if member.get("concept_id") not in concept_ids:
                            concept_ids.append(member.get("concept_id"))
                            out.append(member)
                    elif member["kind"] == "parent":
                        if member.get("parent_id") not in parent_ids:
                            parent_ids.append(member.get("parent_id"))
            out.concept_ids = concept_ids
            out.parent_ids = parent_ids
            out.linked_edge_count = len(linked_edges)
            out.truncated = linked_truncated
            return out
        if concept_ref is not None:
            leaf = self._find_leaf(concept_ref)
            if leaf is None:
                if required:
                    raise FullCycleNotFound(
                        "concept %s does not exist" % concept_ref.hex(),
                        code="concept_not_found")
                return TypedMembers(
                    (), owner="concept", kind="concept",
                    concept_id=concept_ref.hex(), missing=True, bounded=True)
            edge_id = concept_hyperedge_id(concept_ref)
            view = self._get_hyperedge(edge_id)
            if view is None:
                if required:
                    raise FullCycleNotFound(
                        "concept edge %s does not exist" % edge_id.hex(),
                        code="concept_edge_not_found")
                return TypedMembers(
                    (), owner="concept", kind="concept",
                    concept_id=concept_ref.hex(), missing=True,
                    bounded=True)
            parsed = self._parse_view(view)
            members = [member for member in parsed["members"]
                       if include_parents or member["kind"] != "parent"]
            cycle_ids = list(parsed["cycle_ids"])
            concept_ids = list(parsed["concept_ids"])
            parent_ids = list(parsed["parent_ids"])
            edges, truncated = self._concept_edges_for_owner(
                concept_ref, limit)
            seen_edges: set[bytes] = set()
            for link_id in edges:
                if link_id in seen_edges:
                    continue
                seen_edges.add(link_id)
                link_view = self._get_hyperedge(link_id)
                if link_view is None:
                    continue
                link = self._parse_view(link_view)
                if link["kind"] not in ("link", "compose"):
                    continue
                for member in link["members"]:
                    if member["kind"] == "concept" \
                            and member.get("concept_id") == concept_ref.hex():
                        continue
                    if member["kind"] == "parent" and not include_parents:
                        continue
                    members.append(member)
                cycle_ids.extend(link["cycle_ids"])
                concept_ids.extend(link["concept_ids"])
                parent_ids.extend(link["parent_ids"])
            if include_parents:
                for parent_id in list(dict.fromkeys(parent_ids)):
                    parent_view = self._get_hyperedge(parent_id)
                    if parent_view is None:
                        continue
                    parent = self._parse_view(parent_view)
                    members.extend(parent["members"])
                    cycle_ids.extend(parent["cycle_ids"])
                    concept_ids.extend(parent["concept_ids"])
            leaf_digest = _field_any(leaf, ("payload_digest", "digest"))
            if isinstance(leaf_digest, (bytes, bytearray, memoryview)):
                leaf_digest = bytes(leaf_digest).hex()
            return TypedMembers(
                members, owner="concept", kind="concept",
                concept_id=concept_ref.hex(), ref=concept_ref.hex(),
                leaf_id=concept_ref.hex(), hyperedge_id=edge_id.hex(),
                source_digest=None if parsed["source_digest"] is None
                else parsed["source_digest"].hex(),
                value=parsed["value"],
                payload_digest=leaf_digest,
                pair_refs=parsed["pair_refs"], heads=parsed["heads"],
                cycle_ids=list(dict.fromkeys(cycle_ids)),
                concept_ids=[item.hex() for item in
                             dict.fromkeys(concept_ids)],
                parent_ids=[item.hex() for item in
                            dict.fromkeys(parent_ids)],
                linked_edge_count=len(seen_edges),
                truncated=truncated, missing=False, bounded=True)
        if edge_ref is not None:
            view = self._get_hyperedge(edge_ref)
            if view is None:
                if required:
                    raise FullCycleNotFound(
                        "hyperedge %s does not exist" % edge_ref.hex(),
                        code="hyperedge_not_found")
                return TypedMembers(
                    (), owner="hyperedge", kind="hyperedge",
                    hyperedge_id=edge_ref.hex(), missing=True, bounded=True)
            parsed = self._parse_view(view)
            members = [member for member in parsed["members"]
                       if include_parents or member["kind"] != "parent"]
            parent_ids = list(parsed["parent_ids"])
            cycle_ids = list(parsed["cycle_ids"])
            concept_ids = list(parsed["concept_ids"])
            owner_edges, owner_truncated = self._owner_edges_for_edge(
                edge_ref, limit)
            for link_id in dict.fromkeys(owner_edges):
                link_view = self._get_hyperedge(link_id)
                if link_view is None:
                    continue
                link = self._parse_view(link_view)
                for member in link["members"]:
                    if member["kind"] == "parent" and not include_parents:
                        continue
                    if member["kind"] in ("hyperedge",) \
                            and member["id"] == edge_ref.hex():
                        continue
                    members.append(member)
                cycle_ids.extend(link["cycle_ids"])
                concept_ids.extend(link["concept_ids"])
                parent_ids.extend(link["parent_ids"])
            if include_parents:
                for parent_id in list(dict.fromkeys(parent_ids)):
                    parent_view = self._get_hyperedge(parent_id)
                    if parent_view is None:
                        continue
                    parent = self._parse_view(parent_view)
                    members.extend(parent["members"])
                    parent_ids.extend(parent["parent_ids"])
            cycle_value = None
            if parsed["kind"] == "cycle" and parsed["cycle"] is not None:
                non_root = tuple(vertex for vertex in parsed["cycle"]
                                 if vertex != self.root)
                if len(non_root) == 2:
                    cycle_value = cycle_index(
                        non_root[0], non_root[1], self.nodes, self.root)
            return TypedMembers(
                members,
                owner="cycle" if parsed["kind"] == "cycle" else "hyperedge",
                kind=parsed["kind"], ref=edge_ref.hex(),
                cycle=parsed["cycle"],
                cycle_id=parsed["cycle_id"],
                cycle_index=cycle_value,
                hyperedge_id=parsed["hyperedge_id_hex"],
                source_digest=None if parsed["source_digest"] is None
                else parsed["source_digest"].hex(),
                value=parsed["value"],
                payload_digest=None if parsed["payload_digest"] is None
                else parsed["payload_digest"].hex(),
                pair_refs=parsed["pair_refs"], heads=parsed["heads"],
                cycle_ids=list(dict.fromkeys(cycle_ids)),
                concept_ids=[item.hex() for item in
                             dict.fromkeys(concept_ids)],
                parent_ids=[item.hex() for item in
                            dict.fromkeys(parent_ids)],
                linked_edge_count=len(owner_edges),
                truncated=owner_truncated, missing=False, bounded=True)
        if required:
            raise FullCycleNotFound("reference %r does not exist" % (ref,),
                                    code="reference_not_found")
        return TypedMembers((), ref=ref, missing=True, bounded=True)

    cycle_members = members
    concept_members = members
    edge_members = members

    # -- neighbours -----------------------------------------------------
    def _linked_edges_for_target(self, target: bytes,
                                 limit: int = DEFAULT_LINK_LIMIT
                                 ) -> tuple[list[bytes], bool]:
        return self._chain_edges(bytes(target), "object", limit)

    def _cycle_star_candidates(self, cycle: tuple[int, ...]
                               ) -> Iterator[tuple[int, ...]]:
        non_root = [vertex for vertex in cycle if vertex != self.root]
        if len(non_root) != 2:
            return
        first, second = non_root
        heads = [head for head in range(self.nodes) if head != self.root]
        for anchor in (first, second):
            for other in heads:
                if other in (first, second):
                    continue
                yield tuple(sorted((self.root, anchor, other)))

    def neighbors(self, cycle: Any, hops: int = 1,
                  limit: int = DEFAULT_NEIGHBOR_LIMIT, *,
                  include_concepts: bool = True,
                  include_parents: bool = True,
                  scan_limit: int | None = None) -> FullCycleNeighbors:
        """Bounded recursive traversal over cycles, concepts and parents."""
        canonical = normalize_cycle(cycle, self.nodes, self.root)
        hops = _check_bounded_int(hops, "hops", 1, MAX_NEIGHBOR_HOPS)
        limit = _check_bounded_int(limit, "limit", 1, MAX_NEIGHBOR_LIMIT)
        if scan_limit is not None:
            scan_limit = _check_bounded_int(
                scan_limit, "scan_limit", 1, MAX_AUDIT_LIMIT)
        if not self._cycle_exists(canonical):
            return FullCycleNeighbors(
                (), cycle=canonical,
                cycle_id=":".join(str(v) for v in canonical),
                hops=hops, limit=limit, cycle_ids=(), concept_ids=(),
                parent_ids=(), truncated=False, missing=True,
                edges_visited=0, members_visited=0, bounded=True)
        visited: set[tuple[int, ...]] = {canonical}
        frontier: list[tuple[int, ...]] = [canonical]
        results: list[tuple[int, ...]] = []
        concept_ids: set[str] = set()
        parent_ids: set[str] = set()
        parsed_cache: dict[bytes, dict[str, Any]] = {}
        processed_edges: set[bytes] = set()
        processed_concepts: set[bytes] = set()
        edges_visited = 0
        members_visited = 0
        truncated = False
        reason = None
        chain_limit = limit + 1

        def parse_edge(edge_id: bytes) -> dict[str, Any] | None:
            nonlocal edges_visited, members_visited
            if edge_id in parsed_cache:
                return parsed_cache[edge_id]
            view = self._get_hyperedge(edge_id)
            if view is None:
                return None
            parsed = self._parse_view(view)
            parsed_cache[edge_id] = parsed
            edges_visited += 1
            members_visited += len(parsed["members"])
            return parsed

        def add_candidate(candidate: tuple[int, ...]) -> bool:
            nonlocal truncated, reason
            if candidate in visited:
                return False
            if len(results) >= limit:
                truncated = True
                reason = reason or "limit"
                return False
            visited.add(candidate)
            results.append(candidate)
            return True

        for _depth in range(hops):
            if not frontier or truncated:
                break
            next_frontier: list[tuple[int, ...]] = []
            for current in frontier:
                # Star neighbours: cycles sharing one non-root head.
                for candidate in self._cycle_star_candidates(current):
                    if candidate in visited:
                        continue
                    if not self._cycle_exists(candidate):
                        continue
                    if add_candidate(candidate):
                        next_frontier.append(candidate)
                    if truncated:
                        break
                if truncated:
                    break
                current_id = cycle_hyperedge_id(
                    current, self.nodes, self.root)
                edges, chain_truncated = self._linked_edges_for_target(
                    current_id, chain_limit)
                if chain_truncated:
                    truncated = True
                    reason = reason or "scan_budget"
                for edge_id in edges:
                    if edge_id in processed_edges:
                        continue
                    processed_edges.add(edge_id)
                    edge = parse_edge(edge_id)
                    if edge is None:
                        continue
                    if edge["kind"] in ("compose", "link") and \
                            (edge["cycle_ids"] or edge["parent_ids"]):
                        parent_ids.add(edge_id.hex())
                    for member in edge["members"]:
                        if member["kind"] == "cycle":
                            resolved = self._resolve_cycle_id(
                                bytes.fromhex(member["id"]))
                            if resolved is None:
                                continue
                            candidate = resolved[0]
                            if add_candidate(candidate):
                                next_frontier.append(candidate)
                        elif member["kind"] == "concept":
                            concept_id = bytes.fromhex(member["concept_id"])
                            concept_ids.add(concept_id.hex())
                            if not include_concepts or \
                                    concept_id in processed_concepts:
                                continue
                            processed_concepts.add(concept_id)
                            linked, linked_truncated = \
                                self._concept_edges_for_owner(
                                    concept_id, chain_limit)
                            if linked_truncated:
                                truncated = True
                                reason = reason or "scan_budget"
                            for link_id in linked:
                                if link_id in processed_edges:
                                    continue
                                processed_edges.add(link_id)
                                link = parse_edge(link_id)
                                if link is None:
                                    continue
                                for linked_member in link["members"]:
                                    if linked_member["kind"] == "cycle":
                                        resolved = self._resolve_cycle_id(
                                            bytes.fromhex(
                                                linked_member["id"]))
                                        if resolved is None:
                                            continue
                                        if add_candidate(resolved[0]):
                                            next_frontier.append(
                                                resolved[0])
                                    elif linked_member["kind"] == "concept":
                                        concept_ids.add(
                                            linked_member["concept_id"])
                        elif member["kind"] == "parent":
                            parent_id = bytes.fromhex(member["parent_id"])
                            parent_ids.add(parent_id.hex())
                            if not include_parents:
                                continue
                            if parent_id in processed_edges:
                                continue
                            processed_edges.add(parent_id)
                            parent = parse_edge(parent_id)
                            if parent is None:
                                continue
                            for parent_member in parent["members"]:
                                if parent_member["kind"] == "cycle":
                                    resolved = self._resolve_cycle_id(
                                        bytes.fromhex(parent_member["id"]))
                                    if resolved is None:
                                        continue
                                    if add_candidate(resolved[0]):
                                        next_frontier.append(resolved[0])
                    if truncated:
                        break
                if truncated:
                    break
            frontier = next_frontier
        return FullCycleNeighbors(
            results, cycle=canonical,
            cycle_id=":".join(str(v) for v in canonical),
            hops=hops, limit=limit,
            cycle_ids=tuple(":".join(str(v) for v in item)
                            for item in results),
            concept_ids=tuple(sorted(concept_ids)),
            parent_ids=tuple(sorted(parent_ids)),
            edges_visited=edges_visited,
            members_visited=members_visited,
            truncated=truncated, reason=reason, missing=False,
            bounded=True)

    traverse = neighbors

    # -- scan / audit ---------------------------------------------------
    def _scan_cycles(self, limit: int | None = None, collect_limit: int = 0,
                     validate: bool = False) -> dict[str, Any]:
        total = self._hyperedge_count()
        if limit is None:
            limit = total if total is not None else MAX_AUDIT_LIMIT
        limit = _check_bounded_int(limit, "limit", 0, MAX_AUDIT_LIMIT)
        seen = bytearray(self.full_cycle_count)
        entries: list[dict[str, Any]] = []
        problems: list[str] = []
        problem_count = 0

        def add_problem(message: str) -> None:
            nonlocal problem_count
            problem_count += 1
            if len(problems) < MAX_PROBLEMS:
                problems.append(message)

        attached = 0
        duplicates = 0
        scanned = 0
        counts = {"cycle": 0, "concept": 0, "link": 0, "compose": 0,
                  "unknown": 0}
        for view in self._iter_hyperedges(limit):
            scanned += 1
            parsed = self._parse_view(view)
            kind = parsed["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "cycle":
                cycle = parsed["cycle"]
                if cycle is None:
                    add_problem("cycle hyperedge has invalid head members")
                    continue
                non_root = tuple(vertex for vertex in cycle
                                 if vertex != self.root)
                if len(non_root) != 2:
                    add_problem("cycle %s has no dense index"
                                % parsed["cycle_id"])
                    continue
                try:
                    index = cycle_index(
                        non_root[0], non_root[1], self.nodes, self.root)
                except FullCycleHypergraphError:
                    add_problem("cycle %s has no dense index"
                                % parsed["cycle_id"])
                    continue
                if seen[index]:
                    duplicates += 1
                    add_problem("duplicate cycle %s" % parsed["cycle_id"])
                else:
                    seen[index] = 1
                    attached += 1
                if collect_limit and len(entries) < collect_limit:
                    entries.append({
                        "cycle": cycle,
                        "cycle_id": parsed["cycle_id"],
                        "cycle_index": index,
                        "value": parsed["value"],
                        "source_digest": None
                        if parsed["source_digest"] is None
                        else parsed["source_digest"].hex(),
                        "payload_digest": None
                        if parsed["payload_digest"] is None
                        else parsed["payload_digest"].hex(),
                        "hyperedge_id": parsed["hyperedge_id_hex"],
                        "leaf_id": cycle_leaf_id(
                            cycle, self.nodes, self.root).hex(),
                        "pair_refs": parsed["pair_refs"],
                        "heads": parsed["heads"],
                        "member_count": len(parsed["members"]),
                    })
                if validate:
                    expected_id = cycle_hyperedge_id(
                        cycle, self.nodes, self.root)
                    if parsed["hyperedge_id"] != expected_id:
                        add_problem("non-deterministic cycle id for %s"
                                    % parsed["cycle_id"])
                    if expected_id not in parsed["cycle_ids"]:
                        add_problem("missing self cycle member for %s"
                                    % parsed["cycle_id"])
                    expected_pairs = tuple(sorted(
                        pair_reference(*edge) for edge in
                        cycle_edges(cycle, self.nodes, self.root)))
                    if tuple(sorted(parsed["pair_refs"])) != expected_pairs:
                        add_problem("pair references do not match incidence "
                                    "for %s" % parsed["cycle_id"])
                    if tuple(sorted(parsed["heads"])) != tuple(sorted(cycle)):
                        add_problem("head members do not match cycle %s"
                                    % parsed["cycle_id"])
                    if parsed["value"] is None:
                        add_problem("missing raw integer value for %s"
                                    % parsed["cycle_id"])
                    if parsed["payload_digest"] is None:
                        add_problem("missing payload digest for %s"
                                    % parsed["cycle_id"])
                    if parsed["source_digest"] in (None, ZERO_ID):
                        add_problem("missing provenance for %s"
                                    % parsed["cycle_id"])
                    if parsed["duplicate_member_count"]:
                        add_problem("duplicate members for %s"
                                    % parsed["cycle_id"])
                continue
            if kind == "concept":
                if validate:
                    if not parsed["concept_ids"]:
                        add_problem("concept hyperedge has no concept member")
                    else:
                        concept_id = parsed["concept_ids"][0]
                        expected = concept_hyperedge_id(concept_id)
                        if parsed["hyperedge_id"] != expected:
                            add_problem(
                                "non-deterministic concept id for %s"
                                % concept_id.hex())
                        for member_id in parsed["concept_ids"]:
                            if self._find_leaf(member_id) is None:
                                add_problem("dangling concept leaf for %s"
                                            % member_id.hex())
                            else:
                                try:
                                    self._concept_ancestors(member_id)
                                except FullCycleParentCycleError as exc:
                                    add_problem(
                                        "concept parent chain for %s: %s"
                                        % (member_id.hex(), exc))
                    if parsed["source_digest"] in (None, ZERO_ID):
                        add_problem("missing provenance for concept edge %s"
                                    % parsed["hyperedge_id_hex"])
                continue
            if kind in ("link", "compose"):
                if validate:
                    if parsed["source_digest"] in (None, ZERO_ID):
                        add_problem("missing provenance for %s edge %s"
                                    % (kind, parsed["hyperedge_id_hex"]))
                    for identifier in parsed["cycle_ids"]:
                        if self._resolve_cycle_id(identifier) is None:
                            add_problem("dangling cycle member %s in %s edge"
                                        % (identifier.hex(),
                                           parsed["hyperedge_id_hex"]))
                    for concept_id in parsed["concept_ids"]:
                        if self._find_leaf(concept_id) is None:
                            add_problem("dangling concept member %s in %s "
                                        "edge" % (concept_id.hex(),
                                                  parsed["hyperedge_id_hex"]))
                    for parent_id in parsed["parent_ids"]:
                        if self._get_hyperedge(parent_id) is None:
                            add_problem("dangling parent member %s in %s "
                                        "edge" % (parent_id.hex(),
                                                  parsed["hyperedge_id_hex"]))
                        elif self._parent_reaches(parent_id,
                                                  parsed["hyperedge_id"]):
                            add_problem("parent cycle through %s"
                                        % parent_id.hex())
                    if parsed["duplicate_member_count"]:
                        add_problem("duplicate members in %s edge %s"
                                    % (kind, parsed["hyperedge_id_hex"]))
                continue
        truncated = total is not None and scanned < total
        missing_indices = [index for index in range(self.full_cycle_count)
                           if not seen[index]]
        missing_ids = [cycle_id_for_index(index, self.nodes, self.root)
                       for index in missing_indices[:16]]
        return {
            "attached_cycle_count": attached,
            "duplicate_cycle_count": duplicates,
            "missing_cycle_count": self.full_cycle_count - attached,
            "missing_cycle_ids": missing_ids,
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "cycles_checked": attached + duplicates,
            "entries": entries,
            "problems": problems,
            "problem_count": problem_count,
            "problems_truncated": problem_count > len(problems),
            "scanned_hyperedges": scanned,
            "hyperedge_count": total,
            "kind_counts": counts,
            "truncated": truncated,
            "limit": limit,
        }

    def verify_full_coverage(self) -> dict[str, Any]:
        """Verify that all 32,385 deterministic cycle ids resolve."""
        scan = self._scan_cycles(validate=True)
        covered = (scan["attached_cycle_count"] == self.full_cycle_count
                   and scan["missing_cycle_count"] == 0
                   and scan["duplicate_cycle_count"] == 0)
        return {
            "schema": SCHEMA,
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "attached_cycle_count": scan["attached_cycle_count"],
            "missing_cycle_count": scan["missing_cycle_count"],
            "missing_cycle_ids": scan["missing_cycle_ids"],
            "duplicate_cycle_count": scan["duplicate_cycle_count"],
            "covered": covered,
            "truncated": scan["truncated"],
            "bounded": True,
            "canonical_merge": False,
        }

    def audit(self, limit: int | None = None, *,
              require_full: bool = False) -> dict[str, Any]:
        """Audit cycle incidence, typed members, provenance and coverage.

        The default scans the complete resident hyperedge table.  It reports
        ``full_cycle_count=32,385`` for the default 256-head space, never a
        256-entry page.  ``truncated`` is true only when an explicit smaller
        ``limit`` was supplied.
        """
        if limit is not None:
            limit = _check_bounded_int(limit, "limit", 1, MAX_AUDIT_LIMIT)
        if require_full:
            limit = None
        scan = self._scan_cycles(limit=limit, validate=True)
        problems = list(scan["problems"])
        if require_full and scan["attached_cycle_count"] != \
                self.full_cycle_count:
            problems.append(
                "full cycle coverage incomplete: %d of %d"
                % (scan["attached_cycle_count"], self.full_cycle_count))
        covered = (scan["attached_cycle_count"] == self.full_cycle_count
                   and scan["duplicate_cycle_count"] == 0
                   and not scan["truncated"])
        status = self._relation_status()
        counts = scan["kind_counts"]
        return {
            "schema": SCHEMA,
            "ok": not problems,
            "problems": problems,
            "issues": list(problems),
            "problem_count": scan["problem_count"] + (
                1 if require_full and scan["attached_cycle_count"]
                != self.full_cycle_count else 0),
            "problems_truncated": scan["problems_truncated"],
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "attached_cycle_count": scan["attached_cycle_count"],
            "cycle_count": scan["attached_cycle_count"],
            "cycles_checked": scan["cycles_checked"],
            "missing_cycle_count": scan["missing_cycle_count"],
            "missing_cycle_ids": scan["missing_cycle_ids"],
            "duplicate_cycle_count": scan["duplicate_cycle_count"],
            "full_coverage": covered,
            "coverage": (scan["attached_cycle_count"]
                         / self.full_cycle_count
                         if self.full_cycle_count else 1.0),
            "concept_count": counts.get("concept", 0),
            "link_count": counts.get("link", 0),
            "parent_count": counts.get("compose", 0),
            "foreign_edge_count": counts.get("unknown", 0),
            "scanned_hyperedges": scan["scanned_hyperedges"],
            "hyperedge_count": scan["hyperedge_count"],
            "leaf_count": self._leaf_count(),
            "truncated": scan["truncated"],
            "bounded": True,
            "limit": scan["limit"],
            "canonical_merge": False,
            "no_duplicate_storage": True,
            "generation": self.generation,
            "nodes": self.nodes,
            "root": self.root,
            "stores": {
                "relations": type(self.relations).__name__,
                "leaves": type(self.leaves).__name__,
                "cells": None if self.cells is None
                else type(self.cells).__name__,
            },
            "capacity": {
                "relation_capacity": status.get("relation_capacity"),
                "hyperedge_capacity": status.get("hyperedge_capacity"),
                "arena_bytes": status.get("arena_bytes"),
                "arena_used": status.get("arena_used"),
            },
        }

    def snapshot(self, limit: int = DEFAULT_SNAPSHOT_CYCLES, *,
                 include_members: bool = False) -> dict[str, Any]:
        """Return a bounded, JSON-friendly projection of the full space."""
        maximum = min(MAX_SNAPSHOT_LIMIT, self.full_cycle_count)
        if limit is None:
            limit = maximum
        if limit is not None and isinstance(limit, int) \
                and not isinstance(limit, bool) and limit > maximum:
            limit = maximum
        limit = _check_bounded_int(limit, "limit", 1, maximum)
        scan = self._scan_cycles(
            validate=False, collect_limit=limit)
        counts = scan["kind_counts"]
        entries = scan["entries"]
        if include_members:
            for entry in entries:
                members = self.members(entry["cycle"], limit=9)
                entry["members"] = members.as_dict()["members"]
        truncated = bool(scan["truncated"] or
                         scan["attached_cycle_count"] > len(entries))
        return {
            "schema": SCHEMA,
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "attached_cycle_count": scan["attached_cycle_count"],
            "cycle_count": scan["attached_cycle_count"],
            "missing_cycle_count": scan["missing_cycle_count"],
            "duplicate_cycle_count": scan["duplicate_cycle_count"],
            "cycles": entries,
            "returned": len(entries),
            "limit": limit,
            "truncated": truncated,
            "bounded": True,
            "scanned_hyperedges": scan["scanned_hyperedges"],
            "hyperedge_count": scan["hyperedge_count"],
            "concept_count": counts.get("concept", 0),
            "link_count": counts.get("link", 0),
            "parent_count": counts.get("compose", 0),
            "leaf_count": self._leaf_count(),
            "canonical_merge": False,
            "no_duplicate_storage": True,
            "generation": self.generation,
            "nodes": self.nodes,
            "root": self.root,
        }

    # -- disabled canonical merge --------------------------------------
    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Canonical merge is deliberately disabled in this layer."""
        raise FullCycleDenied(
            "canonical cycle merge is disabled",
            code="canonical_merge_disabled",
            details={"canonical_merge": False})

    promote = merge

    # -- reopen ---------------------------------------------------------
    def reopen(self) -> "FullCycleHypergraph":
        """Re-attach the same stores or reopen a host universe."""
        host = self.host
        reopen = _host_attr(host, "reopen", None)
        if callable(reopen):
            try:
                reopened = reopen()
            except TypeError:
                reopened = None
            if reopened is not None:
                host = reopened
        if host is not None and host is not self.host:
            return FullCycleHypergraph(
                host, nodes=self.nodes, root=self.root,
                scan_limit=self.scan_limit)
        relations = self.relations
        leaves = self.leaves
        relation_buffer = _host_attr(self.relations, "_buffer", None)
        if relation_buffer is not None:
            try:
                relations = RelationStore.attach(relation_buffer)
            except Exception:
                relations = self.relations
        leaf_buffer = _host_attr(self.leaves, "_buffer", None)
        if leaf_buffer is not None:
            try:
                leaves = LeafStore.attach(
                    leaf_buffer, base_offset=_host_attr(
                        self.leaves, "base_offset", 0))
            except Exception:
                leaves = self.leaves
        return FullCycleHypergraph(
            relations=relations, leaves=leaves, cells=self.cells,
            nodes=self.nodes, root=self.root, scan_limit=self.scan_limit)

    def status(self) -> dict[str, Any]:
        """Small bounded status projection."""
        scan = self._scan_cycles(validate=False, collect_limit=0)
        return {
            "schema": SCHEMA,
            "full_cycle_count": self.full_cycle_count,
            "cycle_capacity": self.cycle_capacity,
            "attached_cycle_count": scan["attached_cycle_count"],
            "missing_cycle_count": scan["missing_cycle_count"],
            "hyperedge_count": scan["hyperedge_count"],
            "leaf_count": self._leaf_count(),
            "concept_count": scan["kind_counts"].get("concept", 0),
            "link_count": scan["kind_counts"].get("link", 0),
            "parent_count": scan["kind_counts"].get("compose", 0),
            "canonical_merge": False,
            "bounded": True,
            "generation": self.generation,
        }

    def __repr__(self) -> str:
        return ("FullCycleHypergraph(nodes=%d, root=%d, full_cycle_count=%d, "
                "hyperedges=%r)" % (self.nodes, self.root,
                                    self.full_cycle_count,
                                    self._hyperedge_count()))


# Compatibility spellings with the single-cycle resident layer.
CycleMembers = TypedMembers
CycleNeighbors = FullCycleNeighbors
CycleComposition = Composition


__all__ = (
    "SCHEMA", "DEFAULT_NODES", "MAX_NODES", "DEFAULT_ROOT",
    "DEFAULT_EXTRA_EDGES", "DEFAULT_CONCEPT_CAPACITY",
    "DEFAULT_SNAPSHOT_CYCLES", "DEFAULT_NEIGHBOR_LIMIT",
    "DEFAULT_LINK_LIMIT", "MAX_NEIGHBOR_HOPS", "MAX_NEIGHBOR_LIMIT",
    "MAX_SNAPSHOT_LIMIT", "MAX_AUDIT_LIMIT", "MAX_PROBLEMS",
    "MAX_PARENT_DEPTH", "VALUE_BYTES", "FULL_CYCLE_COUNT", "ROOT",
    "ROOT_HEAD", "CYCLE_HEAD_COUNT", "CYCLE_COUNT", "DIMENSION",
    "BASIS_SIZE", "ROOT_CYCLE_COUNT", "full_cycle_count", "MEMBER_LEAF",
    "MEMBER_PAIR", "MEMBER_CYCLE", "MEMBER_VALUE", "MEMBER_HEAD",
    "MEMBER_VERTEX", "MEMBER_PAYLOAD", "MEMBER_CONCEPT", "MEMBER_PARENT",
    "MEMBER_HYPEREDGE", "MEMBER_NAMES", "MEMBER_CODES", "MEMBERSHIP_KINDS",
    "FullCycleHypergraphError", "FullCycleBoundsError",
    "FullCycleProvenanceError", "FullCycleConflictError",
    "FullCycleCorruptionError", "FullCycleNotFound", "FullCycleDenied",
    "FullCycleParentCycleError", "FullCycleError", "FullCycleRangeError",
    "CycleHypergraphError", "CycleBoundsError", "CycleProvenanceError",
    "CycleConflictError", "CycleCorruptionError", "CycleNotFound",
    "CycleDenied", "TypedMembers", "FullCycleNeighbors",
    "Composition", "CycleMembers", "CycleNeighbors", "CycleComposition",
    "root_cycle_count", "normalize_cycle",
    "normalize_root_cycle", "cycle_id", "cycle_leaf_id",
    "cycle_hyperedge_id", "cycle_relation_id", "cycle_edges",
    "pair_reference", "cycle_index", "pair_from_index", "cycle_from_index",
    "cycle_at", "cycle_pair", "pair_of_index", "index_of", "pair_to_index",
    "cycle_row_start", "cycle_id_for_index", "iter_root_pairs",
    "iter_cycle_pairs", "iter_root_cycles", "iter_cycles",
    "root_cycle_pairs", "full_cycle_ids",
    "concept_leaf_id", "concept_hyperedge_id", "FullCycleHypergraph",
)
