"""Cycle observables as first-class resident hypergraph members.

A cycle observable is the fundamental triangle ``(0, i, j)`` over the complete
256-head pair graph.  This module attaches those cycles to the existing
resident stores instead of introducing a second graph:

* :class:`~nanlmind_hermes.resident.relations.RelationStore` stores one typed
  hyperedge per cycle.  The hyperedge members are *references* to the three
  existing pair cells plus the cycle leaf, the raw integer value and the
  payload digest.  Pair cells are never duplicated.
* :class:`~nanlmind_hermes.resident.leaves.LeafStore` stores one deterministic
  cycle leaf per observable.  Its ``payload_digest`` is the content address of
  the semantic payload.
* :class:`~nanlmind_hermes.resident.cells.ResidentCells` is used (when
  supplied) only to resolve the existing pair-cell index.  The canonical
  16-bit pair address is the stable reference written into the hyperedge.

Cycle identity is canonical and mirror-safe: ``(0, i, j)`` and ``(0, j, i)``
name the same observable.  Re-adding the same cycle with the same value and
payload is idempotent (a second provenance is reported as a source mismatch,
never overwritten).  Re-adding a cycle with a different value or payload is a
conflict and is rejected; no canonical merge is performed anywhere.

The implementation is deliberately store-backed.  It keeps no growing Python
dictionary of cycles; lookups use deterministic content-addressed ids and
bounded scans.  All traversal, snapshot and audit operations accept an
explicit bound.

This is classical graph/incidence bookkeeping.  The raw integer value is an
observable supplied by the caller; this module makes no quantum claim.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Iterator

from .cells import (
    PAIR_COUNT,
    _PAIR_CANONICAL,
    canonical_address,
    mirror_address,
    pair_index as cell_pair_index,
    reverse8,
    reverse16,
)
from .leaves import INDEX_ENTRY_BYTES, RECORD_BYTES, HEADER_BYTES as LEAF_HEADER_BYTES
from .leaves import LeafStore
from .relations import (
    RelationBoundsError,
    RelationConflictError,
    RelationStore,
)

SCHEMA = "nanlm.resident-cycle-hypergraph/1"

DEFAULT_NODES = 256
DEFAULT_SNAPSHOT_CYCLES = 64
DEFAULT_AUDIT_SCAN = 4096
MAX_NEIGHBOR_HOPS = 8
MAX_NEIGHBOR_LIMIT = 512
MAX_SNAPSHOT_LIMIT = 512
DEFAULT_CAPACITY = 256

# Custom hyperedge member kinds.  The relation store accepts integer kind
# codes in [0, 255]; unknown codes are preserved verbatim.
MEMBER_PAIR = 3              # same code as relations.MEMBER_PAIR
MEMBER_CYCLE = 5             # cycle leaf reference
MEMBER_VALUE = 6             # signed raw integer, 32-byte two's complement
MEMBER_VERTEX = 7            # graph vertex/head reference
MEMBER_PAYLOAD = 8           # SHA-256 of the canonical semantic payload

MEMBER_NAMES = {
    MEMBER_PAIR: "pair",
    MEMBER_CYCLE: "cycle",
    MEMBER_VALUE: "value",
    MEMBER_VERTEX: "vertex",
    MEMBER_PAYLOAD: "payload",
}
MEMBER_CODES = {name: code for code, name in MEMBER_NAMES.items()}

CYCLE_PREDICATE = hashlib.sha256(b"NLMR-CYCLE-PREDICATE\0").digest()
VALUE_BYTES = 32
VALUE_MIN = -(1 << (VALUE_BYTES * 8 - 1))
VALUE_MAX = (1 << (VALUE_BYTES * 8 - 1)) - 1
ZERO_ID = b"\x00" * 32


class CycleHypergraphError(ValueError):
    """Base error for resident cycle-hypergraph operations."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


class CycleBoundsError(CycleHypergraphError):
    """A bounded capacity, traversal or integer range was exceeded."""


class CycleProvenanceError(CycleHypergraphError):
    """A cycle observation was supplied without usable provenance."""


class CycleConflictError(CycleHypergraphError):
    """The same canonical cycle names different immutable content."""


class CycleCorruptionError(CycleHypergraphError):
    """Persisted cycle links do not agree with their deterministic identity."""


class CycleNotFound(CycleHypergraphError, KeyError):
    """A requested cycle observable does not exist in the resident stores."""


class CycleDenied(CycleHypergraphError):
    """A composition or traversal request is structurally not allowed."""


# ---------------------------------------------------------------------------
# small value types
# ---------------------------------------------------------------------------
class CycleMembers(list):
    """List of incidence members with a dictionary-style metadata view.

    The list itself contains the three pair-cell references in traversal order
    ``(0, i), (i, j), (0, j)``.  Metadata is available as attributes or with
    ``members["value"]`` so callers may treat the result as either a bounded
    list or a small record.
    """

    def __init__(self, members: Iterable[Mapping[str, Any]] = (),
                 **metadata: Any):
        super().__init__(members)
        for name, value in metadata.items():
            setattr(self, name, value)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "members":
                return list(self)
            try:
                return getattr(self, key)
            except AttributeError as exc:
                raise KeyError(key) from exc
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key == "members":
            return list(self)
        return getattr(self, key, default)

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": getattr(self, "cycle", None),
            "cycle_id": getattr(self, "cycle_id", None),
            "value": getattr(self, "value", None),
            "source": getattr(self, "source", None),
            "source_digest": getattr(self, "source_digest", None),
            "payload": getattr(self, "payload", None),
            "payload_digest": getattr(self, "payload_digest", None),
            "hyperedge_id": getattr(self, "hyperedge_id", None),
            "leaf_id": getattr(self, "leaf_id", None),
            "vertices": getattr(self, "vertices", ()),
            "edges": getattr(self, "edges", ()),
            "pair_refs": getattr(self, "pair_refs", ()),
            "members": list(self),
            "missing": getattr(self, "missing", False),
        }


class CycleNeighbors(list):
    """Bounded cycle-neighbour list with traversal metadata."""

    def __init__(self, cycles: Iterable[tuple[int, int, int]] = (),
                 **metadata: Any):
        super().__init__(cycles)
        for name, value in metadata.items():
            setattr(self, name, value)

    @property
    def neighbors(self) -> list[tuple[int, int, int]]:
        return list(self)

    @property
    def cycles(self) -> list[tuple[int, int, int]]:
        return list(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("neighbors", "cycles"):
                return list(self)
            try:
                return getattr(self, key)
            except AttributeError as exc:
                raise KeyError(key) from exc
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key in ("neighbors", "cycles"):
            return list(self)
        return getattr(self, key, default)

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": getattr(self, "cycle", None),
            "hops": getattr(self, "hops", None),
            "limit": getattr(self, "limit", None),
            "neighbors": list(self),
            "cycles": list(self),
            "cycle_ids": getattr(self, "cycle_ids", ()),
            "truncated": getattr(self, "truncated", False),
            "missing": getattr(self, "missing", False),
        }


class CycleComposition(int):
    """Raw integer composition result with descriptive metadata.

    The object is an ``int`` so ``compose(c1, c2)`` can be used directly as a
    raw cycle value, while ``composed.value``, ``composed.cycle``,
    ``composed.members`` or ``composed["cycle"]`` expose the geometric detail.
    Equality accepts an integer value, a cycle tuple or a mapping.
    """

    def __new__(cls, value: int, **metadata: Any):
        obj = int.__new__(cls, value)
        for name, item in metadata.items():
            setattr(obj, name, item)
        return obj

    @property
    def value(self) -> int:
        return int(self)

    @property
    def raw_value(self) -> int:
        return int(self)

    @property
    def cycle_id(self) -> str | None:
        cycle = getattr(self, "cycle", None)
        if isinstance(cycle, (tuple, list)) and all(
                isinstance(item, int) for item in cycle):
            return ":".join(str(item) for item in cycle)
        return None

    @property
    def shared_edges(self):
        return getattr(self, "shared", ())

    @property
    def outer_cycle(self):
        return getattr(self, "cycle", None)

    @property
    def triangle(self):
        """The shortcut triangle of two adjacent fundamental cycles, if any."""
        cycles = tuple(getattr(self, "cycles", ()))
        if len(cycles) != 2:
            return None
        first, second = cycles
        shared_vertices = set(first) & set(second)
        others = sorted((set(first) | set(second)) - shared_vertices)
        if len(others) != 2:
            return None
        return normalize_cycle((0, others[0], others[1]))

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("members", "edges"):
                return list(getattr(self, "members", ()))
            if key in ("cycles",):
                return tuple(getattr(self, "cycles", ()))
            if key == "value":
                return int(self)
            try:
                return getattr(self, key)
            except AttributeError as exc:
                raise KeyError(key) from exc
        try:
            return (int(self), getattr(self, "cycle", None),
                    list(getattr(self, "members", ())))[key]
        except (IndexError, TypeError) as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": getattr(self, "status", "composed"),
            "value": int(self),
            "cycle": getattr(self, "cycle", None),
            "cycles": tuple(getattr(self, "cycles", ())),
            "shared": getattr(self, "shared", ()),
            "members": list(getattr(self, "members", ())),
            "created": False,
            "stored": False,
        }

    def __eq__(self, other):
        if isinstance(other, CycleComposition):
            return (int(self) == int(other)
                    and getattr(self, "cycle", None)
                    == getattr(other, "cycle", None))
        if isinstance(other, bool):
            return False
        if isinstance(other, int):
            return int(self) == other
        if isinstance(other, (tuple, list)):
            return tuple(getattr(self, "cycle", ())) == tuple(other)
        if isinstance(other, Mapping):
            return self.as_dict() == dict(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    __hash__ = int.__hash__

    def __repr__(self):
        return ("CycleComposition(value=%d, cycle=%r, shared=%r)"
                % (int(self), getattr(self, "cycle", None),
                   getattr(self, "shared", ())))


# ---------------------------------------------------------------------------
# generic record helpers
# ---------------------------------------------------------------------------
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


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _check_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleHypergraphError("%s must be an integer" % label,
                                   code="invalid_%s" % label)
    return value


def _check_bounded_int(value: Any, label: str, minimum: int = 0,
                       maximum: int | None = None) -> int:
    number = _check_int(value, label)
    if number < minimum:
        raise CycleBoundsError(
            "%s must be at least %d" % (label, minimum),
            code="%s_out_of_bounds" % label,
            details={label: number})
    if maximum is not None and number > maximum:
        raise CycleBoundsError(
            "%s must be at most %d" % (label, maximum),
            code="%s_too_large" % label,
            details={label: number, "maximum": maximum})
    return number


def _bounded_limit(value: Any, default: int, maximum: int,
                   label: str = "limit") -> int:
    if value is None:
        value = default
    return _check_bounded_int(value, label, 1, maximum)


def _bytes32(value: Any, *, signed: bool = False,
             field: str = "id") -> bytes:
    """Normalize an id/value to exactly 32 big-endian bytes."""
    if isinstance(value, bool):
        raise CycleHypergraphError("%s must not be a boolean" % field,
                                   code="invalid_%s" % field)
    if isinstance(value, int):
        try:
            return value.to_bytes(VALUE_BYTES, "big", signed=signed)
        except OverflowError as exc:
            raise CycleBoundsError(
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
            raise CycleHypergraphError("%s must not be empty" % field,
                                       code="empty_%s" % field)
        if len(raw) >= VALUE_BYTES:
            return raw[:VALUE_BYTES] if len(raw) == VALUE_BYTES \
                else hashlib.sha256(raw).digest()
        return raw.rjust(VALUE_BYTES, b"\x00")
    raise CycleHypergraphError(
        "%s must be an integer, string or bytes" % field,
        code="invalid_%s" % field)


def _source_digest(value: Any) -> bytes:
    if value is None:
        raise CycleProvenanceError(
            "cycle provenance source is required", code="source_required")
    if isinstance(value, bool):
        raise CycleProvenanceError(
            "cycle provenance source must not be a boolean",
            code="invalid_source")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise CycleProvenanceError(
                "cycle provenance source must not be empty",
                code="empty_source")
        if len(raw) == VALUE_BYTES:
            return raw
        if len(raw) < VALUE_BYTES:
            return raw.rjust(VALUE_BYTES, b"\x00")
        return hashlib.sha256(raw).digest()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise CycleProvenanceError(
                "cycle provenance source must not be empty",
                code="empty_source")
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
                b"NLMR-CYCLE-SOURCE\0" + _canonical(value)).digest()
        except (TypeError, ValueError) as exc:
            raise CycleProvenanceError(
                "cycle provenance source is not serializable",
                code="invalid_source") from exc
    raise CycleProvenanceError(
        "cycle provenance source must be text, bytes, a mapping or an integer",
        code="invalid_source")


def _payload_value_and_digest(payload: Any) -> tuple[int, bytes, bytes]:
    """Return ``(raw_int_value, canonical_bytes, digest)`` for a payload."""
    if isinstance(payload, bool):
        raise CycleHypergraphError(
            "semantic payload value must be a raw integer, not a boolean",
            code="invalid_payload")
    if isinstance(payload, int):
        value = payload
        canonical = _canonical({"value": value})
    elif isinstance(payload, Mapping):
        record = dict(payload)
        raw_value = record.get("value", record.get("cycle_value", 0))
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise CycleHypergraphError(
                "semantic payload 'value' must be a raw integer",
                code="invalid_payload_value",
                details={"value": raw_value})
        value = raw_value
        try:
            canonical = _canonical(record)
        except (TypeError, ValueError) as exc:
            raise CycleHypergraphError(
                "semantic payload is not JSON serializable",
                code="invalid_payload") from exc
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        value = 0
        canonical = b"NLMR-CYCLE-PAYLOAD\0" + bytes(payload)
    elif isinstance(payload, str):
        value = 0
        canonical = b"NLMR-CYCLE-PAYLOAD\0" + payload.encode("utf-8")
    else:
        raise CycleHypergraphError(
            "semantic payload must be a raw integer, mapping, string or bytes",
            code="invalid_payload")
    if value < VALUE_MIN or value > VALUE_MAX:
        raise CycleBoundsError(
            "cycle value does not fit in a signed %d-byte integer"
            % VALUE_BYTES,
            code="value_out_of_bounds",
            details={"value": value})
    digest = hashlib.sha256(b"NLMR-CYCLE-PAYLOAD\0" + canonical).digest()
    return value, canonical, digest


# ---------------------------------------------------------------------------
# cycle identity
# ---------------------------------------------------------------------------
def _parse_cycle_text(text: str) -> tuple[int, ...]:
    body = text.strip()
    for prefix in ("cycle:", "triangle:", "hyperedge:"):
        if body.lower().startswith(prefix):
            body = body[len(prefix):]
            break
    parts = body.replace(",", ":").replace("/", ":").split(":")
    parts = [part for part in parts if part != ""]
    if not parts:
        raise CycleHypergraphError("cycle id is empty", code="invalid_cycle")
    try:
        return tuple(int(part, 10) for part in parts)
    except ValueError as exc:
        raise CycleHypergraphError(
            "cycle id must contain integers", code="invalid_cycle",
            details={"cycle": text}) from exc


def normalize_cycle(cycle: Any, nodes: int = DEFAULT_NODES) -> tuple[int, int, int]:
    """Return the canonical, mirror-safe ``(a, b, c)`` cycle id.

    Two-element ``(i, j)`` ids are interpreted as the fundamental triangle
    ``(0, i, j)``.  Three-element ids are sorted, so ``(0, j, i)`` and
    ``(0, i, j)`` are the same cycle.  All vertices must be distinct and in
    ``[0, nodes)``.
    """
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
        raise CycleBoundsError("nodes must be an integer of at least 3",
                               code="invalid_nodes")
    if isinstance(cycle, Mapping):
        if "cycle" in cycle:
            cycle = cycle["cycle"]
        elif "vertices" in cycle:
            cycle = cycle["vertices"]
        elif "i" in cycle and "j" in cycle:
            cycle = (0, cycle["i"], cycle["j"])
        elif "id" in cycle:
            cycle = cycle["id"]
        else:
            raise CycleHypergraphError(
                "cycle mapping needs cycle, vertices, id or i/j",
                code="invalid_cycle")
    if isinstance(cycle, str):
        cycle = _parse_cycle_text(cycle)
    if isinstance(cycle, CycleComposition):
        cycle = getattr(cycle, "cycle", None)
    if isinstance(cycle, (tuple, list)):
        values = tuple(cycle)
    else:
        raise CycleHypergraphError(
            "cycle must be a tuple, list, string or mapping",
            code="invalid_cycle")
    if len(values) == 2:
        values = (0,) + values
    if len(values) != 3:
        raise CycleHypergraphError(
            "cycle id must contain exactly three vertices",
            code="invalid_cycle", details={"cycle": values})
    vertices = tuple(_check_int(value, "cycle vertex") for value in values)
    if len(set(vertices)) != 3:
        raise CycleHypergraphError(
            "cycle vertices must be distinct", code="invalid_cycle",
            details={"cycle": vertices})
    if any(vertex < 0 or vertex >= nodes for vertex in vertices):
        raise CycleBoundsError(
            "cycle vertices must be in [0, %d)" % nodes,
            code="cycle_vertex_out_of_bounds",
            details={"cycle": vertices, "nodes": nodes})
    return tuple(sorted(vertices))  # type: ignore[return-value]


def cycle_id(cycle: Any, nodes: int = DEFAULT_NODES) -> str:
    """Canonical textual cycle id, e.g. ``"0:1:2"``."""
    return ":".join(str(vertex) for vertex in normalize_cycle(cycle, nodes))


def _cycle_key(cycle: tuple[int, int, int]) -> bytes:
    return ("%d:%d:%d" % cycle).encode("ascii")


def cycle_leaf_id(cycle: Any, nodes: int = DEFAULT_NODES) -> bytes:
    """Deterministic 128-bit leaf id for a cycle observable."""
    canonical = normalize_cycle(cycle, nodes)
    return hashlib.sha256(
        b"NLMR-CYCLE-LEAF\0" + _cycle_key(canonical)).digest()[:16]


def cycle_hyperedge_id(cycle: Any, nodes: int = DEFAULT_NODES) -> bytes:
    """Deterministic 256-bit hyperedge id for a cycle observable."""
    canonical = normalize_cycle(cycle, nodes)
    return hashlib.sha256(
        b"NLMR-CYCLE-EDGE\0" + _cycle_key(canonical)).digest()


def cycle_relation_id(cycle: Any, nodes: int = DEFAULT_NODES) -> bytes:
    """Deterministic relation id used to attach the cycle hyperedge."""
    canonical = normalize_cycle(cycle, nodes)
    return hashlib.sha256(
        b"NLMR-CYCLE-REL\0" + _cycle_key(canonical)).digest()


def cycle_edges(cycle: Any, nodes: int = DEFAULT_NODES
                ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Incidence edges of a canonical cycle in traversal order."""
    a, b, c = normalize_cycle(cycle, nodes)
    # For the fundamental triangle (0, i, j), keep the order used by the
    # observable: (0,i), (i,j), (0,j).  For a general triple, sort edges.
    if a == 0:
        return ((0, b), (b, c), (0, c))
    return tuple(sorted(((a, b), (b, c), (a, c))))  # type: ignore[return-value]


def pair_reference(a: int, b: int) -> int:
    """Mirror-safe dense pair-cell index used as a resident reference."""
    return cell_pair_index(int(a), int(b))


def pair_address(a: int, b: int) -> int:
    """Mirror-safe canonical 16-bit pair address for two heads."""
    return canonical_address(int(a), int(b))


def _pair_vertices(reference: int) -> tuple[int, int]:
    value = int(reference)
    if 0 <= value < PAIR_COUNT:
        address = _PAIR_CANONICAL[value]
    else:
        address = value & 0xFFFF
    first = address >> 8
    second = reverse8(address & 0xFF)
    return (first, second) if first <= second else (second, first)


# ---------------------------------------------------------------------------
# store adapters
# ---------------------------------------------------------------------------
def _default_stores(capacity: int = DEFAULT_CAPACITY):
    relations = RelationStore.allocate(
        relation_capacity=max(8, capacity),
        hyperedge_capacity=max(8, capacity),
        arena_bytes=max(8192, capacity * 16 * 33))
    index_capacity = _next_power_of_two(max(16, capacity * 2))
    buffer = bytearray(
        LEAF_HEADER_BYTES + capacity * RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)
    leaves = LeafStore.create(buffer, capacity=capacity)
    return relations, leaves, None


def _resolve_stores(universe_or_stores: Any = None, *,
                    relations: Any = None, leaves: Any = None,
                    cells: Any = None):
    """Resolve a universe, a mapping, a tuple/list or direct stores."""
    host = None
    source = universe_or_stores
    if source is None:
        pass
    elif isinstance(source, Mapping) and (
            "relations" in source or "leaves" in source or "cells" in source):
        relations = relations if relations is not None else source.get("relations")
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
    if relations is None or leaves is None:
        default_relations, default_leaves, _ = _default_stores()
        if relations is None:
            relations = default_relations
        if leaves is None:
            leaves = default_leaves
    return relations, leaves, cells, host


def _record_leaf_id(record: Any) -> bytes | None:
    value = _field_any(record, ("leaf_id", "leaf_id_hex", "id"))
    if value is None and hasattr(record, "to_dict"):
        try:
            return _record_leaf_id(record.to_dict())
        except Exception:
            return None
    if isinstance(value, str):
        text = value[5:] if value.lower().startswith("leaf:") else value
        try:
            return bytes.fromhex(text)
        except ValueError:
            return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def _record_digest(record: Any) -> bytes | None:
    value = _field_any(record, ("payload_digest", "digest"))
    if value is None and hasattr(record, "to_dict"):
        try:
            return _record_digest(record.to_dict())
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return bytes.fromhex(value)
        except ValueError:
            return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


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
                if isinstance(name, str) and name in MEMBER_CODES:
                    code = MEMBER_CODES[name]
            identifier = member.get("id", member.get("member_id"))
            if identifier is None and "id_int" in member:
                identifier = member["id_int"]
        elif isinstance(member, (tuple, list)) and len(member) == 2:
            kind, identifier = member
            code = kind if isinstance(kind, int) and not isinstance(kind, bool) \
                else MEMBER_CODES.get(str(kind), MEMBER_PAIR)
        else:
            if kinds is not None and index < len(kinds):
                kind = kinds[index]
                code = kind if isinstance(kind, int) else MEMBER_CODES.get(
                    str(kind), MEMBER_PAIR)
            else:
                code = MEMBER_PAIR
            identifier = member
        if isinstance(code, bool) or not isinstance(code, int):
            code = MEMBER_PAIR
        out.append((code, _bytes32(identifier, field="member id")))
    return out


def _decode_value(raw: bytes) -> int:
    return int.from_bytes(raw, "big", signed=True)


def _leaf_id_from_member(raw: bytes) -> bytes:
    # Cycle leaf ids are 16 bytes stored left-padded in a 32-byte member id.
    # Strip the padding without damaging a real id that begins with 0x00.
    if len(raw) == VALUE_BYTES:
        return raw[VALUE_BYTES - 16:]
    if len(raw) == 16:
        return raw
    stripped = raw.lstrip(b"\x00")
    return stripped if stripped else raw


class CycleHypergraph:
    """Bounded cycle-observable layer over existing resident stores."""

    def __init__(self, universe_or_stores: Any = None, *,
                 relations: Any = None, leaves: Any = None, cells: Any = None,
                 nodes: int = DEFAULT_NODES,
                 commit: Any = None, flush: Any = None):
        if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
            raise CycleBoundsError("nodes must be an integer of at least 3",
                                   code="invalid_nodes")
        self.nodes = nodes
        self.relations, self.leaves, self.cells, self.host = _resolve_stores(
            universe_or_stores, relations=relations, leaves=leaves,
            cells=cells)
        self._commit = commit if commit is not None else (
            _host_attr(self.host, "_commit") or _host_attr(self.host, "commit"))
        self._flush = flush if flush is not None else _host_attr(
            self.host, "flush")

    @classmethod
    def attach(cls, universe_or_stores: Any = None, **kwargs: Any
               ) -> "CycleHypergraph":
        """Construct an adapter over existing stores (alias of ``__init__``)."""
        return cls(universe_or_stores, **kwargs)

    # -- store metadata -------------------------------------------------
    @property
    def canonical_merge(self) -> bool:
        """Canonical merge is never performed by this layer."""
        return False

    def __len__(self) -> int:
        return self._count_cycle_hyperedges(MAX_SNAPSHOT_LIMIT * 16)

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
        if isinstance(records, Mapping):
            return len(records)
        if isinstance(records, (list, tuple)):
            return len(records)
        return None

    def _leaf_count(self) -> int | None:
        status = _host_attr(self.leaves, "stats", None)
        if callable(status):
            try:
                value = status()
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
            for key in (identifier, bytes(identifier).hex()
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

    def _iter_hyperedges(self, limit: int) -> Iterator[Any]:
        """Yield at most ``limit`` hyperedges from whichever store shape."""
        iterator = _host_attr(self.relations, "iter_hyperedges", None)
        if callable(iterator):
            for index, item in enumerate(iterator()):
                if index >= limit:
                    break
                yield item
            return
        # A duck-typed store may expose a concrete sequence/mapping.  Use it
        # before index lookup so integer-indexed fakes and id-indexed fakes
        # both behave predictably.
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

    def _count_cycle_hyperedges(self, limit: int) -> int:
        count = 0
        for view in self._iter_hyperedges(limit):
            try:
                parsed = self._parse_cycle_view(view)
            except Exception:
                continue
            if parsed is not None:
                count += 1
        return count

    def _check_store_capacity(self, *, leaf_needed: bool,
                              members: int = 9) -> None:
        """Fail before writing when a bounded store has no room left."""
        status = self._relation_status()
        for count_key, capacity_key, code in (
                ("relations", "relation_capacity", "relation_capacity"),
                ("hyperedges", "hyperedge_capacity", "hyperedge_capacity")):
            count = status.get(count_key)
            capacity = status.get(capacity_key)
            if isinstance(count, int) and isinstance(capacity, int) \
                    and count >= capacity:
                raise CycleBoundsError(
                    "%s exhausted" % code, code=code,
                    details={"count": count, "capacity": capacity})
        arena_used = status.get("arena_used")
        arena_bytes = status.get("arena_bytes")
        if isinstance(arena_used, int) and isinstance(arena_bytes, int):
            needed = (members * (VALUE_BYTES + 1) + 7) & ~7
            if arena_used + needed > arena_bytes:
                raise CycleBoundsError(
                    "hyperedge member arena exhausted",
                    code="arena_capacity",
                    details={"used": arena_used, "required": needed,
                             "capacity": arena_bytes})
        if leaf_needed:
            leaf_capacity = _host_attr(self.leaves, "capacity", None)
            leaf_count = _host_attr(self.leaves, "count", None)
            if isinstance(leaf_capacity, int) and isinstance(leaf_count, int) \
                    and leaf_count >= leaf_capacity:
                raise CycleBoundsError(
                    "cycle leaf capacity exhausted",
                    code="leaf_capacity",
                    details={"count": leaf_count,
                             "capacity": leaf_capacity})

    def _hyperedge_identifier(self, view: Any) -> bytes | None:
        value = _field_any(view, ("hyperedge_id", "id"))
        if value is None:
            return None
        return _bytes32(value, field="hyperedge_id")

    def _hyperedge_source(self, view: Any) -> bytes | None:
        value = _field_any(view, ("source_digest", "source"))
        if value is None:
            return None
        return _bytes32(value, field="source_digest")

    def _parse_cycle_view(self, view: Any) -> dict[str, Any] | None:
        """Extract cycle metadata from one hyperedge view."""
        if view is None:
            return None
        pairs: list[tuple[int, bytes]] = _member_pairs(view)
        if not pairs:
            return None
        cycle_leaf = None
        value = None
        payload_digest = None
        vertices: list[int] = []
        pair_refs: list[int] = []
        member_details: list[dict[str, Any]] = []
        for code, raw in pairs:
            if code == MEMBER_CYCLE:
                cycle_leaf = _leaf_id_from_member(raw)
            elif code == MEMBER_VALUE:
                value = _decode_value(raw)
            elif code == MEMBER_PAYLOAD:
                payload_digest = raw
            elif code == MEMBER_VERTEX:
                vertices.append(int.from_bytes(raw, "big"))
            elif code == MEMBER_PAIR:
                pair_refs.append(int.from_bytes(raw, "big"))
                member_details.append({
                    "kind": "pair",
                    "kind_code": MEMBER_PAIR,
                    "id": raw.hex(),
                    "id_int": int.from_bytes(raw, "big"),
                })
        if cycle_leaf is None:
            return None
        if len(vertices) == 3:
            try:
                cycle = normalize_cycle(tuple(vertices), self.nodes)
            except CycleHypergraphError:
                return None
        elif len(pair_refs) == 3:
            counts: dict[int, int] = {}
            for reference in pair_refs:
                for vertex in _pair_vertices(reference):
                    counts[vertex] = counts.get(vertex, 0) + 1
            common = [vertex for vertex, count in counts.items() if count == 2]
            if len(common) != 1:
                return None
            others = sorted(vertex for vertex, count in counts.items()
                            if count == 1)
            if len(others) != 2:
                return None
            cycle = normalize_cycle((common[0], others[0], others[1]),
                                    self.nodes)
        else:
            return None
        edges = cycle_edges(cycle, self.nodes)
        expected_refs = tuple(pair_reference(*edge) for edge in edges)
        # Pair members may be stored in any order by external writers.
        pair_refs_sorted = tuple(sorted(pair_refs))
        expected_sorted = tuple(sorted(expected_refs))
        hyperedge_id = self._hyperedge_identifier(view)
        if hyperedge_id is None:
            hyperedge_id = cycle_hyperedge_id(cycle, self.nodes)
        return {
            "cycle": cycle,
            "cycle_id": ":".join(str(vertex) for vertex in cycle),
            "cycle_leaf": cycle_leaf,
            "leaf_id": cycle_leaf.hex(),
            "value": value,
            "payload_digest": payload_digest,
            "source_digest": self._hyperedge_source(view),
            "hyperedge_id": hyperedge_id,
            "vertices": tuple(vertices),
            "pair_refs": tuple(pair_refs),
            "edges": edges,
            "expected_refs": expected_refs,
            "pair_refs_match": pair_refs_sorted == expected_sorted,
            "members": member_details,
        }

    def _find_cycle_view(self, cycle: tuple[int, int, int]) -> Any | None:
        direct = self._get_hyperedge(cycle_hyperedge_id(cycle, self.nodes))
        if direct is not None:
            try:
                parsed = self._parse_cycle_view(direct)
            except Exception:
                parsed = None
            if parsed is not None and parsed["cycle"] == cycle:
                return direct
        # Bounded fallback for stores that did not use our deterministic id.
        for view in self._iter_hyperedges(DEFAULT_AUDIT_SCAN):
            try:
                parsed = self._parse_cycle_view(view)
            except Exception:
                continue
            if parsed is not None and parsed["cycle"] == cycle:
                return view
        return None

    # -- leaf access ----------------------------------------------------
    def _find_leaf(self, leaf_id: bytes) -> Any | None:
        find = _host_attr(self.leaves, "find", None)
        if callable(find):
            for key in (leaf_id, leaf_id.hex()):
                try:
                    record = find(key)
                except Exception:
                    record = None
                if record is not None:
                    return record
        get = _host_attr(self.leaves, "get", None)
        if callable(get):
            try:
                return get(leaf_id)
            except Exception:
                try:
                    return get(leaf_id.hex())
                except Exception:
                    return None
        return None

    def _insert_leaf(self, leaf_id: bytes, payload_digest: bytes,
                     canonical_payload: bytes,
                     pair_index: int | None) -> Any:
        insert = _host_attr(self.leaves, "insert", None)
        if not callable(insert):
            raise CycleHypergraphError(
                "leaf store does not support insert", code="unsupported_store")
        kwargs: dict[str, Any] = {
            "leaf_id": leaf_id,
            "payload_digest": payload_digest,
            "content": canonical_payload,
            "head": 0,
        }
        if pair_index is not None:
            kwargs["pair_id"] = pair_index
        try:
            return insert("hyperedge", **kwargs)
        except TypeError:
            content = kwargs.pop("content", None)
            try:
                return insert("hyperedge", **kwargs)
            except TypeError:
                kwargs.pop("pair_id", None)
                try:
                    return insert("hyperedge", **kwargs)
                except TypeError:
                    if content is None:
                        raise
                    return insert("hyperedge", content, **kwargs)

    def _pair_index(self, a: int, b: int) -> int | None:
        pair_index = _host_attr(self.cells, "pair_index", None)
        if callable(pair_index):
            try:
                value = pair_index(a, b)
            except Exception:
                return None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    # -- public cycle operations ---------------------------------------
    def add_cycle(self, i: int, j: int, semantic_payload: Any,
                  source: Any = None, *, source_digest: Any = None
                  ) -> dict[str, Any]:
        """Attach one fundamental cycle ``(0, i, j)`` to the stores.

        ``semantic_payload`` must carry a raw integer (either directly or as
        the ``value`` field of a mapping).  ``source`` is required provenance.
        A repeated call with identical value and payload is idempotent; a
        different value or payload for the same canonical cycle raises
        :class:`CycleConflictError`.  A second provenance is reported as a
        duplicate with ``source_mismatch=True`` and never overwrites the
        original source.
        """
        first = _check_bounded_int(i, "i", 1, self.nodes - 1)
        second = _check_bounded_int(j, "j", 1, self.nodes - 1)
        if first == second:
            raise CycleHypergraphError(
                "cycle endpoints must be distinct", code="invalid_cycle",
                details={"i": first, "j": second})
        cycle = normalize_cycle((0, first, second), self.nodes)
        value, canonical_payload, payload_digest = \
            _payload_value_and_digest(semantic_payload)
        if source is None and source_digest is not None:
            source = source_digest
        source_bytes = _source_digest(source)
        leaf_id = cycle_leaf_id(cycle, self.nodes)
        hyperedge_id = cycle_hyperedge_id(cycle, self.nodes)

        existing_view = self._find_cycle_view(cycle)
        if existing_view is not None:
            existing = self._parse_cycle_view(existing_view)
            assert existing is not None
            existing_source = existing["source_digest"]
            existing_digest = existing["payload_digest"]
            differences = []
            if existing["value"] != value:
                differences.append("value")
            if existing_digest != payload_digest:
                differences.append("payload")
            if differences:
                raise CycleConflictError(
                    "cycle %s already names different immutable content"
                    % existing["cycle_id"],
                    code="cycle_conflict",
                    details={"cycle": existing["cycle_id"],
                             "fields": differences})
            # A second provenance for the same immutable observable is an
            # idempotent duplicate; the original source is never overwritten.
            # This mirrors RelationStore's duplicate semantics.
            receipt = self._receipt(
                existing, status="duplicate", created=False, duplicate=True,
                conflict=False, leaf=None)
            receipt["source_mismatch"] = existing_source != source_bytes
            return receipt

        leaf = self._find_leaf(leaf_id)
        if leaf is not None:
            existing_digest = _record_digest(leaf)
            if existing_digest is not None and existing_digest != payload_digest:
                raise CycleConflictError(
                    "cycle leaf %s already names a different payload"
                    % leaf_id.hex(),
                    code="cycle_payload_conflict",
                    details={"cycle": ":".join(str(v) for v in cycle)})
        edges = cycle_edges(cycle, self.nodes)
        pair_index = self._pair_index(*edges[0])
        if pair_index is None:
            try:
                pair_index = pair_reference(*edges[0])
            except Exception:
                pair_index = None
        self._check_store_capacity(leaf_needed=leaf is None)
        if leaf is None:
            try:
                leaf = self._insert_leaf(leaf_id, payload_digest,
                                         canonical_payload, pair_index)
            except CycleHypergraphError:
                raise
            except Exception as exc:
                raise CycleBoundsError(
                    "cycle leaf could not be appended: %s" % exc,
                    code="leaf_capacity",
                    details={"cycle": ":".join(str(v) for v in cycle)}) from exc

        members = self._cycle_members_for_storage(
            cycle, leaf_id, value, payload_digest)
        try:
            receipt = self._append_cycle_hyperedge(
                cycle, leaf_id, hyperedge_id, members, source_bytes)
        except CycleHypergraphError:
            raise
        except RelationConflictError as exc:
            raise CycleConflictError(
                "cycle hyperedge id already names different members",
                code="cycle_hyperedge_conflict",
                details={"cycle": ":".join(str(v) for v in cycle)}) from exc
        except RelationBoundsError as exc:
            raise CycleBoundsError(
                "cycle hyperedge capacity exhausted: %s" % exc,
                code="hyperedge_capacity",
                details={"cycle": ":".join(str(v) for v in cycle)}) from exc
        except Exception as exc:
            raise CycleHypergraphError(
                "cycle hyperedge could not be appended: %s" % exc,
                code="hyperedge_append_failed",
                details={"cycle": ":".join(str(v) for v in cycle)}) from exc
        view = self._find_cycle_view(cycle)
        if view is None:
            raise CycleCorruptionError(
                "cycle hyperedge was not persisted",
                code="missing_cycle_hyperedge",
                details={"cycle": ":".join(str(v) for v in cycle)})
        parsed = self._parse_cycle_view(view)
        assert parsed is not None
        if (parsed["value"] != value
                or parsed["payload_digest"] != payload_digest
                or parsed["source_digest"] != source_bytes):
            raise CycleCorruptionError(
                "persisted cycle hyperedge does not match the receipt",
                code="cycle_readback_mismatch",
                details={"cycle": parsed["cycle_id"]})
        self._maybe_commit("cycle", parsed["cycle_id"])
        out = self._receipt(
            parsed, status="appended", created=True, duplicate=False,
            conflict=False, leaf=leaf)
        out.update({key: value for key, value in receipt.items()
                    if key not in out})
        out["status"] = "appended"
        out["created"] = True
        return out

    add = add_cycle

    def _cycle_members_for_storage(
            self, cycle: tuple[int, int, int], leaf_id: bytes,
            value: int, payload_digest: bytes) -> list[tuple[int, bytes]]:
        a, b, c = cycle
        edges = cycle_edges(cycle, self.nodes)
        members: list[tuple[int, bytes]] = [
            (MEMBER_CYCLE, _bytes32(leaf_id, field="cycle leaf")),
        ]
        for edge in edges:
            members.append((MEMBER_PAIR, _bytes32(pair_reference(*edge),
                                                 field="pair reference")))
        for vertex in (a, b, c):
            members.append((MEMBER_VERTEX, _bytes32(vertex, field="vertex")))
        members.append((MEMBER_VALUE, _bytes32(value, signed=True,
                                               field="cycle value")))
        members.append((MEMBER_PAYLOAD, payload_digest))
        return members

    def _append_cycle_hyperedge(
            self, cycle: tuple[int, int, int], leaf_id: bytes,
            hyperedge_id: bytes, members: list[tuple[int, bytes]],
            source_digest: bytes) -> dict[str, Any]:
        append_edge = _host_attr(self.relations, "append_hyperedge", None)
        if callable(append_edge):
            try:
                result = append_edge(
                    hyperedge_id=hyperedge_id, source_digest=source_digest,
                    members=members)
            except TypeError:
                result = append_edge(hyperedge_id, source_digest, members)
            return dict(result) if isinstance(result, Mapping) else {}
        append_relation = _host_attr(self.relations, "append_relation", None)
        if not callable(append_relation):
            raise CycleHypergraphError(
                "relation store does not support hyperedge append",
                code="unsupported_store")
        edges = cycle_edges(cycle, self.nodes)
        pair_id = pair_reference(*edges[0])
        mirror_id = mirror_address(pair_address(*edges[2]))
        subject = _bytes32(leaf_id, field="subject")
        predicate = CYCLE_PREDICATE
        obj = members[-1][1]  # payload digest, stable for this cycle content
        try:
            result = append_relation(
                subject=subject, predicate=predicate, object=obj,
                polarity="supports", confidence=0.5,
                pair_id=pair_id, mirror_id=mirror_id,
                source_digest=source_digest,
                relation_id=cycle_relation_id(cycle, self.nodes),
                hyperedge_id=hyperedge_id, hyperedge_members=members)
        except TypeError:
            result = append_relation(
                subject=subject, predicate=predicate, object=obj,
                source_digest=source_digest,
                hyperedge_id=hyperedge_id, hyperedge_members=members)
        return dict(result) if isinstance(result, Mapping) else {}

    def _receipt(self, parsed: dict[str, Any], *, status: str, created: bool,
                 duplicate: bool, conflict: bool, leaf: Any) -> dict[str, Any]:
        leaf_id = parsed["leaf_id"]
        return {
            "status": status,
            "created": created,
            "duplicate": duplicate,
            "conflict": conflict,
            "cycle": parsed["cycle"],
            "cycle_id": parsed["cycle_id"],
            "value": parsed["value"],
            "source": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "source_digest": None if parsed["source_digest"] is None
            else parsed["source_digest"].hex(),
            "payload": None if parsed["payload_digest"] is None
            else parsed["payload_digest"].hex(),
            "payload_digest": None if parsed["payload_digest"] is None
            else parsed["payload_digest"].hex(),
            "hyperedge_id": None if parsed["hyperedge_id"] is None
            else parsed["hyperedge_id"].hex(),
            "leaf_id": leaf_id,
            "members": list(parsed["members"]),
            "generation": self._generation(),
            "canonical_merge": False,
        }

    def _generation(self) -> int | None:
        for owner in (self.host, self.relations):
            value = _host_attr(owner, "generation", None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        status = self._relation_status()
        value = status.get("generation")
        return value if isinstance(value, int) else None

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

    # -- public reads ---------------------------------------------------
    def cycle_members(self, cycle: Any) -> CycleMembers:
        """Return the bounded incidence members of one cycle observable.

        The returned list contains the three pair references.  Metadata such
        as the raw integer value, source and payload digest is attached both
        as attributes and through ``members["value"]``.  A missing cycle yields
        an empty list with ``missing=True`` rather than an unbounded scan.
        """
        canonical = normalize_cycle(cycle, self.nodes)
        view = self._find_cycle_view(canonical)
        if view is None:
            return CycleMembers(
                (), cycle=canonical,
                cycle_id=":".join(str(v) for v in canonical), value=None,
                source=None, source_digest=None, payload=None,
                payload_digest=None, hyperedge_id=None,
                leaf_id=None, vertices=canonical,
                edges=cycle_edges(canonical, self.nodes),
                pair_refs=(), missing=True)
        parsed = self._parse_cycle_view(view)
        assert parsed is not None
        source_hex = None if parsed["source_digest"] is None \
            else parsed["source_digest"].hex()
        payload_hex = None if parsed["payload_digest"] is None \
            else parsed["payload_digest"].hex()
        out = CycleMembers(
            (), cycle=parsed["cycle"], cycle_id=parsed["cycle_id"],
            value=parsed["value"], source=source_hex,
            source_digest=source_hex, payload=payload_hex,
            payload_digest=payload_hex,
            hyperedge_id=None if parsed["hyperedge_id"] is None
            else parsed["hyperedge_id"].hex(),
            leaf_id=parsed["leaf_id"], vertices=parsed["vertices"],
            edges=parsed["edges"], pair_refs=parsed["pair_refs"],
            missing=False)
        for position, (edge, reference) in enumerate(
                zip(parsed["edges"], parsed["expected_refs"])):
            a, b = edge
            index = self._pair_index(a, b)
            address = pair_address(a, b)
            out.append({
                "index": position,
                "kind": "pair",
                "kind_code": MEMBER_PAIR,
                "id": _bytes32(reference, field="pair reference").hex(),
                "id_int": reference,
                "address": address,
                "canonical_address": address,
                "mirror": mirror_address(address),
                "pair": (a, b),
                "pair_index": reference if index is None else index,
                "source": "cycle",
            })
        return out

    def _neighbor_candidates(self, cycle: tuple[int, int, int]
                             ) -> Iterator[tuple[int, int, int]]:
        _zero, first, second = cycle
        seen: set[tuple[int, int, int]] = set()
        for anchor in (first, second):
            for other in range(1, self.nodes):
                if other == first or other == second:
                    continue
                candidate = normalize_cycle((0, anchor, other), self.nodes)
                if candidate in seen or candidate == cycle:
                    continue
                seen.add(candidate)
                yield candidate

    def _cycle_exists(self, cycle: tuple[int, int, int],
                      *, scan: bool = False) -> bool:
        direct = self._get_hyperedge(cycle_hyperedge_id(cycle, self.nodes))
        if direct is not None:
            try:
                parsed = self._parse_cycle_view(direct)
            except Exception:
                parsed = None
            if parsed is not None and parsed["cycle"] == cycle:
                return True
        if not scan:
            return False
        return self._find_cycle_view(cycle) is not None

    def neighbors(self, cycle: Any, hops: int = 1,
                  limit: int = DEFAULT_SNAPSHOT_CYCLES) -> CycleNeighbors:
        """Bounded breadth-first traversal of cycles sharing a star edge.

        A fundamental triangle ``(0, i, j)`` shares an edge with
        ``(0, i, k)`` or ``(0, j, k)``.  No pair-cell or cycle records are
        duplicated; each candidate is checked against the deterministic
        resident id.
        """
        canonical = normalize_cycle(cycle, self.nodes)
        if hops is None:
            hops = 1
        hops = _check_bounded_int(hops, "hops", 1, MAX_NEIGHBOR_HOPS)
        limit = _bounded_limit(limit, DEFAULT_SNAPSHOT_CYCLES,
                               MAX_NEIGHBOR_LIMIT)
        # One bounded scan gives a local membership set for this traversal.
        # It is not retained after the call and is capped by DEFAULT_AUDIT_SCAN.
        known: set[tuple[int, int, int]] = set()
        for view in self._iter_hyperedges(DEFAULT_AUDIT_SCAN):
            try:
                parsed = self._parse_cycle_view(view)
            except Exception:
                continue
            if parsed is not None:
                known.add(parsed["cycle"])
        if canonical not in known and not self._cycle_exists(canonical, scan=True):
            return CycleNeighbors(
                (), cycle=canonical,
                cycle_id=":".join(str(v) for v in canonical), hops=hops,
                limit=limit, cycle_ids=(), truncated=False, missing=True)

        def exists(candidate: tuple[int, int, int]) -> bool:
            return candidate in known or self._cycle_exists(candidate)

        visited = {canonical}
        frontier = [canonical]
        results: list[tuple[int, int, int]] = []
        truncated = False
        for _depth in range(hops):
            if not frontier:
                break
            next_frontier: list[tuple[int, int, int]] = []
            for current in frontier:
                for candidate in self._neighbor_candidates(current):
                    if candidate in visited:
                        continue
                    if not exists(candidate):
                        continue
                    if len(results) >= limit:
                        truncated = True
                        break
                    visited.add(candidate)
                    results.append(candidate)
                    next_frontier.append(candidate)
                if truncated:
                    break
            if truncated:
                break
            frontier = next_frontier
        return CycleNeighbors(
            results, cycle=canonical,
            cycle_id=":".join(str(v) for v in canonical), hops=hops,
            limit=limit,
            cycle_ids=tuple(":".join(str(v) for v in item)
                            for item in results),
            truncated=truncated, missing=False)

    def compose(self, c1: Any, c2: Any, *,
                require_shared: bool = False) -> CycleComposition:
        """Compose two existing cycles as a raw integer + incidence union.

        The result value is always ``value(c1) + value(c2)``.  When the two
        observables share exactly one incidence edge, the returned cycle is the
        ordered boundary of the symmetric edge difference (a 4-cycle for two
        fundamental triangles sharing a star edge).  Non-adjacent cycles still
        compose additively; their result is a bounded union with
        ``status="union"``.  Pass ``require_shared=True`` to deny that case.

        Composition is a pure read: it never creates a canonical merged
        record and never changes either source observable.
        """
        first = normalize_cycle(c1, self.nodes)
        second = normalize_cycle(c2, self.nodes)
        left = self.cycle_members(first)
        right = self.cycle_members(second)
        if left.missing:
            raise CycleNotFound(
                "cycle %s does not exist" % left.cycle_id,
                code="cycle_not_found", details={"cycle": left.cycle_id})
        if right.missing:
            raise CycleNotFound(
                "cycle %s does not exist" % right.cycle_id,
                code="cycle_not_found", details={"cycle": right.cycle_id})
        left_refs = set(left.pair_refs)
        right_refs = set(right.pair_refs)
        shared = tuple(sorted(left_refs & right_refs))
        if first == second:
            return CycleComposition(
                int(left.value or 0) + int(right.value or 0),
                status="idempotent", cycle=first,
                cycles=(first, second), shared=shared,
                members=list(left.pair_refs))
        if len(shared) != 1:
            if require_shared:
                raise CycleDenied(
                    "cycle composition requires exactly one shared edge",
                    code="not_composable",
                    details={"c1": left.cycle_id, "c2": right.cycle_id,
                             "shared_edges": len(shared)})
            union_vertices = tuple(sorted(set(first) | set(second)))
            return CycleComposition(
                int(left.value or 0) + int(right.value or 0),
                status="union", cycle=union_vertices,
                cycles=(first, second), shared=shared,
                members=tuple(sorted(left_refs ^ right_refs)))
        remaining = (left_refs ^ right_refs)
        adjacency: dict[int, list[int]] = {}
        for reference in remaining:
            a, b = _pair_vertices(reference)
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if not adjacency or any(len(items) != 2
                                for items in adjacency.values()):
            raise CycleDenied(
                "symmetric difference is not a simple cycle",
                code="not_simple_cycle")
        start = 0 if 0 in adjacency else min(adjacency)
        ordered = [start]
        previous = None
        current = start
        while True:
            options = sorted(vertex for vertex in adjacency[current]
                             if vertex != previous)
            if not options:
                raise CycleDenied(
                    "symmetric difference is not a closed cycle",
                    code="not_simple_cycle")
            nxt = options[0]
            if nxt == start:
                break
            if nxt in ordered:
                raise CycleDenied(
                    "symmetric difference repeats a vertex",
                    code="not_simple_cycle")
            ordered.append(nxt)
            previous, current = current, nxt
            if len(ordered) > self.nodes:
                raise CycleDenied("cycle traversal exceeded the node bound",
                                  code="not_simple_cycle")
        value = int(left.value or 0) + int(right.value or 0)
        members = [pair_reference(ordered[index], ordered[(index + 1)
                                                          % len(ordered)])
                   for index in range(len(ordered))]
        return CycleComposition(
            value, status="composed", cycle=tuple(ordered),
            cycles=(first, second), shared=shared, members=members)

    # -- introspection --------------------------------------------------
    def snapshot(self, limit: int = DEFAULT_SNAPSHOT_CYCLES) -> dict[str, Any]:
        """Return a bounded, JSON-friendly projection of stored cycles."""
        limit = _bounded_limit(limit, DEFAULT_SNAPSHOT_CYCLES,
                               MAX_SNAPSHOT_LIMIT)
        entries: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        seen: set[tuple[int, int, int]] = set()
        for view in self._iter_hyperedges(DEFAULT_AUDIT_SCAN):
            scanned += 1
            try:
                parsed = self._parse_cycle_view(view)
            except Exception:
                continue
            if parsed is None:
                continue
            cycle = parsed["cycle"]
            if cycle in seen:
                continue
            seen.add(cycle)
            if len(entries) >= limit:
                truncated = True
                break
            entries.append({
                "cycle": parsed["cycle"],
                "cycle_id": parsed["cycle_id"],
                "value": parsed["value"],
                "source": None if parsed["source_digest"] is None
                else parsed["source_digest"].hex(),
                "source_digest": None if parsed["source_digest"] is None
                else parsed["source_digest"].hex(),
                "payload": None if parsed["payload_digest"] is None
                else parsed["payload_digest"].hex(),
                "payload_digest": None if parsed["payload_digest"] is None
                else parsed["payload_digest"].hex(),
                "hyperedge_id": None if parsed["hyperedge_id"] is None
                else parsed["hyperedge_id"].hex(),
                "leaf_id": parsed["leaf_id"],
                "edges": parsed["edges"],
                "pair_refs": parsed["pair_refs"],
            })
        hyperedges = self._hyperedge_count()
        if hyperedges is not None and scanned < hyperedges:
            truncated = True
        return {
            "schema": SCHEMA,
            "cycles": entries,
            "cycle_count": len(entries),
            "count": len(entries),
            "returned": len(entries),
            "limit": limit,
            "scanned_hyperedges": scanned,
            "hyperedge_count": hyperedges,
            "leaf_count": self._leaf_count(),
            "truncated": truncated,
            "bounded": True,
            "canonical_merge": False,
            "generation": self._generation(),
            "nodes": self.nodes,
        }

    status = snapshot
    stats = snapshot

    def audit(self, limit: int = DEFAULT_AUDIT_SCAN) -> dict[str, Any]:
        """Check bounded cycle incidence, provenance and deterministic ids.

        The audit is read-only.  It scans at most ``limit`` hyperedges and
        reports ``ok=False`` with concrete problems for dangling leaves,
        mismatched pair references, missing provenance or non-deterministic
        ids.
        """
        limit = _bounded_limit(limit, DEFAULT_AUDIT_SCAN, DEFAULT_AUDIT_SCAN)
        problems: list[str] = []
        checked = 0
        scanned = 0
        seen: set[tuple[int, int, int]] = set()
        for view in self._iter_hyperedges(limit):
            scanned += 1
            try:
                parsed = self._parse_cycle_view(view)
            except Exception as exc:
                problems.append("unreadable hyperedge at scan %d: %s"
                                % (scanned, exc))
                continue
            if parsed is None:
                continue
            checked += 1
            cycle = parsed["cycle"]
            label = parsed["cycle_id"]
            if cycle in seen:
                problems.append("duplicate cycle %s" % label)
            seen.add(cycle)
            expected_hyperedge = cycle_hyperedge_id(cycle, self.nodes)
            if parsed["hyperedge_id"] != expected_hyperedge:
                problems.append("non-deterministic hyperedge id for %s"
                                % label)
            expected_leaf = cycle_leaf_id(cycle, self.nodes)
            if parsed["cycle_leaf"] != expected_leaf:
                problems.append("non-deterministic leaf id for %s" % label)
            if parsed["source_digest"] in (None, ZERO_ID):
                problems.append("missing provenance for %s" % label)
            if parsed["payload_digest"] is None:
                problems.append("missing payload digest for %s" % label)
            if parsed["value"] is None:
                problems.append("missing raw integer value for %s" % label)
            if not parsed["pair_refs_match"]:
                problems.append("pair references do not match incidence for %s"
                                % label)
            leaf = self._find_leaf(expected_leaf)
            if leaf is None:
                problems.append("dangling cycle leaf for %s" % label)
            else:
                digest = _record_digest(leaf)
                if digest is not None and parsed["payload_digest"] is not None \
                        and digest != parsed["payload_digest"]:
                    problems.append("leaf payload digest mismatch for %s"
                                    % label)
        hyperedges = self._hyperedge_count()
        truncated = hyperedges is not None and scanned < hyperedges
        status = self._relation_status()
        capacity_problems = []
        for count_key, capacity_key in (
                ("hyperedges", "hyperedge_capacity"),
                ("relations", "relation_capacity")):
            count = status.get(count_key)
            capacity = status.get(capacity_key)
            if isinstance(count, int) and isinstance(capacity, int) \
                    and count > capacity:
                capacity_problems.append(
                    "%s exceeds %s" % (count_key, capacity_key))
        problems.extend(capacity_problems)
        return {
            "schema": SCHEMA,
            "ok": not problems,
            "problems": problems,
            "issues": list(problems),
            "cycles": checked,
            "cycle_count": checked,
            "cycles_checked": checked,
            "scanned_hyperedges": scanned,
            "hyperedge_count": hyperedges,
            "leaf_count": self._leaf_count(),
            "truncated": truncated,
            "bounded": True,
            "limit": limit,
            "canonical_merge": False,
            "nodes": self.nodes,
            "stores": {
                "relations": type(self.relations).__name__,
                "leaves": type(self.leaves).__name__,
                "cells": None if self.cells is None
                else type(self.cells).__name__,
            },
        }

    def cycle_reference(self, cycle: Any) -> dict[str, Any]:
        """Return a bounded reference that can be embedded in a parent edge."""
        canonical = normalize_cycle(cycle, self.nodes)
        leaf = cycle_leaf_id(canonical, self.nodes)
        edge = cycle_hyperedge_id(canonical, self.nodes)
        return {
            # An integer kind is directly accepted by RelationStore's member
            # normalizer, so this reference can be embedded in a parent edge.
            "kind": MEMBER_CYCLE,
            "kind_name": "cycle",
            "kind_code": MEMBER_CYCLE,
            "id": _bytes32(leaf, field="cycle leaf").hex(),
            "leaf_id": leaf.hex(),
            "hyperedge_id": edge.hex(),
            "cycle": canonical,
            "cycle_id": ":".join(str(vertex) for vertex in canonical),
        }

    cycle_ref = cycle_reference

    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Canonical merge is deliberately disabled in this layer."""
        raise CycleDenied(
            "canonical cycle merge is disabled",
            code="canonical_merge_disabled",
            details={"canonical_merge": False})

    promote = merge

    # -- reopen ---------------------------------------------------------
    def reopen(self) -> "CycleHypergraph":
        """Return a fresh adapter over the same stores or reopened universe.

        If the host exposes a zero-argument ``reopen`` hook it is called first.
        Otherwise the same store objects are re-attached, which exercises the
        deterministic-id path after a process or checkpoint restart.
        """
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
            return CycleHypergraph(host, nodes=self.nodes)
        return CycleHypergraph(
            host if host is not None else self.relations,
            relations=self.relations, leaves=self.leaves, cells=self.cells,
            nodes=self.nodes)


__all__ = (
    "SCHEMA", "DEFAULT_NODES", "DEFAULT_SNAPSHOT_CYCLES",
    "DEFAULT_AUDIT_SCAN", "MAX_NEIGHBOR_HOPS", "MAX_NEIGHBOR_LIMIT",
    "MAX_SNAPSHOT_LIMIT", "DEFAULT_CAPACITY", "MEMBER_PAIR", "MEMBER_CYCLE",
    "MEMBER_VALUE", "MEMBER_VERTEX", "MEMBER_PAYLOAD", "MEMBER_NAMES",
    "MEMBER_CODES", "CycleHypergraphError", "CycleBoundsError",
    "CycleProvenanceError", "CycleConflictError", "CycleCorruptionError",
    "CycleNotFound", "CycleDenied", "CycleMembers", "CycleNeighbors",
    "CycleComposition", "normalize_cycle", "cycle_id", "cycle_leaf_id",
    "cycle_hyperedge_id", "cycle_relation_id", "cycle_edges",
    "pair_reference", "CycleHypergraph",
)
