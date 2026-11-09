"""Append-only resident relation and hyperedge records.

The store is a small fixed-layout section inside a caller-owned mutable buffer
(``bytearray`` or writable ``mmap``), rooted at ``base_offset``.  It never keeps
Python dictionaries as authoritative state: record bodies, chain heads,
content-addressed indexes, counters, generation and the hyperedge member arena
all live in the buffer.  Returned dictionaries are projections of bytes read
back from the buffer.

Section layout (all offsets relative to ``base_offset``)::

    +--------------------------------------+ 0
    | header (104 bytes)                   |
    +--------------------------------------+ 8-byte aligned
    | leaf chain-head hash table           |
    | relation_id -> record index table    |
    | hyperedge_id -> record index table   |
    | fixed relation records               |
    | fixed hyperedge records              |
    | hyperedge member/kind arena          |
    +--------------------------------------+

Relation record (304 bytes, little-endian)::

    offset  size  field
    0       32    relation_id (SHA-256)
    32      32    subject leaf id
    64      32    predicate leaf id
    96      32    object leaf id
    128     32    pair id (zero = absent)
    160     32    mirror id (zero = absent)
    192     32    hyperedge id (zero = absent)
    224     32    source digest
    256     8     next relation index for the subject leaf (index+1, 0 = end)
    264     8     next relation index for the predicate leaf
    272     8     next relation index for the object leaf
    280     8     confidence (IEEE-754 float64)
    288     8     generation
    296     1     polarity (-1 refutes, 0 neutral, +1 supports)
    297     1     flags
    298     2     reserved
    300     4     padding

Hyperedge record (96 bytes, little-endian)::

    offset  size  field
    0       32    hyperedge_id (SHA-256)
    32      32    source digest
    64      8     members offset in arena
    72      8     member-kind bytes offset in arena
    80      8     generation
    88      2     arity
    90      2     flags
    92      1     common member kind (0 = mixed)
    93      1     reserved
    94      2     reserved

Arena entries are ``id[32] * arity`` followed by ``kind[1] * arity``.  The
next allocation is 8-byte aligned.  Existing records are never rewritten;
only append cursors, chain heads and index buckets change.  Re-appending an
existing ``relation_id`` returns the existing record and performs no write.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from . import ALIGN, SECTION_HYPEREDGES, SECTION_RELATIONS

SCHEMA = "nanlm.resident-relations/1"
RELATION_MAGIC = b"NLMRREL1"
HYPEREDGE_MAGIC = b"NLMRHYP1"
VERSION = 1

ID_BYTES = 32
ZERO_ID = b"\x00" * ID_BYTES
INDEX_SLOT = struct.Struct("<32sQ")
CHAIN_SLOT = struct.Struct("<32sQQQ8x")
RELATION_RECORD = struct.Struct("<32s32s32s32s32s32s32s32sQQQdQbBH4x")
HYPEREDGE_RECORD = struct.Struct("<32s32sQQQHHBBH")
HEADER = struct.Struct("<8sIIQQQQQQQQQQQ")

# Header field positions in the HEADER tuple.
H_MAGIC = 0
H_VERSION = 1
H_FLAGS = 2
H_GENERATION = 3
H_RELATION_COUNT = 4
H_HYPEREDGE_COUNT = 5
H_CHAIN_USED = 6
H_RELATION_CAPACITY = 7
H_HYPEREDGE_CAPACITY = 8
H_CHAIN_CAPACITY = 9
H_RELATION_INDEX_CAPACITY = 10
H_HYPEREDGE_INDEX_CAPACITY = 11
H_ARENA_BYTES = 12
H_ARENA_USED = 13

# Byte offsets of scalar header fields (used for in-place counter updates).
_HEADER_GENERATION_OFFSET = 16
_HEADER_RELATION_COUNT_OFFSET = 24
_HEADER_HYPEREDGE_COUNT_OFFSET = 32
_HEADER_CHAIN_USED_OFFSET = 40
_HEADER_ARENA_USED_OFFSET = 96

RELATION_RECORD_BYTES = RELATION_RECORD.size          # 304
HYPEREDGE_RECORD_BYTES = HYPEREDGE_RECORD.size        # 96
INDEX_SLOT_BYTES = INDEX_SLOT.size                    # 40
CHAIN_SLOT_BYTES = CHAIN_SLOT.size                    # 64
HEADER_BYTES = HEADER.size                            # 104

DEFAULT_RELATION_CAPACITY = 65_536
DEFAULT_HYPEREDGE_CAPACITY = 65_536
DEFAULT_WALK_LIMIT = 64
MAX_WALK_LIMIT = 1_000_000

POLARITY_TO_CODE = {"refutes": -1, "neutral": 0, "supports": 1}
CODE_TO_POLARITY = {code: name for name, code in POLARITY_TO_CODE.items()}

MEMBER_LEAF = 1
MEMBER_RELATION = 2
MEMBER_PAIR = 3
MEMBER_MIRROR = 4
MEMBER_KIND_TO_CODE = {
    "leaf": MEMBER_LEAF,
    "relation": MEMBER_RELATION,
    "pair": MEMBER_PAIR,
    "mirror": MEMBER_MIRROR,
}
MEMBER_CODE_TO_KIND = {code: name for name, code in MEMBER_KIND_TO_CODE.items()}

_ROLE_SUBJECT = 0
_ROLE_PREDICATE = 1
_ROLE_OBJECT = 2
_ROLE_TO_NAME = {_ROLE_SUBJECT: "subject", _ROLE_PREDICATE: "predicate",
                 _ROLE_OBJECT: "object"}
_ROLE_FROM_NAME = {name: role for role, name in _ROLE_TO_NAME.items()}

_RELATION_ID_OFFSETS = (
    ("relation_id", 0, 32), ("subject", 32, 32), ("predicate", 64, 32),
    ("object", 96, 32), ("pair_id", 128, 32), ("mirror_id", 160, 32),
    ("hyperedge_id", 192, 32), ("source_digest", 224, 32),
    ("next_subject", 256, 8), ("next_predicate", 264, 8),
    ("next_object", 272, 8), ("confidence", 280, 8),
    ("generation", 288, 8), ("polarity", 296, 1), ("flags", 297, 1),
    ("reserved", 298, 2), ("padding", 300, 4),
)
_HYPEREDGE_ID_OFFSETS = (
    ("hyperedge_id", 0, 32), ("source_digest", 32, 32),
    ("members_offset", 64, 8), ("kinds_offset", 72, 8),
    ("generation", 80, 8), ("arity", 88, 2), ("flags", 90, 2),
    ("member_kind", 92, 1), ("reserved", 93, 1),
    ("reserved2", 94, 2),
)


class RelationError(ValueError):
    """Base error for resident relation/hyperedge operations."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


class RelationBoundsError(RelationError):
    """Raised when a record, arena, chain/index table or walk bound is hit."""


class RelationConflictError(RelationError):
    """Raised when an explicit id already names different immutable content."""


class RelationCorruptionError(RelationError):
    """Raised when persisted links or indexes fail structural validation."""


def relation_layout() -> tuple[dict[str, Any], ...]:
    """Return the documented relation field offsets and sizes."""
    return tuple({"field": name, "offset": offset, "size": size}
                 for name, offset, size in _RELATION_ID_OFFSETS)


def hyperedge_layout() -> tuple[dict[str, Any], ...]:
    """Return the documented hyperedge field offsets and sizes."""
    return tuple({"field": name, "offset": offset, "size": size}
                 for name, offset, size in _HYPEREDGE_ID_OFFSETS)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(value: Any) -> bytes:
    return hashlib.sha256(_canonical(value)).digest()


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _next_pow2(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _check_uint(value: Any, label: str, bits: int = 64) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RelationError("%s must be an integer" % label)
    if value < 0 or value >= (1 << bits):
        raise RelationBoundsError(
            "%s is outside [0, 2**%d)" % (label, bits),
            code="integer_out_of_bounds",
            details={"field": label, "value": value, "bits": bits})
    return value


def _id_bytes(value: Any, *, field: str = "id",
              optional: bool = False) -> bytes:
    """Normalize an external id to a 32-byte resident id.

    * 32-byte values are used verbatim;
    * shorter bytes/ints are left-padded (numeric handles stay stable);
    * 64-character hex strings are decoded;
    * every other string is SHA-256 hashed so existing ``leaf:...`` ids and
      other namespaced identifiers remain content-addressed.
    """
    if value is None:
        if optional:
            return ZERO_ID
        raise RelationError("%s is required" % field)
    if isinstance(value, bool):
        raise RelationError("%s must not be a boolean" % field)
    if isinstance(value, int):
        if value < 0 or value >= (1 << (ID_BYTES * 8)):
            raise RelationBoundsError(
                "%s is outside the resident id range" % field,
                code="id_out_of_bounds", details={"field": field})
        return value.to_bytes(ID_BYTES, "big")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise RelationError("%s must not be empty" % field)
        if len(raw) == ID_BYTES:
            return raw
        if len(raw) < ID_BYTES:
            return raw.rjust(ID_BYTES, b"\x00")
        return hashlib.sha256(raw).digest()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RelationError("%s must not be empty" % field)
        if len(text) == ID_BYTES * 2:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        return hashlib.sha256(text.encode("utf-8")).digest()
    raise RelationError("%s must be a string, integer or bytes" % field)


def _digest_bytes(value: Any, *, field: str = "source_digest") -> bytes:
    if value is None:
        raise RelationError("%s is required" % field)
    return _id_bytes(value, field=field)


def _polarity_code(value: Any) -> int:
    if isinstance(value, bool):
        raise RelationError("polarity must not be a boolean")
    if isinstance(value, str):
        name = value.strip().lower()
        if name not in POLARITY_TO_CODE:
            raise RelationError(
                "polarity must be one of %s" % sorted(POLARITY_TO_CODE))
        return POLARITY_TO_CODE[name]
    if isinstance(value, int):
        if not -128 <= value <= 127:
            raise RelationBoundsError("polarity is outside int8",
                                      code="polarity_out_of_bounds")
        return value
    raise RelationError("polarity must be a string or integer")


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelationError("confidence must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RelationError("confidence must be finite and in [0, 1]")
    return result


def _flags(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise RelationError("flags must be an integer in [0, 255]")
    return value


def _member_kind(value: Any) -> int:
    if isinstance(value, bool):
        raise RelationError("member kind must not be a boolean")
    if isinstance(value, str):
        name = value.strip().lower()
        if name not in MEMBER_KIND_TO_CODE:
            raise RelationError(
                "member kind must be one of %s" % sorted(MEMBER_KIND_TO_CODE))
        return MEMBER_KIND_TO_CODE[name]
    if isinstance(value, int) and 0 <= value <= 0xFF:
        return value
    raise RelationError("member kind must be a known name or byte")


def _normalize_member(value: Any) -> tuple[int, bytes]:
    if isinstance(value, dict):
        if "id" not in value:
            raise RelationError("hyperedge member object requires an id")
        return (_member_kind(value.get("kind", "leaf")),
                _id_bytes(value["id"], field="member id"))
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise RelationError("hyperedge member tuple must be (kind, id)")
        return (_member_kind(value[0]),
                _id_bytes(value[1], field="member id"))
    return MEMBER_LEAF, _id_bytes(value, field="member id")


def _normalize_members(members: Iterable[Any] | None) -> tuple[tuple[int, bytes], ...]:
    if members is None:
        return ()
    if isinstance(members, (str, bytes, bytearray, memoryview, dict)):
        raise RelationError("hyperedge members must be an iterable of members")
    out = tuple(_normalize_member(member) for member in members)
    if not out:
        raise RelationError("hyperedge must contain at least one member")
    if len(out) > 0xFFFF:
        raise RelationBoundsError("hyperedge arity exceeds uint16",
                                  code="arity_out_of_bounds")
    return out


def hyperedge_id_for_members(members: Iterable[Any]) -> str:
    """Deterministic hyperedge id for an ordered member sequence."""
    normalized = _normalize_members(members)
    hasher = hashlib.sha256(b"NLMR-HYPEREDGE\0")
    for kind, member_id in normalized:
        hasher.update(bytes((kind,)))
        hasher.update(member_id)
    return hasher.digest().hex()


def relation_id_for(
    *, subject: Any, predicate: Any, object: Any,
    polarity: Any = "neutral", confidence: Any = 0.5,
    pair_id: Any = None, mirror_id: Any = None, flags: Any = 0,
) -> str:
    """Deterministic id for a relation's immutable semantic content."""
    payload = {
        "object": _id_bytes(object, field="object").hex(),
        "pair_id": _id_bytes(pair_id, field="pair_id", optional=True).hex(),
        "mirror_id": _id_bytes(mirror_id, field="mirror_id", optional=True).hex(),
        "polarity": _polarity_code(polarity),
        "predicate": _id_bytes(predicate, field="predicate").hex(),
        "flags": _flags(flags),
        # float.hex round-trips every finite binary64 value exactly.
        "confidence": float(_confidence(confidence)).hex(),
        "subject": _id_bytes(subject, field="subject").hex(),
    }
    return _sha256(payload).hex()


@dataclass(frozen=True)
class _Layout:
    chain_table: int
    relation_index: int
    hyperedge_index: int
    relation_records: int
    hyperedge_records: int
    arena: int
    end: int


@dataclass(frozen=True)
class _Capacities:
    relation: int
    hyperedge: int
    chain: int
    relation_index: int
    hyperedge_index: int
    arena: int


def _resolve_capacities(
    relation_capacity: int,
    hyperedge_capacity: int,
    chain_capacity: int | None,
    arena_bytes: int | None,
) -> _Capacities:
    for value, label in ((relation_capacity, "relation_capacity"),
                         (hyperedge_capacity, "hyperedge_capacity")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RelationError("%s must be a positive integer" % label)
    relation_index = _next_pow2(max(8, relation_capacity * 2))
    hyperedge_index = _next_pow2(max(8, hyperedge_capacity * 2))
    if chain_capacity is None:
        # Worst case: every relation introduces three new leaf keys, with a
        # load factor below 0.5.
        chain_capacity = _next_pow2(max(16, relation_capacity * 6 + 1))
    if (isinstance(chain_capacity, bool) or not isinstance(chain_capacity, int)
            or chain_capacity <= 0
            or chain_capacity & (chain_capacity - 1)):
        raise RelationError("chain_capacity must be a positive power of two")
    if arena_bytes is None:
        # Six default members: subject, predicate, object, pair, mirror,
        # relation.  A byte of member kind accompanies each 32-byte id.
        arena_bytes = _align8(max(4096, hyperedge_capacity * 6 * (ID_BYTES + 1)))
    if isinstance(arena_bytes, bool) or not isinstance(arena_bytes, int) or arena_bytes <= 0:
        raise RelationError("arena_bytes must be a positive integer")
    return _Capacities(relation_capacity, hyperedge_capacity, chain_capacity,
                       relation_index, hyperedge_index, arena_bytes)


def _compute_layout(base_offset: int, capacities: _Capacities) -> _Layout:
    cursor = _align8(base_offset + HEADER_BYTES)
    chain_table = cursor
    cursor += capacities.chain * CHAIN_SLOT_BYTES
    relation_index = cursor
    cursor += capacities.relation_index * INDEX_SLOT_BYTES
    hyperedge_index = cursor
    cursor += capacities.hyperedge_index * INDEX_SLOT_BYTES
    relation_records = cursor
    cursor += capacities.relation * RELATION_RECORD_BYTES
    hyperedge_records = cursor
    cursor += capacities.hyperedge * HYPEREDGE_RECORD_BYTES
    arena = cursor
    cursor += capacities.arena
    return _Layout(chain_table, relation_index, hyperedge_index,
                   relation_records, hyperedge_records, arena, cursor)


def _hash_slot(key: bytes, capacity: int, person: bytes) -> int:
    digest = hashlib.blake2b(key, digest_size=8, person=person).digest()
    return int.from_bytes(digest, "little") & (capacity - 1)


class RelationStore:
    """Append-only relation/hyperedge section over a mutable buffer."""

    def __init__(self, buffer: Any, base_offset: int = 0):
        if isinstance(base_offset, bool) or not isinstance(base_offset, int) or base_offset < 0:
            raise RelationBoundsError("base_offset must be a non-negative integer",
                                      code="invalid_base_offset")
        try:
            view = memoryview(buffer)
        except TypeError as exc:
            raise RelationError("buffer must support the buffer protocol") from exc
        try:
            view = view.cast("B")
        except TypeError as exc:
            raise RelationError("buffer must be a contiguous byte buffer") from exc
        if view.readonly:
            raise RelationError("buffer must be writable")
        if base_offset + HEADER_BYTES > len(view):
            raise RelationBoundsError(
                "buffer is too small for the relation section header",
                code="buffer_too_small",
                details={"base_offset": base_offset, "buffer_bytes": len(view)})
        self._buffer = buffer
        self._view = view
        self._base = base_offset
        self._lock = threading.RLock()
        header = self._header()
        if header[H_MAGIC] != RELATION_MAGIC:
            raise RelationCorruptionError("resident relation magic mismatch",
                                          code="bad_magic")
        if header[H_VERSION] != VERSION:
            raise RelationCorruptionError("unsupported resident relation version",
                                          code="bad_version")
        capacities = _Capacities(
            _check_uint(header[H_RELATION_CAPACITY], "relation_capacity"),
            _check_uint(header[H_HYPEREDGE_CAPACITY], "hyperedge_capacity"),
            _check_uint(header[H_CHAIN_CAPACITY], "chain_capacity"),
            _check_uint(header[H_RELATION_INDEX_CAPACITY], "relation_index_capacity"),
            _check_uint(header[H_HYPEREDGE_INDEX_CAPACITY], "hyperedge_index_capacity"),
            _check_uint(header[H_ARENA_BYTES], "arena_bytes"),
        )
        if not capacities.chain or capacities.chain & (capacities.chain - 1):
            raise RelationCorruptionError("chain capacity is not a power of two",
                                          code="bad_capacity")
        layout = _compute_layout(base_offset, capacities)
        if layout.end > len(view):
            raise RelationBoundsError(
                "buffer is too small for the persisted relation section",
                code="buffer_too_small",
                details={"required": layout.end, "buffer_bytes": len(view)})
        if (header[H_RELATION_COUNT] > capacities.relation
                or header[H_HYPEREDGE_COUNT] > capacities.hyperedge
                or header[H_CHAIN_USED] > capacities.chain
                or header[H_ARENA_USED] > capacities.arena):
            raise RelationCorruptionError("header counters exceed capacities",
                                          code="bad_counters")
        self._capacities = capacities
        self._layout = layout

    # -- construction ---------------------------------------------------
    @classmethod
    def required_bytes(
        cls, *, relation_capacity: int = DEFAULT_RELATION_CAPACITY,
        hyperedge_capacity: int = DEFAULT_HYPEREDGE_CAPACITY,
        chain_capacity: int | None = None,
        arena_bytes: int | None = None, base_offset: int = 0,
    ) -> int:
        if isinstance(base_offset, bool) or not isinstance(base_offset, int) or base_offset < 0:
            raise RelationBoundsError("base_offset must be a non-negative integer",
                                      code="invalid_base_offset")
        capacities = _resolve_capacities(relation_capacity, hyperedge_capacity,
                                         chain_capacity, arena_bytes)
        return _compute_layout(base_offset, capacities).end

    @classmethod
    def create(
        cls, buffer: Any, *, base_offset: int = 0,
        relation_capacity: int = DEFAULT_RELATION_CAPACITY,
        hyperedge_capacity: int = DEFAULT_HYPEREDGE_CAPACITY,
        chain_capacity: int | None = None,
        arena_bytes: int | None = None,
    ) -> "RelationStore":
        """Initialize a fresh section in ``buffer`` and return its store."""
        if isinstance(base_offset, bool) or not isinstance(base_offset, int) or base_offset < 0:
            raise RelationBoundsError("base_offset must be a non-negative integer",
                                      code="invalid_base_offset")
        capacities = _resolve_capacities(relation_capacity, hyperedge_capacity,
                                         chain_capacity, arena_bytes)
        layout = _compute_layout(base_offset, capacities)
        try:
            view = memoryview(buffer).cast("B")
        except TypeError as exc:
            raise RelationError("buffer must be a contiguous byte buffer") from exc
        if view.readonly:
            raise RelationError("buffer must be writable")
        if layout.end > len(view):
            raise RelationBoundsError(
                "buffer is too small for requested relation capacities",
                code="buffer_too_small",
                details={"required": layout.end, "buffer_bytes": len(view)})
        # Only the new section is zeroed; bytes outside [base, end) are left
        # untouched so several resident sections can share one buffer.
        view[base_offset:layout.end] = b"\x00" * (layout.end - base_offset)
        HEADER.pack_into(
            view, base_offset, RELATION_MAGIC, VERSION, 0, 0, 0, 0, 0,
            capacities.relation, capacities.hyperedge, capacities.chain,
            capacities.relation_index, capacities.hyperedge_index,
            capacities.arena, 0)
        return cls(buffer, base_offset=base_offset)

    @classmethod
    def attach(cls, buffer: Any, *, base_offset: int = 0) -> "RelationStore":
        return cls(buffer, base_offset=base_offset)

    @classmethod
    def allocate(
        cls, *, relation_capacity: int = DEFAULT_RELATION_CAPACITY,
        hyperedge_capacity: int = DEFAULT_HYPEREDGE_CAPACITY,
        chain_capacity: int | None = None,
        arena_bytes: int | None = None, base_offset: int = 0,
    ) -> "RelationStore":
        """Create a store in a newly allocated ``bytearray``."""
        size = cls.required_bytes(
            relation_capacity=relation_capacity,
            hyperedge_capacity=hyperedge_capacity,
            chain_capacity=chain_capacity, arena_bytes=arena_bytes,
            base_offset=base_offset)
        buffer = bytearray(size)
        return cls.create(
            buffer, base_offset=base_offset,
            relation_capacity=relation_capacity,
            hyperedge_capacity=hyperedge_capacity,
            chain_capacity=chain_capacity, arena_bytes=arena_bytes)

    # -- low-level buffer helpers --------------------------------------
    def _header(self) -> tuple[Any, ...]:
        return HEADER.unpack_from(self._view, self._base)

    def _write_u64(self, relative_offset: int, value: int) -> None:
        struct.pack_into("<Q", self._view, self._base + relative_offset,
                         _check_uint(value, "header counter"))

    def _zero_id_at(self, offset: int) -> bool:
        return self._view[offset:offset + ID_BYTES] == ZERO_ID

    def _index_find(self, table_offset: int, capacity: int, key: bytes) -> int | None:
        index = _hash_slot(key, capacity, b"NLMRIDX1")
        for _ in range(capacity):
            slot = table_offset + index * INDEX_SLOT_BYTES
            slot_key = bytes(self._view[slot:slot + ID_BYTES])
            if slot_key == ZERO_ID:
                return None
            if slot_key == key:
                return slot
            index = (index + 1) & (capacity - 1)
        return None

    def _index_insert_slot(self, table_offset: int, capacity: int,
                           key: bytes) -> int:
        index = _hash_slot(key, capacity, b"NLMRIDX1")
        for _ in range(capacity):
            slot = table_offset + index * INDEX_SLOT_BYTES
            slot_key = bytes(self._view[slot:slot + ID_BYTES])
            if slot_key == ZERO_ID or slot_key == key:
                return slot
            index = (index + 1) & (capacity - 1)
        raise RelationBoundsError("relation/hyperedge index table is full",
                                  code="index_full")

    def _index_value(self, slot: int) -> int:
        return struct.unpack_from("<Q", self._view, slot + ID_BYTES)[0]

    def _index_write(self, slot: int, key: bytes, value: int) -> None:
        self._view[slot:slot + ID_BYTES] = key
        struct.pack_into("<Q", self._view, slot + ID_BYTES,
                         _check_uint(value, "index value"))

    def _chain_find(self, key: bytes) -> int | None:
        capacity = self._capacities.chain
        index = _hash_slot(key, capacity, b"NLMRCHN1")
        for _ in range(capacity):
            slot = self._layout.chain_table + index * CHAIN_SLOT_BYTES
            slot_key = bytes(self._view[slot:slot + ID_BYTES])
            if slot_key == ZERO_ID:
                return None
            if slot_key == key:
                return slot
            index = (index + 1) & (capacity - 1)
        return None

    def _chain_slot(self, key: bytes) -> tuple[int, bool]:
        """Find or insert a chain-head slot, returning ``(slot, inserted)``."""
        capacity = self._capacities.chain
        index = _hash_slot(key, capacity, b"NLMRCHN1")
        for _ in range(capacity):
            slot = self._layout.chain_table + index * CHAIN_SLOT_BYTES
            slot_key = bytes(self._view[slot:slot + ID_BYTES])
            if slot_key == key:
                return slot, False
            if slot_key == ZERO_ID:
                used = self._header()[H_CHAIN_USED]
                if used >= capacity:
                    raise RelationBoundsError("leaf chain table is full",
                                              code="chain_table_full")
                self._view[slot:slot + ID_BYTES] = key
                self._write_u64(_HEADER_CHAIN_USED_OFFSET, used + 1)
                return slot, True
            index = (index + 1) & (capacity - 1)
        raise RelationBoundsError("leaf chain table is full",
                                  code="chain_table_full")

    def _chain_head(self, key: bytes, role: int) -> int:
        slot = self._chain_find(key)
        if slot is None:
            return 0
        return struct.unpack_from("<Q", self._view,
                                  slot + ID_BYTES + role * 8)[0]

    def _chain_set_head(self, key: bytes, role: int, value: int) -> None:
        slot, _ = self._chain_slot(key)
        struct.pack_into("<Q", self._view, slot + ID_BYTES + role * 8,
                         _check_uint(value, "chain head"))

    def _chain_heads(self, key: bytes) -> tuple[int, int, int]:
        slot = self._chain_find(key)
        if slot is None:
            return 0, 0, 0
        return struct.unpack_from("<QQQ", self._view, slot + ID_BYTES)

    # -- record access --------------------------------------------------
    def _relation_tuple(self, index: int) -> tuple[Any, ...]:
        count = self._header()[H_RELATION_COUNT]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count:
            raise RelationBoundsError(
                "relation index is outside the appended range",
                code="relation_index_out_of_bounds",
                details={"index": index, "count": count})
        return RELATION_RECORD.unpack_from(
            self._view, self._layout.relation_records + index * RELATION_RECORD_BYTES)

    def _relation_start(self, index: int) -> int:
        return self._layout.relation_records + index * RELATION_RECORD_BYTES

    def _hyperedge_tuple(self, index: int) -> tuple[Any, ...]:
        count = self._header()[H_HYPEREDGE_COUNT]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count:
            raise RelationBoundsError(
                "hyperedge index is outside the appended range",
                code="hyperedge_index_out_of_bounds",
                details={"index": index, "count": count})
        return HYPEREDGE_RECORD.unpack_from(
            self._view, self._layout.hyperedge_records + index * HYPEREDGE_RECORD_BYTES)

    def _relation_index(self, relation_id: bytes) -> int | None:
        header = self._header()
        slot = self._index_find(self._layout.relation_index,
                                header[H_RELATION_INDEX_CAPACITY], relation_id)
        if slot is None:
            return None
        value = self._index_value(slot)
        if value == 0 or value - 1 >= header[H_RELATION_COUNT]:
            raise RelationCorruptionError("relation index points outside records",
                                          code="dangling_relation_index")
        return value - 1

    def _hyperedge_index(self, hyperedge_id: bytes) -> int | None:
        header = self._header()
        slot = self._index_find(self._layout.hyperedge_index,
                                header[H_HYPEREDGE_INDEX_CAPACITY], hyperedge_id)
        if slot is None:
            return None
        value = self._index_value(slot)
        if value == 0 or value - 1 >= header[H_HYPEREDGE_COUNT]:
            raise RelationCorruptionError("hyperedge index points outside records",
                                          code="dangling_hyperedge_index")
        return value - 1

    def _next_relation_id(self, value: int) -> str | None:
        if not value:
            return None
        index = value - 1
        record = self._relation_tuple(index)
        return record[0].hex()

    def _relation_view(self, index: int) -> dict[str, Any]:
        (relation_id, subject, predicate, object_id, pair_id, mirror_id,
         hyperedge_id, source_digest, next_subject, next_predicate,
         next_object, confidence, generation, polarity, flags,
         _reserved) = self._relation_tuple(index)
        return {
            "index": index,
            "relation_id": relation_id.hex(),
            "subject": subject.hex(),
            "predicate": predicate.hex(),
            "object": object_id.hex(),
            "pair_id": pair_id.hex() if pair_id != ZERO_ID else None,
            "mirror_id": mirror_id.hex() if mirror_id != ZERO_ID else None,
            "hyperedge_id": hyperedge_id.hex() if hyperedge_id != ZERO_ID else None,
            "source_digest": source_digest.hex(),
            "generation": generation,
            "confidence": confidence,
            "polarity": CODE_TO_POLARITY.get(polarity, polarity),
            "polarity_code": polarity,
            "flags": flags,
            "next_subject": self._next_relation_id(next_subject),
            "next_subject_index": next_subject - 1 if next_subject else None,
            "next_predicate": self._next_relation_id(next_predicate),
            "next_predicate_index": next_predicate - 1 if next_predicate else None,
            "next_object": self._next_relation_id(next_object),
            "next_object_index": next_object - 1 if next_object else None,
        }

    def _hyperedge_view(self, index: int) -> dict[str, Any]:
        (hyperedge_id, source_digest, members_offset, kinds_offset,
         generation, arity, flags, member_kind, _reserved,
         _reserved2) = self._hyperedge_tuple(index)
        if members_offset + arity * ID_BYTES > self._capacities.arena:
            raise RelationCorruptionError("hyperedge members exceed arena",
                                          code="bad_member_pointer")
        if kinds_offset + arity > self._capacities.arena:
            raise RelationCorruptionError("hyperedge kinds exceed arena",
                                          code="bad_kind_pointer")
        members = []
        member_ids = []
        member_kinds = []
        for member in range(arity):
            start = self._layout.arena + members_offset + member * ID_BYTES
            member_id = bytes(self._view[start:start + ID_BYTES])
            kind_code = self._view[self._layout.arena + kinds_offset + member]
            kind_name = MEMBER_CODE_TO_KIND.get(kind_code, "kind:%d" % kind_code)
            members.append({"kind": kind_name, "kind_code": kind_code,
                            "id": member_id.hex()})
            member_ids.append(member_id.hex())
            member_kinds.append(kind_name)
        return {
            "index": index,
            "hyperedge_id": hyperedge_id.hex(),
            "arity": arity,
            "members": members,
            "member_ids": member_ids,
            "member_kinds": member_kinds,
            "member_kind": MEMBER_CODE_TO_KIND.get(member_kind, "mixed"),
            "source_digest": source_digest.hex(),
            "generation": generation,
            "flags": flags,
            "members_offset": members_offset,
            "kinds_offset": kinds_offset,
        }

    # -- public reads ---------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            header = self._header()
            return {
                "schema": SCHEMA,
                "version": header[H_VERSION],
                "generation": header[H_GENERATION],
                "relations": header[H_RELATION_COUNT],
                "relation_capacity": header[H_RELATION_CAPACITY],
                "hyperedges": header[H_HYPEREDGE_COUNT],
                "hyperedge_capacity": header[H_HYPEREDGE_CAPACITY],
                "chain_used": header[H_CHAIN_USED],
                "chain_capacity": header[H_CHAIN_CAPACITY],
                "relation_index_capacity": header[H_RELATION_INDEX_CAPACITY],
                "hyperedge_index_capacity": header[H_HYPEREDGE_INDEX_CAPACITY],
                "arena_used": header[H_ARENA_USED],
                "arena_bytes": header[H_ARENA_BYTES],
                "relation_record_bytes": RELATION_RECORD_BYTES,
                "hyperedge_record_bytes": HYPEREDGE_RECORD_BYTES,
                "sections": {"relations": SECTION_RELATIONS,
                             "hyperedges": SECTION_HYPEREDGES},
                "record_alignment": 8,
                "section_alignment": ALIGN,
            }

    stats = status

    def __len__(self) -> int:
        with self._lock:
            return self._header()[H_RELATION_COUNT]

    def get_relation(self, relation: Any) -> dict[str, Any] | None:
        """Read by record index (int) or content id."""
        with self._lock:
            if isinstance(relation, int) and not isinstance(relation, bool):
                count = self._header()[H_RELATION_COUNT]
                if 0 <= relation < count:
                    return self._relation_view(relation)
                return None
            key = _id_bytes(relation, field="relation_id")
            index = self._relation_index(key)
            return None if index is None else self._relation_view(index)

    read_relation = get_relation

    def relation_by_id(self, relation_id: Any) -> dict[str, Any] | None:
        with self._lock:
            index = self._relation_index(_id_bytes(relation_id, field="relation_id"))
            return None if index is None else self._relation_view(index)

    def relation_bytes(self, index: int) -> bytes:
        """Return an immutable copy of one relation record body."""
        with self._lock:
            self._relation_tuple(index)  # bounds check
            start = self._relation_start(index)
            return bytes(self._view[start:start + RELATION_RECORD_BYTES])

    def relations(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            count = self._header()[H_RELATION_COUNT]
            if limit is None:
                limit = count
            else:
                limit = self._bounded_limit(limit, maximum=max(count, 1))
            return [self._relation_view(index) for index in range(min(count, limit))]

    def iter_relations(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        yield from self.relations(limit)

    def get_hyperedge(self, hyperedge: Any) -> dict[str, Any] | None:
        with self._lock:
            if isinstance(hyperedge, int) and not isinstance(hyperedge, bool):
                count = self._header()[H_HYPEREDGE_COUNT]
                if 0 <= hyperedge < count:
                    return self._hyperedge_view(hyperedge)
                return None
            key = _id_bytes(hyperedge, field="hyperedge_id")
            index = self._hyperedge_index(key)
            return None if index is None else self._hyperedge_view(index)

    read_hyperedge = get_hyperedge

    def hyperedge_by_id(self, hyperedge_id: Any) -> dict[str, Any] | None:
        with self._lock:
            index = self._hyperedge_index(_id_bytes(hyperedge_id, field="hyperedge_id"))
            return None if index is None else self._hyperedge_view(index)

    def hyperedge_members(self, hyperedge: Any) -> list[dict[str, Any]]:
        view = self.get_hyperedge(hyperedge)
        return [] if view is None else list(view["members"])

    def iter_hyperedge_members(self, hyperedge: Any) -> Iterator[dict[str, Any]]:
        yield from self.hyperedge_members(hyperedge)

    def hyperedge_for_relation(self, relation: Any) -> dict[str, Any] | None:
        with self._lock:
            record = self.get_relation(relation)
            if record is None or record["hyperedge_id"] is None:
                return None
            index = self._hyperedge_index(bytes.fromhex(record["hyperedge_id"]))
            return None if index is None else self._hyperedge_view(index)

    @staticmethod
    def _bounded_limit(limit: Any, *, maximum: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise RelationBoundsError("limit must be a positive integer",
                                      code="invalid_limit")
        if limit > MAX_WALK_LIMIT:
            raise RelationBoundsError(
                "limit exceeds the bounded walk ceiling",
                code="limit_too_large",
                details={"limit": limit, "max": MAX_WALK_LIMIT})
        return min(limit, maximum)

    def walk_chain(self, leaf_id: Any, role: str = "subject",
                   limit: int | None = DEFAULT_WALK_LIMIT) -> list[dict[str, Any]]:
        """Walk newest-to-oldest relations linked by one leaf role."""
        return self.chain_walk(leaf_id, role=role, limit=limit)["relations"]

    def chain_walk(self, leaf_id: Any, role: str = "subject",
                   limit: int | None = DEFAULT_WALK_LIMIT) -> dict[str, Any]:
        if role not in _ROLE_FROM_NAME:
            raise RelationError("role must be subject, predicate or object")
        role_index = _ROLE_FROM_NAME[role]
        key = _id_bytes(leaf_id, field="leaf_id")
        with self._lock:
            count = self._header()[H_RELATION_COUNT]
            if limit is None:
                limit = DEFAULT_WALK_LIMIT
            limit = self._bounded_limit(limit, maximum=max(count, 1))
            head = self._chain_head(key, role_index)
            records: list[dict[str, Any]] = []
            current = head
            while current and len(records) < limit:
                index = current - 1
                if index < 0 or index >= count:
                    raise RelationCorruptionError(
                        "leaf chain points outside relation records",
                        code="dangling_chain")
                if records and index >= records[-1]["index"]:
                    raise RelationCorruptionError(
                        "leaf chain is not strictly decreasing",
                        code="cyclic_chain")
                record = self._relation_view(index)
                records.append(record)
                current = (record["next_subject_index"],
                           record["next_predicate_index"],
                           record["next_object_index"])[role_index]
                if current is not None:
                    current += 1
            truncated = bool(current) and len(records) >= limit
            return {
                "leaf_id": key.hex(),
                "role": role,
                "limit": limit,
                "relations": records,
                "truncated": truncated,
            }

    # -- append ---------------------------------------------------------
    def append_relation(
        self, *, subject: Any, predicate: Any, object: Any,
        polarity: Any = "neutral", confidence: Any = 0.5,
        pair_id: Any = None, mirror_id: Any = None,
        source_digest: Any, generation: int | None = None,
        flags: Any = 0, relation_id: Any = None,
        hyperedge_id: Any = None, hyperedge_members: Iterable[Any] | None = None,
        pair_callback: Any = None, mirror_callback: Any = None,
    ) -> dict[str, Any]:
        """Append one relation, link three leaf chains and bind a hyperedge.

        Passing an already-present ``relation_id`` is idempotent: the existing
        record is returned unchanged and no callback is invoked.
        """
        subject_id = _id_bytes(subject, field="subject")
        predicate_id = _id_bytes(predicate, field="predicate")
        object_id = _id_bytes(object, field="object")
        pair_bytes = _id_bytes(pair_id, field="pair_id", optional=True)
        mirror_bytes = _id_bytes(mirror_id, field="mirror_id", optional=True)
        source = _digest_bytes(source_digest)
        polarity_code = _polarity_code(polarity)
        confidence_value = _confidence(confidence)
        flag_value = _flags(flags)
        if relation_id is None:
            relation_key = bytes.fromhex(relation_id_for(
                subject=subject_id, predicate=predicate_id, object=object_id,
                polarity=polarity_code, confidence=confidence_value,
                pair_id=pair_bytes, mirror_id=mirror_bytes, flags=flag_value))
        else:
            relation_key = _id_bytes(relation_id, field="relation_id")
        if relation_key == ZERO_ID:
            raise RelationError("relation_id must not be all zero")

        with self._lock:
            header = self._header()
            existing = self._relation_index(relation_key)
            if existing is not None:
                record = self._relation_view(existing)
                conflict = (
                    record["subject"] != subject_id.hex()
                    or record["predicate"] != predicate_id.hex()
                    or record["object"] != object_id.hex()
                    or record["polarity_code"] != polarity_code
                    or record["confidence"] != confidence_value
                    or record["flags"] != flag_value
                    or record["pair_id"] != (pair_bytes.hex()
                                             if pair_bytes != ZERO_ID else None)
                    or record["mirror_id"] != (mirror_bytes.hex()
                                               if mirror_bytes != ZERO_ID else None)
                )
                return {
                    "status": "duplicate",
                    "created": False,
                    "duplicate": True,
                    "conflict": conflict,
                    "relation": record,
                    "relation_id": record["relation_id"],
                    "hyperedge_id": record["hyperedge_id"],
                    "generation": record["generation"],
                    "source_digest": record["source_digest"],
                    "store_generation": header[H_GENERATION],
                    "hyperedge_created": False,
                    "pair_linked": False,
                    "mirror_linked": False,
                }

            if generation is None:
                new_generation = header[H_GENERATION] + 1
            else:
                new_generation = _check_uint(generation, "generation")
                if new_generation <= header[H_GENERATION]:
                    raise RelationBoundsError(
                        "generation must increase monotonically",
                        code="stale_generation",
                        details={"generation": new_generation,
                                 "current": header[H_GENERATION]})
            if new_generation >= (1 << 64):
                raise RelationBoundsError("generation exhausted",
                                          code="generation_exhausted")
            if header[H_RELATION_COUNT] >= header[H_RELATION_CAPACITY]:
                raise RelationBoundsError(
                    "relation record capacity exhausted",
                    code="relation_capacity",
                    details={"capacity": header[H_RELATION_CAPACITY]})
            relation_index_capacity = header[H_RELATION_INDEX_CAPACITY]
            relation_index_slot = self._index_insert_slot(
                self._layout.relation_index, relation_index_capacity, relation_key)

            leaf_keys = tuple(dict.fromkeys((subject_id, predicate_id, object_id)))
            new_chain_keys = [key for key in leaf_keys if self._chain_find(key) is None]
            if header[H_CHAIN_USED] + len(new_chain_keys) > header[H_CHAIN_CAPACITY]:
                raise RelationBoundsError(
                    "leaf chain table capacity exhausted",
                    code="chain_capacity",
                    details={"capacity": header[H_CHAIN_CAPACITY],
                             "new_leaves": len(new_chain_keys)})

            if hyperedge_members is None:
                members = [
                    (MEMBER_LEAF, subject_id),
                    (MEMBER_LEAF, predicate_id),
                    (MEMBER_LEAF, object_id),
                ]
                if pair_bytes != ZERO_ID:
                    members.append((MEMBER_PAIR, pair_bytes))
                if mirror_bytes != ZERO_ID:
                    members.append((MEMBER_MIRROR, mirror_bytes))
                members.append((MEMBER_RELATION, relation_key))
                normalized_members = tuple(members)
            else:
                normalized_members = _normalize_members(hyperedge_members)

            if hyperedge_id is None:
                hasher = hashlib.sha256(b"NLMR-HYPEREDGE\0")
                for kind, member_id in normalized_members:
                    hasher.update(bytes((kind,)))
                    hasher.update(member_id)
                hyperedge_key = hasher.digest()
            else:
                hyperedge_key = _id_bytes(hyperedge_id, field="hyperedge_id")
            if hyperedge_key == ZERO_ID:
                raise RelationError("hyperedge_id must not be all zero")

            hyperedge_index = self._hyperedge_index(hyperedge_key)
            hyperedge_created = False
            if hyperedge_index is None:
                if header[H_HYPEREDGE_COUNT] >= header[H_HYPEREDGE_CAPACITY]:
                    raise RelationBoundsError(
                        "hyperedge record capacity exhausted",
                        code="hyperedge_capacity",
                        details={"capacity": header[H_HYPEREDGE_CAPACITY]})
                hyperedge_index_slot = self._index_insert_slot(
                    self._layout.hyperedge_index, header[H_HYPEREDGE_INDEX_CAPACITY],
                    hyperedge_key)
                arena_used = header[H_ARENA_USED]
                arena_end = _align8(arena_used + len(normalized_members) * (ID_BYTES + 1))
                if arena_end > header[H_ARENA_BYTES]:
                    raise RelationBoundsError(
                        "hyperedge member arena exhausted",
                        code="arena_capacity",
                        details={"used": arena_used,
                                 "required": arena_end - arena_used,
                                 "capacity": header[H_ARENA_BYTES]})
                # Commit the hyperedge first; the relation commit below has
                # already passed all capacity checks.
                members_offset = arena_used
                kinds_offset = arena_used + len(normalized_members) * ID_BYTES
                arena_start = self._layout.arena + members_offset
                for position, (kind, member_id) in enumerate(normalized_members):
                    start = arena_start + position * ID_BYTES
                    self._view[start:start + ID_BYTES] = member_id
                    self._view[self._layout.arena + kinds_offset + position] = kind
                kinds = {kind for kind, _ in normalized_members}
                common_kind = next(iter(kinds)) if len(kinds) == 1 else 0
                hyperedge_slot = header[H_HYPEREDGE_COUNT]
                HYPEREDGE_RECORD.pack_into(
                    self._view,
                    self._layout.hyperedge_records + hyperedge_slot * HYPEREDGE_RECORD_BYTES,
                    hyperedge_key, source, members_offset, kinds_offset,
                    new_generation, len(normalized_members), 0, common_kind, 0, 0)
                self._index_write(hyperedge_index_slot, hyperedge_key,
                                  hyperedge_slot + 1)
                self._write_u64(_HEADER_HYPEREDGE_COUNT_OFFSET, hyperedge_slot + 1)
                self._write_u64(_HEADER_ARENA_USED_OFFSET, arena_end)
                hyperedge_created = True
                hyperedge_index = hyperedge_slot
            else:
                # Explicit references may omit the member list; verify only
                # when the caller supplied one.
                if hyperedge_members is not None:
                    existing_edge = self._hyperedge_view(hyperedge_index)
                    existing_ids = [bytes.fromhex(member["id"])
                                    for member in existing_edge["members"]]
                    incoming_ids = [member_id for _, member_id in normalized_members]
                    incoming_kinds = [kind for kind, _ in normalized_members]
                    existing_kinds = [member["kind_code"]
                                      for member in existing_edge["members"]]
                    if (existing_ids != incoming_ids
                            or existing_kinds != incoming_kinds):
                        raise RelationConflictError(
                            "hyperedge_id already names different members",
                            code="hyperedge_conflict",
                            details={"hyperedge_id": hyperedge_key.hex()})

            relation_slot = header[H_RELATION_COUNT]
            old_subject = self._chain_head(subject_id, _ROLE_SUBJECT)
            old_predicate = self._chain_head(predicate_id, _ROLE_PREDICATE)
            old_object = self._chain_head(object_id, _ROLE_OBJECT)
            RELATION_RECORD.pack_into(
                self._view,
                self._layout.relation_records + relation_slot * RELATION_RECORD_BYTES,
                relation_key, subject_id, predicate_id, object_id,
                pair_bytes, mirror_bytes, hyperedge_key, source,
                old_subject, old_predicate, old_object, confidence_value,
                new_generation, polarity_code, flag_value, 0)
            self._index_write(relation_index_slot, relation_key, relation_slot + 1)
            self._chain_set_head(subject_id, _ROLE_SUBJECT, relation_slot + 1)
            self._chain_set_head(predicate_id, _ROLE_PREDICATE, relation_slot + 1)
            self._chain_set_head(object_id, _ROLE_OBJECT, relation_slot + 1)
            self._write_u64(_HEADER_RELATION_COUNT_OFFSET, relation_slot + 1)
            self._write_u64(_HEADER_GENERATION_OFFSET, new_generation)

            pair_linked = self._link_reference(
                pair_callback, pair_bytes, relation_key, new_generation, source)
            mirror_linked = self._link_reference(
                mirror_callback, mirror_bytes, relation_key, new_generation, source)
            record = self._relation_view(relation_slot)
            return {
                "status": "appended",
                "created": True,
                "duplicate": False,
                "conflict": False,
                "relation": record,
                "relation_id": record["relation_id"],
                "hyperedge_id": record["hyperedge_id"],
                "hyperedge_created": hyperedge_created,
                "generation": new_generation,
                "source_digest": source.hex(),
                "store_generation": new_generation,
                "pair_linked": pair_linked,
                "mirror_linked": mirror_linked,
            }

    append = append_relation

    @staticmethod
    def _link_reference(callback: Any, target: bytes, relation_id: bytes,
                        generation: int, source_digest: bytes) -> bool:
        if callback is None or target == ZERO_ID:
            return False
        callback(target_id=target.hex(), relation_id=relation_id.hex(),
                 generation=generation, source_digest=source_digest.hex())
        return True


# Integration-friendly aliases; there is still only one implementation.
ResidentRelations = RelationStore
RelationIndex = RelationStore

__all__ = (
    "SCHEMA", "RELATION_MAGIC", "HYPEREDGE_MAGIC", "VERSION", "ID_BYTES",
    "RELATION_RECORD_BYTES", "HYPEREDGE_RECORD_BYTES", "INDEX_SLOT_BYTES",
    "CHAIN_SLOT_BYTES", "HEADER_BYTES", "DEFAULT_RELATION_CAPACITY",
    "DEFAULT_HYPEREDGE_CAPACITY", "DEFAULT_WALK_LIMIT", "MAX_WALK_LIMIT",
    "POLARITY_TO_CODE", "CODE_TO_POLARITY", "MEMBER_LEAF", "MEMBER_RELATION",
    "MEMBER_PAIR", "MEMBER_MIRROR", "MEMBER_KIND_TO_CODE",
    "MEMBER_CODE_TO_KIND", "RelationError", "RelationBoundsError",
    "RelationConflictError", "RelationCorruptionError", "RelationStore",
    "ResidentRelations", "RelationIndex", "relation_layout",
    "hyperedge_layout", "relation_id_for", "hyperedge_id_for_members",
)
