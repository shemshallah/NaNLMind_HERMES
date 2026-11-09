"""Resident leaf records and the tree/relation links over them.

The store is deliberately small and boring: a caller supplies a mutable buffer
(``bytearray``, writable ``mmap`` or another writable buffer) and a base offset
inside it.  Every byte the store owns lives in that buffer, so the same memory
can be mapped, checkpointed or copied without a Python object graph.

Leaf records are fixed 112-byte structures.  Leaf identifiers are 128-bit
content hashes stored as raw 16-byte values; a bounded open-addressing index
lives next to the records.  A caller may replace that index with ``index_get`` /
``index_set`` callbacks (or an ``index`` mapping-like object) when the IDs live
in another arena.

Record layout (little endian)::

    off  size field
      0     4 magic              b"NLF1"
      4     1 version            1
      5     1 kind_code          1=word .. 6=data
      6     2 flags
      8     2 head_index         0..255, 0xFFFF = none
     10     2 relation_count
     12     4 primary_pair_id    C(256,2) range, 0xFFFFFFFF = none
     16     4 feature_offset
     20     4 feature_length
     24     4 parent             record slot, 0xFFFFFFFF = none
     28     4 first_child        record slot, 0xFFFFFFFF = none
     32     4 next_sibling       record slot, 0xFFFFFFFF = none
     36     4 first_relation     record slot, 0xFFFFFFFF = none
     40     4 next_relation      record slot, 0xFFFFFFFF = none
     44     4 relation_head      owner leaf slot, 0xFFFFFFFF = none
     48     8 generation
     56    16 leaf_id            128-bit content hash bytes
     72    32 payload_digest     SHA-256 of the payload bytes
    104     8 reserved           zero
    ---   112 total

There is no one-word-one-cell rule.  Leaves may repeat a head/pair and may
share the same feature offset/length; the record only stores the reference.
Feature offsets are absolute byte offsets in the caller's buffer unless a
``feature_region=(start, length)`` bounds is supplied.

The header before the records is 48 bytes::

    off  size field
      0     8 magic              b"NLMLEAF1"
      8     2 version
     10     2 header_bytes
     12     2 record_bytes
     14     2 index_entry_bytes
     16     4 capacity
     20     4 index_capacity
     24     4 count
     28     4 flags
     32     8 generation
     40     8 reserved

Index entries are 24 bytes: 16-byte leaf id, 4-byte record slot, 1-byte
occupancy state and 3 bytes of padding.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Callable, Iterator

from . import HEAD_COUNT, PAIR_COUNT

LEAF_MAGIC = b"NLF1"
HEADER_MAGIC = b"NLMLEAF1"
VERSION = 1
HEADER_BYTES = 48
RECORD_BYTES = 112
INDEX_ENTRY_BYTES = 24

NO_LINK = 0xFFFFFFFF
NO_HEAD = 0xFFFF
NO_RELATION = 0xFFFFFFFF
_MAX_U32 = 0xFFFFFFFF
_MAX_U64 = 0xFFFFFFFFFFFFFFFF
_MASK64 = _MAX_U64

KINDS = ("word", "concept", "vector", "relation", "hyperedge", "data")
KIND_CODE = {name: index + 1 for index, name in enumerate(KINDS)}
CODE_KIND = {code: name for name, code in KIND_CODE.items()}

# Field offsets used by link updates.
OFF_PARENT = 24
OFF_FIRST_CHILD = 28
OFF_NEXT_SIBLING = 32
OFF_FIRST_RELATION = 36
OFF_NEXT_RELATION = 40
OFF_RELATION_HEAD = 44
OFF_RELATION_COUNT = 10

HEADER_STRUCT = struct.Struct("<8sHHHHIIIIQ8s")
RECORD_STRUCT = struct.Struct("<4sBBHHHIIIIIIIIIQ16s32s8s")
INDEX_STRUCT = struct.Struct("<16sIB3x")
_ID_STRUCT = struct.Struct("<BBHIII")

assert HEADER_STRUCT.size == HEADER_BYTES
assert RECORD_STRUCT.size == RECORD_BYTES
assert INDEX_STRUCT.size == INDEX_ENTRY_BYTES


class LeafStoreError(Exception):
    """Base error for the resident leaf store."""


class LeafBoundsError(LeafStoreError, ValueError):
    """A record, buffer or numeric field is outside its bounded range."""


class LeafNotFound(LeafStoreError, KeyError):
    """The requested leaf or record slot does not exist."""


class LeafKindError(LeafStoreError, ValueError):
    """The leaf kind is not one of the six resident kinds."""


class LeafIndexFull(LeafBoundsError):
    """The bounded open-address index has no free entry left."""


class LeafCycleError(LeafStoreError, ValueError):
    """A tree/relation link would create a cycle or a corrupt chain."""


class LeafCorruptionError(LeafStoreError):
    """The buffer contents do not agree with the header or an index entry."""


@dataclass(frozen=True, slots=True)
class LeafRecord:
    """Immutable snapshot of one fixed-size leaf record.

    Link fields are record slots (``int``) or ``None``.  Reading the record
    never keeps a pointer into the buffer; call ``LeafStore.get()`` again after
    a mutation to see the current links.
    """

    slot: int
    leaf_id: bytes
    kind: str
    head: int | None
    pair_id: int | None
    feature_offset: int
    feature_length: int
    parent: int | None
    first_child: int | None
    next_sibling: int | None
    first_relation: int | None
    next_relation: int | None
    relation_head: int | None
    relation_count: int
    flags: int
    generation: int
    payload_digest: bytes

    @property
    def leaf_id_hex(self) -> str:
        return self.leaf_id.hex()

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def to_dict(self) -> dict:
        """Return a JSON-friendly, deterministic representation."""
        return {
            "slot": self.slot,
            "leaf_id": self.leaf_id_hex,
            "kind": self.kind,
            "head": self.head,
            "pair_id": self.pair_id,
            "feature_offset": self.feature_offset,
            "feature_length": self.feature_length,
            "parent": self.parent,
            "first_child": self.first_child,
            "next_sibling": self.next_sibling,
            "first_relation": self.first_relation,
            "next_relation": self.next_relation,
            "relation_head": self.relation_head,
            "relation_count": self.relation_count,
            "flags": self.flags,
            "generation": self.generation,
            "payload_digest": self.payload_digest.hex(),
        }


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _hash_index(leaf_id: bytes) -> int:
    """Mix a 128-bit id into a deterministic 64-bit open-address position."""
    left = int.from_bytes(leaf_id[:8], "little")
    right = int.from_bytes(leaf_id[8:], "little")
    mixed = (left ^ (right + 0x9E3779B97F4A7C15 + (left << 6) + (left >> 2)))
    mixed &= _MASK64
    mixed ^= mixed >> 33
    mixed = (mixed * 0xFF51AFD7ED558CCD) & _MASK64
    mixed ^= mixed >> 33
    return mixed


def _coerce_leaf_id(value) -> bytes:
    if isinstance(value, LeafRecord):
        value = value.leaf_id
    if isinstance(value, str):
        text = value[5:] if value.lower().startswith("leaf:") else value
        try:
            raw = bytes.fromhex(text)
        except ValueError as exc:
            raise LeafBoundsError("leaf id must be 32 hex characters") from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise TypeError("leaf id must be bytes, hex text or a LeafRecord")
    if len(raw) != 16:
        raise LeafBoundsError("leaf id must be exactly 16 bytes (128 bits)")
    return raw


def _coerce_digest(value) -> bytes:
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise LeafBoundsError("payload digest must be hex text") from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise TypeError("payload digest must be bytes or hex text")
    if len(raw) != 32:
        raise LeafBoundsError("payload digest must be exactly 32 bytes")
    return raw


def _coerce_content(value) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("content must be bytes-like or text")


def _kind_code(value) -> int:
    if isinstance(value, str):
        code = KIND_CODE.get(value.lower())
    elif isinstance(value, int) and not isinstance(value, bool):
        code = value if value in CODE_KIND else None
    else:
        code = None
    if code is None:
        raise LeafKindError("kind must be one of: %s" % ", ".join(KINDS))
    return code


def _optional_index(value, limit: int, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeafBoundsError("%s must be an integer or None" % name)
    if value < 0 or value >= limit:
        raise LeafBoundsError("%s must be in [0, %d)" % (name, limit))
    return value


def _optional_link(value) -> int:
    return NO_LINK if value is None else int(value)


def _slot_or_none(value: int) -> int | None:
    return None if value == NO_LINK else value


def record_layout() -> tuple[dict, ...]:
    """Describe the exact record layout, useful for audit/reporting."""
    fields = (
        ("magic", 0, 4, "4s"),
        ("version", 4, 1, "B"),
        ("kind_code", 5, 1, "B"),
        ("flags", 6, 2, "H"),
        ("head_index", 8, 2, "H"),
        ("relation_count", 10, 2, "H"),
        ("primary_pair_id", 12, 4, "I"),
        ("feature_offset", 16, 4, "I"),
        ("feature_length", 20, 4, "I"),
        ("parent", 24, 4, "I"),
        ("first_child", 28, 4, "I"),
        ("next_sibling", 32, 4, "I"),
        ("first_relation", 36, 4, "I"),
        ("next_relation", 40, 4, "I"),
        ("relation_head", 44, 4, "I"),
        ("generation", 48, 8, "Q"),
        ("leaf_id", 56, 16, "16s"),
        ("payload_digest", 72, 32, "32s"),
        ("reserved", 104, 8, "8s"),
    )
    return tuple(
        {"name": name, "offset": offset, "size": size, "format": fmt}
        for name, offset, size, fmt in fields
    )


class LeafStore:
    """Bounded leaf/tree store over a caller-owned mutable buffer."""

    def __init__(
        self,
        buffer,
        base_offset: int = 0,
        capacity: int | None = None,
        *,
        index_capacity: int | None = None,
        index_get: Callable[[bytes], int | None] | None = None,
        index_set: Callable[[bytes, int], None] | None = None,
        index=None,
        feature_region: tuple[int, int] | None = None,
        create: bool | None = None,
    ):
        if index is not None:
            if index_get is not None or index_set is not None:
                raise ValueError("pass index or index_get/index_set, not both")
            index_get = index.get
            index_set = lambda key, slot: index.__setitem__(key, slot)
        if (index_get is None) != (index_set is None):
            raise ValueError("index_get and index_set must be supplied together")

        if isinstance(base_offset, bool) or not isinstance(base_offset, int):
            raise LeafBoundsError("base_offset must be an integer")
        if base_offset < 0:
            raise LeafBoundsError("base_offset must be non-negative")
        try:
            size = len(buffer)
        except TypeError as exc:
            raise TypeError("buffer must support len()") from exc
        if base_offset + HEADER_BYTES > size:
            raise LeafBoundsError("buffer is too small for the leaf header")
        view = memoryview(buffer)
        if view.readonly:
            raise LeafBoundsError("leaf store needs a writable buffer")
        del view

        self._buffer = buffer
        self._base_offset = base_offset
        self._index_get = index_get
        self._index_set = index_set
        self._use_external_index = index_get is not None
        self._feature_region = self._check_feature_region(feature_region, size)

        if create is None:
            create = not self._has_header()
        if create:
            if capacity is None:
                raise LeafBoundsError("capacity is required when creating a store")
            self._create_region(capacity, index_capacity)
        else:
            self._load_header()

        self._records_offset = self._base_offset + HEADER_BYTES
        self._index_offset = self._records_offset + self.capacity * RECORD_BYTES
        self._total_bytes = (
            HEADER_BYTES
            + self.capacity * RECORD_BYTES
            + self.index_capacity * INDEX_ENTRY_BYTES
        )
        if self._base_offset + self._total_bytes > size:
            raise LeafBoundsError("buffer is too small for records and index")

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(
        cls,
        buffer,
        base_offset: int = 0,
        capacity: int = 1024,
        **kwargs,
    ) -> "LeafStore":
        kwargs.pop("create", None)
        return cls(
            buffer,
            base_offset=base_offset,
            capacity=capacity,
            create=True,
            **kwargs,
        )

    @classmethod
    def attach(cls, buffer, base_offset: int = 0, **kwargs) -> "LeafStore":
        kwargs.pop("create", None)
        kwargs.pop("capacity", None)
        return cls(buffer, base_offset=base_offset, create=False, **kwargs)

    @classmethod
    def from_serialized(cls, data: bytes, **kwargs) -> "LeafStore":
        """Restore a store from :meth:`serialize` output."""
        raw = bytes(data)
        buffer = bytearray(raw)
        return cls(buffer, base_offset=0, create=False, **kwargs)

    deserialize = from_serialized

    def _has_header(self) -> bool:
        end = self._base_offset + len(HEADER_MAGIC)
        return bytes(self._buffer[self._base_offset:end]) == HEADER_MAGIC

    def _check_feature_region(self, region, size: int):
        if region is None:
            return None
        if not isinstance(region, (tuple, list)) or len(region) != 2:
            raise LeafBoundsError("feature_region must be (start, length)")
        start, length = region
        for value, name in ((start, "feature start"), (length, "feature length")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LeafBoundsError("%s must be a non-negative integer" % name)
        if start + length > size:
            raise LeafBoundsError("feature_region exceeds the buffer")
        return (start, length)

    def _create_region(self, capacity, index_capacity) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise LeafBoundsError("capacity must be an integer")
        if capacity <= 0:
            raise LeafBoundsError("capacity must be positive")

        if self._use_external_index:
            if index_capacity is not None:
                raise ValueError(
                    "index_capacity is unused with caller-provided index callbacks")
            index_capacity = 0
        elif index_capacity is None:
            index_capacity = _next_power_of_two(max(16, capacity * 2))
        else:
            if isinstance(index_capacity, bool) or not isinstance(index_capacity, int):
                raise LeafBoundsError("index_capacity must be an integer")
            if index_capacity < capacity:
                raise LeafBoundsError("index_capacity must be at least capacity")
            if index_capacity & (index_capacity - 1):
                raise LeafBoundsError("index_capacity must be a power of two")
            if index_capacity < 2:
                raise LeafBoundsError("index_capacity must be at least 2")

        total = (
            HEADER_BYTES
            + capacity * RECORD_BYTES
            + index_capacity * INDEX_ENTRY_BYTES
        )
        if self._base_offset + total > len(self._buffer):
            raise LeafBoundsError(
                "buffer is too small: need %d bytes at base %d"
                % (total, self._base_offset)
            )

        start = self._base_offset
        self._buffer[start:start + total] = b"\x00" * total
        self.capacity = capacity
        self.index_capacity = index_capacity
        self.count = 0
        self.generation = 0
        self._write_header()

    def _load_header(self) -> None:
        if not self._has_header():
            raise LeafStoreError("no NLMLEAF1 header at the requested base offset")
        (
            magic,
            version,
            header_bytes,
            record_bytes,
            index_entry_bytes,
            capacity,
            index_capacity,
            count,
            _flags,
            generation,
            _reserved,
        ) = HEADER_STRUCT.unpack_from(self._buffer, self._base_offset)
        if magic != HEADER_MAGIC:
            raise LeafStoreError("bad leaf store header magic")
        if version != VERSION:
            raise LeafStoreError("unsupported leaf store version %d" % version)
        if (header_bytes, record_bytes, index_entry_bytes) != (
            HEADER_BYTES,
            RECORD_BYTES,
            INDEX_ENTRY_BYTES,
        ):
            raise LeafStoreError("incompatible leaf store layout")
        if capacity <= 0 or count > capacity:
            raise LeafCorruptionError("invalid leaf store capacity/count")
        if index_capacity:
            if index_capacity < capacity or index_capacity & (index_capacity - 1):
                raise LeafCorruptionError("invalid leaf index capacity")
        elif not self._use_external_index:
            raise LeafStoreError(
                "store has no resident index; provide index_get/index_set")
        self.capacity = capacity
        self.index_capacity = index_capacity
        self.count = count
        self.generation = generation

    def _write_header(self) -> None:
        HEADER_STRUCT.pack_into(
            self._buffer,
            self._base_offset,
            HEADER_MAGIC,
            VERSION,
            HEADER_BYTES,
            RECORD_BYTES,
            INDEX_ENTRY_BYTES,
            self.capacity,
            self.index_capacity,
            self.count,
            0,
            self.generation,
            b"",
        )

    # --------------------------------------------------------------- metadata
    @property
    def buffer(self):
        return self._buffer

    @property
    def base_offset(self) -> int:
        return self._base_offset

    @property
    def total_bytes(self) -> int:
        """Bytes occupied by the header, records and internal index."""
        return self._total_bytes

    @property
    def feature_region(self) -> tuple[int, int] | None:
        return self._feature_region

    def __len__(self) -> int:
        return self.count

    def stats(self) -> dict:
        return {
            "base_offset": self._base_offset,
            "capacity": self.capacity,
            "count": self.count,
            "index_capacity": self.index_capacity,
            "external_index": self._use_external_index,
            "generation": self.generation,
            "record_bytes": RECORD_BYTES,
            "header_bytes": HEADER_BYTES,
            "index_entry_bytes": INDEX_ENTRY_BYTES,
            "total_bytes": self._total_bytes,
        }

    # ----------------------------------------------------------------- index
    def _check_index_slot(self, slot) -> int:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise LeafCorruptionError("index returned a non-integer slot")
        if slot < 0 or slot >= self.capacity:
            raise LeafCorruptionError("index returned an out-of-range slot")
        if self.leaf_at(slot) is None:
            raise LeafCorruptionError("index points at an empty record slot")
        return slot

    def _lookup_slot(self, leaf_id: bytes) -> int | None:
        if self._use_external_index:
            slot = self._index_get(leaf_id)
            if slot is None:
                return None
            if isinstance(slot, LeafRecord):
                slot = slot.slot
            return self._check_index_slot(slot)

        mask = self.index_capacity - 1
        position = _hash_index(leaf_id) & mask
        for _ in range(self.index_capacity):
            key, slot, state = INDEX_STRUCT.unpack_from(
                self._buffer,
                self._index_offset + position * INDEX_ENTRY_BYTES,
            )
            if state == 0:
                return None
            if state != 1:
                raise LeafCorruptionError("invalid open-address index state")
            if key == leaf_id:
                return self._check_index_slot(slot)
            position = (position + 1) & mask
        return None

    def _free_index_position(self, leaf_id: bytes) -> int:
        mask = self.index_capacity - 1
        position = _hash_index(leaf_id) & mask
        for _ in range(self.index_capacity):
            _key, _slot, state = INDEX_STRUCT.unpack_from(
                self._buffer,
                self._index_offset + position * INDEX_ENTRY_BYTES,
            )
            if state == 0:
                return position
            position = (position + 1) & mask
        raise LeafIndexFull("bounded leaf index is full")

    def _write_index(self, leaf_id: bytes, slot: int, position: int | None) -> None:
        if self._use_external_index:
            self._index_set(leaf_id, slot)
            return
        if position is None:
            position = self._free_index_position(leaf_id)
        INDEX_STRUCT.pack_into(
            self._buffer,
            self._index_offset + position * INDEX_ENTRY_BYTES,
            leaf_id,
            slot,
            1,
        )

    # --------------------------------------------------------------- records
    def _record_offset(self, slot) -> int:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise LeafBoundsError("record slot must be an integer")
        if slot < 0 or slot >= self.capacity:
            raise LeafBoundsError("record slot %r is outside [0, %d)"
                                  % (slot, self.capacity))
        return self._records_offset + slot * RECORD_BYTES

    def _read_record(self, slot: int) -> LeafRecord | None:
        offset = self._record_offset(slot)
        (
            magic,
            version,
            kind_code,
            flags,
            head,
            relation_count,
            pair_id,
            feature_offset,
            feature_length,
            parent,
            first_child,
            next_sibling,
            first_relation,
            next_relation,
            relation_head,
            generation,
            leaf_id,
            payload_digest,
            _reserved,
        ) = RECORD_STRUCT.unpack_from(self._buffer, offset)
        if magic != LEAF_MAGIC:
            if slot < self.count:
                raise LeafCorruptionError("missing record magic in active slot %d"
                                          % slot)
            return None
        if slot >= self.count:
            raise LeafCorruptionError("record beyond the active count in slot %d"
                                      % slot)
        if version != VERSION or kind_code not in CODE_KIND:
            raise LeafCorruptionError("invalid record header in slot %d" % slot)
        return LeafRecord(
            slot=slot,
            leaf_id=bytes(leaf_id),
            kind=CODE_KIND[kind_code],
            head=None if head == NO_HEAD else head,
            pair_id=_slot_or_none(pair_id),
            feature_offset=feature_offset,
            feature_length=feature_length,
            parent=_slot_or_none(parent),
            first_child=_slot_or_none(first_child),
            next_sibling=_slot_or_none(next_sibling),
            first_relation=_slot_or_none(first_relation),
            next_relation=_slot_or_none(next_relation),
            relation_head=_slot_or_none(relation_head),
            relation_count=relation_count,
            flags=flags,
            generation=generation,
            payload_digest=bytes(payload_digest),
        )

    def leaf_at(self, slot: int) -> LeafRecord | None:
        """Read a slot, returning ``None`` when the slot is not in use."""
        return self._read_record(slot)

    def _require(self, ref) -> LeafRecord:
        if isinstance(ref, LeafRecord):
            record = self.leaf_at(ref.slot)
        elif isinstance(ref, int) and not isinstance(ref, bool):
            record = self.leaf_at(ref)
        elif isinstance(ref, (bytes, bytearray, memoryview, str)):
            return self._require_id(ref)
        else:
            raise TypeError("leaf reference must be a slot, id or LeafRecord")
        if record is None:
            raise LeafNotFound("leaf record slot %r does not exist" % (ref,))
        return record

    def _require_id(self, ref) -> LeafRecord:
        leaf_id = _coerce_leaf_id(ref)
        slot = self._lookup_slot(leaf_id)
        if slot is None:
            raise LeafNotFound("leaf %s not found" % leaf_id.hex())
        record = self.leaf_at(slot)
        if record is None or record.leaf_id != leaf_id:
            raise LeafCorruptionError("index entry does not match its record")
        return record

    def find(self, ref) -> LeafRecord | None:
        """Look up by 16-byte id/hex text; return ``None`` when missing."""
        if isinstance(ref, LeafRecord):
            return self.leaf_at(ref.slot)
        if isinstance(ref, int) and not isinstance(ref, bool):
            return self.leaf_at(ref)
        try:
            return self._require_id(ref)
        except LeafNotFound:
            return None

    def get(self, ref) -> LeafRecord:
        """Look up by slot, 16-byte id, hex text or a ``LeafRecord``."""
        return self._require(ref)

    def __contains__(self, ref) -> bool:
        try:
            return self.find(ref) is not None
        except (TypeError, ValueError, LeafStoreError):
            return False

    def __iter__(self) -> Iterator[LeafRecord]:
        return self.records()

    def records(self) -> Iterator[LeafRecord]:
        for slot in range(self.count):
            record = self.leaf_at(slot)
            if record is None:
                raise LeafCorruptionError("hole in active record region")
            yield record

    def _set_u32(self, slot: int, field_offset: int, value) -> None:
        struct.pack_into(
            "<I",
            self._buffer,
            self._record_offset(slot) + field_offset,
            _optional_link(value),
        )

    def _set_relation_count(self, slot: int, value: int) -> None:
        if value < 0 or value > 0xFFFF:
            raise LeafBoundsError("relation count is out of range")
        struct.pack_into(
            "<H",
            self._buffer,
            self._record_offset(slot) + OFF_RELATION_COUNT,
            value,
        )

    # ----------------------------------------------------------------- insert
    def insert(
        self,
        kind,
        *,
        leaf_id=None,
        content=None,
        payload_digest=None,
        head=None,
        pair_id=None,
        feature_offset: int = 0,
        feature_length: int = 0,
        parent=None,
        flags: int = 0,
        generation: int | None = None,
        prepend: bool = False,
    ) -> LeafRecord:
        """Insert a leaf and optionally append it under ``parent``.

        Re-inserting an existing ``leaf_id`` is idempotent: the current record
        is returned and nothing (including the tree) is rewritten.
        """
        kind_code = _kind_code(kind)
        if isinstance(flags, bool) or not isinstance(flags, int):
            raise LeafBoundsError("flags must be an integer")
        if flags < 0 or flags > 0xFFFF:
            raise LeafBoundsError("flags must fit in 16 bits")
        head_value = _optional_index(head, HEAD_COUNT, "head")
        pair_value = _optional_index(pair_id, PAIR_COUNT, "pair_id")
        feature_offset, feature_length = self._check_feature_ref(
            feature_offset, feature_length)

        if leaf_id is None:
            add_nonce = content is None and payload_digest is None
            if payload_digest is None:
                data = b"" if content is None else _coerce_content(content)
                digest = hashlib.sha256(data).digest()
            else:
                digest = _coerce_digest(payload_digest)
                if content is not None:
                    _coerce_content(content)
            seed = _ID_STRUCT.pack(
                kind_code,
                flags,
                NO_HEAD if head_value is None else head_value,
                NO_LINK if pair_value is None else pair_value,
                feature_offset,
                feature_length,
            ) + digest
            if add_nonce:
                seed += struct.pack("<Q", self.generation + 1)
            leaf_id = hashlib.sha256(seed).digest()[:16]
        else:
            leaf_id = _coerce_leaf_id(leaf_id)
            if payload_digest is None:
                data = b"" if content is None else _coerce_content(content)
                digest = hashlib.sha256(data).digest()
            else:
                digest = _coerce_digest(payload_digest)
                if content is not None:
                    _coerce_content(content)

        existing_slot = self._lookup_slot(leaf_id)
        if existing_slot is not None:
            return self._require_id(leaf_id)

        parent_slot = None
        if parent is not None:
            parent_slot = self._require(parent).slot

        if self.count >= self.capacity:
            raise LeafBoundsError(
                "leaf store capacity %d reached" % self.capacity)

        next_generation = self.generation + 1 if generation is None else generation
        if (isinstance(next_generation, bool)
                or not isinstance(next_generation, int)
                or next_generation < 0
                or next_generation > _MAX_U64):
            raise LeafBoundsError("generation must be a 64-bit unsigned integer")

        slot = self.count
        position = None
        if not self._use_external_index:
            position = self._free_index_position(leaf_id)

        self._write_record(
            slot,
            kind_code=kind_code,
            flags=flags,
            head=head_value,
            pair_id=pair_value,
            feature_offset=feature_offset,
            feature_length=feature_length,
            generation=next_generation,
            leaf_id=leaf_id,
            payload_digest=digest,
        )
        try:
            self._write_index(leaf_id, slot, position)
        except Exception:
            self._buffer[
                self._record_offset(slot):self._record_offset(slot) + RECORD_BYTES
            ] = b"\x00" * RECORD_BYTES
            raise

        self.count += 1
        self.generation = max(self.generation, next_generation)
        self._write_header()
        if parent_slot is not None:
            self._append_child(parent_slot, slot, prepend=prepend)
        return self._require_id(leaf_id)

    def _write_record(
        self,
        slot: int,
        *,
        kind_code: int,
        flags: int,
        head: int | None,
        pair_id: int | None,
        feature_offset: int,
        feature_length: int,
        generation: int,
        leaf_id: bytes,
        payload_digest: bytes,
    ) -> None:
        RECORD_STRUCT.pack_into(
            self._buffer,
            self._record_offset(slot),
            LEAF_MAGIC,
            VERSION,
            kind_code,
            flags,
            NO_HEAD if head is None else head,
            0,
            _optional_link(pair_id),
            feature_offset,
            feature_length,
            NO_LINK,
            NO_LINK,
            NO_LINK,
            NO_LINK,
            NO_LINK,
            NO_LINK,
            generation,
            leaf_id,
            payload_digest,
            b"",
        )

    def _check_feature_ref(self, offset, length) -> tuple[int, int]:
        for value, name in ((offset, "feature_offset"),
                            (length, "feature_length")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise LeafBoundsError("%s must be an integer" % name)
            if value < 0 or value > _MAX_U32:
                raise LeafBoundsError("%s must be a 32-bit unsigned integer"
                                      % name)
        if offset + length > _MAX_U32 + 1:
            raise LeafBoundsError("feature range exceeds the 32-bit arena")
        if self._feature_region is not None and (offset or length):
            start, size = self._feature_region
            if offset < start or offset + length > start + size:
                raise LeafBoundsError(
                    "feature range [%d, %d) is outside [%d, %d)"
                    % (offset, offset + length, start, start + size)
                )
        return offset, length

    def feature_bytes(self, ref) -> bytes:
        """Read the shared feature bytes referenced by a leaf."""
        record = self._require(ref)
        if record.feature_length == 0:
            return b""
        end = record.feature_offset + record.feature_length
        if end > len(self._buffer):
            raise LeafBoundsError("feature range exceeds the buffer")
        return bytes(self._buffer[record.feature_offset:end])

    # ------------------------------------------------------------- tree links
    def _child_slots(self, slot: int) -> Iterator[int]:
        record = self._require(slot)
        child = record.first_child
        seen = 0
        while child is not None:
            child_record = self._require(child)
            yield child
            child = child_record.next_sibling
            seen += 1
            if seen > self.capacity:
                raise LeafCycleError("child sibling chain is cyclic")

    def children(self, ref) -> list[LeafRecord]:
        record = self._require(ref)
        return [self._require(child) for child in self._child_slots(record.slot)]

    def _append_child(self, parent_slot: int, child_slot: int,
                      prepend: bool = False) -> None:
        parent = self._require(parent_slot)
        child = self._require(child_slot)
        if child.parent is not None:
            raise LeafStoreError("child is already attached; use move_leaf")
        self._set_u32(child_slot, OFF_NEXT_SIBLING, NO_LINK)
        old_first = parent.first_child
        if old_first is None:
            self._set_u32(parent_slot, OFF_FIRST_CHILD, child_slot)
        elif prepend:
            self._set_u32(child_slot, OFF_NEXT_SIBLING, old_first)
            self._set_u32(parent_slot, OFF_FIRST_CHILD, child_slot)
        else:
            tail = old_first
            seen = 0
            while True:
                tail_record = self._require(tail)
                if tail_record.next_sibling is None:
                    break
                tail = tail_record.next_sibling
                seen += 1
                if seen > self.capacity:
                    raise LeafCycleError("child sibling chain is cyclic")
            self._set_u32(tail, OFF_NEXT_SIBLING, child_slot)
        self._set_u32(child_slot, OFF_PARENT, parent_slot)

    def _detach_child(self, child_slot: int) -> None:
        child = self._require(child_slot)
        parent_slot = child.parent
        if parent_slot is None:
            if child.next_sibling is not None:
                raise LeafCorruptionError("orphan leaf has a sibling link")
            return
        parent = self._require(parent_slot)
        if parent.first_child == child_slot:
            self._set_u32(parent_slot, OFF_FIRST_CHILD, child.next_sibling)
        else:
            previous_slot = parent.first_child
            found = False
            seen = 0
            while previous_slot is not None:
                previous = self._require(previous_slot)
                if previous.next_sibling == child_slot:
                    self._set_u32(
                        previous_slot, OFF_NEXT_SIBLING, child.next_sibling)
                    found = True
                    break
                previous_slot = previous.next_sibling
                seen += 1
                if seen > self.capacity:
                    raise LeafCycleError("child sibling chain is cyclic")
            if not found:
                raise LeafCorruptionError(
                    "leaf %d is not in its parent's child chain" % child_slot)
        self._set_u32(child_slot, OFF_PARENT, NO_LINK)
        self._set_u32(child_slot, OFF_NEXT_SIBLING, NO_LINK)

    def _is_ancestor(self, ancestor_slot: int, node_slot: int) -> bool:
        current = self._require(node_slot).parent
        seen = 0
        while current is not None:
            if current == ancestor_slot:
                return True
            current = self._require(current).parent
            seen += 1
            if seen > self.capacity:
                raise LeafCycleError("parent chain is cyclic")
        return False

    def move_leaf(self, ref, new_parent=None, *, prepend: bool = False
                  ) -> LeafRecord:
        """Relink a leaf without touching its own child subtree."""
        record = self._require(ref)
        target_slot = None
        if new_parent is not None:
            target = self._require(new_parent)
            target_slot = target.slot
            if target_slot == record.slot:
                raise LeafCycleError("cannot move a leaf under itself")
            if self._is_ancestor(record.slot, target_slot):
                raise LeafCycleError("cannot move a leaf under its descendant")
        if record.parent == target_slot:
            return record
        self._detach_child(record.slot)
        if target_slot is not None:
            self._append_child(target_slot, record.slot, prepend=prepend)
        return self._require(record.slot)

    relink = move_leaf

    def link(self, parent, child, *, prepend: bool = False) -> LeafRecord:
        """Attach ``child`` under ``parent`` (parent-first convenience)."""
        return self.move_leaf(child, parent, prepend=prepend)

    def unlink(self, ref) -> LeafRecord:
        return self.move_leaf(ref, None)

    def depth(self, ref) -> int:
        """Return root-relative depth (a root leaf has depth 0)."""
        record = self._require(ref)
        depth = 0
        current = record.parent
        seen = 0
        while current is not None:
            depth += 1
            current = self._require(current).parent
            seen += 1
            if seen > self.capacity:
                raise LeafCycleError("parent chain is cyclic")
        return depth

    depth_of = depth

    def walk_subtree(
        self,
        ref,
        *,
        include_self: bool = True,
        max_depth: int | None = None,
    ) -> Iterator[LeafRecord]:
        """Iteratively walk a subtree in pre-order (no recursion)."""
        root = self._require(ref)
        if max_depth is not None:
            if (isinstance(max_depth, bool) or not isinstance(max_depth, int)
                    or max_depth < 0):
                raise LeafBoundsError("max_depth must be a non-negative integer")
        stack: list[tuple[int, int]] = []
        if include_self:
            stack.append((root.slot, 0))
        else:
            children = list(self._child_slots(root.slot))
            for child in reversed(children):
                stack.append((child, 1))
        seen: set[int] = set()
        while stack:
            slot, depth = stack.pop()
            if slot in seen:
                raise LeafCycleError("subtree contains a cycle")
            seen.add(slot)
            record = self._require(slot)
            yield record
            if max_depth is not None and depth >= max_depth:
                continue
            children = list(self._child_slots(slot))
            for child in reversed(children):
                stack.append((child, depth + 1))

    def subtree(self, ref, **kwargs) -> list[LeafRecord]:
        return list(self.walk_subtree(ref, **kwargs))

    # --------------------------------------------------------- relation links
    def _relation_slots(self, slot: int) -> Iterator[int]:
        record = self._require(slot)
        relation_slot = record.first_relation
        seen = 0
        while relation_slot is not None:
            relation = self._require(relation_slot)
            yield relation_slot
            relation_slot = relation.next_relation
            seen += 1
            if seen > self.capacity:
                raise LeafCycleError("relation chain is cyclic")

    def relations(self, ref) -> list[LeafRecord]:
        record = self._require(ref)
        return [self._require(slot) for slot in self._relation_slots(record.slot)]

    def _detach_relation(self, relation_slot: int) -> None:
        relation = self._require(relation_slot)
        owner_slot = relation.relation_head
        if owner_slot is None:
            if relation.next_relation is not None:
                raise LeafCorruptionError("orphan relation has a next link")
            return
        owner = self._require(owner_slot)
        if owner.first_relation == relation_slot:
            self._set_u32(owner_slot, OFF_FIRST_RELATION,
                          relation.next_relation)
        else:
            previous_slot = owner.first_relation
            found = False
            seen = 0
            while previous_slot is not None:
                previous = self._require(previous_slot)
                if previous.next_relation == relation_slot:
                    self._set_u32(
                        previous_slot, OFF_NEXT_RELATION, relation.next_relation)
                    found = True
                    break
                previous_slot = previous.next_relation
                seen += 1
                if seen > self.capacity:
                    raise LeafCycleError("relation chain is cyclic")
            if not found:
                raise LeafCorruptionError(
                    "relation %d is not in its owner's chain" % relation_slot)
        self._set_relation_count(
            owner_slot, max(0, owner.relation_count - 1))
        self._set_u32(relation_slot, OFF_RELATION_HEAD, NO_LINK)
        self._set_u32(relation_slot, OFF_NEXT_RELATION, NO_LINK)

    def attach_relation(self, head, relation, *, prepend: bool = True
                        ) -> LeafRecord:
        """Attach a relation leaf to a leaf's first-relation chain."""
        head_record = self._require(head)
        relation_record = self._require(relation)
        if relation_record.kind != "relation":
            raise LeafKindError("only relation leaves can be attached as relations")
        if head_record.slot == relation_record.slot:
            raise LeafCycleError("a relation cannot be its own head")
        if relation_record.relation_head == head_record.slot:
            # Already in the requested chain; keep the operation idempotent.
            if relation_record.slot in self._relation_slots(head_record.slot):
                return relation_record
        if relation_record.relation_head is not None:
            self._detach_relation(relation_record.slot)

        head_record = self._require(head_record.slot)
        old_first = head_record.first_relation
        if old_first is not None and not prepend:
            tail_slot = old_first
            seen = 0
            while True:
                tail = self._require(tail_slot)
                if tail.next_relation is None:
                    break
                tail_slot = tail.next_relation
                seen += 1
                if seen > self.capacity:
                    raise LeafCycleError("relation chain is cyclic")
            self._set_u32(tail_slot, OFF_NEXT_RELATION, relation_record.slot)
            self._set_u32(relation_record.slot, OFF_NEXT_RELATION, NO_LINK)
        else:
            self._set_u32(relation_record.slot, OFF_NEXT_RELATION, old_first)
            self._set_u32(head_record.slot, OFF_FIRST_RELATION,
                          relation_record.slot)
        self._set_u32(relation_record.slot, OFF_RELATION_HEAD,
                      head_record.slot)
        self._set_relation_count(
            head_record.slot, min(0xFFFF, head_record.relation_count + 1))
        return self._require(relation_record.slot)

    def detach_relation(self, relation) -> LeafRecord:
        record = self._require(relation)
        if record.kind != "relation":
            raise LeafKindError("only relation leaves can be detached as relations")
        self._detach_relation(record.slot)
        return self._require(record.slot)

    # ----------------------------------------------------------- persistence
    def record_bytes(self, ref) -> bytes:
        """Return the exact 112-byte serialized form of a leaf record."""
        record = self._require(ref)
        offset = self._record_offset(record.slot)
        return bytes(self._buffer[offset:offset + RECORD_BYTES])

    def serialize(self) -> bytes:
        """Return a deterministic image of the header, records and index.

        A caller-provided external index is not part of the image; pass the
        same callbacks again when restoring such a store.
        """
        self._write_header()
        end = self._base_offset + self._total_bytes
        return bytes(self._buffer[self._base_offset:end])

    to_bytes = serialize


__all__ = (
    "LEAF_MAGIC",
    "HEADER_MAGIC",
    "VERSION",
    "HEADER_BYTES",
    "RECORD_BYTES",
    "INDEX_ENTRY_BYTES",
    "NO_LINK",
    "NO_HEAD",
    "NO_RELATION",
    "KINDS",
    "KIND_CODE",
    "CODE_KIND",
    "LeafStoreError",
    "LeafBoundsError",
    "LeafNotFound",
    "LeafKindError",
    "LeafIndexFull",
    "LeafCycleError",
    "LeafCorruptionError",
    "LeafRecord",
    "LeafStore",
    "ResidentLeafStore",
    "record_layout",
)

# A convenient alias for callers that use the "resident" prefix.
ResidentLeafStore = LeafStore
