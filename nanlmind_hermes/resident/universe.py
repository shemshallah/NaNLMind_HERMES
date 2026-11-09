"""One bounded resident universe over a single memory-mapped file.

This module is the integration layer for the independent resident primitives:

* :mod:`nanlmind_hermes.resident.cells` -- 256 heads and 32,640 mirror pairs;
* :mod:`nanlmind_hermes.resident.leaves` -- stable 128-bit leaves and a tree;
* :mod:`nanlmind_hermes.resident.relations` -- typed relations/hyperedges;
* :mod:`nanlmind_hermes.resident.resonance` -- quantized superposition vectors;
* :mod:`nanlmind_hermes.resident.export` -- candidate export/intake;
* :mod:`nanlmind_hermes.resident.persistence` -- atomic checkpoints.

The file is not the canonical BS0 universe.  It is local-derived evidence that
may later be exported as a candidate.  Canonical merge/promotion stays disabled.

File format
-----------

The first 64 KiB are two checksummed JSON header slots.  The rest is a fixed
layout of sections selected from the file size::

    header slots (64 KiB)
    chart placeholder (8 MiB on phone profiles)
    payload arena (text/metadata referenced by leaves)
    cells (256 heads + 32,640 pair cells, 2 MiB)
    leaves (fixed records + bounded open-addressing index)
    relations (typed relation/hyperedge store)
    features (quantized int8/int16 superposition rows)
    journal (fixed 64-byte state-chain records)

Only used pages of the sparse file are faulted in.  There is no 2 GiB
overbootstrap copy.  The default file budget is 256 MiB and the hard ceiling is
1 GiB.

The Python layer is intentionally a coordinator, not a second learner.  It
persists the resident evidence and makes it queryable; the existing temporal LM
and layered learner consume/produce records through this API.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Any, Iterable, Iterator, Mapping

from . import (
    ALIGN,
    DEFAULT_BYTES,
    HEAD_COUNT,
    MAGIC,
    MAX_BYTES,
    MIN_BYTES,
    RESIDENT_VERSION,
    SECTION_ARENA,
    SECTION_CHART,
    SECTION_FEATURES,
    SECTION_HEADS,
    SECTION_HYPEREDGES,
    SECTION_INDEX,
    SECTION_JOURNAL,
    SECTION_LEAVES,
    SECTION_PAIRS,
    SECTION_RELATIONS,
    SECTION_TOKENS,
    align,
    clamp_size,
)
# Sibling modules are imported lazily at this module boundary.  Importing
# ``universe`` must stay harmless on a partial installation; the first
# ``open`` then fails with ResidentIntegrationError naming the missing pieces.
_IMPORT_ERROR: BaseException | None = None
try:
    from .cells import (
        PAIR_COUNT,
        REGION_BYTES as CELLS_BYTES,
        ResidentCells,
        address_of,
        identity_address,
    )
    from .ingest import (
        DEFAULT_FEATURE_DIM,
        DEFAULT_FEATURE_DTYPE,
        derive_features,
        normalize_text,
        prepare_records,
    )
    from .leaves import LeafStore, LeafRecord, \
        RECORD_BYTES as LEAF_RECORD_BYTES, \
        HEADER_BYTES as LEAF_HEADER_BYTES, \
        INDEX_ENTRY_BYTES as LEAF_INDEX_BYTES
    from .relations import RelationStore
    from .resonance import FeatureArena, FeatureStore, ResonanceHit
    from . import export as resident_export
    from . import persistence as resident_persistence
except ImportError as _import_exc:  # pragma: no cover - partial installs
    _IMPORT_ERROR = _import_exc
    PAIR_COUNT = 256 * 255 // 2
    CELLS_BYTES = 2_097_152
    ResidentCells = None  # type: ignore[assignment]
    identity_address = None  # type: ignore[assignment]
    DEFAULT_FEATURE_DIM = 256
    DEFAULT_FEATURE_DTYPE = "int16"
    derive_features = None  # type: ignore[assignment]
    normalize_text = None  # type: ignore[assignment]
    prepare_records = None  # type: ignore[assignment]
    LeafStore = None  # type: ignore[assignment]
    LeafRecord = None  # type: ignore[assignment]
    LEAF_RECORD_BYTES = 112
    LEAF_HEADER_BYTES = 48
    LEAF_INDEX_BYTES = 24
    RelationStore = None  # type: ignore[assignment]
    FeatureArena = None  # type: ignore[assignment]
    FeatureStore = None  # type: ignore[assignment]
    ResonanceHit = None  # type: ignore[assignment]
    resident_export = None  # type: ignore[assignment]
    resident_persistence = None  # type: ignore[assignment]


UNIVERSE_SCHEMA = "nanlm.resident-universe/1"
HEADER_MAGIC = b"NLMUHDR1"
HEADER_VERSION = 1
HEADER_SLOT_BYTES = 4096
HEADER_SLOTS = 2
HEADER_REGION_BYTES = 64 * 1024
HEADER_PREFIX = struct.Struct("<8sII32s")
PAYLOAD_MAGIC = b"NLMPAY01"
PAYLOAD_HEADER = struct.Struct("<8sIIQQ")
PAYLOAD_HEADER_BYTES = PAYLOAD_HEADER.size
JOURNAL_MAGIC = b"NLMJRN01"
JOURNAL_HEADER = struct.Struct("<8sIIQQ")
JOURNAL_RECORD = struct.Struct("<8sQ32s16s")
JOURNAL_RECORD_BYTES = JOURNAL_RECORD.size
JOURNAL_HEADER_BYTES = JOURNAL_HEADER.size
DEFAULT_FEATURE_DIMENSIONS = DEFAULT_FEATURE_DIM
DEFAULT_FEATURE_DTYPE = DEFAULT_FEATURE_DTYPE
MAX_QUERY_TOP_K = 64
MAX_LOOKUP_LIMIT = 64
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
HEADS_BYTES = 8 * 1024
PAIRS_BYTES = CELLS_BYTES - HEADS_BYTES
PLACEHOLDER_SECTION_BYTES = 64 * 1024


class ResidentUniverseError(ValueError):
    """Raised for a bounded or corrupt resident-universe operation."""


class ResidentUniverseFull(ResidentUniverseError, MemoryError):
    """Raised when the selected resident profile is exhausted."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_loads(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def _next_power_of_two(value: int, minimum: int = 2) -> int:
    value = max(minimum, int(value))
    return 1 << (value - 1).bit_length()


def _leaf_id_bytes(value: Any) -> bytes:
    """Map a prepared/candidate leaf id to the 16-byte resident id."""
    text = str(value)
    tail = text.rsplit(":", 1)[-1]
    if len(tail) >= 32:
        try:
            return bytes.fromhex(tail[:32])
        except ValueError:
            pass
    return hashlib.sha256(text.encode("utf-8")).digest()[:16]


def _relation_id_bytes(value: Any) -> bytes:
    text = str(value)
    tail = text.rsplit(":", 1)[-1]
    if len(tail) == 64:
        try:
            return bytes.fromhex(tail)
        except ValueError:
            pass
    return hashlib.sha256(text.encode("utf-8")).digest()


def _relation_leaf_ref(leaf_id: Any) -> bytes:
    return hashlib.sha256(("leaf-ref:" + str(leaf_id)).encode("utf-8")).digest()


def _kind(value: Any) -> str:
    text = normalize_text(value or "data").casefold()
    aliases = {
        "observation": "data",
        "entity": "concept",
        "concept": "concept",
        "word": "word",
        "vector": "vector",
        "relation": "relation",
        "hyperedge": "hyperedge",
        "data": "data",
    }
    return aliases.get(text, "data")


def _head(value: Any, fallback: int = 0) -> int:
    if isinstance(value, bool):
        return fallback % HEAD_COUNT
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback % HEAD_COUNT
    if number < 0 or number >= HEAD_COUNT:
        return number % HEAD_COUNT
    return number


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(result):
        return 0.5
    return min(1.0, max(0.0, result))


def _valid_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) != 64:
        return None
    try:
        bytes.fromhex(text)
    except ValueError:
        return None
    return text


def _source_digest(record: Mapping[str, Any]) -> str:
    value = _valid_digest(record.get("source_digest"))
    if value:
        return value
    seed = _json_bytes({
        "text": record.get("text"),
        "record_id": record.get("record_id") or record.get("leaf_id")
        or record.get("relation_id"),
        "kind": record.get("kind"),
    })
    return hashlib.sha256(seed).hexdigest()


class PayloadArena:
    """Append-only byte arena for leaf text/metadata payloads."""

    def __init__(self, buffer, base_offset: int, capacity: int,
                 create: bool = False):
        self.buffer = buffer
        self.base_offset = base_offset
        self.capacity = capacity
        self.data_offset = base_offset + PAYLOAD_HEADER_BYTES
        self.end_offset = self.data_offset + capacity
        if self.end_offset > len(buffer):
            raise ResidentUniverseError("payload arena exceeds the mapped file")
        if create:
            self.used = 0
            self._write_header()
        else:
            magic, version, header_bytes, cap, used = PAYLOAD_HEADER.unpack_from(
                buffer, base_offset)
            if magic != PAYLOAD_MAGIC or version != 1 or header_bytes != PAYLOAD_HEADER_BYTES:
                raise ResidentUniverseError("bad payload arena header")
            if cap != capacity or used > capacity:
                raise ResidentUniverseError("payload arena capacity mismatch")
            self.used = used

    def _write_header(self) -> None:
        PAYLOAD_HEADER.pack_into(
            self.buffer, self.base_offset, PAYLOAD_MAGIC, 1,
            PAYLOAD_HEADER_BYTES, self.capacity, self.used)

    def append(self, data: bytes) -> int:
        raw = bytes(data)
        start = align(self.used, 8)
        end = start + len(raw)
        if end > self.capacity:
            raise ResidentUniverseFull(
                "resident payload arena is full: need %d more bytes" % len(raw))
        absolute = self.data_offset + start
        self.buffer[absolute:absolute + len(raw)] = raw
        self.used = end
        self._write_header()
        return absolute

    def read(self, offset: int, length: int) -> bytes:
        start = int(offset) - self.data_offset
        if start < 0 or length < 0 or start + length > self.used:
            raise ResidentUniverseError("payload reference is outside the used arena")
        return bytes(self.buffer[offset:offset + length])

    def stats(self) -> dict[str, int]:
        return {
            "base_offset": self.base_offset,
            "capacity_bytes": self.capacity,
            "used_bytes": self.used,
            "free_bytes": self.capacity - self.used,
        }


class StateJournal:
    """Fixed-size monotonic state chain persisted in the resident file."""

    def __init__(self, buffer, base_offset: int, capacity_records: int,
                 create: bool = False):
        self.buffer = buffer
        self.base_offset = base_offset
        self.capacity = capacity_records
        self.records_offset = base_offset + JOURNAL_HEADER_BYTES
        self.end_offset = self.records_offset + capacity_records * JOURNAL_RECORD_BYTES
        if self.end_offset > len(buffer):
            raise ResidentUniverseError("journal exceeds the mapped file")
        if create:
            self.used = 0
            self._write_header()
        else:
            magic, version, header_bytes, capacity, used = JOURNAL_HEADER.unpack_from(
                buffer, base_offset)
            if magic != JOURNAL_MAGIC or version != 1 or header_bytes != JOURNAL_HEADER_BYTES:
                raise ResidentUniverseError("bad state journal header")
            if capacity != capacity_records or used > capacity:
                raise ResidentUniverseError("state journal capacity mismatch")
            self.used = used

    def _write_header(self) -> None:
        JOURNAL_HEADER.pack_into(
            self.buffer, self.base_offset, JOURNAL_MAGIC, 1,
            JOURNAL_HEADER_BYTES, self.capacity, self.used)

    def append(self, kind: str, digest: bytes, *,
               generation: int | None = None) -> int:
        if len(digest) != 32:
            raise ResidentUniverseError("journal digest must be 32 bytes")
        if self.used >= self.capacity:
            raise ResidentUniverseFull("resident state journal is full")
        if generation is None:
            generation = self.used + 1
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ResidentUniverseError("journal generation must be non-negative")
        label = kind.encode("ascii", "replace")[:16].ljust(16, b"\x00")
        JOURNAL_RECORD.pack_into(
            self.buffer, self.records_offset + self.used * JOURNAL_RECORD_BYTES,
            JOURNAL_MAGIC, generation, bytes(digest), label)
        self.used += 1
        self._write_header()
        return generation

    def scan(self) -> list[dict[str, Any]]:
        records = []
        for index in range(self.used):
            offset = self.records_offset + index * JOURNAL_RECORD_BYTES
            magic, generation, digest, kind = JOURNAL_RECORD.unpack_from(
                self.buffer, offset)
            if magic != JOURNAL_MAGIC:
                break
            if not any(digest):
                break
            records.append({
                "sequence": generation,
                "generation": generation,
                "digest": bytes(digest).hex(),
                "kind": kind.rstrip(b"\x00").decode("ascii", "replace"),
            })
        return records

    def stats(self) -> dict[str, int]:
        return {
            "base_offset": self.base_offset,
            "capacity_records": self.capacity,
            "used_records": self.used,
            "record_bytes": JOURNAL_RECORD_BYTES,
        }


def _leaf_region_bytes(capacity: int) -> tuple[int, int]:
    index_capacity = _next_power_of_two(max(16, capacity * 2))
    total = LEAF_HEADER_BYTES + capacity * LEAF_RECORD_BYTES \
        + index_capacity * LEAF_INDEX_BYTES
    return total, index_capacity


def _plan_for_size(file_size: int) -> dict[str, Any]:
    """Choose a bounded profile that fits inside ``file_size``.

    The returned plan contains no absolute offsets; those are assigned in
    ``_build_sections``.  Capacities are reduced until the complete fixed
    layout fits.  This is important for tests and for small phone profiles.
    """
    if file_size <= 48 * 1024 * 1024:
        profile = "fixture"
        chart = 64 * 1024
        leaves, relations, hyperedges, features = 4096, 512, 512, 1024
        payload, journal = 4 * 1024 * 1024, 4096
    elif file_size <= 160 * 1024 * 1024:
        profile = "phone-128"
        chart = 2 * 1024 * 1024
        leaves, relations, hyperedges, features = 32768, 8192, 8192, 8192
        payload, journal = 8 * 1024 * 1024, 32768
    elif file_size <= 768 * 1024 * 1024:
        profile = "phone-256"
        chart = 8 * 1024 * 1024
        leaves, relations, hyperedges, features = 131072, 32768, 32768, 32768
        payload, journal = 32 * 1024 * 1024, 131072
    else:
        profile = "desktop-1g"
        chart = 8 * 1024 * 1024
        leaves, relations, hyperedges, features = 524288, 131072, 131072, 131072
        payload, journal = 128 * 1024 * 1024, 262144

    while True:
        leaf_bytes, leaf_index = _leaf_region_bytes(leaves)
        relation_bytes = RelationStore.required_bytes(
            relation_capacity=relations,
            hyperedge_capacity=hyperedges,
            base_offset=0,
        )
        feature_slot = ((DEFAULT_FEATURE_DIMENSIONS * 2) + 7) & ~7
        feature_bytes = features * feature_slot
        payload_bytes = PAYLOAD_HEADER_BYTES + payload
        journal_bytes = JOURNAL_HEADER_BYTES + journal * JOURNAL_RECORD_BYTES
        total = HEADER_REGION_BYTES + chart + payload_bytes + CELLS_BYTES \
            + leaf_bytes + relation_bytes + feature_bytes + journal_bytes \
            + PLACEHOLDER_SECTION_BYTES * 3
        total = align(total)
        if total <= file_size:
            break
        if leaves <= 256 and relations <= 32 and features <= 64 and payload <= 256 * 1024:
            raise ResidentUniverseError(
                "requested resident size is too small for the fixed geometry")
        leaves = max(256, leaves // 2)
        relations = max(32, relations // 2)
        hyperedges = max(32, hyperedges // 2)
        features = max(64, features // 2)
        payload = max(256 * 1024, payload // 2)
        journal = max(512, journal // 2)
        chart = min(chart, max(64 * 1024, file_size // 16))

    return {
        "profile": profile,
        "chart_bytes": chart,
        "leaf_capacity": leaves,
        "leaf_index_capacity": leaf_index,
        "leaf_bytes": leaf_bytes,
        "relation_capacity": relations,
        "hyperedge_capacity": hyperedges,
        "relation_bytes": relation_bytes,
        "feature_capacity": features,
        "feature_dim": DEFAULT_FEATURE_DIMENSIONS,
        "feature_dtype": DEFAULT_FEATURE_DTYPE,
        "feature_bytes": feature_bytes,
        "payload_capacity": payload,
        "payload_bytes": payload_bytes,
        "journal_capacity": journal,
        "journal_bytes": journal_bytes,
        "used_bytes": total,
    }


def _assign_sections(plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    cursor = HEADER_REGION_BYTES
    sections: dict[int, dict[str, Any]] = {}

    def place(section_id: int, name: str, size: int) -> None:
        nonlocal cursor
        cursor = align(cursor)
        sections[int(section_id)] = {
            "id": int(section_id), "name": name,
            "offset": cursor, "length": int(size),
        }
        cursor = align(cursor + int(size))

    place(SECTION_CHART, "chart", int(plan["chart_bytes"]))
    place(SECTION_ARENA, "payload", int(plan["payload_bytes"]))
    place(SECTION_HEADS, "heads", HEADS_BYTES)
    place(SECTION_PAIRS, "pairs", PAIRS_BYTES)
    place(SECTION_LEAVES, "leaves", int(plan["leaf_bytes"]))
    place(SECTION_RELATIONS, "relations+hyperedges", int(plan["relation_bytes"]))
    place(SECTION_HYPEREDGES, "hyperedge-index", PLACEHOLDER_SECTION_BYTES)
    place(SECTION_FEATURES, "features", int(plan["feature_bytes"]))
    place(SECTION_TOKENS, "tokens", PLACEHOLDER_SECTION_BYTES)
    place(SECTION_INDEX, "index", PLACEHOLDER_SECTION_BYTES)
    place(SECTION_JOURNAL, "journal", int(plan["journal_bytes"]))
    plan["sections"] = list(sections.values())
    plan["committed_bytes"] = cursor
    return sections


def _read_header_slot(buffer, index: int) -> dict[str, Any] | None:
    start = index * HEADER_SLOT_BYTES
    raw = bytes(buffer[start:start + HEADER_SLOT_BYTES])
    if len(raw) < HEADER_PREFIX.size:
        return None
    magic, version, length, digest = HEADER_PREFIX.unpack_from(raw, 0)
    if magic != HEADER_MAGIC or version != HEADER_VERSION:
        return None
    if length == 0 or HEADER_PREFIX.size + length > len(raw):
        return None
    body = raw[HEADER_PREFIX.size:HEADER_PREFIX.size + length]
    if hashlib.sha256(body).digest() != digest:
        return None
    try:
        value = _json_loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class ResidentUniverse:
    """Bounded, persistent, one-file resident universe."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        create: bool = False,
        size: int | None = None,
        state_dir: str | os.PathLike[str] | None = None,
        universe_id: str | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ResidentIntegrationError(
                "ResidentUniverse requires the resident sibling modules; "
                "missing or unusable: %s" % _IMPORT_ERROR) from _IMPORT_ERROR
        self._closed = True
        self._fd = -1
        self._mmap = None
        self.path = self._resolve_path(path, state_dir)
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if create:
            self._create(size if size is not None else DEFAULT_BYTES, universe_id)
        else:
            if not self.path.exists():
                if size is not None:
                    self._create(size, universe_id)
                    return
                raise FileNotFoundError(self.path)
            self._open(universe_id)

    # ------------------------------------------------------------- lifecycle
    @staticmethod
    def _resolve_path(path: str | os.PathLike[str] | None,
                      state_dir: str | os.PathLike[str] | None) -> Path:
        if path is not None:
            return Path(path).expanduser()
        root = Path(state_dir).expanduser() if state_dir else \
            Path.home() / ".local" / "share" / "nanlmind-hermes"
        return root / "resident" / "universe.nlmres"

    @classmethod
    def create(cls, path=None, *, size: int = DEFAULT_BYTES,
               state_dir=None, universe_id=None) -> "ResidentUniverse":
        return cls(path, create=True, size=size, state_dir=state_dir,
                   universe_id=universe_id)

    @classmethod
    def open(cls, path=None, size: int | None = None, *,
             create: bool = False, state_dir=None,
             universe_id=None) -> "ResidentUniverse":
        # ``open`` is the unified create-or-open entry point.  A missing file
        # with no explicit budget starts at the documented 256 MiB default.
        if not create and size is None:
            candidate = cls._resolve_path(path, state_dir)
            if not candidate.exists():
                size = DEFAULT_BYTES
        return cls(path, create=create, size=size, state_dir=state_dir,
                   universe_id=universe_id)

    @classmethod
    def open_or_create(cls, path=None, *, size: int = DEFAULT_BYTES,
                       state_dir=None, universe_id=None) -> "ResidentUniverse":
        candidate = cls._resolve_path(path, state_dir)
        if candidate.exists():
            return cls(candidate, create=False)
        return cls(candidate, create=True, size=size, universe_id=universe_id)

    def _open_fd(self, path: Path) -> tuple[int, mmap.mmap, int]:
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        info = os.fstat(fd)
        if info.st_size < HEADER_REGION_BYTES:
            os.close(fd)
            raise ResidentUniverseError("resident file is shorter than its header")
        if info.st_size > MAX_BYTES:
            os.close(fd)
            raise ResidentUniverseError("resident file exceeds the 1 GiB ceiling")
        mapped = mmap.mmap(fd, info.st_size)
        return fd, mapped, info.st_size

    def _create(self, size: int, universe_id: str | None) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ResidentUniverseError("size must be an integer")
        if size > MAX_BYTES:
            raise ResidentUniverseError(
                "resident universe must not exceed %d bytes (1 GiB); "
                "no 2 GiB overbootstrap copy is permitted" % MAX_BYTES)
        try:
            size = clamp_size(size)
        except ValueError as exc:
            raise ResidentUniverseError(str(exc)) from exc
        plan = _plan_for_size(size)
        if plan["used_bytes"] > size:
            raise ResidentUniverseFull("resident profile does not fit the file")
        sections = _assign_sections(plan)
        self._fd = os.open(
            self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC
            | getattr(os, "O_CLOEXEC", 0), 0o600)
        os.ftruncate(self._fd, size)
        self._mmap = mmap.mmap(self._fd, size)
        created_ns = time.time_ns()
        uid = universe_id or hashlib.sha256(
            ("resident:%s:%s" % (self.path, created_ns)).encode()).hexdigest()[:32]
        self._metadata: dict[str, Any] = {
            "schema": UNIVERSE_SCHEMA,
            "magic": MAGIC.decode("ascii", "replace"),
            "version": RESIDENT_VERSION,
            "universe_id": uid,
            "created_ns": created_ns,
            "file_size": size,
            "plan": plan,
            "sections": [sections[key] for key in sorted(sections)],
            "generation": 0,
            "digest": hashlib.sha256(
                ("genesis:%s:%s:%s" % (uid, size, created_ns)).encode()).hexdigest(),
            "applied_candidates": [],
        }
        self._generation = 0
        self._digest = self._metadata["digest"]
        self._genesis_digest = self._digest
        self._history: dict[int, str] = {0: self._digest}
        self._header_seq = 0
        self._header_slot: int | None = None
        self._source_digests: set[str] = set()
        self._applied_candidates: set[str] = set()
        self._payload_by_leaf: dict[str, dict[str, Any]] = {}
        self._prepared_leaf: dict[str, LeafRecord] = {}
        self._leaf_payload: dict[int, dict[str, Any]] = {}
        self._text_index: dict[str, list[int]] = {}
        self._relation_payloads: dict[str, dict[str, Any]] = {}
        self._root_slots: dict[int, int] = {}
        self._dirty = True
        self._closed = False

        self._initialize_regions(sections, plan)
        # Two complete header copies make a torn creation recoverable.
        self._write_header(fsync=False)
        self._write_header(fsync=True)
        self._load_index()

    def _initialize_regions(self, sections: Mapping[int, Mapping[str, Any]],
                            plan: Mapping[str, Any]) -> None:
        payload = sections[SECTION_ARENA]
        cells = sections[SECTION_HEADS]
        leaves = sections[SECTION_LEAVES]
        relations = sections[SECTION_RELATIONS]
        features = sections[SECTION_FEATURES]
        journal = sections[SECTION_JOURNAL]
        self.payload = PayloadArena(
            self._mmap, int(payload["offset"]),
            int(plan["payload_capacity"]), create=True)
        self.cells = ResidentCells.format_region(
            self._mmap, int(cells["offset"]))
        self.leaves = LeafStore.create(
            self._mmap, base_offset=int(leaves["offset"]),
            capacity=int(plan["leaf_capacity"]),
            index_capacity=int(plan["leaf_index_capacity"]),
            feature_region=(
                int(payload["offset"]) + PAYLOAD_HEADER_BYTES,
                int(plan["payload_capacity"]),
            ),
        )
        self.relations = RelationStore.create(
            self._mmap, base_offset=int(relations["offset"]),
            relation_capacity=int(plan["relation_capacity"]),
            hyperedge_capacity=int(plan["hyperedge_capacity"]),
            arena_bytes=int(plan["hyperedge_arena_bytes"])
            if "hyperedge_arena_bytes" in plan else None,
        )
        self._feature_arena = FeatureArena(
            dimensions=int(plan["feature_dim"]),
            capacity=int(plan["feature_capacity"]),
            dtype=str(plan["feature_dtype"]),
            buffer=self._mmap,
            base_offset=int(features["offset"]),
        )
        self.features = FeatureStore(arena=self._feature_arena)
        self.journal = StateJournal(
            self._mmap, int(journal["offset"]),
            int(plan["journal_capacity"]), create=True)

    def _open(self, universe_id: str | None) -> None:
        self._fd, self._mmap, size = self._open_fd(self.path)
        candidates = []
        for index in range(HEADER_SLOTS):
            value = _read_header_slot(self._mmap, index)
            if value is not None:
                candidates.append((int(value.get("header_seq", 0)), index, value))
        if not candidates:
            os.close(self._fd)
            self._mmap.close()
            raise ResidentUniverseError("resident universe has no valid header")
        _seq, slot, metadata = max(candidates, key=lambda item: (item[0], item[1]))
        if metadata.get("schema") != UNIVERSE_SCHEMA:
            raise ResidentUniverseError("unsupported resident universe schema")
        if universe_id is not None and metadata.get("universe_id") != universe_id:
            raise ResidentUniverseError("resident universe id does not match")
        plan = metadata.get("plan")
        if not isinstance(plan, Mapping) or not isinstance(metadata.get("sections"), list):
            raise ResidentUniverseError("resident plan is missing")
        sections: dict[int, dict[str, Any]] = {}
        for item in metadata["sections"]:
            if not isinstance(item, Mapping):
                continue
            section_id = int(item.get("id", 0))
            offset = int(item.get("offset", -1))
            length = int(item.get("length", 0))
            if section_id <= 0 or offset < HEADER_REGION_BYTES or length <= 0:
                raise ResidentUniverseError("invalid resident section entry")
            if offset + length > size:
                raise ResidentUniverseError("resident section exceeds the file")
            sections[section_id] = dict(item)
        required = (SECTION_ARENA, SECTION_HEADS, SECTION_LEAVES,
                    SECTION_RELATIONS, SECTION_FEATURES, SECTION_JOURNAL)
        if any(section_id not in sections for section_id in required):
            raise ResidentUniverseError("resident section table is incomplete")

        self._metadata = dict(metadata)
        self._header_seq = int(metadata.get("header_seq", 0))
        self._header_slot = slot
        self._generation = int(metadata.get("generation", 0))
        self._digest = str(metadata.get("digest") or "")
        self._genesis_digest = str(metadata.get("genesis_digest") or self._digest)
        self._history = {0: self._genesis_digest}
        self._applied_candidates = {
            str(value) for value in metadata.get("applied_candidates", [])
            if isinstance(value, str)
        }
        self._source_digests = set()
        self._payload_by_leaf = {}
        self._prepared_leaf = {}
        self._leaf_payload = {}
        self._text_index = {}
        self._relation_payloads = {}
        self._root_slots = {}
        self._closed = False

        payload = sections[SECTION_ARENA]
        cells = sections[SECTION_HEADS]
        leaves = sections[SECTION_LEAVES]
        relations = sections[SECTION_RELATIONS]
        features = sections[SECTION_FEATURES]
        journal = sections[SECTION_JOURNAL]
        self.payload = PayloadArena(
            self._mmap, int(payload["offset"]),
            int(plan["payload_capacity"]), create=False)
        self.cells = ResidentCells(self._mmap, int(cells["offset"]))
        self.leaves = LeafStore(
            self._mmap, base_offset=int(leaves["offset"]),
            capacity=int(plan["leaf_capacity"]), create=False,
            feature_region=(
                int(payload["offset"]) + PAYLOAD_HEADER_BYTES,
                int(plan["payload_capacity"]),
            ),
        )
        self.relations = RelationStore.attach(
            self._mmap, base_offset=int(relations["offset"]))
        self._feature_arena = FeatureArena(
            dimensions=int(plan["feature_dim"]),
            capacity=int(plan["feature_capacity"]),
            dtype=str(plan["feature_dtype"]),
            buffer=self._mmap,
            base_offset=int(features["offset"]),
        )
        self.features = FeatureStore(arena=self._feature_arena)
        self.journal = StateJournal(
            self._mmap, int(journal["offset"]),
            int(plan["journal_capacity"]), create=False)
        self._replay_journal()
        self._load_index()

    def _replay_journal(self) -> None:
        records = self.journal.scan()
        for item in records:
            sequence = int(item["sequence"])
            digest = str(item["digest"])
            self._generation = max(self._generation, sequence)
            self._digest = digest
            self._history[sequence] = digest
        if records:
            self._generation = int(records[-1]["sequence"])
            self._digest = str(records[-1]["digest"])
        elif not self._digest:
            self._digest = self._genesis_digest
            self._history[0] = self._digest

    # ---------------------------------------------------------------- headers
    def _header_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        metadata.update({
            "schema": UNIVERSE_SCHEMA,
            "header_seq": self._header_seq + 1,
            "generation": int(self._generation),
            "digest": str(self._digest),
            "genesis_digest": str(self._genesis_digest),
            "applied_candidates": sorted(self._applied_candidates),
        })
        return metadata

    def _write_header(self, *, fsync: bool = False) -> None:
        metadata = self._header_metadata()
        body = _json_bytes(metadata)
        if HEADER_PREFIX.size + len(body) > HEADER_SLOT_BYTES:
            raise ResidentUniverseError("resident header exceeds its 4 KiB slot")
        slot = 0 if self._header_slot is None else (self._header_slot + 1) % 2
        start = slot * HEADER_SLOT_BYTES
        prefix = HEADER_PREFIX.pack(
            HEADER_MAGIC, HEADER_VERSION, len(body), hashlib.sha256(body).digest())
        raw = prefix + body
        self._mmap[start:start + len(raw)] = raw
        if len(raw) < HEADER_SLOT_BYTES:
            self._mmap[start + len(raw):start + HEADER_SLOT_BYTES] = \
                b"\x00" * (HEADER_SLOT_BYTES - len(raw))
        self._header_seq += 1
        self._header_slot = slot
        self._metadata = metadata
        if fsync:
            self._mmap.flush()
            os.fsync(self._fd)
        self._dirty = False

    def flush(self) -> None:
        self._ensure_open()
        self._write_header(fsync=False)
        self._mmap.flush()
        os.fsync(self._fd)

    commit = flush

    # ------------------------------------------------------------- properties
    @property
    def universe_id(self) -> str:
        return str(self._metadata["universe_id"])

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def digest(self) -> str:
        return str(self._digest)

    @property
    def genesis_digest(self) -> str:
        return str(self._genesis_digest)

    @property
    def source_digests(self) -> dict[str, str]:
        return {value: value for value in sorted(self._source_digests)}

    @property
    def external_references(self) -> dict[str, list[str]]:
        return {"leaves": [], "relations": [], "records": []}

    @property
    def applied_candidate_digests(self) -> set[str]:
        return set(self._applied_candidates)

    @property
    def file_size(self) -> int:
        return int(self._metadata["file_size"])

    def digest_at(self, generation: int) -> str | None:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("generation must be an integer")
        if generation < 0:
            return None
        if generation >= self._generation:
            return self._digest
        keys = [key for key in self._history if key <= generation]
        if not keys:
            return self._genesis_digest
        return self._history[max(keys)]

    # ---------------------------------------------------------------- indexes
    def _load_index(self) -> None:
        self._payload_by_leaf.clear()
        self._prepared_leaf.clear()
        self._leaf_payload.clear()
        self._text_index.clear()
        self._relation_payloads.clear()
        self._root_slots.clear()
        self._source_digests.clear()
        for record in self.leaves.records():
            payload = self._read_leaf_payload(record)
            if payload is None:
                continue
            self._leaf_payload[record.slot] = payload
            self._payload_by_leaf[record.leaf_id.hex()] = payload
            prepared = payload.get("prepared_id") or payload.get("leaf_id")
            if isinstance(prepared, str):
                self._prepared_leaf[prepared] = record
            source = _valid_digest(payload.get("source_digest"))
            if source:
                self._source_digests.add(source)
            if payload.get("root"):
                self._root_slots[_head(payload.get("head"))] = record.slot
                continue
            text = normalize_text(payload.get("text"))
            if text:
                self._text_index.setdefault(text, []).append(record.slot)
                folded = text.casefold()
                if folded != text:
                    self._text_index.setdefault(folded, []).append(record.slot)
            if payload.get("kind") == "relation":
                if payload.get("relation_store_id"):
                    self._relation_payloads[str(payload["relation_store_id"])] = payload
                if payload.get("prepared_id"):
                    self._relation_payloads[str(payload["prepared_id"])] = payload
        self._rebuild_features()

    def _clear_features(self) -> None:
        # FeatureStore owns an in-memory index; the arena bytes are rebuilt from
        # durable leaf payloads.  Replacing the index is cheaper and safer than
        # trusting a partially written index after a crash.
        self._feature_arena.close()
        plan = self._metadata["plan"]
        self._feature_arena = FeatureArena(
            dimensions=int(plan["feature_dim"]),
            capacity=int(plan["feature_capacity"]),
            dtype=str(plan["feature_dtype"]),
            buffer=self._mmap,
            base_offset=int(self._section(SECTION_FEATURES)["offset"]),
        )
        self.features = FeatureStore(arena=self._feature_arena)

    def _rebuild_features(self) -> None:
        self._clear_features()
        for record in self.leaves.records():
            payload = self._leaf_payload.get(record.slot)
            if not payload or payload.get("root") or payload.get("no_feature"):
                continue
            text = normalize_text(payload.get("text"))
            prepared = payload.get("prepared_id") or payload.get("leaf_id")
            if not text or not isinstance(prepared, str):
                continue
            try:
                self._add_feature(prepared, text, payload.get("features"),
                                  leaf_id=record.leaf_id.hex())
            except (ValueError, MemoryError):
                continue

    def _add_feature(self, feature_id: str, text: str,
                     values: Any = None, *, leaf_id: str | None = None) -> None:
        if not isinstance(feature_id, str) or not feature_id:
            return
        if feature_id in self.features:
            return
        if values is None:
            values = derive_features(
                text, dim=int(self._metadata["plan"]["feature_dim"]),
                dtype=str(self._metadata["plan"]["feature_dtype"]))["values"]
        vector = list(values)
        if len(vector) != int(self._metadata["plan"]["feature_dim"]):
            return
        self.features.add(vector, feature_id=feature_id,
                          leaf_id=leaf_id or feature_id)

    def _read_leaf_payload(self, record: LeafRecord) -> dict[str, Any] | None:
        if not record.feature_length:
            return None
        try:
            raw = self.payload.read(record.feature_offset, record.feature_length)
            value = _json_loads(raw)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    # ------------------------------------------------------------- integrity
    def _ensure_open(self) -> None:
        if self._closed or self._mmap is None:
            raise ResidentUniverseError("resident universe is closed")

    def _section(self, section_id: int) -> dict[str, Any]:
        for item in self._metadata.get("sections", []):
            if int(item.get("id", 0)) == int(section_id):
                return item
        raise ResidentUniverseError("resident section %d is absent" % section_id)

    def validate(self) -> dict[str, Any]:
        problems: list[str] = []
        for section_id, name in (
            (SECTION_ARENA, "payload"), (SECTION_HEADS, "cells"),
            (SECTION_LEAVES, "leaves"), (SECTION_RELATIONS, "relations"),
            (SECTION_FEATURES, "features"), (SECTION_JOURNAL, "journal"),
        ):
            try:
                item = self._section(section_id)
            except ResidentUniverseError:
                problems.append("missing section: " + name)
                continue
            if int(item["offset"]) % ALIGN:
                problems.append("unaligned section: " + name)
            if int(item["offset"]) + int(item["length"]) > self.file_size:
                problems.append("section exceeds file: " + name)
        if self.leaves.count > self.leaves.capacity:
            problems.append("leaf count exceeds capacity")
        if self.relations.status()["relations"] > self.relations.status()["relation_capacity"]:
            problems.append("relation count exceeds capacity")
        return {
            "schema": UNIVERSE_SCHEMA,
            "ok": not problems,
            "problems": problems,
            "generation": self._generation,
            "digest": self._digest,
        }

    # ----------------------------------------------------------------- commit
    def _commit(self, kind: str, identity: Any, *,
                generation: int | None = None) -> int:
        material = kind.encode("ascii", "replace") + b"\x00" + \
            (identity.encode("utf-8") if isinstance(identity, str)
             else bytes(identity))
        new_digest = hashlib.sha256(
            bytes.fromhex(self._digest) + material).digest()
        if generation is None:
            generation = self._generation + 1
        if isinstance(generation, bool) or not isinstance(generation, int) \
                or generation < 0:
            raise ResidentUniverseError("generation must be non-negative")
        self.journal.append(kind, new_digest, generation=generation)
        self._generation = generation
        self._digest = new_digest.hex()
        self._history[generation] = self._digest
        self._dirty = True
        return generation

    # --------------------------------------------------------------- payloads
    def _payload_for(self, record: Mapping[str, Any],
                     prepared_id: str | None = None) -> dict[str, Any]:
        payload = {key: value for key, value in record.items()
                   if key not in {"features"}}
        payload["prepared_id"] = prepared_id or payload.get("prepared_id") \
            or payload.get("leaf_id") or payload.get("relation_id") \
            or payload.get("record_id") or hashlib.sha256(
                _json_bytes(payload)).hexdigest()[:32]
        return payload

    # ----------------------------------------------------------------- leaves
    def _ensure_root(self, head: int) -> LeafRecord:
        head = _head(head)
        existing = self._root_slots.get(head)
        if existing is not None:
            try:
                return self.leaves.get(existing)
            except Exception:
                pass
        prepared_id = "root:%d" % head
        payload = {
            "schema": UNIVERSE_SCHEMA, "prepared_id": prepared_id,
            "kind": "data", "head": head, "text": "head %d" % head,
            "root": True, "immutable": True,
            "source_digest": hashlib.sha256(
                ("root:%s:%d" % (self.universe_id, head)).encode()).hexdigest(),
        }
        raw = _json_bytes(payload)
        offset = self.payload.append(raw)
        leaf_id = hashlib.sha256(("root:%s:%d" % (self.universe_id, head)).encode()).digest()[:16]
        record = self.leaves.insert(
            "data", leaf_id=leaf_id, content=raw,
            payload_digest=hashlib.sha256(raw).digest(),
            head=head, feature_offset=offset, feature_length=len(raw),
        )
        self._leaf_payload[record.slot] = payload
        self._payload_by_leaf[record.leaf_id.hex()] = payload
        self._root_slots[head] = record.slot
        self._prepared_leaf[prepared_id] = record
        self._source_digests.add(str(payload["source_digest"]))
        self.cells.update_head(head, score_delta=0.0, state_delta=0,
                               leaf_chain=record.slot + 1)
        return record

    def _find_leaf(self, prepared_id: str | None = None,
                   leaf_id: str | None = None) -> LeafRecord | None:
        if prepared_id is not None and prepared_id in self._prepared_leaf:
            return self.leaves.get(self._prepared_leaf[prepared_id].slot)
        reference = leaf_id if leaf_id is not None else prepared_id
        if reference is not None:
            try:
                return self.leaves.get(_leaf_id_bytes(reference))
            except Exception:
                return None
        return None

    def _append_leaf(self, record: Mapping[str, Any],
                     source_digest: str | None = None) -> dict[str, Any]:
        prepared_id = str(
            record.get("leaf_id") or record.get("prepared_id")
            or record.get("record_id") or hashlib.sha256(
                _json_bytes(dict(record))).hexdigest()[:32])
        leaf_id = _leaf_id_bytes(prepared_id)
        existing = self.leaves.find(leaf_id)
        if existing is not None:
            return {
                "status": "duplicate", "created": False, "duplicate": True,
                "leaf_id": existing.leaf_id.hex(), "slot": existing.slot,
                "generation": self._generation,
                "source_digest": source_digest or _source_digest(record),
            }
        payload = self._payload_for(record, prepared_id)
        if source_digest:
            payload["source_digest"] = source_digest
        raw = _json_bytes(payload)
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise ResidentUniverseError("leaf payload exceeds the bounded arena")
        offset = self.payload.append(raw)
        head = _head(record.get("head"))
        kind = _kind(record.get("kind"))
        try:
            leaf = self.leaves.insert(
                kind, leaf_id=leaf_id, content=raw,
                payload_digest=hashlib.sha256(raw).digest(), head=head,
                feature_offset=offset, feature_length=len(raw),
            )
        except Exception:
            raise
        self._payload_by_leaf[leaf.leaf_id.hex()] = payload
        self._leaf_payload[leaf.slot] = payload
        self._prepared_leaf[prepared_id] = leaf
        source = _valid_digest(payload.get("source_digest"))
        if source:
            self._source_digests.add(source)
        text = normalize_text(payload.get("text"))
        if text:
            self._text_index.setdefault(text, []).append(leaf.slot)
            folded = text.casefold()
            if folded != text:
                self._text_index.setdefault(folded, []).append(leaf.slot)
        if kind != "relation" and text:
            try:
                self._add_feature(prepared_id, text, payload.get("features"),
                                  leaf_id=leaf.leaf_id.hex())
            except (ValueError, MemoryError):
                # Feature rows are bounded.  A full resonance arena must not
                # roll back an otherwise durable leaf/relation.
                pass
        try:
            root = self._ensure_root(head)
            self.leaves.link(root.slot, leaf.slot)
        except Exception:
            # A leaf can already be attached by a prior import.  Tree linking
            # remains best-effort; relation membership is authoritative.
            pass
        self.cells.update_head(
            head, score_delta=1.0, state_delta=1,
            leaf_chain=leaf.slot + 1)
        self._commit("leaf", leaf.leaf_id.hex())
        return {
            "status": "appended", "created": True, "duplicate": False,
            "leaf_id": leaf.leaf_id.hex(), "slot": leaf.slot,
            "kind": kind, "head": head, "generation": self._generation,
            "source_digest": source or source_digest,
            "prepared_id": prepared_id,
        }

    def _ensure_leaf_text(self, text: str, *, kind: str = "concept",
                          head: int = 0, source_digest: str | None = None,
                          prepared_id: str | None = None) -> LeafRecord:
        text = normalize_text(text)
        if not text:
            raise ResidentUniverseError("leaf text must not be empty")
        pid = prepared_id or ("text:" + hashlib.sha256(text.encode()).hexdigest()[:32])
        found = self._find_leaf(pid)
        if found is not None:
            return found
        result = self._append_leaf({
            "type": "leaf", "leaf_id": pid, "kind": kind, "head": head,
            "text": text, "source_digest": source_digest or _source_digest({
                "text": text, "record_id": pid,
            }),
        })
        record = self.leaves.get(_leaf_id_bytes(pid))
        return record

    def append_leaf(self, record: Mapping[str, Any],
                    source_digest: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        result = self._append_leaf(dict(record), source_digest)
        self.flush()
        return result

    intern_leaf = append_leaf
    ingest_leaf = append_leaf
    create_leaf = append_leaf
    store_leaf = append_leaf
    add_leaf = append_leaf

    # -------------------------------------------------------------- relations
    def _relation_payload(self, record: Mapping[str, Any], relation_result,
                          source: str) -> dict[str, Any]:
        subject = normalize_text(record.get("subject"))
        predicate = normalize_text(record.get("predicate") or record.get("role"))
        obj = normalize_text(record.get("object"))
        text = normalize_text(record.get("text")) or \
            " ".join(part for part in (subject, predicate, obj) if part)
        payload = {
            "schema": UNIVERSE_SCHEMA,
            "prepared_id": str(record.get("relation_id") or record.get("record_id")
                               or relation_result.get("relation_id")),
            "relation_store_id": str(relation_result.get("relation_id")),
            "kind": "relation", "text": text,
            "subject": subject, "predicate": predicate, "object": obj,
            "subject_leaf_id": record.get("subject_leaf_id"),
            "predicate_leaf_id": record.get("predicate_leaf_id"),
            "object_leaf_id": record.get("object_leaf_id"),
            "source_digest": source,
            "confidence": _confidence(record.get("confidence")),
            "polarity": str(record.get("polarity") or "neutral"),
            "features": record.get("features"),
            "feature_dim": record.get("feature_dim"),
            "feature_dtype": record.get("feature_dtype"),
            "record": {key: value for key, value in record.items()
                       if key not in {"features"}},
        }
        return payload

    def _append_relation(self, record: Mapping[str, Any],
                         source_digest: str | None = None) -> dict[str, Any]:
        source = _valid_digest(source_digest) or _source_digest(record)
        subject = normalize_text(record.get("subject"))
        predicate = normalize_text(record.get("predicate") or record.get("role"))
        obj = normalize_text(record.get("object"))
        if not (subject and predicate and obj):
            return self._append_leaf(record, source)
        head = _head(record.get("head"))
        subject_leaf = self._find_leaf(record.get("subject_leaf_id"))
        predicate_leaf = self._find_leaf(record.get("predicate_leaf_id"))
        object_leaf = self._find_leaf(record.get("object_leaf_id"))
        if subject_leaf is None:
            subject_leaf = self._ensure_leaf_text(
                subject, kind="concept", head=head, source_digest=source,
                prepared_id=record.get("subject_leaf_id"))
        if predicate_leaf is None:
            predicate_leaf = self._ensure_leaf_text(
                predicate, kind="concept", head=head, source_digest=source,
                prepared_id=record.get("predicate_leaf_id"))
        if object_leaf is None:
            object_leaf = self._ensure_leaf_text(
                obj, kind="concept", head=head, source_digest=source,
                prepared_id=record.get("object_leaf_id"))
        relation_id = str(
            record.get("relation_id") or record.get("record_id")
            or hashlib.sha256(("rel:" + subject + "\x1f" + predicate
                               + "\x1f" + obj).encode()).hexdigest())
        # Bind one canonical pair cell and its reverse16 mirror into the
        # relation record/hyperedge.  A single routed head cannot pair with
        # itself, so the object leaf id selects a deterministic partner head.
        pair_head = _head(record.get("head"))
        pair_seed = object_leaf.leaf_id.hex()
        pair_partner = int(hashlib.sha256(
            pair_seed.encode("utf-8")).hexdigest()[:8], 16) % HEAD_COUNT
        if pair_partner == pair_head:
            pair_partner = (pair_head + 1) % HEAD_COUNT
        try:
            pair_id = int(self.cells.pair_index(pair_head, pair_partner))
            mirror_id = int(self.cells.pair_state(
                pair_head, pair_partner).mirror)
        except Exception:
            pair_id = mirror_id = None
        result = self.relations.append_relation(
            subject=_relation_leaf_ref(subject_leaf.leaf_id.hex()),
            predicate=_relation_leaf_ref(predicate_leaf.leaf_id.hex()),
            object=_relation_leaf_ref(object_leaf.leaf_id.hex()),
            polarity=record.get("polarity") or "neutral",
            confidence=_confidence(record.get("confidence")),
            pair_id=pair_id,
            mirror_id=mirror_id,
            source_digest=source,
            relation_id=_relation_id_bytes(relation_id),
        )
        if result.get("duplicate"):
            return {
                "status": "duplicate", "created": False, "duplicate": True,
                "relation_id": result.get("relation_id"),
                "generation": self._generation, "source_digest": source,
            }
        evidence_id = "relleaf:" + hashlib.sha256(
            str(result["relation_id"]).encode()).hexdigest()[:32]
        payload = self._relation_payload(record, result, source)
        payload["relation_leaf_id"] = evidence_id
        raw = _json_bytes(payload)
        offset = self.payload.append(raw)
        evidence_leaf_id = _leaf_id_bytes(evidence_id)
        evidence = self.leaves.find(evidence_leaf_id)
        if evidence is None:
            try:
                evidence = self.leaves.insert(
                    "relation", leaf_id=evidence_leaf_id, content=raw,
                    payload_digest=hashlib.sha256(raw).digest(), head=head,
                    feature_offset=offset, feature_length=len(raw))
                self._leaf_payload[evidence.slot] = payload
                self._payload_by_leaf[evidence.leaf_id.hex()] = payload
                try:
                    self.leaves.link(subject_leaf.slot, evidence.slot)
                except Exception:
                    pass
                try:
                    self.leaves.attach_relation(subject_leaf.slot, evidence.slot)
                except Exception:
                    pass
            except Exception:
                evidence = None
        self._relation_payloads[str(result["relation_id"])] = payload
        self._relation_payloads[str(payload["prepared_id"])] = payload
        if evidence is not None:
            self._prepared_leaf[str(payload["prepared_id"])] = evidence
        self._source_digests.add(source)
        try:
            self._add_feature(payload["prepared_id"], payload["text"],
                              record.get("features"),
                              leaf_id=subject_leaf.leaf_id.hex())
        except (ValueError, MemoryError):
            pass
        heads = []
        for leaf in (subject_leaf, predicate_leaf, object_leaf):
            if leaf.head is not None and leaf.head not in heads:
                heads.append(leaf.head)
        for index, first in enumerate(heads):
            for second in heads[index + 1:]:
                try:
                    self.cells.observe_pair(
                        first, second,
                        weight=_confidence(record.get("confidence")),
                        relation_id=int.from_bytes(
                            _relation_id_bytes(relation_id)[:4], "little"))
                except Exception:
                    pass
        self.cells.update_head(
            head, score_delta=_confidence(record.get("confidence")),
            state_delta=1, relation_chain=result.get("relation", {}).get("index", 0) + 1)
        self._commit("relation", str(result["relation_id"]))
        return {
            "status": "appended", "created": True, "duplicate": False,
            "relation_id": result.get("relation_id"),
            "hyperedge_id": result.get("hyperedge_id"),
            "generation": self._generation, "source_digest": source,
            "subject_leaf_id": subject_leaf.leaf_id.hex(),
            "object_leaf_id": object_leaf.leaf_id.hex(),
            "prepared_id": payload["prepared_id"],
        }

    def append_relation(self, record: Mapping[str, Any],
                        source_digest: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        result = self._append_relation(dict(record), source_digest)
        self.flush()
        return result

    ingest_relation = append_relation
    add_relation = append_relation

    # --------------------------------------------------------------- ingestion
    def append_record(self, record: Mapping[str, Any],
                      source_digest: str | None = None) -> dict[str, Any]:
        item = dict(record)
        if item.get("type") == "relation" or (
                item.get("subject") and item.get("predicate")
                and item.get("object") and item.get("type") != "leaf"):
            return self._append_relation(item, source_digest)
        return self._append_leaf(item, source_digest)

    ingest_record = append_record

    def ingest_records(self, records: Iterable[Mapping[str, Any]],
                       source_digest: str | None = None) -> dict[str, Any]:
        """Append a bounded prepared batch and flush once at the end."""
        self._ensure_open()
        accepted = duplicates = 0
        receipts: list[dict[str, Any]] = []
        for record in records:
            result = self.append_record(record, source_digest)
            if result.get("duplicate"):
                duplicates += 1
            else:
                accepted += 1
            receipts.append(result)
        if receipts:
            self.flush()
        return {
            "accepted": accepted, "duplicates": duplicates,
            "receipts": receipts, "generation": self._generation,
        }

    append_records = ingest_records
    ingest_batch = ingest_records

    def ingest_rows(self, rows: Iterable[Mapping[str, Any]], *,
                    source_digest: str | None = None,
                    feature_dim: int = DEFAULT_FEATURE_DIM,
                    feature_dtype: str = DEFAULT_FEATURE_DTYPE) -> dict[str, Any]:
        prepared = prepare_records(
            rows, source_digest=source_digest,
            feature_dim=feature_dim, feature_dtype=feature_dtype)
        if not prepared:
            return {"accepted": 0, "duplicates": 0, "records": 0,
                    "generation": self._generation}
        result = self.ingest_records(prepared)
        result["records"] = len(prepared)
        return result

    def ingest_jsonl(self, path: str | os.PathLike[str], *,
                     limit: int = 1000, source_digest: str | None = None
                     ) -> dict[str, Any]:
        from .ingest import iter_jsonl
        rows = []
        for index, row in enumerate(iter_jsonl(path)):
            if index >= max(1, min(int(limit), 256)):
                break
            rows.append(row)
        result = self.ingest_rows(rows, source_digest=source_digest)
        result["source"] = str(path)
        return result

    # ------------------------------------------------------------------ reads
    def _leaf_view(self, leaf: LeafRecord) -> dict[str, Any]:
        payload = self._leaf_payload.get(leaf.slot) or \
            self._read_leaf_payload(leaf) or {}
        return {
            "leaf_id": leaf.leaf_id.hex(),
            "slot": leaf.slot,
            "kind": leaf.kind,
            "head": leaf.head,
            "prepared_id": payload.get("prepared_id"),
            "text": payload.get("text"),
            "source_digest": payload.get("source_digest"),
            "parent": leaf.parent,
            "children": len(self.leaves.children(leaf.slot)) if leaf.slot < self.leaves.count else 0,
            "relation_count": leaf.relation_count,
        }

    def _bounded(self, value: Any, maximum: int = MAX_LOOKUP_LIMIT) -> int:
        if isinstance(value, bool):
            raise ValueError("limit must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError("limit must be an integer")
        return max(1, min(number, maximum))

    def lookup(self, *, text: str | None = None,
               subject: str | None = None, predicate: str | None = None,
               object: str | None = None, head: int | None = None,
               limit: int = MAX_LOOKUP_LIMIT) -> dict[str, Any]:
        self._ensure_open()
        limit = self._bounded(limit)
        leaves: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        if text is not None:
            key = normalize_text(text)
            slots = list(dict.fromkeys(self._text_index.get(key, [])))
            for slot in slots[:limit]:
                leaf = self.leaves.leaf_at(slot)
                if leaf is not None:
                    leaves.append(self._leaf_view(leaf))
        elif subject or predicate or object:
            anchor = normalize_text(subject or object or predicate)
            for slot in self._text_index.get(anchor, [])[:limit]:
                leaf = self.leaves.leaf_at(slot)
                if leaf is None:
                    continue
                payload = self._leaf_payload.get(slot) or {}
                prepared = payload.get("prepared_id")
                if not isinstance(prepared, str):
                    continue
                ref = _relation_leaf_ref(_leaf_id_bytes(prepared).hex())
                for role in ("subject", "predicate", "object"):
                    try:
                        walked = self.relations.chain_walk(ref, role=role, limit=limit)
                    except Exception:
                        continue
                    for relation in walked.get("relations", []):
                        view = self._relation_view(relation)
                        subject_text = normalize_text(
                            view.get("subject_text") or view.get("subject"))
                        object_text = normalize_text(
                            view.get("object_text") or view.get("object"))
                        predicate_text = normalize_text(
                            view.get("predicate_text") or view.get("predicate"))
                        if subject and normalize_text(subject) not in {
                                subject_text, object_text}:
                            continue
                        if object and object_text != normalize_text(object):
                            continue
                        if predicate and predicate_text != normalize_text(predicate):
                            continue
                        relations.append(view)
                        if len(relations) >= limit:
                            break
                    if len(relations) >= limit:
                        break
        elif head is not None:
            wanted = _head(head)
            for leaf in self.leaves.records():
                if leaf.head == wanted:
                    leaves.append(self._leaf_view(leaf))
                    if len(leaves) >= limit:
                        break
        else:
            for leaf in self.leaves.records():
                leaves.append(self._leaf_view(leaf))
                if len(leaves) >= limit:
                    break
        return {
            "schema": UNIVERSE_SCHEMA,
            "generation": self._generation,
            "digest": self._digest,
            "leaves": leaves[:limit],
            "relations": relations[:limit],
        }

    def _relation_view(self, relation: Mapping[str, Any]) -> dict[str, Any]:
        view = dict(relation)
        payload = self._relation_payloads.get(str(view.get("relation_id")))
        if payload:
            view.update({
                "subject_text": payload.get("subject"),
                "predicate_text": payload.get("predicate"),
                "object_text": payload.get("object"),
                "text": payload.get("text"),
                "source_digest": payload.get("source_digest"),
                "confidence": payload.get("confidence"),
            })
        return view

    def query(self, text: str | Iterable[float], *, top_k: int = 8,
              mode: str = "cosine", phase: float = 0.0) -> dict[str, Any]:
        self._ensure_open()
        top_k = self._bounded(top_k, MAX_QUERY_TOP_K)
        if isinstance(text, str):
            vector = derive_features(
                text, dim=int(self._metadata["plan"]["feature_dim"]),
                dtype=str(self._metadata["plan"]["feature_dtype"]))["values"]
        else:
            vector = list(text)
        hits: list[ResonanceHit] = self.features.query(
            vector, top_k=top_k, mode=mode, phase=phase)
        rendered = []
        for hit in hits:
            payload = self._relation_payloads.get(str(hit.feature_id)) or \
                self._payload_by_leaf.get(str(hit.leaf_id)) or {}
            if not payload:
                payload = self._payload_for(
                    {"prepared_id": hit.feature_id,
                     "text": str(hit.feature_id)}, str(hit.feature_id))
            rendered.append({
                "feature_id": hit.feature_id,
                "leaf_id": hit.leaf_id,
                "score": round(float(hit.score), 8),
                "cell": hit.cell,
                "offset": hit.offset,
                "kind": payload.get("kind"),
                "text": payload.get("text"),
                "subject": payload.get("subject"),
                "predicate": payload.get("predicate"),
                "object": payload.get("object"),
                "source_digest": payload.get("source_digest"),
            })
        return {
            "schema": UNIVERSE_SCHEMA,
            "generation": self._generation,
            "digest": self._digest,
            "mode": mode,
            "top_k": top_k,
            "hits": rendered,
        }

    def pair_state(self, left: int, right: int) -> dict[str, Any]:
        state = self.cells.pair_state(int(left), int(right))
        return {
            "index": state.index,
            "canonical": state.canonical,
            "mirror": state.mirror,
            "s_plus": {
                "weight": state.forward.weight,
                "amplitude": state.forward.amplitude,
                "phase": state.forward.phase,
                "generation": state.forward.generation,
            },
            "s_minus": {
                "weight": state.reverse.weight,
                "amplitude": state.reverse.amplitude,
                "phase": state.reverse.phase,
                "generation": state.reverse.generation,
            },
        }

    def cycle_field(self):
        """Return the raw-word oriented cycle view over these pair rails."""
        from .cycle_observable import CycleField
        cache = getattr(self, "_cycle_field_object", None)
        if cache is None:
            cache = CycleField(self.cells)
            self._cycle_field_object = cache
        return cache

    def full_cycle_memory(self):
        """Return the complete 32,385-dimensional root-star cycle memory."""
        from .full_cycle_memory import FullCycleMemory
        cache = getattr(self, "_full_cycle_object", None)
        if cache is None:
            cache = FullCycleMemory(self.cells)
            self._full_cycle_object = cache
        return cache

    def evidence_lanes(self):
        """Return the independent support/refute ledger over cycle memory."""
        from .evidence_lanes import EvidenceLanes
        cache = getattr(self, "_evidence_lanes_object", None)
        if cache is None:
            cache = EvidenceLanes(self.full_cycle_memory())
            self._evidence_lanes_object = cache
        return cache

    def evidence_session(self):
        """Return the bounded add/ask session over the evidence lanes."""
        from .evidence_session import EvidenceSession
        cache = getattr(self, "_evidence_session_object", None)
        if cache is None:
            cache = EvidenceSession(self.evidence_lanes())
            self._evidence_session_object = cache
        return cache

    def directed_weight(self, left: int, right: int) -> float:
        """Signed Q16.16 pair weight in the ``left -> right`` direction."""
        state = self.cells.pair_state(int(left), int(right))
        value = (state.forward.weight
                 if address_of(int(left), int(right)) == state.canonical
                 else state.reverse.weight)
        return value / float(1 << 16)

    def cycle_flux(self, left: int, right: int) -> float:
        """Round-23 fundamental circulation relative to head 0.

        The flux is derived from the live antisymmetric pair state; it is not
        an independently invented fact or a quantum observable.
        """
        def anti(a: int, b: int) -> float:
            return 0.5 * (self.directed_weight(a, b)
                          - self.directed_weight(b, a))
        return anti(0, int(left)) + anti(int(left), int(right)) \
            - anti(0, int(right))

    def triangle_flux(self, a: int, b: int, c: int) -> float:
        def anti(x: int, y: int) -> float:
            return 0.5 * (self.directed_weight(x, y)
                          - self.directed_weight(y, x))
        return anti(int(a), int(b)) + anti(int(b), int(c)) \
            + anti(int(c), int(a))

    def hyperedge_members(self, hyperedge: Any,
                          limit: int = 64) -> dict[str, Any]:
        """Round-24 bounded incidence traversal for one hyperedge."""
        value = self.relations.get_hyperedge(hyperedge)
        if value is None:
            return {"hyperedge_id": str(hyperedge), "members": [],
                    "missing": True}
        members = list(value.get("members", []))[:self._bounded(limit)]
        return {
            "hyperedge_id": value.get("hyperedge_id"),
            "arity": value.get("arity"),
            "members": members,
            "truncated": len(value.get("members", [])) > len(members),
        }

    def head_status(self, head: int | None = None) -> dict[str, Any]:
        if head is not None:
            record = self.cells.read_head(_head(head))
            return {
                "head": record.head_id,
                "identity": record.identity,
                "score": record.score,
                "state": record.state,
                "generation": record.generation,
                "leaf_chain": record.leaf_chain,
                "relation_chain": record.relation_chain,
            }
        values = []
        for index in range(HEAD_COUNT):
            record = self.cells.read_head(index)
            values.append({
                "head": index,
                "score": record.score,
                "state": record.state,
                "leaf_chain": record.leaf_chain,
                "relation_chain": record.relation_chain,
            })
        return {"heads": values, "count": len(values)}

    # ------------------------------------------------------------- export/intake
    def delta_records(self, since_generation: int = 0) -> dict[str, list[dict[str, Any]]]:
        if isinstance(since_generation, bool) or not isinstance(since_generation, int):
            raise ValueError("since_generation must be an integer")
        if since_generation >= self._generation:
            return {"leaves": [], "relations": [], "pairs": [], "hyperedges": []}
        leaves = []
        for leaf in self.leaves.records():
            payload = self._leaf_payload.get(leaf.slot) or {}
            if payload.get("root"):
                continue
            item = {
                "id": leaf.leaf_id.hex(), "kind": leaf.kind,
                "head": leaf.head,
                "text": payload.get("text") or leaf.leaf_id.hex(),
                "source_digest": payload.get("source_digest"),
            }
            leaves.append(item)
        relations = []
        for relation in self.relations.relations():
            rid = str(relation.get("relation_id"))
            payload = self._relation_payloads.get(rid) or {}
            subject = payload.get("subject") or relation.get("subject")
            predicate = payload.get("predicate") or "related"
            obj = payload.get("object") or relation.get("object")
            item = {
                "id": rid,
                "subject": str(subject),
                "predicate": str(predicate),
                "object": str(obj),
                "source_digest": payload.get("source_digest")
                or relation.get("source_digest"),
                "polarity": relation.get("polarity"),
                "confidence": relation.get("confidence"),
            }
            for field in ("subject_leaf_id", "predicate_leaf_id",
                          "object_leaf_id", "relation_leaf_id"):
                value = payload.get(field)
                if not isinstance(value, str):
                    continue
                resident_leaf = self._find_leaf(value)
                item[field] = (resident_leaf.leaf_id.hex()
                               if resident_leaf is not None else value)
            evidence_leaf = self._find_leaf(payload.get("relation_leaf_id"))
            if evidence_leaf is not None:
                item["leaf_id"] = evidence_leaf.leaf_id.hex()
            relations.append(item)
        return {"leaves": leaves, "relations": relations,
                "pairs": [], "hyperedges": []}

    def export_candidate(self, destination, *, include_full: bool = False,
                         since_generation: int = 0,
                         max_bytes: int | None = None) -> dict[str, Any]:
        self.flush()
        return resident_export.export_candidate(
            self, destination, include_full=include_full,
            since_generation=since_generation, max_bytes=max_bytes)

    export = export_candidate

    def link(self, parent, child) -> dict[str, Any]:
        """Attach a child leaf under a parent leaf and commit the tree edge."""
        self._ensure_open()
        record = self.leaves.link(parent, child)
        self._commit("tree", "%s:%s" % (parent, child))
        self.flush()
        return {
            "parent": parent, "child": child,
            "child_slot": record.slot, "generation": self._generation,
        }

    link_leaves = link

    def has_candidate(self, candidate_digest: str) -> bool:
        return str(candidate_digest) in self._applied_candidates

    def apply_candidate(self, records: Mapping[str, Iterable[Mapping[str, Any]]],
                        manifest: Mapping[str, Any], *, approved: bool = False,
                        expected_generation: int | None = None,
                        target_generation: int | None = None,
                        candidate_digest: str | None = None) -> dict[str, Any]:
        if approved is not True:
            return {"accepted": False, "reasons": ["approval required"]}
        if expected_generation is not None and expected_generation != self._generation:
            return {"accepted": False, "status": "stale_generation",
                    "reasons": ["generation compare-and-swap failed"]}
        candidate = candidate_digest or str(manifest.get("manifest_digest") or "")
        if candidate and self.has_candidate(candidate):
            return {"accepted": False, "status": "duplicate",
                    "reasons": ["candidate already applied"]}
        appended = 0
        for section in ("leaves", "relations", "pairs", "hyperedges"):
            for record in records.get(section, ()) or ():
                if section == "leaves":
                    item = dict(record)
                    item.setdefault("type", "leaf")
                    item.setdefault("leaf_id", item.get("id"))
                    self.append_record(item)
                    appended += 1
                elif section == "relations":
                    item = dict(record)
                    item.setdefault("type", "relation")
                    item.setdefault("record_id", item.get("id"))
                    # Export relations carry text, not resident leaf ids.
                    self.append_record(item)
                    appended += 1
        if candidate:
            self._applied_candidates.add(candidate)
        commit_generation = target_generation if isinstance(target_generation, int) \
            and not isinstance(target_generation, bool) else self._generation + 1
        self._commit("candidate", candidate or self.digest,
                     generation=commit_generation)
        self.flush()
        return {
            "accepted": True, "applied": appended,
            "generation": self._generation, "candidate_digest": candidate,
            "canonical_commit": False,
        }

    def integrate_local_corpus(self, records, manifest=None, *,
                               approved: bool = False, purpose=None,
                               candidate_digest=None) -> dict[str, Any]:
        if approved is not True:
            return {"accepted": False, "reasons": ["approval required"]}
        appended = 0
        for section in ("leaves", "relations"):
            for record in (records or {}).get(section, ()):
                item = dict(record)
                if section == "leaves":
                    item.setdefault("type", "leaf")
                    item.setdefault("leaf_id", item.get("id"))
                else:
                    item.setdefault("type", "relation")
                    item.setdefault("record_id", item.get("id"))
                self.append_record(item)
                appended += 1
        if candidate_digest:
            self._applied_candidates.add(str(candidate_digest))
        return {
            "accepted": True, "applied": appended,
            "purpose": purpose, "canonical_commit": False,
        }

    # ------------------------------------------------------------- checkpointing
    def checkpoint(self, directory=None) -> dict[str, Any]:
        self.flush()
        root = Path(directory).expanduser() if directory else \
            self.path.with_name(self.path.name + ".checkpoints")
        manifest = {
            "schema": "nanlm.resident-checkpoint-manifest/1",
            "note": "local-derived resident universe; canonical merge disabled",
            "files": [{"path": os.path.abspath(self.path)}],
        }
        result = resident_persistence.checkpoint(
            root, max(1, self._generation), manifest)
        return {
            "schema": "nanlm.resident-universe-checkpoint/1",
            "path": str(root), "generation": result.generation,
            "slot": result.slot, "digest": result.digest,
            "canonical_commit": False,
        }

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        plan = self._metadata.get("plan", {})
        section_views = []
        for item in self._metadata.get("sections", []):
            section_views.append({
                "id": int(item["id"]), "name": item.get("name"),
                "offset": int(item["offset"]), "length": int(item["length"]),
            })
        return {
            "schema": UNIVERSE_SCHEMA,
            "universe_id": self.universe_id,
            "path": str(self.path),
            "generation": self._generation,
            "digest": self._digest,
            "profile": plan.get("profile"),
            "file_bytes": self.file_size,
            "committed_bytes": int(plan.get("used_bytes", self.file_size)),
            "default_bytes": DEFAULT_BYTES,
            "hard_ceiling_bytes": MAX_BYTES,
            "sections": section_views,
            "counts": {
                "leaves": self.leaves.count,
                "leaf_capacity": self.leaves.capacity,
                "relations": len(self.relations),
                "relation_capacity": self.relations.status()["relation_capacity"],
                "hyperedges": self.relations.status()["hyperedges"],
                "features": len(self.features),
                "feature_capacity": self.features.capacity,
                "heads": HEAD_COUNT,
                "pairs": PAIR_COUNT,
                "root_cycles": (HEAD_COUNT - 1) * (HEAD_COUNT - 2) // 2,
                "cycle_words": 32,
            },
            "memory": {
                "payload": self.payload.stats(),
                "journal": self.journal.stats(),
                "features": self.features.stats(),
                "cells_bytes": CELLS_BYTES,
            },
            "canonical_merge": {"enabled": False, "attempted": False},
            "resident": True,
            "memory_mapped": True,
        }

    stats = status

    # ------------------------------------------------------------------- close
    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._feature_arena.close()
            except Exception:
                pass
            # RelationStore keeps a memoryview over the caller buffer for its
            # lifetime.  Release it before closing the mmap; otherwise CPython
            # raises BufferError for the exported pointer.
            try:
                view = getattr(self.relations, "_view", None)
                if view is not None:
                    view.release()
            except Exception:
                pass
            try:
                self._mmap.close()
            finally:
                os.close(self._fd)
                self._closed = True

    def __enter__(self) -> "ResidentUniverse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            if not self._closed:
                self.close()
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# Integration entry points used by ingest/CLI/plugin
# ---------------------------------------------------------------------------


def open_universe(path=None, *, create: bool = False,
                  size: int | None = None, state_dir=None,
                  universe_id=None) -> ResidentUniverse:
    """Open or create a resident universe for the CLI and ingest loaders."""
    if create:
        candidate = ResidentUniverse._resolve_path(path, state_dir)
        if candidate.exists():
            try:
                return ResidentUniverse(candidate, create=False)
            except Exception:
                return ResidentUniverse(candidate, create=True, size=size,
                                        universe_id=universe_id)
        return ResidentUniverse(candidate, create=True,
                                size=size if size is not None else DEFAULT_BYTES,
                                universe_id=universe_id)
    return ResidentUniverse(path, create=False, size=size,
                            state_dir=state_dir, universe_id=universe_id)


# ---------------------------------------------------------------------------
# Unified facade compatibility surface
# ---------------------------------------------------------------------------
#
# The integration implementation above is also used by the CLI, candidate
# export and the corpus loop.  The helpers below expose the smaller, stable
# facade promised to callers (``ingest``/``get_leaf``/``get_relation``/
# ``snapshot``) without duplicating any resident section implementation.

from . import (
    SECTION_HEADER as _SECTION_HEADER,
    SECTION_HYPEREDGES as _SECTION_HYPEREDGES,
    SECTION_INDEX as _SECTION_INDEX,
    SECTION_PAIRS as _SECTION_PAIRS,
    SECTION_TOKENS as _SECTION_TOKENS,
)


FEATURE_DIMENSIONS = DEFAULT_FEATURE_DIMENSIONS
FEATURE_DTYPE = DEFAULT_FEATURE_DTYPE
KNOWN_SECTIONS = tuple(range(1, 13))


class ResidentIntegrationError(ResidentUniverseError):
    """A required sibling resident module or facade API was unavailable."""


ResidentBudgetError = ResidentUniverseError
ResidentUniverseFullError = ResidentUniverseFull


class QueryResult(list):
    """List of resonance hits that also exposes the parent result metadata.

    Conventional callers iterate the ranked hits (``for hit in result``) or
    index them (``result[0]``); integration callers can still read
    ``result["hits"]``, ``result.get("generation")`` or ``result["digest"]``.
    """

    def __init__(self, hits, metadata=None):
        super().__init__(hits)
        self._metadata = dict(metadata or {})

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._metadata[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self._metadata.get(key, default)

    def keys(self):
        return self._metadata.keys()

    def values(self):
        return self._metadata.values()

    def items(self):
        return self._metadata.items()

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self._metadata
        return super().__contains__(item)


def _facade_leaf_id(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = (value.get("leaf_id") or value.get("prepared_id")
                 or value.get("id"))
    elif hasattr(value, "leaf_id"):
        value = value.leaf_id
    return value


def _facade_leaf_record(self: "ResidentUniverse", value: Any):
    reference = _facade_leaf_id(value)
    if reference is None:
        return None
    try:
        record = self.leaves.find(_leaf_id_bytes(str(reference)))
    except Exception:
        record = None
    if record is None:
        record = self._find_leaf(str(reference))
    return record


def _facade_relation_ref(self: "ResidentUniverse", value: Any) -> Any:
    if isinstance(value, Mapping):
        value = (value.get("relation_id") or value.get("record_id")
                 or value.get("id"))
    elif hasattr(value, "relation_id"):
        value = value.relation_id
    if value is None:
        raise ValueError("a relation id is required")
    key = str(value)
    payload = self._relation_payloads.get(key)
    if payload is not None:
        return payload.get("relation_store_id") or key
    return key


def _facade_occupied_pairs(self: "ResidentUniverse") -> int:
    count = 0
    for index in range(PAIR_COUNT):
        try:
            forward = self.cells._read_rail_index(index, 0)
            reverse = self.cells._read_rail_index(index, 1)
        except Exception:
            break
        if (forward.weight or forward.amplitude or reverse.weight
                or reverse.amplitude or forward.relation_chain
                or reverse.relation_chain):
            count += 1
    return count


def _facade_prepare(self: "ResidentUniverse",
                    record: Any) -> list[dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError("ingest record must be an object")
    if (record.get("type") in ("leaf", "relation")
            and record.get("schema") == UNIVERSE_SCHEMA):
        return [dict(record)]
    from .ingest import INGEST_SCHEMA
    if (record.get("type") in ("leaf", "relation")
            and record.get("schema") == INGEST_SCHEMA):
        return [dict(record)]
    source = record.get("source_digest")
    if source is not None and _valid_digest(source) is None:
        source = None
    return prepare_records([record], source_digest=source)


def _facade_observe_pair(self: "ResidentUniverse",
                         prepared: Mapping[str, Any],
                         receipt: Mapping[str, Any]) -> bool:
    """Update one deterministic pair/mirror cell for a committed relation."""
    relation_id = receipt.get("relation_id")
    if not relation_id:
        return False
    primary = _head(prepared.get("head"))
    seed = (prepared.get("object_leaf_id") or prepared.get("object_id")
            or prepared.get("object") or prepared.get("predicate")
            or relation_id)
    secondary = int(hashlib.sha256(str(seed).encode("utf-8"))
                    .hexdigest()[:8], 16) % HEAD_COUNT
    if secondary == primary:
        secondary = (primary + 1) % HEAD_COUNT
    try:
        confidence = _confidence(prepared.get("confidence"))
        polarity = str(prepared.get("polarity") or "neutral").lower()
        weight = -confidence if polarity == "refutes" else confidence
        phase = (int(hashlib.sha256(str(relation_id).encode()).hexdigest()[:8],
                     16) / float(1 << 32)) % 1.0
        relation_number = int(hashlib.sha256(
            str(relation_id).encode()).hexdigest()[:8], 16) or 1
        self.cells.observe_pair(primary, secondary, weight=weight,
                                phase=phase, relation_id=relation_number)
        self.cells.update_head(primary, score_delta=weight)
        self.cells.update_head(secondary, score_delta=weight)
        self._commit("pair", str(relation_id))
        return True
    except Exception:
        return False


def _facade_ingest(self: "ResidentUniverse", record: Any) -> dict[str, Any]:
    """Ingest one raw row or prepared record and return a generation receipt."""
    self._ensure_open()
    if isinstance(record, (list, tuple)):
        prepared = []
        for item in record:
            prepared.extend(_facade_prepare(self, item))
    else:
        prepared = _facade_prepare(self, record)
    before = self.status()["counts"]
    result = self.ingest_records(prepared)
    receipts = list(result.get("receipts", []))
    # Parent relation ingestion observes every distinct head pair it can see;
    # with a single routed head that set is empty.  Add the deterministic
    # object-derived partner cell so every *new* relation always updates a
    # pair cell and both mirror rails coherently.
    prepared_relations = [row for row in prepared
                          if row.get("type") == "relation"]
    relation_receipts_all = [item for item in receipts
                             if item.get("relation_id")]
    for match, item in zip(prepared_relations, relation_receipts_all):
        if item.get("created") and item.get("relation_id"):
            _facade_observe_pair(self, match, item)
    after = self.status()["counts"]
    created = int(result.get("accepted", 0)) > 0
    duplicate = int(result.get("duplicates", 0)) > 0
    relation_receipts = [item for item in receipts if item.get("relation_id")]
    relation_receipt = relation_receipts[-1] if relation_receipts else None
    leaf_ids: list[str] = []
    for item in receipts:
        leaf_id = item.get("leaf_id")
        if isinstance(leaf_id, str) and leaf_id not in leaf_ids:
            leaf_ids.append(leaf_id)
    relation_prepared = None
    for item in prepared:
        if item.get("type") == "relation":
            relation_prepared = item
            for field in ("subject_leaf_id", "predicate_leaf_id",
                          "object_leaf_id"):
                value = item.get(field)
                if isinstance(value, str):
                    resident = _leaf_id_bytes(value).hex()
                    if resident not in leaf_ids:
                        leaf_ids.append(resident)
    relation_leaf_id = None
    if relation_receipt is not None:
        payload = (self._relation_payloads.get(
            str(relation_receipt.get("prepared_id")))
            or self._relation_payloads.get(
                str(relation_receipt.get("relation_id")))
            or {})
        relation_leaf_id = payload.get("relation_leaf_id")
        if isinstance(relation_leaf_id, str):
            relation_leaf_id = _leaf_id_bytes(relation_leaf_id).hex()
            if relation_leaf_id not in leaf_ids:
                leaf_ids.append(relation_leaf_id)
    status = ("appended" if created and not duplicate else
              "duplicate" if duplicate and not created else
              "partial" if created or duplicate else "duplicate")
    receipt = {
        "schema": "nanlm.resident-receipt/1",
        "status": status,
        "created": created,
        "duplicate": duplicate,
        "generation": int(self.generation),
        "digest": str(self.digest),
        "records": len(prepared),
        "leaves_added": max(0, int(after.get("leaves", 0))
                            - int(before.get("leaves", 0))),
        "relations_added": max(0, int(after.get("relations", 0))
                               - int(before.get("relations", 0))),
        "features_added": max(0, int(after.get("features", 0))
                              - int(before.get("features", 0))),
        "pair_updates": len([item for item in relation_receipts
                             if item.get("created")]),
        "leaf_ids": leaf_ids,
        "counts": after,
    }
    if relation_receipt is not None:
        receipt.update({
            "relation_id": relation_receipt.get("relation_id"),
            "ingest_relation_id": (relation_prepared or {}).get("relation_id"),
            "relation_leaf_id": relation_leaf_id,
            "hyperedge_id": relation_receipt.get("hyperedge_id"),
            "source_digest": relation_receipt.get("source_digest"),
            "record_id": (relation_prepared or {}).get("record_id"),
            "polarity": (relation_prepared or {}).get("polarity"),
            "confidence": (relation_prepared or {}).get("confidence"),
        })
    else:
        first = prepared[0] if prepared else {}
        receipt.update({
            "record_id": first.get("record_id")
            or first.get("observation_id"),
            "source_digest": first.get("source_digest"),
        })
    return receipt


def _facade_get_leaf(self: "ResidentUniverse",
                     leaf: Any) -> dict[str, Any] | None:
    self._ensure_open()
    record = _facade_leaf_record(self, leaf)
    return None if record is None else self._leaf_view(record)


def _facade_get_relation(self: "ResidentUniverse",
                         relation: Any) -> dict[str, Any] | None:
    self._ensure_open()
    reference = _facade_relation_ref(self, relation)
    try:
        record = self.relations.get_relation(reference)
    except Exception:
        record = None
    if record is None and isinstance(reference, str):
        try:
            record = self.relations.get_relation(
                _relation_id_bytes(reference))
        except Exception:
            record = None
    return None if record is None else self._relation_view(record)


def _facade_get_hyperedge(self: "ResidentUniverse",
                          hyperedge: Any) -> dict[str, Any] | None:
    self._ensure_open()
    if isinstance(hyperedge, Mapping):
        hyperedge = hyperedge.get("hyperedge_id") or hyperedge.get("id")
    if hasattr(hyperedge, "hyperedge_id"):
        hyperedge = hyperedge.hyperedge_id
    if hyperedge is None:
        raise ValueError("a hyperedge id is required")
    try:
        return self.relations.get_hyperedge(str(hyperedge))
    except Exception:
        return None


def _facade_link(self: "ResidentUniverse", parent: Any,
                 child: Any) -> dict[str, Any]:
    self._ensure_open()
    parent_record = _facade_leaf_record(self, parent)
    child_record = _facade_leaf_record(self, child)
    if parent_record is None:
        raise KeyError("parent leaf not found")
    if child_record is None:
        raise KeyError("child leaf not found")
    if child_record.parent == parent_record.slot:
        return {
            "schema": "nanlm.resident-receipt/1",
            "status": "duplicate", "created": False, "duplicate": True,
            "generation": int(self.generation),
            "parent": parent_record.leaf_id.hex(),
            "child": child_record.leaf_id.hex(),
        }
    record = self.leaves.move_leaf(child_record, parent_record)
    self._commit("tree", "%s:%s" % (parent_record.leaf_id.hex(),
                                    child_record.leaf_id.hex()))
    self.flush()
    return {
        "schema": "nanlm.resident-receipt/1",
        "status": "linked", "created": True, "duplicate": False,
        "generation": int(self.generation),
        "parent": parent_record.leaf_id.hex(),
        "child": record.leaf_id.hex(),
        "depth": int(self.leaves.depth(record)),
    }


def _facade_snapshot(self: "ResidentUniverse") -> dict[str, Any]:
    self._ensure_open()
    status = self.status()
    counts = dict(status.get("counts", {}))
    counts["occupied_pairs"] = _facade_occupied_pairs(self)
    counts["pair_cells"] = PAIR_COUNT
    counts["pair_capacity"] = PAIR_COUNT
    sections: dict[str, dict[str, Any]] = {}
    for item in status.get("sections", []):
        sections[str(int(item["id"]))] = {
            "offset": int(item["offset"]),
            "size": int(item["length"]),
            "used": int(item["length"]),
            "generation": int(self.generation),
            "digest": "",
        }
    heads = sections.get(str(SECTION_HEADS))
    if _SECTION_PAIRS not in [int(key) for key in sections]:
        if heads is not None and heads["size"] >= CELLS_BYTES:
            sections[str(_SECTION_PAIRS)] = {
                "offset": heads["offset"] + (CELLS_BYTES - 2_088_960),
                "size": 2_088_960, "used": 2_088_960,
                "generation": int(self.generation), "digest": "",
            }
        else:
            sections[str(_SECTION_PAIRS)] = {
                "offset": 0, "size": 2_088_960, "used": 2_088_960,
                "generation": int(self.generation), "digest": "",
            }
    relations = sections.get(str(SECTION_RELATIONS))
    if _SECTION_HYPEREDGES not in [int(key) for key in sections]:
        sections[str(_SECTION_HYPEREDGES)] = {
            "offset": int((relations or {}).get("offset", 0)),
            "size": int((relations or {}).get("size", 0)),
            "used": int((relations or {}).get("used", 0)),
            "generation": int(self.generation), "digest": "",
        }
    leaves = sections.get(str(SECTION_LEAVES))
    if _SECTION_INDEX not in [int(key) for key in sections]:
        sections[str(_SECTION_INDEX)] = {
            "offset": int((leaves or {}).get("offset", 0)),
            "size": int((leaves or {}).get("size", 0)),
            "used": int((leaves or {}).get("used", 0)),
            "generation": int(self.generation), "digest": "",
        }
    chart = sections.get(str(SECTION_CHART))
    if _SECTION_TOKENS not in [int(key) for key in sections]:
        sections[str(_SECTION_TOKENS)] = {
            "offset": int((chart or {}).get("offset", 0)),
            "size": int((chart or {}).get("size", 0)),
            "used": int((chart or {}).get("used", 0)),
            "generation": int(self.generation), "digest": "",
        }
    if _SECTION_HEADER not in [int(key) for key in sections]:
        sections[str(_SECTION_HEADER)] = {
            "offset": 0, "size": HEADER_REGION_BYTES,
            "used": HEADER_REGION_BYTES,
            "generation": int(self.generation), "digest": "",
        }
    return {
        "schema": UNIVERSE_SCHEMA,
        "path": str(self.path),
        "size": int(self.file_size),
        "budget": {
            "current_bytes": int(self.file_size),
            "default_bytes": DEFAULT_BYTES,
            "hard_limit_bytes": MAX_BYTES,
            "committed_bytes": int(status.get("committed_bytes", 0)),
        },
        "generation": int(self.generation),
        "digest": str(self.digest),
        "state_digest": str(self.digest),
        "counts": counts,
        "leaves": int(counts.get("leaves", 0)),
        "relations": int(counts.get("relations", 0)),
        "hyperedges": int(counts.get("hyperedges", 0)),
        "features": int(counts.get("features", 0)),
        "pairs": int(counts.get("occupied_pairs", 0)),
        "occupied_pairs": int(counts.get("occupied_pairs", 0)),
        "pair_cells": PAIR_COUNT,
        "sections": sections,
    }


def _facade_query(self: "ResidentUniverse", text: Any, top_k: int = 8, *,
                  mode: str = "cosine", phase: float = 0.0) -> QueryResult:
    if isinstance(text, Mapping):
        if text.get("vector") is not None:
            text = text["vector"]
        else:
            parts = [str(text.get(key)) for key in
                     ("text", "subject", "predicate", "object", "expression")
                     if text.get(key) not in (None, "")]
            text = " ".join(parts) if parts else str(text)
    result = self._facade_original_query(
        text, top_k=top_k, mode=mode, phase=phase)
    hits = list(result.get("hits", []))
    metadata = {key: value for key, value in result.items()
                if key != "hits"}
    metadata["hits"] = hits
    return QueryResult(hits, metadata)


def _facade_checkpoint(self: "ResidentUniverse",
                       directory=None) -> dict[str, Any]:
    root = Path(directory).expanduser() if directory else \
        self.path.with_name(self.path.name + ".checkpoints")
    target = max(1, int(self.generation))
    try:
        existing = resident_persistence.recover(root)
    except Exception:
        existing = None
    if existing is not None and int(existing.generation) >= target:
        snapshot = _facade_snapshot(self)
        return {
            "schema": "nanlm.resident-universe-checkpoint/1",
            "path": str(root),
            "directory": str(root),
            "generation": int(existing.generation),
            "slot": int(existing.slot),
            "digest": snapshot["digest"],
            "manifest_digest": existing.digest,
            "manifest": existing.manifest,
            "snapshot": snapshot,
            "canonical_commit": False,
        }
    self.flush()
    # The resident file is mutable in place.  Once the next append lands,
    # older manifests are payload-invalid; quarantine those slots so the new
    # generation can be published.  A valid generation is never removed by
    # this branch.
    if existing is None and root.is_dir():
        for name in getattr(resident_persistence, "SLOT_NAMES",
                            ("manifest.0.json", "manifest.1.json")):
            slot = root / name
            try:
                if slot.is_file() and not slot.is_symlink():
                    slot.unlink()
            except OSError:
                pass
    snapshot = _facade_snapshot(self)
    committed = int(self._metadata.get("plan", {}).get(
        "used_bytes", self.file_size))
    manifest = {
        "schema": "nanlm.resident-universe-manifest/1",
        "universe_schema": UNIVERSE_SCHEMA,
        "path": os.path.abspath(self.path),
        "generation": target,
        "digest": snapshot["digest"],
        "state_digest": snapshot["digest"],
        "counts": snapshot["counts"],
        "sections": [
            {"section_id": int(key), **value}
            for key, value in sorted(
                snapshot["sections"].items(), key=lambda item: int(item[0]))
        ],
        "files": [{
            "path": os.path.abspath(self.path),
            "size": int(self.file_size),
            "regions": [{
                "offset": 0,
                "length": min(int(self.file_size),
                              max(HEADER_REGION_BYTES, committed)),
            }],
        }],
    }
    checkpoint = resident_persistence.checkpoint(root, target, manifest)
    return {
        "schema": "nanlm.resident-universe-checkpoint/1",
        "path": str(root),
        "directory": str(root),
        "generation": int(checkpoint.generation),
        "slot": int(checkpoint.slot),
        "digest": snapshot["digest"],
        "manifest_digest": checkpoint.digest,
        "manifest": checkpoint.manifest,
        "snapshot": snapshot,
        "canonical_commit": False,
    }


ResidentUniverse._facade_original_query = ResidentUniverse.query
ResidentUniverse.ingest = _facade_ingest
ResidentUniverse.get_leaf = _facade_get_leaf
ResidentUniverse.get_relation = _facade_get_relation
ResidentUniverse.get_hyperedge = _facade_get_hyperedge
ResidentUniverse.link = _facade_link
ResidentUniverse.link_leaves = _facade_link
ResidentUniverse.snapshot = _facade_snapshot
ResidentUniverse.query = _facade_query
ResidentUniverse.checkpoint = _facade_checkpoint


def validate_bundle(bundle, max_bytes: int | None = None) -> Any:
    """CLI-facing candidate validator; never mutates a universe."""
    if max_bytes is None:
        return resident_export.validate_candidate(bundle)
    return resident_export.validate_candidate(bundle, max_bytes=max_bytes)


__all__ = (
    "UNIVERSE_SCHEMA", "HEADER_MAGIC", "HEADER_SLOT_BYTES",
    "DEFAULT_FEATURE_DIMENSIONS", "DEFAULT_FEATURE_DTYPE",
    "FEATURE_DIMENSIONS", "FEATURE_DTYPE", "KNOWN_SECTIONS",
    "ResidentUniverseError", "ResidentUniverseFull", "PayloadArena",
    "StateJournal", "ResidentUniverse", "open_universe", "validate_bundle",
    "ResidentIntegrationError", "ResidentBudgetError",
    "ResidentUniverseFullError", "QueryResult",
)
