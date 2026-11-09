"""Bounded sparse memory-mapped superblock and section layout.

The layout owns one sparse file.  Two checksummed header slots make a torn
header write recoverable: the valid copy with the highest generation wins.  The
header carries the section table, a persisted run list (needed for per-section
digests when allocations are interleaved), a monotonic generation and a
whole-file manifest digest over the used arena prefix.

Physical allocation is a simple append-only bump allocator over the arena that
starts after the two header slots.  Sections are accounting views over runs in
that arena; an allocation is never moved once it has been returned.  This keeps
all offsets stable across flush/reopen while allowing every section to use the
whole remaining budget.
"""
from __future__ import annotations

import hashlib
import hmac
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import (
    ALIGN,
    DEFAULT_BYTES,
    MAGIC,
    MAX_BYTES,
    MIN_BYTES,
    RESIDENT_VERSION,
    SECTION_ARENA,
    SECTION_CHART,
    SECTION_FEATURES,
    SECTION_HEADER,
    SECTION_HEADS,
    SECTION_HYPEREDGES,
    SECTION_INDEX,
    SECTION_JOURNAL,
    SECTION_LEAVES,
    SECTION_PAIRS,
    SECTION_RELATIONS,
    SECTION_TOKENS,
    align,
)

# Public geometry.  The header region is deliberately small and fixed; the
# remainder of the file stays sparse until data is actually written.
HEADER_SLOT_SIZE = 32 * 1024
HEADER_SIZE = HEADER_SLOT_SIZE
HEADER_SLOTS = 2
HEADER_REGION = HEADER_SLOT_SIZE * HEADER_SLOTS
DATA_OFFSET = HEADER_REGION
MAX_SECTIONS = 64
MAX_EXTENTS = 1024

_SECTION_TABLE_OFFSET = 256
_EXTENT_TABLE_OFFSET = 5120
_DIGEST_SIZE = 32
_EMPTY_DIGEST = hashlib.sha256(b"").digest()

# magic, version, slot size, generation, total size, data offset,
# section count, extent count, manifest digest, header checksum
_META = struct.Struct("<8sIIQQQII32s32s")
_CHECKSUM_OFFSET = _META.size - _DIGEST_SIZE
# section_id, offset, size, used, generation, digest
_SECTION = struct.Struct("<B7xQQQQ32s")
# section_id, offset, length
_EXTENT = struct.Struct("<B7xQQ")


class LayoutError(ValueError):
    """Base class for resident layout failures."""


class LayoutCorruptError(LayoutError):
    """The file has an invalid or unrecoverable header/manifest."""


class LayoutFullError(LayoutError, MemoryError):
    """The bounded layout or one of its metadata tables is full."""


class Digest(bytes):
    """A 32-byte digest that can also be called like a small accessor."""

    def __call__(self) -> "Digest":
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "Digest(%s)" % self.hex()


@dataclass(frozen=True)
class Section:
    """One section-table row.

    ``size`` is the aligned capacity consumed by this section and ``used`` is
    the number of aligned bytes allocated from it.  They are equal for this
    append-only layout; keeping both mirrors the on-disk table contract.
    """

    section_id: int
    offset: int
    size: int
    used: int
    generation: int
    digest: bytes
    extents: Tuple[Tuple[int, int], ...] = ()

    @property
    def id(self) -> int:
        return self.section_id

    @property
    def capacity(self) -> int:
        return self.size

    @property
    def checksum(self) -> bytes:
        return self.digest

    @property
    def end(self) -> int:
        return self.offset + self.size if self.size else 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "section_id": self.section_id,
            "offset": self.offset,
            "size": self.size,
            "used": self.used,
            "generation": self.generation,
            "digest": self.digest,
            "extents": self.extents,
        }

    def __getitem__(self, key):
        aliases = {
            "id": "section_id",
            "capacity": "size",
            "checksum": "digest",
            "extents": "extents",
        }
        if isinstance(key, str):
            key = aliases.get(key, key)
            if key in self.as_dict():
                return getattr(self, key)
        raise KeyError(key)


class SectionTable(tuple):
    """Tuple of :class:`Section` rows that also supports lookup by section id.

    ``table[0]`` retains normal tuple indexing when there is no section 0;
    ``table[SECTION_CHART]`` returns that section row when it exists.
    """

    def __new__(cls, sections=()):
        return super().__new__(cls, sections)

    def __getitem__(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            for row in self:
                if row.section_id == key:
                    return row
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default

    def keys(self):
        return tuple(row.section_id for row in self)

    def values(self):
        return tuple(self)

    def items(self):
        return tuple((row.section_id, row) for row in self)


@dataclass(frozen=True)
class Header:
    """An immutable snapshot of the current superblock."""

    magic: bytes
    version: int
    generation: int
    total_size: int
    data_offset: int
    section_count: int
    manifest_digest: bytes
    sections: "SectionTable"
    checksum: bytes = b""
    slot: int = 0

    @property
    def size(self) -> int:
        return self.total_size

    @property
    def digest(self) -> bytes:
        return self.manifest_digest

    @property
    def section_table(self) -> Tuple[Section, ...]:
        return self.sections

    def as_dict(self) -> Dict[str, object]:
        return {
            "magic": self.magic,
            "version": self.version,
            "generation": self.generation,
            "total_size": self.total_size,
            "size": self.total_size,
            "data_offset": self.data_offset,
            "section_count": self.section_count,
            "manifest_digest": self.manifest_digest,
            "digest": self.manifest_digest,
            "sections": self.sections,
            "section_table": self.sections,
            "checksum": self.checksum,
            "slot": self.slot,
        }

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.as_dict():
            return self.as_dict()[key]
        raise KeyError(key)


@dataclass(frozen=True)
class _Extent:
    section_id: int
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


class _SectionState:
    __slots__ = ("section_id", "offset", "size", "used", "generation", "hasher")

    def __init__(self, section_id: int) -> None:
        self.section_id = section_id
        self.offset = 0
        self.size = 0
        self.used = 0
        self.generation = 0
        self.hasher = hashlib.sha256()


@dataclass(frozen=True)
class _Candidate:
    slot: int
    generation: int
    total_size: int
    data_offset: int
    sections: Tuple[Section, ...]
    extents: Tuple[_Extent, ...]
    manifest_digest: bytes
    cursor: int
    section_hashers: Dict[int, "hashlib._Hash"]
    manifest_hasher: "hashlib._Hash"


_KNOWN_SECTIONS = (
    SECTION_HEADER,
    SECTION_CHART,
    SECTION_HEADS,
    SECTION_PAIRS,
    SECTION_LEAVES,
    SECTION_RELATIONS,
    SECTION_HYPEREDGES,
    SECTION_FEATURES,
    SECTION_TOKENS,
    SECTION_INDEX,
    SECTION_ARENA,
    SECTION_JOURNAL,
)


def _validate_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size must be an integer")
    if size < MIN_BYTES:
        raise ValueError(
            "resident layout must be at least %d bytes" % MIN_BYTES)
    if size > MAX_BYTES:
        raise ValueError(
            "resident layout must not exceed %d bytes (1 GiB)" % MAX_BYTES)
    if size < DATA_OFFSET + ALIGN:
        raise ValueError("resident layout is too small for its superblock")
    return size


def _validate_section_id(section_id: int) -> int:
    if isinstance(section_id, bool) or not isinstance(section_id, int):
        raise ValueError("section_id must be an integer")
    if not 1 <= section_id <= 0xFF:
        raise ValueError("section_id must be in the range 1..255")
    return section_id


def _validate_length(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("length must be an integer")
    if length < 0:
        raise ValueError("length must be non-negative")
    return length


def _hash_region(hashers: Tuple[object, ...], view: memoryview) -> None:
    """Feed one mapped region to several hashers, then release the view."""
    try:
        for hasher in hashers:
            hasher.update(view)
    finally:
        view.release()


class ResidentLayout:
    """A bounded sparse resident-universe superblock and allocator.

    ``create`` makes a new sparse file; ``open`` reopens one and verifies both
    header copies and the manifest.  Opening a missing path is equivalent to
    ``create`` so callers can use one entry point during boot.
    """

    MAGIC = MAGIC
    VERSION = RESIDENT_VERSION
    HEADER_SIZE = HEADER_SLOT_SIZE
    HEADER_SLOT_SIZE = HEADER_SLOT_SIZE
    HEADER_SLOTS = HEADER_SLOTS
    DATA_OFFSET = DATA_OFFSET
    MAX_BYTES = MAX_BYTES
    MIN_BYTES = MIN_BYTES

    def __init__(self, path, fd: int, mm, size: int) -> None:
        self.path = os.fspath(path)
        self._fd = fd
        self._mmap = mm
        self._file_size = size
        self._closed = False
        self._generation = 0
        self._cursor = DATA_OFFSET
        self._sections: Dict[int, _SectionState] = {}
        self._section_order = []
        self._extents = []
        self._manifest_hasher = hashlib.sha256()
        self._digests_dirty = False
        self._source_slot = 0

    # ------------------------------------------------------------------
    # Construction / reopen
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, path, size: int = DEFAULT_BYTES) -> "ResidentLayout":
        """Create (or truncate) a sparse file and initialise its superblock."""
        size = _validate_size(size)
        path = os.fspath(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        except BaseException:
            os.close(fd)
            raise

        layout = cls(path, fd, mm, size)
        for section_id in _KNOWN_SECTIONS:
            layout._add_section_state(section_id)
        layout._write_headers()
        layout._mmap.flush()
        return layout

    @classmethod
    def open(cls, path, size: int = DEFAULT_BYTES) -> "ResidentLayout":
        """Open an existing layout, recovering a valid duplicate header.

        ``size`` is accepted for API symmetry with :meth:`create`.  For an
        existing file the on-disk size is authoritative; it must still satisfy
        the configured minimum and the 1 GiB hard ceiling.
        """
        path = os.fspath(path)
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            return cls.create(path, size=size)

        # Validate an explicitly supplied budget too, but never resize an
        # existing universe silently.
        _validate_size(size)
        actual = stat.st_size
        if actual > MAX_BYTES:
            raise ValueError(
                "resident layout must not exceed %d bytes (1 GiB)" % MAX_BYTES)
        if actual < DATA_OFFSET + ALIGN:
            raise LayoutCorruptError("resident layout file is too small")

        fd = os.open(path, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, actual)
        except BaseException:
            os.close(fd)
            raise

        layout = cls(path, fd, mm, actual)
        try:
            layout._load_headers()
            # Refresh both copies so a previously torn slot is healed.
            layout._write_headers()
            layout._mmap.flush()
        except BaseException:
            mm.close()
            os.close(fd)
            layout._closed = True
            raise
        return layout

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------
    @property
    def generation(self) -> int:
        self._ensure_open()
        return self._generation

    @property
    def total_size(self) -> int:
        return self._file_size

    @property
    def size(self) -> int:
        return self._file_size

    @property
    def used_bytes(self) -> int:
        self._ensure_open()
        return self._cursor - DATA_OFFSET

    @property
    def free_bytes(self) -> int:
        self._ensure_open()
        return self._file_size - self._cursor

    @property
    def manifest_digest(self) -> Digest:
        self._ensure_open()
        self._ensure_digests()
        return Digest(self._manifest_hasher.digest())

    @property
    def is_open(self) -> bool:
        return not self._closed

    @property
    def mmap(self):
        """The live shared mapping (callers should prefer read/write helpers)."""
        self._ensure_open()
        return self._mmap

    def header(self) -> Header:
        self._ensure_open()
        self._ensure_digests()
        sections = SectionTable(
            self._section_snapshot(section_id)
            for section_id in self._section_order
        )
        _slot, checksum = self._build_slot()
        return Header(
            magic=MAGIC,
            version=RESIDENT_VERSION,
            generation=self._generation,
            total_size=self._file_size,
            data_offset=DATA_OFFSET,
            section_count=len(sections),
            manifest_digest=Digest(self._manifest_hasher.digest()),
            sections=sections,
            checksum=checksum,
            slot=self._source_slot,
        )

    def section(self, section_id: int) -> Section:
        self._ensure_open()
        section_id = _validate_section_id(section_id)
        self._ensure_digests()
        return self._section_snapshot(section_id)

    def sections(self) -> Tuple[Section, ...]:
        return self.header().sections

    # ------------------------------------------------------------------
    # Allocation and data access
    # ------------------------------------------------------------------
    def allocate(self, section_id: int, length: int) -> int:
        """Align-allocate ``length`` bytes for ``section_id``.

        Returns the stable file offset of the allocation.  The allocation is
        appended to the bounded arena and the section's generation/digest and
        the global generation are advanced.  A full layout raises
        :class:`LayoutFullError` (also a ``MemoryError`` and ``ValueError``).
        """
        self._ensure_open()
        section_id = _validate_section_id(section_id)
        length = _validate_length(length)
        aligned = align(length)
        if aligned == 0:
            return self._cursor

        start = self._cursor
        end = start + aligned
        if end > self._file_size:
            raise LayoutFullError(
                "resident layout is full: need %d bytes, only %d free"
                % (aligned, self._file_size - start))

        state = self._sections.get(section_id)
        is_new_section = state is None
        if is_new_section and len(self._sections) >= MAX_SECTIONS:
            raise LayoutFullError("resident section table is full")

        append_extent = True
        if self._extents and self._extents[-1].section_id == section_id:
            append_extent = False
        if append_extent and len(self._extents) >= MAX_EXTENTS:
            raise LayoutFullError("resident extent table is full")

        if is_new_section:
            state = self._add_section_state(section_id)

        if state.size == 0:
            state.offset = start
        state.size += aligned
        state.used += aligned
        self._generation += 1
        state.generation = self._generation

        if append_extent:
            self._extents.append(_Extent(section_id, start, aligned))
        else:
            last = self._extents[-1]
            self._extents[-1] = _Extent(
                section_id, last.offset, last.length + aligned)

        _hash_region(
            (state.hasher, self._manifest_hasher),
            memoryview(self._mmap)[start:end],
        )
        self._cursor = end
        self._write_headers()
        return start

    def append(self, section_id: int, data) -> int:
        """Allocate and copy ``data``, returning its stable offset."""
        payload = bytes(data)
        offset = self.allocate(section_id, len(payload))
        if payload:
            self._mmap[offset:offset + len(payload)] = payload
            self._digests_dirty = True
            self._write_headers()
        return offset

    def read_at(self, offset: int, length: int) -> bytes:
        """Read previously allocated bytes by absolute file offset."""
        self._ensure_open()
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("offset must be an integer")
        length = _validate_length(length)
        if offset < DATA_OFFSET or offset + length > self._cursor:
            raise ValueError("read is outside the used resident arena")
        return bytes(self._mmap[offset:offset + length])

    def write_at(self, offset: int, data) -> int:
        """Overwrite allocated bytes, returning the number written.

        Digests are refreshed on the next header/flush operation.  Writes are
        restricted to the already-allocated arena so they cannot escape the
        bounded file.
        """
        self._ensure_open()
        payload = bytes(data)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("offset must be an integer")
        if offset < DATA_OFFSET or offset + len(payload) > self._cursor:
            raise ValueError("write is outside the used resident arena")
        self._mmap[offset:offset + len(payload)] = payload
        self._digests_dirty = True
        return len(payload)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Recompute digests and durably write both header copies."""
        self._ensure_open()
        # Recompute from the mapped bytes so direct mmap writes are captured.
        self._ensure_digests(recompute=True)
        self._write_headers()
        self._mmap.flush()
        os.fsync(self._fd)

    def close(self) -> None:
        """Flush and close the mapping.  Safe to call more than once."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._mmap.close()
            finally:
                os.close(self._fd)
                self._closed = True

    def __enter__(self) -> "ResidentLayout":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self):  # pragma: no cover - defensive interpreter cleanup
        try:
            self.close()
        except BaseException:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise LayoutError("resident layout is closed")

    def _add_section_state(self, section_id: int) -> _SectionState:
        state = _SectionState(section_id)
        self._sections[section_id] = state
        self._section_order.append(section_id)
        return state

    def _section_snapshot(self, section_id: int) -> Section:
        state = self._sections.get(section_id)
        if state is None:
            return Section(
                section_id=section_id,
                offset=0,
                size=0,
                used=0,
                generation=0,
                digest=Digest(_EMPTY_DIGEST),
                extents=(),
            )
        extents = tuple(
            (extent.offset, extent.length)
            for extent in self._extents
            if extent.section_id == section_id
        )
        return Section(
            section_id=section_id,
            offset=state.offset,
            size=state.size,
            used=state.used,
            generation=state.generation,
            digest=Digest(state.hasher.digest()),
            extents=extents,
        )

    def _ensure_digests(self, recompute: bool = False) -> None:
        if recompute or self._digests_dirty:
            self._recompute_digests()

    def _recompute_digests(self) -> None:
        hashers = {
            section_id: hashlib.sha256()
            for section_id in self._sections
        }
        view = memoryview(self._mmap)
        for extent in self._extents:
            hasher = hashers.get(extent.section_id)
            if hasher is not None:
                hasher.update(view[extent.offset:extent.end])
        manifest = hashlib.sha256()
        manifest.update(view[DATA_OFFSET:self._cursor])
        for section_id, hasher in hashers.items():
            self._sections[section_id].hasher = hasher
        self._manifest_hasher = manifest
        self._digests_dirty = False

    def _build_slot(self) -> Tuple[bytearray, Digest]:
        self._ensure_digests()
        slot = bytearray(HEADER_SLOT_SIZE)
        _META.pack_into(
            slot, 0,
            MAGIC,
            RESIDENT_VERSION,
            HEADER_SLOT_SIZE,
            self._generation,
            self._file_size,
            DATA_OFFSET,
            len(self._section_order),
            len(self._extents),
            self._manifest_hasher.digest(),
            b"\x00" * _DIGEST_SIZE,
        )
        for index, section_id in enumerate(self._section_order):
            state = self._sections[section_id]
            _SECTION.pack_into(
                slot,
                _SECTION_TABLE_OFFSET + index * _SECTION.size,
                section_id,
                state.offset,
                state.size,
                state.used,
                state.generation,
                state.hasher.digest(),
            )
        for index, extent in enumerate(self._extents):
            _EXTENT.pack_into(
                slot,
                _EXTENT_TABLE_OFFSET + index * _EXTENT.size,
                extent.section_id,
                extent.offset,
                extent.length,
            )
        view = memoryview(slot)
        checksum = hashlib.sha256()
        checksum.update(view[:_CHECKSUM_OFFSET])
        checksum.update(view[_CHECKSUM_OFFSET + _DIGEST_SIZE:])
        digest = checksum.digest()
        slot[_CHECKSUM_OFFSET:_CHECKSUM_OFFSET + _DIGEST_SIZE] = digest
        return slot, Digest(digest)

    def _write_headers(self) -> None:
        slot, _checksum = self._build_slot()

        # Write the first copy, then the second.  A crash in between always
        # leaves at least one complete checksummed copy.
        self._mmap[0:HEADER_SLOT_SIZE] = slot
        self._mmap[HEADER_SLOT_SIZE:HEADER_REGION] = slot

    def _parse_slot(self, index: int) -> Tuple[Optional[_Candidate], bool]:
        """Parse one header slot.

        Returns ``(candidate, magic_ok)``.  A candidate only means the slot was
        structurally readable with a valid checksum; the manifest is verified
        separately so an older intact copy can be used as a fallback.
        """
        start = index * HEADER_SLOT_SIZE
        raw = bytes(self._mmap[start:start + HEADER_SLOT_SIZE])
        if len(raw) < HEADER_SLOT_SIZE:
            return None, False
        magic, version, slot_size, generation, total_size, data_offset, \
            section_count, extent_count, manifest_digest, checksum = \
            _META.unpack_from(raw, 0)
        if magic != MAGIC:
            return None, False
        if version != RESIDENT_VERSION or slot_size != HEADER_SLOT_SIZE:
            return None, True
        if total_size != self._file_size or data_offset != DATA_OFFSET:
            return None, True
        if section_count > MAX_SECTIONS or extent_count > MAX_EXTENTS:
            return None, True

        checksum_hash = hashlib.sha256()
        checksum_hash.update(raw[:_CHECKSUM_OFFSET])
        checksum_hash.update(raw[_CHECKSUM_OFFSET + _DIGEST_SIZE:])
        if not hmac.compare_digest(checksum_hash.digest(), checksum):
            return None, True

        sections = []
        seen = set()
        try:
            for i in range(section_count):
                offset = _SECTION_TABLE_OFFSET + i * _SECTION.size
                (section_id, sec_offset, sec_size, sec_used, sec_generation,
                 sec_digest) = _SECTION.unpack_from(raw, offset)
                if section_id == 0 or section_id in seen:
                    return None, True
                seen.add(section_id)
                if sec_size != sec_used:
                    return None, True
                if sec_size and (sec_offset < DATA_OFFSET or
                                 sec_offset + sec_size > self._file_size):
                    return None, True
                if not sec_size and sec_offset != 0:
                    return None, True
                if sec_generation > generation:
                    return None, True
                sections.append(Section(
                    section_id=section_id,
                    offset=sec_offset,
                    size=sec_size,
                    used=sec_used,
                    generation=sec_generation,
                    digest=Digest(sec_digest),
                ))

            extents = []
            expected = DATA_OFFSET
            for i in range(extent_count):
                offset = _EXTENT_TABLE_OFFSET + i * _EXTENT.size
                section_id, extent_offset, length = _EXTENT.unpack_from(
                    raw, offset)
                if section_id not in seen or length == 0:
                    return None, True
                if extent_offset != expected:
                    return None, True
                if extent_offset < DATA_OFFSET or length > (
                        self._file_size - extent_offset):
                    return None, True
                expected += length
                extents.append(_Extent(section_id, extent_offset, length))
        except struct.error:
            return None, True

        # Verify section aggregates and digests against extents/data.
        section_hashers = {
            section.section_id: hashlib.sha256() for section in sections
        }
        view = memoryview(self._mmap)
        for extent in extents:
            section_hashers[extent.section_id].update(
                view[extent.offset:extent.end])
        for section in sections:
            hasher = section_hashers[section.section_id]
            if hasher.digest() != section.digest:
                return None, True
            matching = [e for e in extents if e.section_id == section.section_id]
            if matching:
                if section.offset != matching[0].offset:
                    return None, True
                total = sum(e.length for e in matching)
                if total != section.size or total != section.used:
                    return None, True
            elif section.size or section.used or section.offset:
                return None, True

        manifest_hasher = hashlib.sha256()
        manifest_hasher.update(view[DATA_OFFSET:expected])
        if not hmac.compare_digest(manifest_hasher.digest(), manifest_digest):
            return None, True

        return _Candidate(
            slot=index,
            generation=generation,
            total_size=total_size,
            data_offset=data_offset,
            sections=tuple(sections),
            extents=tuple(extents),
            manifest_digest=Digest(manifest_digest),
            cursor=expected,
            section_hashers=section_hashers,
            manifest_hasher=manifest_hasher,
        ), True

    def _load_headers(self) -> None:
        candidates = []
        magic_ok = False
        for index in range(HEADER_SLOTS):
            candidate, slot_magic_ok = self._parse_slot(index)
            magic_ok = magic_ok or slot_magic_ok
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            if not magic_ok:
                raise LayoutCorruptError("invalid resident magic")
            raise LayoutCorruptError("resident header checksum mismatch")

        candidates.sort(key=lambda c: (-c.generation, c.slot))
        chosen = candidates[0]

        self._generation = chosen.generation
        self._cursor = chosen.cursor
        self._source_slot = chosen.slot
        self._section_order = [s.section_id for s in chosen.sections]
        self._sections = {}
        for section in chosen.sections:
            state = _SectionState(section.section_id)
            state.offset = section.offset
            state.size = section.size
            state.used = section.used
            state.generation = section.generation
            state.hasher = chosen.section_hashers[section.section_id]
            self._sections[section.section_id] = state
        # Keep the standard sections available even if an older file omitted
        # one of them.
        for section_id in _KNOWN_SECTIONS:
            if section_id not in self._sections:
                self._add_section_state(section_id)
        self._extents = list(chosen.extents)
        self._manifest_hasher = chosen.manifest_hasher
        self._digests_dirty = False


__all__ = (
    "HEADER_SLOT_SIZE",
    "HEADER_SIZE",
    "HEADER_SLOTS",
    "HEADER_REGION",
    "DATA_OFFSET",
    "MAX_SECTIONS",
    "MAX_EXTENTS",
    "Digest",
    "Header",
    "Section",
    "SectionTable",
    "LayoutError",
    "LayoutCorruptError",
    "LayoutFullError",
    "ResidentLayout",
)
