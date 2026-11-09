"""Typed head and pair-cell region for the resident universe.

The region is deliberately independent of any layout module: it is a fixed,
self-describing byte layout that can sit inside a ``bytearray`` or an
``mmap.mmap`` at any non-negative ``base_offset``.  Every record is packed with
an explicit little-endian ``struct.Struct`` and every access validates its
head, pair and rail indices first.

Layout (all sizes in bytes, relative to ``base_offset``)::

    heads   256 records x 32 =   8,192
    pairs    32,640 cells x 64 = 2,088,960   (two 32-byte rails per cell)
    total                       2,097,152   (2 MiB)

Head record ``<BBHiIIQII>``::

    offset  field            type
    0       head_id          u8    permanent identity, never rewritten
    1       flags            u8
    2       identity         u16   permanent pair address (head, head)
    4       score            i32   Q16.16 additive score
    8       state            u32   additive state word
    12      generation       u32   monotonic per-head generation
    16      leaf_chain       u64   offset of this head's leaf chain
    24      relation_chain   u32   relation-chain pointer
    28      reserved         u32

Pair rail record ``<BBHiiiHII2xI>``::

    offset  field            type
    0       source           u8    source head
    1       target           u8    target head
    2       flags            u16   RAIL_FLAG_INITIALIZED, ...
    4       phase            i32   Q16.16 turns, wrap at one full turn
    8       amplitude        i32   Q16.16 accumulated absolute evidence
    12      weight           i32   Q16.16 signed additive weight
    16      mirror           u16   canonical address of the reverse pair
    18      relation_chain   u32   relation-chain pointer
    22      generation       u32   per-cell generation, shared by both rails
    26      (padding)        2
    28      reserved         u32

A pair cell is two adjacent rail records.  ``s+`` (``RAIL_S_PLUS``) is the
forward direction and ``s-`` (``RAIL_S_MINUS``) the reverse direction.  The
canonical pair is ``min(address_of(a, b), reverse16(address_of(a, b)))``; the
mirror is ``reverse16(canonical)`` and is stored in *both* rails.  Observing
``(a, b)`` in canonical orientation updates ``s+``; observing ``(b, a)``
updates ``s-``.  The two rails carry independent weights and phases while the
cell metadata (heads, mirror, generation) stays coherent.

This module makes no quantum claim.  Phase/amplitude/weight are additive fixed
point accumulators intended to make superposition-like readout possible; they
do not create information that was not observed.
"""
from __future__ import annotations

import array
import math
import struct
from typing import NamedTuple

HEAD_COUNT = 256
PAIR_COUNT = 256 * 255 // 2  # C(256, 2) = 32,640
RAILS = 2

FIXED_ONE = 1 << 16
PHASE_ONE = 1 << 16  # one full turn, in Q16.16 phase units

RAIL_S_PLUS = 0
RAIL_S_MINUS = 1
RAIL_NAMES = ("s+", "s-")
RAIL_FLAG_INITIALIZED = 0x0001

# Relative to the caller supplied base offset.
HEADS_REGION_OFFSET = 0
PAIRS_REGION_OFFSET = HEAD_COUNT * 32

HEAD_STRUCT = struct.Struct("<BBHiIIQII")
RAIL_STRUCT = struct.Struct("<BBHiiiHII2xI")

HEAD_RECORD_BYTES = HEAD_STRUCT.size          # 32
RAIL_RECORD_BYTES = RAIL_STRUCT.size          # 32
PAIR_CELL_BYTES = RAIL_RECORD_BYTES * RAILS   # 64
HEAD_REGION_BYTES = HEAD_COUNT * HEAD_RECORD_BYTES
PAIR_REGION_BYTES = PAIR_COUNT * PAIR_CELL_BYTES
REGION_BYTES = HEAD_REGION_BYTES + PAIR_REGION_BYTES

# More explicit aliases for callers that integrate with a future layout.py.
HEAD_RECORD_SIZE = HEAD_RECORD_BYTES
RAIL_RECORD_SIZE = RAIL_RECORD_BYTES
PAIR_CELL_SIZE = PAIR_CELL_BYTES
HEADS_BYTES = HEAD_REGION_BYTES
PAIRS_BYTES = PAIR_REGION_BYTES

_MIN_I32 = -(1 << 31)
_MAX_I32 = (1 << 31) - 1
_MAX_U16 = (1 << 16) - 1
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# address helpers
# ---------------------------------------------------------------------------
_REVERSE8 = bytes(int(format(value, "08b")[::-1], 2) for value in range(256))


def reverse8(value: int) -> int:
    """Reverse the low eight bits of ``value``."""
    _require_int(value, "value")
    return _REVERSE8[value & 0xFF]


def reverse16(value: int) -> int:
    """Reverse all sixteen bits of ``value``."""
    _require_int(value, "value")
    value &= 0xFFFF
    return (_REVERSE8[value & 0xFF] << 8) | _REVERSE8[(value >> 8) & 0xFF]


def _check_head(head: int, name: str = "head") -> int:
    value = _require_int(head, name)
    if not 0 <= value < HEAD_COUNT:
        raise ValueError("%s must be in [0, %d): %r" % (name, HEAD_COUNT, head))
    return value


def _require_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer: %r" % (name, value))
    return value


def _check_address(address: int, name: str = "address") -> int:
    value = _require_int(address, name)
    if not 0 <= value <= _MAX_U16:
        raise ValueError("%s must be a 16-bit address: %r" % (name, address))
    return value


def _check_rail(rail: int) -> int:
    value = _require_int(rail, "rail")
    if value not in (RAIL_S_PLUS, RAIL_S_MINUS):
        raise ValueError("rail must be 0 (s+) or 1 (s-): %r" % (rail,))
    return value


def address_of(a: int, b: int) -> int:
    """Native directed address: ``(a << 8) | reverse8(b)``."""
    first = _check_head(a, "a")
    second = _check_head(b, "b")
    return (first << 8) | _REVERSE8[second]


def canonical_address(a: int, b: int) -> int:
    """Canonical address for an unordered pair (``a == b`` is allowed here)."""
    raw = address_of(a, b)
    return min(raw, reverse16(raw))


def mirror_address(address: int) -> int:
    """Reverse16 mirror of a 16-bit address."""
    return reverse16(_check_address(address))


def identity_address(head: int) -> int:
    """Permanent identity address of one head: ``(head, head)``."""
    value = _check_head(head)
    return (value << 8) | _REVERSE8[value]


def _heads_from_canonical(address: int) -> tuple[int, int]:
    address = _check_address(address)
    first = address >> 8
    second = _REVERSE8[address & 0xFF]
    if first == second:
        raise ValueError("diagonal address %04x has no pair cell" % address)
    return first, second


def _build_pair_index() -> tuple[array.array, array.array]:
    index = array.array("H", [0xFFFF]) * (1 << 16)
    canonicals = array.array("H")
    position = 0
    for first in range(HEAD_COUNT):
        for second in range(first + 1, HEAD_COUNT):
            canonical = canonical_address(first, second)
            if index[canonical] != 0xFFFF:
                raise AssertionError("duplicate canonical pair address")
            index[canonical] = position
            canonicals.append(canonical)
            position += 1
    if position != PAIR_COUNT:
        raise AssertionError("pair index count %d != %d" % (position, PAIR_COUNT))
    return index, canonicals


_PAIR_INDEX, _PAIR_CANONICAL = _build_pair_index()


def pair_index(a: int, b: int) -> int:
    """Return the dense index of the canonical cell for distinct heads."""
    first = _check_head(a, "a")
    second = _check_head(b, "b")
    if first == second:
        raise ValueError("a pair cell needs two distinct heads")
    canonical = canonical_address(first, second)
    index = _PAIR_INDEX[canonical]
    if index == 0xFFFF:
        raise ValueError("pair address %04x is not mapped" % canonical)
    return index


# ---------------------------------------------------------------------------
# fixed point helpers
# ---------------------------------------------------------------------------
def to_fixed(value: float | int, name: str = "value") -> int:
    """Convert a real number to Q16.16, rejecting non-finite/out-of-range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a real number: %r" % (name, value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s must be finite: %r" % (name, value))
    scaled = int(round(value * FIXED_ONE))
    if not _MIN_I32 <= scaled <= _MAX_I32:
        raise ValueError("%s is outside Q16.16 range: %r" % (name, value))
    return scaled


def from_fixed(value: int) -> float:
    """Convert a Q16.16 integer back to a float."""
    return _require_int(value, "value") / FIXED_ONE


def to_phase(value: float | int, name: str = "phase") -> int:
    """Convert turns (1.0 == one full turn) to wrapped Q16.16 phase."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a real number: %r" % (name, value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("%s must be finite: %r" % (name, value))
    return int(round(value * PHASE_ONE)) % PHASE_ONE


def from_phase(value: int) -> float:
    """Convert wrapped Q16.16 phase to turns."""
    return _require_int(value, "value") / PHASE_ONE


def _saturate_i32(value: int) -> int:
    return max(_MIN_I32, min(_MAX_I32, value))


def _saturate_u32(value: int) -> int:
    return max(0, min(_MAX_U32, value))


def _scale_fixed(value: int, units: int) -> int:
    """Multiply a Q16.16 value by a Q16.16 ratio, rounding half away from zero."""
    product = value * units
    if product >= 0:
        return (product + FIXED_ONE // 2) // FIXED_ONE
    return -((-product + FIXED_ONE // 2) // FIXED_ONE)


def _ratio_units(factor: float | int) -> int:
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError("factor must be a real number: %r" % (factor,))
    if isinstance(factor, float) and not math.isfinite(factor):
        raise ValueError("factor must be finite: %r" % (factor,))
    if not 0.0 <= factor <= 1.0:
        raise ValueError("decay factor must be in [0, 1]: %r" % (factor,))
    return int(round(factor * FIXED_ONE))


# ---------------------------------------------------------------------------
# record types
# ---------------------------------------------------------------------------
class HeadRecord(NamedTuple):
    head_id: int
    flags: int
    identity: int
    score: int
    state: int
    generation: int
    leaf_chain: int
    relation_chain: int
    reserved: int

    @property
    def score_value(self) -> float:
        return self.score / FIXED_ONE

    @property
    def score_float(self) -> float:
        return self.score_value

    @property
    def score_word(self) -> int:
        return self.score

    @property
    def state_word(self) -> int:
        return self.state

    @property
    def leaf_chain_head_offset(self) -> int:
        return self.leaf_chain


class RailRecord(NamedTuple):
    source: int
    target: int
    flags: int
    phase: int
    amplitude: int
    weight: int
    mirror: int
    relation_chain: int
    generation: int
    reserved: int

    @property
    def initialized(self) -> bool:
        return bool(self.flags & RAIL_FLAG_INITIALIZED)

    @property
    def source_head(self) -> int:
        return self.source

    @property
    def target_head(self) -> int:
        return self.target

    @property
    def mirror_pair_id(self) -> int:
        return self.mirror

    @property
    def relation_chain_pointer(self) -> int:
        return self.relation_chain

    @property
    def phase_turns(self) -> float:
        return self.phase / PHASE_ONE

    @property
    def phase_float(self) -> float:
        return self.phase_turns

    @property
    def amplitude_value(self) -> float:
        return self.amplitude / FIXED_ONE

    @property
    def amplitude_float(self) -> float:
        return self.amplitude_value

    @property
    def weight_value(self) -> float:
        return self.weight / FIXED_ONE

    @property
    def weight_float(self) -> float:
        return self.weight_value

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "flags": self.flags,
            "phase": self.phase,
            "phase_turns": self.phase_turns,
            "amplitude": self.amplitude,
            "amplitude_value": self.amplitude_value,
            "weight": self.weight,
            "weight_value": self.weight_value,
            "mirror": self.mirror,
            "relation_chain": self.relation_chain,
            "generation": self.generation,
            "initialized": self.initialized,
        }


class PairState(NamedTuple):
    index: int
    canonical: int
    mirror: int
    forward: RailRecord
    reverse: RailRecord

    @property
    def s_plus(self) -> RailRecord:
        return self.forward

    @property
    def s_minus(self) -> RailRecord:
        return self.reverse

    @property
    def canonical_address(self) -> int:
        return self.canonical

    @property
    def mirror_address(self) -> int:
        return self.mirror

    @property
    def total_weight(self) -> int:
        return self.forward.weight + self.reverse.weight

    @property
    def total_amplitude(self) -> int:
        return self.forward.amplitude + self.reverse.amplitude

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "canonical": self.canonical,
            "mirror": self.mirror,
            "s_plus": self.forward.as_dict(),
            "s_minus": self.reverse.as_dict(),
        }


# ---------------------------------------------------------------------------
# region
# ---------------------------------------------------------------------------
class ResidentCells:
    """Typed head/pair region over a caller-owned mutable buffer.

    The buffer is never resized; ``base_offset`` may point anywhere inside a
    larger mmap or bytearray.  A region must be formatted once with
    ``initialize=True`` (or :meth:`format_region`) before heads can be updated.
    Formatting writes the permanent head identities; subsequent updates never
    touch ``head_id`` or ``identity``.
    """

    head_count = HEAD_COUNT
    pair_count = PAIR_COUNT
    rails = RAILS
    region_bytes = REGION_BYTES

    def __init__(self, buffer, base_offset: int = 0, *, initialize: bool = False):
        if isinstance(base_offset, bool) or not isinstance(base_offset, int):
            raise ValueError("base_offset must be an integer")
        if base_offset < 0:
            raise ValueError("base_offset must be non-negative")
        try:
            view = memoryview(buffer)
        except TypeError as error:
            raise TypeError("buffer must be a writable bytes-like object") from error
        if view.readonly:
            raise TypeError("buffer must be writable")
        try:
            view = view.cast("B")
        except TypeError as error:
            raise TypeError("buffer must be a contiguous byte buffer") from error
        length = view.nbytes
        needed = base_offset + REGION_BYTES
        if length < needed:
            raise ValueError(
                "buffer too small: need %d bytes at offset %d, have %d"
                % (REGION_BYTES, base_offset, length))
        self.buffer = buffer
        self.base_offset = base_offset
        self.heads_offset = base_offset + HEADS_REGION_OFFSET
        self.pairs_offset = base_offset + PAIRS_REGION_OFFSET
        self.buffer_bytes = length
        self.generation = 0
        if initialize:
            self.initialize()

    # -- formatting -----------------------------------------------------
    @classmethod
    def format_region(cls, buffer, base_offset: int = 0) -> "ResidentCells":
        """Format ``buffer`` and return a ready region."""
        return cls(buffer, base_offset, initialize=True)

    def initialize(self) -> None:
        """Zero this region and write permanent head/pair identities."""
        start = self.base_offset
        end = start + REGION_BYTES
        self.buffer[start:end] = b"\x00" * REGION_BYTES
        for head in range(HEAD_COUNT):
            HEAD_STRUCT.pack_into(
                self.buffer, self.heads_offset + head * HEAD_RECORD_BYTES,
                head,                # head_id (permanent)
                0,                   # flags
                identity_address(head),  # identity (permanent)
                0,                   # score
                0,                   # state
                0,                   # generation
                0,                   # leaf_chain
                0,                   # relation_chain
                0,                   # reserved
            )
        for first in range(HEAD_COUNT):
            for second in range(first + 1, HEAD_COUNT):
                canonical = canonical_address(first, second)
                index = _PAIR_INDEX[canonical]
                self._write_pair_metadata(index, canonical)
        self.generation = 0

    def _write_pair_metadata(self, index: int, canonical: int) -> None:
        first, second = _heads_from_canonical(canonical)
        mirror = reverse16(canonical)
        offset = self.pairs_offset + index * PAIR_CELL_BYTES
        RAIL_STRUCT.pack_into(
            self.buffer, offset,
            first, second, RAIL_FLAG_INITIALIZED,
            0, 0, 0, mirror, 0, 0, 0)
        RAIL_STRUCT.pack_into(
            self.buffer, offset + RAIL_RECORD_BYTES,
            second, first, RAIL_FLAG_INITIALIZED,
            0, 0, 0, mirror, 0, 0, 0)

    # -- offsets --------------------------------------------------------
    def head_record_offset(self, head: int) -> int:
        return self.heads_offset + _check_head(head) * HEAD_RECORD_BYTES

    def pair_index(self, a: int, b: int) -> int:
        return pair_index(a, b)

    def pair_cell_offset(self, a: int, b: int) -> int:
        index = pair_index(a, b)
        return self.pairs_offset + index * PAIR_CELL_BYTES

    def rail_record_offset(self, a: int, b: int, rail: int) -> int:
        index = pair_index(a, b)
        return self.pairs_offset + index * PAIR_CELL_BYTES + _check_rail(rail) * RAIL_RECORD_BYTES

    # -- heads ----------------------------------------------------------
    def read_head(self, head: int) -> HeadRecord:
        offset = self.head_record_offset(head)
        return HeadRecord(*HEAD_STRUCT.unpack_from(self.buffer, offset))

    def update_head(self, head: int, score_delta: float | int = 0.0,
                    state_delta: int = 0, *, flags: int | None = None,
                    leaf_chain: int | None = None,
                    relation_chain: int | None = None,
                    state: int | None = None) -> HeadRecord:
        """Additively update one head while preserving its permanent identity.

        ``score_delta`` is a real Q16.16 score increment; ``state_delta`` is a
        bounded unsigned increment.  ``flags``, ``leaf_chain`` and
        ``relation_chain`` replace those fields when supplied.  ``head_id`` and
        ``identity`` are read from the region, checked, and written back
        unchanged.
        """
        head = _check_head(head)
        current = self.read_head(head)
        if current.head_id != head or current.identity != identity_address(head):
            raise ValueError(
                "head %d is not initialized with its permanent identity" % head)
        new_flags = current.flags
        if flags is not None:
            new_flags = _bounded_int(flags, 0, 0xFF, "flags")
        new_score = _saturate_i32(current.score + to_fixed(score_delta, "score_delta"))
        if state is None:
            new_state = _saturate_u32(
                current.state + _bounded_int(state_delta, 0, _MAX_U32, "state_delta"))
        else:
            new_state = _bounded_int(state, 0, _MAX_U32, "state")
        new_leaf = current.leaf_chain
        if leaf_chain is not None:
            new_leaf = _bounded_int(leaf_chain, 0, _MAX_U64, "leaf_chain")
        new_relation = current.relation_chain
        if relation_chain is not None:
            new_relation = _bounded_int(relation_chain, 0, _MAX_U32, "relation_chain")
        new_generation = (current.generation + 1) & _MAX_U32
        record = HeadRecord(
            head, new_flags, current.identity, new_score, new_state,
            new_generation, new_leaf, new_relation, current.reserved)
        HEAD_STRUCT.pack_into(
            self.buffer, self.head_record_offset(head),
            record.head_id, record.flags, record.identity, record.score,
            record.state, record.generation, record.leaf_chain,
            record.relation_chain, record.reserved)
        self.generation = max(self.generation, new_generation)
        return record

    # -- pair rails -----------------------------------------------------
    def _resolve_pair(self, a: int, b: int) -> tuple[int, int, int]:
        first = _check_head(a, "a")
        second = _check_head(b, "b")
        if first == second:
            raise ValueError("a pair cell needs two distinct heads")
        canonical = canonical_address(first, second)
        index = pair_index(first, second)
        return index, canonical, reverse16(canonical)

    def _read_rail_index(self, index: int, rail: int) -> RailRecord:
        offset = self.pairs_offset + index * PAIR_CELL_BYTES \
            + _check_rail(rail) * RAIL_RECORD_BYTES
        return RailRecord(*RAIL_STRUCT.unpack_from(self.buffer, offset))

    def _write_rail_index(self, index: int, rail: int, record: RailRecord) -> None:
        offset = self.pairs_offset + index * PAIR_CELL_BYTES \
            + _check_rail(rail) * RAIL_RECORD_BYTES
        RAIL_STRUCT.pack_into(
            self.buffer, offset,
            record.source, record.target, record.flags, record.phase,
            record.amplitude, record.weight, record.mirror,
            record.relation_chain, record.generation, record.reserved)

    def read_rail(self, a: int, b: int, rail: int = RAIL_S_PLUS) -> RailRecord:
        index, _canonical, _mirror = self._resolve_pair(a, b)
        return self._read_rail_index(index, rail)

    def pair_state(self, a: int, b: int) -> PairState:
        """Read both rails of the canonical cell for heads ``a`` and ``b``."""
        index, canonical, mirror = self._resolve_pair(a, b)
        forward = self._read_rail_index(index, RAIL_S_PLUS)
        reverse = self._read_rail_index(index, RAIL_S_MINUS)
        if not forward.initialized:
            forward = _blank_rail(*_heads_from_canonical(canonical), mirror, False)
        if not reverse.initialized:
            first, second = _heads_from_canonical(canonical)
            reverse = _blank_rail(first, second, mirror, True)
        return PairState(index, canonical, mirror, forward, reverse)

    def observe_pair(self, a: int, b: int, weight: float | int = 1.0,
                     phase: float | int = 0.0, relation_id: int = 0, *,
                     rail: int | None = None) -> PairState:
        """Add one weighted phasor observation to a pair cell.

        The update is additive: ``weight`` accumulates with a sign, amplitude
        accumulates the absolute evidence, and ``phase`` accumulates modulo one
        full turn.  With ``rail`` omitted, the canonical orientation updates
        ``s+`` and its reverse updates ``s-``; passing ``rail`` selects a rail
        explicitly.  Both rails receive the same new generation so the mirror
        metadata stays coherent.
        """
        weight_units = to_fixed(weight, "weight")
        phase_units = to_phase(phase, "phase")
        relation_units = _bounded_int(relation_id, 0, _MAX_U32, "relation_id")
        index, canonical, mirror = self._resolve_pair(a, b)
        if rail is None:
            rail = RAIL_S_PLUS if address_of(a, b) == canonical else RAIL_S_MINUS
        else:
            rail = _check_rail(rail)

        forward = self._read_rail_index(index, RAIL_S_PLUS)
        reverse = self._read_rail_index(index, RAIL_S_MINUS)
        if not forward.initialized or not reverse.initialized:
            self._write_pair_metadata(index, canonical)
            forward = self._read_rail_index(index, RAIL_S_PLUS)
            reverse = self._read_rail_index(index, RAIL_S_MINUS)

        generation = (max(forward.generation, reverse.generation) + 1) & _MAX_U32
        forward, reverse = _coherent_rails(forward, reverse, canonical, generation)
        chosen = forward if rail == RAIL_S_PLUS else reverse
        updated = chosen._replace(
            phase=(chosen.phase + phase_units) % PHASE_ONE,
            amplitude=_saturate_i32(chosen.amplitude + abs(weight_units)),
            weight=_saturate_i32(chosen.weight + weight_units),
            relation_chain=(relation_units if relation_units
                            else chosen.relation_chain),
            generation=generation,
            flags=chosen.flags | RAIL_FLAG_INITIALIZED,
        )
        if rail == RAIL_S_PLUS:
            forward = updated
        else:
            reverse = updated
        self._write_rail_index(index, RAIL_S_PLUS, forward)
        self._write_rail_index(index, RAIL_S_MINUS, reverse)
        self.generation = max(self.generation, generation)
        return PairState(index, canonical, mirror, forward, reverse)

    def decay_pair(self, a: int, b: int, factor: float | int = 0.97, *,
                   rail: int | None = None) -> PairState:
        """Multiply weights, amplitudes and phases of a cell by ``factor``.

        With ``rail`` omitted both rails decay together; otherwise only the
        selected rail is scaled.  Both rails share the resulting generation.
        """
        units = _ratio_units(factor)
        index, canonical, mirror = self._resolve_pair(a, b)
        forward = self._read_rail_index(index, RAIL_S_PLUS)
        reverse = self._read_rail_index(index, RAIL_S_MINUS)
        if not forward.initialized or not reverse.initialized:
            self._write_pair_metadata(index, canonical)
            forward = self._read_rail_index(index, RAIL_S_PLUS)
            reverse = self._read_rail_index(index, RAIL_S_MINUS)
        generation = (max(forward.generation, reverse.generation) + 1) & _MAX_U32
        forward, reverse = _coherent_rails(forward, reverse, canonical, generation)
        if rail is None:
            forward = _decayed_rail(forward, units, generation)
            reverse = _decayed_rail(reverse, units, generation)
        elif _check_rail(rail) == RAIL_S_PLUS:
            forward = _decayed_rail(forward, units, generation)
            reverse = reverse._replace(generation=generation)
        else:
            forward = forward._replace(generation=generation)
            reverse = _decayed_rail(reverse, units, generation)
        self._write_rail_index(index, RAIL_S_PLUS, forward)
        self._write_rail_index(index, RAIL_S_MINUS, reverse)
        self.generation = max(self.generation, generation)
        return PairState(index, canonical, mirror, forward, reverse)

    def decay_all(self, factor: float | int = 0.97) -> int:
        """Decay every pair cell; return the number of cells touched."""
        units = _ratio_units(factor)
        for index in range(PAIR_COUNT):
            forward = self._read_rail_index(index, RAIL_S_PLUS)
            reverse = self._read_rail_index(index, RAIL_S_MINUS)
            generation = (max(forward.generation, reverse.generation) + 1) & _MAX_U32
            forward, reverse = _coherent_rails(
                forward, reverse, _PAIR_CANONICAL[index], generation)
            self._write_rail_index(
                index, RAIL_S_PLUS, _decayed_rail(forward, units, generation))
            self._write_rail_index(
                index, RAIL_S_MINUS, _decayed_rail(reverse, units, generation))
            self.generation = max(self.generation, generation)
        return PAIR_COUNT

    # -- introspection --------------------------------------------------
    def stats(self) -> dict:
        return {
            "head_count": HEAD_COUNT,
            "pair_count": PAIR_COUNT,
            "rails": RAILS,
            "head_record_bytes": HEAD_RECORD_BYTES,
            "rail_record_bytes": RAIL_RECORD_BYTES,
            "pair_cell_bytes": PAIR_CELL_BYTES,
            "head_region_bytes": HEAD_REGION_BYTES,
            "pair_region_bytes": PAIR_REGION_BYTES,
            "region_bytes": REGION_BYTES,
            "base_offset": self.base_offset,
            "generation": self.generation,
        }

    @staticmethod
    def layout() -> dict:
        return {
            "heads": HEADS_REGION_OFFSET,
            "pairs": PAIRS_REGION_OFFSET,
            "head_records": HEAD_COUNT,
            "pair_cells": PAIR_COUNT,
            "head_record_bytes": HEAD_RECORD_BYTES,
            "rail_record_bytes": RAIL_RECORD_BYTES,
            "pair_cell_bytes": PAIR_CELL_BYTES,
            "region_bytes": REGION_BYTES,
        }


def _bounded_int(value, minimum: int, maximum: int, name: str) -> int:
    number = _require_int(value, name)
    if not minimum <= number <= maximum:
        raise ValueError("%s must be in [%d, %d]: %r"
                         % (name, minimum, maximum, value))
    return number


def _blank_rail(source: int, target: int, mirror: int, reverse: bool) -> RailRecord:
    if reverse:
        source, target = target, source
    return RailRecord(source, target, RAIL_FLAG_INITIALIZED,
                      0, 0, 0, mirror, 0, 0, 0)


def _coherent_rails(forward: RailRecord, reverse: RailRecord, canonical: int,
                    generation: int) -> tuple[RailRecord, RailRecord]:
    """Re-assert canonical source/target/mirror metadata on both rails."""
    first, second = _heads_from_canonical(canonical)
    mirror = reverse16(canonical)
    return (
        forward._replace(source=first, target=second, mirror=mirror,
                         generation=generation,
                         flags=forward.flags | RAIL_FLAG_INITIALIZED),
        reverse._replace(source=second, target=first, mirror=mirror,
                         generation=generation,
                         flags=reverse.flags | RAIL_FLAG_INITIALIZED),
    )


def _decayed_rail(record: RailRecord, units: int, generation: int) -> RailRecord:
    return record._replace(
        weight=_saturate_i32(_scale_fixed(record.weight, units)),
        amplitude=_saturate_i32(_scale_fixed(record.amplitude, units)),
        phase=_scale_fixed(record.phase, units) % PHASE_ONE,
        generation=generation,
        flags=record.flags | RAIL_FLAG_INITIALIZED,
    )


def format_cells(buffer, base_offset: int = 0) -> ResidentCells:
    """Module-level convenience wrapper around :meth:`ResidentCells.format_region`."""
    return ResidentCells.format_region(buffer, base_offset)


__all__ = (
    "HEAD_COUNT", "PAIR_COUNT", "RAILS", "FIXED_ONE", "PHASE_ONE",
    "RAIL_S_PLUS", "RAIL_S_MINUS", "RAIL_NAMES", "RAIL_FLAG_INITIALIZED",
    "HEADS_REGION_OFFSET", "PAIRS_REGION_OFFSET", "HEAD_STRUCT", "RAIL_STRUCT",
    "HEAD_RECORD_BYTES", "RAIL_RECORD_BYTES", "PAIR_CELL_BYTES",
    "HEAD_REGION_BYTES", "PAIR_REGION_BYTES", "REGION_BYTES",
    "HEAD_RECORD_SIZE", "RAIL_RECORD_SIZE", "PAIR_CELL_SIZE",
    "HEADS_BYTES", "PAIRS_BYTES", "HeadRecord", "RailRecord", "PairState",
    "ResidentCells", "format_cells", "reverse8", "reverse16", "address_of",
    "canonical_address", "mirror_address", "identity_address", "pair_index",
    "to_fixed", "from_fixed", "to_phase", "from_phase",
)
