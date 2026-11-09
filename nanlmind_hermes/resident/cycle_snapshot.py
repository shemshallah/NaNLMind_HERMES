"""Read-only inspector for Round 030 ``NANCYC30`` native snapshots.

A Round 030 snapshot is an 8-byte ASCII marker followed by the exact 8 MiB
native allocation used by ``experiments/round030/native_cycle.cpp``::

    "NANCYC30" | heads[256][128] | pairs[32640][forward 32*float32, reverse 32*float32]

There are 32,640 unordered pair cells.  For canonical heads ``i < j`` the
``forward`` lane stores ``x(i, j, c)`` and the ``reverse`` lane stores
``x(j, i, c)``.  The native ``skew`` helper writes ``-x`` into the opposite
direction, so mirror lanes are antisymmetric as float32 values.

This module is deliberately a *reader and bit mover*, not a new semantic
format.  It never merges canonical identities, never rewrites a snapshot and
never claims that a float32 word and a NaN payload word are the same numerical
object.  ``to_nan_lane`` copies selected 32-bit words into a raw
:mod:`nanlmind_hermes.resident.nan_lane` lane without scaling or conversion.
That operation is lossless only as a bit move; no arithmetic equivalence between
the float interpretation and the NaN payload is asserted.

No NumPy is used.  Parsing, hashing and word access use ``struct`` and ordinary
integer/bytes operations.  Full-file loading is restricted to the roots listed
in the repository's ``experiments/round030/SHA256SUMS``; arbitrary paths,
symlinks and path traversal are rejected.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import struct
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

try:  # Optional at import time, but present in this package.
    from . import nan_lane as _nan_lane
except ImportError:  # pragma: no cover - only on an incomplete installation
    _nan_lane = None

# ---------------------------------------------------------------------------
# fixed Round 030 layout
# ---------------------------------------------------------------------------
SCHEMA = "nanlmind.resident.cycle-snapshot/1"
MARKER = b"NANCYC30"
MARKER_BYTES = 8

HEAD_COUNT = 256
HEAD_BYTES = 128
HEAD_REGION_BYTES = HEAD_COUNT * HEAD_BYTES          # 32,768

PAIR_COUNT = HEAD_COUNT * (HEAD_COUNT - 1) // 2      # 32,640
DIRECTION_WORDS = 32
WORD_BYTES = 4
LANE_BYTES = DIRECTION_WORDS * WORD_BYTES            # 128
PAIR_BYTES = 2 * LANE_BYTES                          # 256
PAIR_REGION_BYTES = PAIR_COUNT * PAIR_BYTES          # 8,355,840

ROOT_BYTES = HEAD_REGION_BYTES + PAIR_REGION_BYTES   # 8,388,608
SNAPSHOT_BYTES = MARKER_BYTES + ROOT_BYTES           # 8,388,616

WORD_BITS = 32
WORD_MODULUS = 1 << WORD_BITS
WORD_MASK = WORD_MODULUS - 1
FLOAT_EXPONENT_MASK = 0x7F800000
FLOAT_SIGN_BIT = 0x80000000

FLUX_DIMENSIONS = 24
FLUX_CHANNELS = 16
DEFAULT_ROOT = 0

# Explicit non-claims.  A raw payload lane is a storage bridge, not an IEEE
# arithmetic identity and not a canonical merge of two universes.
CLAIMS_FLOAT_IDENTITY = False
PERFORMS_CANONICAL_MERGE = False

ROOT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "round030"
SHA256SUMS_PATH = ROOT_DIR / "SHA256SUMS"
ROOT_FILENAME_RE = re.compile(r"^root_([0-9]+)\.bin$")

_PAIR_WORDS = struct.Struct("<64I")
_UINT32 = struct.Struct("<I")
_FLOAT32 = struct.Struct("<f")

assert _PAIR_WORDS.size == PAIR_BYTES
assert _UINT32.size == WORD_BYTES
assert _FLOAT32.size == WORD_BYTES
assert HEAD_REGION_BYTES + PAIR_REGION_BYTES == ROOT_BYTES == 8388608
assert SNAPSHOT_BYTES == 8388616


# ---------------------------------------------------------------------------
# errors and scalar validation
# ---------------------------------------------------------------------------
class SnapshotError(ValueError):
    """Base class for snapshot inspection failures."""


class SnapshotFormatError(SnapshotError):
    """The byte stream is not a valid Round 030 snapshot."""


class SnapshotSizeError(SnapshotFormatError):
    """A snapshot or supplied root has the wrong exact byte count."""


class SnapshotMarkerError(SnapshotFormatError):
    """The eight-byte ``NANCYC30`` marker is absent or wrong."""


class SnapshotPathError(SnapshotError):
    """A path is not one of the supplied, approved root files."""


class SnapshotIntegrityError(SnapshotError):
    """A supplied root does not match its SHA256SUMS digest."""


class SnapshotRangeError(SnapshotError, IndexError):
    """A head, pair, channel, limit or word is outside its fixed bound."""


class SnapshotDependencyError(SnapshotError, RuntimeError):
    """The raw NaN-payload lane module is not available."""


class SnapshotLaneCapacityError(SnapshotRangeError):
    """More 32-bit words were selected than one raw lane can carry."""


def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotRangeError("%s must be an integer: %r" % (name, value))
    return value


def _check_head(value, name: str = "head") -> int:
    value = _require_int(value, name)
    if not 0 <= value < HEAD_COUNT:
        raise SnapshotRangeError(
            "%s must be in [0, %d): %r" % (name, HEAD_COUNT, value))
    return value


def _check_channel(value, name: str = "channel") -> int:
    value = _require_int(value, name)
    if not 0 <= value < DIRECTION_WORDS:
        raise SnapshotRangeError(
            "%s must be in [0, %d): %r" % (name, DIRECTION_WORDS, value))
    return value


def _check_pair_heads(first, second) -> tuple[int, int]:
    left = _check_head(first, "i")
    right = _check_head(second, "j")
    if left == right:
        raise SnapshotRangeError(
            "a pair needs two distinct heads: %r" % (left,))
    return left, right


def _check_pair_index(value, name: str = "index") -> int:
    value = _require_int(value, name)
    if not 0 <= value < PAIR_COUNT:
        raise SnapshotRangeError(
            "%s must be in [0, %d): %r" % (name, PAIR_COUNT, value))
    return value


def _check_limit(value, maximum: int, name: str = "limit") -> int:
    if value is None:
        return maximum
    value = _require_int(value, name)
    if not 0 <= value <= maximum:
        raise SnapshotRangeError(
            "%s must be in [0, %d]: %r" % (name, maximum, value))
    return value


def _check_word(value, name: str = "word") -> int:
    value = _require_int(value, name)
    if not 0 <= value <= WORD_MASK:
        raise SnapshotRangeError(
            "%s must be an unsigned 32-bit word in [0, %d]: %r"
            % (name, WORD_MASK, value))
    return value


# ---------------------------------------------------------------------------
# pair addressing exactly matching native_cycle.cpp
# ---------------------------------------------------------------------------
def pair_index(first: int, second: int) -> int:
    """Return the native dense index of an unordered head pair."""
    left, right = _check_pair_heads(first, second)
    if left > right:
        left, right = right, left
    return left * (2 * HEAD_COUNT - left - 1) // 2 + (right - left - 1)


def pair_heads(index: int) -> tuple[int, int]:
    """Invert :func:`pair_index`, returning heads in increasing order."""
    index = _check_pair_index(index)
    low, high = 0, HEAD_COUNT - 2
    while low < high:
        middle = (low + high + 1) // 2
        if middle * (2 * HEAD_COUNT - middle - 1) // 2 <= index:
            low = middle
        else:
            high = middle - 1
    first = low
    second = index - first * (2 * HEAD_COUNT - first - 1) // 2 + first + 1
    if not (0 <= first < second < HEAD_COUNT):  # pragma: no cover - invariant
        raise SnapshotRangeError("pair index does not map to two heads")
    return first, second


class PairIdentity(NamedTuple):
    """Canonical orientation of one shared pair cell."""

    index: int
    first: int
    second: int

    @property
    def canonical(self) -> tuple[int, int]:
        return self.first, self.second

    @property
    def mirror(self) -> tuple[int, int]:
        return self.second, self.first

    @property
    def mirror_index(self) -> int:
        """The mirror direction addresses the same physical pair cell."""
        return self.index


def pair_identity(first: int, second: int | None = None) -> PairIdentity:
    """Return identity for ``(i, j)`` or for a dense pair index.

    ``pair_identity(3, 240)`` and ``pair_identity(pair_index(3, 240))`` both
    return the canonical low-to-high orientation.  Reversing ``(i, j)`` keeps
    the same cell index and swaps :attr:`PairIdentity.mirror`.
    """
    if second is None:
        index = _check_pair_index(first)
        left, right = pair_heads(index)
    else:
        left, right = _check_pair_heads(first, second)
        index = pair_index(left, right)
        left, right = pair_heads(index)
    return PairIdentity(index, left, right)


class PairCell(NamedTuple):
    """One 256-byte pair cell with both 32-word float32 lanes as raw bits.

    ``forward_words`` are the bit patterns of the canonical ``i < j`` lane and
    ``reverse_words`` are the bit patterns of the ``j -> i`` lane.  The raw
    words are authoritative; the float views are convenience decodings only.
    """

    index: int
    first: int
    second: int
    forward_words: tuple[int, ...]
    reverse_words: tuple[int, ...]

    @property
    def canonical(self) -> tuple[int, int]:
        return self.first, self.second

    @property
    def mirror(self) -> tuple[int, int]:
        return self.second, self.first

    @property
    def mirror_index(self) -> int:
        return self.index

    @property
    def forward(self) -> tuple[int, ...]:
        """Alias for the canonical forward raw words."""
        return self.forward_words

    @property
    def reverse(self) -> tuple[int, ...]:
        """Alias for the canonical reverse raw words."""
        return self.reverse_words

    @property
    def forward_float_words(self) -> tuple[float, ...]:
        return tuple(_float_from_bits(word) for word in self.forward_words)

    @property
    def reverse_float_words(self) -> tuple[float, ...]:
        return tuple(_float_from_bits(word) for word in self.reverse_words)

    # Explicit spellings for callers that want the decoded convenience view.
    forward_floats = forward_float_words
    reverse_floats = reverse_float_words

    def oriented_words(self, source: int, target: int) -> tuple[int, ...]:
        """Return the raw words stored for the ``source -> target`` direction."""
        left, right = _check_pair_heads(source, target)
        if pair_index(left, right) != self.index:
            raise SnapshotRangeError(
                "direction %r does not belong to pair %d"
                % ((source, target), self.index))
        return self.forward_words if left < right else self.reverse_words

    def oriented_float_words(self, source: int,
                             target: int) -> tuple[float, ...]:
        return tuple(_float_from_bits(word)
                     for word in self.oriented_words(source, target))


# ---------------------------------------------------------------------------
# raw/float word helpers (no NumPy, no implicit payload canonicalisation)
# ---------------------------------------------------------------------------
def _float_from_bits(word: int) -> float:
    word = _check_word(word)
    return _FLOAT32.unpack(_UINT32.pack(word))[0]


def _float_to_bits(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotRangeError("float value must be int or float: %r"
                                 % (value,))
    try:
        packed = _FLOAT32.pack(float(value))
    except (OverflowError, struct.error) as error:
        raise SnapshotRangeError(
            "value is outside the binary32 range: %r" % (value,)) from error
    return _UINT32.unpack(packed)[0]


def _round_float32(value: float) -> float:
    """Round one arithmetic result to binary32, matching native ``float``."""
    try:
        return _FLOAT32.unpack(_FLOAT32.pack(value))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def _float_is_finite_bits(word: int) -> bool:
    return (word & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK


def _mirror_words_are_antisymmetric(forward: int, reverse: int) -> bool:
    """Check the sign-bit mirror relation, allowing either zero encoding.

    For every finite non-zero float32 value ``v`` the native ``-v`` operation
    flips exactly the sign bit.  The C++ initialiser leaves both directions at
    positive zero, and ``+0.0 == -0.0`` under float comparison, so either zero
    sign is accepted.
    """
    if (forward & ~FLOAT_SIGN_BIT) == 0 and (reverse & ~FLOAT_SIGN_BIT) == 0:
        return True
    return (forward ^ reverse) == FLOAT_SIGN_BIT


# ---------------------------------------------------------------------------
# bounded snapshot reader
# ---------------------------------------------------------------------------
def _as_snapshot_bytes(data, name: str = "snapshot") -> bytes:
    if isinstance(data, str):
        raise TypeError("%s data must be bytes-like, not str" % name)
    try:
        view = memoryview(data)
    except TypeError as error:
        raise TypeError("%s data must be a bytes-like object" % name) from error
    try:
        view = view.cast("B")
    except TypeError:
        view = memoryview(view.tobytes())
    raw = view.tobytes()
    if len(raw) != SNAPSHOT_BYTES:
        raise SnapshotSizeError(
            "%s must be exactly %d bytes, got %d"
            % (name, SNAPSHOT_BYTES, len(raw)))
    if raw[:MARKER_BYTES] != MARKER:
        raise SnapshotMarkerError(
            "%s marker must be %r, got %r"
            % (name, MARKER, raw[:MARKER_BYTES]))
    return raw


class Snapshot:
    """Parsed, immutable view of one ``NANCYC30`` byte stream."""

    __slots__ = ("_data", "_root", "_source", "_sha256")

    def __init__(self, data, *, source: str = "<memory>"):
        raw = _as_snapshot_bytes(data)
        self._data = raw
        self._root = raw[MARKER_BYTES:]
        self._source = str(source)
        self._sha256 = hashlib.sha256(raw).hexdigest()

    # -- whole-file metadata -------------------------------------------
    @classmethod
    def from_bytes(cls, data, *, source: str = "<memory>") -> "Snapshot":
        return cls(data, source=source)

    @property
    def schema(self) -> str:
        return SCHEMA

    @property
    def source(self) -> str:
        return self._source

    @property
    def marker(self) -> bytes:
        return MARKER

    @property
    def raw(self) -> bytes:
        """The complete marker-prefixed file bytes."""
        return self._data

    @property
    def root(self) -> bytes:
        """The exact 8 MiB root region without the marker."""
        return self._root

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def file_bytes(self) -> int:
        return len(self._data)

    @property
    def root_bytes(self) -> int:
        return len(self._root)

    @property
    def head_count(self) -> int:
        return HEAD_COUNT

    @property
    def pair_count(self) -> int:
        return PAIR_COUNT

    @property
    def direction_words(self) -> int:
        return DIRECTION_WORDS

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return ("Snapshot(source=%r, pairs=%d, sha256=%s)"
                % (self._source, PAIR_COUNT, self._sha256[:12]))

    # -- 256 head records ----------------------------------------------
    def head_bytes(self, index: int) -> bytes:
        """Return the exact 128 raw bytes of one head record."""
        index = _check_head(index)
        offset = index * HEAD_BYTES
        return self._root[offset:offset + HEAD_BYTES]

    # Short spelling used by snapshot inspectors.
    head = head_bytes

    @property
    def heads(self) -> tuple[bytes, ...]:
        """All 256 head records as exact 128-byte immutable values."""
        return tuple(self.iter_heads())

    def iter_heads(self, limit=None):
        """Yield at most ``limit`` head records (all 256 by default)."""
        count = _check_limit(limit, HEAD_COUNT)
        for index in range(count):
            yield self.head_bytes(index)

    # -- 32,640 pair cells ---------------------------------------------
    def pair_cell(self, first: int, second: int | None = None) -> PairCell:
        """Read one cell by dense index or by two distinct heads."""
        if second is None:
            index = _check_pair_index(first)
            left, right = pair_heads(index)
        else:
            left, right = _check_pair_heads(first, second)
            index = pair_index(left, right)
            left, right = pair_heads(index)
        offset = PAIR_REGION_OFFSET + index * PAIR_BYTES
        words = _PAIR_WORDS.unpack_from(self._root, offset)
        return PairCell(index, left, right, words[:DIRECTION_WORDS],
                        words[DIRECTION_WORDS:])

    def iter_pairs(self, limit=None):
        """Yield bounded :class:`PairCell` values in native address order."""
        count = _check_limit(limit, PAIR_COUNT)
        for index in range(count):
            yield self.pair_cell(index)

    # Common spellings; both are bounded generators.
    pair_cells = iter_pairs
    iter_pair_cells = iter_pairs

    def pair_identity(self, first: int, second: int | None = None):
        return pair_identity(first, second)

    # -- lanes and triangles -------------------------------------------
    def raw_words(self, first: int, second: int, channel=None):
        """Return raw 32-bit words for the ``first -> second`` direction."""
        return snapshot_raw_lanes(self, first, second, channel)

    def float_words(self, first: int, second: int, channel=None):
        """Return binary32 decodings for the ``first -> second`` direction."""
        return snapshot_float_words(self, first, second, channel)

    def cycle(self, i: int, j: int, c: int, root: int = DEFAULT_ROOT) -> float:
        return snapshot_cycle(self, i, j, c, root=root)

    def audit(self, *, limit=None):
        return audit(self, limit=limit)


PAIR_REGION_OFFSET = HEAD_REGION_BYTES


def _coerce_snapshot(snapshot) -> Snapshot:
    if isinstance(snapshot, Snapshot):
        return snapshot
    return Snapshot(snapshot)


def snapshot_head_bytes(snapshot, index: int) -> bytes:
    """Return one 128-byte head record from a snapshot or raw bytes."""
    return _coerce_snapshot(snapshot).head_bytes(index)


def snapshot_pair_cell(snapshot, first: int, second: int | None = None):
    """Return one :class:`PairCell` by dense index or two heads."""
    return _coerce_snapshot(snapshot).pair_cell(first, second)


def snapshot_pair_cells(snapshot, limit=None):
    """Yield bounded :class:`PairCell` values in native address order."""
    return _coerce_snapshot(snapshot).iter_pairs(limit=limit)


def _oriented_words(snapshot: Snapshot, first: int, second: int
                    ) -> tuple[int, ...]:
    left, right = _check_pair_heads(first, second)
    identity = pair_identity(left, right)
    offset = PAIR_REGION_OFFSET + identity.index * PAIR_BYTES
    words = _PAIR_WORDS.unpack_from(snapshot.root, offset)
    if left < right:
        return words[:DIRECTION_WORDS]
    return words[DIRECTION_WORDS:]


def _edge_float(snapshot: Snapshot, first: int, second: int,
                channel: int) -> float:
    return _float_from_bits(_oriented_words(snapshot, first, second)[channel])


def snapshot_raw_lanes(snapshot, first: int, second: int, channel=None):
    """Return bit-preserving 32-bit words for the ``first -> second`` lane.

    With ``channel=None`` the result is a tuple of 32 unsigned words.  With a
    channel index the result is one unsigned word.  No float interpretation is
    performed, so NaN payloads, infinities, subnormals and signed zeros are
    returned exactly as stored.
    """
    snap = _coerce_snapshot(snapshot)
    words = _oriented_words(snap, first, second)
    if channel is None:
        return words
    return words[_check_channel(channel)]


def snapshot_raw_lane(snapshot, first: int, second: int, channel=None):
    """Alias for :func:`snapshot_raw_lanes` (one word per named lane)."""
    return snapshot_raw_lanes(snapshot, first, second, channel)


def snapshot_float_words(snapshot, first: int, second: int, channel=None):
    """Decode the ``first -> second`` float32 lane as ordinary Python floats.

    This is a convenience view for reading the native trained values.  The
    authoritative representation remains :func:`snapshot_raw_lanes`; no claim
    is made that these floats are NaN payload lanes or vice versa.
    """
    snap = _coerce_snapshot(snapshot)
    words = _oriented_words(snap, first, second)
    if channel is None:
        return tuple(_float_from_bits(word) for word in words)
    return _float_from_bits(words[_check_channel(channel)])


def snapshot_cycle(snapshot, i: int, j: int, c: int,
                   root: int = DEFAULT_ROOT) -> float:
    """Read three float32 lanes and compute the native triangle flux.

    ``flux = x(root, i, c) + x(i, j, c) - x(root, j, c)``

    Each intermediate result is rounded to binary32, matching the native C++
    ``float`` expression.  The function returns a value; it does not assert
    that the same value is representable or equivalent in a NaN payload lane.
    """
    snap = _coerce_snapshot(snapshot)
    left, middle = _check_pair_heads(i, j)
    channel = _check_channel(c)
    anchor = _check_head(root, "root")
    if anchor == left or anchor == middle:
        raise SnapshotRangeError(
            "root must differ from both cycle heads: %r" % (anchor,))
    first = _edge_float(snap, anchor, left, channel)
    second = _edge_float(snap, left, middle, channel)
    third = _edge_float(snap, anchor, middle, channel)
    return _round_float32(_round_float32(first + second) - third)


def snapshot_flux(snapshot, dimension: int, channel: int) -> float:
    """Round 030 ``flux(d, c)`` mapping ``a = 1 + 2*d``, ``b = a + 1``."""
    dimension = _require_int(dimension, "dimension")
    if not 0 <= dimension < FLUX_DIMENSIONS:
        raise SnapshotRangeError(
            "dimension must be in [0, %d): %r"
            % (FLUX_DIMENSIONS, dimension))
    channel = _check_channel(channel)
    first = 1 + 2 * dimension
    return snapshot_cycle(snapshot, first, first + 1, channel,
                          root=DEFAULT_ROOT)


# ---------------------------------------------------------------------------
# audit report
# ---------------------------------------------------------------------------
class AuditReport(dict):
    """Dictionary report whose truth value is its ``ok`` field."""

    def __bool__(self) -> bool:
        return bool(self.get("ok"))

    @property
    def ok(self) -> bool:
        return bool(self.get("ok"))

    @property
    def passed(self) -> bool:
        return bool(self.get("ok"))

    def __repr__(self) -> str:
        return "AuditReport(ok=%r, pairs=%r, finite=%r, mirror=%r)" % (
            self.get("ok"), self.get("pairs_checked"),
            self.get("finite_bits"), self.get("mirror_antisymmetry"))


def audit(snapshot, *, limit=None) -> AuditReport:
    """Check size/marker, float finiteness and mirror antisymmetry.

    ``limit`` bounds the number of pair cells inspected; the default is all
    32,640 cells.  This is a fixed, bounded scan, never an open-ended sampler.
    """
    snap = _coerce_snapshot(snapshot)
    pairs_checked = _check_limit(limit, PAIR_COUNT)
    mirror_ok = True
    finite_ok = True
    mirror_mismatches = 0
    nonfinite_words = 0
    first_bad_pair = None
    first_bad_channel = None

    for index in range(pairs_checked):
        offset = PAIR_REGION_OFFSET + index * PAIR_BYTES
        words = _PAIR_WORDS.unpack_from(snap.root, offset)
        for channel in range(DIRECTION_WORDS):
            forward = words[channel]
            reverse = words[DIRECTION_WORDS + channel]
            if not _float_is_finite_bits(forward):
                nonfinite_words += 1
                finite_ok = False
                if first_bad_pair is None:
                    first_bad_pair = index
                    first_bad_channel = channel
            if not _float_is_finite_bits(reverse):
                nonfinite_words += 1
                finite_ok = False
                if first_bad_pair is None:
                    first_bad_pair = index
                    first_bad_channel = channel
            if not _mirror_words_are_antisymmetric(forward, reverse):
                mirror_mismatches += 1
                mirror_ok = False
                if first_bad_pair is None:
                    first_bad_pair = index
                    first_bad_channel = channel

    size_ok = snap.file_bytes == SNAPSHOT_BYTES and snap.root_bytes == ROOT_BYTES
    marker_ok = snap.marker == MARKER
    report = AuditReport(
        schema=SCHEMA,
        source=snap.source,
        sha256=snap.sha256,
        file_bytes=snap.file_bytes,
        expected_file_bytes=SNAPSHOT_BYTES,
        root_bytes=snap.root_bytes,
        expected_root_bytes=ROOT_BYTES,
        size_ok=size_ok,
        marker_ok=marker_ok,
        pair_count=PAIR_COUNT,
        pairs_checked=pairs_checked,
        all_pairs_checked=pairs_checked == PAIR_COUNT,
        mirror_antisymmetry=mirror_ok,
        mirror_mismatches=mirror_mismatches,
        finite_bits=finite_ok,
        nonfinite_words=nonfinite_words,
        first_bad_pair=first_bad_pair,
        first_bad_channel=first_bad_channel,
        float_identity_claim=CLAIMS_FLOAT_IDENTITY,
        canonical_merge=PERFORMS_CANONICAL_MERGE,
        ok=bool(size_ok and marker_ok and mirror_ok and finite_ok),
    )
    return report


# ---------------------------------------------------------------------------
# lossless bit bridge into raw NaN-payload lanes
# ---------------------------------------------------------------------------
def _looks_like_snapshot_bytes(source) -> bool:
    if isinstance(source, (str, int, bool)):
        return False
    try:
        return memoryview(source).nbytes == SNAPSHOT_BYTES
    except TypeError:
        return False


def _coerce_lane_words(source, i, j, channel):
    if isinstance(source, Snapshot) or _looks_like_snapshot_bytes(source):
        snap = _coerce_snapshot(source)
        if i is None or j is None:
            raise SnapshotRangeError(
                "a snapshot bridge needs i, j and an explicit channel")
        if channel is None:
            raise SnapshotLaneCapacityError(
                "one directional lane has 32 float32 words but a raw nan_lane "
                "Lane carries only %d payload channels; select one channel "
                "or pass at most %d raw words"
                % (_nan_lane.CHANNEL_COUNT, _nan_lane.CHANNEL_COUNT))
        return (snapshot_raw_lanes(snap, i, j, _check_channel(channel)),)

    if i is not None or j is not None or channel is not None:
        raise SnapshotRangeError(
            "raw lane words do not take head/channel arguments")

    if isinstance(source, bool):
        raise SnapshotRangeError("raw lane word must not be bool")
    if isinstance(source, int):
        return (_check_word(source),)
    if isinstance(source, str):
        raise TypeError("raw lane words must be integers, not str")
    if isinstance(source, (bytes, bytearray, memoryview)):
        raise TypeError("packed bytes are not a list of 32-bit words")
    try:
        words = tuple(source)
    except TypeError as error:
        raise TypeError(
            "raw lane words must be an int or an iterable of ints") from error
    return tuple(_check_word(word, "word[%d]" % index)
                 for index, word in enumerate(words))


def to_nan_lane(source, i=None, j=None, channel=None):
    """Move selected 32-bit words into a raw ``nan_lane.Lane``.

    Accepted forms::

        to_nan_lane(snapshot, i, j, channel)   # one float32 bit pattern
        to_nan_lane(0x7FC00001)                # one raw word
        to_nan_lane([word_0, ..., word_27])    # at most 28 raw words

    A :class:`~nanlmind_hermes.resident.nan_lane.Lane` has 28 payload channels,
    so a complete 32-word directional lane is *not* silently truncated.  Pass
    an explicit channel for snapshot input.  Padding channels are zero and are
    explicit, not lost data.  ``TAG_RAW`` accepts every bit pattern, including
    signalling NaNs, quiet NaNs, infinities, subnormals and signed zero.
    """
    if _nan_lane is None:  # pragma: no cover - optional installation guard
        raise SnapshotDependencyError(
            "nan_lane is unavailable; cannot bridge raw payload lanes")
    words = _coerce_lane_words(source, i, j, channel)
    capacity = _nan_lane.CHANNEL_COUNT
    if len(words) > capacity:
        raise SnapshotLaneCapacityError(
            "%d raw words exceed the %d payload channels of one nan_lane.Lane"
            % (len(words), capacity))
    if not words:
        raise SnapshotLaneCapacityError("at least one raw word is required")
    padded = words + (0,) * (capacity - len(words))
    return _nan_lane.Lane.new(_nan_lane.TAG_RAW, channels=padded)


def from_nan_lane(lane, *, limit=None):
    """Return raw channel words from a raw ``nan_lane.Lane`` without decoding.

    Only the 28 payload channels are returned; tag/version, generation and
    reserved words are structural and are never treated as float payload.
    """
    if _nan_lane is None:  # pragma: no cover - optional installation guard
        raise SnapshotDependencyError(
            "nan_lane is unavailable; cannot inspect payload lanes")
    if not isinstance(lane, _nan_lane.Lane):
        raise TypeError("from_nan_lane expects a nan_lane.Lane")
    channels = lane.channels
    count = _check_limit(limit, len(channels), "limit")
    return channels[:count]


def to_nan_lane_bytes(source, i=None, j=None, channel=None) -> bytes:
    """Return the 128-byte encoded form of :func:`to_nan_lane`."""
    return to_nan_lane(source, i, j, channel).to_bytes()


# ---------------------------------------------------------------------------
# approved supplied-root loading and SHA256SUMS verification
# ---------------------------------------------------------------------------
def _sha256sums(path=None) -> MappingProxyType:
    checksum_path = Path(SHA256SUMS_PATH if path is None else path)
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise SnapshotPathError(
            "SHA256SUMS is unavailable at the approved root directory") from error
    entries = {}
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SnapshotIntegrityError(
                "malformed SHA256SUMS line %d" % number)
        digest, filename = parts
        digest = digest.lower()
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise SnapshotIntegrityError(
                "malformed SHA256SUMS digest on line %d" % number)
        if (filename.startswith("*") or "/" in filename
                or "\\" in filename or filename in (".", "..")):
            raise SnapshotIntegrityError(
                "unsafe SHA256SUMS filename on line %d" % number)
        if filename in entries:
            raise SnapshotIntegrityError(
                "duplicate SHA256SUMS filename %r" % filename)
        entries[filename] = digest
    return MappingProxyType(entries)


def supplied_root_names() -> tuple[str, ...]:
    """Return the sorted ``root_*.bin`` names listed in SHA256SUMS."""
    entries = _sha256sums()
    return tuple(sorted(name for name in entries
                        if ROOT_FILENAME_RE.match(name)))


def supplied_root_digests() -> MappingProxyType:
    """Return an immutable filename-to-SHA256 mapping for supplied roots."""
    entries = _sha256sums()
    return MappingProxyType({
        name: digest for name, digest in entries.items()
        if ROOT_FILENAME_RE.match(name)})


def _resolve_root_name(source, entries) -> str:
    if isinstance(source, bool):
        raise SnapshotPathError("root source must be a seed or filename")
    if isinstance(source, int):
        seeds = {}
        for name in entries:
            match = ROOT_FILENAME_RE.match(name)
            if match is None:
                continue
            seed = int(match.group(1))
            if seed in seeds:
                raise SnapshotPathError(
                    "ambiguous supplied seed %d" % seed)
            seeds[seed] = name
        if source not in seeds:
            raise SnapshotPathError(
                "seed %d is not one of the supplied roots" % source)
        return seeds[source]

    if isinstance(source, bytes):
        raise SnapshotPathError("root source must be text, not bytes")
    try:
        text = os.fspath(source)
    except TypeError as error:
        raise SnapshotPathError(
            "root source must be an int seed, filename or path") from error
    if not isinstance(text, str):
        raise SnapshotPathError("root source path must be text")
    if not text or "\x00" in text:
        raise SnapshotPathError("root source path is empty or contains NUL")
    if "\\" in text:
        raise SnapshotPathError("backslash path separators are not accepted")
    path = Path(text)
    if path.name in ("", ".", "..") or not ROOT_FILENAME_RE.match(path.name):
        raise SnapshotPathError(
            "only supplied root_<seed>.bin files are accepted")
    if len(path.parts) == 1:
        name = path.name
    else:
        if ".." in path.parts:
            raise SnapshotPathError("path traversal is not accepted")
        try:
            resolved = path.resolve(strict=False)
            expected = (ROOT_DIR / path.name).resolve(strict=False)
        except OSError as error:
            raise SnapshotPathError("root path could not be resolved") from error
        if resolved != expected:
            raise SnapshotPathError(
                "only the supplied roots in the approved directory are read")
        name = path.name
    if name not in entries:
        raise SnapshotPathError(
            "%r is not listed in the approved SHA256SUMS" % name)
    return name


def _read_root_file(name: str) -> bytes:
    path = ROOT_DIR / name
    try:
        link_stat = os.lstat(path)
    except OSError as error:
        raise SnapshotPathError(
            "supplied root %r is not available" % name) from error
    if stat.S_ISLNK(link_stat.st_mode):
        raise SnapshotPathError(
            "supplied root %r is a symlink and is rejected" % name)
    if not stat.S_ISREG(link_stat.st_mode):
        raise SnapshotPathError(
            "supplied root %r is not a regular file" % name)

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SnapshotPathError(
            "supplied root %r could not be opened safely" % name) from error
    try:
        file_stat = os.fstat(descriptor)
        if stat.S_ISLNK(file_stat.st_mode):
            raise SnapshotPathError(
                "supplied root %r became a symlink and is rejected" % name)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SnapshotPathError(
                "supplied root %r is not a regular file" % name)
        if file_stat.st_size != SNAPSHOT_BYTES:
            raise SnapshotSizeError(
                "supplied root %r must be exactly %d bytes, got %d"
                % (name, SNAPSHOT_BYTES, file_stat.st_size))
        chunks = []
        remaining = SNAPSHOT_BYTES
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise SnapshotSizeError(
                    "supplied root %r ended before %d bytes"
                    % (name, SNAPSHOT_BYTES))
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = os.read(descriptor, 1)
        if extra:
            raise SnapshotSizeError(
                "supplied root %r has bytes beyond %d"
                % (name, SNAPSHOT_BYTES))
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_snapshot(source, *, verify_sha256: bool = True) -> Snapshot:
    """Load one *supplied* root by seed, filename or safe exact path.

    The source is accepted only when it names an entry listed in the approved
    ``SHA256SUMS``.  The final path component must not be a symlink and the
    complete file is hashed before parsing.  There is no parameter for an
    arbitrary directory.  :func:`parse_snapshot` remains available for callers
    that already hold bytes in memory.
    """
    entries = _sha256sums()
    name = _resolve_root_name(source, entries)
    expected = entries[name]
    data = _read_root_file(name)
    actual = hashlib.sha256(data).hexdigest()
    if verify_sha256 and actual != expected:
        raise SnapshotIntegrityError(
            "supplied root %r SHA256 mismatch: expected %s, got %s"
            % (name, expected, actual))
    return Snapshot(data, source=name)


def parse_snapshot(data, *, source: str = "<memory>",
                   expected_sha256: str | None = None) -> Snapshot:
    """Parse bytes; optionally require a caller-supplied whole-file digest."""
    snapshot = _coerce_snapshot(data)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str):
            raise TypeError("expected_sha256 must be hex text or None")
        if snapshot.sha256 != expected_sha256.lower():
            raise SnapshotIntegrityError(
                "snapshot SHA256 mismatch: expected %s, got %s"
                % (expected_sha256.lower(), snapshot.sha256))
    if source != "<memory>" and snapshot.source == "<memory>":
        snapshot._source = str(source)
    return snapshot


def verify_supplied_roots(names=None):
    """Load and hash supplied roots, returning a per-file verification report.

    This is intentionally a read-only convenience: it never writes, repairs or
    merges a snapshot.
    """
    entries = _sha256sums()
    if names is None:
        selected = tuple(sorted(name for name in entries
                                if ROOT_FILENAME_RE.match(name)))
    else:
        if isinstance(names, (str, bytes)):
            names = (names,)
        selected = tuple(_resolve_root_name(name, entries) for name in names)
    report = {}
    for name in selected:
        snapshot = load_snapshot(name)
        report[name] = {
            "sha256": snapshot.sha256,
            "expected_sha256": entries[name],
            "matches": snapshot.sha256 == entries[name],
            "file_bytes": snapshot.file_bytes,
            "source": snapshot.source,
        }
    return MappingProxyType(report)


# Common explicit aliases.
parse_snapshot_bytes = parse_snapshot
load_supplied_root = load_snapshot
verify_sha256sums = verify_supplied_roots


__all__ = (
    "SCHEMA", "MARKER", "MARKER_BYTES", "HEAD_COUNT", "HEAD_BYTES",
    "HEAD_REGION_BYTES", "PAIR_COUNT", "DIRECTION_WORDS", "WORD_BYTES",
    "LANE_BYTES", "PAIR_BYTES", "PAIR_REGION_BYTES", "PAIR_REGION_OFFSET",
    "ROOT_BYTES", "SNAPSHOT_BYTES", "WORD_BITS", "WORD_MODULUS", "WORD_MASK",
    "FLOAT_EXPONENT_MASK", "FLOAT_SIGN_BIT", "FLUX_DIMENSIONS",
    "FLUX_CHANNELS", "DEFAULT_ROOT", "CLAIMS_FLOAT_IDENTITY",
    "PERFORMS_CANONICAL_MERGE", "ROOT_DIR", "SHA256SUMS_PATH",
    "ROOT_FILENAME_RE", "SnapshotError", "SnapshotFormatError",
    "SnapshotSizeError", "SnapshotMarkerError", "SnapshotPathError",
    "SnapshotIntegrityError", "SnapshotRangeError",
    "SnapshotDependencyError", "SnapshotLaneCapacityError", "PairIdentity",
    "PairCell", "Snapshot", "pair_index", "pair_heads", "pair_identity",
    "snapshot_head_bytes", "snapshot_pair_cell", "snapshot_pair_cells",
    "snapshot_raw_lanes", "snapshot_raw_lane", "snapshot_float_words",
    "snapshot_cycle", "snapshot_flux", "AuditReport", "audit",
    "to_nan_lane", "from_nan_lane", "to_nan_lane_bytes",
    "_float_from_bits", "_float_to_bits", "supplied_root_names",
    "supplied_root_digests", "load_snapshot", "parse_snapshot",
    "parse_snapshot_bytes", "load_supplied_root", "verify_supplied_roots",
    "verify_sha256sums",
)
