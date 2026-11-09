"""Raw 32-bit oriented cycle observables over resident pair rails.

The resident :class:`~nanlmind_hermes.resident.cells.ResidentCells` region
stores a Q16.16 ``weight`` in each directional rail.  This module deliberately
ignores the float interpretation of those fixed-point lanes: a rail's 32-bit
weight word is read and written as an element of ``Z / 2**32 Z``.  No scaling,
rounding or float conversion is performed.  The integer amplitude/phase lanes
and all pair/head metadata stay untouched.

For distinct heads ``i`` and ``j`` let ``x(i, j)`` be the raw word on the rail
that points from ``i`` to ``j``.  The fundamental cycle relative to ``root`` is

    c(i, j, root) = x(root, i) + x(i, j) - x(root, j)   (mod 2**32).

If a vertex gauge ``g`` is added by ``x(i, j) += g[i] - g[j]`` then

    c' = x(root, i) + g[root] - g[i]
       + x(i, j)    + g[i]    - g[j]
       - x(root, j) - g[root] + g[j]
      = c                                                     (mod 2**32).

Every intermediate addition/subtraction is ordinary integer arithmetic followed
by one mask to 32 bits, so the cancellation is exact for all 2**32 words.  The
complete-graph cycle space rooted at head 0 has C(255, 2) = 32,385 fundamental
cycles; :meth:`CycleField.batch_cycles` is bounded by that fixed count.

``CycleField`` accepts three kinds of cell store:

* ``ResidentCells`` itself - the raw weight lane of each directional rail is
  used directly;
* a duck-typed store with ``read_rail(a, b, rail)`` plus ``write_rail(...)`` or
  ``_write_rail_index(...)`` - useful for fakes and alternate layouts;
* a store with ``read_raw_word(a, b)`` / ``write_raw_word(a, b, word)`` (or the
  ``read_word`` / ``write_word`` spellings) that already owns raw word lanes.

Writes never touch the head region.  Before a write, the permanent
``head_id``/``identity`` of every involved head is verified, and initialized
pair-rail source/target/mirror metadata must already agree with the cached
canonical identity.  A mismatch raises :class:`CycleFieldIdentityError` instead
of repairing the metadata by rewriting it.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from .cells import (
    HEAD_COUNT,
    PAIR_CELL_BYTES,
    PAIR_COUNT,
    RAIL_FLAG_INITIALIZED,
    RAIL_RECORD_BYTES,
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    RAIL_STRUCT,
    RailRecord,
    address_of,
    canonical_address,
    identity_address,
    pair_index,
    reverse8,
    reverse16,
)

try:  # cells.py already owns the authoritative dense identity cache.
    from .cells import _PAIR_CANONICAL as _CELL_CANONICALS
except (ImportError, AttributeError):  # pragma: no cover - defensive fallback
    _CELL_CANONICALS = None

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
WORD_MODULUS = 1 << WORD_BITS
ROOT_CYCLE_COUNT = (HEAD_COUNT - 1) * (HEAD_COUNT - 2) // 2  # 32,385
DEFAULT_ROOT = 0
SCHEMA = "nanlmind.resident.cycle-observable/1"


# ---------------------------------------------------------------------------
# errors and small value helpers
# ---------------------------------------------------------------------------
class CycleFieldError(ValueError):
    """Base error for invalid cycle-field input or state."""


class CycleFieldRangeError(CycleFieldError):
    """A head, pair, limit or gauge index is outside its fixed bound."""


class CycleFieldWordError(CycleFieldError):
    """A raw word is not an unsigned 32-bit integer."""


class CycleFieldIdentityError(CycleFieldError):
    """A write would contradict permanent head/pair identity metadata."""


class CycleFieldBackendError(CycleFieldError):
    """The injected cell store returned invalid data or lacks write access."""


def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleFieldRangeError(
            "%s must be an integer: %r" % (name, value))
    return value


def _check_head(value, name: str = "head") -> int:
    value = _require_int(value, name)
    if not 0 <= value < HEAD_COUNT:
        raise CycleFieldRangeError(
            "%s must be in [0, %d): %r" % (name, HEAD_COUNT, value))
    return value


def _check_pair(first, second) -> tuple[int, int]:
    left = _check_head(first, "i")
    right = _check_head(second, "j")
    if left == right:
        raise CycleFieldRangeError(
            "a directional pair needs two distinct heads: %r" % (left,))
    return left, right


def _check_triple(first, second, third) -> tuple[int, int, int]:
    values = (
        _check_head(first, "a"),
        _check_head(second, "b"),
        _check_head(third, "c"),
    )
    if len(set(values)) != 3:
        raise CycleFieldRangeError(
            "a triangle needs three distinct heads: %r" % (values,))
    return values


def _check_word(value, name: str = "raw_word") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleFieldWordError(
            "%s must be an integer: %r" % (name, value))
    if not 0 <= value <= WORD_MASK:
        raise CycleFieldWordError(
            "%s must be an unsigned 32-bit word in [0, %d]: %r"
            % (name, WORD_MASK, value))
    return value


def _to_signed(word: int) -> int:
    """Map an unsigned word to the two's-complement value of an i32 lane."""
    word &= WORD_MASK
    return word if word < (1 << 31) else word - WORD_MODULUS


def _to_word(lane: int) -> int:
    """Read any signed i32 lane as its exact unsigned 32-bit word."""
    return int(lane) & WORD_MASK


def word_distance(word: int) -> int:
    """Unsigned modular distance from zero (0 .. 2**31)."""
    word = _check_word(word)
    return min(word, WORD_MODULUS - word)


def heads_from_canonical(canonical: int) -> tuple[int, int]:
    """Decode the stored forward orientation from a canonical address."""
    canonical = _require_int(canonical, "canonical")
    if not 0 <= canonical <= 0xFFFF:
        raise CycleFieldRangeError(
            "canonical must be a 16-bit address: %r" % (canonical,))
    first = canonical >> 8
    second = reverse8(canonical & 0xFF)
    if first == second:
        raise CycleFieldRangeError(
            "diagonal address %04x has no pair cell" % canonical)
    return first, second


def _build_canonical_cache() -> tuple[int, ...]:
    if _CELL_CANONICALS is not None and len(_CELL_CANONICALS) == PAIR_COUNT:
        return tuple(_CELL_CANONICALS)
    # Fallback only; the normal path shares cells.py's cached identity table.
    return tuple(
        canonical_address(first, second)
        for first in range(HEAD_COUNT)
        for second in range(first + 1, HEAD_COUNT)
    )


PAIR_CANONICALS = _build_canonical_cache()


class PairIdentity(NamedTuple):
    index: int
    canonical: int
    mirror: int
    first: int
    second: int


class PairWords(NamedTuple):
    """Two raw words in the cell's canonical forward/reverse orientation."""

    index: int
    canonical: int
    mirror: int
    forward: int
    reverse: int


class CycleSample(NamedTuple):
    root: int
    left: int
    right: int
    word: int


def pair_identity(index: int) -> PairIdentity:
    """Return cached canonical/mirror identity for a dense pair index."""
    index = _require_int(index, "index")
    if not 0 <= index < PAIR_COUNT:
        raise CycleFieldRangeError(
            "index must be in [0, %d): %r" % (PAIR_COUNT, index))
    canonical = PAIR_CANONICALS[index]
    first, second = heads_from_canonical(canonical)
    return PairIdentity(index, canonical, reverse16(canonical), first, second)


def _rail_for(first: int, second: int, canonical: int) -> int:
    return (RAIL_S_PLUS if address_of(first, second) == canonical
            else RAIL_S_MINUS)


def _coerce_gauge(raw_gauge_vector) -> tuple[int, ...]:
    """Validate exactly 256 unsigned words without unbounded consumption."""
    if isinstance(raw_gauge_vector, (str, bytes, bytearray)):
        raise CycleFieldWordError("raw_gauge_vector must be a sequence of ints")
    try:
        iterator = iter(raw_gauge_vector)
    except TypeError as error:
        raise CycleFieldWordError(
            "raw_gauge_vector must be an iterable of 256 ints") from error
    words = []
    for index in range(HEAD_COUNT):
        try:
            value = next(iterator)
        except StopIteration as error:
            raise CycleFieldWordError(
                "raw_gauge_vector must contain exactly %d words" % HEAD_COUNT
            ) from error
        words.append(_check_word(value, "raw_gauge_vector[%d]" % index))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise CycleFieldWordError(
            "raw_gauge_vector must contain at most %d words" % HEAD_COUNT)
    return tuple(words)


# ---------------------------------------------------------------------------
# cell-store adapters
# ---------------------------------------------------------------------------
class _Backend:
    """Small adapter surface used by :class:`CycleField`."""

    def __init__(self, cells):
        self.cells = cells

    def read_word(self, first: int, second: int) -> int:
        raise NotImplementedError

    def write_word(self, first: int, second: int, word: int) -> None:
        raise NotImplementedError

    def pair_words(self, index: int) -> tuple[int, int]:
        first, second = heads_from_canonical(PAIR_CANONICALS[index])
        return (self.read_word(first, second),
                self.read_word(second, first))

    def apply_gauge(self, gauge: tuple[int, ...]) -> int:
        count = 0
        for index in range(PAIR_COUNT):
            first, second = heads_from_canonical(PAIR_CANONICALS[index])
            delta = (gauge[first] - gauge[second]) & WORD_MASK
            forward = (self.read_word(first, second) + delta) & WORD_MASK
            reverse = (self.read_word(second, first) - delta) & WORD_MASK
            self.write_word(first, second, forward)
            self.write_word(second, first, reverse)
            count += 2
        return count


class _RailBackend(_Backend):
    """Use the two 32-byte rail records of each resident pair cell."""

    def __init__(self, cells):
        super().__init__(cells)
        self._buffer = getattr(cells, "buffer", None)
        self._pairs_offset = getattr(cells, "pairs_offset", None)
        self._direct_buffer = (
            self._buffer is not None and isinstance(self._pairs_offset, int))
        self._read_rail = getattr(cells, "read_rail", None)
        self._write_rail = getattr(cells, "write_rail", None)
        self._write_rail_index = getattr(cells, "_write_rail_index", None)
        self._rail_record_offset = getattr(cells, "rail_record_offset", None)
        self._pair_index_method = getattr(cells, "pair_index", None)
        if self._read_rail is None:
            raise TypeError("rail backend requires read_rail(a, b, rail)")
        if not any((self._direct_buffer, self._write_rail,
                    self._write_rail_index, self._rail_record_offset)):
            raise TypeError(
                "rail backend requires writable rail access or a raw buffer")

    def _index(self, first: int, second: int) -> int:
        if self._pair_index_method is not None:
            return int(self._pair_index_method(first, second))
        return pair_index(first, second)

    def _read_pair_records(self, index: int) -> tuple[RailRecord, RailRecord]:
        if self._direct_buffer:
            offset = self._pairs_offset + index * PAIR_CELL_BYTES
            forward = RailRecord(*RAIL_STRUCT.unpack_from(self._buffer, offset))
            reverse = RailRecord(*RAIL_STRUCT.unpack_from(
                self._buffer, offset + RAIL_RECORD_BYTES))
            return forward, reverse
        first, second = heads_from_canonical(PAIR_CANONICALS[index])
        return (self._read_rail(first, second, RAIL_S_PLUS),
                self._read_rail(first, second, RAIL_S_MINUS))

    def _write_pair_records(self, index: int, forward: RailRecord,
                            reverse: RailRecord) -> None:
        if self._direct_buffer:
            offset = self._pairs_offset + index * PAIR_CELL_BYTES
            RAIL_STRUCT.pack_into(self._buffer, offset, *forward)
            RAIL_STRUCT.pack_into(
                self._buffer, offset + RAIL_RECORD_BYTES, *reverse)
            return
        first, second = heads_from_canonical(PAIR_CANONICALS[index])
        if self._write_rail is not None:
            self._write_rail(first, second, RAIL_S_PLUS, forward)
            self._write_rail(first, second, RAIL_S_MINUS, reverse)
            return
        if self._write_rail_index is not None:
            self._write_rail_index(index, RAIL_S_PLUS, forward)
            self._write_rail_index(index, RAIL_S_MINUS, reverse)
            return
        if self._rail_record_offset is not None and self._buffer is not None:
            offset = self._rail_record_offset(first, second, RAIL_S_PLUS)
            RAIL_STRUCT.pack_into(self._buffer, offset, *forward)
            offset = self._rail_record_offset(first, second, RAIL_S_MINUS)
            RAIL_STRUCT.pack_into(self._buffer, offset, *reverse)
            return
        raise CycleFieldBackendError("rail backend has no write path")

    @staticmethod
    def _normalize_pair(index: int, forward: RailRecord,
                        reverse: RailRecord) -> tuple[RailRecord, RailRecord]:
        canonical = PAIR_CANONICALS[index]
        first, second = heads_from_canonical(canonical)
        mirror = reverse16(canonical)
        records = []
        for rail, record in enumerate((forward, reverse)):
            source = first if rail == RAIL_S_PLUS else second
            target = second if rail == RAIL_S_PLUS else first
            if record.flags & RAIL_FLAG_INITIALIZED:
                if (record.source != source or record.target != target
                        or record.mirror != mirror):
                    raise CycleFieldIdentityError(
                        "pair %d rail identity mismatch: expected %d->%d "
                        "mirror %04x, found %d->%d mirror %04x"
                        % (index, source, target, mirror,
                           record.source, record.target, record.mirror))
                records.append(record)
            else:
                # Only derived pair metadata is initialized.  No head record
                # is ever written by this adapter.
                records.append(RailRecord(
                    source, target, record.flags | RAIL_FLAG_INITIALIZED,
                    record.phase, record.amplitude, record.weight,
                    mirror, record.relation_chain, record.generation,
                    record.reserved))
        return records[0], records[1]

    def _bump_generation(self, generation: int) -> None:
        current = getattr(self.cells, "generation", None)
        if isinstance(current, int) and not isinstance(current, bool):
            self.cells.generation = max(current, generation)

    def read_word(self, first: int, second: int) -> int:
        index = self._index(first, second)
        canonical = PAIR_CANONICALS[index]
        rail = _rail_for(first, second, canonical)
        forward, reverse = self._read_pair_records(index)
        return _to_word(forward.weight if rail == RAIL_S_PLUS
                        else reverse.weight)

    def write_word(self, first: int, second: int, word: int) -> None:
        index = self._index(first, second)
        canonical = PAIR_CANONICALS[index]
        rail = _rail_for(first, second, canonical)
        forward, reverse = self._read_pair_records(index)
        forward, reverse = self._normalize_pair(index, forward, reverse)
        generation = (max(forward.generation, reverse.generation) + 1) \
            & WORD_MASK
        if rail == RAIL_S_PLUS:
            forward = forward._replace(
                weight=_to_signed(word), flags=forward.flags
                | RAIL_FLAG_INITIALIZED, generation=generation)
            reverse = reverse._replace(
                flags=reverse.flags | RAIL_FLAG_INITIALIZED,
                generation=generation)
        else:
            reverse = reverse._replace(
                weight=_to_signed(word), flags=reverse.flags
                | RAIL_FLAG_INITIALIZED, generation=generation)
            forward = forward._replace(
                flags=forward.flags | RAIL_FLAG_INITIALIZED,
                generation=generation)
        self._write_pair_records(index, forward, reverse)
        self._bump_generation(generation)

    def pair_words(self, index: int) -> tuple[int, int]:
        forward, reverse = self._read_pair_records(index)
        return _to_word(forward.weight), _to_word(reverse.weight)

    def apply_gauge(self, gauge: tuple[int, ...]) -> int:
        count = 0
        for index in range(PAIR_COUNT):
            first, second = heads_from_canonical(PAIR_CANONICALS[index])
            forward, reverse = self._read_pair_records(index)
            forward, reverse = self._normalize_pair(index, forward, reverse)
            delta = (gauge[first] - gauge[second]) & WORD_MASK
            generation = (max(forward.generation, reverse.generation) + 1) \
                & WORD_MASK
            forward = forward._replace(
                weight=_to_signed((_to_word(forward.weight) + delta)
                                  & WORD_MASK),
                flags=forward.flags | RAIL_FLAG_INITIALIZED,
                generation=generation)
            reverse = reverse._replace(
                weight=_to_signed((_to_word(reverse.weight) - delta)
                                  & WORD_MASK),
                flags=reverse.flags | RAIL_FLAG_INITIALIZED,
                generation=generation)
            self._write_pair_records(index, forward, reverse)
            self._bump_generation(generation)
            count += 2
        return count


class _RawWordBackend(_Backend):
    """Adapter for a store that already exposes raw oriented word lanes."""

    def __init__(self, cells):
        super().__init__(cells)
        self._read = getattr(cells, "read_raw_word", None)
        self._write = getattr(cells, "write_raw_word", None)
        if self._read is None or self._write is None:
            self._read = getattr(cells, "read_word", None)
            self._write = getattr(cells, "write_word", None)
        if self._read is None or self._write is None:
            raise TypeError(
                "raw-word backend requires read_raw_word/write_raw_word")

    def read_word(self, first: int, second: int) -> int:
        value = self._read(first, second)
        if isinstance(value, bool) or not isinstance(value, int):
            raise CycleFieldBackendError(
                "raw-word store returned a non-integer: %r" % (value,))
        return value & WORD_MASK

    def write_word(self, first: int, second: int, word: int) -> None:
        self._write(first, second, word)


def _select_backend(cells) -> _Backend:
    if hasattr(cells, "read_rail"):
        if (hasattr(cells, "write_rail")
                or hasattr(cells, "_write_rail_index")
                or (getattr(cells, "buffer", None) is not None
                    and isinstance(getattr(cells, "pairs_offset", None), int))
                or hasattr(cells, "rail_record_offset")):
            return _RailBackend(cells)
    if (hasattr(cells, "read_raw_word")
            and hasattr(cells, "write_raw_word")):
        return _RawWordBackend(cells)
    if hasattr(cells, "read_word") and hasattr(cells, "write_word"):
        return _RawWordBackend(cells)
    raise TypeError(
        "cells must expose raw rail access or raw oriented word methods")


# ---------------------------------------------------------------------------
# public field view
# ---------------------------------------------------------------------------
class CycleField:
    """Raw-word cycle observables over the resident pair cells.

    The object is a view, not a copy: reads see the current rails, writes go
    straight to the pair lanes, and the 256 head records are never touched.
    """

    head_count = HEAD_COUNT
    pair_count = PAIR_COUNT
    word_bits = WORD_BITS
    root_cycle_count = ROOT_CYCLE_COUNT

    def __init__(self, cells):
        if cells is None:
            raise TypeError("cells must not be None")
        head_count = getattr(cells, "head_count", HEAD_COUNT)
        pair_count = getattr(cells, "pair_count", PAIR_COUNT)
        if head_count != HEAD_COUNT or pair_count != PAIR_COUNT:
            raise CycleFieldRangeError(
                "cycle observables require %d heads and %d pairs, got %r/%r"
                % (HEAD_COUNT, PAIR_COUNT, head_count, pair_count))
        self.cells = cells
        self._backend = _select_backend(cells)

    @property
    def generation(self) -> int:
        value = getattr(self.cells, "generation", 0)
        return value if isinstance(value, int) and not isinstance(value, bool) \
            else 0

    # -- raw directional words -----------------------------------------
    def get_oriented(self, i: int, j: int) -> int:
        """Return the exact raw 32-bit word stored for ``i -> j``."""
        first, second = _check_pair(i, j)
        return self._backend.read_word(first, second)

    def set_oriented(self, i: int, j: int, raw_word: int) -> int:
        """Replace only the raw word in the ``i -> j`` rail lane."""
        first, second = _check_pair(i, j)
        word = _check_word(raw_word)
        self._validate_heads(first, second)
        self._backend.write_word(first, second, word)
        return word

    def mirror(self, i: int, j: int) -> int:
        """Set ``x(j, i) = -x(i, j)`` modulo 2**32 and return that word."""
        first, second = _check_pair(i, j)
        self._validate_heads(first, second)
        word = self._backend.read_word(first, second)
        mirror_word = (-word) & WORD_MASK
        self._backend.write_word(second, first, mirror_word)
        return mirror_word

    # -- cycle observables ---------------------------------------------
    def cycle(self, i: int, j: int, root: int = DEFAULT_ROOT) -> int:
        """Return ``x(root,i) + x(i,j) - x(root,j)`` modulo 2**32."""
        first, second = _check_pair(i, j)
        root = _check_head(root, "root")
        if root == first or root == second:
            raise CycleFieldRangeError(
                "root must differ from both cycle heads: %r" % (root,))
        return (self._backend.read_word(root, first)
                + self._backend.read_word(first, second)
                - self._backend.read_word(root, second)) & WORD_MASK

    def triangle(self, a: int, b: int, c: int) -> int:
        """Return the oriented three-head circulation modulo 2**32."""
        first, second, third = _check_triple(a, b, c)
        return (self._backend.read_word(first, second)
                + self._backend.read_word(second, third)
                + self._backend.read_word(third, first)) & WORD_MASK

    # -- gauge operations ----------------------------------------------
    def gauge_shift(self, raw_gauge_vector) -> int:
        """Add ``g[i] - g[j]`` to every oriented rail; return lane count.

        The complete 256-word vector is validated before any write, so an
        invalid gauge cannot leave a partially shifted field.
        """
        gauge = _coerce_gauge(raw_gauge_vector)
        self._validate_all_heads()
        return int(self._backend.apply_gauge(gauge))

    def gauge_residual(self, root: int = DEFAULT_ROOT) -> int:
        """Maximum modular residual over every fundamental cycle at ``root``."""
        root = _check_head(root, "root")
        root_words = [0] * HEAD_COUNT
        for head in range(HEAD_COUNT):
            if head != root:
                root_words[head] = self._backend.read_word(root, head)
        worst = 0
        for left in range(HEAD_COUNT):
            if left == root:
                continue
            for right in range(left + 1, HEAD_COUNT):
                if right == root:
                    continue
                word = (root_words[left]
                        + self._backend.read_word(left, right)
                        - root_words[right]) & WORD_MASK
                distance = word_distance(word)
                if distance > worst:
                    worst = distance
        return worst

    # -- bounded batches -----------------------------------------------
    @staticmethod
    def _check_limit(limit) -> int:
        if limit is None:
            return ROOT_CYCLE_COUNT
        value = _require_int(limit, "limit")
        if not 0 <= value <= ROOT_CYCLE_COUNT:
            raise CycleFieldRangeError(
                "limit must be in [0, %d]: %r"
                % (ROOT_CYCLE_COUNT, limit))
        return value

    @staticmethod
    def _iter_root_pairs(root: int):
        for left in range(HEAD_COUNT):
            if left == root:
                continue
            for right in range(left + 1, HEAD_COUNT):
                if right == root:
                    continue
                yield left, right

    def iter_cycles(self, subset=None, root: int = DEFAULT_ROOT,
                    limit=None):
        """Yield bounded :class:`CycleSample` values without a full allocation.

        With ``subset=None`` the caller receives the first ``limit`` canonical
        root cycles (all 32,385 by default).  An explicitly selected subset is
        also capped and rejected if it is longer than ``limit``.
        """
        root = _check_head(root, "root")
        cap = self._check_limit(limit)
        root_words = [0] * HEAD_COUNT
        if subset is None:
            for head in range(HEAD_COUNT):
                if head != root:
                    root_words[head] = self._backend.read_word(root, head)
            count = 0
            for left, right in self._iter_root_pairs(root):
                if count >= cap:
                    break
                word = (root_words[left]
                        + self._backend.read_word(left, right)
                        - root_words[right]) & WORD_MASK
                yield CycleSample(root, left, right, word)
                count += 1
            return

        try:
            iterator = iter(subset)
        except TypeError as error:
            raise CycleFieldRangeError("subset must be iterable") from error
        count = 0
        for item in iterator:
            if count >= cap:
                raise CycleFieldRangeError(
                    "subset contains more than limit=%d entries" % cap)
            if isinstance(item, (str, bytes, bytearray)):
                raise CycleFieldRangeError(
                    "subset entries must be (i, j), not text")
            try:
                left, right = item
            except (TypeError, ValueError) as error:
                raise CycleFieldRangeError(
                    "subset entries must be (i, j) pairs") from error
            left, right = _check_pair(left, right)
            if left == root or right == root:
                raise CycleFieldRangeError(
                    "root must differ from both subset heads")
            word = (self._backend.read_word(root, left)
                    + self._backend.read_word(left, right)
                    - self._backend.read_word(root, right)) & WORD_MASK
            yield CycleSample(root, left, right, word)
            count += 1

    def batch_cycles(self, subset=None, root: int = DEFAULT_ROOT,
                     limit=None) -> tuple[CycleSample, ...]:
        """Return a bounded tuple of root cycles (all 32,385 by default)."""
        return tuple(self.iter_cycles(subset=subset, root=root, limit=limit))

    # Common spellings, kept as aliases because the API is deliberately small.
    batch_root_cycles = batch_cycles
    cycle_batch = batch_cycles
    iter_root_cycles = iter_cycles

    def root_cycle_pairs(self, root: int = DEFAULT_ROOT,
                         limit=None) -> tuple[tuple[int, int], ...]:
        """Return the bounded ``(i, j)`` pairs used by ``batch_cycles``."""
        root = _check_head(root, "root")
        cap = self._check_limit(limit)
        pairs = []
        for left, right in self._iter_root_pairs(root):
            if len(pairs) >= cap:
                break
            pairs.append((left, right))
        return tuple(pairs)

    # -- read-only snapshot --------------------------------------------
    def snapshot(self):
        """Return an immutable copy of the raw words and every root cycle.

        The top-level mapping is a ``MappingProxyType`` and every record is an
        immutable ``NamedTuple``; no buffer, memoryview or live rail object is
        exposed.  Later field writes do not change an existing snapshot.
        """
        pairs = []
        for index in range(PAIR_COUNT):
            identity = pair_identity(index)
            forward, reverse = self._backend.pair_words(index)
            pairs.append(PairWords(
                index, identity.canonical, identity.mirror,
                forward, reverse))
        pair_words = tuple(pairs)
        cycles = self.batch_cycles(root=DEFAULT_ROOT)
        return MappingProxyType({
            "schema": SCHEMA,
            "mode": "raw-u32",
            "word_bits": WORD_BITS,
            "head_count": HEAD_COUNT,
            "pair_count": PAIR_COUNT,
            "direction_count": PAIR_COUNT * 2,
            "root": DEFAULT_ROOT,
            "root_cycle_count": ROOT_CYCLE_COUNT,
            "generation": self.generation,
            "pairs": pair_words,
            "pair_words": pair_words,
            "cycles": cycles,
            "root_cycles": cycles,
        })

    # -- identity checks -----------------------------------------------
    def _head_identity_error(self, head: int, record) -> None:
        if isinstance(record, Mapping):
            head_id = record.get("head_id")
            identity = record.get("identity")
        else:
            head_id = getattr(record, "head_id", None)
            identity = getattr(record, "identity", None)
        if head_id is not None and head_id != head:
            raise CycleFieldIdentityError(
                "head %d has permanent head_id %r" % (head, head_id))
        if identity is not None and identity != identity_address(head):
            raise CycleFieldIdentityError(
                "head %d has identity %r, expected %04x"
                % (head, identity, identity_address(head)))

    def _validate_heads(self, *heads) -> None:
        reader = getattr(self.cells, "read_head", None)
        if reader is None:
            return
        for head in dict.fromkeys(heads):
            try:
                record = reader(head)
            except Exception as error:  # noqa: BLE001 - backend boundary
                raise CycleFieldBackendError(
                    "could not read head %d: %s" % (head, error)) from error
            self._head_identity_error(head, record)

    def _validate_all_heads(self) -> None:
        reader = getattr(self.cells, "read_head", None)
        if reader is None:
            return
        for head in range(HEAD_COUNT):
            try:
                record = reader(head)
            except Exception as error:  # noqa: BLE001 - backend boundary
                raise CycleFieldBackendError(
                    "could not read head %d: %s" % (head, error)) from error
            self._head_identity_error(head, record)

    def __repr__(self) -> str:
        return ("CycleField(heads=%d, pairs=%d, generation=%d, root_cycles=%d)"
                % (HEAD_COUNT, PAIR_COUNT, self.generation,
                   ROOT_CYCLE_COUNT))


__all__ = (
    "WORD_BITS", "WORD_MASK", "WORD_MODULUS", "ROOT_CYCLE_COUNT",
    "DEFAULT_ROOT", "SCHEMA", "PAIR_CANONICALS", "PairIdentity", "PairWords",
    "CycleSample", "CycleFieldError", "CycleFieldRangeError",
    "CycleFieldWordError", "CycleFieldIdentityError", "CycleFieldBackendError",
    "word_distance", "heads_from_canonical", "pair_identity", "CycleField",
)
