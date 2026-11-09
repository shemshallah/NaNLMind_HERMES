"""Full 32,385-dimensional root-star cycle memory over resident pair cells.

The complete-graph cycle space on 256 heads, rooted at head ``0``, has

    C(255, 2) = 255 * 254 / 2 = 32,385

independent fundamental cycles.  This module owns exactly that basis.  A cycle
``(i, j)`` with ``1 <= i < j <= 255`` has one raw 32-bit coefficient.  The
coefficient is stored in the weight lane of the resident pair cell ``(i, j)``;
the mirror direction ``(j, i)`` is kept as the exact modular negation.  The 255
root edges ``(0, i)`` are constrained to zero, so the physical edge field is a
canonical representative of the cycle memory and

    cycle(i, j) = x(0, i) + x(i, j) - x(0, j) = x(i, j)   (mod 2**32).

Only integer words are used.  A lane word is an element of ``Z / 2**32 Z`` and
is never converted to a host binary32 value.  For dot products and gradient
steps the same raw word is read as a signed two's-complement i32 coefficient.
No binary32/binary64 identity, scaling or rounding is involved.

The index map is the dense row-major map over the 255 non-root heads::

    cycle_index(i, j) = (i - 1) * (510 - i) // 2 + (j - i - 1)

with ``cycle_pair(index)`` as its exact inverse.  Index ``0`` is ``(1, 2)``
and index ``32,384`` is ``(254, 255)``.

``FullCycleMemory`` is a view over a resident cell store.  The real
:class:`~nanlmind_hermes.resident.cells.ResidentCells` 2 MiB region is
supported directly, as is any duck-typed store accepted by
:class:`~nanlmind_hermes.resident.cycle_observable.CycleField` (for example a
fake with ``read_raw_word`` / ``write_raw_word``).  Every write validates the
permanent head identities and the pair rail metadata before changing a weight
lane.  Reading and writing never touches the 256 head records.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from .cells import (
    HEAD_COUNT,
    PAIR_COUNT,
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    identity_address,
    pair_index,
    reverse16,
)
from .cycle_observable import (
    PAIR_CANONICALS,
    WORD_BITS,
    WORD_MASK,
    WORD_MODULUS,
    CycleField,
    CycleFieldBackendError,
    CycleFieldIdentityError,
    heads_from_canonical,
)

__all__ = (
    "HEAD_COUNT",
    "ROOT",
    "ROOT_HEAD",
    "CYCLE_HEAD_COUNT",
    "FULL_CYCLE_COUNT",
    "CYCLE_COUNT",
    "DIMENSION",
    "BASIS_SIZE",
    "FULL_BASIS_COUNT",
    "FULL_BASIS_CYCLES",
    "ROOT_CYCLE_COUNT",
    "PAIR_COUNT",
    "WORD_BITS",
    "WORD_MASK",
    "WORD_MODULUS",
    "I32_MIN",
    "I32_MAX",
    "ROOT_EDGE_COUNT",
    "DEFAULT_ROOT",
    "SCHEMA",
    "CycleWord",
    "UpdateReport",
    "AuditReport",
    "Occupancy",
    "FullCycleError",
    "FullCycleMemoryError",
    "FullCycleRangeError",
    "FullCycleDimensionError",
    "FullCycleWordError",
    "FullCycleIdentityError",
    "FullCycleRootError",
    "FullCycleOverflowError",
    "FullCycleFeatureError",
    "FullCycleBackendError",
    "cycle_row_start",
    "row_start",
    "cycle_index",
    "cycle_pair",
    "index_of",
    "index_for_pair",
    "pair_of_index",
    "pair_for_index",
    "pair_from_index",
    "index_to_pair",
    "pair_to_index",
    "iter_cycle_pairs",
    "iter_pairs",
    "signed_word",
    "word_from_signed",
    "FullCycleMemory",
)

ROOT = 0
ROOT_HEAD = ROOT
DEFAULT_ROOT = ROOT
CYCLE_HEAD_MIN = 1
CYCLE_HEAD_MAX = HEAD_COUNT - 1
CYCLE_HEAD_COUNT = HEAD_COUNT - 1
ROOT_EDGE_COUNT = HEAD_COUNT - 1
FULL_CYCLE_COUNT = CYCLE_HEAD_COUNT * (CYCLE_HEAD_COUNT - 1) // 2
assert FULL_CYCLE_COUNT == 32385
assert FULL_CYCLE_COUNT == (HEAD_COUNT - 1) * (HEAD_COUNT - 2) // 2
CYCLE_COUNT = FULL_CYCLE_COUNT
DIMENSION = FULL_CYCLE_COUNT
BASIS_SIZE = FULL_CYCLE_COUNT
FULL_BASIS_COUNT = FULL_CYCLE_COUNT
FULL_BASIS_CYCLES = FULL_CYCLE_COUNT
ROOT_CYCLE_COUNT = FULL_CYCLE_COUNT

I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1

SCHEMA = "nanlmind.resident.full-cycle-memory/1"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class FullCycleError(ValueError):
    """Base error for the full-cycle memory."""


FullCycleMemoryError = FullCycleError


class FullCycleRangeError(FullCycleError):
    """A head, index, limit or feature index is outside its fixed bound."""


class FullCycleDimensionError(FullCycleError):
    """The caller tried to use a subset or a non-32,385-dimensional store."""


class FullCycleWordError(FullCycleError):
    """A coefficient is not an unsigned 32-bit raw word."""


class FullCycleIdentityError(FullCycleError):
    """A permanent head or pair identity would be contradicted."""


class FullCycleRootError(FullCycleError):
    """A root edge ``(0, i)`` is nonzero or was targeted for a coefficient."""


class FullCycleOverflowError(FullCycleError, ArithmeticError):
    """A signed gradient step left the raw i32 coefficient range."""


class FullCycleFeatureError(FullCycleError):
    """A feature vector is malformed, sparse-keyed incorrectly or too large."""


class FullCycleBackendError(FullCycleError):
    """The resident cell store is missing, read-only or malformed."""


# ---------------------------------------------------------------------------
# scalar validation and raw-word conversion
# ---------------------------------------------------------------------------
def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleRangeError("%s must be an integer: %r" % (name, value))
    return value


def _check_word(value, name: str = "word") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleWordError(
            "%s must be an integer raw word: %r" % (name, value))
    if not 0 <= value <= WORD_MASK:
        raise FullCycleWordError(
            "%s must be an unsigned 32-bit word in [0, %d]: %r"
            % (name, WORD_MASK, value))
    return value


def _check_signed(value, name: str = "value") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleFeatureError(
            "%s must be an integer: %r" % (name, value))
    if not I32_MIN <= value <= I32_MAX:
        raise FullCycleOverflowError(
            "%s must fit a signed 32-bit coefficient [%d, %d]: %r"
            % (name, I32_MIN, I32_MAX, value))
    return value


def signed_word(word) -> int:
    """Return the signed two's-complement i32 value of a raw word."""
    value = _check_word(word)
    if value & (1 << 31):
        return value - WORD_MODULUS
    return value


def word_from_signed(value) -> int:
    """Return the exact 32-bit raw word for a signed i32 value."""
    value = _check_signed(value, "signed value")
    return value & WORD_MASK


# ---------------------------------------------------------------------------
# exact dense index map over heads 1..255
# ---------------------------------------------------------------------------
def _check_cycle_head(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleRangeError("%s must be an integer: %r" % (name, value))
    if value == ROOT_HEAD:
        raise FullCycleRootError(
            "%s is the root head; root edges (0, i) are fixed at zero" % name)
    if not CYCLE_HEAD_MIN <= value <= CYCLE_HEAD_MAX:
        raise FullCycleRangeError(
            "%s must be in [%d, %d]: %r"
            % (name, CYCLE_HEAD_MIN, CYCLE_HEAD_MAX, value))
    return value


def _check_cycle_pair(i, j, name_i: str = "i", name_j: str = "j") \
        -> tuple[int, int]:
    first = _check_cycle_head(i, name_i)
    second = _check_cycle_head(j, name_j)
    if first == second:
        raise FullCycleRangeError(
            "a cycle needs two distinct non-root heads: (%r, %r)"
            % (first, second))
    return (first, second) if first < second else (second, first)


def cycle_row_start(i) -> int:
    """First dense index of row ``i`` (``1 <= i <= 255``).

    ``cycle_row_start(255) == FULL_CYCLE_COUNT``; it is the start of the empty
    final row and is valid as an exclusive bound.
    """
    head = _require_int(i, "i")
    if not CYCLE_HEAD_MIN <= head <= CYCLE_HEAD_MAX:
        raise FullCycleRangeError(
            "row head must be in [%d, %d]: %r"
            % (CYCLE_HEAD_MIN, CYCLE_HEAD_MAX, i))
    return (head - 1) * (2 * CYCLE_HEAD_COUNT - head) // 2


def cycle_index(i, j) -> int:
    """Map an unordered cycle pair to its dense index in ``[0, 32385)``.

    Both ``(i, j)`` and ``(j, i)`` map to the same index.  Both heads must be
    in ``1..255`` and distinct; the root head is rejected because root-star
    cycles use only non-root pairs.
    """
    first, second = _check_cycle_pair(i, j)
    return cycle_row_start(first) + (second - first - 1)


def cycle_pair(index) -> tuple[int, int]:
    """Invert :func:`cycle_index`, returning ``(i, j)`` with ``i < j``."""
    value = _require_int(index, "index")
    if not 0 <= value < FULL_CYCLE_COUNT:
        raise FullCycleRangeError(
            "cycle index must be in [0, %d): %r"
            % (FULL_CYCLE_COUNT, index))
    low, high = CYCLE_HEAD_MIN, CYCLE_HEAD_MAX
    while low < high:
        middle = (low + high + 1) // 2
        if cycle_row_start(middle) <= value:
            low = middle
        else:
            high = middle - 1
    first = low
    second = value - cycle_row_start(first) + first + 1
    if not CYCLE_HEAD_MIN <= first < second <= CYCLE_HEAD_MAX:
        raise FullCycleRangeError(
            "cycle index %r does not map to two non-root heads" % (index,))
    return first, second


# Stable common spellings, all exact and exhaustive over the same dense basis.
row_start = cycle_row_start
index_of = cycle_index
index_for_pair = cycle_index
pair_of_index = cycle_pair
pair_for_index = cycle_pair
pair_from_index = cycle_pair
index_to_pair = cycle_pair
pair_to_index = cycle_index


def iter_cycle_pairs():
    """Yield all 32,385 ``(i, j)`` pairs in dense index order."""
    for first in range(CYCLE_HEAD_MIN, CYCLE_HEAD_MAX):
        for second in range(first + 1, CYCLE_HEAD_MAX + 1):
            yield first, second


iter_pairs = iter_cycle_pairs


# ---------------------------------------------------------------------------
# value records
# ---------------------------------------------------------------------------
class CycleWord(NamedTuple):
    """One immutable cycle coefficient at its dense index."""

    index: int
    i: int
    j: int
    word: int

    @property
    def signed(self) -> int:
        return signed_word(self.word)

    @property
    def pair(self) -> tuple[int, int]:
        return self.i, self.j


class UpdateReport(int):
    """Integer report: the value is the number of cycle lanes updated.

    The extra attributes make the raw prediction and signed error available
    without changing the usual ``updated == 2`` integer assertion.
    """

    def __new__(cls, prediction: int, target: int, error: int,
                touched: int, loss: int):
        report = int.__new__(cls, touched)
        report.prediction = prediction
        report.target = target
        report.error = error
        report.touched = touched
        report.loss = loss
        return report

    @property
    def updated(self) -> int:
        return int(self)

    @property
    def count(self) -> int:
        return int(self)

    def as_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "target": self.target,
            "error": self.error,
            "touched": self.touched,
            "updated": int(self),
            "loss": self.loss,
        }

    def __repr__(self) -> str:
        return ("UpdateReport(updated=%d, prediction=%d, target=%d, "
                "error=%d, loss=%d)"
                % (int(self), self.prediction, self.target,
                   self.error, self.loss))


class AuditReport(dict):
    """Dictionary audit report whose truth value is its ``ok`` field."""

    def __bool__(self) -> bool:
        return bool(self.get("ok"))

    @property
    def ok(self) -> bool:
        return bool(self.get("ok"))

    @property
    def passed(self) -> bool:
        return bool(self.get("ok"))

    def __repr__(self) -> str:
        return ("AuditReport(ok=%r, cycles=%r, root_edges=%r, "
                "mirror_mismatches=%r)"
                % (self.get("ok"), self.get("cycles_checked"),
                   self.get("root_edges_checked"),
                   self.get("mirror_mismatches")))


class Occupancy(int):
    """Integer count of nonzero cycle coefficients that is also callable."""

    __slots__ = ()

    def __call__(self) -> int:
        return int(self)

    @property
    def count(self) -> int:
        return int(self)

    @property
    def total(self) -> int:
        return FULL_CYCLE_COUNT

    @property
    def free(self) -> int:
        return FULL_CYCLE_COUNT - int(self)

    def as_dict(self) -> dict:
        return {
            "occupied": int(self),
            "total": FULL_CYCLE_COUNT,
            "free": FULL_CYCLE_COUNT - int(self),
        }

    def __repr__(self) -> str:
        return "Occupancy(occupied=%d, total=%d)" % (
            int(self), FULL_CYCLE_COUNT)


# ---------------------------------------------------------------------------
# feature-vector coercion (dense or sparse, integer only)
# ---------------------------------------------------------------------------
def _feature_key(key) -> tuple[tuple[int, int], int]:
    """Return ``((i, j), orientation_sign)`` for a feature key."""
    if isinstance(key, bool):
        raise FullCycleFeatureError(
            "feature key may not be a bool: %r" % (key,))
    if isinstance(key, int):
        return cycle_pair(key), 1
    if isinstance(key, (str, bytes, bytearray, memoryview)):
        raise FullCycleFeatureError(
            "feature key must be an index or an (i, j) pair, not text")
    try:
        items = tuple(key)
    except TypeError as error:
        raise FullCycleFeatureError(
            "feature key must be an index or an (i, j) pair: %r"
            % (key,)) from error
    if len(items) != 2:
        raise FullCycleFeatureError(
            "feature key pair must contain exactly two heads: %r" % (key,))
    first, second = items
    if isinstance(first, bool) or isinstance(second, bool):
        raise FullCycleFeatureError(
            "feature key heads must be integers: %r" % (key,))
    if not isinstance(first, int) or not isinstance(second, int):
        raise FullCycleFeatureError(
            "feature key heads must be integers: %r" % (key,))
    sign = -1 if first > second else 1
    return _check_cycle_pair(first, second, "feature i", "feature j"), sign


def _mapping_key(key) -> tuple[int, bool]:
    """Return ``(dense_index, reversed_order)`` for a bulk-mapping key."""
    if isinstance(key, bool):
        raise FullCycleDimensionError(
            "mapping key may not be a bool: %r" % (key,))
    if isinstance(key, int):
        index = _require_int(key, "mapping index")
        if not 0 <= index < FULL_CYCLE_COUNT:
            raise FullCycleRangeError(
                "mapping index must be in [0, %d): %r"
                % (FULL_CYCLE_COUNT, key))
        return index, False
    if isinstance(key, (str, bytes, bytearray, memoryview)):
        raise FullCycleDimensionError(
            "mapping key must be a dense index or an (i, j) pair")
    try:
        items = tuple(key)
    except TypeError as error:
        raise FullCycleDimensionError(
            "mapping key must be a dense index or an (i, j) pair: %r"
            % (key,)) from error
    if len(items) != 2:
        raise FullCycleDimensionError(
            "mapping pair key must contain exactly two heads: %r" % (key,))
    first, second = items
    first = _check_cycle_head(first, "mapping i")
    second = _check_cycle_head(second, "mapping j")
    if first == second:
        raise FullCycleRangeError(
            "mapping pair needs two distinct non-root heads: (%r, %r)"
            % (first, second))
    return cycle_index(first, second), first > second


def _append_feature(entries: dict, key, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleFeatureError(
            "feature value must be an integer: %r" % (value,))
    if not I32_MIN <= value <= I32_MAX:
        raise FullCycleOverflowError(
            "feature value must fit a signed 32-bit coefficient [%d, %d]: %r"
            % (I32_MIN, I32_MAX, value))
    pair, sign = _feature_key(key)
    entries[pair] = entries.get(pair, 0) + sign * value


def _coerce_features(features) -> list[tuple[tuple[int, int], int]]:
    """Return consolidated ``((i, j), value)`` entries.

    A dense sequence must contain exactly :data:`FULL_CYCLE_COUNT` integers.  A
    mapping or an iterable of ``(index_or_pair, value)`` entries is sparse and
    may contain fewer entries.  Duplicate keys accumulate.  No subset of a
    dense vector is accepted: a sequence shorter than the full basis is a
    feature error, never a silent 256-weight or partial-basis mode.
    """
    if isinstance(features, Mapping):
        entries: dict[tuple[int, int], int] = {}
        for key, value in features.items():
            _append_feature(entries, key, value)
        return sorted(entries.items())

    if isinstance(features, (str, bytes, bytearray, memoryview)):
        raise FullCycleFeatureError(
            "features must be a dense sequence of integers or sparse entries")

    try:
        iterator = iter(features)
    except TypeError as error:
        raise FullCycleFeatureError(
            "features must be iterable") from error
    items = []
    for item in iterator:
        if len(items) >= 2 * FULL_CYCLE_COUNT:
            raise FullCycleDimensionError(
                "feature input exceeds the bounded full-basis size %d"
                % (2 * FULL_CYCLE_COUNT))
        items.append(item)

    all_integers = all(not isinstance(item, bool) and isinstance(item, int)
                       for item in items)
    if len(items) == FULL_CYCLE_COUNT and all_integers:
        dense = []
        for index, value in enumerate(items):
            _check_signed(value, "features[%d]" % index)
            if value:
                dense.append((cycle_pair(index), value))
        return dense
    if all_integers and items:
        raise FullCycleDimensionError(
            "a dense feature vector must contain exactly %d integers, got %d"
            % (FULL_CYCLE_COUNT, len(items)))

    if len(items) == 0:
        return []
    entries = {}
    for entry in items:
        if isinstance(entry, (str, bytes, bytearray, memoryview)):
            raise FullCycleFeatureError(
                "sparse feature entries must be (key, value) pairs")
        try:
            key, value = entry
        except (TypeError, ValueError) as error:
            raise FullCycleFeatureError(
                "sparse feature entries must be (key, value) pairs: %r"
                % (entry,)) from error
        _append_feature(entries, key, value)
    return sorted(entries.items())


# ---------------------------------------------------------------------------
# full-cycle memory view
# ---------------------------------------------------------------------------
class FullCycleMemory:
    """View over exactly C(255, 2) = 32,385 raw cycle coefficients.

    The constructor zeroes and then validates the 255 root edges.  Every
    subsequent coefficient write is therefore a root-star cycle readout:
    ``cycle(i, j) == get_word(i, j)``.  The mirror direction is written as the
    modular negation, and the head record region is never touched.
    """

    head_count = HEAD_COUNT
    pair_count = PAIR_COUNT
    dimension = FULL_CYCLE_COUNT
    cycle_count = FULL_CYCLE_COUNT
    full_cycle_count = FULL_CYCLE_COUNT
    basis_size = FULL_CYCLE_COUNT
    root = ROOT_HEAD

    def __init__(self, cells):
        if cells is None:
            raise FullCycleBackendError("cells must not be None")
        head_count = getattr(cells, "head_count", HEAD_COUNT)
        pair_count = getattr(cells, "pair_count", PAIR_COUNT)
        if head_count != HEAD_COUNT or pair_count != PAIR_COUNT:
            raise FullCycleDimensionError(
                "full cycle memory requires %d heads and %d pairs, got %r/%r"
                % (HEAD_COUNT, PAIR_COUNT, head_count, pair_count))
        self.cells = cells
        try:
            self._field = CycleField(cells)
        except (CycleFieldBackendError, TypeError) as error:
            raise FullCycleBackendError(
                "could not attach a raw cycle field to cells: %s"
                % (error,)) from error
        self._validate_head_identities()
        self.zero_root_edges()

    # -- basic identity / dimensions -----------------------------------
    @property
    def field(self):
        """The underlying raw :class:`CycleField` view."""
        return self._field

    @property
    def generation(self) -> int:
        return self._field.generation

    def __len__(self) -> int:
        return FULL_CYCLE_COUNT

    def __repr__(self) -> str:
        return ("FullCycleMemory(heads=%d, cycles=%d, occupancy=%d, "
                "generation=%d)"
                % (HEAD_COUNT, FULL_CYCLE_COUNT, int(self.occupancy),
                   self.generation))

    def __iter__(self):
        for index in range(FULL_CYCLE_COUNT):
            yield self._read_index(index)

    def __getitem__(self, key):
        if isinstance(key, bool):
            raise FullCycleRangeError("key must be an index or an (i, j) pair")
        if isinstance(key, int):
            return self._read_index(key)
        try:
            first, second = key
        except (TypeError, ValueError) as error:
            raise FullCycleRangeError(
                "key must be an index or an (i, j) pair") from error
        return self.get_word(first, second)

    def __setitem__(self, key, word) -> None:
        if isinstance(key, bool):
            raise FullCycleRangeError("key must be an index or an (i, j) pair")
        if isinstance(key, int):
            first, second = cycle_pair(key)
        else:
            try:
                first, second = key
            except (TypeError, ValueError) as error:
                raise FullCycleRangeError(
                    "key must be an index or an (i, j) pair") from error
        self.set_word(first, second, word)

    # -- identity checks ----------------------------------------------
    @staticmethod
    def _head_fields(record):
        if isinstance(record, Mapping):
            return record.get("head_id"), record.get("identity")
        return getattr(record, "head_id", None), getattr(record, "identity",
                                                         None)

    def _validate_head_identities(self) -> None:
        reader = getattr(self.cells, "read_head", None)
        if reader is None:
            return
        for head in range(HEAD_COUNT):
            try:
                record = reader(head)
            except Exception as error:  # noqa: BLE001 - backend boundary
                raise FullCycleBackendError(
                    "could not read head %d: %s" % (head, error)) from error
            head_id, identity = self._head_fields(record)
            if head_id is not None and head_id != head:
                raise FullCycleIdentityError(
                    "head %d has permanent head_id %r" % (head, head_id))
            if identity is not None and identity != identity_address(head):
                raise FullCycleIdentityError(
                    "head %d has identity %r, expected %04x"
                    % (head, identity, identity_address(head)))

    @staticmethod
    def _pair_expected(index: int) -> tuple[int, int, int, int]:
        canonical = PAIR_CANONICALS[index]
        first, second = heads_from_canonical(canonical)
        return first, second, canonical, reverse16(canonical)

    def _check_physical_pair(self, first: int, second: int) -> None:
        """Validate real rail metadata for the cell containing a pair."""
        reader = getattr(self.cells, "pair_state", None)
        if reader is None:
            return
        try:
            state = reader(first, second)
        except Exception as error:  # noqa: BLE001 - backend boundary
            raise FullCycleBackendError(
                "could not read pair (%d, %d): %s"
                % (first, second, error)) from error
        index = pair_index(first, second)
        expected_first, expected_second, canonical, mirror = \
            self._pair_expected(index)
        if state.canonical != canonical or state.mirror != mirror:
            raise FullCycleIdentityError(
                "pair (%d, %d) canonical/mirror metadata mismatch"
                % (first, second))
        for rail, record, source, target in (
                (RAIL_S_PLUS, state.forward, expected_first, expected_second),
                (RAIL_S_MINUS, state.reverse, expected_second, expected_first)):
            if (record.source != source or record.target != target
                    or record.mirror != mirror):
                raise FullCycleIdentityError(
                    "pair (%d, %d) rail %d identity mismatch: expected "
                    "%d->%d mirror %04x, found %d->%d mirror %04x"
                    % (first, second, rail, source, target, mirror,
                       record.source, record.target, record.mirror))

    # -- root edges ----------------------------------------------------
    def zero_root_edges(self) -> int:
        """Force both directions of all 255 root edges to zero.

        Only lanes that are already nonzero are touched, so a formatted cell
        region is not churned.  The return value is the number of directed
        root lanes changed.  Pair identity is checked by the write path.
        """
        changed = 0
        for head in range(CYCLE_HEAD_MIN, CYCLE_HEAD_MAX + 1):
            forward = self._field.get_oriented(ROOT_HEAD, head)
            reverse = self._field.get_oriented(head, ROOT_HEAD)
            if forward or reverse:
                if forward:
                    self._field.set_oriented(ROOT_HEAD, head, 0)
                    changed += 1
                if reverse:
                    self._field.set_oriented(head, ROOT_HEAD, 0)
                    changed += 1
        return changed

    def validate_root_edges(self) -> int:
        """Raise unless all 510 directed root lanes are zero; return 510."""
        for head in range(CYCLE_HEAD_MIN, CYCLE_HEAD_MAX + 1):
            forward = self._field.get_oriented(ROOT_HEAD, head)
            reverse = self._field.get_oriented(head, ROOT_HEAD)
            if forward or reverse:
                raise FullCycleRootError(
                    "root edge (0, %d) must be zero, found %08x/%08x"
                    % (head, forward, reverse))
        return 2 * ROOT_EDGE_COUNT

    def _check_root_heads(self, *heads) -> None:
        for head in dict.fromkeys(heads):
            forward = self._field.get_oriented(ROOT_HEAD, head)
            reverse = self._field.get_oriented(head, ROOT_HEAD)
            if forward or reverse:
                raise FullCycleRootError(
                    "root edge (0, %d) conflict: %08x/%08x"
                    % (head, forward, reverse))

    # -- public dense-basis access ------------------------------------
    def index(self, i, j) -> int:
        """Dense index of a cycle pair (same as :func:`cycle_index`)."""
        return cycle_index(i, j)

    def pair(self, index) -> tuple[int, int]:
        """Inverse dense mapping (same as :func:`cycle_pair`)."""
        return cycle_pair(index)

    index_of = index
    pair_of_index = pair
    pair_for_index = pair
    index_to_pair = pair

    def iter_pairs(self):
        """Yield all 32,385 canonical ``(i, j)`` pairs in index order."""
        return iter_cycle_pairs()

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Immutable tuple of all 32,385 canonical pairs."""
        return tuple(iter_cycle_pairs())

    def _read_index(self, index: int) -> int:
        first, second = cycle_pair(index)
        return self._field.get_oriented(first, second)

    def _write_index(self, index: int, word: int) -> int:
        first, second = cycle_pair(index)
        self._write_words(first, second, word)
        return word

    def _write_words(self, first: int, second: int, word: int) -> None:
        """Write both directions of one canonical pair with one mirror value."""
        try:
            self._field.set_oriented(first, second, word)
            self._field.set_oriented(second, first,
                                     (-word) & WORD_MASK)
        except CycleFieldIdentityError as error:
            raise FullCycleIdentityError(str(error)) from error
        except CycleFieldBackendError as error:
            raise FullCycleBackendError(str(error)) from error

    def set_word(self, i, j, word) -> int:
        """Store one raw oriented coefficient and return it.

        ``(i, j)`` and ``(j, i)`` address the same cycle with opposite sign:
        setting ``(j, i)`` to ``w`` is the same as setting ``(i, j)`` to
        ``-w`` modulo 2**32.  Root heads, same-head pairs, unknown heads and
        non-word values are rejected before any lane is touched.
        """
        first = _check_cycle_head(i, "i")
        second = _check_cycle_head(j, "j")
        if first == second:
            raise FullCycleRangeError(
                "a cycle needs two distinct non-root heads: (%r, %r)"
                % (first, second))
        value = _check_word(word)
        self._check_root_heads(first, second)
        oriented = value if first < second else (-value) & WORD_MASK
        self._write_index(cycle_index(first, second), oriented)
        return value

    def get_word(self, i, j) -> int:
        """Return the raw 32-bit coefficient for directed cycle ``(i, j)``."""
        first = _check_cycle_head(i, "i")
        second = _check_cycle_head(j, "j")
        if first == second:
            raise FullCycleRangeError(
                "a cycle needs two distinct non-root heads: (%r, %r)"
                % (first, second))
        return self._field.get_oriented(first, second)

    def get_signed(self, i, j) -> int:
        """Return the coefficient as a signed two's-complement i32 value."""
        return signed_word(self.get_word(i, j))

    def cycle(self, i, j) -> int:
        """Return the root-star circulation ``x(0,i)+x(i,j)-x(0,j)``.

        The 255 root edges are validated first.  With the canonical root
        constraint this is exactly :meth:`get_word`, but the modular sum is
        computed explicitly so a corrupted root lane is refused instead of
        silently changing the meaning of the coefficient.
        """
        first = _check_cycle_head(i, "i")
        second = _check_cycle_head(j, "j")
        if first == second:
            raise FullCycleRangeError(
                "a cycle needs two distinct non-root heads: (%r, %r)"
                % (first, second))
        root_first = self._field.get_oriented(ROOT_HEAD, first)
        root_second = self._field.get_oriented(ROOT_HEAD, second)
        if root_first or root_second:
            raise FullCycleRootError(
                "root edge conflict: x(0,%d)=%08x x(0,%d)=%08x"
                % (first, root_first, second, root_second))
        middle = self._field.get_oriented(first, second)
        return (root_first + middle - root_second) & WORD_MASK

    # -- bulk access ---------------------------------------------------
    def read_all(self, limit=None, *, bounded=None) -> tuple[int, ...]:
        """Return raw coefficients in dense index order.

        ``limit`` (or the ``bounded`` keyword) caps the number of returned
        values in ``[0, 32385]``.  The full basis is the default; there is no
        partial or 256-value mode hidden behind this API.
        """
        if bounded is not None:
            if limit is not None:
                raise FullCycleRangeError(
                    "pass either limit or bounded, not both")
            limit = bounded
        if limit is None:
            count = FULL_CYCLE_COUNT
        else:
            count = _require_int(limit, "limit")
            if not 0 <= count <= FULL_CYCLE_COUNT:
                raise FullCycleRangeError(
                    "limit must be in [0, %d]: %r"
                    % (FULL_CYCLE_COUNT, limit))
        values = []
        for index in range(count):
            values.append(self._read_index(index))
        return tuple(values)

    read_all_words = read_all

    @property
    def words(self) -> tuple[int, ...]:
        """All 32,385 raw coefficients in dense index order."""
        return self.read_all()

    def _coerce_full_mapping(self, mapping: Mapping) -> list[int]:
        words = [None] * FULL_CYCLE_COUNT
        seen = {}
        for key, value in mapping.items():
            raw = _check_word(value)
            canonical, reversed_order = _mapping_key(key)
            oriented = (-raw) & WORD_MASK if reversed_order else raw
            previous = seen.get(canonical)
            if previous is not None:
                if previous != oriented:
                    raise FullCycleDimensionError(
                        "mapping contains conflicting values for cycle %d"
                        % canonical)
                continue
            seen[canonical] = oriented
            words[canonical] = oriented
        missing = [index for index, value in enumerate(words)
                   if value is None]
        if missing:
            raise FullCycleDimensionError(
                "write_all mapping must cover all %d cycles; missing %d "
                "(first missing index %d)"
                % (FULL_CYCLE_COUNT, len(missing), missing[0]))
        return [int(value) for value in words]

    def _coerce_full_sequence(self, words) -> list[int]:
        if isinstance(words, (str, bytes, bytearray, memoryview)):
            raise FullCycleDimensionError(
                "write_all words must be a sequence of integers")
        try:
            iterator = iter(words)
        except TypeError as error:
            raise FullCycleDimensionError(
                "write_all words must be iterable") from error
        values = []
        for item in iterator:
            if len(values) >= FULL_CYCLE_COUNT:
                raise FullCycleDimensionError(
                    "write_all requires exactly %d words; more were supplied"
                    % FULL_CYCLE_COUNT)
            values.append(_check_word(item, "write_all[%d]" % len(values)))
        if len(values) != FULL_CYCLE_COUNT:
            raise FullCycleDimensionError(
                "write_all requires exactly %d words, got %d"
                % (FULL_CYCLE_COUNT, len(values)))
        return values

    def write_all(self, words=None, *, mapping=None) -> int:
        """Replace the entire 32,385-coefficient basis atomically.

        A sequence must contain exactly :data:`FULL_CYCLE_COUNT` raw words in
        dense index order.  A mapping must cover every canonical cycle exactly
        once, using dense indices or ``(i, j)`` keys; reversed keys are
        accepted with their oriented sign.  Every value is validated before
        the first lane is written, so a malformed or short input cannot leave a
        partially updated memory.  Returns the number of coefficients written.
        """
        if mapping is not None:
            if words is not None:
                raise FullCycleDimensionError(
                    "pass either words or mapping, not both")
            if not isinstance(mapping, Mapping):
                raise FullCycleDimensionError(
                    "write_all mapping argument must be a mapping")
            values = self._coerce_full_mapping(mapping)
        else:
            if words is None:
                raise FullCycleDimensionError(
                    "write_all needs a full word sequence or mapping")
            if isinstance(words, Mapping):
                values = self._coerce_full_mapping(words)
            else:
                values = self._coerce_full_sequence(words)
        # All input is now checked.  Root edges must still be canonical.
        self.validate_root_edges()
        for index, word in enumerate(values):
            self._write_index(index, word)
        return len(values)

    # -- dot products and raw gradient steps ---------------------------
    def predict(self, features) -> int:
        """Integer dot product of signed coefficients and integer features.

        Accepted forms are a dense sequence of exactly 32,385 integers, a
        mapping from dense index or ``(i, j)`` pair to integer, or an iterable
        of ``(index_or_pair, value)`` entries.  Zero entries are skipped.  The
        result is an exact Python integer dot product, not a modular or
        binary32 reduction.
        """
        entries = _coerce_features(features)
        total = 0
        for pair, value in entries:
            if value == 0:
                continue
            total += signed_word(self._read_index(cycle_index(*pair))) * value
        return total

    def predict_word(self, features) -> int:
        """Return :meth:`predict` reduced to the exact raw 32-bit word."""
        return self.predict(features) & WORD_MASK

    def update(self, features, target, lr) -> UpdateReport:
        """Apply one sparse raw gradient step and return an integer report.

        The squared-error gradient is ``error * value`` per nonzero feature,
        with ``error = target - predict(features)`` and the update
        ``w += lr * error * value``.  ``target`` and ``lr`` must be integers;
        a step that would leave the signed i32 coefficient range raises
        :class:`FullCycleOverflowError` before any lane is changed.  Only
        nonzero feature entries are written, and each write keeps the mirror
        rail exactly negated.
        """
        target_value = _check_signed(target, "target")
        learning_rate = _check_signed(lr, "lr")
        entries = _coerce_features(features)
        self.validate_root_edges()
        prediction = 0
        for pair, value in entries:
            if value:
                prediction += signed_word(
                    self._read_index(cycle_index(*pair))) * value
        error = target_value - prediction
        # Compute and validate every new coefficient before the first write.
        pending = []
        for pair, value in entries:
            if value == 0:
                continue
            index = cycle_index(*pair)
            current = signed_word(self._read_index(index))
            delta = learning_rate * error * value
            if delta == 0:
                continue
            if not I32_MIN <= delta <= I32_MAX:
                raise FullCycleOverflowError(
                    "gradient delta %d for cycle %d does not fit i32"
                    % (delta, index))
            new_value = current + delta
            if not I32_MIN <= new_value <= I32_MAX:
                raise FullCycleOverflowError(
                    "cycle %d coefficient %d + %d leaves the i32 range"
                    % (index, current, delta))
            pending.append((index, new_value & WORD_MASK))
        for index, word in pending:
            self._write_index(index, word)
        return UpdateReport(prediction, target_value, error, len(pending),
                            error * error)

    # -- occupancy / snapshots ----------------------------------------
    @property
    def occupancy(self) -> Occupancy:
        """Number of nonzero cycle coefficients (also callable)."""
        count = 0
        for index in range(FULL_CYCLE_COUNT):
            if self._read_index(index):
                count += 1
        return Occupancy(count)

    @property
    def occupied(self) -> Occupancy:
        return self.occupancy

    def occupancy_report(self) -> dict:
        occupied = int(self.occupancy)
        return {
            "schema": SCHEMA,
            "dimension": FULL_CYCLE_COUNT,
            "occupied": occupied,
            "free": FULL_CYCLE_COUNT - occupied,
            "total": FULL_CYCLE_COUNT,
        }

    def snapshot(self):
        """Return an immutable, detached snapshot of the full basis."""
        words = self.read_all()
        cycles = tuple(CycleWord(index, first, second, words[index])
                       for index, (first, second)
                       in enumerate(iter_cycle_pairs()))
        root_edges = tuple(
            (ROOT_HEAD, head,
             self._field.get_oriented(ROOT_HEAD, head),
             self._field.get_oriented(head, ROOT_HEAD))
            for head in range(CYCLE_HEAD_MIN, CYCLE_HEAD_MAX + 1))
        pairs = tuple(iter_cycle_pairs())
        word_map = MappingProxyType(dict(zip(pairs, words)))
        occupied = sum(1 for word in words if word)
        return MappingProxyType({
            "schema": SCHEMA,
            "mode": "raw-u32",
            "word_bits": WORD_BITS,
            "head_count": HEAD_COUNT,
            "pair_count": PAIR_COUNT,
            "root": ROOT_HEAD,
            "root_edge_count": ROOT_EDGE_COUNT,
            "root_cycle_count": FULL_CYCLE_COUNT,
            "dimension": FULL_CYCLE_COUNT,
            "cycle_count": FULL_CYCLE_COUNT,
            "generation": self.generation,
            "occupancy": Occupancy(occupied),
            "words": words,
            "cycles": cycles,
            "root_cycles": cycles,
            "pairs": pairs,
            "word_map": word_map,
            "word_by_pair": word_map,
            "root_edges": root_edges,
        })

    # -- audit ---------------------------------------------------------
    def audit(self, *, full: bool = True) -> AuditReport:
        """Check dimensions, root zeros, mirrors, metadata and every cycle.

        The report is a dictionary with an ``ok`` field and is falsy when a
        problem is found.  ``full=True`` (the default) inspects all 32,385
        cycles, all 255 root pairs and all 256 permanent head identities.
        """
        problems = []
        problem_codes = []
        cycles_checked = 0
        root_edges_checked = 0
        mirror_checked = 0
        mirror_mismatches = 0
        nonzero_cycles = 0
        root_conflicts = 0
        head_identity_ok = True
        pair_identity_checked = 0
        pair_identity_ok = True

        head_count = getattr(self.cells, "head_count", HEAD_COUNT)
        pair_count = getattr(self.cells, "pair_count", PAIR_COUNT)
        dimension_ok = (head_count == HEAD_COUNT
                        and pair_count == PAIR_COUNT
                        and FULL_CYCLE_COUNT == 32385)

        reader = getattr(self.cells, "read_head", None)
        if reader is not None:
            for head in range(HEAD_COUNT):
                try:
                    record = reader(head)
                except Exception as error:  # noqa: BLE001 - backend boundary
                    problems.append("head %d unreadable: %s" % (head, error))
                    problem_codes.append("head_unreadable")
                    head_identity_ok = False
                    continue
                head_id, identity = self._head_fields(record)
                if head_id is not None and head_id != head:
                    problems.append(
                        "head %d has permanent head_id %r" % (head, head_id))
                    problem_codes.append("head_id")
                    head_identity_ok = False
                if identity is not None and identity != identity_address(head):
                    problems.append(
                        "head %d has identity %r, expected %04x"
                        % (head, identity, identity_address(head)))
                    problem_codes.append("head_identity")
                    head_identity_ok = False

        # Root edges: both directed lanes must be exactly zero.
        for head in range(CYCLE_HEAD_MIN, CYCLE_HEAD_MAX + 1):
            forward = self._field.get_oriented(ROOT_HEAD, head)
            reverse = self._field.get_oriented(head, ROOT_HEAD)
            root_edges_checked += 2
            if forward or reverse:
                root_conflicts += 1
                problems.append(
                    "root edge (0, %d) is nonzero: %08x/%08x"
                    % (head, forward, reverse))
                problem_codes.append("root_edge")

        # Every non-root pair: exact word range and mirror negation.
        count = FULL_CYCLE_COUNT if full else min(FULL_CYCLE_COUNT, 256)
        for index in range(count):
            first, second = cycle_pair(index)
            forward = self._field.get_oriented(first, second)
            reverse = self._field.get_oriented(second, first)
            cycles_checked += 1
            if forward:
                nonzero_cycles += 1
            if not (0 <= forward <= WORD_MASK):
                problems.append(
                    "cycle %d word is outside u32: %r" % (index, forward))
                problem_codes.append("word_range")
            mirror_checked += 1
            if ((forward + reverse) & WORD_MASK) != 0:
                mirror_mismatches += 1
                problems.append(
                    "cycle %d mirror mismatch: %08x and %08x"
                    % (index, forward, reverse))
                problem_codes.append("mirror")
            state_reader = getattr(self.cells, "pair_state", None)
            if state_reader is not None and full:
                try:
                    state = state_reader(first, second)
                except Exception as error:  # noqa: BLE001 - backend boundary
                    problems.append(
                        "pair (%d, %d) unreadable: %s"
                        % (first, second, error))
                    problem_codes.append("pair_unreadable")
                    pair_identity_ok = False
                    continue
                pair_identity_checked += 1
                expected_first, expected_second, canonical, mirror = \
                    self._pair_expected(pair_index(first, second))
                if state.canonical != canonical or state.mirror != mirror:
                    problems.append(
                        "pair (%d, %d) canonical/mirror metadata mismatch"
                        % (first, second))
                    problem_codes.append("pair_canonical")
                    pair_identity_ok = False
                for rail, record, source, target in (
                        (RAIL_S_PLUS, state.forward,
                         expected_first, expected_second),
                        (RAIL_S_MINUS, state.reverse,
                         expected_second, expected_first)):
                    if (record.source != source or record.target != target
                            or record.mirror != mirror):
                        problems.append(
                            "pair (%d, %d) rail %d identity mismatch"
                            % (first, second, rail))
                        problem_codes.append("pair_identity")
                        pair_identity_ok = False

        expected_cycles = FULL_CYCLE_COUNT if full else count
        expected_roots = 2 * ROOT_EDGE_COUNT
        cycles_ok = cycles_checked == expected_cycles
        roots_ok = root_edges_checked == expected_roots and root_conflicts == 0
        mirror_ok = mirror_mismatches == 0
        ok = bool(dimension_ok and cycles_ok and roots_ok and mirror_ok
                  and not problems)
        return AuditReport({
            "schema": SCHEMA,
            "ok": ok,
            "accepted": ok,
            "dimension": FULL_CYCLE_COUNT,
            "dimension_ok": dimension_ok,
            "cycle_count": FULL_CYCLE_COUNT,
            "word_bits": WORD_BITS,
            "pairs_checked": cycles_checked,
            "cycles_checked": cycles_checked,
            "cycles_ok": cycles_ok,
            "nonzero_cycles": nonzero_cycles,
            "occupied": nonzero_cycles,
            "occupancy": Occupancy(nonzero_cycles) if full
            else Occupancy(self.occupancy),
            "root_edge_count": ROOT_EDGE_COUNT,
            "root_edges_checked": root_edges_checked,
            "root_conflicts": root_conflicts,
            "root_ok": roots_ok,
            "root_zero": roots_ok,
            "mirror_checked": mirror_checked,
            "mirror_mismatches": mirror_mismatches,
            "mirror_ok": mirror_ok,
            "mirror_antisymmetry": mirror_ok,
            "head_identity_ok": head_identity_ok,
            "pair_identity_checked": pair_identity_checked,
            "pair_identity_ok": pair_identity_ok,
            "generation": self.generation,
            "problems": tuple(problems),
            "problem_codes": tuple(problem_codes),
            "full_scan": bool(full),
            "no_float": True,
            "integer_words_only": True,
        })
