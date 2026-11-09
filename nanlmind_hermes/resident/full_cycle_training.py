"""Fixed-point supervised training of the complete 32,385-cycle basis.

Round 031 exposed only 256 effective triangle weights even though the complete
graph on 256 heads has ``C(255, 2) = 32,385`` independent fundamental cycles
relative to the star root.  This module removes that accidental 256-cycle
window.  A :class:`FullCycleTrainer` always allocates, validates and addresses
the complete coefficient vector; sparse feature maps merely select which of the
32,385 coefficients receive a non-zero gradient on a given step.

Raw arithmetic
--------------

Coefficients, features, targets, gradients and update deltas are Python
integers in Q16.16 units.  Fractions are accepted at the boundary and rounded
once to Q16.16; Python ``float`` values are refused for every value that can
reach a cycle word.  Reporting metrics such as :meth:`FullCycleTrainer.loss`
and receipt ``loss_before``/``loss_after`` fields may be floats, but they are
derived from the exact integer result and are never written back.

Memory protocol
---------------

``FullCycleTrainer(memory)`` accepts any object with this small protocol::

    memory.read_cycle(index) -> int              # signed Q16.16 word
    memory.commit_cycles(updates) -> generation  # atomic (index, value) batch

``read_word``/``get_cycle``/``read_coefficient`` and
``write_cycles``/``commit_weights`` are accepted as spellings.  A memory that
only offers ``set_cycle(index, value)`` is still usable through a snapshot and
readback guard.  Optional ``generation``, ``snapshot``/``restore`` and
``cycle_features``/``cycle_readout`` hooks are used when present.
:class:`FullCycleMemory` is the deterministic in-memory reference store.

Gauge invariance
----------------

The reference store also owns raw pair lanes.  A root-star-anchored vertex
gauge ``x(i, j) -> x(i, j) + g[i] - g[j]`` leaves every fundamental cycle
``x(root, i) + x(i, j) - x(root, j)`` unchanged exactly.  Applying that gauge
therefore leaves the cycle feature vector and the trained coefficient vector
unchanged; only the pair lanes move.

No canonical merge, semantic-truth claim, language-model claim or quantum claim
is made here.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from array import array
from fractions import Fraction
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from .cells import FIXED_ONE, HEAD_COUNT
from .cycle_observable import DEFAULT_ROOT, WORD_MASK, WORD_MODULUS

# ---------------------------------------------------------------------------
# dimensions and fixed-point bounds
# ---------------------------------------------------------------------------
FULL_CYCLE_COUNT = (HEAD_COUNT - 1) * (HEAD_COUNT - 2) // 2  # 32,385
FULL_CYCLE_DIMENSION = FULL_CYCLE_COUNT
CYCLE_SPACE_DIMENSION = FULL_CYCLE_COUNT
CYCLE_COUNT = FULL_CYCLE_COUNT
ROOT_CYCLE_COUNT = FULL_CYCLE_COUNT
ROOT_HEAD = DEFAULT_ROOT

WORD_BITS = 32
HALF_FIXED = FIXED_ONE // 2
_I32_MAX = (1 << 31) - 1
PAYLOAD_MIN = -_I32_MAX
PAYLOAD_MAX = _I32_MAX

DEFAULT_LEARNING_RATE_UNITS = FIXED_ONE // 10
DEFAULT_MAX_LEARNING_RATE_UNITS = 16 * FIXED_ONE
DEFAULT_MAX_STEP_UNITS = 1 << 26
DEFAULT_MAX_STEPS = 1_000_000
DEFAULT_MAX_STEPS_PER_CALL = 1024
DEFAULT_MAX_BATCH_SIZE = 1024

FULL_CYCLE_TRAINING_SCHEMA = "nanlm.resident-full-cycle-training/1"
FULL_CYCLE_STEP_SCHEMA = "nanlm.resident-full-cycle-step/1"
FULL_CYCLE_MEMORY_SCHEMA = "nanlm.resident-full-cycle-memory/1"
FULL_CYCLE_GAUGE_SCHEMA = "nanlm.resident-full-cycle-gauge/1"

# Explicit declarations: training is evidence, not canonical truth.
CANONICAL_MERGE = False
FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH = False


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class FullCycleTrainingError(ValueError):
    """Base class for malformed full-cycle observations and refused steps."""


class FullCycleValidationError(FullCycleTrainingError):
    """Raised when a feature, target, option or memory handle is malformed."""


class FullCycleDimensionError(FullCycleValidationError):
    """Raised when an input does not cover the complete 32,385 dimensions."""


class FullCycleTruncationError(FullCycleDimensionError):
    """Raised when a memory or vector is a 256-style truncated window."""


class FullCycleOverflowError(FullCycleTrainingError):
    """Raised instead of saturating an out-of-range word or step."""


class FullCycleStepRejected(FullCycleTrainingError):
    """Raised when a bounded step would not improve the measured loss."""


class FullCycleFieldError(FullCycleTrainingError):
    """Raised when a commit fails or readback disagrees with the proposal."""


class FullCycleGaugeError(FullCycleTrainingError):
    """Raised when a gauge operation would change a cycle or a weight."""


class FullCycleLimitError(FullCycleTrainingError):
    """Raised when a learning-rate, step or batch bound is exceeded."""


# ---------------------------------------------------------------------------
# integer and cycle-index helpers
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleValidationError("%s must be an integer: %r"
                                       % (name, value))
    return value


def _fixed_units(value: Any, name: str) -> int:
    """Return an exact Q16.16 integer; floats are deliberately refused."""
    if isinstance(value, bool) or isinstance(value, float):
        raise FullCycleValidationError(
            "%s must be an integer in Q16.16 units or a Fraction; floats are "
            "not accepted: %r" % (name, value))
    if isinstance(value, Fraction):
        return int(round(value * FIXED_ONE))
    if not isinstance(value, int):
        raise FullCycleValidationError(
            "%s must be an integer in Q16.16 units or a Fraction: %r"
            % (name, value))
    return value


def _check_payload(value: Any, name: str) -> int:
    value = _require_int(value, name)
    if not PAYLOAD_MIN <= value <= PAYLOAD_MAX:
        raise FullCycleOverflowError(
            "%s %d is outside [%d, %d]"
            % (name, value, PAYLOAD_MIN, PAYLOAD_MAX))
    return value


def _check_head(value: Any, name: str = "head") -> int:
    value = _require_int(value, name)
    if not 0 <= value < HEAD_COUNT:
        raise FullCycleValidationError(
            "%s must be in [0, %d): %r" % (name, HEAD_COUNT, value))
    return value


def _round_div(numerator: int, denominator: int) -> int:
    """Round an exact integer quotient, half away from zero."""
    if denominator <= 0:
        raise FullCycleValidationError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def cycle_index(first: Any, second: Any) -> int:
    """Return the dense index of fundamental cycle ``(first, second)``.

    The star root is head 0, so fundamental cycle heads live in
    ``1 .. HEAD_COUNT-1`` and are ordered lexicographically.
    """
    left = _check_head(first, "first")
    right = _check_head(second, "second")
    if left == right:
        raise FullCycleValidationError(
            "a fundamental cycle needs two distinct heads: %r" % (left,))
    if left == 0 or right == 0:
        raise FullCycleValidationError(
            "root edges are gauge coordinates, not fundamental-cycle heads")
    if left > right:
        left, right = right, left
    return ((left - 1) * (HEAD_COUNT - 1)
            - (left - 1) * left // 2 + (right - left - 1))


def cycle_pair(index: Any) -> tuple[int, int]:
    """Inverse of :func:`cycle_index` for ``0 <= index < 32,385``."""
    index = _require_int(index, "cycle index")
    if not 0 <= index < FULL_CYCLE_COUNT:
        raise FullCycleDimensionError(
            "cycle index must be in [0, %d): %r"
            % (FULL_CYCLE_COUNT, index))
    remaining = index
    for first in range(1, HEAD_COUNT):
        width = HEAD_COUNT - 1 - first
        if remaining < width:
            return first, first + 1 + remaining
        remaining -= width
    raise AssertionError("cycle index inverse fell outside the basis")


CYCLE_BASIS = tuple(cycle_pair(index) for index in range(FULL_CYCLE_COUNT))
CYCLE_PAIRS = CYCLE_BASIS


def _coerce_cycle_key(value: Any, name: str = "cycle key") -> int:
    """Accept a dense index, an ``(i, j)`` pair or ``"i,j"`` text."""
    if isinstance(value, bool):
        raise FullCycleValidationError("%s must not be bool" % name)
    if isinstance(value, int):
        cycle_pair(value)  # validate the dense index range
        return value
    if isinstance(value, str):
        text = value.strip()
        for separator in (",", ":", "-", "/"):
            if separator in text:
                pieces = text.split(separator)
                break
        else:
            pieces = [text]
        if len(pieces) == 2:
            try:
                return cycle_index(int(pieces[0]), int(pieces[1]))
            except (TypeError, ValueError) as error:
                raise FullCycleValidationError(
                    "%s is not a cycle pair: %r" % (name, value)) from error
        raise FullCycleValidationError(
            "%s must be an index or an (i, j) pair: %r" % (name, value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return cycle_index(value[0], value[1])
        except FullCycleTrainingError:
            raise
        except Exception as error:  # noqa: BLE001 - boundary normalization
            raise FullCycleValidationError(
                "%s is not a cycle pair: %r" % (name, value)) from error
    raise FullCycleValidationError(
        "%s must be an index or an (i, j) pair: %r" % (name, value))


def _to_word(value: int) -> int:
    """Read a signed i32 lane as its exact unsigned 32-bit word."""
    return int(value) & WORD_MASK


def _from_word(value: int) -> int:
    """Interpret a raw unsigned 32-bit word as a signed Q16.16 payload."""
    word = _require_int(value, "raw word")
    if not 0 <= word <= WORD_MASK:
        raise FullCycleOverflowError(
            "raw word must be an unsigned 32-bit integer: %r" % (value,))
    return word if word < (1 << 31) else word - WORD_MODULUS


def _normalize_payload(value: Any, name: str = "cycle word") -> int:
    """Accept a signed payload or an unsigned raw word without conversion."""
    value = _require_int(value, name)
    if PAYLOAD_MIN <= value <= PAYLOAD_MAX:
        return value
    if 0 <= value <= WORD_MASK:
        signed = _from_word(value)
        if PAYLOAD_MIN <= signed <= PAYLOAD_MAX:
            return signed
    raise FullCycleOverflowError(
        "%s %d is outside the signed payload range or a raw 32-bit word"
        % (name, value))


def _digest_words(words: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(words), 4096):
        chunk = words[start:start + 4096]
        digest.update(struct.pack("<%dI" % len(chunk), *chunk))
    return digest.hexdigest()


def _coerce_gauge(gauge: Any) -> tuple[int, ...]:
    if isinstance(gauge, (str, bytes, bytearray)):
        raise FullCycleValidationError("gauge must be a sequence of integers")
    try:
        iterator = iter(gauge)
    except TypeError as error:
        raise FullCycleValidationError(
            "gauge must be an iterable of %d integers" % HEAD_COUNT) from error
    values = []
    for index in range(HEAD_COUNT):
        try:
            value = next(iterator)
        except StopIteration as error:
            raise FullCycleValidationError(
                "gauge must contain exactly %d integers" % HEAD_COUNT) from error
        values.append(_fixed_units(value, "gauge[%d]" % index))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise FullCycleValidationError(
            "gauge must contain exactly %d integers" % HEAD_COUNT)
    return tuple(values)


def deterministic_batches(samples: Iterable[Any], batch_size: int = 1, *,
                          seed: int = 0, shuffle: bool = True
                          ) -> tuple[tuple[Any, ...], ...]:
    """Split samples into reproducible batches.

    The same ``(samples, batch_size, seed, shuffle)`` always yields the same
    batch order.  No floating-point or process-global RNG state is used.
    """
    items = list(samples)
    batch_size = _require_int(batch_size, "batch_size")
    if batch_size <= 0:
        raise FullCycleValidationError("batch_size must be positive")
    seed = _require_int(seed, "seed")
    if not isinstance(shuffle, bool):
        raise FullCycleValidationError("shuffle must be bool")
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    ordered = [items[index] for index in order]
    return tuple(
        tuple(ordered[start:start + batch_size])
        for start in range(0, len(ordered), batch_size)
    )


# ---------------------------------------------------------------------------
# reference raw-word memory
# ---------------------------------------------------------------------------
class FullCycleMemory:
    """Deterministic in-memory store for all 32,385 unsigned cycle words.

    ``weights`` are exposed as signed Q16.16 integers while the underlying
    ``array('i')`` and :meth:`read_word` preserve the exact raw 32-bit word.
    Neither a sparse commit nor a single write ever resizes or truncates the
    vector.
    """

    schema = FULL_CYCLE_MEMORY_SCHEMA
    dimension = FULL_CYCLE_COUNT
    cycle_count = FULL_CYCLE_COUNT
    root_cycle_count = FULL_CYCLE_COUNT
    word_count = FULL_CYCLE_COUNT
    byte_count = FULL_CYCLE_COUNT * 4
    size = FULL_CYCLE_COUNT

    def __init__(self, weights: Any = None, *,
                 payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX):
        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        if not PAYLOAD_MIN <= payload_min < payload_max <= PAYLOAD_MAX:
            raise FullCycleValidationError(
                "payload bounds must be inside symmetric int32")
        self.payload_min = payload_min
        self.payload_max = payload_max
        self._coefficients = array("i", [0]) * FULL_CYCLE_COUNT
        self._lanes: dict[tuple[int, int], int] = {}
        self._generation = 0
        if weights is not None:
            values = self._coerce_full_weights(weights)
            for index, value in enumerate(values):
                self._coefficients[index] = value

    # -- dimension -------------------------------------------------------
    def _coerce_full_weights(self, weights: Any) -> tuple[int, ...]:
        if isinstance(weights, (str, bytes, bytearray)):
            raise FullCycleDimensionError("weights must be a sequence")
        try:
            values = tuple(weights)
        except TypeError as error:
            raise FullCycleDimensionError(
                "weights must be an iterable of integers") from error
        if len(values) != FULL_CYCLE_COUNT:
            raise FullCycleTruncationError(
                "weights must contain all %d cycle words, got %d; a 256-word "
                "window is refused" % (FULL_CYCLE_COUNT, len(values)))
        cleaned = []
        for index, value in enumerate(values):
            value = _normalize_payload(value, "weights[%d]" % index)
            if not self.payload_min <= value <= self.payload_max:
                raise FullCycleOverflowError(
                    "weights[%d]=%d is outside [%d, %d]"
                    % (index, value, self.payload_min, self.payload_max))
            cleaned.append(value)
        return tuple(cleaned)

    def __len__(self) -> int:
        return FULL_CYCLE_COUNT

    def __getitem__(self, index: int) -> int:
        return self.read_cycle(index)

    def __setitem__(self, index: int, value: int) -> None:
        self.commit_cycles(((index, value),))

    @property
    def generation(self) -> int:
        return self._generation

    # -- coefficient words ----------------------------------------------
    def read_cycle(self, index: int) -> int:
        index = _require_int(index, "cycle index")
        if not 0 <= index < FULL_CYCLE_COUNT:
            raise FullCycleDimensionError(
                "cycle index must be in [0, %d): %r"
                % (FULL_CYCLE_COUNT, index))
        value = int(self._coefficients[index])
        if not self.payload_min <= value <= self.payload_max:
            raise FullCycleOverflowError(
                "stored cycle word %d=%d is outside [%d, %d]"
                % (index, value, self.payload_min, self.payload_max))
        return value

    def read_word(self, index: int) -> int:
        return _to_word(self.read_cycle(index))

    def read_all(self) -> tuple[int, ...]:
        return tuple(self._coefficients)

    @property
    def weights(self) -> tuple[int, ...]:
        return tuple(self._coefficients)

    @property
    def coefficients(self) -> tuple[int, ...]:
        return tuple(self._coefficients)

    @property
    def raw_words(self) -> tuple[int, ...]:
        return tuple(_to_word(value) for value in self._coefficients)

    def commit_cycles(self, updates: Iterable[Any]) -> int:
        normalized = self._normalize_cycle_updates(updates)
        previous: dict[int, int] = {}
        try:
            for index, value in normalized:
                previous[index] = int(self._coefficients[index])
                self._coefficients[index] = value
        except Exception:
            for index, value in previous.items():
                self._coefficients[index] = value
            raise
        self._generation += 1
        return self._generation

    def write_cycles(self, updates: Iterable[Any]) -> int:
        return self.commit_cycles(updates)

    def commit_words(self, updates: Iterable[Any]) -> int:
        signed = []
        for index, word in updates:
            signed.append((index, _from_word(word)))
        return self.commit_cycles(signed)

    def _normalize_cycle_updates(
            self, updates: Any) -> tuple[tuple[int, int], ...]:
        if isinstance(updates, Mapping):
            items = updates.items()
        else:
            try:
                items = iter(updates)
            except TypeError as error:
                raise FullCycleValidationError(
                    "updates must be an iterable of (index, value)") from error
        seen: dict[int, int] = {}
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise FullCycleValidationError(
                    "each update must be (index, value): %r" % (item,))
            index = _coerce_cycle_key(item[0])
            value = _normalize_payload(item[1])
            if not self.payload_min <= value <= self.payload_max:
                raise FullCycleOverflowError(
                    "cycle word %d=%d is outside [%d, %d]"
                    % (index, value, self.payload_min, self.payload_max))
            if index in seen:
                raise FullCycleValidationError(
                    "duplicate cycle update: %r" % (index,))
            seen[index] = value
        return tuple((index, seen[index]) for index in sorted(seen))

    # -- raw pair lanes (gauge reference) -------------------------------
    def _check_pair(self, source: int, target: int) -> tuple[int, int]:
        first = _check_head(source, "source")
        second = _check_head(target, "target")
        if first == second:
            raise FullCycleValidationError(
                "a pair needs two distinct heads: %r" % (first,))
        return first, second

    def read_lane(self, source: int, target: int) -> int:
        first, second = self._check_pair(source, target)
        if first < second:
            value = self._lanes.get((first, second), 0)
        else:
            value = -self._lanes.get((second, first), 0)
        return _check_payload(value, "lane")

    def commit_pairs(self, updates: Iterable[Any]) -> int:
        normalized = self._normalize_pair_updates(updates)
        previous: dict[tuple[int, int], int] = {}
        try:
            for first, second, value in normalized:
                previous[(first, second)] = self._lanes.get((first, second), 0)
                self._lanes[(first, second)] = value
        except Exception:
            for pair, value in previous.items():
                self._lanes[pair] = value
            raise
        self._generation += 1
        return self._generation

    def write_pairs(self, updates: Iterable[Any]) -> int:
        return self.commit_pairs(updates)

    def _normalize_pair_updates(
            self, updates: Any) -> tuple[tuple[int, int, int], ...]:
        if isinstance(updates, Mapping):
            items = updates.items()
        else:
            try:
                items = iter(updates)
            except TypeError as error:
                raise FullCycleValidationError(
                    "pair updates must be an iterable") from error
        seen: dict[tuple[int, int], int] = {}
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise FullCycleValidationError(
                    "each pair update must be (left, right, value)")
            first, second = self._check_pair(item[0], item[1])
            value = _check_payload(item[2], "lane value")
            if first > second:
                first, second = second, first
                value = -value
            _check_payload(-value, "mirror lane value")
            if (first, second) in seen:
                raise FullCycleValidationError(
                    "duplicate canonical pair update: %r"
                    % ((first, second),))
            seen[(first, second)] = value
        return tuple((first, second, seen[(first, second)])
                     for first, second in sorted(seen))

    def cycle_readout(self, first: Any, second: Any = None,
                      root: int = ROOT_HEAD) -> int:
        """Return ``x(root, i) + x(i, j) - x(root, j)`` exactly."""
        if second is None:
            if not isinstance(first, int):
                raise FullCycleValidationError(
                    "pass a cycle index or two heads")
            index = _require_int(first, "cycle index")
            if not 0 <= index < FULL_CYCLE_COUNT:
                raise FullCycleDimensionError(
                    "cycle index must be in [0, %d): %r"
                    % (FULL_CYCLE_COUNT, index))
            left, right = cycle_pair(index)
        else:
            left, right = self._check_pair(first, second)
        root = _check_head(root, "root")
        if root in (left, right):
            raise FullCycleValidationError(
                "root must differ from both cycle heads")
        return (self.read_lane(root, left) + self.read_lane(left, right)
                - self.read_lane(root, right))

    def cycle_features(self, root: int = ROOT_HEAD) -> tuple[int, ...]:
        """Return all 32,385 fundamental-cycle readouts for ``root``."""
        root = _check_head(root, "root")
        values = []
        for first in range(HEAD_COUNT):
            if first == root:
                continue
            for second in range(first + 1, HEAD_COUNT):
                if second == root:
                    continue
                values.append(self.cycle_readout(first, second, root=root))
        if len(values) != FULL_CYCLE_COUNT:
            raise AssertionError(
                "cycle feature vector has %d entries, expected %d"
                % (len(values), FULL_CYCLE_COUNT))
        return tuple(values)

    def apply_root_star_gauge(self, gauge: Any,
                              *, root: int = ROOT_HEAD) -> int:
        """Apply the root-star-anchored vertex gauge to every pair lane.

        The 256 gauge coordinates are Q16.16 integers.  Every fundamental
        cycle is exactly invariant because both root edges and the non-root
        edge receive ``g[i] - g[j]``.  The coefficient vector is untouched.
        """
        values = _coerce_gauge(gauge)
        root = _check_head(root, "root")
        updates = []
        for first in range(HEAD_COUNT):
            for second in range(first + 1, HEAD_COUNT):
                value = (self._lanes.get((first, second), 0)
                         + values[first] - values[second])
                updates.append((first, second, value))
        return self.commit_pairs(updates)

    def apply_gauge(self, gauge: Any, *, root: int = ROOT_HEAD) -> int:
        return self.apply_root_star_gauge(gauge, root=root)

    def apply_root_edge_gauge(self, gauge: Any,
                              *, root: int = ROOT_HEAD) -> int:
        """Shift only root lanes; kept as a non-invariant control."""
        values = _coerce_gauge(gauge)
        root = _check_head(root, "root")
        updates = []
        for head in range(HEAD_COUNT):
            if head == root:
                continue
            first, second = (root, head) if root < head else (head, root)
            current = self._lanes.get((first, second), 0)
            delta = values[root] - values[head]
            value = current + (delta if first == root else -delta)
            updates.append((first, second, value))
        return self.commit_pairs(updates)

    # -- snapshots and identity -----------------------------------------
    def snapshot(self) -> tuple[tuple[int, ...], dict[tuple[int, int], int], int]:
        return (tuple(self._coefficients), dict(self._lanes),
                self._generation)

    def restore(self, snapshot: Any) -> None:
        if not isinstance(snapshot, (tuple, list)) or len(snapshot) != 3:
            raise FullCycleValidationError(
                "snapshot must be (weights, lanes, generation)")
        weights, lanes, generation = snapshot
        values = self._coerce_full_weights(weights)
        if not isinstance(lanes, Mapping):
            raise FullCycleValidationError("snapshot lanes must be a mapping")
        cleaned_lanes: dict[tuple[int, int], int] = {}
        for pair, value in lanes.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise FullCycleValidationError(
                    "snapshot lane keys must be (left, right)")
            first, second = self._check_pair(pair[0], pair[1])
            if first > second:
                first, second = second, first
            cleaned_lanes[(first, second)] = _check_payload(
                value, "snapshot lane")
        self._coefficients = array("i", values)
        self._lanes = cleaned_lanes
        self._generation = _require_int(generation, "snapshot generation")
        if self._generation < 0:
            raise FullCycleValidationError(
                "snapshot generation must be non-negative")

    def digest(self) -> str:
        return _digest_words(tuple(_to_word(value)
                                   for value in self._coefficients))

    def nonzero_count(self) -> int:
        return sum(1 for value in self._coefficients if value)

    def validate_full_basis(self) -> dict[str, Any]:
        for index in range(FULL_CYCLE_COUNT):
            self.read_cycle(index)
        return {
            "schema": FULL_CYCLE_MEMORY_SCHEMA,
            "dimension": FULL_CYCLE_COUNT,
            "readable": FULL_CYCLE_COUNT,
            "full_basis": True,
            "truncated": False,
        }

    def set_cycle(self, index: int, value: int) -> int:
        return self.commit_cycles(((index, value),))

    write_cycle = set_cycle
    set_cycle_weight = set_cycle
    write_cycle_weight = set_cycle

    read_cycle_weight = read_cycle
    get_cycle = read_cycle
    read_coefficient = read_cycle
    readout = cycle_readout
    features = cycle_features

    def __repr__(self) -> str:
        return ("FullCycleMemory(dimension=%d, generation=%d, nonzero=%d)"
                % (FULL_CYCLE_COUNT, self._generation,
                   self.nonzero_count()))


# Common spellings for the same reference store.
IntegerFullCycleMemory = FullCycleMemory
FullCycleStore = FullCycleMemory


# ---------------------------------------------------------------------------
# memory protocol resolution
# ---------------------------------------------------------------------------
_READER_NAMES = (
    "read_cycle", "read_cycle_weight", "read_cycle_coefficient",
    "get_cycle", "get_cycle_weight", "get_cycle_coefficient",
    "read_coefficient", "read_weight", "get_weight", "read_word", "get_word",
)
_WORD_READER_NAMES = ("read_word", "get_word")
_WORD_COMMITTER_NAMES = (
    "commit_words", "write_words", "commit_cycle_words",
    "write_cycle_words",
)
_COMMITTER_NAMES = (
    "commit_cycles", "write_cycles", "commit_cycle_updates", "update_cycles",
    "commit_weights", "write_weights", "commit_cycle_weights",
    "write_cycle_weights", "commit_words", "write_words",
    "commit_cycle_words", "write_cycle_words", "commit",
)
_WORD_WRITER_NAMES = ("write_word", "set_word")
_SINGLE_WRITER_NAMES = (
    "set_cycle", "write_cycle", "set_cycle_weight", "write_cycle_weight",
    "set_cycle_coefficient", "write_cycle_coefficient", "set_coefficient",
    "write_weight", "set_weight", "write_word", "set_word",
)
_COEFFICIENT_SEQUENCE_NAMES = (
    "weights", "cycle_weights", "cycle_coefficients", "coefficients",
    "raw_words",
)


def _memory_dimension(memory: Any) -> int:
    for name in ("cycle_count", "cycle_dimension", "dimension",
                 "root_cycle_count"):
        value = getattr(memory, name, None)
        if callable(value):
            value = value()
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value != FULL_CYCLE_COUNT:
            raise FullCycleTruncationError(
                "memory %s=%d cannot represent all %d cycles"
                % (name, value, FULL_CYCLE_COUNT))
    for name in _COEFFICIENT_SEQUENCE_NAMES:
        value = getattr(memory, name, None)
        if callable(value) or value is None:
            continue
        try:
            length = len(value)
        except TypeError:
            continue
        if length != FULL_CYCLE_COUNT:
            raise FullCycleTruncationError(
                "memory %s has %d words, expected all %d"
                % (name, length, FULL_CYCLE_COUNT))
    return FULL_CYCLE_COUNT


def _resolve_reader(memory: Any):
    for name in _READER_NAMES:
        method = getattr(memory, name, None)
        if callable(method):
            return method, name in _WORD_READER_NAMES
    for name in _COEFFICIENT_SEQUENCE_NAMES:
        sequence = getattr(memory, name, None)
        if callable(sequence) or sequence is None:
            continue
        if hasattr(sequence, "__getitem__"):
            def reader(index, _name=name):
                return getattr(memory, _name)[index]
            return reader, name == "raw_words"
    raise TypeError(
        "memory is missing read_cycle(index) or an indexable weights vector")


def _resolve_committer(memory: Any):
    for name in _COMMITTER_NAMES:
        method = getattr(memory, name, None)
        if callable(method):
            if name in _WORD_COMMITTER_NAMES:
                def word_committer(updates, _method=method):
                    return _method(tuple(
                        (index, _to_word(value))
                        for index, value in updates))
                return word_committer
            return method
    return None


def _resolve_single_writer(memory: Any):
    for name in _SINGLE_WRITER_NAMES:
        method = getattr(memory, name, None)
        if callable(method):
            if name in _WORD_WRITER_NAMES:
                def word_writer(index, value, _method=method):
                    return _method(index, _to_word(value))
                return word_writer
            return method
    for name in _COEFFICIENT_SEQUENCE_NAMES:
        sequence = getattr(memory, name, None)
        if callable(sequence) or sequence is None:
            continue
        if hasattr(sequence, "__setitem__"):
            return lambda index, value, _seq=sequence: _seq.__setitem__(
                index, value)
    setter = getattr(memory, "__setitem__", None)
    if callable(setter):
        return lambda index, value: setter(index, value)
    return None


def _memory_generation(memory: Any) -> int | None:
    value = getattr(memory, "generation", None)
    if value is None:
        return None
    if callable(value):
        value = value()
    value = _require_int(value, "memory generation")
    if value < 0:
        raise FullCycleValidationError(
            "memory generation must be non-negative: %r" % (value,))
    return value


def _memory_snapshot_hooks(memory: Any):
    snapshot = getattr(memory, "snapshot", None)
    restore = getattr(memory, "restore", None)
    if callable(snapshot) and callable(restore):
        return snapshot, restore
    return None


# ---------------------------------------------------------------------------
# prepared observations
# ---------------------------------------------------------------------------
class _PreparedSample(NamedTuple):
    mode: str
    features: Any
    feature_kind: str
    targets: Any
    target_kind: str
    active_features: tuple[int, ...]
    relevant_indices: tuple[int, ...]

    def feature(self, index: int) -> int:
        if self.feature_kind == "sparse":
            return self.features.get(index, 0)
        return self.features[index]

    def target(self, index: int) -> int:
        if self.target_kind == "sparse":
            return self.targets.get(index, 0)
        return self.targets[index]


def _sparse_pairs_to_mapping(items: Sequence[Any], label: str,
                             payload_min: int, payload_max: int
                             ) -> tuple[dict[int, int], tuple[int, ...]]:
    values: dict[int, int] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise FullCycleValidationError(
                "each sparse %s entry must be (cycle, value)" % label)
        index = _coerce_cycle_key(item[0])
        if index in values:
            raise FullCycleValidationError(
                "duplicate sparse %s index: %r" % (label, index))
        value = _fixed_units(item[1], "%s[%d]" % (label, index))
        if not payload_min <= value <= payload_max:
            raise FullCycleOverflowError(
                "%s[%d]=%d is outside [%d, %d]"
                % (label, index, value, payload_min, payload_max))
        values[index] = value
    active = tuple(sorted(index for index, value in values.items()
                          if value != 0))
    return values, active


def _parse_feature_vector(value: Any, payload_min: int,
                          payload_max: int) -> tuple[str, Any,
                                                    tuple[int, ...]]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
        features, active = _sparse_pairs_to_mapping(
            pairs, "feature", payload_min, payload_max)
        return "sparse", features, active
    if isinstance(value, (str, bytes, bytearray)):
        raise FullCycleValidationError("features must be a vector or mapping")
    if isinstance(value, (int, Fraction)) and not isinstance(value, bool):
        raise FullCycleValidationError(
            "features must be a full vector or a sparse mapping; a scalar "
            "feature is ambiguous")
    try:
        dense = tuple(value)
    except TypeError as error:
        raise FullCycleValidationError(
            "features must be a sequence or mapping") from error
    if dense and all(isinstance(item, (tuple, list)) and len(item) == 2
                     for item in dense):
        features, active = _sparse_pairs_to_mapping(
            dense, "feature", payload_min, payload_max)
        return "sparse", features, active
    if len(dense) != FULL_CYCLE_COUNT:
        raise FullCycleTruncationError(
            "features must contain all %d cycle dimensions, got %d; "
            "refusing to train a truncated window"
            % (FULL_CYCLE_COUNT, len(dense)))
    cleaned = []
    for index, item in enumerate(dense):
        item = _fixed_units(item, "features[%d]" % index)
        if not payload_min <= item <= payload_max:
            raise FullCycleOverflowError(
                "features[%d]=%d is outside [%d, %d]"
                % (index, item, payload_min, payload_max))
        cleaned.append(item)
    dense = tuple(cleaned)
    active = tuple(index for index, item in enumerate(dense) if item != 0)
    return "dense", dense, active


def _parse_target_scores(value: Any, payload_min: int,
                         payload_max: int) -> tuple[str, Any,
                                                     tuple[int, ...]]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
        targets, nonzero = _sparse_pairs_to_mapping(
            pairs, "target", payload_min, payload_max)
        return "sparse", targets, nonzero
    if isinstance(value, bool) or isinstance(value, float):
        raise FullCycleValidationError(
            "target scores must be integers in Q16.16 units, a Fraction, a "
            "full vector or a sparse mapping: %r" % (value,))
    if isinstance(value, (int, Fraction)):
        item = _fixed_units(value, "target score")
        if not payload_min <= item <= payload_max:
            raise FullCycleOverflowError(
                "target score %d is outside [%d, %d]"
                % (item, payload_min, payload_max))
        return "scalar", item, ()
    if isinstance(value, (str, bytes, bytearray)):
        raise FullCycleValidationError(
            "target scores must be a vector, mapping or integer")
    try:
        dense = tuple(value)
    except TypeError as error:
        raise FullCycleValidationError(
            "target scores must be a sequence, mapping or integer") from error
    if dense and all(isinstance(item, (tuple, list)) and len(item) == 2
                     for item in dense):
        targets, nonzero = _sparse_pairs_to_mapping(
            dense, "target", payload_min, payload_max)
        return "sparse", targets, nonzero
    if len(dense) != FULL_CYCLE_COUNT:
        raise FullCycleTruncationError(
            "target scores must contain all %d cycle dimensions, got %d; "
            "refusing to truncate to a 256-style window"
            % (FULL_CYCLE_COUNT, len(dense)))
    cleaned = []
    for index, item in enumerate(dense):
        item = _fixed_units(item, "target[%d]" % index)
        if not payload_min <= item <= payload_max:
            raise FullCycleOverflowError(
                "target[%d]=%d is outside [%d, %d]"
                % (index, item, payload_min, payload_max))
        cleaned.append(item)
    dense = tuple(cleaned)
    nonzero = tuple(index for index, item in enumerate(dense) if item != 0)
    return "dense", dense, nonzero


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class FullCycleTrainer:
    """Bounded fixed-point SGD over the complete 32,385-cycle coefficient vector.

    ``sgd_step(features, target_scores, lr)`` performs one exact integer
    gradient step.  A dense feature/target sequence must have exactly 32,385
    entries.  A sparse mapping may activate any subset of that fixed basis; the
    gradient and update vectors are still full-width, with zeros elsewhere.
    """

    dimension = FULL_CYCLE_COUNT
    cycle_count = FULL_CYCLE_COUNT
    full_basis = True

    def __init__(self, memory: Any = None, *,
                 learning_rate: int | Fraction = DEFAULT_LEARNING_RATE_UNITS,
                 max_learning_rate_units: int = DEFAULT_MAX_LEARNING_RATE_UNITS,
                 max_step_units: int = DEFAULT_MAX_STEP_UNITS,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 max_steps_per_call: int = DEFAULT_MAX_STEPS_PER_CALL,
                 max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                 payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX,
                 require_improvement: bool = False):
        if memory is None:
            memory = FullCycleMemory(payload_min=payload_min,
                                     payload_max=payload_max)
        if isinstance(memory, (str, bytes, bytearray)):
            raise TypeError("memory must be a cycle-word store, not text")
        _memory_dimension(memory)

        self.memory = memory
        self._reader, self._reader_is_word = _resolve_reader(memory)
        self._committer = _resolve_committer(memory)
        self._single_writer = _resolve_single_writer(memory)
        if self._committer is None and self._single_writer is None:
            raise TypeError(
                "memory is missing commit_cycles(updates) or set_cycle()")
        self._snapshot_hooks = _memory_snapshot_hooks(memory)

        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        memory_min = getattr(memory, "payload_min", PAYLOAD_MIN)
        memory_max = getattr(memory, "payload_max", PAYLOAD_MAX)
        if isinstance(memory_min, int) and not isinstance(memory_min, bool):
            payload_min = max(payload_min, memory_min)
        if isinstance(memory_max, int) and not isinstance(memory_max, bool):
            payload_max = min(payload_max, memory_max)
        if not PAYLOAD_MIN <= payload_min < payload_max <= PAYLOAD_MAX:
            raise FullCycleValidationError(
                "payload bounds must be inside symmetric int32")
        self.payload_min = payload_min
        self.payload_max = payload_max

        self.max_learning_rate_units = _require_int(
            max_learning_rate_units, "max_learning_rate_units")
        if self.max_learning_rate_units <= 0:
            raise FullCycleLimitError(
                "max_learning_rate_units must be positive")
        self.learning_rate_units = self._validate_learning_rate(learning_rate)

        self.max_step_units = _require_int(max_step_units, "max_step_units")
        if self.max_step_units <= 0:
            raise FullCycleLimitError("max_step_units must be positive")
        self.max_steps = _require_int(max_steps, "max_steps")
        if self.max_steps <= 0:
            raise FullCycleLimitError("max_steps must be positive")
        self.max_steps_per_call = _require_int(
            max_steps_per_call, "max_steps_per_call")
        if self.max_steps_per_call <= 0:
            raise FullCycleLimitError("max_steps_per_call must be positive")
        self.max_batch_size = _require_int(max_batch_size, "max_batch_size")
        if self.max_batch_size <= 0:
            raise FullCycleLimitError("max_batch_size must be positive")
        if not isinstance(require_improvement, bool):
            raise FullCycleValidationError("require_improvement must be bool")
        self.require_improvement = require_improvement

        self._frozen: set[int] = set()
        self._local_generation = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.no_op_steps = 0
        self.samples_seen = 0
        self.updates_applied = 0
        self.last_result: dict[str, Any] | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.last_rejection: dict[str, Any] | None = None

    # -- validation and readout -----------------------------------------
    def _validate_learning_rate(self, value: Any) -> int:
        units = _fixed_units(value, "learning rate")
        if units <= 0:
            raise FullCycleLimitError("learning rate must be positive")
        if units > self.max_learning_rate_units:
            raise FullCycleLimitError(
                "learning rate %d exceeds max_learning_rate_units=%d"
                % (units, self.max_learning_rate_units))
        return units

    @property
    def generation(self) -> int:
        value = _memory_generation(self.memory)
        if value is None:
            return self._local_generation
        return max(value, self._local_generation)

    @property
    def frozen(self) -> frozenset[int]:
        return frozenset(self._frozen)

    @property
    def frozen_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._frozen))

    @property
    def weights(self) -> tuple[int, ...]:
        return self._read_all()

    @property
    def coefficients(self) -> tuple[int, ...]:
        return self._read_all()

    @property
    def raw_words(self) -> tuple[int, ...]:
        return tuple(_to_word(value) for value in self._read_all())

    @property
    def gradient_shape(self) -> tuple[int]:
        return (FULL_CYCLE_COUNT,)

    @property
    def parameter_count(self) -> int:
        return FULL_CYCLE_COUNT

    @property
    def receipt(self) -> dict[str, Any] | None:
        return self.last_receipt

    def weight(self, index: Any) -> int:
        return self._read_signed(_coerce_cycle_key(index))

    def _read_signed(self, index: int) -> int:
        try:
            value = self._reader(index)
        except FullCycleTrainingError:
            raise
        except Exception as error:  # noqa: BLE001 - backend boundary
            raise FullCycleFieldError(
                "memory read failed at cycle %d: %r" % (index, error)
            ) from error
        if isinstance(value, bool) or not isinstance(value, int):
            raise FullCycleFieldError(
                "memory returned a non-integer cycle word at %d: %r"
                % (index, value))
        if self._reader_is_word and 0 <= value <= WORD_MASK:
            return _normalize_payload(_from_word(value), "cycle word")
        return _normalize_payload(value, "cycle word")

    def _read_all(self) -> tuple[int, ...]:
        read_all = getattr(self.memory, "read_all", None)
        values = None
        if callable(read_all):
            values = read_all()
        else:
            for name in _COEFFICIENT_SEQUENCE_NAMES:
                candidate = getattr(self.memory, name, None)
                if callable(candidate) or candidate is None:
                    continue
                values = candidate
                break
        if values is not None:
            if isinstance(values, Mapping):
                if len(values) != FULL_CYCLE_COUNT:
                    raise FullCycleTruncationError(
                        "memory read_all returned %d words, expected %d"
                        % (len(values), FULL_CYCLE_COUNT))
                cleaned = []
                for index in range(FULL_CYCLE_COUNT):
                    if index not in values:
                        raise FullCycleDimensionError(
                            "memory read_all is missing cycle %d" % index)
                    cleaned.append(_normalize_payload(
                        values[index], "cycle word"))
                return tuple(cleaned)
            try:
                dense = tuple(values)
            except TypeError as error:
                raise FullCycleFieldError(
                    "memory read_all must return a sequence") from error
            if len(dense) != FULL_CYCLE_COUNT:
                raise FullCycleTruncationError(
                    "memory exposes %d cycle words, expected all %d"
                    % (len(dense), FULL_CYCLE_COUNT))
            return tuple(_normalize_payload(value, "cycle word")
                         for value in dense)
        return tuple(self._read_signed(index)
                     for index in range(FULL_CYCLE_COUNT))

    def _prepare_sample(self, features: Any,
                        target_scores: Any) -> _PreparedSample:
        feature_kind, parsed_features, active = _parse_feature_vector(
            features, self.payload_min, self.payload_max)
        target_kind, parsed_targets, target_nonzero = _parse_target_scores(
            target_scores, self.payload_min, self.payload_max)
        if target_kind == "scalar":
            mode = "linear"
            relevant = active
        else:
            mode = "elementwise"
            relevant = tuple(sorted(set(active) | set(target_nonzero)))
        return _PreparedSample(
            mode=mode,
            features=parsed_features,
            feature_kind=feature_kind,
            targets=parsed_targets,
            target_kind=target_kind,
            active_features=active,
            relevant_indices=relevant,
        )

    def validate_features(self, features: Any) -> dict[str, Any]:
        _kind, _values, active = _parse_feature_vector(
            features, self.payload_min, self.payload_max)
        return {
            "dimension": FULL_CYCLE_COUNT,
            "active_features": len(active),
            "full_basis": True,
            "truncated": False,
        }

    def validate_targets(self, target_scores: Any) -> dict[str, Any]:
        kind, _values, nonzero = _parse_target_scores(
            target_scores, self.payload_min, self.payload_max)
        return {
            "dimension": FULL_CYCLE_COUNT,
            "kind": kind,
            "nonzero_targets": len(nonzero),
            "full_basis": True,
            "truncated": False,
        }

    def validate_full_basis(self) -> dict[str, Any]:
        values = self._read_all()
        if len(values) != FULL_CYCLE_COUNT:
            raise FullCycleTruncationError(
                "coefficient vector has %d entries, expected %d"
                % (len(values), FULL_CYCLE_COUNT))
        return {
            "schema": FULL_CYCLE_TRAINING_SCHEMA,
            "dimension": FULL_CYCLE_COUNT,
            "readable": len(values),
            "nonzero": sum(1 for value in values if value),
            "full_basis": True,
            "truncated": False,
        }

    def coverage(self) -> dict[str, Any]:
        return self.validate_full_basis()

    validate_dimension = validate_full_basis

    def cycle_readout(self, first: Any, second: Any = None,
                      root: int = ROOT_HEAD) -> int:
        method = getattr(self.memory, "cycle_readout", None)
        if callable(method):
            try:
                value = method(first, second, root=root)
            except TypeError:
                if second is None:
                    value = method(first)
                else:
                    value = method(first, second)
            return _require_int(value, "cycle readout")
        if second is None:
            left, right = cycle_pair(_require_int(first, "cycle index"))
        else:
            left, right = _check_head(first), _check_head(second)
        root = _check_head(root, "root")
        reader = getattr(self.memory, "read_lane", None)
        if not callable(reader):
            raise FullCycleFieldError(
                "memory has no cycle_readout() or read_lane() hook")
        return (reader(root, left) + reader(left, right)
                - reader(root, right))

    readout = cycle_readout

    def cycle_features(self, root: int = ROOT_HEAD) -> tuple[int, ...]:
        method = getattr(self.memory, "cycle_features", None)
        if callable(method):
            try:
                values = tuple(method(root=root))
            except TypeError:
                values = tuple(method())
            if len(values) != FULL_CYCLE_COUNT:
                raise FullCycleTruncationError(
                    "memory cycle_features returned %d values, expected %d"
                    % (len(values), FULL_CYCLE_COUNT))
            return tuple(_require_int(value, "cycle feature")
                         for value in values)
        values = []
        for first, second in CYCLE_BASIS:
            values.append(self.cycle_readout(first, second, root=root))
        return tuple(values)

    features_from_memory = cycle_features

    # -- exact loss and gradient ----------------------------------------
    def _elementwise_loss_and_gradient(
            self, sample: _PreparedSample, weights: Sequence[int]
    ) -> tuple[int, dict[int, int]]:
        loss_units = 0
        gradient: dict[int, int] = {}
        for index in sample.relevant_indices:
            feature = sample.feature(index)
            prediction = (_round_div(weights[index] * feature, FIXED_ONE)
                          if feature else 0)
            residual = prediction - sample.target(index)
            loss_units += residual * residual * HALF_FIXED
            if feature and index not in self._frozen:
                value = _round_div(residual * feature, FIXED_ONE)
                if value:
                    gradient[index] = value
        return loss_units, gradient

    def _linear_loss_and_gradient(
            self, sample: _PreparedSample, weights: Sequence[int]
    ) -> tuple[int, dict[int, int]]:
        total = 0
        for index in sample.active_features:
            total += weights[index] * sample.feature(index)
        prediction = _round_div(total, FIXED_ONE)
        residual = prediction - sample.targets
        loss_units = residual * residual * HALF_FIXED
        gradient: dict[int, int] = {}
        for index in sample.active_features:
            if index in self._frozen:
                continue
            value = _round_div(residual * sample.feature(index), FIXED_ONE)
            if value:
                gradient[index] = value
        return loss_units, gradient

    def _sample_loss_and_gradient(
            self, sample: _PreparedSample, weights: Sequence[int]
    ) -> tuple[int, dict[int, int]]:
        if sample.mode == "linear":
            return self._linear_loss_and_gradient(sample, weights)
        return self._elementwise_loss_and_gradient(sample, weights)

    def _sample_loss_units(self, sample: _PreparedSample,
                           weights: Sequence[int],
                           updates: Mapping[int, int] | None = None) -> int:
        def current(index: int) -> int:
            if updates is not None and index in updates:
                return updates[index]
            return weights[index]

        if sample.mode == "linear":
            total = 0
            for index in sample.active_features:
                total += current(index) * sample.feature(index)
            residual = _round_div(total, FIXED_ONE) - sample.targets
            return residual * residual * HALF_FIXED
        loss_units = 0
        for index in sample.relevant_indices:
            feature = sample.feature(index)
            prediction = (_round_div(current(index) * feature, FIXED_ONE)
                          if feature else 0)
            residual = prediction - sample.target(index)
            loss_units += residual * residual * HALF_FIXED
        return loss_units

    def _loss_denominator(self, sample: _PreparedSample) -> int:
        if sample.mode == "linear":
            return 1
        return FULL_CYCLE_COUNT

    def _metric_loss(self, loss_units: int, samples: Sequence[_PreparedSample],
                     *, mean: bool) -> float:
        if not mean:
            return loss_units / FIXED_ONE
        denominator = sum(self._loss_denominator(sample) for sample in samples)
        if denominator <= 0:
            return 0.0
        return loss_units / FIXED_ONE / denominator

    def gradient(self, features: Any,
                 target_scores: Any) -> tuple[int, ...]:
        """Return the full 32,385-entry exact integer gradient."""
        sample = self._prepare_sample(features, target_scores)
        weights = self._read_all()
        _loss, gradient = self._sample_loss_and_gradient(sample, weights)
        full = [0] * FULL_CYCLE_COUNT
        for index, value in gradient.items():
            full[index] = value
        return tuple(full)

    gradient_units = gradient
    gradients = gradient

    def loss_units(self, features: Any, target_scores: Any, *,
                   mean: bool = False) -> int:
        sample = self._prepare_sample(features, target_scores)
        weights = self._read_all()
        units = self._sample_loss_units(sample, weights)
        if mean:
            denominator = self._loss_denominator(sample)
            if denominator:
                units //= denominator
        return units

    def loss(self, features: Any, target_scores: Any, *,
             mean: bool = True) -> float:
        """Reporting metric only; never fed back into a cycle word."""
        sample = self._prepare_sample(features, target_scores)
        weights = self._read_all()
        units = self._sample_loss_units(sample, weights)
        return self._metric_loss(units, (sample,), mean=mean)

    mse_loss = loss

    def _softplus(self, value: float) -> float:
        if value > 40.0:
            return value
        if value < -40.0:
            return math.exp(value)
        return math.log1p(math.exp(value))

    def ce_like_loss(self, features: Any, target_scores: Any) -> float:
        """Bounded one-vs-zero logistic metric derived from exact integers.

        This is a reporting metric only.  Training always uses the exact
        integer MSE gradient; no float is converted into a cycle word.
        """
        sample = self._prepare_sample(features, target_scores)
        weights = self._read_all()
        inverse = float(FIXED_ONE)
        if sample.mode == "linear":
            total = 0
            for index in sample.active_features:
                total += weights[index] * sample.feature(index)
            prediction = _round_div(total, FIXED_ONE) / inverse
            target = sample.targets / inverse
            return self._softplus(-target * prediction)
        total = 0.0
        count = 0
        for index in sample.active_features:
            feature = sample.feature(index)
            prediction = _round_div(weights[index] * feature,
                                    FIXED_ONE) / inverse
            target = sample.target(index) / inverse
            total += self._softplus(-target * prediction)
            count += 1
        return total / count if count else 0.0

    ce_loss = ce_like_loss

    # -- freeze/unfreeze -------------------------------------------------
    def _normalize_freeze_indices(self, indices: Any) -> tuple[int, ...]:
        if indices is None:
            return tuple(range(FULL_CYCLE_COUNT))
        if isinstance(indices, (int, str)) or hasattr(indices, "keys"):
            indices = (indices,)
        elif isinstance(indices, tuple) and not indices:
            return ()
        try:
            values = tuple(_coerce_cycle_key(value) for value in indices)
        except TypeError as error:
            raise FullCycleValidationError(
                "indices must be one cycle key or an iterable") from error
        if len(set(values)) != len(values):
            raise FullCycleValidationError("freeze indices must be unique")
        return tuple(sorted(values))

    def freeze(self, indices: Any = None) -> int:
        values = self._normalize_freeze_indices(indices)
        self._frozen.update(values)
        return len(values)

    def unfreeze(self, indices: Any = None) -> int:
        if indices is None:
            removed = len(self._frozen)
            self._frozen.clear()
            return removed
        values = self._normalize_freeze_indices(indices)
        removed = 0
        for value in values:
            if value in self._frozen:
                self._frozen.remove(value)
                removed += 1
        return removed

    def is_frozen(self, index: Any) -> bool:
        return _coerce_cycle_key(index) in self._frozen

    # -- snapshots and commits ------------------------------------------
    def _take_state(self) -> tuple[str, Any]:
        if self._snapshot_hooks is not None:
            return "memory", self._snapshot_hooks[0]()
        return "weights", self._read_all()

    def _restore_state(self, state: tuple[str, Any]) -> None:
        kind, payload = state
        if kind == "memory":
            if self._snapshot_hooks is None:
                raise FullCycleFieldError(
                    "memory snapshot hooks disappeared during rollback")
            self._snapshot_hooks[1](payload)
            return
        values = tuple(_normalize_payload(value, "rollback word")
                       for value in payload)
        if len(values) != FULL_CYCLE_COUNT:
            raise FullCycleTruncationError(
                "rollback vector has %d words, expected %d"
                % (len(values), FULL_CYCLE_COUNT))
        if self._committer is not None:
            self._committer(tuple(enumerate(values)))
        elif self._single_writer is not None:
            for index, value in enumerate(values):
                self._single_writer(index, value)
        else:
            raise FullCycleFieldError("memory has no rollback write path")

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": FULL_CYCLE_TRAINING_SCHEMA,
            "generation": self.generation,
            "weights_digest": self.weight_digest(),
            "state": self._take_state(),
            "frozen": tuple(sorted(self._frozen)),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, Mapping):
            raise FullCycleValidationError("snapshot must be a mapping")
        if "state" in snapshot:
            state = snapshot["state"]
            frozen = snapshot.get("frozen")
            if frozen is not None:
                self._frozen = set(self._normalize_freeze_indices(frozen))
        else:  # tolerate a direct memory snapshot
            state = ("memory", snapshot)
        self._restore_state(state)
        return self.stats()

    def weight_digest(self) -> str:
        return _digest_words(tuple(_to_word(value)
                                   for value in self._read_all()))

    def _commit_updates(self, updates: Sequence[tuple[int, int]]) -> int:
        generation_before = self.generation
        state = self._take_state()
        try:
            if self._committer is not None:
                self._committer(updates)
            else:
                for index, value in updates:
                    self._single_writer(index, value)
        except Exception as error:  # noqa: BLE001 - backend boundary
            try:
                self._restore_state(state)
            except Exception as restore_error:  # noqa: BLE001
                raise FullCycleFieldError(
                    "field commit failed and rollback also failed: %r / %r"
                    % (error, restore_error)) from error
            raise FullCycleFieldError(
                "field commit failed; no step was accepted: %r" % (error,)
            ) from error

        generation_after = self.generation
        memory_generation = _memory_generation(self.memory)
        if self._committer is None or memory_generation is None:
            self._local_generation = max(self._local_generation,
                                         generation_before) + 1
            generation_after = self._local_generation
        elif generation_after <= generation_before:
            try:
                self._restore_state(state)
            except Exception:  # noqa: BLE001 - preserve original failure
                pass
            raise FullCycleFieldError(
                "memory commit did not advance generation (%d -> %d)"
                % (generation_before, generation_after))
        for index, value in updates:
            try:
                actual = self._read_signed(index)
            except FullCycleTrainingError:
                raise
            if actual != value:
                try:
                    self._restore_state(state)
                except Exception:  # noqa: BLE001 - preserve original failure
                    pass
                raise FullCycleFieldError(
                    "field readback at cycle %d is %d, proposed %d"
                    % (index, actual, value))
        return generation_after

    def _propose_updates(self, gradient: Mapping[int, int], weights: Sequence[int],
                         lr_units: int) -> tuple[list[tuple[int, int]],
                                                 dict[int, int]]:
        updates: list[tuple[int, int]] = []
        values: dict[int, int] = {}
        for index in sorted(gradient):
            value = gradient[index]
            if value == 0 or index in self._frozen:
                continue
            magnitude = _round_div(lr_units * abs(value), FIXED_ONE)
            if magnitude > self.max_step_units:
                raise FullCycleOverflowError(
                    "step magnitude %d at cycle %d exceeds "
                    "max_step_units=%d"
                    % (magnitude, index, self.max_step_units))
            delta = -magnitude if value > 0 else magnitude
            if delta == 0:
                continue
            proposed = weights[index] + delta
            if not self.payload_min <= proposed <= self.payload_max:
                raise FullCycleOverflowError(
                    "proposed cycle word %d=%d is outside [%d, %d]"
                    % (index, proposed, self.payload_min,
                       self.payload_max))
            if proposed != weights[index]:
                updates.append((index, proposed))
                values[index] = proposed
        return updates, values

    def _sgd_prepared(self, samples: Sequence[_PreparedSample],
                      lr_units: int, *, batch: bool) -> dict[str, Any]:
        if not samples:
            raise FullCycleValidationError("a step needs at least one sample")
        if self.accepted_steps >= self.max_steps:
            raise FullCycleLimitError(
                "step limit reached: max_steps=%d" % self.max_steps)
        generation_before = self.generation
        weights = self._read_all()

        loss_before_units = 0
        total_gradient: dict[int, int] = {}
        max_abs_gradient = 0
        active_features = 0
        for sample in samples:
            units, gradient = self._sample_loss_and_gradient(sample, weights)
            loss_before_units += units
            active_features += len(sample.active_features)
            for index, value in gradient.items():
                total_gradient[index] = total_gradient.get(index, 0) + value
                if abs(value) > max_abs_gradient:
                    max_abs_gradient = abs(value)

        updates, proposed = self._propose_updates(
            total_gradient, weights, lr_units)
        loss_after_units = sum(
            self._sample_loss_units(sample, weights, proposed)
            for sample in samples)

        if self.require_improvement and loss_after_units > loss_before_units:
            raise FullCycleStepRejected(
                "step would increase loss from %d to %d fixed-point units"
                % (loss_before_units, loss_after_units))

        max_abs_step = 0
        for index, value in updates:
            delta = abs(value - weights[index])
            if delta > max_abs_step:
                max_abs_step = delta

        if not updates:
            self.accepted_steps += 1
            self.no_op_steps += 1
            self.samples_seen += len(samples)
            receipt = self._receipt(
                samples, generation_before, generation_before,
                loss_before_units, loss_after_units, total_gradient,
                max_abs_gradient, max_abs_step, lr_units,
                updates=0, changed=False, batch=batch,
                active_features=active_features)
            self.last_result = receipt
            self.last_receipt = receipt
            return receipt

        generation_after = self._commit_updates(updates)

        self.accepted_steps += 1
        self.samples_seen += len(samples)
        self.updates_applied += len(updates)
        receipt = self._receipt(
            samples, generation_before, generation_after,
            loss_before_units, loss_after_units, total_gradient,
            max_abs_gradient, max_abs_step, lr_units,
            updates=len(updates), changed=True, batch=batch,
            active_features=active_features)
        self.last_result = receipt
        self.last_receipt = receipt
        return receipt

    def _receipt(self, samples: Sequence[_PreparedSample],
                 generation_before: int, generation_after: int,
                 loss_before_units: int, loss_after_units: int,
                 gradient: Mapping[int, int], max_abs_gradient: int,
                 max_abs_step: int, lr_units: int, *,
                 updates: int, changed: bool, batch: bool,
                 active_features: int) -> dict[str, Any]:
        gradient_values = tuple(gradient.values())
        mean_gradient = (sum(gradient_values) // len(gradient_values)
                         if gradient_values else 0)
        return {
            "schema": FULL_CYCLE_STEP_SCHEMA,
            "accepted": True,
            "changed": changed,
            "batch": batch,
            "samples": len(samples),
            "dimension": FULL_CYCLE_COUNT,
            "full_basis": True,
            "truncated": False,
            "active_features": active_features,
            "frozen_weights": len(self._frozen),
            "weights_updated": updates,
            "generation_before": generation_before,
            "generation_after": generation_after,
            "generation": generation_after,
            "loss_units_before": loss_before_units,
            "loss_units_after": loss_after_units,
            "loss_before": self._metric_loss(
                loss_before_units, samples, mean=True),
            "loss_after": self._metric_loss(
                loss_after_units, samples, mean=True),
            "mean_gradient_units": mean_gradient,
            "max_abs_gradient_units": max_abs_gradient,
            "max_abs_step_units": max_abs_step,
            "learning_rate_units": lr_units,
            "weights_digest": self.weight_digest(),
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
        }

    # -- public steps ----------------------------------------------------
    def sgd_step(self, features: Any, target_scores: Any = None,
                 lr: int | Fraction | None = None, *,
                 steps: int = 1) -> dict[str, Any]:
        """Run one or more deterministic fixed-point SGD steps.

        Passing ``target_scores=None`` treats ``features`` as an iterable of
        ``(features, target_scores)`` samples and performs one batch step.
        """
        if target_scores is None:
            return self.train_batch(features, lr=lr)
        steps = _require_int(steps, "steps")
        if steps <= 0:
            raise FullCycleLimitError("steps must be positive")
        if steps > self.max_steps_per_call:
            raise FullCycleLimitError(
                "steps=%d exceeds max_steps_per_call=%d"
                % (steps, self.max_steps_per_call))
        if self.accepted_steps + steps > self.max_steps:
            raise FullCycleLimitError(
                "steps=%d would exceed max_steps=%d"
                % (steps, self.max_steps))
        lr_units = (self.learning_rate_units if lr is None
                    else self._validate_learning_rate(lr))
        sample = self._prepare_sample(features, target_scores)
        receipts = []
        for _ in range(steps):
            receipts.append(self._sgd_prepared((sample,), lr_units,
                                               batch=False))
        if len(receipts) == 1:
            return receipts[0]
        first = dict(receipts[0])
        last = receipts[-1]
        first.update({
            "steps_run": len(receipts),
            "step_receipts": receipts,
            "generation_before": receipts[0]["generation_before"],
            "generation_after": last["generation_after"],
            "generation": last["generation"],
            "loss_units_before": receipts[0]["loss_units_before"],
            "loss_units_after": last["loss_units_after"],
            "loss_before": receipts[0]["loss_before"],
            "loss_after": last["loss_after"],
            "weights_updated": sum(item["weights_updated"]
                                   for item in receipts),
            "changed": any(item["changed"] for item in receipts),
        })
        return first

    step = sgd_step
    train_step = sgd_step
    sgd = sgd_step

    def train_batch(self, batch: Iterable[Any],
                    lr: int | Fraction | None = None) -> dict[str, Any]:
        samples = list(batch)
        if not samples:
            raise FullCycleValidationError("batch must not be empty")
        if len(samples) > self.max_batch_size:
            raise FullCycleLimitError(
                "batch has %d samples, above max_batch_size=%d"
                % (len(samples), self.max_batch_size))
        prepared = []
        for item in samples:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise FullCycleValidationError(
                    "each batch sample must be (features, target_scores)")
            prepared.append(self._prepare_sample(item[0], item[1]))
        lr_units = (self.learning_rate_units if lr is None
                    else self._validate_learning_rate(lr))
        return self._sgd_prepared(tuple(prepared), lr_units, batch=True)

    sgd_batch = train_batch
    batch_step = train_batch

    def deterministic_batches(self, samples: Iterable[Any],
                              batch_size: int = 1, *, seed: int = 0,
                              shuffle: bool = True
                              ) -> tuple[tuple[Any, ...], ...]:
        if batch_size > self.max_batch_size:
            raise FullCycleLimitError(
                "batch_size=%d exceeds max_batch_size=%d"
                % (batch_size, self.max_batch_size))
        return deterministic_batches(samples, batch_size, seed=seed,
                                     shuffle=shuffle)

    make_batches = deterministic_batches

    def fit(self, samples: Iterable[Any], *, epochs: int = 1,
            batch_size: int = 1, lr: int | Fraction | None = None,
            seed: int = 0, shuffle: bool = True) -> dict[str, Any]:
        """Train deterministic batches without any hidden RNG state."""
        items = list(samples)
        epochs = _require_int(epochs, "epochs")
        if epochs <= 0:
            raise FullCycleLimitError("epochs must be positive")
        receipts = []
        for _ in range(epochs):
            for batch in deterministic_batches(
                    items, batch_size, seed=seed, shuffle=shuffle):
                receipts.append(self.train_batch(batch, lr=lr))
        return {
            "schema": FULL_CYCLE_TRAINING_SCHEMA,
            "epochs": epochs,
            "batches": len(receipts),
            "receipts": receipts,
            "dimension": FULL_CYCLE_COUNT,
            "full_basis": True,
            "weights_digest": self.weight_digest(),
        }

    def try_sgd_step(self, features: Any, target_scores: Any = None,
                     lr: int | Fraction | None = None, *,
                     steps: int = 1) -> dict[str, Any]:
        try:
            return self.sgd_step(features, target_scores, lr, steps=steps)
        except FullCycleTrainingError as error:
            rejection = {
                "schema": FULL_CYCLE_STEP_SCHEMA,
                "accepted": False,
                "changed": False,
                "reason": type(error).__name__,
                "detail": str(error),
                "generation": self.generation,
                "dimension": FULL_CYCLE_COUNT,
                "full_basis": True,
                "canonical_merge": CANONICAL_MERGE,
                "semantic_truth": FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
            }
            self.rejected_steps += 1
            self.last_rejection = rejection
            return rejection

    try_step = try_sgd_step

    # -- gauge -----------------------------------------------------------
    def apply_root_star_gauge(self, gauge: Any, *,
                              root: int = ROOT_HEAD) -> dict[str, Any]:
        method = getattr(self.memory, "apply_root_star_gauge", None)
        if not callable(method):
            method = getattr(self.memory, "apply_gauge", None)
        if not callable(method):
            raise FullCycleGaugeError(
                "memory has no apply_root_star_gauge() hook")
        before_features = self.cycle_features(root=root)
        before_digest = self.weight_digest()
        generation_before = self.generation
        try:
            applied = method(gauge, root=root)
        except TypeError:
            applied = method(gauge)
        after_features = self.cycle_features(root=root)
        after_digest = self.weight_digest()
        if before_features != after_features:
            raise FullCycleGaugeError(
                "root-star gauge changed the cycle readout")
        if before_digest != after_digest:
            raise FullCycleGaugeError(
                "root-star gauge changed the trained cycle weights")
        return {
            "schema": FULL_CYCLE_GAUGE_SCHEMA,
            "applied": applied,
            "generation_before": generation_before,
            "generation_after": self.generation,
            "generation": self.generation,
            "dimension": FULL_CYCLE_COUNT,
            "full_basis": True,
            "cycle_readout_unchanged": True,
            "weights_unchanged": True,
            "weights_digest": after_digest,
            "readout_digest": _digest_words(
                tuple(_to_word(value) for value in after_features)),
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
        }

    gauge_shift = apply_root_star_gauge
    root_star_gauge = apply_root_star_gauge

    # -- reporting -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "schema": FULL_CYCLE_TRAINING_SCHEMA,
            "memory": type(self.memory).__name__,
            "generation": self.generation,
            "dimension": FULL_CYCLE_COUNT,
            "full_basis": True,
            "truncated": False,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "no_op_steps": self.no_op_steps,
            "samples_seen": self.samples_seen,
            "updates_applied": self.updates_applied,
            "frozen_weights": len(self._frozen),
            "learning_rate_units": self.learning_rate_units,
            "max_learning_rate_units": self.max_learning_rate_units,
            "max_step_units": self.max_step_units,
            "max_steps": self.max_steps,
            "max_steps_per_call": self.max_steps_per_call,
            "max_batch_size": self.max_batch_size,
            "payload_min": self.payload_min,
            "payload_max": self.payload_max,
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
        }

    def __repr__(self) -> str:
        return ("FullCycleTrainer(dimension=%d, generation=%d, steps=%d, "
                "frozen=%d)"
                % (FULL_CYCLE_COUNT, self.generation, self.accepted_steps,
                   len(self._frozen)))


def apply_root_star_gauge(memory: Any, gauge: Any, *,
                          root: int = ROOT_HEAD) -> Any:
    """Apply a root-star-anchored gauge to a memory's pair lanes."""
    method = getattr(memory, "apply_root_star_gauge", None)
    if not callable(method):
        method = getattr(memory, "apply_gauge", None)
    if not callable(method):
        raise FullCycleGaugeError(
            "memory has no apply_root_star_gauge() hook")
    try:
        return method(gauge, root=root)
    except TypeError:
        return method(gauge)


__all__ = (
    "FULL_CYCLE_COUNT", "FULL_CYCLE_DIMENSION", "CYCLE_SPACE_DIMENSION",
    "CYCLE_COUNT", "ROOT_CYCLE_COUNT", "ROOT_HEAD", "CYCLE_BASIS",
    "CYCLE_PAIRS", "WORD_BITS", "WORD_MASK", "WORD_MODULUS", "HALF_FIXED",
    "PAYLOAD_MIN", "PAYLOAD_MAX", "DEFAULT_LEARNING_RATE_UNITS",
    "DEFAULT_MAX_LEARNING_RATE_UNITS", "DEFAULT_MAX_STEP_UNITS",
    "DEFAULT_MAX_STEPS", "DEFAULT_MAX_STEPS_PER_CALL",
    "DEFAULT_MAX_BATCH_SIZE", "FULL_CYCLE_TRAINING_SCHEMA",
    "FULL_CYCLE_STEP_SCHEMA", "FULL_CYCLE_MEMORY_SCHEMA",
    "FULL_CYCLE_GAUGE_SCHEMA", "CANONICAL_MERGE",
    "FULL_CYCLE_TRAINING_IS_SEMANTIC_TRUTH", "FullCycleTrainingError",
    "FullCycleValidationError", "FullCycleDimensionError",
    "FullCycleTruncationError", "FullCycleOverflowError",
    "FullCycleStepRejected", "FullCycleFieldError", "FullCycleGaugeError",
    "FullCycleLimitError", "cycle_index", "cycle_pair",
    "deterministic_batches", "FullCycleMemory", "IntegerFullCycleMemory",
    "FullCycleStore", "FullCycleTrainer", "apply_root_star_gauge",
)
