"""Fixed-point supervised training of cycle readouts on raw pair lanes.

Round 029 demonstrated that the oriented triangle observable::

    F(0, i, j) = x(0, i) + x(i, j) + x(j, 0)

is exactly invariant under an additive vertex gauge
``x(a, b) -> x(a, b) + g(a) - g(b)``.  This module trains that readout in
resident memory without ever converting an identity, gauge, edge or cycle
value to a Python ``float``.

Two value families are kept deliberately separate:

* **Payload values** (edge lanes, gauges, targets, predictions, gradients,
  losses in fixed point) are Python integers in Q16.16 units.
* **Reporting metrics** (``loss()`` and receipt helpers) may be floats.  They
  are derived from the exact integer result and are never fed back into a
  write.

Field protocol
--------------

``CycleTrainer(field)`` accepts any object with this small protocol::

    field.read_lane(source, target) -> int
    field.commit_pairs(updates) -> object        # atomic, mirror-safe
    field.generation -> int

``commit_pairs`` receives ``(left, right, value)`` triples with
``left < right``.  ``value`` is the raw lane in the ``left -> right``
direction; the implementation must write ``right -> left = -value`` and
increment ``generation`` once.  The commit must either apply the complete
batch or raise without changing the field.

An optional ``field.snapshot()`` / ``field.restore(snapshot)`` pair is used
for rollback if a field raises after a partial write.  :class:`CellsLaneField`
adapts a :class:`~nanlmind_hermes.resident.cells.ResidentCells` region directly
and :class:`IntegerLaneField` is the deterministic in-memory reference field.
:class:`CycleObservableLaneField` adapts the raw-u32
``nanlmind_hermes.resident.cycle_observable.CycleField`` view.  Passing a
``ResidentCells``, a ``CycleField`` or an object with a ``.cells`` attribute
holding ``ResidentCells`` to :class:`CycleTrainer` wraps it automatically.

Boundaries
----------

* Writes are bounded by ``max_step_units``; out-of-range targets, lanes or
  steps are denied rather than saturated.
* A whole batch is validated and proposed before one atomic commit.
* The field generation advances only after an accepted commit.
* No canonical merge, truth claim or language-model claim is made here.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from .cells import (
    FIXED_ONE,
    HEAD_COUNT,
    RAIL_FLAG_INITIALIZED,
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    RAIL_STRUCT,
    ResidentCells,
    address_of,
)

CYCLE_TRAINING_SCHEMA = "nanlm.resident-cycle-training/1"
CYCLE_STEP_SCHEMA = "nanlm.resident-cycle-step/1"
CYCLE_EXPERIMENT_SCHEMA = "nanlm.resident-cycle-gauge-experiment/1"

# Symmetric payload range: a value and its exact mirror must both fit in i32.
_I32_MAX = (1 << 31) - 1
PAYLOAD_MIN = -_I32_MAX
PAYLOAD_MAX = _I32_MAX
HALF_FIXED = FIXED_ONE // 2

DEFAULT_LEARNING_RATE_UNITS = FIXED_ONE // 10  # 0.1, as in native_mirror.cpp
DEFAULT_MAX_STEP_UNITS = 1 << 26                # 1024.0 in Q16.16
DEFAULT_MAX_BATCH_SIZE = 1024
DEFAULT_MAX_PAIRS_PER_STEP = 4 * DEFAULT_MAX_BATCH_SIZE

# Explicit declarations: training is evidence, not canonical truth.
CANONICAL_MERGE = False
CYCLE_TRAINING_IS_SEMANTIC_TRUTH = False


class CycleTrainingError(ValueError):
    """Base class for malformed observations and refused cycle steps."""


class CycleValidationError(CycleTrainingError):
    """Raised when an observation, batch or trainer option is malformed."""


class CycleOverflowError(CycleTrainingError):
    """Raised instead of saturating an out-of-range payload or step."""


class CycleMirrorError(CycleTrainingError):
    """Raised when the field's two directed lanes are not exact mirrors."""


class CycleStepRejected(CycleTrainingError):
    """Raised when a validated step would not improve the bounded loss."""


class CycleFieldError(CycleTrainingError):
    """Raised when a field commit fails or readback is inconsistent."""


# ---------------------------------------------------------------------------
# integer helpers
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleValidationError("%s must be an integer: %r" % (name, value))
    return value


def _check_head(value: Any, name: str = "head") -> int:
    head = _require_int(value, name)
    if not 0 <= head < HEAD_COUNT:
        raise CycleValidationError(
            "%s must be in [0, %d): %r" % (name, HEAD_COUNT, value))
    return head


def _check_pair(source: Any, target: Any) -> tuple[int, int]:
    first = _check_head(source, "source")
    second = _check_head(target, "target")
    if first == second:
        raise CycleValidationError("a pair needs two distinct heads")
    return first, second


def _check_triangle(left: Any, right: Any) -> tuple[int, int]:
    i, j = _check_pair(left, right)
    if i == 0 or j == 0:
        raise CycleValidationError(
            "triangle heads must be non-zero and distinct: %r, %r"
            % (left, right))
    return i, j


def _round_div(numerator: int, denominator: int) -> int:
    """Round an exact integer quotient, half away from zero."""
    if denominator <= 0:
        raise CycleValidationError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def fixed_ratio(numerator: int, denominator: int = 1) -> int:
    """Return an exact Q16.16 integer for ``numerator / denominator``."""
    numerator = _require_int(numerator, "numerator")
    denominator = _require_int(denominator, "denominator")
    if denominator == 0:
        raise CycleValidationError("denominator must not be zero")
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    product = numerator * FIXED_ONE
    if product >= 0:
        return (product + denominator // 2) // denominator
    return -((-product + denominator // 2) // denominator)


def _learning_rate_units(value: Any) -> int:
    if isinstance(value, bool):
        raise CycleValidationError("learning rate must be a fixed-point integer")
    if isinstance(value, Fraction):
        units = int(round(value * FIXED_ONE))
    elif isinstance(value, int):
        units = value
    else:
        raise CycleValidationError(
            "learning rate must be an integer in Q16.16 units or a Fraction; "
            "floats are not accepted: %r" % (value,))
    if units <= 0:
        raise CycleValidationError("learning rate must be positive")
    return units


def _field_generation(field: Any) -> int:
    value = getattr(field, "generation", None)
    if callable(value):
        value = value()
    return _require_int(value, "field generation")


def _field_reader(field: Any):
    reader = getattr(field, "read_lane", None)
    if not callable(reader):
        reader = getattr(field, "read_payload", None)
    if not callable(reader):
        raise TypeError("field is missing read_lane(source, target)")
    return reader


def _field_committer(field: Any):
    committer = getattr(field, "commit_pairs", None)
    if not callable(committer):
        committer = getattr(field, "write_pairs", None)
    if not callable(committer):
        raise TypeError("field is missing commit_pairs(updates)")
    return committer


def _snapshot_hooks(field: Any):
    snapshot = getattr(field, "snapshot", None)
    restore = getattr(field, "restore", None)
    if callable(snapshot) and callable(restore):
        return snapshot, restore
    return None


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
class CycleObservation(NamedTuple):
    """One triangle target in raw Q16.16 payload units."""

    left: int
    right: int
    target: int

    @property
    def edges(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return ((0, self.left), (self.left, self.right), (self.right, 0))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _looks_single(value: Any) -> bool:
    if isinstance(value, (CycleObservation, Mapping)):
        return True
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return all(_is_int(item) for item in value)
    return False


def _mapping_value(record: Mapping, names: Sequence[str], label: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise CycleValidationError("observation is missing %s: %r" % (label, record))


def _parse_observation(
        value: Any, target_min: int, target_max: int) -> CycleObservation:
    if isinstance(value, Mapping):
        left = _mapping_value(value, ("left", "i", "source"), "left head")
        right = _mapping_value(value, ("right", "j", "target_head"), "right head")
        target = _mapping_value(value, ("target", "value", "payload"), "target")
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        left, right, target = value
    else:
        raise CycleValidationError(
            "observation must be (left, right, target) or a mapping: %r"
            % (value,))

    i, j = _check_triangle(left, right)
    target_units = _require_int(target, "target")
    if not target_min <= target_units <= target_max:
        raise CycleOverflowError(
            "target %d is outside [%d, %d]" % (target_units, target_min,
                                               target_max))
    return CycleObservation(i, j, target_units)


def _normalize_updates(
        updates: Iterable[tuple[int, int, int]],
        payload_min: int, payload_max: int) -> tuple[tuple[int, int, int], ...]:
    """Validate and sort canonical ``left < right`` pair writes."""
    try:
        iterator = iter(updates)
    except TypeError as error:
        raise CycleValidationError("updates must be an iterable") from error
    seen: dict[tuple[int, int], int] = {}
    for item in iterator:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise CycleValidationError("each update must be (left, right, value)")
        left = _check_head(item[0], "left")
        right = _check_head(item[1], "right")
        if left >= right:
            raise CycleValidationError(
                "canonical updates require left < right: %r" % (item,))
        value = _require_int(item[2], "lane value")
        if not payload_min <= value <= payload_max:
            raise CycleOverflowError(
                "lane value %d is outside [%d, %d]"
                % (value, payload_min, payload_max))
        if not payload_min <= -value <= payload_max:
            raise CycleOverflowError(
                "mirror lane %d is outside [%d, %d]"
                % (-value, payload_min, payload_max))
        key = (left, right)
        if key in seen:
            raise CycleValidationError("duplicate canonical update: %r" % (key,))
        seen[key] = value
    return tuple((left, right, seen[(left, right)]) for left, right in sorted(seen))


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------
class IntegerLaneField:
    """Deterministic in-memory raw lane field used by tests and experiments.

    Only integers are stored.  ``commit_pairs`` validates the whole update
    first, applies both mirror lanes, and advances ``generation`` once.
    """

    def __init__(self, *, payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX):
        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        if payload_min >= payload_max:
            raise CycleValidationError("payload_min must be below payload_max")
        self.payload_min = payload_min
        self.payload_max = payload_max
        self._lanes: dict[tuple[int, int], int] = {}
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def read_lane(self, source: int, target: int) -> int:
        first, second = _check_pair(source, target)
        value = self._lanes.get((first, second), 0)
        if not self.payload_min <= value <= self.payload_max:
            raise CycleOverflowError(
                "stored lane %r=%d is outside [%d, %d]"
                % ((first, second), value, self.payload_min, self.payload_max))
        return value

    def commit_pairs(self, updates: Iterable[tuple[int, int, int]]) -> int:
        normalized = _normalize_updates(
            updates, self.payload_min, self.payload_max)
        previous = {
            (left, right): self._lanes.get((left, right), 0)
            for left, right, _value in normalized
        }
        try:
            for left, right, value in normalized:
                self._lanes[(left, right)] = value
                self._lanes[(right, left)] = -value
        except Exception:
            for (left, right), value in previous.items():
                self._lanes[(left, right)] = value
                self._lanes[(right, left)] = -value
            raise
        self._generation += 1
        return self._generation

    def snapshot(self) -> tuple[dict[tuple[int, int], int], int]:
        return dict(self._lanes), self._generation

    def restore(self, snapshot: tuple[dict[tuple[int, int], int], int]) -> None:
        lanes, generation = snapshot
        self._lanes = dict(lanes)
        self._generation = _require_int(generation, "generation")

    def apply_gauge(self, gauge: Sequence[int]) -> int:
        """Apply ``x(a,b) += g(a)-g(b)`` to every unordered pair exactly."""
        values = tuple(_require_int(value, "gauge value") for value in gauge)
        if len(values) != HEAD_COUNT:
            raise CycleValidationError(
                "gauge needs %d integer values, got %d"
                % (HEAD_COUNT, len(values)))
        updates = []
        for left in range(HEAD_COUNT):
            for right in range(left + 1, HEAD_COUNT):
                value = (self._lanes.get((left, right), 0)
                         + values[left] - values[right])
                updates.append((left, right, value))
        return self.commit_pairs(updates)


class CellsLaneField:
    """Exact raw-lane adapter over a :class:`ResidentCells` region.

    Pair weights are written directly as signed Q16.16 words.  Both rails of a
    cell are updated with one shared generation, so the mirror relation is
    exact rather than merely float-close.  Head identities, phases, amplitudes
    and permanent metadata are left untouched.
    """

    def __init__(self, cells: ResidentCells, *,
                 payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX):
        if not isinstance(cells, ResidentCells):
            raise TypeError("cells must be a ResidentCells region")
        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        if payload_min >= payload_max:
            raise CycleValidationError("payload_min must be below payload_max")
        if payload_min < PAYLOAD_MIN or payload_max > PAYLOAD_MAX:
            raise CycleValidationError("payload bounds must fit symmetric i32")
        self.cells = cells
        self.payload_min = payload_min
        self.payload_max = payload_max
        self._generation = _require_int(
            getattr(cells, "generation", 0), "cells generation")

    @property
    def generation(self) -> int:
        return self._generation

    def read_lane(self, source: int, target: int) -> int:
        first, second = _check_pair(source, target)
        state = self.cells.pair_state(first, second)
        if address_of(first, second) == state.canonical:
            value = state.forward.weight
        else:
            value = state.reverse.weight
        if not self.payload_min <= value <= self.payload_max:
            raise CycleOverflowError(
                "stored lane %r=%d is outside [%d, %d]"
                % ((first, second), value, self.payload_min, self.payload_max))
        return int(value)

    def commit_pairs(self, updates: Iterable[tuple[int, int, int]]) -> int:
        normalized = _normalize_updates(
            updates, self.payload_min, self.payload_max)
        records: list[tuple[int, tuple[int, ...]]] = []
        previous: list[tuple[int, tuple[int, ...]]] = []
        cell_generation = self.cells.generation
        for left, right, value in normalized:
            state = self.cells.pair_state(left, right)
            canonical_value = (
                value if address_of(left, right) == state.canonical else -value)
            generation = (max(state.forward.generation,
                              state.reverse.generation) + 1) & 0xFFFFFFFF
            cell_generation = max(cell_generation, generation)
            forward = state.forward._replace(
                weight=canonical_value,
                generation=generation,
                flags=state.forward.flags | RAIL_FLAG_INITIALIZED)
            reverse = state.reverse._replace(
                weight=-canonical_value,
                generation=generation,
                flags=state.reverse.flags | RAIL_FLAG_INITIALIZED)
            forward_offset = self.cells.rail_record_offset(
                left, right, RAIL_S_PLUS)
            reverse_offset = self.cells.rail_record_offset(
                left, right, RAIL_S_MINUS)
            previous.append((forward_offset, tuple(state.forward)))
            previous.append((reverse_offset, tuple(state.reverse)))
            records.append((forward_offset, tuple(forward)))
            records.append((reverse_offset, tuple(reverse)))
        try:
            for offset, record in records:
                RAIL_STRUCT.pack_into(self.cells.buffer, offset, *record)
        except Exception:
            for offset, record in previous:
                RAIL_STRUCT.pack_into(self.cells.buffer, offset, *record)
            raise
        self.cells.generation = max(self.cells.generation, cell_generation)
        self._generation = max(self._generation + 1, cell_generation)
        return self._generation

    def cycle_observable(self, left: int, right: int) -> int:
        """Raw oriented triangle readout ``x(0,left)+x(left,right)+x(right,0)``."""
        i, j = _check_triangle(left, right)
        return (self.read_lane(0, i) + self.read_lane(i, j)
                + self.read_lane(j, 0))


ResidentCellsLaneField = CellsLaneField

_WORD_MASK = (1 << 32) - 1
_WORD_SIGN = 1 << 31
_WORD_MODULUS = 1 << 32


def _is_cycle_observable_field(field: Any) -> bool:
    """True for the raw-u32 ``cycle_observable.CycleField`` protocol."""
    if isinstance(field, CellsLaneField):
        return False
    return (callable(getattr(field, "get_oriented", None))
            and callable(getattr(field, "set_oriented", None))
            and hasattr(field, "generation"))


class CycleObservableLaneField:
    """Signed Q16.16 lane view over a raw-u32 ``CycleField``.

    ``cycle_observable.CycleField`` stores the same 32 bits as an unsigned
    word.  This adapter applies the two's-complement interpretation required
    by :class:`CycleTrainer` and writes both directions of each pair so the
    mirror relation is exact.  The signed symmetric payload range deliberately
    excludes ``0x80000000`` because that word is its own mirror.
    """

    def __init__(self, field: Any, *,
                 payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX):
        if not _is_cycle_observable_field(field):
            raise TypeError(
                "field must expose get_oriented/set_oriented/generation")
        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        if payload_min >= payload_max:
            raise CycleValidationError("payload_min must be below payload_max")
        if payload_min < PAYLOAD_MIN or payload_max > PAYLOAD_MAX:
            raise CycleValidationError("payload bounds must fit symmetric i32")
        self.field = field
        self.payload_min = payload_min
        self.payload_max = payload_max

    @property
    def generation(self) -> int:
        return _field_generation(self.field)

    def read_lane(self, source: int, target: int) -> int:
        first, second = _check_pair(source, target)
        word = self.field.get_oriented(first, second)
        word = _require_int(word, "raw word") & _WORD_MASK
        value = word if word < _WORD_SIGN else word - _WORD_MODULUS
        if not self.payload_min <= value <= self.payload_max:
            raise CycleOverflowError(
                "raw lane %r=%d is outside [%d, %d]"
                % ((first, second), value, self.payload_min, self.payload_max))
        return value

    def commit_pairs(self, updates: Iterable[tuple[int, int, int]]) -> int:
        normalized = _normalize_updates(
            updates, self.payload_min, self.payload_max)
        previous: dict[tuple[int, int], int] = {}
        for left, right, _value in normalized:
            previous[(left, right)] = (
                self.field.get_oriented(left, right) & _WORD_MASK)
            previous[(right, left)] = (
                self.field.get_oriented(right, left) & _WORD_MASK)
        try:
            for left, right, value in normalized:
                self.field.set_oriented(left, right, value & _WORD_MASK)
                self.field.set_oriented(
                    right, left, (-value) & _WORD_MASK)
        except Exception:
            for (left, right), word in previous.items():
                try:
                    self.field.set_oriented(left, right, word)
                except Exception:
                    pass
            raise
        return self.generation

    def cycle_observable(self, left: int, right: int) -> int:
        """Raw signed triangle readout through the adapted CycleField."""
        i, j = _check_triangle(left, right)
        return (self.read_lane(0, i) + self.read_lane(i, j)
                + self.read_lane(j, 0))


CycleFieldLaneField = CycleObservableLaneField


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class CycleTrainer:
    """Supervised fixed-point trainer for the oriented triangle readout.

    With ``normalize_gradient=False`` (the default) each of the three raw
    edges moves by ``-learning_rate * gradient`` exactly as
    ``native_mirror.cpp`` does, so the circulation moves by three times that
    amount.  Set ``normalize_gradient=True`` to distribute the same gradient
    over the three edges instead, which makes one unit-rate step move the
    circulation by exactly one gradient unit and is useful for exact
    reconstruction experiments.
    """

    def __init__(self, field: Any, *,
                 learning_rate: int | Fraction = DEFAULT_LEARNING_RATE_UNITS,
                 normalize_gradient: bool = False,
                 max_step_units: int = DEFAULT_MAX_STEP_UNITS,
                 max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                 max_pairs_per_step: int = DEFAULT_MAX_PAIRS_PER_STEP,
                 payload_min: int = PAYLOAD_MIN,
                 payload_max: int = PAYLOAD_MAX,
                 target_min: int = PAYLOAD_MIN,
                 target_max: int = PAYLOAD_MAX,
                 require_improvement: bool = True):
        if isinstance(field, ResidentCells):
            field = CellsLaneField(field)
        elif isinstance(field, (CellsLaneField, CycleObservableLaneField)):
            pass
        elif _is_cycle_observable_field(field):
            field = CycleObservableLaneField(field)
        elif isinstance(getattr(field, "cells", None), ResidentCells):
            field = CellsLaneField(field.cells)
        self.field = field
        self._read_lane = _field_reader(field)
        self._commit_pairs = _field_committer(field)
        _field_generation(field)  # fail early on a malformed field

        self.learning_rate_units = _learning_rate_units(learning_rate)
        if not isinstance(normalize_gradient, bool):
            raise CycleValidationError("normalize_gradient must be bool")
        self.normalize_gradient = normalize_gradient
        self.gradient_spread = 3 if normalize_gradient else 1
        self.max_step_units = _require_int(max_step_units, "max_step_units")
        if self.max_step_units <= 0:
            raise CycleValidationError("max_step_units must be positive")
        self.max_batch_size = _require_int(max_batch_size, "max_batch_size")
        if self.max_batch_size <= 0:
            raise CycleValidationError("max_batch_size must be positive")
        self.max_pairs_per_step = _require_int(
            max_pairs_per_step, "max_pairs_per_step")
        if self.max_pairs_per_step <= 0:
            raise CycleValidationError("max_pairs_per_step must be positive")
        payload_min = _require_int(payload_min, "payload_min")
        payload_max = _require_int(payload_max, "payload_max")
        if not PAYLOAD_MIN <= payload_min < payload_max <= PAYLOAD_MAX:
            raise CycleValidationError(
                "payload bounds must be inside symmetric int32")
        self.payload_min = payload_min
        self.payload_max = payload_max
        target_min = _require_int(target_min, "target_min")
        target_max = _require_int(target_max, "target_max")
        if target_min >= target_max:
            raise CycleValidationError("target_min must be below target_max")
        self.target_min = target_min
        self.target_max = target_max
        if not isinstance(require_improvement, bool):
            raise CycleValidationError("require_improvement must be bool")
        self.require_improvement = require_improvement
        self._snapshot_hooks = _snapshot_hooks(field)

        self.accepted_steps = 0
        self.rejected_steps = 0
        self.no_op_steps = 0
        self.observations_seen = 0
        self.updates_applied = 0
        self.last_observations: tuple[CycleObservation, ...] | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_rejection: dict[str, Any] | None = None

    # -- readout ---------------------------------------------------------
    @property
    def generation(self) -> int:
        return _field_generation(self.field)

    def _read_raw(self, source: int, target: int) -> int:
        value = self._read_lane(source, target)
        return _require_int(value, "lane value")

    def _read_edge(self, source: int, target: int) -> int:
        forward = self._read_raw(source, target)
        reverse = self._read_raw(target, source)
        if forward != -reverse:
            raise CycleMirrorError(
                "lane %r=%d is not the exact mirror of %r=%d"
                % ((source, target), forward, (target, source), reverse))
        if not self.payload_min <= forward <= self.payload_max:
            raise CycleOverflowError(
                "lane %r=%d is outside [%d, %d]"
                % ((source, target), forward, self.payload_min,
                   self.payload_max))
        if not self.payload_min <= reverse <= self.payload_max:
            raise CycleOverflowError(
                "mirror lane %r=%d is outside [%d, %d]"
                % ((target, source), reverse, self.payload_min,
                   self.payload_max))
        return forward

    def predict(self, left: int, right: int) -> int:
        """Return the raw integer circulation of triangle ``(0,left,right)``."""
        i, j = _check_triangle(left, right)
        return (self._read_edge(0, i) + self._read_edge(i, j)
                + self._read_edge(j, 0))

    # -- observation parsing --------------------------------------------
    def _parse_observations(self, observations: Any, *,
                            allow_last: bool = False) -> tuple[CycleObservation, ...]:
        if observations is None:
            if allow_last and self.last_observations is not None:
                return self.last_observations
            raise CycleValidationError("observations are required")
        if _looks_single(observations):
            return (_parse_observation(
                observations, self.target_min, self.target_max),)
        if isinstance(observations, (str, bytes, bytearray)):
            raise CycleValidationError("observations must be a sequence")
        try:
            iterator = iter(observations)
        except TypeError as error:
            raise CycleValidationError(
                "observations must be one observation or an iterable") from error
        parsed: list[CycleObservation] = []
        for item in iterator:
            parsed.append(_parse_observation(
                item, self.target_min, self.target_max))
            if len(parsed) > self.max_batch_size:
                raise CycleValidationError(
                    "batch exceeds max_batch_size=%d" % self.max_batch_size)
        if not parsed:
            raise CycleValidationError("batch must not be empty")
        return tuple(parsed)

    def _single(self, observations: Any) -> CycleObservation:
        parsed = self._parse_observations(observations)
        if len(parsed) != 1:
            raise CycleValidationError("expected exactly one observation")
        return parsed[0]

    def _coerce_single(self, left: Any, right: Any, target: Any) -> CycleObservation:
        if right is None and target is None:
            return self._single(left)
        if right is None or target is None:
            raise CycleValidationError(
                "pass an observation or all of left, right and target")
        return self._single((left, right, target))

    # -- loss and gradient ----------------------------------------------
    @staticmethod
    def _loss_units_for_residual(residual: int) -> int:
        # 0.5 * residual^2 in Q16.16; HALF_FIXED makes this exact.
        return residual * residual * HALF_FIXED

    def _residual(self, observation: CycleObservation) -> int:
        return (self.predict(observation.left, observation.right)
                - observation.target)

    def _loss_units(self, observations: Sequence[CycleObservation], *,
                    mean: bool = False) -> int:
        total = 0
        for observation in observations:
            residual = self._residual(observation)
            total += self._loss_units_for_residual(residual)
        if mean:
            total //= len(observations)
        return total

    def residual(self, left: Any, right: Any = None,
                 target: Any = None) -> int:
        """Return ``predict - target`` in raw payload units."""
        return self._residual(self._coerce_single(left, right, target))

    def gradient(self, left: Any, right: Any = None, target: Any = None) -> int:
        """Exact ``dLoss/dFlux = predict - target`` in raw payload units.

        For ``L = residual**2 / 2`` this is the complete integer gradient; no
        float approximation is involved.
        """
        return self._residual(self._coerce_single(left, right, target))

    def gradients(self, observations: Any) -> tuple[int, ...]:
        parsed = self._parse_observations(observations)
        return tuple(self._residual(observation) for observation in parsed)

    def loss_units(self, observations: Any = None, *,
                   mean: bool = False) -> int:
        """Exact Q16.16-scaled loss for one observation or a deterministic batch."""
        parsed = self._parse_observations(observations, allow_last=True)
        return self._loss_units(parsed, mean=mean)

    def loss(self, observations: Any = None, *, mean: bool = False) -> float:
        """Reporting metric only: ``0.5 * residual**2`` in payload-squared units."""
        return self.loss_units(observations, mean=mean) / FIXED_ONE

    def finite_difference_check(self, left: Any, right: Any = None,
                                target: Any = None, *,
                                delta: int = 1024) -> dict[str, Any]:
        """Check the analytic gradient with an exact integer central difference.

        The field is only read.  Changing any one of the three oriented edges
        by ``+delta`` changes the circulation by exactly ``+delta``; the
        symmetric loss difference then yields ``residual * FIXED_ONE`` with no
        rounding error for any integer ``delta``.
        """
        observation = self._coerce_single(left, right, target)
        delta = _require_int(delta, "delta")
        if delta <= 0:
            raise CycleValidationError("delta must be positive")
        flux = self.predict(observation.left, observation.right)
        residual = flux - observation.target
        loss_plus = self._loss_units_for_residual(residual + delta)
        loss_minus = self._loss_units_for_residual(residual - delta)
        finite_difference = (loss_plus - loss_minus) // (2 * delta)
        analytic = residual * FIXED_ONE
        return {
            "flux": flux,
            "target": observation.target,
            "residual": residual,
            "delta": delta,
            "loss_plus_units": loss_plus,
            "loss_minus_units": loss_minus,
            "gradient_units": analytic,
            "finite_difference_units": finite_difference,
            "error_units": finite_difference - analytic,
            "exact": finite_difference == analytic,
            "generation": self.generation,
        }

    # -- step ------------------------------------------------------------
    def _step_delta(self, gradient: int) -> int:
        if gradient == 0:
            return 0
        denominator = FIXED_ONE * self.gradient_spread
        magnitude = _round_div(
            self.learning_rate_units * abs(gradient), denominator)
        if magnitude > self.max_step_units:
            raise CycleOverflowError(
                "step magnitude %d exceeds max_step_units=%d"
                % (magnitude, self.max_step_units))
        return -magnitude if gradient > 0 else magnitude

    def step(self, observations: Any = None) -> dict[str, Any]:
        """Validate, apply and commit one deterministic fixed-point batch."""
        parsed = self._parse_observations(observations)
        generation_before = _field_generation(self.field)

        flux_before: list[int] = []
        residuals: list[int] = []
        deltas: list[int] = []
        for observation in parsed:
            flux = self.predict(observation.left, observation.right)
            gradient = flux - observation.target
            flux_before.append(flux)
            residuals.append(gradient)
            deltas.append(self._step_delta(gradient))

        pairs: set[tuple[int, int]] = set()
        for observation in parsed:
            for source, target in observation.edges:
                pairs.add((source, target) if source < target
                          else (target, source))
        old_values = {
            pair: self._read_edge(pair[0], pair[1]) for pair in pairs
        }
        aggregate = {pair: 0 for pair in pairs}
        for observation, delta in zip(parsed, deltas):
            for source, target in observation.edges:
                pair = ((source, target) if source < target
                        else (target, source))
                aggregate[pair] += delta if source < target else -delta

        new_values: dict[tuple[int, int], int] = {}
        updates: list[tuple[int, int, int]] = []
        for pair in sorted(pairs):
            value = old_values[pair] + aggregate[pair]
            if not self.payload_min <= value <= self.payload_max:
                raise CycleOverflowError(
                    "proposed lane %r=%d is outside [%d, %d]"
                    % (pair, value, self.payload_min, self.payload_max))
            new_values[pair] = value
            if value != old_values[pair]:
                updates.append((pair[0], pair[1], value))
        if len(updates) > self.max_pairs_per_step:
            raise CycleOverflowError(
                "step touches %d pairs, above max_pairs_per_step=%d"
                % (len(updates), self.max_pairs_per_step))

        flux_after: list[int] = []
        for observation in parsed:
            flux = 0
            for source, target in observation.edges:
                pair = ((source, target) if source < target
                        else (target, source))
                value = new_values[pair]
                flux += value if source < target else -value
            flux_after.append(flux)

        loss_before_units = sum(
            self._loss_units_for_residual(residual)
            for residual in residuals)
        loss_after_units = sum(
            self._loss_units_for_residual(flux - observation.target)
            for flux, observation in zip(flux_after, parsed))
        if self.require_improvement and loss_after_units > loss_before_units:
            raise CycleStepRejected(
                "step would increase loss from %d to %d fixed-point units"
                % (loss_before_units, loss_after_units))

        if not updates:
            self.accepted_steps += 1
            self.no_op_steps += 1
            self.observations_seen += len(parsed)
            self.last_observations = parsed
            result = self._receipt(
                parsed, generation_before, generation_before,
                loss_before_units, loss_after_units, residuals, deltas,
                updates=0, changed=False)
            self.last_result = result
            return result

        snapshot = None
        if self._snapshot_hooks is not None:
            snapshot = self._snapshot_hooks[0]()

        try:
            self._commit_pairs(updates)
        except Exception as error:
            if snapshot is not None:
                try:
                    self._snapshot_hooks[1](snapshot)
                except Exception:
                    pass
            raise CycleFieldError(
                "field commit failed; no step was accepted: %r" % (error,)
            ) from error

        generation_after = _field_generation(self.field)
        if generation_after <= generation_before:
            if snapshot is not None:
                try:
                    self._snapshot_hooks[1](snapshot)
                except Exception:
                    pass
            raise CycleFieldError(
                "field commit did not advance generation (%d -> %d)"
                % (generation_before, generation_after))

        actual_after_units = self._loss_units(parsed)
        if actual_after_units != loss_after_units:
            if snapshot is not None:
                try:
                    self._snapshot_hooks[1](snapshot)
                except Exception:
                    pass
            raise CycleFieldError(
                "field readback disagrees with exact proposal (%d != %d)"
                % (actual_after_units, loss_after_units))

        self.accepted_steps += 1
        self.observations_seen += len(parsed)
        self.updates_applied += len(updates)
        self.last_observations = parsed
        result = self._receipt(
            parsed, generation_before, generation_after,
            loss_before_units, actual_after_units, residuals, deltas,
            updates=len(updates), changed=True)
        self.last_result = result
        return result

    def _receipt(self, parsed: Sequence[CycleObservation],
                 generation_before: int, generation_after: int,
                 loss_before_units: int, loss_after_units: int,
                 residuals: Sequence[int], deltas: Sequence[int], *,
                 updates: int, changed: bool) -> dict[str, Any]:
        return {
            "schema": CYCLE_STEP_SCHEMA,
            "accepted": True,
            "changed": changed,
            "observations": len(parsed),
            "pairs_updated": updates,
            "generation_before": generation_before,
            "generation_after": generation_after,
            "generation": generation_after,
            "loss_units_before": loss_before_units,
            "loss_units_after": loss_after_units,
            "loss_before": loss_before_units / FIXED_ONE,
            "loss_after": loss_after_units / FIXED_ONE,
            "mean_gradient_units": sum(residuals) // len(residuals),
            "max_abs_gradient_units": max(abs(value) for value in residuals),
            "max_abs_step_units": max(abs(value) for value in deltas),
            "learning_rate_units": self.learning_rate_units,
            "normalize_gradient": self.normalize_gradient,
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
        }

    def try_step(self, observations: Any = None) -> dict[str, Any]:
        """Run :meth:`step` but report a denial instead of raising."""
        try:
            return self.step(observations)
        except CycleTrainingError as error:
            generation = _field_generation(self.field)
            rejection = {
                "schema": CYCLE_STEP_SCHEMA,
                "accepted": False,
                "changed": False,
                "reason": type(error).__name__,
                "detail": str(error),
                "generation": generation,
                "canonical_merge": CANONICAL_MERGE,
                "semantic_truth": CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
            }
            self.rejected_steps += 1
            self.last_rejection = rejection
            return rejection

    def stats(self) -> dict[str, Any]:
        return {
            "schema": CYCLE_TRAINING_SCHEMA,
            "field": type(self.field).__name__,
            "generation": self.generation,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "no_op_steps": self.no_op_steps,
            "observations_seen": self.observations_seen,
            "updates_applied": self.updates_applied,
            "learning_rate_units": self.learning_rate_units,
            "normalize_gradient": self.normalize_gradient,
            "max_step_units": self.max_step_units,
            "max_batch_size": self.max_batch_size,
            "payload_min": self.payload_min,
            "payload_max": self.payload_max,
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": CYCLE_TRAINING_IS_SEMANTIC_TRUTH,
        }


# ---------------------------------------------------------------------------
# synthetic gauge-shift experiment
# ---------------------------------------------------------------------------
def _fit_origin_scale(features: Sequence[int],
                      targets: Sequence[int]) -> int:
    """Least-squares scale through the origin, returned as exact Q16.16."""
    numerator = sum(feature * target for feature, target in zip(features,
                                                               targets))
    denominator = sum(feature * feature for feature in features)
    if denominator == 0:
        return 0
    return int(round(Fraction(numerator, denominator) * FIXED_ONE))


def _apply_scale(scale_units: int, value: int) -> int:
    return _round_div(scale_units * value, FIXED_ONE)


def synthetic_gauge_shift_experiment(
        *, samples: int = 8, seed: int = 2029) -> dict[str, Any]:
    """Train cycle and raw-edge readouts, then expose an unseen vertex gauge.

    The experiment is deliberately small and exact.  ``samples`` disjoint
    triangles are trained through :class:`CycleTrainer` with targets that are
    integer multiples of three raw payload units.  A scalar cycle readout and
    a scalar raw-edge readout are both calibrated on the same training data;
    both reconstruct every training target exactly.  A previously unseen
    integer vertex gauge is then applied to every pair lane.  The cycle readout
    is unchanged and still exact because the gauge terms telescope.  The raw
    edge readout is shifted by ``3*(g(left)-g(right))`` and is no longer exact.

    This demonstrates gauge invariance of the cycle observable, not a language
    model win.  No held-out sample or LM claim is made.
    """
    samples = _require_int(samples, "samples")
    if not 1 <= samples <= 32:
        raise CycleValidationError("samples must be in [1, 32]")
    seed = _require_int(seed, "seed")

    field = IntegerLaneField()
    trainer = CycleTrainer(
        field,
        learning_rate=FIXED_ONE,
        normalize_gradient=True,
        max_step_units=DEFAULT_MAX_STEP_UNITS,
        target_min=PAYLOAD_MIN,
        target_max=PAYLOAD_MAX,
    )

    triangles: list[tuple[int, int]] = []
    targets: list[int] = []
    for index in range(samples):
        left = 2 * index + 1
        right = 2 * index + 2
        triangles.append((left, right))
        sign = -1 if index % 2 else 1
        # Multiples of three let one exact distributed gradient step solve it.
        targets.append(sign * 3 * (index + 1) * FIXED_ONE)

    step_receipts = []
    for (left, right), target in zip(triangles, targets):
        step_receipts.append(trainer.step((left, right, target)))

    cycle_train = [trainer.predict(left, right)
                   for left, right in triangles]
    raw_train = [field.read_lane(left, right)
                 for left, right in triangles]
    cycle_scale = _fit_origin_scale(cycle_train, targets)
    raw_scale = _fit_origin_scale(raw_train, targets)
    cycle_train_predictions = [
        _apply_scale(cycle_scale, value) for value in cycle_train
    ]
    raw_train_predictions = [
        _apply_scale(raw_scale, value) for value in raw_train
    ]

    # A deterministic gauge that differs on every consecutive pair used above.
    gauge = tuple(
        (((index * 5 + seed) % 17) - 8) * FIXED_ONE
        for index in range(HEAD_COUNT))
    field.apply_gauge(gauge)

    cycle_test = [trainer.predict(left, right)
                  for left, right in triangles]
    raw_test = [field.read_lane(left, right)
                for left, right in triangles]
    cycle_test_predictions = [
        _apply_scale(cycle_scale, value) for value in cycle_test
    ]
    raw_test_predictions = [
        _apply_scale(raw_scale, value) for value in raw_test
    ]

    def exact_count(predictions: Sequence[int]) -> int:
        return sum(1 for prediction, target in zip(predictions, targets)
                   if prediction == target)

    def error_sum(predictions: Sequence[int]) -> int:
        return sum(abs(prediction - target)
                   for prediction, target in zip(predictions, targets))

    cycle_train_exact = exact_count(cycle_train_predictions)
    cycle_test_exact = exact_count(cycle_test_predictions)
    raw_train_exact = exact_count(raw_train_predictions)
    raw_test_exact = exact_count(raw_test_predictions)
    cycle_unchanged = cycle_train == cycle_test
    raw_changed = raw_train != raw_test

    return {
        "schema": CYCLE_EXPERIMENT_SCHEMA,
        "seed": seed,
        "train_samples": samples,
        "held_out_gauge_only": True,
        "cycle": {
            "scale_units": cycle_scale,
            "train_exact": cycle_train_exact,
            "test_exact": cycle_test_exact,
            "predictions_unchanged": cycle_unchanged,
            "train_error_units": error_sum(cycle_train_predictions),
            "test_error_units": error_sum(cycle_test_predictions),
            "train_predictions": list(cycle_train_predictions),
            "test_predictions": list(cycle_test_predictions),
        },
        "raw_edge": {
            "scale_units": raw_scale,
            "train_exact": raw_train_exact,
            "test_exact": raw_test_exact,
            "predictions_changed": raw_changed,
            "train_error_units": error_sum(raw_train_predictions),
            "test_error_units": error_sum(raw_test_predictions),
            "train_predictions": list(raw_train_predictions),
            "test_predictions": list(raw_test_predictions),
        },
        "targets": list(targets),
        "step_receipts": step_receipts,
        "gauge_max_abs_units": max(abs(value) for value in gauge),
        "claims": {
            "cycle_gauge_invariance": True,
            "cycle_exact_reconstruction": (
                cycle_train_exact == samples and cycle_test_exact == samples),
            "language_model_win": False,
            "held_out_samples": False,
            "canonical_merge": False,
            "semantic_truth": False,
        },
        "note": (
            "Synthetic fixed-point gauge shift only; cycle invariance is "
            "structural, not evidence of a language-model advantage."),
    }


__all__ = (
    "CYCLE_TRAINING_SCHEMA", "CYCLE_STEP_SCHEMA", "CYCLE_EXPERIMENT_SCHEMA",
    "FIXED_ONE", "HEAD_COUNT", "PAYLOAD_MIN", "PAYLOAD_MAX", "HALF_FIXED",
    "DEFAULT_LEARNING_RATE_UNITS", "DEFAULT_MAX_STEP_UNITS",
    "DEFAULT_MAX_BATCH_SIZE", "DEFAULT_MAX_PAIRS_PER_STEP",
    "CANONICAL_MERGE", "CYCLE_TRAINING_IS_SEMANTIC_TRUTH",
    "CycleTrainingError", "CycleValidationError", "CycleOverflowError",
    "CycleMirrorError", "CycleStepRejected", "CycleFieldError",
    "CycleObservation", "fixed_ratio", "IntegerLaneField", "CellsLaneField",
    "ResidentCellsLaneField", "CycleObservableLaneField",
    "CycleFieldLaneField", "CycleTrainer",
    "synthetic_gauge_shift_experiment",
)
