"""Resonance and superposition retrieval over resident quantized vectors.

The feature store is a mutable, byte-addressed arena of compact int8/int16
rows.  A physical row (a *cell*) can be shared by one or more concept/leaf ids;
ingest composes a weighted vector into the row in place::

    cell <- decay * cell + weight * vector

so several concepts may live in one superposition without a dense per-concept
copy.  Queries scan the resident rows and keep a bounded top-k heap; they never
fall back to an exact key lookup and never materialise the complete score list.

Three score modes are provided:

``dot``
    The dequantized inner product ``sum(q_i * s_i)``.
``cosine``
    ``dot / (||q|| * ||s||)``; zero vectors score 0.0.
``phase`` (aliases ``phase_coherent``/``resonance``)
    The real phase-coherent overlap ``cosine * cos(theta_s - theta_q)``.
    Each cell has an optional global phase; the query may supply its own.
    This is classical signal processing, not a quantum claim.

Quantization uses a per-cell scale, so a cell always reconstructs its logical
amplitudes as ``stored_integer * scale``.  Offsets exposed by references are
byte offsets into the arena and remain stable across superposition updates.
"""
from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_DIMENSIONS = 128
DEFAULT_CAPACITY = 256
DEFAULT_DTYPE = "int16"
MODES = ("dot", "cosine", "phase")
FEATURE_ID = int | str

# dtype -> (item size, minimum, maximum, struct-style format character)
_DTYPE_SPECS = {
    "int8": (1, -128, 127, "b"),
    "i8": (1, -128, 127, "b"),
    "b": (1, -128, 127, "b"),
    "int16": (2, -32768, 32767, "h"),
    "i16": (2, -32768, 32767, "h"),
    "h": (2, -32768, 32767, "h"),
}
_MODE_ALIASES = {
    "dot": "dot",
    "inner": "dot",
    "inner_product": "dot",
    "cosine": "cosine",
    "cos": "cosine",
    "phase": "phase",
    "phase_coherent": "phase",
    "resonance": "phase",
}


class ResonanceError(ValueError):
    """Base error for invalid resonance-store operations."""


class ArenaFullError(ResonanceError):
    """Raised when no further cell can be allocated in the arena."""


def _dimensions(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("dimensions must be an integer")
    if value <= 0:
        raise ValueError("dimensions must be positive")
    if value > (1 << 20):
        raise ValueError("dimensions is unreasonably large")
    return value


def _capacity(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("capacity must be an integer")
    if value < 0:
        raise ValueError("capacity must be non-negative")
    return value


def _dtype(value: str | int) -> str:
    if isinstance(value, bool):
        raise TypeError("dtype must be a string or 8/16")
    if isinstance(value, int):
        value = "int%d" % value
    if not isinstance(value, str):
        raise TypeError("dtype must be a string or 8/16")
    key = value.strip().lower().replace("-", "").replace("_", "")
    if key.startswith("np."):
        key = key[3:]
    spec = _DTYPE_SPECS.get(key)
    if spec is None:
        raise ValueError("dtype must be 'int8' or 'int16'")
    return "int8" if spec[0] == 1 else "int16"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("%s must be a finite number" % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("%s must be a finite number" % name) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _non_negative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError("%s must be positive" % name)
    return result


def _phase(value: Any) -> float:
    """Normalise a global phase to [0, 2*pi)."""
    return _finite(value, "phase") % math.tau


def _feature_id(value: Any, name: str = "feature_id") -> FEATURE_ID:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("%s must be an int or str" % name)
    if isinstance(value, str) and not value:
        raise ValueError("%s must not be empty" % name)
    return value


def _id_key(value: FEATURE_ID) -> tuple[int, Any]:
    """Deterministic ordering key: numeric ids first, then lexical ids."""
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _coerce_vector(values: Iterable[float], dimensions: int,
                   name: str = "vector") -> list[float]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("%s must be a sequence of numbers" % name)
    try:
        raw = list(values)
    except TypeError as exc:
        raise TypeError("%s must be an iterable of numbers" % name) from exc
    if len(raw) != dimensions:
        raise ValueError("%s must have %d dimensions, got %d"
                         % (name, dimensions, len(raw)))
    result: list[float] = []
    for value in raw:
        if isinstance(value, bool):
            raise TypeError("%s elements must be numbers, not bool" % name)
        try:
            item = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("%s elements must be numbers" % name) from exc
        if not math.isfinite(item):
            raise ValueError("%s elements must be finite" % name)
        result.append(item)
    return result


def _top_k(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("top_k must be an integer")
    if value < 1:
        raise ValueError("top_k must be at least 1")
    return value


def _target_first(vector: Any, feature_id: Any) -> tuple[Any, Any]:
    """Accept both ``ingest(vector, id)`` and ``ingest(id, vector)`` forms."""
    if (feature_id is not None
            and isinstance(vector, (int, str)) and not isinstance(vector, bool)
            and not isinstance(feature_id, (int, str))):
        return feature_id, vector
    return vector, feature_id


def _mode(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("mode must be a string")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError("mode must be one of %s" % (", ".join(MODES),)) from exc


def _cell_index(value: Any) -> int:
    if isinstance(value, CellRef):
        value = value.cell
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cell must be an integer or CellRef")
    if value < 0:
        raise IndexError("cell index out of range")
    return value


class _Record:
    """Small mapping facet so records work as attributes or JSON-like dicts."""

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def keys(self) -> tuple[str, ...]:
        return tuple(self.__dataclass_fields__)

    def values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)

    def items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((name, getattr(self, name))
                     for name in self.__dataclass_fields__)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True)
class FeatureRef(_Record):
    """A concept/leaf view onto a physical arena cell."""

    feature_id: FEATURE_ID
    leaf_id: FEATURE_ID
    cell: int
    offset: int
    dimensions: int
    scale: float
    phase: float


@dataclass(frozen=True)
class CellRef(_Record):
    """Snapshot of a physical resident cell and the ids sharing it."""

    cell: int
    offset: int
    dimensions: int
    scale: float
    phase: float
    features: tuple[FEATURE_ID, ...]

    def as_dict(self) -> dict[str, Any]:
        data = super().as_dict()
        data["features"] = list(self.features)
        return data


@dataclass(frozen=True)
class ResonanceHit(_Record):
    """One feature id in a bounded top-k resonance result."""

    feature_id: FEATURE_ID
    leaf_id: FEATURE_ID
    score: float
    cell: int
    offset: int
    dimensions: int
    scale: float
    phase: float


class _ReverseKey:
    """Make a min-heap evict the largest deterministic tie key first."""

    __slots__ = ("key",)

    def __init__(self, key: tuple[int, Any]):
        self.key = key

    def __lt__(self, other: "_ReverseKey") -> bool:
        return self.key > other.key


@dataclass
class _Cell:
    index: int
    offset: int
    scale: float
    phase: float
    features: list[FEATURE_ID]


@dataclass(frozen=True)
class _Feature:
    feature_id: FEATURE_ID
    leaf_id: FEATURE_ID
    cell: int


class FeatureArena:
    """A fixed, mutable byte arena of quantized feature rows.

    Rows are allocated append-only.  ``offset_for_slot``/``slot_for_offset``
    convert between slot numbers and byte offsets; the offsets returned by
    :meth:`allocate` are stable for the lifetime of the arena.  A caller may
    supply any writable buffer (``bytearray`` or ``mmap``) to keep the vectors
    inside an existing resident universe.
    """

    __slots__ = (
        "dimensions", "capacity", "dtype", "item_size", "qmin", "qmax",
        "slot_bytes", "byte_size", "base_offset", "_buffer", "_view", "_used",
    )

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS,
                 capacity: int = DEFAULT_CAPACITY, dtype: str | int = DEFAULT_DTYPE,
                 buffer: Any = None, base_offset: int = 0):
        self.dimensions = _dimensions(dimensions)
        self.capacity = _capacity(capacity)
        self.dtype = _dtype(dtype)
        self.item_size, self.qmin, self.qmax, code = _DTYPE_SPECS[self.dtype]
        raw_slot = self.dimensions * self.item_size
        # Keep rows aligned for mmap-backed universes.
        self.slot_bytes = (raw_slot + 7) & ~7
        self.byte_size = self.capacity * self.slot_bytes
        if isinstance(base_offset, bool) or not isinstance(base_offset, int):
            raise TypeError("base_offset must be an integer")
        if base_offset < 0:
            raise ValueError("base_offset must be non-negative")
        if base_offset % self.item_size:
            raise ValueError("base_offset must be aligned to the item size")
        self.base_offset = base_offset
        end = base_offset + self.byte_size

        if buffer is None:
            self._buffer = bytearray(end)
            raw = memoryview(self._buffer)
        else:
            try:
                raw = memoryview(buffer)
            except TypeError as exc:
                raise TypeError("buffer must support the writable buffer protocol") from exc
            if raw.readonly:
                raise ValueError("buffer must be writable")
            if raw.format != "B":
                try:
                    raw = raw.cast("B")
                except TypeError as exc:
                    raise TypeError("buffer must be a contiguous byte buffer") from exc
            if len(raw) < end:
                raise ValueError("buffer is too small for %d bytes" % end)
            self._buffer = buffer
        segment = raw[base_offset:end]
        self._view = segment.cast(code)
        self._used = 0

    # -- allocation -----------------------------------------------------
    def offset_for_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        if slot < 0 or slot >= self.capacity:
            raise IndexError("slot out of range")
        return self.base_offset + slot * self.slot_bytes

    def slot_for_offset(self, offset: int) -> int:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("offset must be an integer")
        delta = offset - self.base_offset
        if delta < 0 or delta % self.slot_bytes:
            raise ValueError("offset is not aligned to an arena row")
        slot = delta // self.slot_bytes
        if slot >= self.capacity:
            raise IndexError("offset out of range")
        return slot

    def allocate(self) -> int:
        """Allocate the next zeroed row and return its byte offset."""
        if self._used >= self.capacity:
            raise ArenaFullError("feature arena is full (%d rows)" % self.capacity)
        slot = self._used
        start = slot * self.dimensions
        for index in range(self.dimensions):
            self._view[start + index] = 0
        self._used += 1
        return self.offset_for_slot(slot)

    def _slot(self, offset: int) -> int:
        slot = self.slot_for_offset(offset)
        if slot >= self._used:
            raise ValueError("arena offset has not been allocated")
        return slot

    # -- row access -----------------------------------------------------
    def quantized(self, offset: int) -> memoryview:
        """Read-only view of the stored integers for one allocated row."""
        slot = self._slot(offset)
        start = slot * self.dimensions
        return self._view[start:start + self.dimensions]

    def read_quantized(self, offset: int) -> list[int]:
        return list(self.quantized(offset))

    def read_vector(self, offset: int, scale: float = 1.0) -> list[float]:
        scale = _positive(scale, "scale")
        return [value * scale for value in self.quantized(offset)]

    def write_vector(self, offset: int, values: Iterable[float],
                     scale: float | None = None) -> float:
        """Quantize and write one row; return the scale that was used.

        With ``scale=None`` the row is auto-scaled to use the full integer
        range.  Values outside an explicit scale saturate to the dtype bounds.
        """
        values = _coerce_vector(values, self.dimensions, name="values")
        slot = self._slot(offset)
        if scale is None:
            peak = max((abs(value) for value in values), default=0.0)
            used_scale = (peak / self.qmax) if peak > 0.0 else 1.0
        else:
            used_scale = _positive(scale, "scale")
        start = slot * self.dimensions
        for index, value in enumerate(values):
            scaled = value / used_scale
            if scaled >= self.qmax:
                quantized = self.qmax
            elif scaled <= self.qmin:
                quantized = self.qmin
            else:
                quantized = int(round(scaled))
            self._view[start + index] = quantized
        return used_scale

    def clear(self) -> None:
        """Zero used rows and make the whole arena allocatable again."""
        for slot in range(self._used):
            start = slot * self.dimensions
            for index in range(self.dimensions):
                self._view[start + index] = 0
        self._used = 0

    def close(self) -> None:
        """Release the memoryview; an externally supplied buffer is not closed."""
        view = getattr(self, "_view", None)
        if view is not None:
            view.release()
            self._view = None

    @property
    def used(self) -> int:
        return self._used

    @property
    def free(self) -> int:
        return self.capacity - self._used

    @property
    def bytes_used(self) -> int:
        return self._used * self.slot_bytes

    @property
    def buffer(self) -> Any:
        """The backing writable buffer (for resident persistence)."""
        return self._buffer

    def stats(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "capacity": self.capacity,
            "dtype": self.dtype,
            "item_size": self.item_size,
            "slot_bytes": self.slot_bytes,
            "base_offset": self.base_offset,
            "used": self._used,
            "free": self.free,
            "bytes_used": self.bytes_used,
            "bytes_total": self.byte_size,
        }


class FeatureStore:
    """Mutable resident feature table with resonance query and ingest.

    Parameters are intentionally small and phone-friendly by default: 128
    dimensions, int16 rows, and 256 physical cells.  Pass an existing
    :class:`FeatureArena` (or a raw writable byte buffer) with ``arena=`` to
    place the table inside a resident universe.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS,
                 capacity: int = DEFAULT_CAPACITY, dtype: str | int = DEFAULT_DTYPE,
                 arena: FeatureArena | Any = None):
        if isinstance(arena, FeatureArena):
            self.arena = arena
        elif arena is not None:
            self.arena = FeatureArena(dimensions, capacity, dtype, buffer=arena)
        else:
            self.arena = FeatureArena(dimensions, capacity, dtype)
        self.dimensions = self.arena.dimensions
        self.dtype = self.arena.dtype
        self.capacity = self.arena.capacity
        self._cells: list[_Cell] = []
        self._features: dict[FEATURE_ID, _Feature] = {}
        self._next_auto_id = 0
        self._lock = threading.RLock()

    # -- small helpers --------------------------------------------------
    def _auto_feature_id(self) -> int:
        while self._next_auto_id in self._features:
            self._next_auto_id += 1
        value = self._next_auto_id
        self._next_auto_id += 1
        return value

    def _cell_at(self, index: int) -> _Cell:
        index = _cell_index(index)
        if index >= len(self._cells):
            raise IndexError("cell index out of range")
        return self._cells[index]

    def _feature_at(self, feature_id: FEATURE_ID) -> _Feature:
        try:
            return self._features[feature_id]
        except KeyError as exc:
            raise KeyError("unknown feature id %r" % (feature_id,)) from exc

    def _cell_ref(self, cell: _Cell) -> CellRef:
        return CellRef(
            cell=cell.index,
            offset=cell.offset,
            dimensions=self.dimensions,
            scale=cell.scale,
            phase=cell.phase,
            features=tuple(cell.features),
        )

    def _feature_ref(self, feature_id: FEATURE_ID) -> FeatureRef:
        record = self._feature_at(feature_id)
        cell = self._cells[record.cell]
        return FeatureRef(
            feature_id=record.feature_id,
            leaf_id=record.leaf_id,
            cell=cell.index,
            offset=cell.offset,
            dimensions=self.dimensions,
            scale=cell.scale,
            phase=cell.phase,
        )

    def _hit(self, feature_id: FEATURE_ID, score: float) -> ResonanceHit:
        record = self._feature_at(feature_id)
        cell = self._cells[record.cell]
        return ResonanceHit(
            feature_id=record.feature_id,
            leaf_id=record.leaf_id,
            score=score,
            cell=cell.index,
            offset=cell.offset,
            dimensions=self.dimensions,
            scale=cell.scale,
            phase=cell.phase,
        )

    # -- add / ingest ---------------------------------------------------
    def add(self, vector: Iterable[float], feature_id: FEATURE_ID | None = None,
            leaf_id: FEATURE_ID | None = None, *, weight: float = 1.0,
            decay: float = 1.0, phase: float | None = None,
            scale: float | None = None) -> FeatureRef:
        """Create a feature row, or superpose onto it when it already exists.

        For a new id the stored row is ``weight * vector``.  For an existing
        id the row is updated to ``decay * old + weight * vector`` (see
        :meth:`ingest`).  ``phase`` is preserved on update unless supplied.
        """
        values = _coerce_vector(vector, self.dimensions)
        weight = _finite(weight, "weight")
        decay = _non_negative(decay, "decay")
        phase_value = None if phase is None else _phase(phase)
        if scale is not None:
            _positive(scale, "scale")
        with self._lock:
            if feature_id is None:
                feature_id = self._auto_feature_id()
            else:
                feature_id = _feature_id(feature_id)
            if feature_id in self._features:
                record = self._features[feature_id]
                if leaf_id is not None:
                    leaf = _feature_id(leaf_id, "leaf_id")
                    if leaf != record.leaf_id:
                        raise ValueError(
                            "leaf_id %r does not match existing %r for feature %r"
                            % (leaf, record.leaf_id, feature_id))
                return self._ingest_cell_locked(
                    record.cell, values, weight=weight, decay=decay,
                    phase=phase_value, scale=scale, feature_id=feature_id)

            leaf = feature_id if leaf_id is None else _feature_id(leaf_id, "leaf_id")
            combined = [weight * value for value in values]
            offset = self.arena.allocate()
            used_scale = self.arena.write_vector(offset, combined, scale=scale)
            cell = _Cell(
                index=len(self._cells),
                offset=offset,
                scale=used_scale,
                phase=0.0 if phase_value is None else phase_value,
                features=[feature_id],
            )
            self._cells.append(cell)
            self._features[feature_id] = _Feature(feature_id, leaf, cell.index)
            return self._feature_ref(feature_id)

    def ingest(self, vector: Iterable[float],
               feature_id: FEATURE_ID | None = None, *,
               cell: int | CellRef | None = None, weight: float = 1.0,
               decay: float = 1.0, phase: float | None = None,
               scale: float | None = None) -> FeatureRef | CellRef:
        """Superpose a weighted vector into an existing row.

        The target is either ``feature_id`` (all ids sharing that cell observe
        the update) or an explicit ``cell``.  ``decay`` multiplies the old row
        before the weighted vector is added.  With an explicit ``scale=None``
        the composed row is re-quantized using a fresh auto scale.

        Both ``ingest(vector, feature_id)`` and ``ingest(feature_id, vector)``
        are accepted.
        """
        vector, feature_id = _target_first(vector, feature_id)
        values = _coerce_vector(vector, self.dimensions)
        weight = _finite(weight, "weight")
        decay = _non_negative(decay, "decay")
        phase_value = None if phase is None else _phase(phase)
        if scale is not None:
            _positive(scale, "scale")
        with self._lock:
            if feature_id is not None and cell is not None:
                raise ValueError("specify feature_id or cell, not both")
            if feature_id is not None:
                feature_id = _feature_id(feature_id)
                record = self._feature_at(feature_id)
                return self._ingest_cell_locked(
                    record.cell, values, weight=weight, decay=decay,
                    phase=phase_value, scale=scale, feature_id=feature_id)
            if cell is not None:
                target = self._cell_at(_cell_index(cell))
                return self._ingest_cell_locked(
                    target.index, values, weight=weight, decay=decay,
                    phase=phase_value, scale=scale, feature_id=None)
            raise ValueError("ingest requires feature_id or cell")

    def ingest_cell(self, cell: int | CellRef, vector: Iterable[float], *,
                    weight: float = 1.0, decay: float = 1.0,
                    phase: float | None = None,
                    scale: float | None = None) -> CellRef:
        """Superpose into a physical cell; every alias sees the change."""
        result = self.ingest(vector, cell=cell, weight=weight, decay=decay,
                             phase=phase, scale=scale)
        assert isinstance(result, CellRef)
        return result

    def superpose(self, vector: Iterable[float],
                  feature_id: FEATURE_ID | None = None, *,
                  cell: int | CellRef | None = None, weight: float = 1.0,
                  decay: float = 1.0, phase: float | None = None,
                  scale: float | None = None) -> FeatureRef | CellRef:
        """Create a missing feature or ingest into an existing target.

        This is the permissive counterpart to :meth:`ingest`: a call with a
        feature id that does not exist yet creates it, which keeps the
        superposition API convenient for first writes.
        """
        vector, feature_id = _target_first(vector, feature_id)
        if feature_id is not None and cell is None:
            feature_id = _feature_id(feature_id)
            with self._lock:
                exists = feature_id in self._features
            if not exists:
                return self.add(vector, feature_id=feature_id, weight=weight,
                                decay=decay, phase=phase, scale=scale)
        return self.ingest(vector, feature_id=feature_id, cell=cell,
                           weight=weight, decay=decay, phase=phase,
                           scale=scale)

    def share_cell(self, feature_id: FEATURE_ID,
                   source_feature_id: FEATURE_ID | None = None, *,
                   cell: int | CellRef | None = None,
                   leaf_id: FEATURE_ID | None = None) -> FeatureRef:
        """Attach a new concept/leaf id to an existing physical cell.

        This is how several concepts share one superposition cell.  The new
        id immediately participates in queries and in future ingests through
        the shared row.
        """
        feature_id = _feature_id(feature_id)
        if source_feature_id is None and cell is None:
            raise ValueError("share_cell requires source_feature_id or cell")
        if source_feature_id is not None and cell is not None:
            raise ValueError("specify source_feature_id or cell, not both")
        with self._lock:
            if feature_id in self._features:
                raise ValueError("feature id %r already exists" % (feature_id,))
            if source_feature_id is not None:
                source = self._feature_at(_feature_id(source_feature_id))
                target = self._cells[source.cell]
            else:
                target = self._cell_at(_cell_index(cell))
            leaf = feature_id if leaf_id is None else _feature_id(leaf_id, "leaf_id")
            target.features.append(feature_id)
            self._features[feature_id] = _Feature(feature_id, leaf, target.index)
            return self._feature_ref(feature_id)

    def attach(self, feature_id: FEATURE_ID, offset: int, *,
               leaf_id: FEATURE_ID | None = None, scale: float = 1.0,
               phase: float = 0.0) -> FeatureRef:
        """Register an already-allocated arena row as a feature cell.

        This is the bridge for a resident superblock that allocated feature
        rows before constructing the index.  ``scale`` is the row's
        dequantization scale; ``offset`` is the byte offset returned by
        :meth:`FeatureArena.allocate`.  Attaching a second id to the same
        offset shares the existing cell.
        """
        feature_id = _feature_id(feature_id)
        scale_value = _positive(scale, "scale")
        phase_value = _phase(phase)
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("offset must be an integer")
        with self._lock:
            if feature_id in self._features:
                raise ValueError("feature id %r already exists" % (feature_id,))
            slot = self.arena.slot_for_offset(offset)
            if slot >= self.arena.used:
                raise ValueError("arena offset has not been allocated")
            leaf = feature_id if leaf_id is None else _feature_id(leaf_id, "leaf_id")
            for cell in self._cells:
                if cell.offset == offset:
                    cell.features.append(feature_id)
                    self._features[feature_id] = _Feature(
                        feature_id, leaf, cell.index)
                    return self._feature_ref(feature_id)
            cell = _Cell(
                index=len(self._cells),
                offset=offset,
                scale=scale_value,
                phase=phase_value,
                features=[feature_id],
            )
            self._cells.append(cell)
            self._features[feature_id] = _Feature(feature_id, leaf, cell.index)
            return self._feature_ref(feature_id)

    def _ingest_cell_locked(self, cell_index: int,
                            values: Sequence[float], *, weight: float,
                            decay: float, phase: float | None,
                            scale: float | None,
                            feature_id: FEATURE_ID | None) -> FeatureRef | CellRef:
        cell = self._cells[cell_index]
        old = self.arena.read_vector(cell.offset, cell.scale)
        combined = [decay * previous + weight * value
                    for previous, value in zip(old, values)]
        cell.scale = self.arena.write_vector(cell.offset, combined, scale=scale)
        if phase is not None:
            cell.phase = _phase(phase)
        if feature_id is not None:
            return self._feature_ref(feature_id)
        return self._cell_ref(cell)

    # -- reads ----------------------------------------------------------
    def feature(self, feature_id: FEATURE_ID) -> FeatureRef:
        with self._lock:
            return self._feature_ref(_feature_id(feature_id))

    def cell(self, cell: int | CellRef) -> CellRef:
        with self._lock:
            return self._cell_ref(self._cell_at(_cell_index(cell)))

    def feature_ids(self) -> tuple[FEATURE_ID, ...]:
        with self._lock:
            return tuple(self._features)

    def cells(self) -> tuple[CellRef, ...]:
        with self._lock:
            return tuple(self._cell_ref(cell) for cell in self._cells)

    def read_feature(self, feature_id: FEATURE_ID) -> list[float]:
        with self._lock:
            record = self._feature_at(_feature_id(feature_id))
            cell = self._cells[record.cell]
            return self.arena.read_vector(cell.offset, cell.scale)

    def read_cell(self, cell: int | CellRef) -> list[float]:
        with self._lock:
            target = self._cell_at(_cell_index(cell))
            return self.arena.read_vector(target.offset, target.scale)

    # -- query ----------------------------------------------------------
    @staticmethod
    def _cell_score(cell: _Cell, query: Sequence[float], query_norm: float,
                    mode: str, query_phase: float,
                    arena: "FeatureArena") -> float:
        row = arena.quantized(cell.offset)
        if mode == "dot":
            raw = 0.0
            for q_value, stored in zip(query, row):
                raw += q_value * stored
            return raw * cell.scale

        raw = 0.0
        stored_norm_sq = 0.0
        for q_value, stored in zip(query, row):
            raw += q_value * stored
            stored_norm_sq += stored * stored
        if query_norm <= 0.0 or stored_norm_sq <= 0.0:
            return 0.0
        cosine = raw / (query_norm * math.sqrt(stored_norm_sq))
        # Guard against a last-bit overshoot from floating point.
        cosine = max(-1.0, min(1.0, cosine))
        if mode == "cosine":
            return cosine
        return cosine * math.cos(cell.phase - query_phase)

    def query(self, vector: Iterable[float], top_k: int = 1,
              mode: str = "dot", phase: float = 0.0) -> list[ResonanceHit]:
        """Return the bounded top-k features by resident resonance score.

        Scores are scanned directly from the arena rows.  Only ``top_k``
        candidates are retained in a heap, so the score vector is never
        materialised.  Ties are broken by ascending feature id (numeric ids
        before lexical ids) and the result is stable across runs.
        """
        values = _coerce_vector(vector, self.dimensions)
        top_k = _top_k(top_k)
        mode = _mode(mode)
        query_phase = 0.0 if phase is None else _phase(phase)
        query_norm = math.sqrt(sum(value * value for value in values))
        with self._lock:
            if not self._cells:
                return []
            heap: list[tuple[float, _ReverseKey, FEATURE_ID]] = []
            for cell in self._cells:
                score = self._cell_score(cell, values, query_norm, mode,
                                         query_phase, self.arena)
                for feature_id in cell.features:
                    tie = _id_key(feature_id)
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, _ReverseKey(tie), feature_id))
                    elif (score > heap[0][0] or
                          (score == heap[0][0] and tie < heap[0][1].key)):
                        heapq.heapreplace(heap, (score, _ReverseKey(tie), feature_id))
            ordered = sorted(heap, key=lambda item: (-item[0], item[1].key))
            return [self._hit(feature_id, score)
                    for score, _, feature_id in ordered]

    # -- diagnostics ----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            data = self.arena.stats()
            data.update({
                "cells": len(self._cells),
                "features": len(self._features),
            })
            return data

    def __len__(self) -> int:
        with self._lock:
            return len(self._features)

    def __contains__(self, feature_id: object) -> bool:
        with self._lock:
            return feature_id in self._features

    def __iter__(self) -> Iterator[FeatureRef]:
        for feature_id in self.feature_ids():
            yield self.feature(feature_id)

    def __repr__(self) -> str:
        return ("FeatureStore(dimensions=%d, dtype=%r, cells=%d, features=%d)"
                % (self.dimensions, self.dtype, len(self._cells),
                   len(self._features)))


# The names below make the intended resident roles explicit without adding
# parallel implementations.
ResidentArena = FeatureArena
ResonanceStore = FeatureStore


def query(store: FeatureStore, vector: Iterable[float], top_k: int = 1,
          mode: str = "dot", phase: float = 0.0) -> list[ResonanceHit]:
    """Module-level convenience wrapper for :meth:`FeatureStore.query`."""
    if not isinstance(store, FeatureStore):
        raise TypeError("store must be a FeatureStore")
    return store.query(vector, top_k=top_k, mode=mode, phase=phase)


def superpose(store: FeatureStore, vector: Iterable[float],
              feature_id: FEATURE_ID | None = None, **kwargs: Any
              ) -> FeatureRef | CellRef:
    """Module-level convenience wrapper for :meth:`FeatureStore.superpose`."""
    if not isinstance(store, FeatureStore):
        raise TypeError("store must be a FeatureStore")
    return store.superpose(vector, feature_id, **kwargs)


__all__ = (
    "DEFAULT_DIMENSIONS", "DEFAULT_CAPACITY", "DEFAULT_DTYPE", "MODES",
    "FEATURE_ID", "ResonanceError", "ArenaFullError", "FeatureArena",
    "ResidentArena", "FeatureStore", "ResonanceStore", "FeatureRef",
    "CellRef", "ResonanceHit", "query", "superpose",
)
