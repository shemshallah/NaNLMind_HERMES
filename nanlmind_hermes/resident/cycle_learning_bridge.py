"""Gauge-invariant cycle readout channel beside temporal and field learning.

This module adds a **third** learning channel without replacing either existing
channel.  The temporal model keeps its own next-token distribution and the head
field keeps its independent per-head activity.  The cycle bridge reads the
token/head activation pattern, selects a bounded set of sparse fundamental
triangles, measures their oriented flux, and commits only cycle-consistent edge
updates through a scratch-first adapter.

The channel is deliberately small and auditable:

* **Sparse integer cycles.**  A cycle is a sorted triple ``a < b < c`` encoded
  as one raw integer ``(a << 16) | (b << 8) | c``.  Edges use the raw integer
  ``(left << 8) | right``.  All proposal payloads (ids, feature values and edge
  deltas) are integers; floats appear only in final bounded readout metrics.
* **Gauge invariance.**  A cycle feature is the triangle flux
  ``x(a,b) + x(b,c) - x(a,c)``.  Under a vertex gauge
  ``x'(a,b) = x(a,b) + g[b] - g[a]`` every flux is unchanged, so gauge-shifted
  inputs produce identical cycle ids, features and bounded scores.
* **Scratch first.**  Candidate edge deltas are staged by ``cycle_field.propose``
  and only reach the adapter after every invariant passes.  A rejected or
  denied proposal is discarded without calling ``apply_proposal``.
* **Dedicated denial.**  :class:`CycleLearningDenied` is raised for non-integer
  cycle payloads and for a non-zero gauge residual.  A gauge residual is the
  maximum absolute divergence of an edge update; it is exactly zero for a sum
  of fundamental triangle cycles.
* **Bounded state.**  Active heads, active cycles, cycle-ledger size, edge
  writes per sentence and total lifetime writes are all capped.
* **No canonical merge.**  This is local evidence, never a semantic fact or a
  canonical promotion.  No merge/promotion API is imported or exposed.

Adapter protocol (duck typed)::

    edge_value(edge_id: int) -> int                 # optional but preferred
    read_edges() -> Mapping[int, int]               # fallback readout
    propose(update: Mapping[str, Any]) -> Any       # scratch staging only
    apply_proposal(proposal) -> bool | dict | None  # commit staged edges
    discard_proposal(proposal) -> bool              # optional
    gauge_residual() -> int | float                 # optional, must be zero
    edge_unit -> int                                # optional quantisation

``propose`` receives ``cycles`` (``{cycle_id: integer_feature}``) and ``edges``
(``{edge_id: integer_delta}``).  A payload whose edge deltas do not sum to a
divergence-free cycle field is denied, not silently projected.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from ..temporal_lm import BOS, EOS, fnv_head, mirror8, tokens
from .cells import FIXED_ONE, HEAD_COUNT, canonical_address

CYCLE_SCHEMA = "nanlm.resident-cycle-learning/1"
CYCLE_PROPOSAL_SCHEMA = "nanlm.resident-cycle-proposal/1"
CYCLE_ID_SCHEMA = "nanlm.resident-cycle-id/1"
EDGE_ID_SCHEMA = "nanlm.resident-cycle-edge-id/1"

DEFAULT_MAX_ACTIVE_HEADS = 48
DEFAULT_MAX_ACTIVE_CYCLES = 24
DEFAULT_MAX_TOTAL_CYCLES = 4096
DEFAULT_MAX_WRITES_PER_SENTENCE = 96
DEFAULT_TOTAL_WRITE_BUDGET = 1_000_000
DEFAULT_MAX_CYCLE_DELTA = 8
DEFAULT_CYCLE_GAIN = 1
DEFAULT_FIELD_GAIN = 0.25
DEFAULT_MIRROR_GAIN = 0.25
DEFAULT_PAIR_WINDOW = 2
DEFAULT_ACTIVE_EPSILON = 1e-9
MAX_BATCH_RECORDS = 100_000
MAX_CYCLE_POOL = 32
MAX_TOTAL_CYCLES_SETTING = 1 << 20

# Explicit declarations: cycle evidence is not canonical truth.
CANONICAL_MERGE = False
CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH = False
HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH = False


class CycleLearningError(ValueError):
    """Base error for malformed cycle payloads or adapters."""


class CycleLearningDenied(CycleLearningError):
    """Dedicated denial for non-integer payloads or a non-zero gauge residual."""

    def __init__(self, message: str, *, code: str = "denied",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "denied",
            "code": self.code,
            "error": str(self),
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# integer id and gauge helpers
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str, *, minimum: int | None = None,
                 maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleLearningDenied(
            "%s must be a raw integer: %r" % (name, value),
            code="non_integer", details={"name": name, "value": value})
    if minimum is not None and value < minimum:
        raise CycleLearningDenied(
            "%s is below %d: %r" % (name, minimum, value),
            code="out_of_range",
            details={"name": name, "value": value, "minimum": minimum})
    if maximum is not None and value > maximum:
        raise CycleLearningDenied(
            "%s is above %d: %r" % (name, maximum, value),
            code="out_of_range",
            details={"name": name, "value": value, "maximum": maximum})
    return value


def _require_head(value: Any, name: str = "head") -> int:
    return _require_int(value, name, minimum=0, maximum=HEAD_COUNT - 1)


def encode_cycle_id(first: Any, second: Any, third: Any) -> int:
    """Encode a sorted ``a < b < c`` head triple as one raw integer."""
    heads = sorted((_require_head(first, "first"),
                    _require_head(second, "second"),
                    _require_head(third, "third")))
    if len(set(heads)) != 3:
        raise CycleLearningDenied(
            "a cycle needs three distinct heads: %r" % (heads,),
            code="invalid_cycle_id", details={"heads": heads})
    a, b, c = heads
    return (a << 16) | (b << 8) | c


def decode_cycle_id(cycle_id: Any) -> tuple[int, int, int]:
    """Decode one raw integer cycle id into a sorted head triple."""
    value = _require_int(cycle_id, "cycle_id", minimum=0,
                         maximum=(HEAD_COUNT - 1) * 0x010101)
    a = (value >> 16) & 0xFF
    b = (value >> 8) & 0xFF
    c = value & 0xFF
    if not (a < b < c):
        raise CycleLearningDenied(
            "cycle id does not encode a sorted triple: %r" % (cycle_id,),
            code="invalid_cycle_id",
            details={"cycle_id": cycle_id, "heads": [a, b, c]})
    return a, b, c


def encode_edge_id(first: Any, second: Any) -> int:
    """Encode an unordered pair ``a < b`` as one raw integer edge id."""
    a = _require_head(first, "first")
    b = _require_head(second, "second")
    if a == b:
        raise CycleLearningDenied(
            "an edge needs two distinct heads: %r" % (a,),
            code="invalid_edge_id", details={"head": a})
    if a > b:
        a, b = b, a
    return (a << 8) | b


def decode_edge_id(edge_id: Any) -> tuple[int, int]:
    """Decode one raw integer edge id into a sorted head pair."""
    value = _require_int(edge_id, "edge_id", minimum=0,
                         maximum=(HEAD_COUNT - 1) * 0x0101)
    a = (value >> 8) & 0xFF
    b = value & 0xFF
    if not (a < b):
        raise CycleLearningDenied(
            "edge id does not encode two distinct sorted heads: %r"
            % (edge_id,),
            code="invalid_edge_id", details={"edge_id": edge_id,
                                             "heads": [a, b]})
    return a, b


def cycle_heads(cycle_id: Any) -> tuple[int, int, int]:
    """Alias for :func:`decode_cycle_id`."""
    return decode_cycle_id(cycle_id)


def cycle_edge_delta(cycle_id: Any, amount: Any = 1) -> dict[int, int]:
    """Return the oriented triangle edge delta for one cycle.

    The orientation is ``(a,b) + amount``, ``(b,c) + amount`` and
    ``(a,c) - amount``.  Its divergence is exactly zero, so applying it cannot
    change the vertex-gauge component of the field.
    """
    a, b, c = decode_cycle_id(cycle_id)
    scale = _require_int(amount, "amount")
    if scale == 0:
        return {}
    return {
        encode_edge_id(a, b): scale,
        encode_edge_id(b, c): scale,
        encode_edge_id(a, c): -scale,
    }


def edge_divergence(edges: Mapping[int, int]) -> list[int]:
    """Return the integer vertex divergence of a raw edge mapping."""
    if not isinstance(edges, Mapping):
        raise CycleLearningDenied(
            "edge payload must be a mapping: %r" % (edges,),
            code="invalid_proposal")
    divergence = [0] * HEAD_COUNT
    for edge_id, value in edges.items():
        left, right = decode_edge_id(edge_id)
        delta = _require_int(value, "edge delta")
        divergence[left] -= delta
        divergence[right] += delta
    return divergence


def gauge_residual(edges: Mapping[int, int]) -> int:
    """Maximum absolute edge divergence; zero means a pure cycle update."""
    divergence = edge_divergence(edges)
    return max((abs(value) for value in divergence), default=0)


def triangle_flux(edges: Mapping[int, int] | Any,
                  cycle_id: Any) -> int | float:
    """Raw oriented triangle flux ``x(a,b) + x(b,c) - x(a,c)``."""
    a, b, c = decode_cycle_id(cycle_id)

    def value(left: int, right: int) -> int | float:
        edge_id = encode_edge_id(left, right)
        if callable(edges):
            return edges(edge_id)
        if isinstance(edges, Mapping):
            raw = edges.get(edge_id, edges.get((left, right), 0))
        else:
            raw = getattr(edges, "edge_value")(edge_id)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise CycleLearningDenied(
                "edge value must be numeric: %r" % (raw,),
                code="non_integer", details={"edge_id": edge_id})
        return raw

    return value(a, b) + value(b, c) - value(a, c)


def _normalize_cycles(updates: Any) -> dict[int, int]:
    if updates is None:
        return {}
    if not isinstance(updates, Mapping):
        raise CycleLearningDenied(
            "cycles must be a mapping of raw integer cycle ids",
            code="invalid_proposal")
    result: dict[int, int] = {}
    for key, value in updates.items():
        cycle_id = decode_cycle_id(key)
        encoded = encode_cycle_id(*cycle_id)
        feature = _require_int(value, "cycle feature")
        result[encoded] = result.get(encoded, 0) + feature
    return result


def _normalize_edges(updates: Any) -> dict[int, int]:
    if updates is None:
        return {}
    if not isinstance(updates, Mapping):
        raise CycleLearningDenied(
            "edges must be a mapping of raw integer edge ids",
            code="invalid_proposal")
    result: dict[int, int] = {}
    for key, value in updates.items():
        edge_id = decode_edge_id(key)
        encoded = encode_edge_id(*edge_id)
        delta = _require_int(value, "edge delta")
        result[encoded] = result.get(encoded, 0) + delta
    return result


def _proposal_value(proposal: Any, *names: str) -> Any:
    for name in names:
        if isinstance(proposal, Mapping) and name in proposal:
            return proposal[name]
        if not isinstance(proposal, Mapping) and hasattr(proposal, name):
            return getattr(proposal, name)
    return None


def _mapping_close(first: Mapping[int, Any],
                   second: Mapping[int, Any]) -> bool:
    if set(first) != set(second):
        return False
    return all(first[key] == second[key] for key in first)


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CycleLearningDenied(
            "%s must be numeric: %r" % (name, value),
            code="non_numeric", details={"name": name, "value": value}
        ) from error
    if not math.isfinite(number):
        raise CycleLearningDenied(
            "%s must be finite: %r" % (name, value),
            code="non_finite", details={"name": name, "value": value})
    return number


def _limit(value: Any, default: int, name: str, *,
           minimum: int = 1, maximum: int = MAX_BATCH_RECORDS) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer" % name)
    return max(minimum, min(value, maximum))


def activation_entropy(values: Iterable[float]) -> float:
    """Shannon entropy in bits for a non-negative activation sequence."""
    positive = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise CycleLearningError("activation values must be finite")
        if number > 0.0:
            positive.append(number)
    total = math.fsum(positive)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in positive:
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def _snapshot_digest(values: Mapping[int, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(("%d:%r\n" % (key, values[key])).encode("ascii"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
class _CycleScratchAdapter:
    """Shared scratch staging; subclasses own the main cycle edge field."""

    edge_unit = 1

    def __init__(self, *, proposal_prefix: str = "cycle"):
        self._proposal_prefix = str(proposal_prefix)
        self._scratch: dict[str, dict[str, Any]] = {}
        self._proposal_counter = 0
        self._last_residual = 0
        self.discarded_proposals = 0

    @property
    def scratch(self) -> dict[str, dict[str, Any]]:
        """Read-only copy of staged proposals."""
        return {
            key: {
                **value,
                "cycles": dict(value.get("cycles", {})),
                "edges": dict(value.get("edges", {})),
            }
            for key, value in self._scratch.items()
        }

    def propose(self, update: Mapping[str, Any]) -> dict[str, Any]:
        self._last_residual = 0
        if not isinstance(update, Mapping):
            raise CycleLearningDenied(
                "cycle proposal must be a mapping", code="invalid_proposal")
        cycles = _normalize_cycles(
            update.get("cycles", update.get("cycle_features")))
        edges = _normalize_edges(
            update.get("edges", update.get("cycle_edges")))
        declared = update.get("gauge_residual")
        if declared is not None:
            if isinstance(declared, bool) \
                    or not isinstance(declared, (int, float)):
                raise CycleLearningDenied(
                    "declared gauge_residual must be numeric",
                    code="non_numeric", details={"gauge_residual": declared})
            if isinstance(declared, float) and not math.isfinite(declared):
                raise CycleLearningDenied(
                    "declared gauge_residual must be finite",
                    code="non_finite", details={"gauge_residual": declared})
            if declared != 0:
                self._last_residual = int(declared) \
                    if isinstance(declared, int) else declared
                raise CycleLearningDenied(
                    "cycle proposal declares a non-zero gauge residual",
                    code="gauge_residual",
                    details={"gauge_residual": declared})
        residual = gauge_residual(edges)
        if residual != 0:
            self._last_residual = residual
            raise CycleLearningDenied(
                "cycle proposal has a non-zero gauge residual",
                code="gauge_residual",
                details={"gauge_residual": residual})
        self._proposal_counter += 1
        proposal_id = "%s-%08d" % (
            self._proposal_prefix, self._proposal_counter)
        proposal = {
            "schema": CYCLE_PROPOSAL_SCHEMA,
            "proposal_id": proposal_id,
            "cycles": cycles,
            "edges": edges,
            "cycle_ids": sorted(cycles),
            "gauge_residual": residual,
            "generation": int(getattr(self, "generation", 0)),
            "snapshot_digest": update.get("snapshot_digest"),
            "sentence_digest": update.get("sentence_digest"),
            "source": update.get("source"),
        }
        self._scratch[proposal_id] = proposal
        self._last_residual = 0
        return {
            **proposal,
            "cycles": dict(cycles),
            "edges": dict(edges),
            "cycle_ids": list(proposal["cycle_ids"]),
        }

    def _resolve(self, proposal: Any) -> dict[str, Any] | None:
        if isinstance(proposal, str):
            proposal_id = proposal
        else:
            proposal_id = _proposal_value(proposal, "proposal_id", "id")
        if not isinstance(proposal_id, str):
            return None
        return self._scratch.get(proposal_id)

    def discard_proposal(self, proposal: Any) -> bool:
        staged = self._resolve(proposal)
        if staged is None:
            return False
        self._scratch.pop(staged["proposal_id"], None)
        self.discarded_proposals += 1
        self._last_residual = 0
        return True

    def gauge_residual(self) -> int:
        """Residual of the most recent staged proposal; always zero if staged."""
        return int(self._last_residual)


class CycleFieldAdapter(_CycleScratchAdapter):
    """Deterministic in-memory scratch-first cycle edge field.

    ``edges`` is a raw ``{edge_id: integer_value}`` table.  ``cycles`` stores
    accumulated raw integer cycle features.  ``propose`` only stages a copy;
    ``apply_proposal`` is the sole commit path and refuses any non-cycle-
    consistent payload.
    """

    def __init__(self, edges: Mapping[int, int] | None = None,
                 cycles: Mapping[int, int] | None = None, *,
                 proposal_prefix: str = "cycle"):
        super().__init__(proposal_prefix=proposal_prefix)
        self.edges: dict[int, int] = _normalize_edge_state(edges)
        self.cycles: dict[int, int] = _normalize_cycles(cycles)
        self.generation = 0
        self.edge_writes = 0
        self.cycle_writes = 0

    def read_edges(self) -> dict[int, int]:
        return dict(self.edges)

    def read_cycles(self) -> dict[int, int]:
        return dict(self.cycles)

    def read_all(self) -> dict[int, int]:
        return dict(self.edges)

    def edge_value(self, edge_id: Any) -> int:
        edge_id = decode_edge_id(edge_id)
        return self.edges.get(encode_edge_id(*edge_id), 0)

    def cycle_value(self, cycle_id: Any) -> int:
        cycle_id = decode_cycle_id(cycle_id)
        return self.cycles.get(encode_cycle_id(*cycle_id), 0)

    def cycle_activation(self) -> int:
        return sum(abs(value) for value in self.cycles.values())

    def field_activation(self) -> int:
        return self.cycle_activation()

    def with_gauge(self, potential: Iterable[int]) -> "CycleFieldAdapter":
        """Return the same cycle content in another integer vertex gauge."""
        shifts = list(potential)
        if len(shifts) != HEAD_COUNT:
            raise CycleLearningDenied(
                "gauge potential needs %d entries" % HEAD_COUNT,
                code="invalid_gauge",
                details={"length": len(shifts)})
        cleaned = [
            _require_int(value, "gauge potential") for value in shifts
        ]
        shifted: dict[int, int] = {}
        for edge_id, value in self.edges.items():
            left, right = decode_edge_id(edge_id)
            shifted[edge_id] = value + cleaned[right] - cleaned[left]
        result = CycleFieldAdapter(
            shifted, self.cycles, proposal_prefix=self._proposal_prefix)
        result.generation = self.generation
        return result

    def apply_proposal(self, proposal: Any) -> bool:
        staged = self._resolve(proposal)
        if staged is None:
            return False
        declared = _proposal_value(proposal, "gauge_residual")
        if declared not in (None, 0) or staged.get("gauge_residual", 0) != 0:
            residual = declared if declared not in (None, 0) \
                else staged.get("gauge_residual")
            self._last_residual = residual
            raise CycleLearningDenied(
                "refusing to commit a declared non-zero gauge residual",
                code="gauge_residual",
                details={"gauge_residual": residual})
        updates = _normalize_edges(staged.get("edges"))
        residual = gauge_residual(updates)
        if residual != 0:
            self._last_residual = residual
            raise CycleLearningDenied(
                "refusing to commit a non-zero gauge residual",
                code="gauge_residual",
                details={"gauge_residual": residual})
        for edge_id in sorted(updates):
            self.edges[edge_id] = self.edges.get(edge_id, 0) + updates[edge_id]
        cycles = _normalize_cycles(staged.get("cycles"))
        for cycle_id in sorted(cycles):
            self.cycles[cycle_id] = self.cycles.get(cycle_id, 0) + cycles[cycle_id]
        self.generation += 1
        self.edge_writes += len(updates)
        self.cycle_writes += len(cycles)
        self._scratch.pop(staged["proposal_id"], None)
        self._last_residual = 0
        return True


def _normalize_edge_state(edges: Mapping[int, int] | None) -> dict[int, int]:
    if edges is None:
        return {}
    return _normalize_edges(edges)


class ResidentCellCycleAdapter(_CycleScratchAdapter):
    """Scratch-first cycle field backed by :class:`ResidentCells` pair rails.

    Edge values are raw Q16.16 integer antisymmetric pair weights.  A committed
    delta of ``1`` adds one Q16.16 unit to the directed pair weight.  Only the
    additive rail weights are touched; permanent head identities and canonical
    pair metadata are preserved by the resident store.
    """

    edge_unit = FIXED_ONE

    def __init__(self, cells: Any, *, proposal_prefix: str = "resident-cycle"):
        for name in ("pair_state", "observe_pair"):
            if not callable(getattr(cells, name, None)):
                raise TypeError("cells is missing %s()" % name)
        super().__init__(proposal_prefix=proposal_prefix)
        self.cells = cells
        self.cycles: dict[int, int] = {}
        self.edge_writes = 0
        self.cycle_writes = 0

    @property
    def generation(self) -> int:
        return int(getattr(self.cells, "generation", 0))

    def read_cycles(self) -> dict[int, int]:
        return dict(self.cycles)

    def cycle_value(self, cycle_id: Any) -> int:
        cycle_id = decode_cycle_id(cycle_id)
        return self.cycles.get(encode_cycle_id(*cycle_id), 0)

    def cycle_activation(self) -> int:
        return sum(abs(value) for value in self.cycles.values())

    def field_activation(self) -> int:
        return self.cycle_activation()

    def edge_value(self, edge_id: Any) -> int:
        left, right = decode_edge_id(edge_id)
        state = self.cells.pair_state(left, right)
        # Orient the antisymmetric difference along left -> right.
        if state.canonical == canonical_address(left, right):
            return int(state.forward.weight) - int(state.reverse.weight)
        return int(state.reverse.weight) - int(state.forward.weight)

    def read_edges(self) -> dict[int, int]:
        values: dict[int, int] = {}
        for left in range(HEAD_COUNT):
            for right in range(left + 1, HEAD_COUNT):
                edge_id = encode_edge_id(left, right)
                values[edge_id] = self.edge_value(edge_id)
        return values

    def apply_proposal(self, proposal: Any) -> bool:
        staged = self._resolve(proposal)
        if staged is None:
            return False
        declared = _proposal_value(proposal, "gauge_residual")
        if declared not in (None, 0) or staged.get("gauge_residual", 0) != 0:
            residual = declared if declared not in (None, 0) \
                else staged.get("gauge_residual")
            self._last_residual = residual
            raise CycleLearningDenied(
                "refusing to commit a declared non-zero gauge residual",
                code="gauge_residual",
                details={"gauge_residual": residual})
        updates = _normalize_edges(staged.get("edges"))
        residual = gauge_residual(updates)
        if residual != 0:
            self._last_residual = residual
            raise CycleLearningDenied(
                "refusing to commit a non-zero gauge residual",
                code="gauge_residual",
                details={"gauge_residual": residual})
        for edge_id in sorted(updates):
            left, right = decode_edge_id(edge_id)
            delta = updates[edge_id] / float(FIXED_ONE)
            self.cells.observe_pair(left, right, weight=delta)
        cycles = _normalize_cycles(staged.get("cycles"))
        for cycle_id in sorted(cycles):
            self.cycles[cycle_id] = self.cycles.get(cycle_id, 0) + cycles[cycle_id]
        self.edge_writes += len(updates)
        self.cycle_writes += len(cycles)
        self._scratch.pop(staged["proposal_id"], None)
        self._last_residual = 0
        return True


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------
class CycleLearningBridge:
    """Add a gauge-invariant cycle channel beside temporal and field learning.

    ``field`` may be a :class:`~nanlmind_hermes.resident.field_learning.FieldLearner`
    or a bare field adapter.  When it exposes ``observe`` *and* that learner is
    driven by the same temporal instance, the bridge delegates the observation
    so the temporal model is updated exactly once.  Otherwise the bridge calls
    ``temporal.observe`` itself.  In both cases the temporal and field channels
    keep their own state; the cycle channel never rewrites them.
    """

    head_count = HEAD_COUNT

    def __init__(self, temporal: Any, field: Any, cycle_field: Any, *,
                 max_active_heads: int = DEFAULT_MAX_ACTIVE_HEADS,
                 max_active_cycles: int = DEFAULT_MAX_ACTIVE_CYCLES,
                 max_total_cycles: int = DEFAULT_MAX_TOTAL_CYCLES,
                 max_writes_per_sentence: int = DEFAULT_MAX_WRITES_PER_SENTENCE,
                 total_write_budget: int | None = DEFAULT_TOTAL_WRITE_BUDGET,
                 max_cycle_delta: int = DEFAULT_MAX_CYCLE_DELTA,
                 cycle_gain: int = DEFAULT_CYCLE_GAIN,
                 field_gain: float = DEFAULT_FIELD_GAIN,
                 mirror_gain: float = DEFAULT_MIRROR_GAIN,
                 pair_window: int = DEFAULT_PAIR_WINDOW,
                 active_epsilon: float = DEFAULT_ACTIVE_EPSILON,
                 strict: bool = True):
        for name in ("propose", "apply_proposal"):
            if not callable(getattr(cycle_field, name, None)):
                raise TypeError("cycle_field adapter is missing %s()" % name)
        if not any(callable(getattr(cycle_field, name, None))
                   for name in ("edge_value", "read_edges", "read_all")):
            raise TypeError(
                "cycle_field adapter needs edge_value(), read_edges() or read_all()")
        self.temporal = temporal
        self.field = field
        self.cycle_field = cycle_field
        self.max_active_heads = _limit(
            max_active_heads, DEFAULT_MAX_ACTIVE_HEADS, "max_active_heads",
            maximum=HEAD_COUNT)
        self.max_active_cycles = _limit(
            max_active_cycles, DEFAULT_MAX_ACTIVE_CYCLES, "max_active_cycles",
            maximum=HEAD_COUNT * (HEAD_COUNT - 1) * (HEAD_COUNT - 2) // 6)
        self.max_total_cycles = _limit(
            max_total_cycles, DEFAULT_MAX_TOTAL_CYCLES, "max_total_cycles",
            maximum=MAX_TOTAL_CYCLES_SETTING)
        self.max_writes_per_sentence = _limit(
            max_writes_per_sentence, DEFAULT_MAX_WRITES_PER_SENTENCE,
            "max_writes_per_sentence", maximum=HEAD_COUNT * HEAD_COUNT)
        if total_write_budget is not None:
            if isinstance(total_write_budget, bool) \
                    or not isinstance(total_write_budget, int):
                raise ValueError("total_write_budget must be an integer or None")
            total_write_budget = max(0, total_write_budget)
        self.total_write_budget = total_write_budget
        self.max_cycle_delta = _limit(
            max_cycle_delta, DEFAULT_MAX_CYCLE_DELTA, "max_cycle_delta",
            minimum=0, maximum=MAX_BATCH_RECORDS)
        self.cycle_gain = _limit(
            cycle_gain, DEFAULT_CYCLE_GAIN, "cycle_gain",
            minimum=0, maximum=MAX_BATCH_RECORDS)
        self.field_gain = max(0.0, _finite(field_gain, "field_gain"))
        self.mirror_gain = max(0.0, _finite(mirror_gain, "mirror_gain"))
        self.pair_window = _limit(
            pair_window, DEFAULT_PAIR_WINDOW, "pair_window",
            maximum=HEAD_COUNT)
        self.active_epsilon = max(
            0.0, _finite(active_epsilon, "active_epsilon"))
        self.strict = bool(strict)
        try:
            edge_unit = int(getattr(cycle_field, "edge_unit", 1))
        except (TypeError, ValueError) as error:
            raise ValueError("cycle_field.edge_unit must be an integer") from error
        if edge_unit <= 0:
            raise ValueError("cycle_field.edge_unit must be positive")
        self.edge_unit = edge_unit

        self.head_ledger: list[dict[str, Any]] = [
            {
                "head": head,
                "attempts": 0,
                "accepted": 0,
                "rejected": 0,
                "skipped": 0,
                "activation": 0.0,
                "loss_sum": 0.0,
                "loss_count": 0,
                "last_nll": None,
            }
            for head in range(HEAD_COUNT)
        ]
        self.cycle_ledger: dict[int, dict[str, Any]] = {}
        self.sentences = 0
        self.tokens_seen = 0
        self.collisions = 0
        self.writes = 0
        self.accepted_proposals = 0
        self.rejected_proposals = 0
        self.denials = 0
        self.budget_rejections = 0
        self.cycle_budget_rejections = 0
        self.temporal_observations = 0
        self.field_observations = 0
        self.temporal_loss_sum = 0.0
        self.temporal_loss_tokens = 0
        self._cycle_edge_state: dict[int, int] = {}
        self.last_result: dict[str, Any] | None = None
        self.last_rejection: dict[str, Any] | None = None
        self.last_denial: dict[str, Any] | None = None

    # -- base/readout helpers -------------------------------------------
    @staticmethod
    def _base_result() -> dict[str, Any]:
        return {
            "schema": CYCLE_SCHEMA,
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH,
            "cycle_evidence_is_semantic_truth":
                CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH,
            "head_activations_are_semantic_truth":
                HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH,
        }

    def _remaining_writes(self) -> int | None:
        if self.total_write_budget is None:
            return None
        return max(0, self.total_write_budget - self.writes)

    def _head_for_token(self, token: str) -> int:
        vocab = getattr(self.temporal, "vocab", None)
        if isinstance(vocab, Mapping):
            value = vocab.get(token, fnv_head(token))
        else:
            value = fnv_head(token)
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = fnv_head(token)
        return value % self.head_count

    def _mirror_head(self, head: int) -> int:
        return mirror8(head) % self.head_count

    def _token_sequence(self, text: str) -> list[tuple[str, int, bool]]:
        raw = tokens(text)
        sequence = [(BOS, self._head_for_token(BOS), True)]
        sequence.extend(
            (token, self._head_for_token(token), False) for token in raw)
        sequence.append((EOS, self._head_for_token(EOS), True))
        return sequence

    def _field_vector(self) -> tuple[float, ...]:
        owner = self.field
        read = getattr(owner, "read_all", None)
        if not callable(read):
            nested = getattr(owner, "field", None)
            read = getattr(nested, "read_all", None)
        if not callable(read):
            return (0.0,) * HEAD_COUNT
        try:
            raw = read()
        except Exception:
            return (0.0,) * HEAD_COUNT
        if isinstance(raw, Mapping):
            values = [raw.get(head, 0.0) for head in range(HEAD_COUNT)]
        else:
            try:
                values = list(raw)
            except TypeError:
                return (0.0,) * HEAD_COUNT
        if len(values) < HEAD_COUNT:
            values = values + [0.0] * (HEAD_COUNT - len(values))
        result = []
        for item in values[:HEAD_COUNT]:
            if isinstance(item, Mapping):
                item = item.get("score", item.get("activation", 0.0))
            else:
                candidate = getattr(item, "score", item)
                item = candidate
            try:
                number = float(item)
            except (TypeError, ValueError):
                number = 0.0
            result.append(number if math.isfinite(number) else 0.0)
        return tuple(result)

    def _head_weights(self, text: str) -> tuple[
            dict[int, float], int, int]:
        sequence = self._token_sequence(text)
        counts: Counter[int] = Counter()
        token_for_head: dict[int, str] = {}
        collisions = 0
        token_count = 0
        for token, head, boundary in sequence:
            if not boundary:
                token_count += 1
                previous = token_for_head.get(head)
                if previous is not None and previous != token:
                    collisions += 1
                token_for_head[head] = token
            weight = 0.5 if boundary else 1.0
            counts[head] += weight
            mirror = self._mirror_head(head)
            if mirror != head:
                counts[mirror] += self.mirror_gain * weight

        vector = self._field_vector()
        total = math.fsum(abs(value) for value in vector)
        if total > 0.0 and self.field_gain > 0.0:
            scale = max(1.0, math.fsum(counts.values()))
            for head, value in enumerate(vector):
                if value:
                    counts[head] += (
                        self.field_gain * scale * abs(value) / total)

        selected = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:self.max_active_heads]
        weights = {head: weight for head, weight in selected}
        # A triangle needs three distinct heads.  Add deterministic fallbacks.
        fallback = 0
        while len(weights) < 3 and fallback < HEAD_COUNT:
            weights.setdefault(fallback, 0.0)
            fallback += 1
        return weights, collisions, token_count

    def _select_cycles(self, head_weights: Mapping[int, float]) -> list[int]:
        heads = sorted(head_weights, key=lambda head: (-head_weights[head], head))
        if len(heads) < 3:
            return []
        anchor = 0
        pool = [head for head in heads if head != anchor]
        candidates: set[int] = set()
        # Star-anchored triangles, bounded to the strongest MAX_CYCLE_POOL heads.
        pool = pool[:MAX_CYCLE_POOL]
        for index, first in enumerate(pool):
            for second in pool[index + 1:]:
                candidates.add(encode_cycle_id(anchor, first, second))
        # Consecutive triples retain local token order structure.
        for index in range(len(heads) - 2):
            candidates.add(encode_cycle_id(*heads[index:index + 3]))
        scored = []
        for cycle_id in candidates:
            a, b, c = decode_cycle_id(cycle_id)
            score = (head_weights.get(a, 0.0)
                     + head_weights.get(b, 0.0)
                     + head_weights.get(c, 0.0))
            scored.append((-score, cycle_id))
        scored.sort()
        return [cycle_id for _score, cycle_id in
                scored[:self.max_active_cycles]]

    def _text_edge_counts(self, text: str) -> dict[int, int]:
        heads = [head for _token, head, _boundary in self._token_sequence(text)]
        counts: Counter[int] = Counter()
        for index, left in enumerate(heads):
            stop = min(len(heads), index + self.pair_window + 1)
            for other in range(index + 1, stop):
                right = heads[other]
                if left == right:
                    continue
                counts[encode_edge_id(left, right)] += 1
        return dict(counts)

    def _cycle_edge_ids(self, cycles: Iterable[int]) -> list[int]:
        edge_ids: set[int] = set()
        for cycle_id in cycles:
            a, b, c = decode_cycle_id(cycle_id)
            edge_ids.add(encode_edge_id(a, b))
            edge_ids.add(encode_edge_id(b, c))
            edge_ids.add(encode_edge_id(a, c))
        return sorted(edge_ids)

    def _coerce_edge_number(self, raw: Any, edge_id: int) -> int | float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise CycleLearningDenied(
                "adapter edge value must be numeric: %r" % (raw,),
                code="non_integer", details={"edge_id": edge_id})
        if isinstance(raw, float) and not math.isfinite(raw):
            raise CycleLearningDenied(
                "adapter edge value must be finite",
                code="non_finite", details={"edge_id": edge_id})
        return raw

    def _read_edge_snapshot(self, edge_ids: Iterable[int]) -> dict[int, int | float]:
        requested = []
        for edge_id in edge_ids:
            decode_edge_id(edge_id)
            requested.append(edge_id)
        reader = getattr(self.cycle_field, "edge_value", None)
        if callable(reader):
            snapshot = {}
            for edge_id in sorted(requested):
                snapshot[edge_id] = self._coerce_edge_number(
                    reader(edge_id), edge_id)
            return snapshot
        mapping = None
        for name in ("read_edges", "read_all"):
            read = getattr(self.cycle_field, name, None)
            if callable(read):
                mapping = read()
                break
        if not isinstance(mapping, Mapping):
            raise CycleLearningDenied(
                "cycle_field did not return an edge mapping",
                code="invalid_proposal")
        snapshot = {}
        for edge_id in sorted(requested):
            snapshot[edge_id] = self._coerce_edge_number(
                mapping.get(edge_id, 0), edge_id)
        return snapshot

    def _cycle_feature(self, cycle_id: int,
                       text_edges: Mapping[int, int],
                       snapshot: Mapping[int, int | float]) -> int | float:
        a, b, c = decode_cycle_id(cycle_id)

        def value(left: int, right: int) -> int | float:
            edge_id = encode_edge_id(left, right)
            base = snapshot.get(edge_id, 0)
            extra = text_edges.get(edge_id, 0) * self.edge_unit
            return base + extra

        return value(a, b) + value(b, c) - value(a, c)

    @staticmethod
    def _metric(value: int | float) -> int | float:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _last_token_head(self, text: str) -> int:
        raw = tokens(text)
        return self._head_for_token(raw[-1] if raw else BOS)

    def _bounded_prediction(
            self, features: Mapping[int, int | float],
            evidence: Mapping[int, int],
            next_head: int | None = None) -> tuple[dict[str, Any],
                                                   dict[int, float]]:
        if not features:
            return ({
                "head": None,
                "class": None,
                "score": 0.0,
                "probability": 0.5,
                "bounded": True,
                "next_token_head": next_head,
                "next_token_score": 0.0,
            }, {})
        scale = max(1, math.fsum(abs(value) for value in features.values()))
        class_scores: Counter[int] = Counter()
        cycle_scores: dict[int, float] = {}
        for cycle_id, feature in features.items():
            score = math.tanh(float(feature) / scale)
            if evidence.get(cycle_id, 0) < 0:
                score = -score
            cycle_scores[cycle_id] = score
            a, b, c = decode_cycle_id(cycle_id)
            class_scores[a] += score
            class_scores[b] += score
            class_scores[c] += score
        heads = sorted(class_scores)
        predicted = max(
            heads,
            key=lambda head: (class_scores[head], -head))
        denominator = max(1, len(features))
        raw = class_scores[predicted] / denominator
        bounded = max(-1.0, min(1.0, math.tanh(raw)))
        probability = 0.5 * (bounded + 1.0)
        target = predicted if next_head is None else next_head
        next_raw = class_scores.get(target, 0.0) / denominator
        next_bounded = max(-1.0, min(1.0, math.tanh(next_raw)))
        return ({
            "head": predicted,
            "class": predicted,
            "score": round(bounded, 6),
            "probability": round(probability, 6),
            "bounded": True,
            "next_token_head": target,
            "next_token_score": round(next_bounded, 6),
        }, cycle_scores)

    # -- temporal channel ----------------------------------------------
    def _temporal_losses(self, text: str) -> list[tuple[int, float]] | None:
        probability = getattr(self.temporal, "_prob", None)
        if not callable(probability):
            return None
        sequence = [BOS] + tokens(text) + [EOS]
        losses: list[tuple[int, float]] = []
        for index in range(1, len(sequence)):
            target = sequence[index]
            context2 = sequence[index - 2] if index >= 2 else None
            context1 = sequence[index - 1]
            try:
                value = float(probability(context2, context1, target))
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            value = max(value, 1e-12)
            losses.append((self._head_for_token(target), -math.log(value)))
        return losses

    def _temporal_observe(self, text: str) -> tuple[bool | None, str | None]:
        observe = getattr(self.temporal, "observe", None)
        if not callable(observe):
            return None, "missing observe()"
        try:
            return bool(observe(text)), None
        except Exception as error:
            return None, type(error).__name__

    def _observe_channels(self, text: str) -> dict[str, Any]:
        temporal_learned = None
        temporal_error = None
        field_learned = None
        field_error = None
        delegated = False
        field_observe = getattr(self.field, "observe", None)
        if callable(field_observe):
            delegates = getattr(self.field, "temporal", None) is self.temporal
            try:
                result = field_observe(text)
            except Exception as error:
                field_error = type(error).__name__
            else:
                if isinstance(result, Mapping):
                    field_learned = result.get("learned")
                    if field_learned is None:
                        field_learned = result.get("accepted")
                    if "temporal_learned" in result:
                        temporal_learned = result.get("temporal_learned")
                    if result.get("temporal_error") is not None:
                        temporal_error = result.get("temporal_error")
                else:
                    field_learned = bool(result)
                delegated = delegates
        if not delegated:
            temporal_learned, temporal_error = self._temporal_observe(text)
        if temporal_learned:
            self.temporal_observations += 1
        if field_learned:
            self.field_observations += 1
        return {
            "temporal_learned": temporal_learned,
            "temporal_error": temporal_error,
            "field_learned": field_learned,
            "field_error": field_error,
        }

    # -- invariant/denial helpers --------------------------------------
    def _adapter_gauge_residual(self) -> int | float:
        probe = getattr(self.cycle_field, "gauge_residual", None)
        if not callable(probe):
            return 0
        value = probe()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CycleLearningDenied(
                "cycle_field gauge_residual() must be numeric",
                code="non_numeric", details={"value": value})
        if isinstance(value, float) and not math.isfinite(value):
            raise CycleLearningDenied(
                "cycle_field gauge_residual() must be finite",
                code="non_finite", details={"value": value})
        return value

    def _safe_adapter_residual(self) -> int | float | None:
        try:
            return self._adapter_gauge_residual()
        except CycleLearningDenied:
            return None
        except Exception:
            return None

    def _deny(self, message: str, code: str,
              details: Mapping[str, Any] | None,
              strict: bool, result: Mapping[str, Any]) -> dict[str, Any]:
        denial = {
            "code": code,
            "error": message,
            "details": dict(details or {}),
        }
        self.denials += 1
        self.last_denial = denial
        if strict:
            raise CycleLearningDenied(
                message, code=code, details=denial["details"])
        return {
            **result,
            "learned": False,
            "accepted": False,
            "cycle_learned": False,
            "denied": True,
            "reason": code,
            "denial": denial,
        }

    @staticmethod
    def _proposal_updates(proposal: Any) -> tuple[
            dict[int, int], dict[int, int]] | None:
        cycles_raw = _proposal_value(proposal, "cycles", "cycle_features")
        edges_raw = _proposal_value(proposal, "edges", "cycle_edges")
        if cycles_raw is None and edges_raw is None:
            return None
        return _normalize_cycles(cycles_raw), _normalize_edges(edges_raw)

    def _check_invariants(
            self, submitted_cycles: Mapping[int, int],
            submitted_edges: Mapping[int, int], proposal: Any,
            before: Mapping[int, int | float], capacity: int,
            active_heads: int, active_cycles: int) -> dict[str, bool]:
        checks: dict[str, bool] = {"proposal_staged": proposal is not None}
        staged_cycles: dict[int, int] = {}
        staged_edges: dict[int, int] = {}
        staged_visible = False
        if proposal is not None:
            try:
                updates = self._proposal_updates(proposal)
            except CycleLearningDenied:
                raise
            except (TypeError, ValueError):
                checks["proposal_shape_valid"] = False
                updates = None
            else:
                checks["proposal_shape_valid"] = True
            if updates is not None:
                staged_visible = True
                staged_cycles, staged_edges = updates

        # Validate the union without double-counting an exact staging copy.
        combined_edges = dict(submitted_edges)
        for edge_id, delta in staged_edges.items():
            if combined_edges.get(edge_id) == delta:
                continue
            combined_edges[edge_id] = combined_edges.get(edge_id, 0) + delta
        combined_cycles = dict(submitted_cycles)
        for cycle_id, feature in staged_cycles.items():
            if combined_cycles.get(cycle_id) == feature:
                continue
            combined_cycles[cycle_id] = combined_cycles.get(cycle_id, 0) + feature

        checks["cycle_ids_valid"] = all(
            decode_cycle_id(cycle_id) for cycle_id in combined_cycles)
        checks["edge_ids_valid"] = all(
            decode_edge_id(edge_id) for edge_id in combined_edges)
        checks["deltas_integer"] = all(
            isinstance(delta, int) and not isinstance(delta, bool)
            for delta in combined_edges.values())
        checks["cycle_features_integer"] = all(
            isinstance(feature, int) and not isinstance(feature, bool)
            for feature in combined_cycles.values())
        checks["deltas_bounded"] = all(
            abs(delta) <= self.max_cycle_delta for delta in combined_edges.values())
        checks["head_budget"] = active_heads <= self.max_active_heads
        checks["cycle_budget"] = active_cycles <= self.max_active_cycles
        checks["write_budget"] = (
            len(combined_edges) <= capacity
            and len(combined_edges) <= self.max_writes_per_sentence)
        checks["staged_matches_submitted"] = (
            not staged_visible
            or (_mapping_close(submitted_cycles, staged_cycles)
                and _mapping_close(submitted_edges, staged_edges)))
        try:
            residual = gauge_residual(combined_edges)
            proposal_residual = _proposal_value(
                proposal, "gauge_residual", "residual") if proposal else 0
            if proposal_residual is None:
                proposal_residual = 0
            if isinstance(proposal_residual, bool) \
                    or not isinstance(proposal_residual, (int, float)):
                raise CycleLearningDenied(
                    "proposal gauge_residual must be numeric",
                    code="non_numeric", details={"value": proposal_residual})
            adapter_residual = self._adapter_gauge_residual()
            checks["gauge_residual_zero"] = (
                residual == 0 and proposal_residual == 0
                and adapter_residual == 0)
            checks["adapter_gauge_zero"] = adapter_residual == 0
        except CycleLearningDenied:
            checks["gauge_residual_zero"] = False
            checks["adapter_gauge_zero"] = False
        try:
            after = self._read_edge_snapshot(sorted(before))
            checks["main_cycle_field_unchanged"] = (
                _mapping_close(dict(before), after))
        except Exception:
            checks["main_cycle_field_unchanged"] = False
        return checks

    # -- exposure --------------------------------------------------------
    def observe(self, text: str, *, source: str | None = None,
                strict: bool | None = None) -> dict[str, Any]:
        """Observe one sentence through temporal, field and cycle channels."""
        raise_strict = self.strict if strict is None else bool(strict)
        value = str(text or "").strip()
        base = self._base_result()
        if len(value) < 2:
            return {
                **base,
                "learned": False,
                "accepted": False,
                "cycle_learned": False,
                "denied": False,
                "reason": "short",
                "temporal_learned": None,
                "field_learned": None,
                "active_heads": 0,
                "active_head_count": 0,
                "active_cycles": 0,
                "active_cycle_count": 0,
                "applied_cycles": 0,
                "gauge_residual": 0,
                "temporal_nll": None,
                "writes": 0,
            }

        self.sentences += 1
        losses = self._temporal_losses(value)
        if losses:
            for head, nll in losses:
                row = self.head_ledger[head]
                row["loss_sum"] += nll
                row["loss_count"] += 1
                row["last_nll"] = nll
            loss = (math.fsum(item for _head, item in losses) / len(losses))
            self.temporal_loss_sum += math.fsum(item for _head, item in losses)
            self.temporal_loss_tokens += len(losses)
        else:
            loss = None

        channels = self._observe_channels(value)

        try:
            head_weights, collisions, token_count = self._head_weights(value)
            active_heads = sorted(
                head_weights, key=lambda head: (-head_weights[head], head))
            cycles = self._select_cycles(head_weights)
            text_edges = self._text_edge_counts(value)
            edge_ids = self._cycle_edge_ids(cycles)
            before = self._read_edge_snapshot(edge_ids)
        except CycleLearningDenied as error:
            result = {
                **base,
                **channels,
                "learned": False,
                "accepted": False,
                "cycle_learned": False,
                "denied": True,
                "reason": error.code,
                "active_heads": 0,
                "active_head_count": 0,
                "active_cycles": 0,
                "active_cycle_count": 0,
                "applied_cycles": 0,
                "gauge_residual": 0,
                "temporal_nll": loss,
                "writes": 0,
            }
            self.last_result = result
            return self._deny(
                str(error), error.code, error.details,
                raise_strict, result)
        except Exception as error:
            result = {
                **base,
                **channels,
                "learned": False,
                "accepted": False,
                "cycle_learned": False,
                "denied": False,
                "reason": "activation_error",
                "error": type(error).__name__,
                "active_heads": 0,
                "active_head_count": 0,
                "active_cycles": 0,
                "active_cycle_count": 0,
                "applied_cycles": 0,
                "gauge_residual": 0,
                "temporal_nll": loss,
                "writes": 0,
            }
            self.last_result = result
            return result

        self.tokens_seen += token_count
        self.collisions += collisions
        features = {
            cycle_id: self._cycle_feature(cycle_id, text_edges, before)
            for cycle_id in cycles
        }
        feature_ints = {
            cycle_id: int(round(float(feature)))
            for cycle_id, feature in features.items()
        }
        previous_evidence = {
            cycle_id: self._cycle_row(cycle_id, create=False)["evidence"]
            if self._cycle_row(cycle_id, create=False) is not None else 0
            for cycle_id in cycles
        }
        prediction, cycle_scores = self._bounded_prediction(
            features, previous_evidence, self._last_token_head(value))

        for head in active_heads:
            self.head_ledger[head]["attempts"] += 1
            self.head_ledger[head]["activation"] += head_weights[head]

        remaining = self._remaining_writes()
        capacity = self.max_writes_per_sentence
        if remaining is not None:
            capacity = min(capacity, remaining)
        updates, chosen = self._build_updates(
            features, cycles, capacity)
        if not updates:
            self.budget_rejections += 1
            for cycle_id in cycles:
                row = self._cycle_row(cycle_id)
                if row is not None:
                    row["attempts"] += 1
                    row["skipped"] += 1
                    row["last_feature"] = feature_ints[cycle_id]
            result = {
                **base,
                **channels,
                "learned": False,
                "accepted": False,
                "cycle_learned": False,
                "denied": False,
                "reason": ("write_budget_exhausted" if capacity <= 0
                           else "no_cycle_signal"),
                "active_heads": len(active_heads),
                "active_head_count": len(active_heads),
                "active_head_ids": list(active_heads),
                "active_cycles": len(cycles),
                "active_cycle_count": len(cycles),
                "active_cycle_ids": list(cycles),
                "applied_cycles": 0,
                "cycle_ids": list(cycles),
                "cycle_features": {
                    str(cycle_id): self._metric(features[cycle_id])
                    for cycle_id in cycles},
                "cycles": self._cycle_rows(features, chosen, {}, cycle_scores),
                "head_updates": [
                    {"head": head, "weight": round(head_weights[head], 6)}
                    for head in active_heads
                ],
                "prediction": prediction,
                "prediction_score": prediction["score"],
                "predicted_class": prediction["class"],
                "gauge_residual": gauge_residual(self._cycle_edge_state),
                "temporal_nll": loss,
                "loss": loss,
                "loss_tokens": len(losses) if losses else 0,
                "writes": 0,
                "budget": self._budget_status(),
            }
            self.last_result = result
            return result

        submitted_cycles = {
            cycle_id: feature_ints[cycle_id] for cycle_id in chosen
        }
        residual = gauge_residual(updates)
        if residual != 0:
            result = {
                **base,
                **channels,
                "active_heads": len(active_heads),
                "active_head_count": len(active_heads),
                "active_cycles": len(cycles),
                "active_cycle_count": len(cycles),
                "applied_cycles": 0,
                "gauge_residual": residual,
                "temporal_nll": loss,
                "writes": 0,
            }
            return self._deny(
                "cycle updates are not divergence-free", "gauge_residual",
                {"gauge_residual": residual}, raise_strict, result)

        sentence_digest = hashlib.sha256(
            value.casefold().encode("utf-8")).hexdigest()
        payload = {
            "cycles": submitted_cycles,
            "edges": updates,
            "gauge_residual": 0,
            "heads": list(active_heads),
            "cycle_ids": list(chosen),
            "snapshot_digest": _snapshot_digest(before),
            "sentence_digest": sentence_digest,
            "source": source,
        }
        proposal = None
        proposal_error = None
        try:
            proposal = self.cycle_field.propose(payload)
        except CycleLearningDenied as error:
            result = {
                **base,
                **channels,
                "active_heads": len(active_heads),
                "active_head_count": len(active_heads),
                "active_cycles": len(cycles),
                "active_cycle_count": len(cycles),
                "applied_cycles": 0,
                "gauge_residual": residual,
                "temporal_nll": loss,
                "writes": 0,
            }
            self.rejected_proposals += 1
            return self._deny(
                str(error), error.code,
                {**dict(error.details),
                 "proposal_error": "CycleLearningDenied"},
                raise_strict, result)
        except Exception as error:
            proposal_error = type(error).__name__

        try:
            checks = self._check_invariants(
                submitted_cycles, updates, proposal, before, capacity,
                len(active_heads), len(cycles))
        except CycleLearningDenied as error:
            if proposal is not None:
                discard = getattr(self.cycle_field, "discard_proposal", None)
                if callable(discard):
                    try:
                        discard(proposal)
                    except Exception:
                        pass
            self.rejected_proposals += 1
            result = {
                **base,
                **channels,
                "active_heads": len(active_heads),
                "active_head_count": len(active_heads),
                "active_cycles": len(cycles),
                "active_cycle_count": len(cycles),
                "applied_cycles": 0,
                "gauge_residual": residual,
                "temporal_nll": loss,
                "writes": 0,
            }
            return self._deny(
                str(error), error.code, error.details,
                raise_strict, result)
        gauge_ok = bool(checks.get("gauge_residual_zero", False)
                        and checks.get("adapter_gauge_zero", False))
        accepted = all(checks.values())

        if not gauge_ok:
            if proposal is not None:
                discard = getattr(self.cycle_field, "discard_proposal", None)
                if callable(discard):
                    try:
                        discard(proposal)
                    except Exception:
                        pass
            for cycle_id in chosen:
                row = self._cycle_row(cycle_id, create=False)
                if row is not None:
                    row["rejected"] += 1
            for head in active_heads:
                self.head_ledger[head]["rejected"] += 1
            self.rejected_proposals += 1
            result = {
                **base,
                **channels,
                "active_heads": len(active_heads),
                "active_head_count": len(active_heads),
                "active_cycles": len(cycles),
                "active_cycle_count": len(cycles),
                "applied_cycles": 0,
                "gauge_residual": max(
                    residual,
                    int(checks.get("adapter_gauge_zero", True) is False)),
                "temporal_nll": loss,
                "writes": 0,
                "invariants": checks,
            }
            return self._deny(
                "cycle proposal failed the gauge residual invariant",
                "gauge_residual", {"invariants": checks},
                raise_strict, result)

        applied_heads = 0
        applied_coupling = 0
        apply_error = None
        applied = False
        if accepted:
            try:
                applied_result = self.cycle_field.apply_proposal(proposal)
            except CycleLearningDenied:
                applied_result = False
                apply_error = "CycleLearningDenied"
            except Exception as error:
                applied_result = False
                apply_error = type(error).__name__
            if applied_result is False or (
                    isinstance(applied_result, Mapping)
                    and applied_result.get(
                        "applied", applied_result.get("accepted", True)) is False):
                accepted = False
                checks["apply_proposal_succeeded"] = False
                apply_error = apply_error or "apply_proposal_returned_false"
            else:
                checks["apply_proposal_succeeded"] = True
                applied = True

        if applied:
            after = self._read_edge_snapshot(updates)
            expected = {
                edge_id: before.get(edge_id, 0) + delta
                for edge_id, delta in updates.items()
            }
            checks["applied_edges_match"] = _mapping_close(expected, after)
            if not checks["applied_edges_match"]:
                accepted = False
                applied = False
                apply_error = "applied_edge_mismatch"

        committed = bool(accepted and applied)
        if committed:
            self.accepted_proposals += 1
            for edge_id, delta in updates.items():
                self._cycle_edge_state[edge_id] = (
                    self._cycle_edge_state.get(edge_id, 0) + delta)
            for cycle_id in chosen:
                row = self._cycle_row(cycle_id)
                if row is not None:
                    row["attempts"] += 1
                    row["accepted"] += 1
                    row["last_feature"] = feature_ints[cycle_id]
                    row["feature_sum"] += feature_ints[cycle_id]
                    row["evidence"] += feature_ints[cycle_id]
                    row["last_score"] = cycle_scores.get(cycle_id)
            for head in active_heads:
                self.head_ledger[head]["accepted"] += 1
            applied_heads = len(active_heads)
            applied_coupling = len(updates)
            self.writes += len(updates)
        else:
            self.rejected_proposals += 1
            for cycle_id in chosen:
                row = self._cycle_row(cycle_id)
                if row is not None:
                    row["attempts"] += 1
                    row["rejected"] += 1
                    row["last_feature"] = feature_ints[cycle_id]
            for head in active_heads:
                self.head_ledger[head]["rejected"] += 1
            if proposal is not None:
                discard = getattr(self.cycle_field, "discard_proposal", None)
                if callable(discard):
                    try:
                        discard(proposal)
                    except Exception:
                        pass
            self.last_rejection = {
                "sentence_digest": sentence_digest,
                "reason": ("proposal_error" if proposal_error
                           else "apply_error" if apply_error
                           else "invariant_failure"),
                "error": proposal_error or apply_error,
                "invariants": checks,
            }

        result = {
            **base,
            **channels,
            "learned": committed,
            "accepted": committed,
            "cycle_learned": committed,
            "denied": False,
            "reason": None if committed else (
                "proposal_error" if proposal_error else
                "apply_error" if apply_error else "invariant_failure"),
            "sentence_digest": sentence_digest,
            "active_heads": len(active_heads),
            "active_head_count": len(active_heads),
            "active_head_ids": list(active_heads),
            "active_cycles": len(cycles),
            "active_cycle_count": len(cycles),
            "active_cycle_ids": list(cycles),
            "applied_cycles": len(chosen) if committed else 0,
            "cycle_ids": list(cycles),
            "cycle_features": {
                str(cycle_id): self._metric(features[cycle_id])
                for cycle_id in cycles},
            "cycles": self._cycle_rows(
                features, chosen, updates if committed else {},
                cycle_scores),
            "head_updates": [
                {"head": head, "weight": round(head_weights[head], 6)}
                for head in active_heads
            ],
            "cycle_updates": [
                {"cycle_id": cycle_id,
                 "feature": feature_ints[cycle_id],
                 "delta": (self.cycle_gain if features[cycle_id] > 0
                           else -self.cycle_gain)}
                for cycle_id in chosen
            ],
            "edge_updates": [
                {"edge_id": edge_id,
                 "left": decode_edge_id(edge_id)[0],
                 "right": decode_edge_id(edge_id)[1],
                 "delta": delta}
                for edge_id, delta in sorted(updates.items())
            ],
            "applied_heads": applied_heads,
            "applied_edges": applied_coupling,
            "prediction": prediction,
            "prediction_score": prediction["score"],
            "predicted_class": prediction["class"],
            "gauge_residual": gauge_residual(self._cycle_edge_state),
            "temporal_nll": loss,
            "loss": loss,
            "loss_tokens": len(losses) if losses else 0,
            "writes": len(updates) if committed else 0,
            "proposal": {
                "id": _proposal_value(proposal, "proposal_id", "id"),
                "staged": proposal is not None,
                "accepted": committed,
            },
            "invariants": checks,
            "budget": self._budget_status(),
        }
        self.last_result = result
        return result

    def _build_updates(
            self, features: Mapping[int, int | float],
            cycles: Iterable[int], capacity: int) -> tuple[dict[int, int],
                                                           list[int]]:
        if capacity <= 0 or self.cycle_gain <= 0 or self.max_cycle_delta <= 0:
            return {}, []
        ranked = sorted(
            (cycle_id for cycle_id in cycles if features.get(cycle_id, 0) != 0),
            key=lambda cycle_id: (-abs(features[cycle_id]), cycle_id))
        by_delta = self.max_cycle_delta // self.cycle_gain
        by_writes = capacity // 3
        limit = min(len(ranked), by_delta, by_writes)
        chosen = ranked[:max(0, limit)]
        updates: Counter[int] = Counter()
        for cycle_id in chosen:
            feature = features[cycle_id]
            sign = 1 if feature > 0 else -1
            for edge_id, delta in cycle_edge_delta(
                    cycle_id, sign * self.cycle_gain).items():
                updates[edge_id] += delta
        return dict(updates), chosen

    def _cycle_row(self, cycle_id: int, *, create: bool = True
                   ) -> dict[str, Any] | None:
        cycle_id = encode_cycle_id(*decode_cycle_id(cycle_id))
        row = self.cycle_ledger.get(cycle_id)
        if row is not None or not create:
            return row
        if len(self.cycle_ledger) >= self.max_total_cycles:
            self.cycle_budget_rejections += 1
            return None
        a, b, c = decode_cycle_id(cycle_id)
        row = {
            "cycle_id": cycle_id,
            "heads": (a, b, c),
            "attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
            "feature_sum": 0,
            "last_feature": None,
            "evidence": 0,
            "last_score": None,
        }
        self.cycle_ledger[cycle_id] = row
        return row

    def _cycle_rows(self, features: Mapping[int, int | float],
                    chosen: Iterable[int],
                    updates: Mapping[int, int],
                    scores: Mapping[int, float] | None = None
                    ) -> list[dict[str, Any]]:
        chosen_set = set(chosen)
        rows = []
        for cycle_id in sorted(features):
            a, b, c = decode_cycle_id(cycle_id)
            row = self._cycle_row(cycle_id, create=False)
            rows.append({
                "cycle_id": cycle_id,
                "heads": [a, b, c],
                "feature": self._metric(features[cycle_id]),
                "score": round(scores.get(cycle_id, 0.0), 6)
                if scores else 0.0,
                "evidence": int(row["evidence"]) if row else 0,
                "selected": cycle_id in chosen_set,
                "applied": bool(updates and cycle_id in chosen_set),
            })
        return rows

    # -- batches ---------------------------------------------------------
    @staticmethod
    def _rows(records: Iterable[Any], limit: int | None) -> list[Any]:
        if limit is None:
            return list(records)
        return list(itertools.islice(records, _limit(limit, 512, "limit")))

    @staticmethod
    def _record_text(record: Any) -> str:
        if isinstance(record, Mapping):
            value = record.get("text")
        else:
            value = record
        return str(value or "").strip()

    def observe_batch(self, records: Iterable[Any], *,
                      limit: int | None = None) -> dict[str, Any]:
        """Observe a bounded batch and aggregate cycle-channel metrics."""
        rows = self._rows(records, limit)
        learned = skipped = denied = temporal_learned = field_learned = 0
        accepted_before = self.accepted_proposals
        rejected_before = self.rejected_proposals
        writes_before = self.writes
        active_heads: set[int] = set()
        active_cycles: set[int] = set()
        cycle_counts: list[int] = []
        loss_sum = 0.0
        loss_tokens = 0
        for record in rows:
            text = self._record_text(record)
            if len(text) < 2:
                skipped += 1
                continue
            source = record.get("source") if isinstance(record, Mapping) else None
            result = self.observe(text, source=source, strict=False)
            if result.get("temporal_learned"):
                temporal_learned += 1
            if result.get("field_learned"):
                field_learned += 1
            if result.get("denied"):
                denied += 1
                continue
            if not result.get("learned"):
                skipped += 1
                continue
            learned += 1
            for item in result.get("head_updates", []):
                active_heads.add(int(item["head"]))
            if not result.get("head_updates"):
                for item in result.get("edge_updates", []):
                    left, right = decode_edge_id(item["edge_id"])
                    active_heads.add(left)
                    active_heads.add(right)
            active_cycles.update(
                int(cycle_id) for cycle_id in result.get("cycle_ids", []))
            cycle_counts.append(int(result.get("active_cycles", 0)))
            if result.get("loss") is not None and result.get("loss_tokens"):
                loss_sum += float(result["loss"]) * int(result["loss_tokens"])
                loss_tokens += int(result["loss_tokens"])
        return {
            **self._base_result(),
            "records": len(rows),
            "learned": learned,
            "skipped": skipped,
            "denied": denied,
            "temporal_learned": temporal_learned,
            "field_learned": field_learned,
            "active_heads": len(active_heads),
            "active_head_count": len(active_heads),
            "active_head_ids": sorted(active_heads),
            "active_cycles": len(active_cycles),
            "active_cycle_count": len(active_cycles),
            "active_cycle_ids": sorted(active_cycles),
            "mean_active_cycles": round(
                math.fsum(cycle_counts) / len(cycle_counts), 6)
            if cycle_counts else 0.0,
            "accepted_proposals": self.accepted_proposals - accepted_before,
            "rejected_proposals": self.rejected_proposals - rejected_before,
            "writes": self.writes - writes_before,
            "temporal_nll": round(loss_sum / loss_tokens, 6)
            if loss_tokens else None,
            "gauge_residual": gauge_residual(self._cycle_edge_state),
            "budget": self._budget_status(),
        }

    def evaluate(self, records: Iterable[Any], *,
                 limit: int | None = None) -> dict[str, Any]:
        """Read-only cycle readout plus temporal NLL.

        Evaluation never calls ``observe``, ``propose``, ``apply_proposal`` or
        ``discard_proposal``.  It reports the cycles and bounded scores the
        current channel would produce for each record.
        """
        if self.strict:
            adapter_residual = self._adapter_gauge_residual()
            if adapter_residual != 0:
                raise CycleLearningDenied(
                    "cycle field has a non-zero gauge residual",
                    code="gauge_residual",
                    details={"gauge_residual": adapter_residual})
        rows = self._rows(records, limit)
        active_heads: set[int] = set()
        active_cycles: set[int] = set()
        cycle_counts: list[int] = []
        energy: list[float] = []
        predictions: list[float] = []
        collisions = 0
        tokens_seen = 0
        loss_sum = 0.0
        loss_tokens = 0
        temporal_available = callable(getattr(self.temporal, "_prob", None))
        evaluated = skipped = 0
        for record in rows:
            text = self._record_text(record)
            if len(text) < 2:
                skipped += 1
                continue
            try:
                head_weights, record_collisions, token_count = \
                    self._head_weights(text)
                cycles = self._select_cycles(head_weights)
                text_edges = self._text_edge_counts(text)
                edge_ids = self._cycle_edge_ids(cycles)
                before = self._read_edge_snapshot(edge_ids)
            except CycleLearningDenied:
                raise
            except Exception:
                skipped += 1
                continue
            features = {
                cycle_id: self._cycle_feature(cycle_id, text_edges, before)
                for cycle_id in cycles
            }
            evidence = {}
            for cycle_id in cycles:
                row = self._cycle_row(cycle_id, create=False)
                evidence[cycle_id] = int(row["evidence"]) if row else 0
            prediction, _scores = self._bounded_prediction(
                features, evidence, self._last_token_head(text))
            evaluated += 1
            collisions += record_collisions
            tokens_seen += token_count
            active_heads.update(head_weights)
            active_cycles.update(cycles)
            cycle_counts.append(len(cycles))
            energy.append(math.fsum(abs(value) for value in features.values()))
            predictions.append(float(prediction["score"]))
            losses = self._temporal_losses(text)
            if losses is None:
                temporal_available = False
            else:
                for _head, nll in losses:
                    loss_sum += nll
                    loss_tokens += 1
        mean_nll = round(loss_sum / loss_tokens, 6) if loss_tokens else None
        return {
            **self._base_result(),
            "records": len(rows),
            "evaluated": evaluated,
            "skipped": skipped,
            "tokens": tokens_seen,
            "active_heads": len(active_heads),
            "active_head_count": len(active_heads),
            "active_head_ids": sorted(active_heads),
            "active_cycles": len(active_cycles),
            "active_cycle_count": len(active_cycles),
            "active_cycle_ids": sorted(active_cycles),
            "mean_active_cycles": round(
                math.fsum(cycle_counts) / len(cycle_counts), 6)
            if cycle_counts else 0.0,
            "max_active_cycles": max(cycle_counts) if cycle_counts else 0,
            "cycle_feature_energy": round(
                math.fsum(energy) / len(energy), 6) if energy else 0.0,
            "prediction_score": round(
                math.fsum(predictions) / len(predictions), 6)
            if predictions else 0.0,
            "collisions": collisions,
            "gauge_residual": gauge_residual(self._cycle_edge_state),
            "temporal": {
                "available": bool(temporal_available),
                "tokens": loss_tokens,
                "mean_negative_log_likelihood": mean_nll,
                "perplexity": (round(math.exp(min(mean_nll, 50.0)), 6)
                               if mean_nll is not None else None),
            },
            "temporal_nll": mean_nll,
            "budget": self._budget_status(),
        }

    # -- reporting -------------------------------------------------------
    def _budget_status(self) -> dict[str, Any]:
        remaining = self._remaining_writes()
        return {
            "max_active_heads": self.max_active_heads,
            "max_active_cycles": self.max_active_cycles,
            "max_total_cycles": self.max_total_cycles,
            "max_writes_per_sentence": self.max_writes_per_sentence,
            "max_cycle_delta": self.max_cycle_delta,
            "total_write_budget": self.total_write_budget,
            "writes": self.writes,
            "writes_remaining": remaining,
        }

    def head_ledger_rows(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.head_ledger:
            count = int(row["loss_count"])
            rows.append({
                "head": int(row["head"]),
                "attempts": int(row["attempts"]),
                "accepted": int(row["accepted"]),
                "rejected": int(row["rejected"]),
                "skipped": int(row["skipped"]),
                "activation": float(row["activation"]),
                "loss_sum": float(row["loss_sum"]),
                "loss_count": count,
                "mean_loss": (float(row["loss_sum"]) / count) if count else None,
                "last_nll": row["last_nll"],
            })
        return rows

    def cycle_ledger_rows(self) -> list[dict[str, Any]]:
        rows = []
        for cycle_id in sorted(self.cycle_ledger):
            row = self.cycle_ledger[cycle_id]
            rows.append({
                "cycle_id": int(cycle_id),
                "heads": list(row["heads"]),
                "attempts": int(row["attempts"]),
                "accepted": int(row["accepted"]),
                "rejected": int(row["rejected"]),
                "skipped": int(row["skipped"]),
                "feature_sum": int(row["feature_sum"]),
                "last_feature": row["last_feature"],
                "evidence": int(row["evidence"]),
                "last_score": row["last_score"],
            })
        return rows

    def stats(self) -> dict[str, Any]:
        """Return deterministic active-head/cycle, gauge and NLL counters."""
        vector = self._field_vector()
        field_active = {
            head for head, value in enumerate(vector)
            if value > self.active_epsilon
        }
        ledger_active_heads = {
            row["head"] for row in self.head_ledger if row["attempts"] > 0
        }
        active_heads = field_active | ledger_active_heads
        ledger_active_cycles = {
            cycle_id for cycle_id, row in self.cycle_ledger.items()
            if row["attempts"] > 0
        }
        evidence_cycles = {
            cycle_id for cycle_id, row in self.cycle_ledger.items()
            if row["evidence"] != 0
        }
        temporal_mean_nll = (
            round(self.temporal_loss_sum / self.temporal_loss_tokens, 6)
            if self.temporal_loss_tokens else None)
        total_proposals = self.accepted_proposals + self.rejected_proposals
        return {
            **self._base_result(),
            "sentences": self.sentences,
            "tokens": self.tokens_seen,
            "active_heads": len(active_heads),
            "active_head_count": len(active_heads),
            "active_head_ids": sorted(active_heads),
            "field_active_heads": len(field_active),
            "ledger_active_heads": len(ledger_active_heads),
            "active_cycles": len(ledger_active_cycles),
            "active_cycle_count": len(ledger_active_cycles),
            "active_cycle_ids": sorted(ledger_active_cycles),
            "evidence_cycles": len(evidence_cycles),
            "ledger_cycles": len(self.cycle_ledger),
            "cycle_ledger": self.cycle_ledger_rows(),
            "cycle_activation": sum(
                abs(int(row["evidence"]))
                for row in self.cycle_ledger.values()),
            "collisions": self.collisions,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "denials": self.denials,
            "budget_rejections": self.budget_rejections,
            "cycle_budget_rejections": self.cycle_budget_rejections,
            "proposal_acceptance_rate": (
                round(self.accepted_proposals / total_proposals, 6)
                if total_proposals else 0.0),
            "proposals": {
                "accepted": self.accepted_proposals,
                "rejected": self.rejected_proposals,
                "total": total_proposals,
            },
            "writes": self.writes,
            "gauge_residual": gauge_residual(self._cycle_edge_state),
            "cycle_field_gauge_residual": self._safe_adapter_residual(),
            "temporal_nll": temporal_mean_nll,
            "temporal": {
                "vocab": len(getattr(self.temporal, "vocab", {}) or {}),
                "observations": self.temporal_observations,
                "has_probability": callable(
                    getattr(self.temporal, "_prob", None)),
                "loss_tokens": self.temporal_loss_tokens,
            },
            "field": {
                "observations": self.field_observations,
                "available": callable(getattr(self.field, "read_all", None))
                or callable(getattr(getattr(self.field, "field", None),
                                    "read_all", None)),
            },
            "budget": self._budget_status(),
            "last_rejection": self.last_rejection,
            "last_denial": self.last_denial,
        }


__all__ = (
    "CYCLE_SCHEMA", "CYCLE_PROPOSAL_SCHEMA", "CYCLE_ID_SCHEMA",
    "EDGE_ID_SCHEMA", "CANONICAL_MERGE",
    "CYCLE_EVIDENCE_IS_SEMANTIC_TRUTH",
    "HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH",
    "CycleLearningError", "CycleLearningDenied",
    "CycleFieldAdapter", "ResidentCellCycleAdapter", "CycleLearningBridge",
    "encode_cycle_id", "decode_cycle_id", "cycle_heads",
    "encode_edge_id", "decode_edge_id", "cycle_edge_delta",
    "edge_divergence", "gauge_residual", "triangle_flux",
    "activation_entropy",
    "DEFAULT_MAX_ACTIVE_HEADS", "DEFAULT_MAX_ACTIVE_CYCLES",
    "DEFAULT_MAX_TOTAL_CYCLES", "DEFAULT_MAX_WRITES_PER_SENTENCE",
    "DEFAULT_TOTAL_WRITE_BUDGET", "DEFAULT_MAX_CYCLE_DELTA",
)
