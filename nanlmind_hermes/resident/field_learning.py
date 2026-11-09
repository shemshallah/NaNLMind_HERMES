"""Independent 256-head field learning beside :class:`TemporalMirrorLM`.

This module deliberately adds a *second* learning channel instead of replacing
the temporal word predictor.  The temporal model remains responsible for its
own next-token distribution.  The head field learns a bounded, per-head
activation pattern plus a sparse coupling matrix for every exposed sentence.

The design has three hard boundaries:

* **Scratch first.**  A sentence is first turned into per-head deltas and
  coupling deltas.  Those deltas are handed to the adapter's ``propose`` method
  and only reach the main field after every Boolean/consistency invariant
  passes.  Rejected proposals are discarded without calling ``activate``.
* **Bounded writes.**  The number of head updates, coupling updates, per-head
  magnitude and lifetime writes are all capped.  Nothing is written outside
  those caps, even when the adapter stages a larger proposal.
* **No canonical merge, no truth claim.**  This module never imports or calls a
  canonical merge gate, and an activated head is evidence about exposure, not a
  semantic fact.  The result dictionaries carry explicit
  ``canonical_merge``/``semantic_truth`` false flags.

Adapter protocol (duck typed, no inheritance required)::

    activate(head: int, amount: float) -> float       # independent head write
    read_all() -> Sequence[float] | Mapping[int, float]
    field_activation() -> float
    propose(update: Mapping[str, Any]) -> Any         # scratch staging only
    apply_proposal(proposal) -> bool | dict | None    # commit field coupling

``propose`` receives a mapping with ``heads`` (``{head: delta}``) and
``coupling`` (``{(left, right): delta}``) entries.  ``apply_proposal`` is
expected to commit the staged *field coupling*; accepted head deltas are
committed independently through ``activate``.  A custom adapter may also expose
``discard_proposal``; when present it is called for rejected proposals.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from ..temporal_lm import BOS, EOS, fnv_head, mirror8, tokens
from .cells import HEAD_COUNT, ResidentCells

FIELD_SCHEMA = "nanlm.resident-field-learning/1"
FIELD_PROPOSAL_SCHEMA = "nanlm.resident-field-proposal/1"

DEFAULT_MAX_ACTIVE_HEADS = 64
DEFAULT_MAX_COUPLINGS = 128
DEFAULT_MAX_WRITES_PER_SENTENCE = 512
DEFAULT_TOTAL_WRITE_BUDGET = 1_000_000
DEFAULT_MAX_HEAD_DELTA = 4.0
DEFAULT_MAX_COUPLING_DELTA = 2.0
DEFAULT_HEAD_GAIN = 1.0
DEFAULT_COUPLING_GAIN = 0.5
DEFAULT_MIRROR_GAIN = 0.25
DEFAULT_PAIR_WINDOW = 2
DEFAULT_ACTIVE_EPSILON = 1e-9
MAX_BATCH_RECORDS = 100_000

# Explicit declarations: activation is learned evidence, not canonical truth.
CANONICAL_MERGE = False
HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH = False


class FieldLearningError(ValueError):
    """Raised for malformed adapters, proposals or bounded-update requests."""


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------
def activation_entropy(values: Iterable[float]) -> float:
    """Shannon entropy (bits) of a non-negative head-activation vector."""
    positive = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise FieldLearningError("activation values must be finite")
        if number > 0.0:
            positive.append(number)
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in positive:
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def _vector_digest(values: Iterable[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(("%.12g\n" % float(value)).encode("ascii"))
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FieldLearningError("%s must be numeric: %r" % (name, value)) from error
    if not math.isfinite(number):
        raise FieldLearningError("%s must be finite: %r" % (name, value))
    return number


def _head_index(value: Any) -> int:
    if isinstance(value, bool):
        raise FieldLearningError("head index must be an integer, not bool")
    try:
        head = int(value)
    except (TypeError, ValueError) as error:
        raise FieldLearningError("head index must be an integer: %r" % (value,)) from error
    if not 0 <= head < HEAD_COUNT:
        raise FieldLearningError(
            "head index must be in [0, %d): %r" % (HEAD_COUNT, value))
    return head


def _pair_key(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        separator = ":" if ":" in value else ","
        parts = value.split(separator)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        parts = list(value)
    else:
        raise FieldLearningError("coupling key must be a pair: %r" % (value,))
    left, right = _head_index(parts[0]), _head_index(parts[1])
    if left == right:
        raise FieldLearningError("a coupling pair needs two distinct heads")
    return (left, right) if left < right else (right, left)


def _normalize_heads(updates: Any) -> dict[int, float]:
    if updates is None:
        return {}
    items = updates.items() if isinstance(updates, Mapping) else updates
    result: dict[int, float] = {}
    try:
        iterator = iter(items)
    except TypeError as error:
        raise FieldLearningError("head updates must be a mapping or pairs") from error
    for key, value in iterator:
        head = _head_index(key)
        delta = _finite(value, "head delta")
        if delta < 0.0:
            raise FieldLearningError("head deltas must be non-negative")
        result[head] = result.get(head, 0.0) + delta
    return result


def _normalize_coupling(updates: Any) -> dict[tuple[int, int], float]:
    if updates is None:
        return {}
    items = updates.items() if isinstance(updates, Mapping) else updates
    result: dict[tuple[int, int], float] = {}
    try:
        iterator = iter(items)
    except TypeError as error:
        raise FieldLearningError(
            "coupling updates must be a mapping or pairs") from error
    for key, value in iterator:
        pair = _pair_key(key)
        delta = _finite(value, "coupling delta")
        result[pair] = result.get(pair, 0.0) + delta
    return result


def _proposal_value(proposal: Any, *names: str) -> Any:
    for name in names:
        if isinstance(proposal, Mapping) and name in proposal:
            return proposal[name]
        if not isinstance(proposal, Mapping) and hasattr(proposal, name):
            return getattr(proposal, name)
    return None


def _mapping_close(first: Mapping[Any, float],
                   second: Mapping[Any, float]) -> bool:
    if set(first) != set(second):
        return False
    return all(abs(float(first[key]) - float(second[key])) <= 1e-9
               for key in first)


def _coerce_vector(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, Mapping):
        values = [float(raw.get(head, 0.0)) for head in range(HEAD_COUNT)]
    else:
        try:
            values = [float(value) for value in raw]
        except TypeError as error:
            raise FieldLearningError("read_all() must return a sequence") from error
    if len(values) != HEAD_COUNT:
        raise FieldLearningError(
            "read_all() must return %d values, got %d" % (HEAD_COUNT, len(values)))
    if any(not math.isfinite(value) for value in values):
        raise FieldLearningError("read_all() returned a non-finite activation")
    return tuple(values)


def _coerce_scalar(raw: Any) -> float:
    if isinstance(raw, Mapping):
        raw = sum(float(value) for value in raw.values())
    elif isinstance(raw, (list, tuple)):
        raw = sum(float(value) for value in raw)
    value = _finite(raw, "field activation")
    return value


def _limit(value: Any, default: int, name: str,
           maximum: int = MAX_BATCH_RECORDS) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("%s must be an integer" % name)
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be an integer" % name) from error
    return max(1, min(number, maximum))


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
class _ScratchAdapter:
    """Shared bounded scratch staging; subclasses own the main field."""

    def __init__(self, *, proposal_prefix: str = "field"):
        self._proposal_prefix = str(proposal_prefix)
        self._scratch: dict[str, dict[str, Any]] = {}
        self._proposal_counter = 0
        self.discarded_proposals = 0

    @property
    def scratch(self) -> dict[str, dict[str, Any]]:
        """Read-only view of staged proposals (copy of the mapping)."""
        return {key: dict(value) for key, value in self._scratch.items()}

    def propose(self, update: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(update, Mapping):
            raise FieldLearningError("proposal payload must be a mapping")
        heads = _normalize_heads(update.get("heads"))
        coupling = _normalize_coupling(update.get("coupling"))
        self._proposal_counter += 1
        proposal_id = "%s-%08d" % (self._proposal_prefix, self._proposal_counter)
        proposal = {
            "schema": FIELD_PROPOSAL_SCHEMA,
            "proposal_id": proposal_id,
            "heads": heads,
            "coupling": coupling,
            "generation": int(getattr(self, "generation", 0)),
            "snapshot_digest": update.get("snapshot_digest"),
            "sentence_digest": update.get("sentence_digest"),
            "field": _finite(update.get("field", 0.0), "field delta"),
        }
        self._scratch[proposal_id] = proposal
        return dict(proposal)

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
        return True


class HeadFieldAdapter(_ScratchAdapter):
    """Deterministic in-memory adapter used for tests and lightweight use.

    Head activations are independent scalars.  Field coupling is a sparse
    symmetric table keyed by sorted head pairs.  ``propose`` only copies values
    into scratch; ``apply_proposal`` commits coupling, while ``activate``
    commits one head at a time.
    """

    def __init__(self, *, proposal_prefix: str = "field"):
        super().__init__(proposal_prefix=proposal_prefix)
        self.activations = [0.0] * HEAD_COUNT
        self.coupling: dict[tuple[int, int], float] = {}
        self.generation = 0
        self.activation_writes = 0
        self.coupling_writes = 0

    def activate(self, head: int, amount: float = 1.0) -> float:
        index = _head_index(head)
        delta = _finite(amount, "activation amount")
        if delta < 0.0:
            raise FieldLearningError("activation amount must be non-negative")
        self.activations[index] += delta
        self.generation += 1
        self.activation_writes += 1
        return self.activations[index]

    def read_all(self) -> tuple[float, ...]:
        return tuple(self.activations)

    def field_activation(self) -> float:
        return float(sum(abs(value) for value in self.coupling.values()))

    def coupling_weight(self, left: int, right: int) -> float:
        pair = (min(_head_index(left), _head_index(right)),
                max(_head_index(left), _head_index(right)))
        if pair[0] == pair[1]:
            return 0.0
        return float(self.coupling.get(pair, 0.0))

    def apply_proposal(self, proposal: Any) -> bool:
        staged = self._resolve(proposal)
        if staged is None:
            return False
        coupling = _normalize_coupling(staged.get("coupling"))
        for pair in sorted(coupling):
            self.coupling[pair] = self.coupling.get(pair, 0.0) + coupling[pair]
        self.generation += 1
        self.coupling_writes += len(coupling)
        self._scratch.pop(staged["proposal_id"], None)
        return True


class ResidentCellFieldAdapter(_ScratchAdapter):
    """Field adapter backed by a real :class:`ResidentCells` region.

    ``activate`` writes only the additive head score/state words.  Permanent
    ``head_id``/``identity`` fields are never supplied to ``update_head`` and
    therefore cannot be rewritten.  Field coupling is committed as weighted
    pair observations on the resident rails, which also preserves the permanent
    pair metadata.  No universe merge or canonical promotion is involved.
    """

    def __init__(self, cells: ResidentCells, *,
                 proposal_prefix: str = "resident-field",
                 state_scale: float = 1024.0):
        if not isinstance(cells, ResidentCells):
            raise TypeError("cells must be a ResidentCells region")
        super().__init__(proposal_prefix=proposal_prefix)
        self.cells = cells
        self.state_scale = max(0.0, float(state_scale))
        self.coupling: dict[tuple[int, int], float] = {}
        self.activation_writes = 0
        self.coupling_writes = 0

    @property
    def generation(self) -> int:
        return int(self.cells.generation)

    def activate(self, head: int, amount: float = 1.0) -> float:
        index = _head_index(head)
        delta = _finite(amount, "activation amount")
        if delta < 0.0:
            raise FieldLearningError("activation amount must be non-negative")
        state_delta = int(round(min(delta, 4096.0) * self.state_scale))
        record = self.cells.update_head(
            index, score_delta=delta, state_delta=state_delta)
        self.activation_writes += 1
        return record.score_value

    def read_all(self) -> tuple[float, ...]:
        return tuple(self.cells.read_head(head).score_value
                     for head in range(HEAD_COUNT))

    def field_activation(self) -> float:
        return float(sum(abs(value) for value in self.coupling.values()))

    def coupling_weight(self, left: int, right: int) -> float:
        first, second = _head_index(left), _head_index(right)
        if first == second:
            return 0.0
        pair = (first, second) if first < second else (second, first)
        return float(self.coupling.get(pair, 0.0))

    def apply_proposal(self, proposal: Any) -> bool:
        staged = self._resolve(proposal)
        if staged is None:
            return False
        coupling = _normalize_coupling(staged.get("coupling"))
        for pair in sorted(coupling):
            self.cells.observe_pair(pair[0], pair[1], weight=coupling[pair])
            self.coupling[pair] = self.coupling.get(pair, 0.0) + coupling[pair]
        self.coupling_writes += len(coupling)
        self._scratch.pop(staged["proposal_id"], None)
        return True


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------
class FieldLearner:
    """Learn independent head activity and field coupling sentence by sentence.

    The temporal model is *not* replaced.  Its ``observe`` method is called for
    the same sentence, and its ``_prob`` method (when present) provides a
    pre-update token NLL for the ledger and evaluation.  The head field evolves
    separately, so a temporal rejection (for example a duplicate) does not stop
    head activity, and a field rejection never rolls back temporal learning.
    """

    head_count = HEAD_COUNT

    def __init__(self, temporal: Any, field: Any, *,
                 max_active_heads: int = DEFAULT_MAX_ACTIVE_HEADS,
                 max_couplings: int = DEFAULT_MAX_COUPLINGS,
                 max_writes_per_sentence: int = DEFAULT_MAX_WRITES_PER_SENTENCE,
                 total_write_budget: int | None = DEFAULT_TOTAL_WRITE_BUDGET,
                 max_head_delta: float = DEFAULT_MAX_HEAD_DELTA,
                 max_coupling_delta: float = DEFAULT_MAX_COUPLING_DELTA,
                 head_gain: float = DEFAULT_HEAD_GAIN,
                 coupling_gain: float = DEFAULT_COUPLING_GAIN,
                 mirror_gain: float = DEFAULT_MIRROR_GAIN,
                 pair_window: int = DEFAULT_PAIR_WINDOW,
                 active_epsilon: float = DEFAULT_ACTIVE_EPSILON):
        for name in ("activate", "read_all", "field_activation",
                     "propose", "apply_proposal"):
            if not callable(getattr(field, name, None)):
                raise TypeError("field adapter is missing %s()" % name)
        self.temporal = temporal
        self.field = field
        self.max_active_heads = _limit(
            max_active_heads, DEFAULT_MAX_ACTIVE_HEADS, "max_active_heads",
            HEAD_COUNT)
        self.max_couplings = _limit(
            max_couplings, DEFAULT_MAX_COUPLINGS, "max_couplings",
            HEAD_COUNT * (HEAD_COUNT - 1) // 2)
        self.max_writes_per_sentence = _limit(
            max_writes_per_sentence, DEFAULT_MAX_WRITES_PER_SENTENCE,
            "max_writes_per_sentence", HEAD_COUNT * HEAD_COUNT)
        if total_write_budget is not None:
            if isinstance(total_write_budget, bool):
                raise ValueError("total_write_budget must be an integer or None")
            total_write_budget = max(0, int(total_write_budget))
        self.total_write_budget = total_write_budget
        self.max_head_delta = max(0.0, _finite(max_head_delta, "max_head_delta"))
        self.max_coupling_delta = max(
            0.0, _finite(max_coupling_delta, "max_coupling_delta"))
        self.head_gain = max(0.0, _finite(head_gain, "head_gain"))
        self.coupling_gain = max(0.0, _finite(coupling_gain, "coupling_gain"))
        self.mirror_gain = max(0.0, _finite(mirror_gain, "mirror_gain"))
        self.pair_window = _limit(pair_window, DEFAULT_PAIR_WINDOW, "pair_window",
                                  HEAD_COUNT)
        self.active_epsilon = max(
            0.0, _finite(active_epsilon, "active_epsilon"))

        self.head_ledger: list[dict[str, Any]] = [
            {
                "head": head,
                "attempts": 0,
                "accepted": 0,
                "rejected": 0,
                "activation": 0.0,
                "applied": 0.0,
                "loss_sum": 0.0,
                "loss_count": 0,
                "last_nll": None,
            }
            for head in range(HEAD_COUNT)
        ]
        # Short alias for callers that inspect the ledger directly.
        self.ledger = self.head_ledger
        self.sentences = 0
        self.tokens_seen = 0
        self.collisions = 0
        self.writes = 0
        self.accepted_proposals = 0
        self.rejected_proposals = 0
        self.budget_rejections = 0
        self.temporal_observations = 0
        self.last_result: dict[str, Any] | None = None
        self.last_rejection: dict[str, Any] | None = None

    # -- adapter readout -------------------------------------------------
    def _vector(self) -> tuple[float, ...]:
        return _coerce_vector(self.field.read_all())

    def _snapshot_digest(self) -> str:
        return _vector_digest(self._vector())

    def _field_activation(self) -> float:
        try:
            return _coerce_scalar(self.field.field_activation())
        except Exception:
            return 0.0

    def _remaining_writes(self) -> int | None:
        if self.total_write_budget is None:
            return None
        return max(0, self.total_write_budget - self.writes)

    # -- token -> head mapping -------------------------------------------
    def _head_for_token(self, token: str) -> int:
        vocab = getattr(self.temporal, "vocab", None)
        if isinstance(vocab, Mapping):
            value = vocab.get(token, fnv_head(token))
        else:
            value = fnv_head(token)
        try:
            head = int(value) % self.head_count
        except (TypeError, ValueError):
            head = fnv_head(token) % self.head_count
        return head

    def _mirror_head(self, head: int) -> int:
        return mirror8(head) % self.head_count

    def _token_sequence(self, text: str) -> list[tuple[str, int, bool]]:
        raw = tokens(text)
        sequence = [(BOS, self._head_for_token(BOS), True)]
        sequence.extend(
            (token, self._head_for_token(token), False) for token in raw)
        sequence.append((EOS, self._head_for_token(EOS), True))
        return sequence

    def _activation_map(self, text: str) -> tuple[
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
        selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        deltas = {
            head: min(self.max_head_delta, self.head_gain * weight)
            for head, weight in selected[:self.max_active_heads]
        }
        return deltas, collisions, token_count

    def _coupling_map(self, text: str) -> dict[tuple[int, int], float]:
        sequence = self._token_sequence(text)
        heads = [head for _token, head, _boundary in sequence]
        counts: Counter[tuple[int, int]] = Counter()
        for index, left in enumerate(heads):
            stop = min(len(heads), index + self.pair_window + 1)
            for other in range(index + 1, stop):
                right = heads[other]
                if left == right:
                    continue
                pair = (left, right) if left < right else (right, left)
                counts[pair] += 1.0 / (other - index)
        selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return {
            pair: min(self.max_coupling_delta, self.coupling_gain * weight)
            for pair, weight in selected[:self.max_couplings]
        }

    def _bound_updates(
            self, heads: dict[int, float],
            coupling: dict[tuple[int, int], float]
    ) -> tuple[dict[int, float], dict[tuple[int, int], float], dict[str, Any]]:
        remaining = self._remaining_writes()
        capacity = self.max_writes_per_sentence
        if remaining is not None:
            capacity = min(capacity, remaining)
        head_items = list(heads.items())[:max(0, capacity)]
        slots = max(0, capacity - len(head_items))
        coupling_items = list(coupling.items())[:slots]
        return (dict(head_items), dict(coupling_items), {
            "capacity": int(capacity),
            "head_updates": len(head_items),
            "coupling_updates": len(coupling_items),
            "total_write_budget": self.total_write_budget,
            "writes_remaining": remaining,
        })

    # -- temporal channel ------------------------------------------------
    def _temporal_losses(self, text: str) -> list[tuple[int, float]] | None:
        probability = getattr(self.temporal, "_prob", None)
        if not callable(probability):
            return None
        raw = tokens(text)
        sequence = [BOS] + raw + [EOS]
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

    # -- invariants ------------------------------------------------------
    @staticmethod
    def _proposal_updates(proposal: Any) -> tuple[
            dict[int, float], dict[tuple[int, int], float]] | None:
        heads_raw = _proposal_value(proposal, "heads", "head_updates")
        coupling_raw = _proposal_value(
            proposal, "coupling", "coupling_updates")
        if heads_raw is None and coupling_raw is None:
            return None
        return _normalize_heads(heads_raw), _normalize_coupling(coupling_raw)

    def _check_invariants(
            self, submitted_heads: dict[int, float],
            submitted_coupling: dict[tuple[int, int], float],
            proposal: Any, snapshot_before: str,
            capacity: int) -> dict[str, bool]:
        checks: dict[str, bool] = {"proposal_staged": proposal is not None}
        staged_heads: dict[int, float] = {}
        staged_coupling: dict[tuple[int, int], float] = {}
        staged_visible = False
        if proposal is not None:
            try:
                updates = self._proposal_updates(proposal)
            except (FieldLearningError, TypeError, ValueError):
                checks["proposal_shape_valid"] = False
                updates = None
            else:
                checks["proposal_shape_valid"] = True
            if updates is not None:
                staged_visible = True
                staged_heads, staged_coupling = updates

        # Validate the union without double-counting an exact staging copy.
        combined_heads = dict(submitted_heads)
        for head, delta in staged_heads.items():
            combined_heads[head] = max(combined_heads.get(head, 0.0), delta)
        combined_coupling = dict(submitted_coupling)
        for pair, delta in staged_coupling.items():
            previous = combined_coupling.get(pair)
            if previous is None or abs(delta) > abs(previous):
                combined_coupling[pair] = delta

        checks["head_indices_valid"] = all(
            0 <= head < self.head_count for head in combined_heads)
        checks["head_deltas_finite"] = all(
            math.isfinite(delta) for delta in combined_heads.values())
        checks["head_deltas_nonnegative"] = all(
            delta >= 0.0 for delta in combined_heads.values())
        checks["head_deltas_bounded"] = all(
            delta <= self.max_head_delta + 1e-12
            for delta in combined_heads.values())
        checks["coupling_pairs_distinct"] = all(
            left != right for left, right in combined_coupling)
        checks["coupling_deltas_finite"] = all(
            math.isfinite(delta) for delta in combined_coupling.values())
        checks["coupling_deltas_nonnegative"] = all(
            delta >= 0.0 for delta in combined_coupling.values())
        checks["coupling_deltas_bounded"] = all(
            abs(delta) <= self.max_coupling_delta + 1e-12
            for delta in combined_coupling.values())
        checks["head_budget"] = len(combined_heads) <= self.max_active_heads
        checks["coupling_budget"] = (
            len(combined_coupling) <= self.max_couplings)
        total_updates = len(combined_heads) + len(combined_coupling)
        checks["write_budget"] = (
            total_updates <= capacity
            and total_updates <= self.max_writes_per_sentence)
        checks["staged_matches_submitted"] = (
            not staged_visible
            or (_mapping_close(submitted_heads, staged_heads)
                and _mapping_close(submitted_coupling, staged_coupling)))
        try:
            checks["main_field_unchanged"] = (
                self._snapshot_digest() == snapshot_before)
        except Exception:
            checks["main_field_unchanged"] = False
        try:
            checks["field_activation_finite"] = math.isfinite(
                self._field_activation())
        except Exception:
            checks["field_activation_finite"] = False
        return checks

    # -- exposure --------------------------------------------------------
    def observe(self, text: str, *, source: str | None = None) -> dict[str, Any]:
        """Learn one sentence through the temporal and independent field paths."""
        value = str(text or "").strip()
        base = self._base_result()
        if len(value) < 2:
            return {**base, "learned": False, "accepted": False,
                    "reason": "short", "active_heads": 0,
                    "active_head_count": 0, "entropy": 0.0, "entropy_bits": 0.0,
                    "collisions": 0, "writes": 0,
                    "field_activation": round(self._field_activation(), 6),
                    "temporal_learned": None}

        self.sentences += 1
        losses = self._temporal_losses(value)
        if losses:
            for head, nll in losses:
                row = self.head_ledger[head]
                row["loss_sum"] += nll
                row["loss_count"] += 1
                row["last_nll"] = nll
        loss = (sum(value for _head, value in losses) / len(losses)
                if losses else None)

        temporal_learned = None
        temporal_error = None
        observe = getattr(self.temporal, "observe", None)
        if callable(observe):
            try:
                temporal_learned = bool(observe(value))
            except Exception as error:
                temporal_error = type(error).__name__
        else:
            temporal_error = "missing observe()"
        if temporal_learned:
            self.temporal_observations += 1

        try:
            heads, collisions, token_count = self._activation_map(value)
            coupling = self._coupling_map(value)
        except Exception as error:
            return {**base, "learned": False, "accepted": False,
                    "reason": "activation_error",
                    "error": type(error).__name__,
                    "temporal_learned": temporal_learned,
                    "active_heads": 0, "active_head_count": 0,
                    "entropy": 0.0, "entropy_bits": 0.0,
                    "collisions": 0, "writes": 0,
                    "field_activation": round(self._field_activation(), 6)}
        self.tokens_seen += token_count
        self.collisions += collisions

        for head, delta in heads.items():
            row = self.head_ledger[head]
            row["attempts"] += 1
            row["activation"] += delta

        heads, coupling, budget = self._bound_updates(heads, coupling)
        if not heads and not coupling:
            self.budget_rejections += 1
            result = {
                **base,
                "learned": False,
                "accepted": False,
                "reason": "write_budget_exhausted",
                "temporal_learned": temporal_learned,
                "temporal_error": temporal_error,
                "active_heads": 0,
                "active_head_count": 0,
                "entropy": 0.0,
                "entropy_bits": 0.0,
                "collisions": collisions,
                "loss": loss,
                "loss_tokens": len(losses) if losses else 0,
                "writes": 0,
                "field_activation": round(self._field_activation(), 6),
                "budget": budget,
            }
            self.last_result = result
            return result

        entropy = activation_entropy(heads.values())
        try:
            snapshot_before = self._snapshot_digest()
        except Exception as error:
            result = {
                **base, "learned": False, "accepted": False,
                "reason": "field_read_error",
                "error": type(error).__name__,
                "temporal_learned": temporal_learned,
                "active_heads": len(heads),
                "active_head_count": len(heads),
                "entropy": round(entropy, 6),
                "entropy_bits": round(entropy, 6),
                "collisions": collisions, "writes": 0,
                "field_activation": round(self._field_activation(), 6),
                "budget": budget,
            }
            self.last_result = result
            return result

        sentence_digest = hashlib.sha256(
            value.casefold().encode("utf-8")).hexdigest()
        payload = {
            "heads": heads,
            "coupling": coupling,
            "field": sum(abs(value) for value in coupling.values()),
            "snapshot_digest": snapshot_before,
            "sentence_digest": sentence_digest,
            "source": source,
        }
        proposal = None
        proposal_error = None
        try:
            proposal = self.field.propose(payload)
        except Exception as error:
            proposal_error = type(error).__name__

        checks = self._check_invariants(
            heads, coupling, proposal, snapshot_before, budget["capacity"])
        accepted = all(checks.values())
        applied_heads: dict[int, float] = {}
        applied_coupling = 0
        apply_error = None
        if accepted:
            try:
                applied = self.field.apply_proposal(proposal)
            except Exception as error:
                applied = False
                apply_error = type(error).__name__
            if applied is False or (
                    isinstance(applied, Mapping)
                    and applied.get("applied", applied.get("accepted", True)) is False):
                accepted = False
                checks["apply_proposal_succeeded"] = False
                apply_error = apply_error or "apply_proposal_returned_false"
            else:
                checks["apply_proposal_succeeded"] = True
                applied_coupling = len(coupling)
                for head, delta in heads.items():
                    try:
                        self.field.activate(head, delta)
                    except Exception as error:
                        apply_error = type(error).__name__
                        checks["activate_succeeded"] = False
                        accepted = False
                        break
                    applied_heads[head] = delta
                else:
                    checks["activate_succeeded"] = True

        committed = accepted and len(applied_heads) == len(heads)
        if committed:
            self.accepted_proposals += 1
            for head, delta in heads.items():
                row = self.head_ledger[head]
                row["accepted"] += 1
                row["applied"] += delta
        else:
            self.rejected_proposals += 1
            for head in heads:
                self.head_ledger[head]["rejected"] += 1
            if proposal is not None:
                discard = getattr(self.field, "discard_proposal", None)
                if callable(discard):
                    try:
                        discard(proposal)
                    except Exception:
                        pass
            if proposal_error:
                checks["proposal_staged"] = checks.get("proposal_staged", False)
                self.last_rejection = {
                    "sentence_digest": sentence_digest,
                    "reason": "proposal_error",
                    "error": proposal_error,
                    "invariants": checks,
                }
            else:
                self.last_rejection = {
                    "sentence_digest": sentence_digest,
                    "reason": "invariant_failure",
                    "error": apply_error,
                    "invariants": checks,
                }

        self.writes += len(applied_heads) + applied_coupling
        result = {
            **base,
            "learned": committed,
            "accepted": committed,
            "reason": None if committed else (
                "proposal_error" if proposal_error else
                "apply_error" if apply_error else "invariant_failure"),
            "temporal_learned": temporal_learned,
            "temporal_error": temporal_error,
            "proposal_error": proposal_error,
            "sentence_digest": sentence_digest,
            "active_heads": len(heads),
            "active_head_count": len(heads),
            "entropy": round(entropy, 6),
            "entropy_bits": round(entropy, 6),
            "collisions": collisions,
            "head_updates": [
                {"head": head, "delta": delta}
                for head, delta in sorted(heads.items())
            ],
            "coupling_updates": [
                {"left": pair[0], "right": pair[1], "delta": delta}
                for pair, delta in sorted(coupling.items())
            ],
            "applied_heads": len(applied_heads),
            "applied_coupling": applied_coupling,
            "writes": len(applied_heads) + applied_coupling,
            "loss": loss,
            "loss_tokens": len(losses) if losses else 0,
            "field_activation": round(self._field_activation(), 6),
            "proposal": {
                "id": _proposal_value(proposal, "proposal_id", "id"),
                "staged": proposal is not None,
                "accepted": committed,
            },
            "invariants": checks,
            "budget": budget,
        }
        self.last_result = result
        return result

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
        """Observe a bounded batch and aggregate independent field metrics."""
        rows = self._rows(records, limit)
        learned = skipped = temporal_learned = 0
        accepted_before = self.accepted_proposals
        rejected_before = self.rejected_proposals
        collisions_before = self.collisions
        writes_before = self.writes
        active_union: set[int] = set()
        active_counts: list[int] = []
        entropies: list[float] = []
        loss_sum = 0.0
        loss_tokens = 0
        for record in rows:
            text = self._record_text(record)
            if len(text) < 2:
                skipped += 1
                continue
            source = record.get("source") if isinstance(record, Mapping) else None
            result = self.observe(text, source=source)
            if result.get("temporal_learned"):
                temporal_learned += 1
            if not result.get("learned"):
                skipped += 1
                continue
            learned += 1
            active_union.update(item["head"] for item in result["head_updates"])
            active_counts.append(int(result["active_heads"]))
            entropies.append(float(result["entropy"]))
            if result.get("loss") is not None and result.get("loss_tokens"):
                loss_sum += float(result["loss"]) * int(result["loss_tokens"])
                loss_tokens += int(result["loss_tokens"])
        return {
            **self._base_result(),
            "records": len(rows),
            "learned": learned,
            "skipped": skipped,
            "temporal_learned": temporal_learned,
            "active_heads": len(active_union),
            "active_head_count": len(active_union),
            "mean_active_heads": round(
                sum(active_counts) / len(active_counts), 6) if active_counts else 0.0,
            "entropy": round(sum(entropies) / len(entropies), 6)
            if entropies else 0.0,
            "entropy_bits": round(sum(entropies) / len(entropies), 6)
            if entropies else 0.0,
            "collisions": self.collisions - collisions_before,
            "accepted_proposals": self.accepted_proposals - accepted_before,
            "rejected_proposals": self.rejected_proposals - rejected_before,
            "writes": self.writes - writes_before,
            "temporal_nll": round(loss_sum / loss_tokens, 6)
            if loss_tokens else None,
            "field_activation": round(self._field_activation(), 6),
            "budget": self._budget_status(),
        }

    def evaluate(self, records: Iterable[Any], *,
                 limit: int | None = None) -> dict[str, Any]:
        """Read-only head-activity and temporal-NLL evaluation.

        Evaluation never calls ``observe``, ``propose``, ``activate`` or
        ``apply_proposal``.  It reports the activity the current learner would
        induce for each record, plus temporal next-token NLL when the temporal
        model exposes ``_prob``.
        """
        rows = self._rows(records, limit)
        per_head: Counter[int] = Counter()
        active_counts: list[int] = []
        entropies: list[float] = []
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
                heads, record_collisions, token_count = self._activation_map(text)
            except (FieldLearningError, TypeError, ValueError):
                skipped += 1
                continue
            evaluated += 1
            collisions += record_collisions
            tokens_seen += token_count
            per_head.update(heads.keys())
            active_counts.append(len(heads))
            entropies.append(activation_entropy(heads.values()))
            losses = self._temporal_losses(text)
            if losses is None:
                temporal_available = False
            else:
                for _head, nll in losses:
                    loss_sum += nll
                    loss_tokens += 1
        mean_nll = round(loss_sum / loss_tokens, 6) if loss_tokens else None
        independent = {
            "active_heads": len(per_head),
            "active_head_fraction": round(len(per_head) / self.head_count, 6),
            "mean_active_heads_per_record": round(
                sum(active_counts) / len(active_counts), 6)
            if active_counts else 0.0,
            "max_active_heads": max(active_counts) if active_counts else 0,
            "entropy_bits": round(sum(entropies) / len(entropies), 6)
            if entropies else 0.0,
            "collisions": collisions,
            "head_activity": {
                str(head): count for head, count in sorted(per_head.items())
            },
        }
        temporal = {
            "available": bool(temporal_available),
            "tokens": loss_tokens,
            "mean_negative_log_likelihood": mean_nll,
            "perplexity": (round(math.exp(min(mean_nll, 50.0)), 6)
                           if mean_nll is not None else None),
        }
        return {
            **self._base_result(),
            "records": len(rows),
            "evaluated": evaluated,
            "skipped": skipped,
            "tokens": tokens_seen,
            "active_heads": independent["active_heads"],
            "active_head_count": independent["active_heads"],
            "entropy": independent["entropy_bits"],
            "entropy_bits": independent["entropy_bits"],
            "collisions": collisions,
            "independent": independent,
            "temporal": temporal,
            "temporal_nll": mean_nll,
            "field_activation": round(self._field_activation(), 6),
            "budget": self._budget_status(),
        }

    # -- reporting -------------------------------------------------------
    @staticmethod
    def _base_result() -> dict[str, Any]:
        return {
            "schema": FIELD_SCHEMA,
            "canonical_merge": CANONICAL_MERGE,
            "semantic_truth": HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH,
            "head_activations_are_semantic_truth":
                HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH,
        }

    def _budget_status(self) -> dict[str, Any]:
        remaining = self._remaining_writes()
        return {
            "max_active_heads": self.max_active_heads,
            "max_couplings": self.max_couplings,
            "max_writes_per_sentence": self.max_writes_per_sentence,
            "total_write_budget": self.total_write_budget,
            "writes": self.writes,
            "writes_remaining": remaining,
        }

    @staticmethod
    def _ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
        count = int(row["loss_count"])
        return {
            "head": int(row["head"]),
            "attempts": int(row["attempts"]),
            "accepted": int(row["accepted"]),
            "rejected": int(row["rejected"]),
            "activation": float(row["activation"]),
            "applied": float(row["applied"]),
            "loss_sum": float(row["loss_sum"]),
            "loss_count": count,
            "mean_loss": (float(row["loss_sum"]) / count) if count else None,
            "last_nll": row["last_nll"],
        }

    def ledger_rows(self) -> list[dict[str, Any]]:
        """Return a copy of all 256 per-head ledger rows."""
        return [self._ledger_row(row) for row in self.head_ledger]

    def stats(self) -> dict[str, Any]:
        """Return active-head, entropy, collision and proposal counters."""
        try:
            vector = self._vector()
        except Exception:
            vector = (0.0,) * HEAD_COUNT
        active = [value for value in vector if value > self.active_epsilon]
        entropy = round(activation_entropy(vector), 6)
        ledger_active = sum(
            1 for row in self.head_ledger if row["activation"] > 0.0)
        ledger_applied = sum(
            1 for row in self.head_ledger if row["applied"] > 0.0)
        total_proposals = self.accepted_proposals + self.rejected_proposals
        temporal = {
            "vocab": len(getattr(self.temporal, "vocab", {}) or {}),
            "observations": self.temporal_observations,
            "has_probability": callable(getattr(self.temporal, "_prob", None)),
        }
        return {
            **self._base_result(),
            "sentences": self.sentences,
            "tokens": self.tokens_seen,
            "active_heads": len(active),
            "active_head_count": len(active),
            "ledger_active_heads": ledger_active,
            "ledger_applied_heads": ledger_applied,
            "entropy": entropy,
            "entropy_bits": entropy,
            "collisions": self.collisions,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "budget_rejections": self.budget_rejections,
            "proposal_acceptance_rate": (
                round(self.accepted_proposals / total_proposals, 6)
                if total_proposals else 0.0),
            "proposals": {
                "accepted": self.accepted_proposals,
                "rejected": self.rejected_proposals,
                "total": total_proposals,
            },
            "writes": self.writes,
            "field_activation": round(self._field_activation(), 6),
            "temporal": temporal,
            "budget": self._budget_status(),
            "last_rejection": self.last_rejection,
        }

    @classmethod
    def from_cells(cls, temporal: Any, cells: ResidentCells,
                   **kwargs: Any) -> "FieldLearner":
        """Convenience constructor over a real resident cell region."""
        return cls(temporal, ResidentCellFieldAdapter(cells), **kwargs)


__all__ = (
    "FIELD_SCHEMA", "FIELD_PROPOSAL_SCHEMA", "CANONICAL_MERGE",
    "HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH", "FieldLearningError",
    "HeadFieldAdapter", "ResidentCellFieldAdapter", "FieldLearner",
    "activation_entropy", "DEFAULT_MAX_ACTIVE_HEADS", "DEFAULT_MAX_COUPLINGS",
    "DEFAULT_MAX_WRITES_PER_SENTENCE", "DEFAULT_TOTAL_WRITE_BUDGET",
    "DEFAULT_MAX_HEAD_DELTA", "DEFAULT_MAX_COUPLING_DELTA",
)
