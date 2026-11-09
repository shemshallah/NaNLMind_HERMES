"""Independent 256-head activation field over resident head cells.

:class:`ResidentHeadField` is a thin adapter around the typed head records and
pair rails in :mod:`nanlmind_hermes.resident.cells`.  Each head owns an
additive Q16.16 score; activating one head never changes another head's score,
state, identity or per-head generation.  The permanent ``head_id`` and identity
address written by ``ResidentCells.initialize`` are never rewritten here.

The adapter accepts either a ``ResidentCells`` instance directly or a host
object that exposes it as ``cells`` and optionally supplies ``generation``,
``_commit`` and ``flush`` hooks.  Any duck-typed store with ``read_head``,
``update_head`` and ``observe_pair`` also works, which keeps the field usable
with test fakes.

Pair coupling delegates to ``observe_pair``.  Pair rails accumulate evidence
only; they do not alter head identity.  Proposals are scratch records: a
``propose`` call validates and previews a delta without touching resident
bytes.  Only ``apply_proposal(token, approved=True)`` (or an explicitly
approved proposal) commits the delta.

Everything returned by :meth:`snapshot` is a bounded, JSON-friendly
projection: at most ``SNAPSHOT_HEADS`` head summaries, a fixed 256-element
activation vector and a bounded event/proposal tail.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from collections import OrderedDict, deque
from collections.abc import Mapping

from .cells import FIXED_ONE, HEAD_COUNT, identity_address, to_fixed, to_phase

SCHEMA = "nanlm.resident-head-field/1"

# A score word is a signed Q16.16 i32, so no single delta may exceed the
# representable fixed-point span.  Larger requests are denied rather than
# silently wrapped.
MAX_DELTA = 32767.0
MAX_COUPLE_WEIGHT = MAX_DELTA
MIN_SCORE_WORD = -(1 << 31)
MAX_SCORE_WORD = (1 << 31) - 1
MIN_SCORE = MIN_SCORE_WORD / FIXED_ONE
MAX_SCORE = MAX_SCORE_WORD / FIXED_ONE

ACTIVE_EPSILON = 1e-9

SNAPSHOT_HEADS = 32
SNAPSHOT_EVENTS = 16
SNAPSHOT_PROPOSALS = 8
EVENT_LIMIT = 64
MAX_PENDING_PROPOSALS = 256


class HeadFieldError(ValueError):
    """Base error for resident head-field operations."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "status": "error",
            "code": self.code,
            "error": str(self),
            "details": self.details,
        }


class HeadFieldDenied(HeadFieldError):
    """Raised when validation or approval rules refuse an operation."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _check_head(head) -> int:
    if isinstance(head, bool) or not isinstance(head, int) \
            or not 0 <= head < HEAD_COUNT:
        raise HeadFieldDenied(
            "unknown head %r; expected an integer in [0, %d)"
            % (head, HEAD_COUNT),
            code="unknown_head", details={"head": head})
    return head


def _check_real(value, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeadFieldDenied(
            "%s must be a real number: %r" % (name, value),
            code="invalid_%s" % name, details={name: value})
    if isinstance(value, float) and not math.isfinite(value):
        raise HeadFieldDenied(
            "%s must be finite: %r" % (name, value),
            code="non_finite_%s" % name, details={name: value})
    if abs(value) > maximum:
        raise HeadFieldDenied(
            "%s is too large: %r (limit %r)" % (name, value, maximum),
            code="%s_too_large" % name,
            details={name: value, "maximum": maximum})
    try:
        to_fixed(value, name)
    except ValueError as error:
        raise HeadFieldDenied(
            str(error), code="invalid_%s" % name,
            details={name: value}) from error
    return float(value)


def _check_phase(phase) -> float:
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise HeadFieldDenied(
            "phase must be a real number: %r" % (phase,),
            code="invalid_phase", details={"phase": phase})
    if isinstance(phase, float) and not math.isfinite(phase):
        raise HeadFieldDenied(
            "phase must be finite: %r" % (phase,),
            code="non_finite_phase", details={"phase": phase})
    try:
        to_phase(phase, "phase")
    except ValueError as error:
        raise HeadFieldDenied(
            str(error), code="invalid_phase",
            details={"phase": phase}) from error
    return float(phase)


def _check_generation(value, name: str = "generation") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HeadFieldError(
            "%s must be a non-negative integer: %r" % (name, value),
            code="invalid_generation", details={name: value})
    return value


def _reason_text(reason) -> str:
    if reason is None:
        return ""
    return str(reason)[:240]


def _host_attr(host, name: str, default=None):
    if isinstance(host, Mapping):
        return host.get(name, default)
    return getattr(host, name, default)


def _host_generation(host) -> int:
    try:
        value = _host_attr(host, "generation", 0)
    except Exception:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _resolve_cells(cells):
    """Return ``(store, host)`` for a direct store or a wrapper object."""
    if callable(getattr(cells, "read_head", None)) \
            and callable(getattr(cells, "update_head", None)):
        return cells, cells
    nested = _host_attr(cells, "cells")
    if nested is not None \
            and callable(getattr(nested, "read_head", None)) \
            and callable(getattr(nested, "update_head", None)):
        return nested, cells
    raise TypeError(
        "cells must expose read_head/update_head, directly or via .cells")


def _first_callable(host, store, name: str):
    for candidate_owner in (host, store):
        candidate = _host_attr(candidate_owner, name)
        if callable(candidate):
            return candidate
    return None


def _call_commit(callback, kind: str, identity: str) -> None:
    """Call a zero-argument hook or a universe-style ``_commit(kind, id)``."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback()
        return
    positional = [
        parameter for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD)
    ]
    varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values())
    if varargs or len(positional) >= 2:
        callback(kind, identity)
    elif len(positional) == 1:
        callback(kind)
    else:
        callback()


def _record_field(record, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _integer_field(record, name: str, default: int = 0) -> int:
    value = _record_field(record, name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return default
    return value


def _score_parts(record) -> tuple[int, float]:
    raw = _record_field(record, "score", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise HeadFieldError(
            "resident head score is not numeric", code="invalid_head_record")
    if isinstance(raw, int):
        word = raw
        value = raw / FIXED_ONE
    else:
        value = float(raw)
        if not math.isfinite(value):
            raise HeadFieldError(
                "resident head score is not finite", code="invalid_head_record")
        word = int(round(value * FIXED_ONE))
    return word, value


def _fixed_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        return value / FIXED_ONE
    return float(value)


def _token_id(token) -> str:
    if isinstance(token, Mapping):
        token = token.get("token")
    if isinstance(token, bool) or token is None:
        raise HeadFieldDenied(
            "a proposal token is required", code="unknown_proposal")
    if isinstance(token, int):
        token = str(token)
    if not isinstance(token, str) or not token.strip():
        raise HeadFieldDenied(
            "a proposal token is required", code="unknown_proposal")
    return token.strip()


# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------
class ResidentHeadField:
    """Additive 256-head score field backed by resident cells.

    ``cells`` may be a :class:`~nanlmind_hermes.resident.cells.ResidentCells`
    instance, or a host object whose ``cells`` attribute is the store.  Host
    ``generation`` is used as a monotonic floor; ``_commit`` and ``flush`` are
    invoked after a successful mutation when present (or when explicitly
    supplied).
    """

    head_count = HEAD_COUNT
    schema = SCHEMA

    def __init__(self, cells, generation: int = 0, commit=None, flush=None):
        store, host = _resolve_cells(cells)
        initial = _check_generation(generation)
        self.cells = store
        self.host = host
        self._store = store
        self._host = host
        self.generation = max(
            initial, _host_generation(host), _host_generation(store))

        if commit is not None and not callable(commit):
            raise TypeError("commit must be callable")
        if flush is not None and not callable(flush):
            raise TypeError("flush must be callable")
        self._commit_callback = commit if commit is not None \
            else _first_callable(host, store, "_commit")
        if self._commit_callback is None:
            self._commit_callback = _first_callable(host, store, "commit")
        self._flush_callback = flush if flush is not None \
            else _first_callable(host, store, "flush")
        if self._flush_callback is self._commit_callback:
            self._flush_callback = None

        self._events = deque(maxlen=EVENT_LIMIT)
        self._proposals: OrderedDict[str, dict] = OrderedDict()
        self._token_counter = 0
        self._coupling_count = 0
        self._proposal_count = 0
        self._applied_count = 0

    # -- reading --------------------------------------------------------
    def read(self, head: int) -> dict:
        """Return one head's score/state/identity/generation projection."""
        head = _check_head(head)
        record = self._store.read_head(head)
        word, score = _score_parts(record)
        state = _integer_field(record, "state", 0)
        head_generation = _integer_field(record, "generation", 0)
        head_id = _integer_field(record, "head_id", head)
        identity = _record_field(
            record, "identity", identity_address(head))
        flags = _integer_field(record, "flags", 0)
        return {
            "head": head,
            "head_id": head_id,
            "identity": identity,
            "identity_address": identity,
            "score": score,
            "score_word": word,
            "score_fixed": word,
            "state": state,
            "state_word": state,
            "generation": head_generation,
            "head_generation": head_generation,
            "flags": flags,
            "active": score > ACTIVE_EPSILON,
        }

    def read_all(self) -> list[dict]:
        """Return all 256 head projections in head order."""
        return [self.read(head) for head in range(HEAD_COUNT)]

    # -- activation -----------------------------------------------------
    def activate(self, head: int, delta, *, reason: str = "") -> dict:
        """Add ``delta`` to one head's score, leaving every other head alone."""
        head = _check_head(head)
        value = _check_real(delta, "delta", MAX_DELTA)
        reason = _reason_text(reason)
        before, after = self._add_score(
            head, value, reason=reason, event_kind="activate",
            identity="head:%d" % head)
        return {
            "status": "activated",
            "head": head,
            "delta": value,
            "previous_score": before["score"],
            "score": after["score"],
            "state": after["state"],
            "identity": after["identity"],
            "head_generation": after["generation"],
            "field_generation": self.generation,
            "reason": reason,
        }

    def field_activation(self) -> dict:
        """Return the normalized 256-vector plus entropy/active-head metrics.

        Negative scores are clamped to zero because they carry no activation
        mass.  When the field is empty the vector is all zeros.
        """
        records = self.read_all()
        weights = [max(0.0, item["score"]) for item in records]
        total = math.fsum(weights)
        if total > ACTIVE_EPSILON:
            vector = [weight / total for weight in weights]
            # Keep the normalized vector summing to one despite float rounding.
            residual = 1.0 - math.fsum(vector)
            peak = max(range(HEAD_COUNT), key=lambda index: vector[index])
            vector[peak] += residual
            active = [
                index for index, weight in enumerate(weights)
                if weight > ACTIVE_EPSILON
            ]
            entropy = -math.fsum(
                probability * math.log(probability)
                for probability in vector if probability > 0.0)
            peak_head = peak
            peak_value = vector[peak]
        else:
            vector = [0.0] * HEAD_COUNT
            active = []
            entropy = 0.0
            peak_head = None
            peak_value = 0.0

        entropy_bits = entropy / math.log(2.0) if entropy > 0.0 else 0.0
        active_count = len(active)
        entropy_max = math.log(active_count) if active_count > 1 else 0.0
        entropy_ratio = entropy / entropy_max if entropy_max > 0.0 else 0.0
        metrics = {
            "heads": HEAD_COUNT,
            "total": total,
            "active_heads": active_count,
            "active_head_count": active_count,
            "active": active_count,
            "active_fraction": active_count / HEAD_COUNT,
            "active_indices": active,
            "peak_head": peak_head,
            "peak": peak_value,
            "entropy": entropy,
            "entropy_bits": entropy_bits,
            "entropy_max": entropy_max,
            "entropy_ratio": entropy_ratio,
            "generation": self.generation,
        }
        result = dict(metrics)
        result["vector"] = vector
        result["activation"] = vector
        result["metrics"] = dict(metrics)
        return result

    # -- pair coupling --------------------------------------------------
    def couple(self, left: int, right: int, weight,
               phase=0) -> dict:
        """Accumulate one pair observation without touching head identities."""
        left = _check_head(left)
        right = _check_head(right)
        if left == right:
            raise HeadFieldDenied(
                "a pair needs two distinct heads",
                code="same_head", details={"head": left})
        value = _check_real(weight, "weight", MAX_COUPLE_WEIGHT)
        phase_value = _check_phase(phase)
        observe = getattr(self._store, "observe_pair", None)
        if not callable(observe):
            raise HeadFieldError(
                "resident store does not support observe_pair",
                code="pair_unsupported")

        before_left = self.read(left)
        before_right = self.read(right)
        state = observe(left, right, weight=value, phase=phase_value)
        after_left = self.read(left)
        after_right = self.read(right)
        self._bump_generation()

        identity_preserved = (
            before_left["identity"] == after_left["identity"]
            and before_right["identity"] == after_right["identity"]
            and before_left["head_id"] == after_left["head_id"]
            and before_right["head_id"] == after_right["head_id"])
        if not identity_preserved:
            raise HeadFieldError(
                "pair observation rewrote a permanent head identity",
                code="identity_rewritten")

        pair = {}
        for name in ("canonical", "mirror", "index"):
            value_part = getattr(state, name, None)
            if isinstance(value_part, int) and not isinstance(value_part, bool):
                pair[name] = value_part
        forward = getattr(state, "forward", None)
        reverse = getattr(state, "reverse", None)
        if forward is not None:
            pair["forward_weight"] = _fixed_float(
                getattr(forward, "weight", None))
            pair["forward_amplitude"] = _fixed_float(
                getattr(forward, "amplitude", None))
            pair["forward_phase"] = _fixed_float(
                getattr(forward, "phase", None))
        if reverse is not None:
            pair["reverse_weight"] = _fixed_float(
                getattr(reverse, "weight", None))
            pair["reverse_amplitude"] = _fixed_float(
                getattr(reverse, "amplitude", None))
            pair["reverse_phase"] = _fixed_float(
                getattr(reverse, "phase", None))

        self._coupling_count += 1
        self._remember({
            "kind": "couple",
            "left": left,
            "right": right,
            "weight": value,
            "phase": phase_value,
            "generation": self.generation,
        })
        self._notify("pair", "%d:%d" % (left, right))

        result = {
            "status": "coupled",
            "left": left,
            "right": right,
            "weight": value,
            "phase": phase_value,
            "identity_preserved": True,
            "pair": pair,
            "field_generation": self.generation,
        }
        result.update(pair)
        return result

    # -- proposals ------------------------------------------------------
    def propose(self, head: int, delta, *, approved: bool = False) -> dict:
        """Stage a delta in scratch; only explicit approval mutates cells.

        With ``approved=True`` the proposal is staged and immediately committed
        as an explicit approved write.  Otherwise :meth:`apply_proposal` with
        ``approved=True`` is required.
        """
        head = _check_head(head)
        value = _check_real(delta, "delta", MAX_DELTA)
        current = self.read(head)
        delta_word = to_fixed(value, "delta")
        proposed_word = max(
            MIN_SCORE_WORD, min(MAX_SCORE_WORD, current["score_word"] + delta_word))
        token = self._new_token()
        proposal = {
            "token": token,
            "head": head,
            "delta": value,
            "previous_score": current["score"],
            "proposed_score": proposed_word / FIXED_ONE,
            "identity": current["identity"],
            "status": "proposed",
            "approved": approved is True,
            "applied": False,
            "created_generation": self.generation,
            "applied_generation": None,
            "applied_score": None,
        }
        self._store_proposal(proposal)
        self._proposal_count += 1
        self._remember({
            "kind": "propose",
            "token": token,
            "head": head,
            "delta": value,
            "generation": self.generation,
        })
        if approved is True:
            return self.apply_proposal(token, approved=True)
        return self._proposal_view(proposal)

    def apply_proposal(self, token, *, approved: bool = False) -> dict:
        """Commit a scratch proposal; unapproved applies are denied."""
        token = _token_id(token)
        proposal = self._proposals.get(token)
        if proposal is None:
            raise HeadFieldDenied(
                "unknown proposal token", code="unknown_proposal",
                details={"token": token})
        if approved is not True:
            raise HeadFieldDenied(
                "proposal commit requires explicit approval",
                code="unapproved_apply", details={"token": token})
        if proposal.get("applied"):
            view = self._proposal_view(proposal)
            view["status"] = "already_applied"
            view["applied"] = False
            return view

        head = _check_head(proposal["head"])
        value = _check_real(proposal["delta"], "delta", MAX_DELTA)
        before, after = self._add_score(
            head, value, reason="proposal:%s" % token, event_kind="apply",
            identity=token)
        proposal["status"] = "applied"
        proposal["applied"] = True
        proposal["applied_generation"] = self.generation
        proposal["applied_score"] = after["score"]
        self._applied_count += 1
        view = self._proposal_view(proposal)
        view.update({
            "status": "applied",
            "applied": True,
            "previous_score": before["score"],
            "score": after["score"],
            "state": after["state"],
            "identity": after["identity"],
            "head_generation": after["generation"],
            "field_generation": self.generation,
        })
        return view

    # -- snapshot -------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a bounded JSON-friendly projection of field state."""
        activation = self.field_activation()
        records = self.read_all()
        ordered = sorted(
            records, key=lambda item: (-item["score"], item["head"]))
        heads = [
            {
                "head": item["head"],
                "score": round(item["score"], 8),
                "state": item["state"],
                "identity": item["identity"],
                "generation": item["generation"],
            }
            for item in ordered[:SNAPSHOT_HEADS]
        ]
        vector = [round(value, 8) for value in activation["vector"]]
        identity_material = ",".join(
            str(item["identity"]) for item in records).encode("ascii")
        proposals = [
            self._proposal_view(proposal)
            for proposal in list(self._proposals.values())[-SNAPSHOT_PROPOSALS:]
        ]
        pending = sum(
            1 for proposal in self._proposals.values()
            if not proposal.get("applied"))
        metrics = {
            "active_heads": activation["active_heads"],
            "active_indices": list(activation["active_indices"]),
            "entropy": activation["entropy"],
            "entropy_bits": activation["entropy_bits"],
            "peak_head": activation["peak_head"],
            "peak": activation["peak"],
            "total": activation["total"],
        }
        return {
            "schema": SCHEMA,
            "generation": self.generation,
            "head_count": HEAD_COUNT,
            "heads": heads,
            "heads_returned": len(heads),
            "heads_truncated": len(records) > len(heads),
            "identity_digest": hashlib.sha256(
                identity_material).hexdigest()[:32],
            "active_heads": activation["active_heads"],
            "active_indices": list(activation["active_indices"]),
            "entropy": activation["entropy"],
            "entropy_bits": activation["entropy_bits"],
            "peak_head": activation["peak_head"],
            "peak": activation["peak"],
            "total": activation["total"],
            "vector": vector,
            "activation": {"vector": list(vector), "metrics": metrics},
            "coupling_count": self._coupling_count,
            "proposal_count": self._proposal_count,
            "applied_proposals": self._applied_count,
            "pending_proposals": pending,
            "proposals": proposals,
            "events": [dict(event) for event in
                       list(self._events)[-SNAPSHOT_EVENTS:]],
        }

    # -- internals ------------------------------------------------------
    def _add_score(self, head: int, value: float, *, reason: str,
                   event_kind: str, identity: str):
        before = self.read(head)
        self._store.update_head(head, score_delta=value)
        self._bump_generation()
        after = self.read(head)
        self._remember({
            "kind": event_kind,
            "head": head,
            "delta": value,
            "score": after["score"],
            "reason": reason,
            "generation": self.generation,
        })
        self._notify(event_kind, identity)
        return before, after

    def _new_token(self) -> str:
        self._token_counter += 1
        return "hf-%08x" % self._token_counter

    def _store_proposal(self, proposal: dict) -> None:
        while len(self._proposals) >= MAX_PENDING_PROPOSALS:
            evicted = False
            for token, existing in self._proposals.items():
                if existing.get("applied"):
                    del self._proposals[token]
                    evicted = True
                    break
            if not evicted:
                self._proposals.popitem(last=False)
        self._proposals[proposal["token"]] = proposal

    def _proposal_view(self, proposal: dict) -> dict:
        return {
            "token": proposal["token"],
            "head": proposal["head"],
            "delta": proposal["delta"],
            "previous_score": proposal["previous_score"],
            "proposed_score": proposal["proposed_score"],
            "identity": proposal["identity"],
            "status": proposal["status"],
            "approved": proposal["approved"],
            "applied": bool(proposal["applied"]),
            "created_generation": proposal["created_generation"],
            "applied_generation": proposal["applied_generation"],
            "applied_score": proposal["applied_score"],
        }

    def _remember(self, event: dict) -> None:
        self._events.append(event)

    def _bump_generation(self) -> None:
        self.generation += 1
        self._sync_generation()

    def _sync_generation(self) -> None:
        for candidate in (self._host, self._store):
            try:
                value = _host_generation(candidate)
            except Exception:
                value = 0
            if value > self.generation:
                self.generation = value

    def _notify(self, kind: str, identity: str) -> None:
        if self._commit_callback is not None:
            _call_commit(self._commit_callback, kind, identity)
        if self._flush_callback is not None \
                and self._flush_callback is not self._commit_callback:
            self._flush_callback()
        self._sync_generation()


__all__ = (
    "SCHEMA",
    "MAX_DELTA",
    "MAX_COUPLE_WEIGHT",
    "MIN_SCORE",
    "MAX_SCORE",
    "ACTIVE_EPSILON",
    "SNAPSHOT_HEADS",
    "SNAPSHOT_EVENTS",
    "SNAPSHOT_PROPOSALS",
    "EVENT_LIMIT",
    "MAX_PENDING_PROPOSALS",
    "HeadFieldError",
    "HeadFieldDenied",
    "ResidentHeadField",
)
