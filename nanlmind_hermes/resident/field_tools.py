"""Bounded, dependency-injected tool handlers for resident head fields.

The handlers in this module are deliberately backend agnostic.  A later plugin
integration supplies a backend object with the small contract below; these
functions only validate inputs, enforce explicit approval, bound every
collection and string, and render JSON-safe output.  Nothing here opens a file,
touches a network, exposes a raw pointer/offset, or enables canonical
merge/promotion.

Backend contract (all calls are keyword-free)::

    snapshot() -> mapping
    activate(head, delta) -> mapping
    propose(head, delta, reason) -> mapping
    commit(token, expected_generation) -> mapping
    head_metrics(head) -> mapping
    field_metrics() -> mapping
    resonance(text, top_k) -> mapping or sequence

``snapshot`` should report ``generation``, ``opened``, ``budget`` and optional
``heads``/``active_heads``.  ``activate`` should apply one bounded local
activation and return the new ``generation``.  ``propose`` must stage a scratch
proposal only and return an opaque ``token`` plus ``reasons``.  ``commit`` must
consume that token, re-check the generation, and return ``accepted`` or
``accepted=False`` with ``reasons``.  Query methods are strictly read-only.

Every response includes ``canonical_merge.enabled=False``,
``canonical_merge.attempted=False`` and ``canonical_promotion=False``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
import json
import math
import re
from typing import Any, Callable

SCHEMA = "nanlm.resident-field-tools/1"

HEAD_COUNT = 256
MAX_DELTA = 1.0
MIN_DELTA = -1.0
MAX_TEXT_BYTES = 4096
MAX_HIT_TEXT_BYTES = 512
MAX_TOKEN_BYTES = 256
MAX_REASON_BYTES = 240
MAX_REASONS = 16
MAX_TOP_K = 32
DEFAULT_TOP_K = 8
MAX_STATUS_LIMIT = HEAD_COUNT
DEFAULT_STATUS_LIMIT = HEAD_COUNT
MAX_OUTPUT_BYTES = 64 * 1024
MAX_U32 = (1 << 32) - 1
MAX_GENERATION = (1 << 63) - 1
DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024
MAX_BUDGET_BYTES = 1024 * 1024 * 1024
MAX_UNKNOWN_PROPERTIES = 8

# Tokens are opaque but must not be able to smuggle paths, whitespace,
# pointers or control characters back through a tool response.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MISSING = object()


class FieldToolError(ValueError):
    """Base error for invalid field-tool requests or backend results."""


class FieldValidationError(FieldToolError):
    """A request property is missing, unknown, or outside its bound."""


class FieldApprovalError(FieldToolError):
    """An operation that changes local state lacks explicit approval."""


class FieldBackendError(FieldToolError):
    """The injected backend returned an invalid or unsafe result."""


class FieldConflictError(FieldToolError):
    """A backend rejected a commit; the handler renders it as denied."""

    def __init__(self, message: str = "commit denied",
                 reasons: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = tuple(reasons or ())


class FieldInvariantError(FieldToolError):
    """A backend violated read-only or scratch-only handler invariants."""


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------
def _params(params: Any, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(params, Mapping):
        raise FieldValidationError("params must be an object")
    unknown: list[str] = []
    for key in params:
        if not isinstance(key, str):
            raise FieldValidationError("property names must be strings")
        if len(key) > 64:
            raise FieldValidationError("property name exceeds 64 characters")
        if key not in allowed:
            unknown.append(key)
            if len(unknown) >= MAX_UNKNOWN_PROPERTIES:
                break
    if unknown:
        raise FieldValidationError(
            "unknown properties: %s" % ", ".join(sorted(unknown)))
    return params


def _required(params: Mapping[str, Any], name: str) -> Any:
    if name not in params:
        raise FieldValidationError("%s is required" % name)
    return params[name]


def _require_approved(params: Mapping[str, Any]) -> None:
    if params.get("approved") is not True:
        raise FieldApprovalError("explicit approved=true is required")


def _input_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldValidationError("%s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise FieldValidationError(
            "%s must be in [%d, %d]" % (name, minimum, maximum))
    return value


def _input_head(value: Any) -> int:
    return _input_int(value, "head", 0, HEAD_COUNT - 1)


def _input_generation(value: Any, name: str = "expected_generation") -> int:
    return _input_int(value, name, 0, MAX_GENERATION)


def _input_limit(value: Any, name: str, maximum: int) -> int:
    return _input_int(value, name, 1, maximum)


def _input_delta(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldValidationError("delta must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FieldValidationError("delta must be a finite number") from exc
    if not math.isfinite(result):
        raise FieldValidationError("delta must be finite")
    if result < MIN_DELTA or result > MAX_DELTA:
        raise FieldValidationError(
            "delta must be in [%.1f, %.1f]" % (MIN_DELTA, MAX_DELTA))
    return round(result, 8)


def _bounded_string(value: Any, name: str, maximum: int,
                    error_type: type[FieldToolError],
                    *, allow_empty: bool) -> str:
    if not isinstance(value, str):
        raise error_type("%s must be a string" % name)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise error_type("%s is not valid UTF-8 text" % name) from exc
    if size > maximum:
        raise error_type("%s exceeds %d bytes" % (name, maximum))
    if not allow_empty and not value.strip():
        raise error_type("%s must not be empty" % name)
    return value


def _input_text(value: Any) -> str:
    return _bounded_string(value, "text", MAX_TEXT_BYTES,
                           FieldValidationError, allow_empty=False)


def _input_reason(value: Any) -> str:
    return _bounded_string(value, "reason", MAX_REASON_BYTES,
                           FieldValidationError, allow_empty=True)


def _input_token(value: Any) -> str:
    if not isinstance(value, str):
        raise FieldValidationError("token must be a string")
    if not value:
        raise FieldValidationError("token must not be empty")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise FieldValidationError("token is not valid UTF-8 text") from exc
    if size > MAX_TOKEN_BYTES:
        raise FieldValidationError(
            "token exceeds %d bytes" % MAX_TOKEN_BYTES)
    if _TOKEN_RE.fullmatch(value) is None:
        raise FieldValidationError(
            "token must contain only letters, digits, '.', '_', ':' or '-'")
    return value


# ---------------------------------------------------------------------------
# backend result validation / sanitisation
# ---------------------------------------------------------------------------
def _backend_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FieldBackendError("backend %s must return an object" % name)
    return value


def _backend_int(value: Any, name: str, minimum: int = 0,
                 maximum: int = MAX_GENERATION, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldBackendError("backend %s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise FieldBackendError(
            "backend %s is outside [%d, %d]" % (name, minimum, maximum))
    return value


def _backend_float(value: Any, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldBackendError("backend %s must be a number" % name)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FieldBackendError("backend %s must be a number" % name) from exc
    if not math.isfinite(result):
        raise FieldBackendError("backend %s must be finite" % name)
    return round(result, 8)


def _backend_ratio(value: Any, name: str, default: float = 0.0) -> float:
    result = _backend_float(value, name, default)
    if result < 0.0 or result > 1.0:
        raise FieldBackendError("backend %s must be in [0, 1]" % name)
    return result


def _backend_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise FieldBackendError("backend %s must be a boolean" % name)
    return value


def _backend_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or \
            not isinstance(value, Sequence):
        raise FieldBackendError("backend reasons must be a sequence")
    if len(value) > MAX_REASONS:
        raise FieldBackendError(
            "backend reasons exceed %d entries" % MAX_REASONS)
    return [
        _bounded_string(item, "reason", MAX_REASON_BYTES,
                        FieldBackendError, allow_empty=True)
        for item in value
    ]


def _backend_hit_text(value: Any, name: str) -> str:
    return _bounded_string(value, name, MAX_HIT_TEXT_BYTES,
                           FieldBackendError, allow_empty=True)


def _call(backend: Any, method: str, *args: Any) -> Any:
    if backend is None:
        raise FieldValidationError("backend is required")
    function = getattr(backend, method, None)
    if not callable(function):
        raise FieldBackendError("backend must provide %s()" % method)
    try:
        return function(*args)
    except FieldToolError:
        raise
    except Exception as exc:  # Keep backend internals out of tool output.
        raise FieldBackendError(
            "backend %s failed: %s" % (method, type(exc).__name__)) from exc


def _snapshot(backend: Any) -> Mapping[str, Any]:
    return _backend_mapping(_call(backend, "snapshot"), "snapshot()")


def _generation_from_snapshot(snapshot: Mapping[str, Any],
                              backend: Any) -> int:
    value = snapshot.get("generation")
    if value is None:
        field = snapshot.get("field")
        if isinstance(field, Mapping):
            value = field.get("generation")
    if value is None:
        return _current_generation(backend)
    return _input_generation(value, "generation")


def _current_generation(backend: Any) -> int:
    value = getattr(backend, "generation", None)
    if callable(value):
        value = value()
    if value is not None:
        return _input_generation(value, "generation")
    snapshot = _snapshot(backend)
    value = snapshot.get("generation")
    if value is None:
        field = snapshot.get("field")
        if isinstance(field, Mapping):
            value = field.get("generation")
    if value is None:
        raise FieldBackendError("backend must expose generation")
    return _input_generation(value, "generation")


def _active_heads(value: Any, error_type: type[FieldToolError]) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)) or \
            not isinstance(value, Sequence):
        raise error_type("active_heads must be a sequence")
    if len(value) > HEAD_COUNT:
        raise error_type("active_heads exceeds %d entries" % HEAD_COUNT)
    heads: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise error_type("active_heads entries must be integers")
        if not 0 <= item < HEAD_COUNT:
            raise error_type(
                "active_heads entries must be in [0, %d]" % (HEAD_COUNT - 1))
        heads.append(item)
    return sorted(set(heads))


def _normalize_heads(value: Any) -> list[dict[str, Any]]:
    """Project only safe head metrics, never backend extras or offsets."""
    if value is None:
        return []
    entries: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if len(value) > HEAD_COUNT:
            raise FieldBackendError("snapshot heads exceed %d entries"
                                    % HEAD_COUNT)
        for key, item in value.items():
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if isinstance(item, Mapping):
                record = dict(item)
                record.setdefault("head", key)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                record = {"head": key, "score": item}
            else:
                raise FieldBackendError(
                    "snapshot heads values must be objects or numbers")
            entries.append(record)
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        if len(value) > HEAD_COUNT:
            raise FieldBackendError("snapshot heads exceed %d entries"
                                    % HEAD_COUNT)
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise FieldBackendError(
                    "snapshot heads entries must be objects")
            record = dict(item)
            record.setdefault("head", index)
            entries.append(record)
    else:
        raise FieldBackendError("snapshot heads must be an object or sequence")

    heads: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in entries:
        head = _input_head(item.get("head"))
        if head in seen:
            raise FieldBackendError("snapshot contains duplicate head %d"
                                    % head)
        seen.add(head)
        score = _backend_float(item.get("score", item.get("weight")),
                               "head score")
        state = _backend_int(item.get("state"), "head state",
                             0, MAX_U32)
        generation = _backend_int(item.get("generation"),
                                  "head generation", 0, MAX_GENERATION)
        active_value = item.get("active")
        active = (bool(score != 0.0 or state != 0)
                  if active_value is None
                  else _backend_bool(active_value, "head active"))
        heads.append({
            "head": head,
            "score": score,
            "state": state,
            "generation": generation,
            "active": active,
        })
    heads.sort(key=lambda item: item["head"])
    return heads


def _normalize_budget(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    budget = snapshot.get("budget")
    if not isinstance(budget, Mapping):
        budget = {}

    def first(source: Mapping[str, Any], names: tuple[str, ...],
              default: Any) -> Any:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return default

    file_bytes = _backend_int(
        first(budget, ("bytes", "file_bytes", "budget_bytes"),
              first(snapshot, ("budget_bytes", "file_bytes"),
                    DEFAULT_BUDGET_BYTES)),
        "budget bytes", 0, MAX_BUDGET_BYTES, DEFAULT_BUDGET_BYTES)
    max_bytes = _backend_int(
        first(budget, ("max_bytes", "ceiling_bytes"),
              first(snapshot, ("max_bytes",), MAX_BUDGET_BYTES)),
        "budget max_bytes", 0, MAX_BUDGET_BYTES, MAX_BUDGET_BYTES)
    used_bytes = _backend_int(
        first(budget, ("used_bytes", "used"),
              first(snapshot, ("used_bytes",), 0)),
        "budget used_bytes", 0, MAX_BUDGET_BYTES, 0)
    exceeded = bool(file_bytes and used_bytes > file_bytes)
    return {
        "bytes": file_bytes,
        "used_bytes": used_bytes,
        "max_bytes": max_bytes,
        "hard_ceiling_bytes": MAX_BUDGET_BYTES,
        "bounded": True,
        "exceeded": exceeded,
        "within_budget": not exceeded,
    }


def _normalize_head_metrics(value: Any, head: int) -> dict[str, Any]:
    base = _backend_mapping(value, "head_metrics()")
    nested = base.get("metrics")
    if isinstance(nested, Mapping):
        base = nested
    if "head" in base:
        reported = _input_head(base.get("head"))
        if reported != head:
            raise FieldBackendError(
                "backend head_metrics() returned a different head")
    result: dict[str, Any] = {"head": head}
    for key in ("score", "weight", "phase", "amplitude", "incoming",
                "outgoing", "degree", "mean_weight"):
        if key in base and base[key] is not None:
            result[key] = _backend_float(base[key], "head %s" % key)
    for key in ("state", "generation", "pair_count", "leaf_count",
                "relation_count"):
        if key in base and base[key] is not None:
            result[key] = _backend_int(base[key], "head %s" % key)
    if "active" in base and base["active"] is not None:
        result["active"] = _backend_bool(base["active"], "head active")
    elif "score" in result or "state" in result:
        result["active"] = bool(result.get("score", 0.0) != 0.0 or
                                result.get("state", 0) != 0)
    return result


def _normalize_field_metrics(value: Any) -> dict[str, Any]:
    base = _backend_mapping(value, "field_metrics()")
    nested = base.get("field")
    if isinstance(nested, Mapping):
        base = nested
    result: dict[str, Any] = {
        "head_count": _backend_int(base.get("head_count"), "head_count",
                                   0, HEAD_COUNT, HEAD_COUNT),
    }
    active: list[int] | None = None
    if "active_heads" in base and base["active_heads"] is not None:
        active = _active_heads(base["active_heads"], FieldBackendError)
        result["active_heads"] = active
    if "active_count" in base and base["active_count"] is not None:
        result["active_count"] = _backend_int(
            base["active_count"], "active_count", 0, HEAD_COUNT)
    elif active is not None:
        result["active_count"] = len(active)
    if "coverage" in base and base["coverage"] is not None:
        result["coverage"] = _backend_ratio(base["coverage"], "coverage")
    elif active is not None and result["head_count"]:
        result["coverage"] = round(
            len(active) / float(result["head_count"]), 8)
    for key in ("mean_weight", "max_share", "score_sum"):
        if key in base and base[key] is not None:
            result[key] = _backend_float(base[key], key)
    for key in ("pair_cells", "occupied_pairs", "leaves", "relations",
                "features", "generation"):
        if key in base and base[key] is not None:
            result[key] = _backend_int(base[key], key)
    return result


def _normalize_hits(value: Any, top_k: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        raw = value.get("hits")
        if raw is None:
            raw = value.get("results")
        if raw is None:
            if "head" in value or "score" in value:
                raw = [value]
            else:
                return []
    else:
        raw = value
    if isinstance(raw, (str, bytes, bytearray)) or \
            not isinstance(raw, Sequence):
        raise FieldBackendError("backend resonance hits must be a sequence")
    hits: list[dict[str, Any]] = []
    for item in list(raw)[:top_k]:
        if isinstance(item, Mapping):
            hit: dict[str, Any] = {}
            if item.get("head") is not None:
                hit["head"] = _input_head(item.get("head"))
            for key in ("score", "weight", "phase", "amplitude"):
                if key in item and item[key] is not None:
                    hit[key] = _backend_float(item[key], "hit %s" % key)
            for key in ("text", "kind", "leaf_id", "feature_id"):
                if key in item and item[key] is not None:
                    text = _backend_hit_text(item[key], "hit %s" % key)
                    if text:
                        hit[key] = text
            if hit:
                hits.append(hit)
        elif isinstance(item, int) and not isinstance(item, bool):
            hits.append({"head": _input_head(item)})
        else:
            raise FieldBackendError(
                "backend resonance hits must be objects or head integers")
    return hits


def _policy_flags() -> dict[str, Any]:
    return {
        "canonical_merge": {"enabled": False, "attempted": False},
        "canonical_merge_enabled": False,
        "canonical_promotion": False,
        "canonical_promotion_enabled": False,
        "promotion_enabled": False,
    }


def _json_safe(value: dict[str, Any], operation: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False,
                             separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise FieldBackendError(
            "%s produced non-JSON-safe output" % operation) from exc
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise FieldBackendError(
            "%s output exceeds %d bytes" % (operation, MAX_OUTPUT_BYTES))
    return value


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------
def status(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return a bounded read-only head-field snapshot."""
    args = _params(params, frozenset({"limit"}))
    limit = _input_limit(args.get("limit", DEFAULT_STATUS_LIMIT),
                         "limit", MAX_STATUS_LIMIT)
    snapshot = _snapshot(backend)
    generation = _generation_from_snapshot(snapshot, backend)
    field = snapshot.get("field")
    if not isinstance(field, Mapping):
        field = {}

    raw_heads = snapshot.get("heads")
    if raw_heads is None:
        raw_heads = field.get("heads")
    heads = _normalize_heads(raw_heads)

    raw_active = snapshot.get("active_heads")
    if raw_active is None:
        raw_active = field.get("active_heads")
    if raw_active is None:
        all_active = sorted(item["head"] for item in heads if item["active"])
    else:
        all_active = _active_heads(raw_active, FieldBackendError)
    active = all_active[:limit]
    top = sorted(heads, key=lambda item: (-item["score"], item["head"]))
    top = [{"head": item["head"], "score": item["score"],
            "active": item["active"]} for item in top[:min(limit, 16)]]
    budget = _normalize_budget(snapshot)
    opened = _backend_bool(snapshot.get("opened"), "opened", False)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "opened": opened,
        "generation": generation,
        "field": {
            "head_count": HEAD_COUNT,
            "materialized_heads": len(heads),
            "active_count": len(all_active),
            "active_heads": active,
            "active_truncated": len(all_active) > len(active),
            "coverage": round(len(all_active) / float(HEAD_COUNT), 8),
            "top": top,
        },
        "active_heads": active,
        "heads": heads,
        "budget": budget,
        "budget_exceeded": budget["exceeded"],
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "status")


def activate(params: Any, *, backend: Any) -> dict[str, Any]:
    """Apply exactly one bounded local head activation after approval."""
    args = _params(params, frozenset({"approved", "head", "delta"}))
    _require_approved(args)
    head = _input_head(_required(args, "head"))
    delta = _input_delta(args.get("delta", 1.0))

    previous_generation = _current_generation(backend)
    value = _backend_mapping(
        _call(backend, "activate", head, delta), "activate()")
    accepted = _backend_bool(value.get("accepted"), "accepted", True)
    reasons = _backend_reasons(value.get("reasons"))
    default_generation = previous_generation + 1 if accepted \
        else previous_generation
    generation = _input_generation(
        value.get("generation", default_generation), "generation")
    if accepted and generation <= previous_generation:
        raise FieldInvariantError(
            "accepted activation must advance the main generation")
    if not accepted and generation != previous_generation:
        raise FieldInvariantError(
            "denied activation must not change the main generation")

    receipt: dict[str, Any] = {
        "operation": "activate",
        "accepted": accepted,
        "applied": bool(accepted),
        "head": head,
        "delta": delta,
        "previous_generation": previous_generation,
        "generation": generation,
        "reasons": reasons,
    }
    if value.get("score") is not None:
        receipt["score"] = _backend_float(value["score"], "activation score")
    if value.get("state") is not None:
        receipt["state"] = _backend_int(value["state"], "activation state",
                                        0, MAX_U32)
    if value.get("active") is not None:
        receipt["active"] = _backend_bool(value["active"], "activation active")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "activate",
        "accepted": accepted,
        "head": head,
        "delta": delta,
        "previous_generation": previous_generation,
        "generation": generation,
        "reasons": reasons,
        "receipt": receipt,
        "read_only": False,
        "writes_main": accepted,
    }
    result.update(_policy_flags())
    return _json_safe(result, "activate")


def propose(params: Any, *, backend: Any) -> dict[str, Any]:
    """Stage a scratch proposal without touching main head state."""
    args = _params(params, frozenset({"head", "delta", "reason"}))
    head = _input_head(_required(args, "head"))
    delta = _input_delta(args.get("delta", 1.0))
    reason = _input_reason(args.get("reason", ""))

    previous_generation = _current_generation(backend)
    value = _backend_mapping(
        _call(backend, "propose", head, delta, reason), "propose()")
    token = _input_token(value.get("token"))
    reasons = _backend_reasons(value.get("reasons"))
    if not reasons:
        reasons = ["scratch_only"]
    after_generation = _current_generation(backend)
    if after_generation != previous_generation:
        raise FieldInvariantError(
            "propose must not write main state or advance generation")

    receipt: dict[str, Any] = {
        "operation": "propose",
        "head": head,
        "delta": delta,
        "reason": reason,
        "token": token,
        "generation": previous_generation,
        "reasons": reasons,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "propose",
        "staged": True,
        "accepted": True,
        "token": token,
        "reasons": reasons,
        "head": head,
        "delta": delta,
        "generation": previous_generation,
        "main_generation": previous_generation,
        "main_unchanged": True,
        "scratch_only": True,
        "writes_main": False,
        "receipt": receipt,
        "read_only": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "propose")


def commit(params: Any, *, backend: Any) -> dict[str, Any]:
    """Consume an approved scratch proposal at an exact generation."""
    args = _params(params, frozenset({"approved", "token",
                                      "expected_generation"}))
    _require_approved(args)
    token = _input_token(_required(args, "token"))
    expected = _input_generation(_required(args, "expected_generation"))

    current = _current_generation(backend)
    if expected != current:
        return _commit_denied(token, expected, current,
                              ["generation_mismatch"])

    try:
        value = _backend_mapping(
            _call(backend, "commit", token, expected), "commit()")
    except FieldConflictError as exc:
        reasons = _backend_reasons(exc.reasons) or ["commit_denied"]
        after = _current_generation(backend)
        if after != current:
            raise FieldInvariantError(
                "denied commit must not change the main generation") from exc
        return _commit_denied(token, expected, after, reasons)

    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise FieldBackendError("backend commit() must return accepted=true/false")
    reasons = _backend_reasons(value.get("reasons"))
    if not accepted:
        after = _current_generation(backend)
        if after != current:
            raise FieldInvariantError(
                "denied commit must not change the main generation")
        return _commit_denied(token, expected, after,
                              reasons or ["commit_denied"])

    generation = _input_generation(
        value.get("generation", current + 1), "generation")
    if generation <= current:
        raise FieldInvariantError(
            "accepted commit must advance the main generation")
    head = value.get("head")
    delta = value.get("delta")
    receipt = {
        "operation": "commit",
        "accepted": True,
        "token": token,
        "expected_generation": expected,
        "previous_generation": current,
        "generation": generation,
        "head": None if head is None else _input_head(head),
        "delta": None if delta is None else _input_delta(delta),
        "reasons": reasons,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "commit",
        "accepted": True,
        "denied": False,
        "token": token,
        "expected_generation": expected,
        "previous_generation": current,
        "generation": generation,
        "head": receipt["head"],
        "delta": receipt["delta"],
        "reasons": reasons,
        "receipt": receipt,
        "read_only": False,
        "writes_main": True,
    }
    result.update(_policy_flags())
    return _json_safe(result, "commit")


def _commit_denied(token: str, expected: int, generation: int,
                   reasons: Sequence[str]) -> dict[str, Any]:
    bounded_reasons = _backend_reasons(list(reasons)) or ["commit_denied"]
    receipt = {
        "operation": "commit",
        "accepted": False,
        "token": token,
        "expected_generation": expected,
        "generation": generation,
        "reasons": bounded_reasons,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "commit",
        "accepted": False,
        "denied": True,
        "token": token,
        "expected_generation": expected,
        "previous_generation": generation,
        "generation": generation,
        "reasons": bounded_reasons,
        "receipt": receipt,
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "commit")


def query(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return read-only head/field metrics or a bounded resonance lookup."""
    args = _params(params, frozenset({"head", "text", "top_k"}))
    if "head" in args and "text" in args:
        raise FieldValidationError(
            "query accepts either head or text, not both")
    if "top_k" in args and "text" not in args:
        raise FieldValidationError("top_k requires text")

    previous_generation = _current_generation(backend)
    if "head" in args:
        mode = "head"
        head = _input_head(args["head"])
        metrics = _normalize_head_metrics(
            _call(backend, "head_metrics", head), head)
        payload: dict[str, Any] = {"head": head, "metrics": metrics}
    elif "text" in args:
        mode = "resonance"
        text = _input_text(args["text"])
        top_k = _input_limit(args.get("top_k", DEFAULT_TOP_K),
                             "top_k", MAX_TOP_K)
        hits = _normalize_hits(_call(backend, "resonance", text, top_k),
                               top_k)
        payload = {"text": text, "top_k": top_k, "hits": hits}
    else:
        mode = "field"
        payload = {"field": _normalize_field_metrics(
            _call(backend, "field_metrics"))}

    after_generation = _current_generation(backend)
    if after_generation != previous_generation:
        raise FieldInvariantError("query must be read-only")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "query",
        "mode": mode,
        "generation": previous_generation,
        "read_only": True,
        "writes_main": False,
    }
    result.update(payload)
    result.update(_policy_flags())
    return _json_safe(result, "query")


HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "status": status,
    "activate": activate,
    "propose": propose,
    "commit": commit,
    "query": query,
}


def build_handlers(backend: Any) -> dict[str, Callable[[Any],
                                                       dict[str, Any]]]:
    """Bind all handlers to one backend for later plugin registration."""
    return {name: partial(handler, backend=backend)
            for name, handler in HANDLERS.items()}


__all__ = (
    "SCHEMA", "HEAD_COUNT", "MAX_DELTA", "MIN_DELTA", "MAX_TEXT_BYTES",
    "MAX_TOKEN_BYTES", "MAX_REASON_BYTES", "MAX_REASONS", "MAX_TOP_K",
    "DEFAULT_TOP_K", "MAX_STATUS_LIMIT", "DEFAULT_STATUS_LIMIT",
    "MAX_OUTPUT_BYTES", "DEFAULT_BUDGET_BYTES", "MAX_BUDGET_BYTES",
    "FieldToolError", "FieldValidationError", "FieldApprovalError",
    "FieldBackendError", "FieldConflictError", "FieldInvariantError",
    "status", "activate", "propose", "commit", "query", "HANDLERS",
    "build_handlers",
)
