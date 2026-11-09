"""Conservative self-modification gate for the 256 resident heads.

Every candidate change is first written to an L0-L9 scratch layer.  A small,
bounded Boolean rule table then evaluates the staged proposal.  Only a literal
``approved=True``, an accepted evaluation and an exact main-generation
compare-and-swap can produce exactly one main-register write.

The gate is deliberately fail-closed:

* unknown heads, layers and operations are denied;
* non-finite, oversized or non-JSON values are denied;
* any canonical merge/promotion wording or flag is denied;
* a caller-supplied logic object may add contradictions but can never bypass
  the mandatory rules;
* every denial is a structured result and performs no scratch or main write.

No canonical merge, canonical promotion or six-runner barrier is performed or
represented here.  Receipts always report those authorities as false/absent.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import secrets
import threading
from typing import Any, Callable


SCHEMA = "nanlm.resident.selfmod-gate/1"
PROPOSAL_SCHEMA = "nanlm.resident.selfmod-proposal/1"
COMMIT_SCHEMA = "nanlm.resident.selfmod-commit/1"

HEAD_COUNT = 256
LAYER_COUNT = 10
LAYER_NAMES = (
    "bit", "nibble", "byte", "phonetic", "morpheme",
    "lexical", "word_completion", "sentence", "paragraph", "narrative",
)

MAX_RULES = 32
MAX_VALUE_BYTES = 4096
MAX_VALUE_DEPTH = 8
MAX_VALUE_NODES = 512
MAX_ABS_NUMBER = 1_000_000_000
MAX_REASON_CHARS = 512
MAX_TOKEN_CHARS = 128
MAX_GENERATION = 2 ** 63 - 1

OPERATION_ALLOWLIST = frozenset({
    "set", "add", "remove", "update", "replace", "increment", "decrement",
    "reset", "append", "write", "modify", "tune", "learn",
})

# Stable denial codes.  They are intentionally greppable and never carry a
# mutation receipt.
UNKNOWN_HEAD = "unknown_head"
UNKNOWN_LAYER = "unknown_layer"
UNKNOWN_OPERATION = "unknown_operation"
UNAPPROVED = "unapproved"
STALE_GENERATION = "stale_generation"
LOGIC_CONTRADICTION = "logic_contradiction"
DUPLICATE_TOKEN = "duplicate_token"
OVERSIZED_VALUE = "oversized_value"
CANONICAL_MERGE_ATTEMPT = "canonical_merge_attempt"
INVALID_VALUE = "invalid_value"
INVALID_REASON = "invalid_reason"
INVALID_TOKEN = "invalid_token"
UNKNOWN_TOKEN = "unknown_token"
NOT_EVALUATED = "not_evaluated"
NO_MAIN_BACKEND = "no_main_backend"
MAIN_GENERATION_UNAVAILABLE = "main_generation_unavailable"
SCRATCH_WRITE_FAILED = "scratch_write_failed"
SCRATCH_WRITE_REJECTED = "scratch_write_rejected"
MAIN_WRITE_FAILED = "main_write_failed"
MAIN_WRITE_REJECTED = "main_write_rejected"

DENIAL_CODES = frozenset({
    UNKNOWN_HEAD, UNKNOWN_LAYER, UNKNOWN_OPERATION, UNAPPROVED,
    STALE_GENERATION, LOGIC_CONTRADICTION, DUPLICATE_TOKEN,
    OVERSIZED_VALUE, CANONICAL_MERGE_ATTEMPT, INVALID_VALUE,
    INVALID_REASON, INVALID_TOKEN, UNKNOWN_TOKEN, NOT_EVALUATED,
    NO_MAIN_BACKEND, MAIN_GENERATION_UNAVAILABLE, SCRATCH_WRITE_FAILED,
    SCRATCH_WRITE_REJECTED, MAIN_WRITE_FAILED, MAIN_WRITE_REJECTED,
})

DENIAL_MESSAGES = {
    UNKNOWN_HEAD: "head must be an integer in 0..255",
    UNKNOWN_LAYER: "layer must be an integer 0..9 (or its L0-L9 name)",
    UNKNOWN_OPERATION: "operation is not in the bounded allowlist",
    UNAPPROVED: "commit requires literal approved=True",
    STALE_GENERATION: "main generation does not match expected_generation",
    LOGIC_CONTRADICTION: "Boolean logic/invariants rejected the proposal",
    DUPLICATE_TOKEN: "proposal token is unknown, consumed or duplicated",
    OVERSIZED_VALUE: "value exceeds the bounded self-modification limits",
    CANONICAL_MERGE_ATTEMPT: "canonical merge/promotion is forbidden",
    INVALID_VALUE: "value must be finite, JSON-safe and structurally bounded",
    INVALID_REASON: "reason must be a short non-empty string",
    INVALID_TOKEN: "token must be a short non-empty string",
    UNKNOWN_TOKEN: "token does not identify a staged proposal",
    NOT_EVALUATED: "proposal must be evaluated before commit",
    NO_MAIN_BACKEND: "commit requires a main backend",
    MAIN_GENERATION_UNAVAILABLE: "main backend does not expose a generation",
    SCRATCH_WRITE_FAILED: "scratch backend refused or failed the proposal write",
    SCRATCH_WRITE_REJECTED: "scratch backend rejected the proposal write",
    MAIN_WRITE_FAILED: "main backend failed the single commit write",
    MAIN_WRITE_REJECTED: "main backend rejected the commit write",
}

_CANONICAL_MARKERS = ("canonical", "merge", "promot")
_MISSING = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_size(value: Any) -> int:
    try:
        return len(_canonical_json(value))
    except (TypeError, ValueError, RecursionError):
        return MAX_VALUE_BYTES + 1


def _digest(value: Any) -> str:
    try:
        return hashlib.sha256(_canonical_json(value)).hexdigest()
    except (TypeError, ValueError, RecursionError):
        return ""


def _safe_flags() -> dict[str, Any]:
    """The complete non-authority surface shared by every result."""
    return {
        "canonical": False,
        "canonical_merge": False,
        "canonical_merge_enabled": False,
        "canonical_authority": False,
        "canonical_commit": False,
        "canonical_state_modified": False,
        "merge_enabled": False,
        "promotion_allowed": False,
        "six_runner_barrier": "absent",
    }


def normalize_layer(layer: Any) -> int:
    """Return an L0-L9 index or raise ``ValueError``.

    Integers are the canonical form.  ``"L3"``, ``"3"`` and the layer names
    used by the scratch register (``"lexical"``, ...) are accepted as
    conveniences and normalised before storage.
    """
    number: int
    if isinstance(layer, bool):
        raise ValueError(DENIAL_MESSAGES[UNKNOWN_LAYER])
    if isinstance(layer, int):
        number = layer
    elif isinstance(layer, str):
        text = layer.strip().casefold()
        if text.startswith("l") and text[1:].isdigit():
            number = int(text[1:])
        elif text.isdigit():
            number = int(text)
        elif text in LAYER_NAMES:
            number = LAYER_NAMES.index(text)
        else:
            raise ValueError(DENIAL_MESSAGES[UNKNOWN_LAYER])
    else:
        raise ValueError(DENIAL_MESSAGES[UNKNOWN_LAYER])
    if not 0 <= number < LAYER_COUNT:
        raise ValueError(DENIAL_MESSAGES[UNKNOWN_LAYER])
    return number


def normalize_operation(operation: Any) -> Any:
    if not isinstance(operation, str):
        return operation
    return operation.strip().casefold()


def _marker_in(text: str) -> bool:
    folded = str(text).casefold()
    return any(marker in folded for marker in _CANONICAL_MARKERS)


def _value_attempts_canonical(value: Any, depth: int = 0) -> bool:
    if depth > MAX_VALUE_DEPTH:
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _marker_in(key):
                if item is not False and item is not None and item != 0 and item != "":
                    return True
            if _value_attempts_canonical(item, depth + 1):
                return True
    elif isinstance(value, (list, tuple)):
        for item in value:
            if _value_attempts_canonical(item, depth + 1):
                return True
    elif isinstance(value, str):
        folded = value.strip().casefold()
        if folded in {
            "canonical", "canonical_merge", "merge", "promote", "promotion",
        }:
            return True
        if "canonical merge" in folded:
            return True
    return False


def _inspect_value(value: Any) -> str | None:
    """Return a denial code for a value, or ``None`` when bounded."""
    nodes = [0]

    def walk(item: Any, depth: int) -> str | None:
        nodes[0] += 1
        if nodes[0] > MAX_VALUE_NODES:
            return OVERSIZED_VALUE
        if depth > MAX_VALUE_DEPTH:
            return OVERSIZED_VALUE
        if item is None or isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                return INVALID_VALUE
            if abs(item) > MAX_ABS_NUMBER:
                return OVERSIZED_VALUE
            return None
        if isinstance(item, str):
            if len(item.encode("utf-8")) > MAX_VALUE_BYTES:
                return OVERSIZED_VALUE
            return None
        if isinstance(item, (list, tuple)):
            if len(item) > MAX_VALUE_NODES:
                return OVERSIZED_VALUE
            for child in item:
                issue = walk(child, depth + 1)
                if issue is not None:
                    return issue
            return None
        if isinstance(item, Mapping):
            if len(item) > MAX_VALUE_NODES:
                return OVERSIZED_VALUE
            for key, child in item.items():
                if not isinstance(key, str):
                    return INVALID_VALUE
                if len(key.encode("utf-8")) > MAX_VALUE_BYTES:
                    return OVERSIZED_VALUE
                issue = walk(child, depth + 1)
                if issue is not None:
                    return issue
            return None
        return INVALID_VALUE

    issue = walk(value, 0)
    if issue is not None:
        return issue
    if _json_size(value) > MAX_VALUE_BYTES:
        return OVERSIZED_VALUE
    return None


def _check_head(proposal: "_Proposal") -> str | None:
    if isinstance(proposal.head, bool) or not isinstance(proposal.head, int):
        return UNKNOWN_HEAD
    if not 0 <= proposal.head < HEAD_COUNT:
        return UNKNOWN_HEAD
    return None


def _check_layer(proposal: "_Proposal") -> str | None:
    if isinstance(proposal.layer, bool) or not isinstance(proposal.layer, int):
        return UNKNOWN_LAYER
    if not 0 <= proposal.layer < LAYER_COUNT:
        return UNKNOWN_LAYER
    return None


def _check_canonical(proposal: "_Proposal") -> str | None:
    if isinstance(proposal.operation, str) and _marker_in(proposal.operation):
        return CANONICAL_MERGE_ATTEMPT
    if isinstance(proposal.reason, str) and _marker_in(proposal.reason):
        return CANONICAL_MERGE_ATTEMPT
    if _value_attempts_canonical(proposal.value):
        return CANONICAL_MERGE_ATTEMPT
    return None


def _check_operation(proposal: "_Proposal") -> str | None:
    if not isinstance(proposal.operation, str):
        return UNKNOWN_OPERATION
    if proposal.operation not in OPERATION_ALLOWLIST:
        return UNKNOWN_OPERATION
    return None


def _check_reason(proposal: "_Proposal") -> str | None:
    if not isinstance(proposal.reason, str):
        return INVALID_REASON
    reason = proposal.reason.strip()
    if not reason or len(reason) > MAX_REASON_CHARS:
        return INVALID_REASON
    return None


def _check_value(proposal: "_Proposal") -> str | None:
    return _inspect_value(proposal.value)


@dataclass(frozen=True)
class Rule:
    """One mandatory Boolean rule in the bounded table."""

    name: str
    check: Callable[["_Proposal"], str | None]


DEFAULT_RULE_TABLE: tuple[Rule, ...] = (
    Rule("head_range", _check_head),
    Rule("layer_range", _check_layer),
    Rule("canonical_forbidden", _check_canonical),
    Rule("operation_allowlist", _check_operation),
    Rule("reason_present", _check_reason),
    Rule("value_bounded", _check_value),
)

# Convenience aliases for callers that use the more explicit names.
ALLOWED_OPERATIONS = OPERATION_ALLOWLIST
RULE_TABLE = DEFAULT_RULE_TABLE


@dataclass(frozen=True)
class _Proposal:
    token: str
    head: Any
    layer: Any
    operation: Any
    value: Any
    reason: Any
    created_at: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "token": self.token,
            "head": self.head,
            "layer": self.layer,
            "layer_name": (LAYER_NAMES[self.layer]
                           if isinstance(self.layer, int)
                           and 0 <= self.layer < LAYER_COUNT else None),
            "operation": self.operation,
            "value": deepcopy(self.value),
            "reason": self.reason,
            "created_at": self.created_at,
            "state": "staged_scratch",
            **_safe_flags(),
        }

    def as_logic_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "token": self.token,
            "head": self.head,
            "layer": self.layer,
            "layer_name": (LAYER_NAMES[self.layer]
                           if isinstance(self.layer, int)
                           and 0 <= self.layer < LAYER_COUNT else None),
            "operation": self.operation,
            "value": deepcopy(self.value),
            "reason": self.reason,
            "created_at": self.created_at,
        }


class SelfModificationGate:
    """Stage, evaluate and conservatively commit one head change at a time.

    ``scratch_backend`` must expose a write method understood by the adapter
    (``append``/``write``/``put``/``store``, a mapping with ``entries``, or a
    list).  ``main_backend`` is optional but required for commit; it must
    expose a ``generation`` (property, callable or mapping key) and a write
    method such as ``commit``/``write``/``apply``/``append``.

    ``logic`` is optional.  It may be a callable, an object with
    ``evaluate``/``check``, or a mapping of Boolean invariants.  It can only
    reject a proposal; mandatory rules always run.
    """

    RULE_TABLE = DEFAULT_RULE_TABLE
    MAX_RULES = MAX_RULES
    MAX_VALUE_BYTES = MAX_VALUE_BYTES
    HEAD_COUNT = HEAD_COUNT
    LAYER_COUNT = LAYER_COUNT
    OPERATION_ALLOWLIST = OPERATION_ALLOWLIST

    def __init__(self, scratch_backend: Any, main_backend: Any = None,
                 logic: Any = None):
        if scratch_backend is None:
            raise ValueError("scratch_backend is required")
        self.scratch_backend = scratch_backend
        self.main_backend = main_backend
        self.logic = logic
        self._rules = tuple(self.RULE_TABLE)
        self.rule_table = self._rules
        if len(self._rules) > MAX_RULES:
            raise ValueError("rule table exceeds MAX_RULES")
        if isinstance(logic, Mapping) and len(logic) > MAX_RULES - len(self._rules):
            raise ValueError("custom logic rules exceed the bounded table")
        self._lock = threading.RLock()
        self._proposals: dict[str, _Proposal] = {}
        self._evaluations: dict[str, dict[str, Any]] = {}
        self._committed: dict[str, dict[str, Any]] = {}
        self._consumed: set[str] = set()
        self._main_fallback_generation = 0
        self._scratch_write_count = 0
        self._main_write_count = 0

    # ------------------------------------------------------------- public API
    def propose(self, head: int, layer: int | str, operation: str,
                value: Any, *, reason: str) -> Any:
        """Stage a proposal in scratch and return its token.

        Invalid proposals return a structured denial and write nowhere.
        """
        with self._lock:
            if isinstance(operation, str):
                operation_value: Any = normalize_operation(operation)
            else:
                operation_value = operation
            try:
                layer_value: Any = normalize_layer(layer)
            except (TypeError, ValueError):
                layer_value = layer
            try:
                value_copy = deepcopy(value)
            except Exception:  # pragma: no cover - unusual hostile objects
                return self._denial(INVALID_VALUE, details={
                    "head": head, "layer": layer_value,
                    "operation": operation_value,
                })
            proposal = _Proposal(
                token="", head=head, layer=layer_value,
                operation=operation_value, value=value_copy,
                reason=reason, created_at=_now(),
            )
            code = self._first_failure(proposal)
            if code is not None:
                return self._denial(code, details={
                    "head": head, "layer": layer_value,
                    "operation": operation_value,
                })

            token = self._new_token()
            if (not isinstance(token, str) or not token
                    or len(token) > MAX_TOKEN_CHARS):
                return self._denial(INVALID_TOKEN)
            if token in self._proposals or token in self._evaluations \
                    or token in self._committed or token in self._consumed:
                return self._denial(DUPLICATE_TOKEN, token=token)

            proposal = _Proposal(
                token=token, head=proposal.head, layer=proposal.layer,
                operation=proposal.operation, value=deepcopy(proposal.value),
                reason=proposal.reason, created_at=proposal.created_at,
            )
            payload = proposal.as_payload()
            try:
                result = self._write_scratch(payload)
            except Exception as exc:
                return self._denial(SCRATCH_WRITE_FAILED, token=token, details={
                    "error": type(exc).__name__,
                })
            rejection = self._result_rejection(result)
            if rejection is not None:
                return self._denial(SCRATCH_WRITE_REJECTED, token=token, details={
                    "backend_reason": rejection,
                })

            self._proposals[token] = proposal
            self._evaluations.pop(token, None)
            self._scratch_write_count += 1
            return token

    def evaluate(self, token: str) -> dict[str, Any]:
        """Run the Boolean rule table and optional logic; never write."""
        with self._lock:
            if not isinstance(token, str) or not token:
                return self._evaluation(False, [DENIAL_MESSAGES[INVALID_TOKEN]],
                                        {"denial": INVALID_TOKEN})
            cached = self._evaluations.get(token)
            if cached is not None:
                return deepcopy(cached)
            proposal = self._proposals.get(token)
            if proposal is None:
                if token in self._committed or token in self._consumed:
                    stored = self._committed.get(token)
                    if stored is not None and isinstance(stored.get("evaluation"), dict):
                        return deepcopy(stored["evaluation"])
                return self._evaluation(
                    False, [DENIAL_MESSAGES[UNKNOWN_TOKEN]],
                    {"denial": UNKNOWN_TOKEN, "token": token})
            return self._evaluate_proposal(proposal)

    def commit(self, token: str, *, approved: bool = False,
               expected_generation: int) -> dict[str, Any]:
        """Commit once to main after approval, evaluation and exact CAS."""
        with self._lock:
            if not isinstance(token, str) or not token:
                return self._denial(INVALID_TOKEN)
            if token in self._committed or token in self._consumed:
                return self._denial(DUPLICATE_TOKEN, token=token)
            if approved is not True:
                return self._denial(UNAPPROVED, token=token)
            proposal = self._proposals.get(token)
            if proposal is None:
                return self._denial(DUPLICATE_TOKEN, token=token)
            evaluation = self._evaluations.get(token)
            if evaluation is None:
                return self._denial(NOT_EVALUATED, token=token)
            if not evaluation.get("accepted"):
                metrics = evaluation.get("metrics") or {}
                code = metrics.get("denial") or LOGIC_CONTRADICTION
                if code not in DENIAL_CODES:
                    code = LOGIC_CONTRADICTION
                return self._denial(
                    code, token=token,
                    message=(evaluation.get("reasons") or [None])[0],
                    details={"evaluation": deepcopy(evaluation)},
                )

            # Re-run mandatory rules defensively; a staged proposal is never
            # trusted merely because it was once written to scratch.
            code = self._first_failure(proposal)
            if code is not None:
                return self._denial(code, token=token)

            if isinstance(expected_generation, bool) \
                    or not isinstance(expected_generation, int) \
                    or not 0 <= expected_generation < MAX_GENERATION:
                return self._denial(
                    STALE_GENERATION, token=token,
                    message="expected_generation must be an integer >= 0",
                    details={"expected_generation": expected_generation},
                )
            if self.main_backend is None:
                return self._denial(NO_MAIN_BACKEND, token=token)
            observed = self._read_main_generation()
            if observed is None:
                return self._denial(
                    MAIN_GENERATION_UNAVAILABLE, token=token)
            if observed != expected_generation:
                return self._denial(
                    STALE_GENERATION, token=token,
                    details={
                        "expected_generation": expected_generation,
                        "observed_generation": observed,
                    },
                )

            generation = expected_generation + 1
            record = {
                "schema": COMMIT_SCHEMA,
                "token": token,
                "head": proposal.head,
                "layer": proposal.layer,
                "layer_name": LAYER_NAMES[proposal.layer],
                "operation": proposal.operation,
                "value": deepcopy(proposal.value),
                "reason": proposal.reason,
                "approved": True,
                "expected_generation": expected_generation,
                "previous_generation": observed,
                "generation": generation,
                "evaluation_digest": _digest(evaluation),
                "created_at": _now(),
                **_safe_flags(),
            }
            try:
                result = self._write_main(record)
            except Exception as exc:
                return self._denial(MAIN_WRITE_FAILED, token=token, details={
                    "error": type(exc).__name__,
                })
            rejection = self._result_rejection(result)
            if rejection is not None:
                return self._denial(MAIN_WRITE_REJECTED, token=token, details={
                    "backend_reason": rejection,
                })

            reported = self._extract_generation(result)
            if reported is not None and reported < generation:
                return self._denial(
                    MAIN_WRITE_FAILED, token=token,
                    message="main backend reported a non-monotonic generation",
                    details={
                        "generation": reported,
                        "required_generation": generation,
                    },
                )
            after = self._read_main_generation()
            if after is not None and after < generation:
                # A plain object backend may expose a writable generation but
                # not advance it in its write method.  Advance it exactly once
                # here so the next CAS sees a monotonic register.
                try:
                    setattr(self.main_backend, "generation", generation)
                except (AttributeError, TypeError, ValueError):
                    pass
                after = self._read_main_generation()
            committed_generation = max(
                generation, reported or 0, after or 0)

            self._consumed.add(token)
            self._main_write_count += 1
            receipt = {
                "schema": SCHEMA,
                "status": "committed",
                "committed": True,
                "applied": True,
                "duplicate": False,
                "accepted": True,
                "denied": False,
                "code": None,
                "denial": None,
                "token": token,
                "head": proposal.head,
                "layer": proposal.layer,
                "layer_name": LAYER_NAMES[proposal.layer],
                "operation": proposal.operation,
                "value_digest": _digest(proposal.value),
                "approved": True,
                "expected_generation": expected_generation,
                "previous_generation": observed,
                "generation": committed_generation,
                "main_generation": committed_generation,
                "generation_cas": {
                    "expected": expected_generation,
                    "observed": observed,
                    "result": committed_generation,
                    "ok": observed == expected_generation,
                },
                "main_writes": 1,
                "scratch_writes": 1,
                "reasons": [],
                "metrics": {
                    "rules_total": len(self._rules),
                    "main_writes": 1,
                    "scratch_writes": 1,
                    "value_digest": _digest(proposal.value),
                },
                **_safe_flags(),
            }
            self._committed[token] = receipt
            return deepcopy(receipt)

    def status(self) -> dict[str, Any]:
        """Return a bounded, non-authority status view."""
        with self._lock:
            return {
                "schema": SCHEMA,
                "head_count": HEAD_COUNT,
                "layer_count": LAYER_COUNT,
                "rule_count": len(self._rules),
                "max_rules": MAX_RULES,
                "rules": [rule.name for rule in self._rules],
                "operation_allowlist": sorted(OPERATION_ALLOWLIST),
                "proposals": len(self._proposals),
                "evaluations": len(self._evaluations),
                "committed": len(self._committed),
                "scratch_writes": self._scratch_write_count,
                "main_writes": self._main_write_count,
                "main_backend": self.main_backend is not None,
                "main_generation": self._read_main_generation(),
                **_safe_flags(),
            }

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    # --------------------------------------------------------------- internal
    def _new_token(self) -> str:
        return "smod_" + secrets.token_hex(16)

    def _rule_results(self, proposal: _Proposal) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for rule in self._rules:
            try:
                code = rule.check(proposal)
            except Exception:
                code = LOGIC_CONTRADICTION
            results.append({
                "name": rule.name,
                "passed": code is None,
                "code": code,
            })
        return results

    def _first_failure(self, proposal: _Proposal) -> str | None:
        for result in self._rule_results(proposal):
            if not result["passed"]:
                return str(result["code"])
        return None

    def _evaluate_proposal(self, proposal: _Proposal) -> dict[str, Any]:
        rule_results = self._rule_results(proposal)
        failures = [result for result in rule_results if not result["passed"]]
        reasons: list[str] = []
        code: str | None = None
        for failure in failures:
            code = code or str(failure["code"])
            reasons.append(DENIAL_MESSAGES.get(
                str(failure["code"]), str(failure["code"])))

        logic_passed, logic_reasons, logic_metrics = self._run_logic(proposal)
        if not logic_passed:
            code = code or LOGIC_CONTRADICTION
            reasons.extend(logic_reasons or [DENIAL_MESSAGES[LOGIC_CONTRADICTION]])

        accepted = code is None
        metrics = {
            "accepted": accepted,
            "token": proposal.token,
            "head": proposal.head,
            "layer": proposal.layer,
            "layer_name": (LAYER_NAMES[proposal.layer]
                           if isinstance(proposal.layer, int)
                           and 0 <= proposal.layer < LAYER_COUNT else None),
            "operation": proposal.operation,
            "value_bytes": _json_size(proposal.value),
            "value_digest": _digest(proposal.value),
            "rules_total": len(rule_results),
            "rules_passed": len(rule_results) - len(failures),
            "rules_failed": len(failures),
            "rules": rule_results,
            "logic": logic_metrics,
            "denial": code,
            "main_writes": 0,
            "scratch_writes": 0,
        }
        result = self._evaluation(accepted, [] if accepted else reasons, metrics)
        self._evaluations[proposal.token] = result
        return deepcopy(result)

    @staticmethod
    def _evaluation(accepted: bool, reasons: list[str],
                    metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "accepted": bool(accepted),
            "reasons": list(reasons),
            "metrics": dict(metrics),
        }

    def _denial(self, code: str, *, token: str | None = None,
                message: str | None = None,
                details: dict[str, Any] | None = None) -> dict[str, Any]:
        text = message or DENIAL_MESSAGES.get(code, code)
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "denied",
            "denied": True,
            "accepted": False,
            "committed": False,
            "code": code,
            "denial": code,
            "reasons": [text],
            "reason": text,
            "token": token,
            "main_writes": 0,
            "scratch_writes": 0,
            "metrics": {
                "denial": code,
                "main_writes": 0,
                "scratch_writes": 0,
            },
            **_safe_flags(),
        }
        if details:
            result["details"] = deepcopy(details)
        return result

    # --------------------------------------------------------------- scratch
    def _write_scratch(self, record: dict[str, Any]) -> Any:
        backend = self.scratch_backend
        if isinstance(backend, MutableMapping):
            self._append_mapping(backend, record)
            return {"accepted": True, "storage": "mapping"}
        if isinstance(backend, list):
            backend.append(deepcopy(record))
            return {"accepted": True, "storage": "list"}

        for name in ("append_proposal", "store_proposal", "write_proposal"):
            method = getattr(backend, name, None)
            if callable(method):
                return self._call_scratch_method(method, record)
        for name in ("append", "write", "put", "store", "record"):
            method = getattr(backend, name, None)
            if callable(method):
                return self._call_scratch_method(method, record)
        entries = getattr(backend, "entries", None)
        if isinstance(entries, list):
            entries.append(deepcopy(record))
            return {"accepted": True, "storage": "entries"}
        for name, method in getattr(backend, "__dict__", {}).items():
            if name.startswith("_") or not callable(method):
                continue
            folded = name.casefold()
            if any(marker in folded for marker in
                   ("append", "write", "store", "put", "record")):
                return self._call_scratch_method(method, record)
        if callable(backend):
            return backend(record)
        raise TypeError("scratch backend has no supported write method")

    @staticmethod
    def _append_mapping(mapping: MutableMapping, record: dict[str, Any]) -> None:
        for key in ("entries", "records", "writes"):
            target = mapping.get(key)
            if isinstance(target, list):
                target.append(deepcopy(record))
                return
        layers = mapping.get("layers")
        if isinstance(layers, MutableMapping):
            bucket = layers.setdefault(record["layer"], [])
            if isinstance(bucket, list):
                bucket.append(deepcopy(record))
                return
        mapping.setdefault("entries", [])
        mapping["entries"].append(deepcopy(record))
        by_token = mapping.get("by_token")
        if isinstance(by_token, MutableMapping):
            by_token[record["token"]] = deepcopy(record)

    @staticmethod
    def _call_scratch_method(method: Callable[..., Any],
                             record: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return method(record)
        parameters = signature.parameters
        names = set(parameters)
        has_var_kw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values())
        positional = [
            parameter for parameter in parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if {"lane", "writer"} <= names:
            return method(
                lane="learning_proposal", writer="nanlmind", payload=record)
        if "layer" in names and "payload" in names:
            kwargs: dict[str, Any] = {
                "layer": record["layer"],
                "head": record["head"],
                "payload": record,
            }
            if "leaf_id" in names:
                kwargs["leaf_id"] = record["token"]
            if "kind" in names:
                kwargs["kind"] = "selfmod-proposal"
            return method(**kwargs)
        if "layer" in names and "head" in names:
            kwargs = {"layer": record["layer"], "head": record["head"]}
            if "record" in names:
                kwargs["record"] = record
            elif "payload" in names:
                kwargs["payload"] = record
            elif {"operation", "value"} <= names:
                kwargs.update({
                    "operation": record["operation"],
                    "value": record["value"],
                })
                if "reason" in names:
                    kwargs["reason"] = record["reason"]
                if "token" in names:
                    kwargs["token"] = record["token"]
            if len(kwargs) > 2:
                return method(**kwargs)
        if "token" in names and "record" in names:
            return method(token=record["token"], record=record)
        if "record" in names:
            return method(record=record)
        if "payload" in names and len(positional) == 1:
            return method(payload=record)
        if len(positional) == 1:
            return method(record)
        if has_var_kw:
            return method(
                record=record, layer=record["layer"], head=record["head"],
                token=record["token"])
        if not names:
            return method()
        return method(record)

    # ------------------------------------------------------------------ main
    def _read_main_generation(self) -> int | None:
        backend = self.main_backend
        if backend is None:
            return None
        if isinstance(backend, Mapping):
            value = backend.get("generation", 0)
            parsed = self._as_generation(value)
            return parsed if parsed is not None else None

        value: Any = _MISSING
        for name in ("generation", "get_generation", "current_generation"):
            if hasattr(backend, name):
                value = getattr(backend, name)
                break
        if value is not _MISSING:
            if callable(value):
                try:
                    value = value()
                except Exception:
                    return None
            parsed = self._as_generation(value)
            if parsed is not None:
                return parsed

        for name in ("status", "stats"):
            method = getattr(backend, name, None)
            if not callable(method):
                continue
            try:
                data = method()
            except Exception:
                continue
            if isinstance(data, Mapping):
                parsed = self._as_generation(data.get("generation"))
                if parsed is not None:
                    return parsed
        return int(self._main_fallback_generation)

    @staticmethod
    def _as_generation(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if not 0 <= value < MAX_GENERATION:
            return None
        return value

    def _write_main(self, record: dict[str, Any]) -> Any:
        backend = self.main_backend
        if backend is None:
            raise RuntimeError("main backend is not configured")
        if isinstance(backend, MutableMapping):
            key = None
            for candidate in ("records", "writes", "entries", "commits", "applied"):
                if isinstance(backend.get(candidate), list):
                    key = candidate
                    break
            if key is None:
                key = "records"
                backend.setdefault(key, [])
            backend[key].append(deepcopy(record))
            backend["generation"] = record["generation"]
            self._main_fallback_generation = record["generation"]
            return {"accepted": True, "generation": record["generation"],
                    "storage": "mapping"}
        if isinstance(backend, list):
            backend.append(deepcopy(record))
            self._main_fallback_generation = record["generation"]
            return {"accepted": True, "generation": record["generation"],
                    "storage": "list"}

        candidates: list[Callable[..., Any]] = []
        seen: set[str] = set()
        for name in (
            "apply_selfmodification", "commit_selfmodification", "commit",
            "apply", "write", "append", "update", "set",
        ):
            method = getattr(backend, name, None)
            if callable(method):
                candidates.append(method)
                seen.add(name)
        for name, method in getattr(backend, "__dict__", {}).items():
            if name in seen or name.startswith("_") or not callable(method):
                continue
            folded = name.casefold()
            if any(marker in folded for marker in
                   ("write", "commit", "apply", "append", "update")):
                candidates.append(method)
                seen.add(name)
        if not candidates and callable(backend):
            candidates.append(backend)

        for method in candidates:
            result = self._call_main_method(
                method, record, record["expected_generation"],
                record["generation"])
            if self._result_rejection(result) is not None:
                # A rejected write is not a mutation: leave the observed
                # generation untouched so a retry can use the same CAS.
                return result
            self._main_fallback_generation = record["generation"]
            direct = self._read_direct_generation(backend)
            if direct is None or direct < record["generation"]:
                try:
                    setattr(backend, "generation", record["generation"])
                except (AttributeError, TypeError, ValueError):
                    pass
            if isinstance(result, Mapping):
                reported = result.get("generation")
                if isinstance(reported, int) \
                        and not isinstance(reported, bool) \
                        and reported < record["generation"]:
                    # Some plain object backends echo their pre-write
                    # generation.  The gate wrote exactly this record, so
                    # report the monotonic target generation instead.
                    result = dict(result)
                    result["generation"] = record["generation"]
            return result
        raise TypeError("main backend has no supported write method")

    def _read_direct_generation(self, backend: Any) -> int | None:
        if not hasattr(backend, "generation"):
            return None
        value = getattr(backend, "generation")
        if callable(value):
            try:
                value = value()
            except Exception:
                return None
        return self._as_generation(value)

    @staticmethod
    def _call_main_method(method: Callable[..., Any], record: dict[str, Any],
                          expected: int, generation: int) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return method(record)
        parameters = signature.parameters
        names = set(parameters)
        positional = [
            parameter for parameter in parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        extras: dict[str, Any] = {}
        if "expected_generation" in names:
            extras["expected_generation"] = expected
        if "generation" in names:
            extras["generation"] = generation
        if "approved" in names:
            extras["approved"] = True
        if "record" in names:
            if "token" in names:
                return method(token=record["token"], record=record, **extras)
            return method(record=record, **extras)
        if "payload" in names:
            return method(payload=record, **extras)
        if "token" in names and len(positional) == 1:
            return method(token=record["token"], **extras)
        if {"head", "layer", "operation", "value"} <= names:
            kwargs = {
                "head": record["head"], "layer": record["layer"],
                "operation": record["operation"], "value": record["value"],
            }
            if "reason" in names:
                kwargs["reason"] = record["reason"]
            if "token" in names:
                kwargs["token"] = record["token"]
            return method(**kwargs, **extras)
        if not positional:
            return method(record=record, **extras)
        return method(record, **extras)

    @staticmethod
    def _result_rejection(result: Any) -> str | None:
        if result is False:
            return "backend returned False"
        if isinstance(result, Mapping):
            if result.get("accepted") is False or result.get("ok") is False:
                return str(result.get("reason") or result.get("reasons")
                           or "accepted is false")
            status = str(result.get("status") or "").casefold()
            if status in {"denied", "rejected", "failed", "error"}:
                return str(result.get("reason") or status)
        return None

    @staticmethod
    def _extract_generation(result: Any) -> int | None:
        if isinstance(result, Mapping):
            value = result.get("generation")
            if isinstance(value, int) and not isinstance(value, bool) \
                    and value >= 0:
                return value
        return None

    # ----------------------------------------------------------------- logic
    def _run_logic(self, proposal: _Proposal) -> tuple[bool, list[str],
                                                       dict[str, Any]]:
        if self.logic is None:
            return True, [], {
                "present": False,
                "passed": True,
                "reasons": [],
            }
        payload = proposal.as_logic_payload()
        try:
            raw = self._call_logic(payload)
        except Exception as exc:
            reason = "logic evaluation failed (%s)" % type(exc).__name__
            return False, [reason], {
                "present": True,
                "passed": False,
                "error": type(exc).__name__,
                "reasons": [reason],
            }
        passed, reasons, detail = self._interpret_logic(raw)
        return passed, reasons, {
            "present": True,
            "passed": passed,
            "result_type": type(raw).__name__,
            "reasons": list(reasons),
            "detail": detail,
        }

    def _call_logic(self, payload: dict[str, Any]) -> Any:
        logic = self.logic
        if isinstance(logic, Mapping):
            checks = []
            for key, value in logic.items():
                if callable(value):
                    verdict, _reasons, _detail = self._interpret_logic(
                        self._invoke_logic(value, payload))
                    checks.append((str(key), bool(verdict)))
                else:
                    checks.append((str(key), bool(value)))
            return {name: ok for name, ok in checks}
        if callable(logic):
            return self._invoke_logic(logic, payload)
        evaluate = getattr(logic, "evaluate", None)
        if callable(evaluate):
            return self._invoke_logic(evaluate, payload)
        check = getattr(logic, "check", None)
        if callable(check):
            return self._invoke_logic(check, payload)
        raise TypeError("logic does not expose a callable rule set")

    @staticmethod
    def _invoke_logic(function: Callable[..., Any], payload: dict[str, Any]) -> Any:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return function(payload)
        parameters = signature.parameters
        names = set(parameters)
        positional = [
            parameter for parameter in parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        field_names = {"head", "layer", "operation", "value", "reason"}
        if names & field_names:
            kwargs = {name: payload[name] for name in field_names if name in names}
            if "token" in names:
                kwargs["token"] = payload["token"]
            return function(**kwargs)
        if "token" in names and "proposal" in names:
            return function(token=payload["token"], proposal=payload)
        if "token" in names and "record" in names:
            return function(token=payload["token"], record=payload)
        if "token" in names:
            return function(token=payload["token"])
        if "proposal" in names:
            return function(proposal=payload)
        if "record" in names:
            return function(record=payload)
        if "payload" in names:
            return function(payload=payload)
        has_var_pos = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters.values())
        has_var_kw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values())
        if len(positional) >= 1 or has_var_pos:
            return function(payload)
        if has_var_kw:
            return function(**payload)
        return function()

    @staticmethod
    def _interpret_logic(result: Any) -> tuple[bool, list[str], Any]:
        if isinstance(result, bool):
            return result, ([] if result else [DENIAL_MESSAGES[LOGIC_CONTRADICTION]]), result
        if result is None:
            return False, ["logic returned no decision"], None
        if isinstance(result, Mapping):
            reasons: list[str] = []
            denied = False
            recognized = False
            for key in (
                "contradiction", "conflict", "inconsistent",
                "logic_contradiction", "denied", "rejected",
            ):
                if key in result:
                    recognized = True
                    if bool(result[key]):
                        denied = True
                        reasons.append(
                            str(result.get("reason") or DENIAL_MESSAGES[LOGIC_CONTRADICTION]))
            for key in ("accepted", "ok", "consistent", "valid"):
                if key in result:
                    recognized = True
                    if result[key] is False:
                        denied = True
            for key in ("reason", "reasons"):
                if key in result:
                    raw = result[key]
                    if isinstance(raw, (list, tuple)):
                        reasons.extend(str(item) for item in raw)
                    elif raw not in (None, ""):
                        reasons.append(str(raw))
            invariants = result.get("invariants")
            if invariants is not None:
                recognized = True
                if isinstance(invariants, Mapping):
                    failed = [name for name, ok in invariants.items() if not ok]
                    if failed:
                        denied = True
                        reasons.extend("invariant failed: %s" % name for name in failed)
                elif isinstance(invariants, (list, tuple)):
                    failed = [str(item) for item in invariants if not item]
                    if failed:
                        denied = True
                        reasons.extend("invariant failed: %s" % item for item in failed)
                elif not invariants:
                    denied = True
                    reasons.append(DENIAL_MESSAGES[LOGIC_CONTRADICTION])
            if not recognized:
                if result and all(isinstance(value, bool) for value in result.values()):
                    failed = [name for name, ok in result.items() if not ok]
                    if failed:
                        denied = True
                        reasons.extend("rule failed: %s" % name for name in failed)
                else:
                    denied = True
                    reasons.append("logic result has no Boolean decision")
            if denied:
                if not reasons:
                    reasons.append(DENIAL_MESSAGES[LOGIC_CONTRADICTION])
                return False, reasons, dict(result)
            return True, [], dict(result)
        if isinstance(result, (list, tuple, set, frozenset)):
            values = list(result)
            if not values:
                return True, [], values
            if all(isinstance(value, bool) for value in values):
                if all(values):
                    return True, [], values
                return False, [DENIAL_MESSAGES[LOGIC_CONTRADICTION]], values
            return False, [str(value) for value in values], values
        if isinstance(result, str):
            if result.strip().casefold() in {
                "ok", "pass", "passed", "accepted", "true", "consistent",
            }:
                return True, [], result
            return False, [result], result
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            if isinstance(result, float) and not math.isfinite(result):
                return False, [DENIAL_MESSAGES[LOGIC_CONTRADICTION]], result
            return bool(result), ([] if result else [DENIAL_MESSAGES[LOGIC_CONTRADICTION]]), result
        return False, ["unsupported logic result"], repr(result)


__all__ = (
    "SCHEMA", "PROPOSAL_SCHEMA", "COMMIT_SCHEMA", "HEAD_COUNT", "LAYER_COUNT",
    "LAYER_NAMES", "MAX_RULES", "MAX_VALUE_BYTES", "OPERATION_ALLOWLIST",
    "ALLOWED_OPERATIONS", "RULE_TABLE", "DENIAL_CODES", "DENIAL_MESSAGES",
    "Rule", "DEFAULT_RULE_TABLE", "SelfModificationGate", "normalize_layer",
    "normalize_operation",
)
