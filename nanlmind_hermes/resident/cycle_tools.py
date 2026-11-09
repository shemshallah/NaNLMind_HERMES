"""Bounded, dependency-injected tool handlers for raw cycle geometry.

The handlers in this module are backend agnostic.  A host integration supplies
one object with the small contract below; these functions only validate
properties, enforce explicit approval, bound every collection/string, and
render JSON-safe output.  Nothing here opens a file, performs a network
request, exposes a process pointer or byte offset, or enables canonical
merge/promotion.

The backend stores *raw signed integer words* (the Q16.16 representation used
by the resident pair rails).  Arithmetic in this module is exact integer
arithmetic: a triangle flux is

    F_root(i, j) = x(root, i) + x(i, j) - x(root, j)

so an additive vertex gauge ``x(a, b) -> x(a, b) + g[b] - g[a]`` cancels
exactly, with no floating-point rounding.

Canonical backend contract (aliases listed below are also accepted)::

    snapshot() -> mapping
    read_word(i, j) -> int or mapping with ``word``
    set_word(i, j, word[, expected_generation=...]) -> mapping
    train(text, batch, steps[, expected_generation=...]) -> mapping
    cycle(i, j, root) -> int or mapping with ``value``      (optional)
    gauge(i, j, root) -> mapping                            (optional)
    query(text[, limit]) -> mapping or sequence             (optional)

``snapshot`` should report ``nodes``, ``root``, ``generation``, ``digest`` and
optionally ``words``/``word_count`` plus ``scratch``/``main`` phase records.
Accepted writes should report ``accepted``, ``generation``, ``reasons`` and,
when the backend stages writes, ``scratch``/``main`` phase records and an
ordered ``events`` sequence.  A scratch write must be reported before the main
write.  Denied or stale writes must not advance generation or change digest.

Every response includes ``canonical_merge.enabled=False``,
``canonical_merge.attempted=False`` and ``canonical_promotion=False``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
import hashlib
import inspect
import json
import math
import re
from typing import Any, Callable

SCHEMA = "nanlm.resident-cycle-tools/1"

NODE_COUNT = 256
MAX_NODE = NODE_COUNT - 1
EDGE_COUNT = NODE_COUNT * (NODE_COUNT - 1) // 2
CYCLE_DIMENSION = EDGE_COUNT - (NODE_COUNT - 1)
TRIANGLE_COUNT = NODE_COUNT * (NODE_COUNT - 1) * (NODE_COUNT - 2) // 6

MIN_WORD = -(1 << 31)
MAX_WORD = (1 << 31) - 1
MAX_CYCLE_MAGNITUDE = 3 * (1 << 31)

MAX_TEXT_BYTES = 4096
MAX_HIT_TEXT_BYTES = 512
MAX_METRIC_TEXT_BYTES = 240
MAX_BATCH = 64
DEFAULT_BATCH = 1
MAX_STEPS = 256
DEFAULT_STEPS = 1
MAX_SNAPSHOT_WORDS = 512
DEFAULT_SNAPSHOT_WORDS = 64
DEFAULT_STATUS_WORDS = 16
MAX_QUERY_HITS = 32
DEFAULT_QUERY_HITS = 8
MAX_REASONS = 16
MAX_REASON_BYTES = 240
MAX_EVENTS = 16
MAX_METRICS = 32
MAX_OUTPUT_BYTES = 64 * 1024
MAX_GENERATION = (1 << 63) - 1
MAX_UNKNOWN_PROPERTIES = 8
MAX_RAW_WORDS = 65536

_EDGE_KEY_RE = re.compile(r"^(\d{1,3})[,:](\d{1,3})$")
_EVENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_METRIC_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MISSING = object()

_SCRATCH_EVENTS = frozenset({"scratch", "stage", "staged", "prepare"})
_MAIN_EVENTS = frozenset(
    {"main", "commit", "committed", "apply", "applied", "write"})
_POINTER_KEYS = frozenset({
    "pointer", "ptr", "raw_pointer", "offset", "address", "buffer",
    "memoryview", "cell_offset", "raw_offset", "file_offset", "mmap",
})


class CycleToolError(ValueError):
    """Base error for invalid cycle-tool requests or backend results."""


class CycleValidationError(CycleToolError):
    """A request property is missing, unknown, or outside its bound."""


class CycleApprovalError(CycleToolError):
    """A main-state operation lacks the literal ``approved=true`` value."""


class CycleBackendError(CycleToolError):
    """The injected backend returned an invalid or unsafe result."""


class CycleConflictError(CycleToolError):
    """A backend rejected a write; the handler renders it as denied."""

    def __init__(self, message: str = "write denied",
                 reasons: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = tuple(reasons or ())


class CycleInvariantError(CycleToolError):
    """A backend violated a read-only, ordering, or generation invariant."""


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------
def _params(params: Any, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(params, Mapping):
        raise CycleValidationError("params must be an object")
    unknown: list[str] = []
    for key in params:
        if not isinstance(key, str):
            raise CycleValidationError("property names must be strings")
        if len(key) > 64:
            raise CycleValidationError(
                "property name exceeds 64 characters")
        if key not in allowed:
            unknown.append(key)
            if len(unknown) >= MAX_UNKNOWN_PROPERTIES:
                break
    if unknown:
        raise CycleValidationError(
            "unknown properties: %s" % ", ".join(sorted(unknown)))
    return params


def _required(params: Mapping[str, Any], name: str) -> Any:
    if name not in params:
        raise CycleValidationError("%s is required" % name)
    return params[name]


def _require_approved(params: Mapping[str, Any]) -> None:
    if params.get("approved") is not True:
        raise CycleApprovalError("explicit approved=true is required")


def _input_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleValidationError("%s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise CycleValidationError(
            "%s must be in [%d, %d]" % (name, minimum, maximum))
    return value


def _input_node(value: Any, name: str) -> int:
    return _input_int(value, name, 0, MAX_NODE)


def _input_generation(value: Any,
                      name: str = "expected_generation") -> int:
    return _input_int(value, name, 0, MAX_GENERATION)


def _input_word(value: Any, name: str = "word") -> int:
    return _input_int(value, name, MIN_WORD, MAX_WORD)


def _input_limit(value: Any, name: str, maximum: int) -> int:
    return _input_int(value, name, 1, maximum)


def _bounded_string(value: Any, name: str, maximum: int,
                    error_type: type[CycleToolError],
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
                           CycleValidationError, allow_empty=False)


def _input_triangle(args: Mapping[str, Any], *,
                    require_root: bool) -> tuple[int, int, int | None]:
    first = _input_node(_required(args, "i"), "i")
    second = _input_node(_required(args, "j"), "j")
    if first == second:
        raise CycleValidationError("i and j must be distinct")
    root: int | None = None
    if "root" in args:
        root = _input_node(args["root"], "root")
        if root == first or root == second:
            raise CycleValidationError(
                "root must be distinct from i and j")
    elif require_root:
        raise CycleValidationError("root is required")
    return first, second, root


def _resolve_root(args: Mapping[str, Any], snapshot: Mapping[str, Any]) -> int:
    if "root" in args:
        return _input_node(args["root"], "root")
    value = snapshot.get("root")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleBackendError("backend root must be an integer")
    if not 0 <= value <= MAX_NODE:
        raise CycleBackendError("backend root is outside [0, %d]" % MAX_NODE)
    return value


# ---------------------------------------------------------------------------
# backend invocation / result validation
# ---------------------------------------------------------------------------
def _backend_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CycleBackendError("backend %s must return an object" % name)
    return value


def _backend_int(value: Any, name: str, minimum: int = MIN_WORD,
                 maximum: int = MAX_WORD, default: int | None = None
                 ) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleBackendError("backend %s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise CycleBackendError(
            "backend %s is outside [%d, %d]" % (name, minimum, maximum))
    return value


def _backend_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CycleBackendError("backend %s must be a number" % name)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CycleBackendError("backend %s must be a number" % name) from exc
    if not math.isfinite(result):
        raise CycleBackendError("backend %s must be finite" % name)
    return round(result, 8)


def _backend_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise CycleBackendError("backend %s must be a boolean" % name)
    return value


def _backend_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or \
            not isinstance(value, Sequence):
        raise CycleBackendError("backend reasons must be a sequence")
    if len(value) > MAX_REASONS:
        raise CycleBackendError(
            "backend reasons exceed %d entries" % MAX_REASONS)
    return [
        _bounded_string(item, "reason", MAX_REASON_BYTES,
                        CycleBackendError, allow_empty=True)
        for item in value
    ]


def _backend_digest(value: Any, name: str = "digest") -> str | None:
    if value is None:
        return None
    return _bounded_string(value, name, 128, CycleBackendError,
                           allow_empty=True) or None


def _find_method(backend: Any, names: Sequence[str]) -> tuple[str, Any] | None:
    for name in names:
        function = getattr(backend, name, None)
        if callable(function):
            return name, function
    return None


def _accepts_keyword(function: Any, name: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    return any(parameter.kind == parameter.VAR_KEYWORD
               for parameter in signature.parameters.values()) or \
        name in signature.parameters


def _positional_names(function: Any) -> list[str]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return []
    return [
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD)
    ]


def _call_train(backend: Any, text: str, batch: int, steps: int,
                expected: int | None) -> Any:
    found = _find_method(backend, ("train",))
    names = _positional_names(found[1]) if found is not None else []
    arguments: tuple[Any, ...]
    if names[:3] == ["batch", "steps", "text"]:
        arguments = (batch, steps, text)
    else:
        arguments = (text, batch, steps)
    return _call(backend, ("train",), *arguments,
                 expected_generation=(_MISSING if expected is None
                                      else expected))


def _call(backend: Any, names: Sequence[str], *args: Any,
          expected_generation: Any = _MISSING, **kwargs: Any) -> Any:
    if backend is None:
        raise CycleValidationError("backend is required")
    found = _find_method(backend, names)
    if found is None:
        raise CycleBackendError("backend must provide %s()" % names[0])
    method_name, function = found
    call_kwargs = {}
    if expected_generation is not _MISSING and \
            _accepts_keyword(function, "expected_generation"):
        call_kwargs["expected_generation"] = expected_generation
    for key, value in kwargs.items():
        if _accepts_keyword(function, key):
            call_kwargs[key] = value
    try:
        return function(*args, **call_kwargs)
    except CycleToolError:
        raise
    except Exception as exc:  # Keep backend internals out of tool output.
        raise CycleBackendError(
            "backend %s failed: %s" % (method_name, type(exc).__name__)
        ) from exc


def _snapshot(backend: Any) -> Mapping[str, Any]:
    return _backend_mapping(
        _call(backend, ("snapshot",)), "snapshot()")


def _operation_snapshot(backend: Any) -> Mapping[str, Any]:
    """Take a snapshot for read-only/write invariant checks."""
    try:
        return _snapshot(backend)
    except CycleToolError:
        raise
    except Exception as exc:
        raise CycleBackendError("backend snapshot() failed") from exc


def _nested_mapping(snapshot: Mapping[str, Any],
                    name: str) -> Mapping[str, Any]:
    value = snapshot.get(name)
    return value if isinstance(value, Mapping) else {}


def _generation_from(value: Any, name: str = "generation") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleBackendError("backend %s must be an integer" % name)
    if not 0 <= value <= MAX_GENERATION:
        raise CycleBackendError(
            "backend %s is outside [0, %d]" % (name, MAX_GENERATION))
    return value


def _generation_from_snapshot(snapshot: Mapping[str, Any],
                              backend: Any = None) -> int | None:
    value = snapshot.get("generation")
    if value is None:
        value = _nested_mapping(snapshot, "field").get("generation")
    if value is None and backend is not None:
        value = getattr(backend, "generation", None)
        if callable(value):
            value = value()
    return _generation_from(value)


def _digest_from_snapshot(snapshot: Mapping[str, Any]) -> str | None:
    for key in ("digest", "state_digest", "field_digest"):
        value = snapshot.get(key)
        if value is not None:
            return _backend_digest(value, "backend %s" % key)
    field = _nested_mapping(snapshot, "field")
    for key in ("digest", "state_digest", "field_digest"):
        value = field.get(key)
        if value is not None:
            return _backend_digest(value, "backend %s" % key)
    return None


def _phase_record(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CycleBackendError("backend %s must be an object" % name)
    result: dict[str, Any] = {}
    if "written" in value and value["written"] is not None:
        result["written"] = _backend_bool(
            value["written"], "%s written" % name)
    generation = _backend_int(
        value.get("generation"), "%s generation" % name,
        0, MAX_GENERATION, None)
    if generation is not None:
        result["generation"] = generation
        result.setdefault("written", True)
    digest = _backend_digest(value.get("digest"), "backend %s digest" % name)
    if digest is not None:
        result["digest"] = digest
    result.setdefault("written", False)
    return result


def _events(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or \
            not isinstance(value, Sequence):
        raise CycleBackendError("backend events must be a sequence")
    if len(value) > MAX_EVENTS:
        raise CycleBackendError(
            "backend events exceed %d entries" % MAX_EVENTS)
    events: list[str] = []
    for item in value:
        if not isinstance(item, str) or _EVENT_RE.fullmatch(item) is None:
            raise CycleBackendError(
                "backend events must be short lowercase names")
        events.append(item)
    return events


def _validate_write_phases(value: Mapping[str, Any],
                           accepted: bool) -> dict[str, Any]:
    scratch = _phase_record(value.get("scratch"), "scratch")
    main = _phase_record(value.get("main"), "main")
    events = _events(value.get("events"))

    scratch_positions = [index for index, event in enumerate(events)
                         if event in _SCRATCH_EVENTS]
    main_positions = [index for index, event in enumerate(events)
                      if event in _MAIN_EVENTS]
    if scratch_positions and main_positions and \
            min(scratch_positions) > min(main_positions):
        raise CycleInvariantError(
            "scratch write must be reported before the main write")
    if main_positions:
        first_scratch = min(scratch_positions) if scratch_positions else None
        first_main = min(main_positions)
        if first_scratch is not None and first_scratch > first_main:
            raise CycleInvariantError(
                "scratch write must be reported before the main write")

    if not accepted and (main_positions or
                         (main is not None and main.get("written"))):
        raise CycleInvariantError(
            "denied write must not report a main write")
    if main is not None and main.get("written") and scratch is not None \
            and not scratch.get("written"):
        raise CycleInvariantError(
            "main write requires a completed scratch write")

    if scratch is not None and main is not None:
        scratch_generation = scratch.get("generation")
        main_generation = main.get("generation")
        if scratch_generation is not None and main_generation is not None \
                and main_generation < scratch_generation:
            raise CycleInvariantError(
                "main generation must not precede scratch generation")

    reported = value.get("scratch_before_main")
    if reported is not None:
        if not isinstance(reported, bool):
            raise CycleBackendError(
                "backend scratch_before_main must be a boolean")
        if not reported and scratch is not None and main is not None \
                and scratch.get("written") and main.get("written"):
            raise CycleInvariantError(
                "backend reported main before scratch")

    if main_positions and scratch_positions:
        ordering: bool | None = min(scratch_positions) < min(main_positions)
    elif scratch is not None and main is not None:
        ordering = bool(scratch.get("written")) or \
            not bool(main.get("written"))
    elif main is not None and main.get("written"):
        ordering = None
    else:
        ordering = True
    return {
        "scratch": scratch,
        "main": main,
        "events": events,
        "scratch_before_main": ordering,
    }


def _check_generation_transition(previous: int | None, current: int | None,
                                 accepted: bool, operation: str) -> None:
    if previous is None or current is None:
        return
    if accepted:
        if current <= previous:
            raise CycleInvariantError(
                "accepted %s must advance main generation" % operation)
    elif current != previous:
        raise CycleInvariantError(
            "denied %s must not change main generation" % operation)


def _assert_no_main_mutation(before: Mapping[str, Any],
                             after: Mapping[str, Any],
                             operation: str) -> None:
    before_generation = _generation_from_snapshot(before)
    after_generation = _generation_from_snapshot(after)
    if before_generation is not None and after_generation is not None \
            and before_generation != after_generation:
        raise CycleInvariantError(
            "failed/denied %s changed main generation" % operation)
    before_digest = _digest_from_snapshot(before)
    after_digest = _digest_from_snapshot(after)
    if before_digest is not None and after_digest is not None \
            and before_digest != after_digest:
        raise CycleInvariantError(
            "failed/denied %s changed the main digest" % operation)


def _snapshot_after(backend: Any) -> Mapping[str, Any] | None:
    try:
        return _snapshot(backend)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# word / snapshot projections
# ---------------------------------------------------------------------------
def _edge_key(key: Any) -> tuple[int, int]:
    if isinstance(key, (tuple, list)) and len(key) == 2:
        first, second = key
    elif isinstance(key, str):
        match = _EDGE_KEY_RE.fullmatch(key)
        if match is None:
            raise CycleBackendError(
                "snapshot word keys must look like 'i,j'")
        first, second = int(match.group(1)), int(match.group(2))
    else:
        raise CycleBackendError(
            "snapshot word keys must be pairs or 'i,j' strings")
    first = _input_int(first, "snapshot i", 0, MAX_NODE)
    second = _input_int(second, "snapshot j", 0, MAX_NODE)
    if first == second:
        raise CycleBackendError("snapshot word keys must be distinct")
    return first, second


def _edge_word(value: Any) -> int:
    if isinstance(value, Mapping):
        value = value.get("word", value.get("value"))
    result = _backend_int(value, "snapshot word")
    if result is None:
        raise CycleBackendError("snapshot word is missing")
    return result


def _raw_words(value: Any) -> list[dict[str, int]]:
    if value is None:
        return []
    words: dict[tuple[int, int], int] = {}
    if isinstance(value, Mapping):
        if len(value) > MAX_RAW_WORDS:
            raise CycleBackendError(
                "snapshot words exceed %d entries" % MAX_RAW_WORDS)
        entries = value.items()
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_RAW_WORDS:
            raise CycleBackendError(
                "snapshot words exceed %d entries" % MAX_RAW_WORDS)
        pairs: list[tuple[Any, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                pairs.append(((item.get("i"), item.get("j")), item))
            elif isinstance(item, Sequence) and \
                    not isinstance(item, (str, bytes, bytearray)) and \
                    len(item) >= 3:
                pairs.append(((item[0], item[1]), item[2]))
            else:
                raise CycleBackendError(
                    "snapshot word entries must be objects or triples")
        entries = pairs
    else:
        raise CycleBackendError(
            "snapshot words must be an object or sequence")

    for key, item in entries:
        edge = _edge_key(key)
        if edge in words:
            raise CycleBackendError(
                "snapshot contains duplicate word %d,%d" % edge)
        words[edge] = _edge_word(item)
    return [
        {"i": first, "j": second, "word": word}
        for (first, second), word in sorted(words.items())
    ]


def _snapshot_words(snapshot: Mapping[str, Any]) -> list[dict[str, int]]:
    raw = snapshot.get("words")
    if raw is None:
        raw = snapshot.get("edges")
    if raw is None:
        raw = _nested_mapping(snapshot, "field").get("words")
    return _raw_words(raw)


def _snapshot_word_count(snapshot: Mapping[str, Any],
                         words: Sequence[Mapping[str, int]]) -> int:
    value = snapshot.get("word_count")
    if value is None:
        value = snapshot.get("count")
    if value is None:
        value = _nested_mapping(snapshot, "field").get("word_count")
    if value is None:
        return len(words)
    count = _backend_int(value, "word_count", 0, MAX_RAW_WORDS)
    return max(int(count or 0), len(words))


def _bounded_words(snapshot: Mapping[str, Any],
                   limit: int) -> tuple[list[dict[str, int]], int, bool,
                                       list[dict[str, int]]]:
    words = _snapshot_words(snapshot)
    count = _snapshot_word_count(snapshot, words)
    return (words[:limit], count, count > limit or len(words) > limit,
            words)


def _snapshot_fallback_digest(
        snapshot: Mapping[str, Any],
        words: Sequence[Mapping[str, int]] | None = None) -> str:
    digest = _digest_from_snapshot(snapshot)
    if digest is not None:
        return digest
    if words is None:
        words = _snapshot_words(snapshot)
    return _words_digest(words)


def _snapshot_nodes(snapshot: Mapping[str, Any]) -> int:
    value = snapshot.get("nodes")
    if value is None:
        value = snapshot.get("node_count")
    if value is None:
        return NODE_COUNT
    nodes = _backend_int(value, "nodes", 2, NODE_COUNT)
    return int(nodes)


def _snapshot_root(snapshot: Mapping[str, Any], nodes: int) -> int:
    value = snapshot.get("root")
    if value is None:
        return 0
    root = _backend_int(value, "root", 0, nodes - 1)
    return int(root)


def _phase_from_snapshot(snapshot: Mapping[str, Any],
                         name: str) -> dict[str, Any] | None:
    value = snapshot.get(name)
    if value is None:
        value = snapshot.get("%s_generation" % name)
        if value is None:
            return None
        return {"generation": _generation_from(value, "%s_generation" % name),
                "written": False}
    return _phase_record(value, name)


def _state_meta(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    nodes = _snapshot_nodes(snapshot)
    generation = _generation_from_snapshot(snapshot)
    digest = _digest_from_snapshot(snapshot)
    result: dict[str, Any] = {
        "nodes": nodes,
        "root": _snapshot_root(snapshot, nodes),
        "generation": generation,
        "digest": digest,
        "opened": _backend_bool(snapshot.get("opened"), "opened", True),
    }
    for name in ("scratch", "main"):
        phase = _phase_from_snapshot(snapshot, name)
        if phase is not None:
            result[name] = phase
    return result


def _words_digest(words: Sequence[Mapping[str, int]]) -> str:
    material = json.dumps(
        [[item["i"], item["j"], item["word"]] for item in words],
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# ---------------------------------------------------------------------------
# read helpers
# ---------------------------------------------------------------------------
def _read_word(backend: Any, first: int, second: int) -> tuple[int, Any]:
    value = _call(backend, ("read_word", "get_word", "read_edge", "word"),
                  first, second)
    metadata = value if isinstance(value, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get("word", value.get("value"))
    word = _backend_int(value, "read_word")
    if word is None:
        raise CycleBackendError("backend read_word() returned no word")
    return int(word), metadata


def _cycle_from_value(value: Any, name: str = "cycle()") -> int:
    metadata = value if isinstance(value, Mapping) else None
    if isinstance(value, Mapping):
        for key in ("value", "flux", "cycle", "word"):
            if key in value:
                value = value[key]
                break
        else:
            raise CycleBackendError("backend %s returned no value" % name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CycleBackendError("backend %s must return an integer" % name)
    if not -MAX_CYCLE_MAGNITUDE <= value <= MAX_CYCLE_MAGNITUDE:
        raise CycleBackendError(
            "backend %s is outside the bounded cycle range" % name)
    if metadata is not None:
        digest = _backend_digest(metadata.get("digest"))
        if digest is not None:
            return int(value)
    return int(value)


def _exact_cycle(backend: Any, first: int, second: int,
                 root: int) -> tuple[int, dict[str, int]]:
    root_i, _ = _read_word(backend, root, first)
    first_second, _ = _read_word(backend, first, second)
    root_second, _ = _read_word(backend, root, second)
    raw = {
        "root_i": root_i,
        "i_j": first_second,
        "root_j": root_second,
    }
    return root_i + first_second - root_second, raw


def _read_only_guard(before: Mapping[str, Any],
                     after: Mapping[str, Any], operation: str) -> None:
    before_generation = _generation_from_snapshot(before)
    after_generation = _generation_from_snapshot(after)
    if before_generation is not None and after_generation is not None \
            and before_generation != after_generation:
        raise CycleInvariantError("%s must be read-only" % operation)
    before_digest = _digest_from_snapshot(before)
    after_digest = _digest_from_snapshot(after)
    if before_digest is not None and after_digest is not None \
            and before_digest != after_digest:
        raise CycleInvariantError("%s must be read-only" % operation)


# ---------------------------------------------------------------------------
# query / metric projections
# ---------------------------------------------------------------------------
def _normalize_hit(item: Any) -> dict[str, Any] | None:
    if isinstance(item, int) and not isinstance(item, bool):
        return {"word": _input_word(item, "hit word")}
    if not isinstance(item, Mapping):
        raise CycleBackendError("query hits must be objects")
    hit: dict[str, Any] = {}
    for key in ("i", "j", "root"):
        if item.get(key) is not None:
            hit[key] = _input_node(item[key], "hit %s" % key)
    for key in ("word", "value"):
        if item.get(key) is not None:
            hit["word"] = _input_word(item[key], "hit word")
            break
    if item.get("score") is not None:
        hit["score"] = _backend_float(item["score"], "hit score")
    for key in ("text", "kind", "leaf_id", "feature_id"):
        if item.get(key) is not None:
            text = _bounded_string(
                item[key], "hit %s" % key, MAX_HIT_TEXT_BYTES,
                CycleBackendError, allow_empty=True)
            if text:
                hit[key] = text
    return hit or None


def _normalize_query(value: Any, limit: int) -> dict[str, Any]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"hits": [{"word": _input_word(value, "query word")}],
                "count": 1}
    if isinstance(value, Mapping):
        base: dict[str, Any] = {}
        if value.get("mode") is not None:
            base["mode"] = _bounded_string(
                value["mode"], "query mode", 32,
                CycleBackendError, allow_empty=False)
        if value.get("text") is not None:
            base["text"] = _bounded_string(
                value["text"], "query text", MAX_TEXT_BYTES,
                CycleBackendError, allow_empty=True)
        if value.get("word") is not None or value.get("value") is not None:
            base["word"] = _input_word(
                value.get("word", value.get("value")), "query word")
        raw = value.get("hits")
        if raw is None:
            raw = value.get("results")
        if raw is None and "word" not in base and "value" not in base:
            raw = []
        elif raw is None:
            raw = [value]
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        base = {}
        raw = value
    else:
        raise CycleBackendError(
            "backend query() must return an object, sequence or integer")

    if not isinstance(raw, (str, bytes, bytearray)) and \
            isinstance(raw, Sequence):
        if len(raw) > MAX_RAW_WORDS:
            raise CycleBackendError(
                "backend query hits exceed %d entries" % MAX_RAW_WORDS)
        hits = []
        for item in list(raw)[:limit]:
            hit = _normalize_hit(item)
            if hit is not None:
                hits.append(hit)
        base["hits"] = hits
        base["count"] = len(hits)
    else:
        raise CycleBackendError(
            "backend query() hits must be a sequence")
    return base


def _normalize_metrics(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    base = _backend_mapping(value, "metrics")
    if len(base) > MAX_METRICS:
        raise CycleBackendError(
            "backend metrics exceed %d entries" % MAX_METRICS)
    result: dict[str, Any] = {}
    for key, item in base.items():
        if not isinstance(key, str) or _METRIC_KEY_RE.fullmatch(key) is None:
            raise CycleBackendError("backend metric names must be safe")
        if key in _POINTER_KEYS:
            continue
        if item is None:
            continue
        if isinstance(item, bool):
            result[key] = item
        elif isinstance(item, int):
            result[key] = _backend_int(
                item, "metric %s" % key, -(1 << 62), (1 << 62))
        elif isinstance(item, float):
            result[key] = _backend_float(item, "metric %s" % key)
        elif isinstance(item, str):
            result[key] = _bounded_string(
                item, "metric %s" % key, MAX_METRIC_TEXT_BYTES,
                CycleBackendError, allow_empty=True)
        else:
            raise CycleBackendError(
                "backend metric %s must be scalar" % key)
    return result


# ---------------------------------------------------------------------------
# response policy / JSON safety
# ---------------------------------------------------------------------------
def _policy_flags() -> dict[str, Any]:
    return {
        "canonical_merge": {"enabled": False, "attempted": False},
        "canonical_merge_enabled": False,
        "canonical_promotion": False,
        "canonical_promotion_attempted": False,
        "canonical_promotion_enabled": False,
        "promotion_enabled": False,
    }


def _json_safe(value: dict[str, Any], operation: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False,
                             separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise CycleBackendError(
            "%s produced non-JSON-safe output" % operation) from exc
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise CycleBackendError(
            "%s output exceeds %d bytes" % (operation, MAX_OUTPUT_BYTES))
    return value


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------
def status(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return a bounded read-only summary of the raw cycle field."""
    args = _params(params, frozenset({"limit"}))
    limit = _input_limit(args.get("limit", DEFAULT_STATUS_WORDS),
                         "limit", MAX_SNAPSHOT_WORDS)
    snapshot = _snapshot(backend)
    meta = _state_meta(snapshot)
    words, word_count, truncated, all_words = _bounded_words(snapshot, limit)
    nodes = meta["nodes"]
    edges = nodes * (nodes - 1) // 2
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "status",
        **meta,
        "word_count": word_count,
        "words": words,
        "truncated": truncated,
        "geometry": {
            "nodes": nodes,
            "edges": edges,
            "cycle_dimension": edges - (nodes - 1),
            "triangles": nodes * (nodes - 1) * (nodes - 2) // 6,
            "max_node": nodes - 1,
        },
        "raw_words": True,
        "bounded": True,
        "read_only": True,
        "writes_main": False,
    }
    if result["digest"] is None:
        result["digest"] = _words_digest(all_words)
    result.update(_policy_flags())
    return _json_safe(result, "status")


def snapshot(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return a bounded, sanitised raw-word snapshot and state digest."""
    args = _params(params, frozenset({"limit"}))
    limit = _input_limit(args.get("limit", DEFAULT_SNAPSHOT_WORDS),
                         "limit", MAX_SNAPSHOT_WORDS)
    state = _snapshot(backend)
    meta = _state_meta(state)
    words, word_count, truncated, all_words = _bounded_words(state, limit)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "snapshot",
        **meta,
        "word_count": word_count,
        "words": words,
        "truncated": truncated,
        "raw_words": True,
        "bounded": True,
        "read_only": True,
        "writes_main": False,
    }
    if result["digest"] is None:
        result["digest"] = _words_digest(all_words)
    result.update(_policy_flags())
    return _json_safe(result, "snapshot")


def cycle(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return the exact integer triangle flux for one bounded triangle."""
    args = _params(params, frozenset({"i", "j", "root"}))
    first, second, _root_arg = _input_triangle(args, require_root=False)
    before = _operation_snapshot(backend)
    root = _resolve_root(args, before)
    if root == first or root == second:
        raise CycleValidationError("root must be distinct from i and j")

    raw: dict[str, int] | None = None
    exact_available = _find_method(
        backend, ("read_word", "get_word", "read_edge", "word")) is not None
    value: int | None = None
    if exact_available:
        try:
            value, raw = _exact_cycle(backend, first, second, root)
        except CycleBackendError:
            if _find_method(backend, ("cycle", "cycle_flux", "flux")) is None:
                raise
    backend_cycle = _find_method(backend, ("cycle", "cycle_flux", "flux"))
    if backend_cycle is not None:
        reported = _cycle_from_value(
            _call(backend, ("cycle", "cycle_flux", "flux"),
                  first, second, root))
        if value is not None and reported != value:
            raise CycleInvariantError(
                "backend cycle() disagrees with exact raw-word geometry")
        value = reported
    if value is None:
        raise CycleBackendError(
            "backend must provide read_word() or cycle()")

    after = _operation_snapshot(backend)
    _read_only_guard(before, after, "cycle")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "cycle",
        "i": first,
        "j": second,
        "root": root,
        "value": value,
        "word": value,
        "generation": _generation_from_snapshot(before),
        "digest": _snapshot_fallback_digest(before),
        "exact": True,
        "raw_words": True,
        "read_only": True,
        "writes_main": False,
    }
    if raw is not None:
        result["raw_edges"] = {
            "root_i": raw["root_i"],
            "i_j": raw["i_j"],
            "root_j": raw["root_j"],
        }
    result.update(_policy_flags())
    return _json_safe(result, "cycle")


def gauge(params: Any, *, backend: Any) -> dict[str, Any]:
    """Show exact vertex-gauge cancellation around one bounded triangle."""
    args = _params(params, frozenset({"i", "j", "root"}))
    first, second, _root_arg = _input_triangle(args, require_root=False)
    before = _operation_snapshot(backend)
    root = _resolve_root(args, before)
    if root == first or root == second:
        raise CycleValidationError("root must be distinct from i and j")

    if _find_method(backend, ("read_word", "get_word", "read_edge", "word")) \
            is None:
        value = _call(backend, ("gauge", "gauge_cancel"),
                      first, second, root)
        base = _backend_mapping(value, "gauge()")
        cycle_value = _cycle_from_value(
            base.get("cycle", base.get("value", base.get("flux"))),
            "gauge()")
        gauge_cycle = _cycle_from_value(
            base.get("gauge_cycle", 0), "gauge() gauge_cycle")
        residual = _cycle_from_value(
            base.get("cancellation_residual", cycle_value - gauge_cycle),
            "gauge() cancellation_residual")
        result = {
            "schema": SCHEMA,
            "operation": "gauge",
            "i": first,
            "j": second,
            "root": root,
            "generation": _generation_from_snapshot(before),
            "digest": _snapshot_fallback_digest(before),
            "cycle": cycle_value,
            "invariant_cycle": cycle_value,
            "gauge_cycle": gauge_cycle,
            "cancellation_residual": residual,
            "cancelled": gauge_cycle == 0 and residual == 0,
            "exact": True,
            "raw_words": True,
            "read_only": True,
            "writes_main": False,
        }
        after = _operation_snapshot(backend)
        _read_only_guard(before, after, "gauge")
        result.update(_policy_flags())
        return _json_safe(result, "gauge")

    cycle_value, raw = _exact_cycle(backend, first, second, root)
    # Star gauge relative to root: p(root)=0, p(i)=x(root,i),
    # p(j)=x(root,j).  Its triangle circulation is exactly zero.
    potentials = {
        "root": 0,
        "i": raw["root_i"],
        "j": raw["root_j"],
    }
    gauge_edges = {
        "root_i": potentials["i"] - potentials["root"],
        "i_j": potentials["j"] - potentials["i"],
        "root_j": potentials["j"] - potentials["root"],
    }
    gauge_cycle = gauge_edges["root_i"] + gauge_edges["i_j"] \
        - gauge_edges["root_j"]
    residual_edges = {
        "root_i": raw["root_i"] - gauge_edges["root_i"],
        "i_j": raw["i_j"] - gauge_edges["i_j"],
        "root_j": raw["root_j"] - gauge_edges["root_j"],
    }
    residual_cycle = residual_edges["root_i"] + residual_edges["i_j"] \
        - residual_edges["root_j"]
    cancellation_residual = cycle_value - gauge_cycle - residual_cycle

    after = _operation_snapshot(backend)
    _read_only_guard(before, after, "gauge")

    result = {
        "schema": SCHEMA,
        "operation": "gauge",
        "i": first,
        "j": second,
        "root": root,
        "generation": _generation_from_snapshot(before),
        "digest": _snapshot_fallback_digest(before),
        "potentials": potentials,
        "raw_edges": {
            "root_i": raw["root_i"],
            "i_j": raw["i_j"],
            "root_j": raw["root_j"],
        },
        "gauge_edges": gauge_edges,
        "residual_edges": residual_edges,
        "cycle": cycle_value,
        "invariant_cycle": residual_cycle,
        "gauge_cycle": gauge_cycle,
        "cancellation_residual": cancellation_residual,
        "cancelled": gauge_cycle == 0 and cancellation_residual == 0,
        "exact": True,
        "raw_words": True,
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "gauge")


def _write_denied(operation: str, fields: Mapping[str, Any],
                  reasons: Sequence[str], generation: int | None,
                  expected_generation: int | None) -> dict[str, Any]:
    bounded_reasons = _backend_reasons(list(reasons)) or ["write_denied"]
    receipt = {
        "operation": operation,
        "accepted": False,
        "previous_generation": generation,
        "generation": generation,
        "expected_generation": expected_generation,
        "reasons": bounded_reasons,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": operation,
        **dict(fields),
        "accepted": False,
        "denied": True,
        "previous_generation": generation,
        "generation": generation,
        "expected_generation": expected_generation,
        "reasons": bounded_reasons,
        "scratch": None,
        "main": None,
        "events": [],
        "scratch_before_main": True,
        "receipt": receipt,
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, operation)


def set(params: Any, *, backend: Any) -> dict[str, Any]:
    """Set one directed raw edge word after explicit approval."""
    args = _params(params, frozenset({
        "approved", "i", "j", "word", "expected_generation"}))
    _require_approved(args)
    first, second, _root = _input_triangle(args, require_root=False)
    word = _input_word(_required(args, "word"))
    expected = None if "expected_generation" not in args else \
        _input_generation(args["expected_generation"])

    before = _operation_snapshot(backend)
    previous_generation = _generation_from_snapshot(before)
    fields = {"i": first, "j": second, "word": word}
    if expected is not None:
        if previous_generation is None:
            raise CycleValidationError(
                "backend must expose generation for expected_generation")
        if expected != previous_generation:
            return _write_denied(
                "set", fields, ["stale_generation"],
                previous_generation, expected)

    try:
        value = _backend_mapping(
            _call(backend, ("set_word", "write_word", "set_edge"),
                  first, second, word,
                  expected_generation=(_MISSING if expected is None
                                       else expected)),
            "set_word()")
    except CycleToolError as exc:
        after = _snapshot_after(backend)
        if after is not None:
            _assert_no_main_mutation(before, after, "set")
        raise exc

    accepted = _backend_bool(value.get("accepted"), "accepted", True)
    reasons = _backend_reasons(value.get("reasons"))
    reported_generation = _generation_from(
        value.get("generation"), "generation")
    phases = _validate_write_phases(value, accepted)

    after = _snapshot_after(backend)
    generation = reported_generation
    if after is not None:
        after_generation = _generation_from_snapshot(after)
        if generation is None:
            generation = after_generation
        if not accepted:
            _assert_no_main_mutation(before, after, "set")
        elif previous_generation is not None and after_generation is not None \
                and after_generation <= previous_generation:
            raise CycleInvariantError(
                "accepted set must advance main generation")
    _check_generation_transition(
        previous_generation, generation, accepted, "set")

    stored_word = value.get("word", value.get("value"))
    if stored_word is not None:
        stored_word = _input_word(stored_word, "stored word")
    else:
        stored_word = word

    receipt = {
        "operation": "set",
        "accepted": accepted,
        "i": first,
        "j": second,
        "word": word,
        "stored_word": stored_word,
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        **phases,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "set",
        "accepted": accepted,
        "denied": not accepted,
        "i": first,
        "j": second,
        "word": word,
        "stored_word": stored_word,
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        "scratch": phases["scratch"],
        "main": phases["main"],
        "events": phases["events"],
        "scratch_before_main": phases["scratch_before_main"],
        "receipt": receipt,
        "read_only": not accepted,
        "writes_main": accepted,
    }
    result.update(_policy_flags())
    return _json_safe(result, "set")


def train(params: Any, *, backend: Any) -> dict[str, Any]:
    """Run a bounded, explicitly approved local cycle-field training step."""
    args = _params(params, frozenset({
        "approved", "text", "batch", "steps", "expected_generation"}))
    _require_approved(args)
    text = _input_text(_required(args, "text"))
    batch = _input_int(args.get("batch", DEFAULT_BATCH),
                       "batch", 1, MAX_BATCH)
    steps = _input_int(args.get("steps", DEFAULT_STEPS),
                       "steps", 1, MAX_STEPS)
    expected = None if "expected_generation" not in args else \
        _input_generation(args["expected_generation"])

    before = _operation_snapshot(backend)
    previous_generation = _generation_from_snapshot(before)
    fields = {"text": text, "batch": batch, "steps": steps}
    if expected is not None:
        if previous_generation is None:
            raise CycleValidationError(
                "backend must expose generation for expected_generation")
        if expected != previous_generation:
            return _write_denied(
                "train", fields, ["stale_generation"],
                previous_generation, expected)

    try:
        value = _backend_mapping(
            _call_train(backend, text, batch, steps, expected),
            "train()")
    except CycleToolError as exc:
        after = _snapshot_after(backend)
        if after is not None:
            _assert_no_main_mutation(before, after, "train")
        raise exc

    accepted = _backend_bool(value.get("accepted"), "accepted", True)
    reasons = _backend_reasons(value.get("reasons"))
    reported_generation = _generation_from(
        value.get("generation"), "generation")
    phases = _validate_write_phases(value, accepted)
    metrics = _normalize_metrics(value.get("metrics"))

    after = _snapshot_after(backend)
    generation = reported_generation
    if after is not None:
        after_generation = _generation_from_snapshot(after)
        if generation is None:
            generation = after_generation
        if not accepted:
            _assert_no_main_mutation(before, after, "train")
        elif previous_generation is not None and after_generation is not None \
                and after_generation <= previous_generation:
            raise CycleInvariantError(
                "accepted train must advance main generation")
    _check_generation_transition(
        previous_generation, generation, accepted, "train")

    receipt = {
        "operation": "train",
        "accepted": accepted,
        "text": text,
        "batch": batch,
        "steps": steps,
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        "metrics": metrics,
        **phases,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "train",
        "accepted": accepted,
        "denied": not accepted,
        "applied": bool(accepted),
        "text": text,
        "batch": batch,
        "steps": steps,
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        "metrics": metrics,
        "scratch": phases["scratch"],
        "main": phases["main"],
        "events": phases["events"],
        "scratch_before_main": phases["scratch_before_main"],
        "receipt": receipt,
        "read_only": not accepted,
        "writes_main": accepted,
    }
    result.update(_policy_flags())
    return _json_safe(result, "train")


def query(params: Any, *, backend: Any) -> dict[str, Any]:
    """Read one word/triangle or a bounded text query without mutation."""
    args = _params(params, frozenset({
        "i", "j", "root", "text", "limit"}))
    has_text = "text" in args
    has_edge = "i" in args or "j" in args
    if has_text and (has_edge or "root" in args):
        raise CycleValidationError(
            "query accepts either text or i/j/root, not both")
    if "limit" in args and not has_text:
        raise CycleValidationError("limit requires text")

    before = _operation_snapshot(backend)
    if has_edge:
        if "i" not in args or "j" not in args:
            raise CycleValidationError(
                "query requires both i and j")
        first, second, _root_arg = _input_triangle(args, require_root=False)
        word, metadata = _read_word(backend, first, second)
        payload: dict[str, Any] = {
            "mode": "word",
            "i": first,
            "j": second,
            "word": word,
            "value": word,
        }
        if metadata is not None:
            digest = _backend_digest(metadata.get("digest"), "word digest")
            if digest is not None:
                payload["word_digest"] = digest
        if "root" in args:
            root = _resolve_root(args, before)
            if root == first or root == second:
                raise CycleValidationError(
                    "root must be distinct from i and j")
            cycle_value, raw = _exact_cycle(backend, first, second, root)
            payload.update({
                "root": root,
                "cycle": cycle_value,
                "raw_edges": raw,
            })
    elif has_text:
        text = _input_text(args["text"])
        limit = _input_limit(args.get("limit", DEFAULT_QUERY_HITS),
                             "limit", MAX_QUERY_HITS)
        method = _find_method(backend, ("query", "search", "resonance"))
        if method is None:
            raise CycleBackendError(
                "backend must provide query() for text queries")
        function = method[1]
        if _accepts_keyword(function, "limit"):
            value = _call(backend, ("query", "search", "resonance"),
                          text, limit=limit)
        else:
            value = _call(backend, ("query", "search", "resonance"), text)
        payload = _normalize_query(value, limit)
        payload.setdefault("mode", "text")
        payload["text"] = text
        payload["top_k"] = limit
    else:
        raise CycleValidationError(
            "query requires text or both i and j")

    after = _operation_snapshot(backend)
    _read_only_guard(before, after, "query")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "query",
        "generation": _generation_from_snapshot(before),
        "digest": _digest_from_snapshot(before),
        **payload,
        "read_only": True,
        "writes_main": False,
    }
    if result["digest"] is None:
        result["digest"] = _snapshot_fallback_digest(before)
    result.update(_policy_flags())
    return _json_safe(result, "query")


HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "status": status,
    "set": set,
    "cycle": cycle,
    "gauge": gauge,
    "train": train,
    "snapshot": snapshot,
    "query": query,
}

# Convenient non-shadowing alias for callers that prefer it.
set_edge = set


def build_handlers(backend: Any) -> dict[str, Callable[[Any],
                                                       dict[str, Any]]]:
    """Bind all handlers to one backend for later plugin registration."""
    return {name: partial(handler, backend=backend)
            for name, handler in HANDLERS.items()}


__all__ = (
    "SCHEMA", "NODE_COUNT", "MAX_NODE", "EDGE_COUNT", "CYCLE_DIMENSION",
    "TRIANGLE_COUNT", "MIN_WORD", "MAX_WORD", "MAX_TEXT_BYTES", "MAX_BATCH",
    "MAX_STEPS", "MAX_SNAPSHOT_WORDS", "DEFAULT_SNAPSHOT_WORDS",
    "DEFAULT_STATUS_WORDS", "MAX_QUERY_HITS", "DEFAULT_QUERY_HITS",
    "MAX_OUTPUT_BYTES", "CycleToolError", "CycleValidationError",
    "CycleApprovalError", "CycleBackendError", "CycleConflictError",
    "CycleInvariantError", "status", "set", "set_edge", "cycle", "gauge",
    "train", "snapshot", "query", "HANDLERS", "build_handlers",
)
