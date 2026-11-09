"""Bounded tooling and persistence for the complete 32,385-cycle basis.

The resident pair rails define one oriented raw word ``x(i, j)`` for each
ordered pair of heads.  Relative to a root head ``r`` the star-basis flux is

    c_r(i, j) = x(r, i) + x(i, j) - x(r, j)

and there are ``C(255, 2) = 32,385`` independent such cycles.  The nearby
``cycle_tools`` module exposes individual edges/triangles; this module exposes
the *whole* basis through bounded, dependency-injected handlers and through a
sharded persistence adapter.

Two deliberately separate layers live here:

``FullCyclePersistence``
    Stores a canonical 32,385-entry basis as immutable, content-addressed
    shards plus one manifest and one version anchor inside a
    ``ResidentUniverse`` (or any object with ``ingest``/``get_leaf``).  Main
    writes require the literal ``approved=True``.  A load reconstructs all
    32,385 entries, verifies every shard digest and the basis digest, and
    rejects missing, extra, duplicated or truncated entries.

Tool handlers
    ``status``, ``write``, ``predict``, ``train``, ``snapshot``, ``audit`` and
    ``query`` operate on a small backend contract.  Every request property,
    collection and string is bounded; every response is JSON-safe, contains no
    raw pointer/offset fields, and declares canonical merge/promotion false.

Backend contract (aliases in parentheses are accepted)::

    snapshot() -> mapping
    read_word(i, j) -> int | mapping with ``word``       (get_word/read_edge)
    write_word(i, j, word[, expected_generation=...])     (set_word/set_edge)
    predict(text[, features=..., limit=...]) -> mapping or sequence
    train(text, batch, steps[, features=..., expected_generation=...])
    query(text[, limit]) -> mapping or sequence            (search/resonance)

``snapshot`` should report ``nodes``/``root``/``generation``/``digest`` plus
optional full-basis metadata (``full_basis``/``basis``, ``basis_count``/
``word_count``, ``basis_digest``, ``feature_count``, ``words``).  A backend may
also expose an optional ``audit()``; otherwise the audit is derived from the
snapshot.  Read-only actions are checked against generation and digest before
and after the call.

No file is opened here, no network request is made, no byte offset or live
buffer is exposed, and no canonical merge, promotion or truth claim is enabled.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable

SCHEMA = "nanlm.resident-full-cycle-tools/1"
PERSISTENCE_SCHEMA = "nanlm.resident-full-cycle-persistence/1"
BASIS_SCHEMA = "nanlm.resident-full-cycle-basis/1"
MANIFEST_SCHEMA = "nanlm.resident-full-cycle-manifest/1"
SHARD_SCHEMA = "nanlm.resident-full-cycle-shard/1"
ANCHOR_SCHEMA = "nanlm.resident-full-cycle-anchor/1"
AUDIT_SCHEMA = "nanlm.resident-full-cycle-audit/1"
RECEIPT_SCHEMA = "nanlm.resident-full-cycle-receipt/1"
UNIVERSE_SCHEMA = "nanlm.resident-universe/1"

HEAD_COUNT = 256
NODE_COUNT = HEAD_COUNT
MAX_NODE = NODE_COUNT - 1
PAIR_COUNT = NODE_COUNT * (NODE_COUNT - 1) // 2          # 32,640
EDGE_COUNT = PAIR_COUNT
CYCLE_DIMENSION = PAIR_COUNT - (NODE_COUNT - 1)          # 32,385
FULL_BASIS_CYCLES = CYCLE_DIMENSION
FULL_CYCLE_COUNT = CYCLE_DIMENSION
ROOT_CYCLE_COUNT = CYCLE_DIMENSION
BASIS_COUNT = CYCLE_DIMENSION
TRIANGLE_COUNT = NODE_COUNT * (NODE_COUNT - 1) * (NODE_COUNT - 2) // 6
DEFAULT_ROOT = 0

# Raw cycle words follow the resident CyclePersistence envelope: a signed or
# unsigned 64-bit integer is accepted and the exact integer is preserved.
WORD_BITS = 64
WORD_MODULUS = 1 << WORD_BITS
WORD_MASK = WORD_MODULUS - 1
WORD_MIN = -(1 << (WORD_BITS - 1))
WORD_MAX = WORD_MASK
MIN_WORD = WORD_MIN
MAX_WORD = WORD_MAX
U32_MASK = (1 << 32) - 1
MIN_CYCLE_WORD = -(1 << 31)
MAX_CYCLE_WORD = U32_MASK

MAX_FEATURES = CYCLE_DIMENSION
MAX_FEATURE_ENTRIES = CYCLE_DIMENSION
MAX_BASIS_ENTRIES = CYCLE_DIMENSION
MAX_SNAPSHOT_WORDS = CYCLE_DIMENSION
DEFAULT_SNAPSHOT_WORDS = 64
DEFAULT_STATUS_WORDS = 16
MAX_PREDICTIONS = CYCLE_DIMENSION
DEFAULT_PREDICTIONS = 8
MAX_QUERY_HITS = 32
DEFAULT_QUERY_HITS = 8
MAX_TEXT_BYTES = 4096
MAX_HIT_TEXT_BYTES = 512
MAX_REASON_BYTES = 240
MAX_REASONS = 16
MAX_EVENTS = 16
MAX_METRICS = 32
MAX_BATCH = 64
DEFAULT_BATCH = 1
MAX_STEPS = 256
MAX_TRAIN_STEPS = MAX_STEPS
DEFAULT_STEPS = 1
MAX_GENERATION = (1 << 63) - 1
MAX_OUTPUT_BYTES = 64 * 1024
MAX_UNKNOWN_PROPERTIES = 8

# Persistence shards keep every leaf comfortably inside the bounded envelope.
SHARD_CYCLES = 512
MAX_SHARD_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ANCHOR_BYTES = 8 * 1024
MAX_SOURCE_BYTES = 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_VERSIONS = 256
VERSION_GAP = 8
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 65536
MAX_STRING_BYTES = 4096

HEX_WORD_RE = __import__("re").compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
DEC_WORD_RE = __import__("re").compile(r"^[+-]?[0-9]+$")
EDGE_KEY_RE = __import__("re").compile(r"^(\d{1,3})[,:](\d{1,3})$")
EVENT_RE = __import__("re").compile(r"^[a-z][a-z0-9_]{0,31}$")
METRIC_KEY_RE = __import__("re").compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
MISSING = object()

_SCRATCH_EVENTS = frozenset({"scratch", "stage", "staged", "prepare"})
_MAIN_EVENTS = frozenset(
    {"main", "commit", "committed", "apply", "applied", "write"})
_POINTER_KEYS = frozenset({
    "pointer", "ptr", "raw_pointer", "raw_ptr", "offset", "address", "addr",
    "buffer", "buffer_address", "memory_address", "memoryview", "mmap",
    "cell_offset", "raw_offset", "file_offset", "slot", "handle",
})
_PATH_KEYS = frozenset({
    "cwd", "dir", "directory", "file", "filename", "filepath", "folder",
    "path", "uri", "url",
})
_SECRET_KEYS = frozenset({
    "apikey", "accesskey", "credential", "credentials", "password", "passwd",
    "passphrase", "privatekey", "secret", "secretkey", "session", "token",
})


class FullCycleError(ValueError):
    """Base error for full-cycle tooling and persistence."""


class FullCycleValidationError(FullCycleError):
    """A request property is missing, unknown, or outside its bound."""


class FullCycleApprovalError(FullCycleError):
    """A main-state operation lacks the literal ``approved=True`` value."""


class FullCycleBackendError(FullCycleError):
    """The injected backend returned an invalid or unsafe result."""


class FullCycleConflictError(FullCycleError):
    """A backend rejected a write; the handler renders it as denied."""

    def __init__(self, message: str = "write denied",
                 reasons: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = tuple(reasons or ())


class FullCycleInvariantError(FullCycleError):
    """A backend violated a read-only, ordering or generation invariant."""


class FullCyclePersistenceError(FullCycleError):
    """A persisted full basis cannot be represented or recovered safely."""


class FullCycleBoundsError(FullCyclePersistenceError):
    """A payload or entry count exceeds the bounded full-basis envelope."""


class FullCycleCorruptionError(FullCyclePersistenceError):
    """A stored shard, manifest, anchor or digest failed its check."""


class FullCyclePathError(FullCyclePersistenceError):
    """A payload contains filesystem-path-shaped material."""


class FullCycleSecretError(FullCyclePersistenceError):
    """A payload contains secret-shaped keys or values."""


class FullCyclePointerError(FullCyclePersistenceError):
    """A payload contains a raw-pointer-shaped key."""


class FullCycleReceipt(dict):
    """Persistence receipt whose truth value follows ``accepted``."""

    def __bool__(self) -> bool:
        return bool(self.get("accepted", False))


class FullBasisRecord(dict):
    """Loaded full basis.

    The mapping carries the metadata and both ``words`` (index order) and
    ``basis`` (``index``/``i``/``j``/``word`` records).  ``len(record)`` is the
    exact number of cycles, so callers can assert the 32,385 count directly.
    """

    def __len__(self) -> int:
        try:
            return int(self.get("basis_count", 0))
        except (TypeError, ValueError):
            return 0

    def __int__(self) -> int:
        return len(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def words(self) -> list[int]:
        return list(self.get("words", ()))

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.items()}


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------
def _params(params: Any, allowed: frozenset[str]) -> Mapping[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise FullCycleValidationError("params must be an object")
    unknown: list[str] = []
    for key in params:
        if not isinstance(key, str):
            raise FullCycleValidationError("property names must be strings")
        if len(key) > 64:
            raise FullCycleValidationError(
                "property name exceeds 64 characters")
        if key not in allowed:
            unknown.append(key)
            if len(unknown) >= MAX_UNKNOWN_PROPERTIES:
                break
    if unknown:
        raise FullCycleValidationError(
            "unknown properties: %s" % ", ".join(sorted(unknown)))
    return params


def _required(params: Mapping[str, Any], name: str) -> Any:
    if name not in params:
        raise FullCycleValidationError("%s is required" % name)
    return params[name]


def _require_approved(params: Mapping[str, Any]) -> None:
    if params.get("approved") is not True:
        raise FullCycleApprovalError("explicit approved=true is required")


def _input_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleValidationError("%s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise FullCycleValidationError(
            "%s must be in [%d, %d]" % (name, minimum, maximum))
    return value


def _input_node(value: Any, name: str = "node") -> int:
    return _input_int(value, name, 0, MAX_NODE)


def _input_root(value: Any, name: str = "root") -> int:
    return _input_node(value, name)


def _input_generation(value: Any,
                      name: str = "expected_generation") -> int:
    return _input_int(value, name, 0, MAX_GENERATION)


def _input_limit(value: Any, name: str, maximum: int) -> int:
    return _input_int(value, name, 1, maximum)


def _input_word(value: Any, name: str = "word") -> int:
    if isinstance(value, bool):
        raise FullCycleValidationError("%s must be an integer" % name)
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise FullCycleValidationError("%s must not be empty" % name)
        if HEX_WORD_RE.match(text):
            number = int(text, 16)
        elif DEC_WORD_RE.match(text):
            number = int(text, 10)
        else:
            raise FullCycleValidationError(
                "%s must be a decimal or 0x-hexadecimal integer" % name)
    elif isinstance(value, float):
        raise FullCycleValidationError(
            "%s must be an integer; floats are not permitted" % name)
    else:
        raise FullCycleValidationError(
            "%s must be an integer" % name)
    if not WORD_MIN <= number <= WORD_MAX:
        raise FullCycleValidationError(
            "%s is outside the signed/unsigned 64-bit range" % name)
    return int(number)


def _normalize_word(value: Any, name: str = "word") -> int:
    return _input_word(value, name) & WORD_MASK


def _signed_word(value: int) -> int:
    value &= WORD_MASK
    return value - WORD_MODULUS if value >= (1 << (WORD_BITS - 1)) else value


def _word_hex(value: int) -> str:
    return "0x%x" % (int(value) & WORD_MASK)


def _bounded_string(value: Any, name: str, maximum: int,
                    error_type: type[FullCycleError],
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
                           FullCycleValidationError, allow_empty=False)


def _input_source(value: Any) -> str:
    if value is None:
        return "unspecified"
    return _bounded_string(value, "source", MAX_SOURCE_BYTES,
                           FullCycleValidationError, allow_empty=True) \
        or "unspecified"


def _input_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FullCycleValidationError("metadata must be an object or null")
    try:
        text = json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FullCycleValidationError(
            "metadata must be JSON-safe") from exc
    if len(text.encode("utf-8")) > MAX_METADATA_BYTES:
        raise FullCycleValidationError(
            "metadata exceeds %d bytes" % MAX_METADATA_BYTES)
    _check_json(dict(value), label="metadata", max_bytes=MAX_METADATA_BYTES)
    return json.loads(text)


# ---------------------------------------------------------------------------
# basis coordinates and normalization
# ---------------------------------------------------------------------------
def _check_basis_pair(i: Any, j: Any, root: Any = DEFAULT_ROOT) -> tuple[int, int, int]:
    first = _input_node(i, "i")
    second = _input_node(j, "j")
    root_value = _input_root(root)
    if first == second:
        raise FullCycleValidationError("i and j must be distinct")
    if first == root_value or second == root_value:
        raise FullCycleValidationError(
            "root must be distinct from i and j")
    return (first, second, root_value) if first < second else \
        (second, first, root_value)


def _iter_basis_pairs(root: int = DEFAULT_ROOT):
    root = _input_root(root)
    for first in range(NODE_COUNT):
        if first == root:
            continue
        for second in range(first + 1, NODE_COUNT):
            if second == root:
                continue
            yield first, second


def full_basis_pairs(root: int = DEFAULT_ROOT) -> tuple[tuple[int, int], ...]:
    """Return the exact 32,385 star-basis pairs for ``root``."""
    pairs = tuple(_iter_basis_pairs(root))
    if len(pairs) != FULL_BASIS_CYCLES:
        raise AssertionError("full-basis enumeration is incomplete")
    return pairs


def _left_pair_count(left: int, root: int) -> int:
    return (NODE_COUNT - left - 1) - (1 if left < root else 0)


def basis_index(i: Any, j: Any, root: Any = DEFAULT_ROOT) -> int:
    """Return the canonical 0..32,384 index for one basis pair."""
    first, second, root_value = _check_basis_pair(i, j, root)
    if first > second:
        first, second = second, first
    index = 0
    for left in range(first):
        if left != root_value:
            index += _left_pair_count(left, root_value)
    within = second - first - 1
    if first < root_value < second:
        within -= 1
    return index + max(0, within)


def basis_pair(index: Any, root: Any = DEFAULT_ROOT) -> tuple[int, int]:
    """Return the ``(i, j)`` coordinate for a canonical basis index."""
    value = _input_int(index, "index", 0, FULL_BASIS_CYCLES - 1)
    root_value = _input_root(root)
    remaining = value
    for left in range(NODE_COUNT):
        if left == root_value:
            continue
        count = _left_pair_count(left, root_value)
        if remaining < count:
            second = left + 1 + remaining
            if left < root_value <= second:
                second += 1
            if second == root_value:
                second += 1
            return left, second
        remaining -= count
    raise FullCycleValidationError("basis index is outside the full basis")


def _basis_digest(words: Sequence[int], root: int = DEFAULT_ROOT) -> str:
    material = json.dumps(
        {"root": int(root), "head_count": HEAD_COUNT,
         "count": len(words), "words": [int(word) for word in words]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _entries_digest(entries: Sequence[Mapping[str, Any]],
                    root: int = DEFAULT_ROOT) -> str:
    material = json.dumps(
        {"root": int(root), "head_count": HEAD_COUNT,
         "entries": [[int(item["index"]), int(item["i"]), int(item["j"]),
                      int(item["word"])] for item in entries]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _coerce_basis_value(value: Any, *, label: str) -> int:
    try:
        return _input_word(value, label)
    except FullCycleValidationError as exc:
        raise FullCycleBoundsError(str(exc)) from exc


def _basis_entry_for_key(key: Any, value: Any,
                         root: int) -> tuple[int, int, int, int]:
    """Normalize one mapping entry into ``index, i, j, word``."""
    if isinstance(key, bool):
        raise FullCycleBoundsError("basis keys must not be booleans")
    index: int | None = None
    first: int | None = None
    second: int | None = None
    if isinstance(key, int):
        index = _input_int(key, "basis index", 0, FULL_BASIS_CYCLES - 1)
    elif isinstance(key, str):
        text = key.strip()
        match = EDGE_KEY_RE.fullmatch(text)
        if match is not None:
            first, second = int(match.group(1)), int(match.group(2))
        elif text.isdigit():
            index = _input_int(int(text), "basis index", 0,
                               FULL_BASIS_CYCLES - 1)
        else:
            raise FullCycleBoundsError(
                "basis keys must be indices or 'i:j' pairs")
    elif isinstance(key, (tuple, list)) and len(key) == 2:
        first, second = key[0], key[1]
    else:
        raise FullCycleBoundsError(
            "basis keys must be indices, 'i:j' strings or (i, j) pairs")

    if index is None:
        first, second, root = _check_basis_pair(first, second, root)
        index = basis_index(first, second, root)
    else:
        first, second = basis_pair(index, root)
    word = _coerce_basis_value(value, label="word")
    return int(index), int(first), int(second), int(word)


def _sequence_entry(item: Any, position: int,
                    root: int) -> tuple[int, int, int, int]:
    if isinstance(item, Mapping):
        if "index" in item:
            index = _input_int(item["index"], "basis index", 0,
                               FULL_BASIS_CYCLES - 1)
            first, second = basis_pair(index, root)
            raw = item.get("word", item.get("raw_word", item.get("value")))
            if raw is None:
                raise FullCycleBoundsError(
                    "basis entry %d is missing word" % position)
            return index, first, second, _coerce_basis_value(
                raw, label="basis word")
        raw = item.get("word", item.get("raw_word", item.get("value")))
        if raw is None:
            raise FullCycleBoundsError(
                "basis entry %d is missing word" % position)
        return _basis_entry_for_key(
            (item.get("i"), item.get("j")), raw, root)
    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            if isinstance(item[0], (tuple, list)) and len(item[0]) == 2:
                return _basis_entry_for_key(item[0], item[1], root)
            return _basis_entry_for_key(item[0], item[1], root)
        if len(item) == 3:
            return _basis_entry_for_key((item[0], item[1]), item[2], root)
        if len(item) == 4:
            index = _input_int(item[0], "basis index", 0,
                               FULL_BASIS_CYCLES - 1)
            first, second = _check_basis_pair(item[1], item[2], root)
            if basis_index(first, second, root) != index:
                raise FullCycleBoundsError(
                    "basis entry %d index does not match its pair" % position)
            return index, first, second, _coerce_basis_value(
                item[3], label="basis word")
    if isinstance(item, (str, bytes, bytearray)):
        raise FullCycleBoundsError(
            "basis entries must be integers, objects or triples")
    # A flat ordered coefficient list is handled by the caller.
    raise FullCycleBoundsError(
        "basis entry %d has an unsupported shape" % position)


def _normalize_full_basis(value: Any, *, root: Any = DEFAULT_ROOT,
                          require_full: bool = True
                          ) -> tuple[list[int], dict[str, Any]]:
    """Return exactly 32,385 words in canonical index order.

    Accepted shapes are an ordered 32,385-integer sequence, a mapping keyed by
    index/``"i:j"``/pair, or a sequence of entry objects/triples.  Anything
    shorter, longer, duplicated or out of range is denied before persistence.
    """
    root_value = _input_root(root)
    metadata: dict[str, Any] = {"root": root_value}
    if isinstance(value, FullBasisRecord):
        metadata = {key: value[key] for key in value
                    if key not in ("words", "basis", "features", "cycles")}
        metadata["root"] = root_value
        value = value.get("words", value.get("basis"))
    elif isinstance(value, Mapping):
        source = value
        if "root" in source:
            try:
                root_value = _input_root(source["root"])
            except FullCycleValidationError as exc:
                raise FullCycleBoundsError(str(exc)) from exc
            metadata["root"] = root_value
        for key in ("words", "basis", "cycles", "coefficients",
                    "full_basis", "feature_map"):
            if key in source:
                candidate = source[key]
                if key == "words" and isinstance(candidate, (list, tuple)) \
                        and len(candidate) == FULL_BASIS_CYCLES and all(
                            isinstance(item, int)
                            and not isinstance(item, bool)
                            for item in candidate):
                    value = candidate
                else:
                    value = candidate
                break
        else:
            value = source
        if "metadata" in source and isinstance(source["metadata"], Mapping):
            metadata["metadata"] = dict(source["metadata"])
        if "source" in source and isinstance(source["source"], str):
            metadata["source"] = source["source"]

    if isinstance(value, Mapping):
        entries: list[tuple[int, int, int, int]] = []
        for key, item in value.items():
            entries.append(_basis_entry_for_key(key, item, root_value))
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if all(isinstance(item, int) and not isinstance(item, bool)
               for item in items):
            if len(items) != FULL_BASIS_CYCLES:
                raise FullCycleBoundsError(
                    "full basis is truncated: expected %d coefficients, "
                    "found %d" % (FULL_BASIS_CYCLES, len(items)))
            words = [_input_word(item, "basis word") for item in items]
            return words, metadata
        if len(items) != FULL_BASIS_CYCLES:
            raise FullCycleBoundsError(
                "full basis is truncated: expected %d entries, found %d"
                % (FULL_BASIS_CYCLES, len(items)))
        entries = [
            _sequence_entry(item, position, root_value)
            for position, item in enumerate(items)
        ]
    else:
        raise FullCycleBoundsError(
            "full basis must be an ordered sequence or a mapping")

    if len(entries) != FULL_BASIS_CYCLES:
        raise FullCycleBoundsError(
            "full basis is truncated: expected %d entries, found %d"
            % (FULL_BASIS_CYCLES, len(entries)))
    words = [0] * FULL_BASIS_CYCLES
    seen: set[int] = set()
    for index, first, second, word in entries:
        if index in seen:
            raise FullCycleBoundsError(
                "full basis contains duplicate index %d" % index)
        seen.add(index)
        expected = basis_pair(index, root_value)
        if (first, second) != expected:
            raise FullCycleBoundsError(
                "basis index %d has the wrong coordinate" % index)
        words[index] = int(word)
    if require_full and len(seen) != FULL_BASIS_CYCLES:
        raise FullCycleBoundsError(
            "full basis is truncated: expected %d entries, found %d"
            % (FULL_BASIS_CYCLES, len(seen)))
    return words, metadata


def _feature_entries(value: Any, *, root: Any = DEFAULT_ROOT,
                     limit: int = MAX_FEATURES
                     ) -> list[dict[str, int]]:
    """Normalize a bounded sparse feature map to index-ordered entries."""
    if value is None:
        return []
    root_value = _input_root(root)
    if isinstance(value, Mapping):
        entries = [
            _basis_entry_for_key(key, item, root_value)
            for key, item in value.items()
        ]
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) > limit:
            raise FullCycleBoundsError(
                "feature map exceeds %d entries" % limit)
        if all(isinstance(item, int) and not isinstance(item, bool)
               for item in items):
            if len(items) > FULL_BASIS_CYCLES:
                raise FullCycleBoundsError(
                    "feature map exceeds %d entries" % FULL_BASIS_CYCLES)
            entries = [
                (index, *basis_pair(index, root_value), _input_word(
                    item, "feature word"))
                for index, item in enumerate(items)
            ]
        else:
            entries = [
                _sequence_entry(item, position, root_value)
                for position, item in enumerate(items)
            ]
    else:
        raise FullCycleBoundsError(
            "feature map must be an object or sequence")
    if len(entries) > limit:
        raise FullCycleBoundsError(
            "feature map exceeds %d entries" % limit)
    result: dict[int, dict[str, int]] = {}
    for index, first, second, word in entries:
        if index in result:
            raise FullCycleBoundsError(
                "feature map contains duplicate index %d" % index)
        result[index] = {
            "index": int(index), "i": int(first), "j": int(second),
            "word": int(word),
        }
    return [result[index] for index in sorted(result)]


def _bounded_feature_sample(entries: Sequence[Mapping[str, int]],
                            limit: int
                            ) -> tuple[list[dict[str, int]], int, bool]:
    items = list(entries)
    return items[:limit], len(items), len(items) > limit


# ---------------------------------------------------------------------------
# JSON/payload validation for persisted data
# ---------------------------------------------------------------------------
def _compact_key(key: str) -> str:
    return "".join(character for character in key.casefold()
                   if character.isalnum())


def _check_payload_key(key: str, *, label: str) -> None:
    compact = _compact_key(key)
    if compact in _POINTER_KEYS or any(
            compact.endswith(suffix) for suffix in
            ("pointer", "ptr", "offset", "address", "buffer", "mmap")):
        raise FullCyclePointerError(
            "%s contains a raw-pointer-shaped key %r" % (label, key))
    if compact in _SECRET_KEYS or any(
            compact.endswith(suffix) for suffix in _SECRET_KEYS):
        raise FullCycleSecretError(
            "%s contains a secret-shaped key %r" % (label, key))


def _check_payload_string(value: str, *, label: str, key: str | None = None) -> None:
    if "\x00" in value:
        raise FullCyclePathError("%s contains a NUL byte" % label)
    text = value.strip()
    if text.startswith(("/", "\\", "~")):
        raise FullCyclePathError(
            "%s contains an absolute or home-relative path" % label)
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        raise FullCyclePathError("%s contains a drive path" % label)
    parts = text.replace("\\", "/").split("/")
    if ".." in parts:
        raise FullCyclePathError("%s contains path traversal" % label)
    if key is not None and _compact_key(key) in _PATH_KEYS:
        raise FullCyclePathError(
            "%s key %r contains a path-like value" % (label, key))
    if text.casefold().startswith("-----begin ") and "private key" in \
            text.casefold():
        raise FullCycleSecretError(
            "%s contains secret-shaped data" % label)


def _check_json(value: Any, *, label: str, max_bytes: int | None = None,
                depth: int = 0, counter: list[int] | None = None,
                key: str | None = None) -> Any:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise FullCycleBoundsError(
            "%s exceeds %d JSON nodes" % (label, MAX_JSON_NODES))
    if depth > MAX_JSON_DEPTH:
        raise FullCycleBoundsError(
            "%s exceeds JSON depth %d" % (label, MAX_JSON_DEPTH))
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise FullCycleBoundsError(
                "%s contains an oversized integer" % label)
        return value
    if isinstance(value, float):
        raise FullCycleBoundsError(
            "%s contains a float; full-cycle payloads require integers only"
            % label)
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise FullCycleBoundsError(
                "%s contains an oversized string" % label)
        _check_payload_string(value, label=label, key=key)
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_BASIS_ENTRIES:
            raise FullCycleBoundsError(
                "%s exceeds %d entries" % (label, MAX_BASIS_ENTRIES))
        return [
            _check_json(item, label=label, depth=depth + 1,
                        counter=counter, key=key)
            for item in value
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                raise FullCyclePersistenceError(
                    "%s object keys must be strings" % label)
            if not item_key or len(item_key.encode("utf-8")) > 256:
                raise FullCycleBoundsError(
                    "%s contains an invalid object key" % label)
            _check_payload_key(item_key, label=label)
            _check_payload_string(item_key, label="%s key" % label)
            result[item_key] = _check_json(
                item, label=label, depth=depth + 1,
                counter=counter, key=item_key)
        return result
    raise FullCyclePersistenceError(
        "%s contains a non-JSON value" % label)


def _canonical_text(value: Any, *, label: str,
                    max_bytes: int) -> tuple[str, Any]:
    clean = _check_json(value, label=label)
    try:
        text = json.dumps(
            clean, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FullCyclePersistenceError(
            "%s cannot be encoded as canonical JSON" % label) from exc
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise FullCycleBoundsError(
            "%s is %d bytes; the limit is %d bytes"
            % (label, size, max_bytes))
    return text, clean


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shard_id(digest: str) -> str:
    return "full-cycle-shard:" + digest[:32]


def _manifest_id(digest: str) -> str:
    return "full-cycle-manifest:" + digest[:32]


def _anchor_id(version: int) -> str:
    return "full-cycle-anchor:%d" % int(version)


def _valid_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if len(text) == 64 and all(
        character in "0123456789abcdef" for character in text) else None


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
class FullCyclePersistence:
    """Sharded, content-addressed persistence of a full 32,385-cycle basis.

    Only ``ingest`` and ``get_leaf`` are required from the universe facade;
    ``snapshot`` supplies generation/digest receipts and ``checkpoint`` is
    delegated when available.  Main writes require literal approval.  The
    latest version is loaded by default, and every load verifies all shard
    counts and digests before returning data.
    """

    def __init__(self, universe: Any):
        for name in ("ingest", "get_leaf"):
            if not callable(getattr(universe, name, None)):
                raise TypeError("universe must expose %s()" % name)
        self.universe = universe

    # ------------------------------------------------------------- public API
    def save_full_basis(self, basis: Any, *, approved: bool = False,
                        root: Any = DEFAULT_ROOT, source: Any = None,
                        metadata: Any = None,
                        expected_generation: Any = None) -> FullCycleReceipt:
        """Commit one complete basis as a new immutable version.

        Validation happens before the universe is touched.  Without
        ``approved=True`` or with a stale ``expected_generation`` a falsey
        denial receipt is returned and no leaf is written.
        """
        root_value = _input_root(root)
        words, detected = _normalize_full_basis(basis, root=root_value)
        if "root" in detected and int(detected["root"]) != root_value:
            raise FullCycleBoundsError(
                "basis root does not match the requested root")
        basis_digest = _basis_digest(words, root_value)
        source_text = _input_source(source)
        clean_metadata = _input_metadata(metadata)
        state_before = self._state()

        if expected_generation is not None:
            try:
                expected = _input_generation(expected_generation)
            except FullCycleValidationError as exc:
                raise FullCycleBoundsError(str(exc)) from exc
            if expected != state_before["generation"]:
                return self._denial(
                    "stale_generation",
                    "expected generation %d does not match current %d"
                    % (expected, state_before["generation"]),
                    approved=approved is True, root=root_value,
                    basis_count=FULL_BASIS_CYCLES,
                    basis_digest=basis_digest,
                    expected_generation=expected,
                    previous_generation=state_before["generation"])

        if approved is not True:
            return self._denial(
                "approval_required",
                "explicit approved=True is required for a main full-basis "
                "commit",
                approved=False, root=root_value,
                basis_count=FULL_BASIS_CYCLES,
                basis_digest=basis_digest,
                previous_generation=state_before["generation"])

        latest = self._latest_anchor()
        if latest is not None and latest.get("basis_digest") == basis_digest \
                and int(latest.get("root", DEFAULT_ROOT)) == root_value:
            return self._receipt(
                status="duplicate", accepted=True, created=False,
                duplicate=True, approved=True, root=root_value,
                basis_count=FULL_BASIS_CYCLES, basis_digest=basis_digest,
                version=int(latest.get("version", 0)),
                anchor_leaf_id=latest.get("anchor_leaf_id"),
                manifest_leaf_id=latest.get("manifest_leaf_id"),
                manifest_digest=latest.get("manifest_digest"),
                generation_before=state_before["generation"],
                data_generation=latest.get("data_generation"),
                commit_generation=latest.get("commit_generation"),
                source=source_text, metadata=clean_metadata)
        version = 0 if latest is None else int(latest.get("version", 0)) + 1
        if version >= MAX_VERSIONS:
            raise FullCycleBoundsError(
                "full-basis version limit (%d) reached" % MAX_VERSIONS)

        # Build and bound every leaf *before* touching the universe.
        shards: list[dict[str, Any]] = []
        shard_records: list[tuple[str, str, str]] = []
        for start in range(0, FULL_BASIS_CYCLES, SHARD_CYCLES):
            chunk = words[start:start + SHARD_CYCLES]
            body = {
                "schema": SHARD_SCHEMA,
                "kind": "full-cycle-shard",
                "root": root_value,
                "start": start,
                "count": len(chunk),
                "words": [int(word) for word in chunk],
            }
            text, _clean = _canonical_text(
                body, label="full-cycle shard", max_bytes=MAX_SHARD_BYTES)
            digest = _sha256_text(text)
            leaf_id = _shard_id(digest)
            shards.append({
                "start": start, "count": len(chunk), "digest": digest,
                "leaf_id": leaf_id,
            })
            shard_records.append((leaf_id, text, digest))
        if not shards or sum(item["count"] for item in shards) != \
                FULL_BASIS_CYCLES:
            raise FullCycleBoundsError(
                "shard plan does not cover exactly %d cycles"
                % FULL_BASIS_CYCLES)

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "kind": "full-cycle-manifest",
            "root": root_value,
            "head_count": HEAD_COUNT,
            "basis_count": FULL_BASIS_CYCLES,
            "index_count": FULL_BASIS_CYCLES,
            "first_index": 0,
            "last_index": FULL_BASIS_CYCLES - 1,
            "basis_digest": basis_digest,
            "source": source_text,
            "metadata": clean_metadata,
            "shards": shards,
            "shard_count": len(shards),
            "no_floats": True,
            "no_raw_pointers": True,
        }
        manifest_text, _manifest_clean = _canonical_text(
            manifest, label="full-cycle manifest", max_bytes=MAX_MANIFEST_BYTES)
        manifest_digest = _sha256_text(manifest_text)
        manifest_leaf_id = _manifest_id(manifest_digest)

        data_generation = 0
        leaf_ids: list[str] = []
        for leaf_id, text, _digest in shard_records:
            result = self._ingest_leaf(leaf_id, text, head=0,
                                       label="full-cycle shard")
            data_generation = max(data_generation,
                                  self._generation(result))
            leaf_ids.append(leaf_id)
        result = self._ingest_leaf(
            manifest_leaf_id, manifest_text, head=0,
            label="full-cycle manifest")
        data_generation = max(data_generation, self._generation(result))
        leaf_ids.append(manifest_leaf_id)

        state_after_data = self._state()
        anchor = {
            "schema": ANCHOR_SCHEMA,
            "kind": "full-cycle-anchor",
            "version": int(version),
            "root": root_value,
            "head_count": HEAD_COUNT,
            "basis_count": FULL_BASIS_CYCLES,
            "basis_digest": basis_digest,
            "manifest_leaf_id": manifest_leaf_id,
            "manifest_digest": manifest_digest,
            "data_generation": int(data_generation),
            "commit_generation": int(state_after_data["generation"]),
            "read_only": True,
            "no_raw_pointers": True,
        }
        anchor_text, _anchor_clean = _canonical_text(
            anchor, label="full-cycle anchor", max_bytes=MAX_ANCHOR_BYTES)
        anchor_leaf_id = _anchor_id(version)
        anchor_result = self._ingest_leaf(
            anchor_leaf_id, anchor_text, head=0, label="full-cycle anchor")
        leaf_ids.append(anchor_leaf_id)

        stored = self._read_anchor(version)
        if stored is None or stored.get("basis_digest") != basis_digest:
            raise FullCycleCorruptionError(
                "full-cycle anchor %s did not commit the requested basis"
                % anchor_leaf_id)
        return self._receipt(
            status="committed", accepted=True, created=True, duplicate=False,
            approved=True, root=root_value, basis_count=FULL_BASIS_CYCLES,
            index_count=FULL_BASIS_CYCLES, basis_digest=basis_digest,
            version=int(version), anchor_leaf_id=anchor_leaf_id,
            manifest_leaf_id=manifest_leaf_id,
            manifest_digest=manifest_digest, shard_count=len(shards),
            leaf_ids=leaf_ids,
            data_generation=self._generation(anchor_result),
            commit_generation=stored.get("commit_generation"),
            generation_before=state_before["generation"],
            source=source_text, metadata=clean_metadata,
            main_modified=True, main_cycle_modified=True)

    def load_full_basis(self, version: int | None = None, *,
                        verify: bool = True) -> FullBasisRecord | None:
        """Load a complete basis, latest by default, or return ``None``.

        The returned mapping has ``words`` in canonical index order and a
        ``basis`` list of coordinate records.  A short/extra shard, digest
        mismatch or non-exact count raises ``FullCycleCorruptionError``.
        """
        if version is None:
            anchor = self._latest_anchor()
        else:
            try:
                version_value = _input_int(
                    version, "version", 0, MAX_VERSIONS - 1)
            except FullCycleValidationError as exc:
                raise FullCycleBoundsError(str(exc)) from exc
            anchor = self._read_anchor(version_value)
        if anchor is None:
            return None
        root_value = int(anchor["root"])
        manifest = self._read_manifest(anchor)
        shards = manifest.get("shards")
        if not isinstance(shards, list) or len(shards) != int(
                manifest.get("shard_count", -1)):
            raise FullCycleCorruptionError(
                "full-cycle manifest has an inconsistent shard list")
        words: list[int] = []
        for expected_start, item in enumerate(shards):
            if not isinstance(item, Mapping):
                raise FullCycleCorruptionError(
                    "full-cycle manifest shard is not an object")
            start = int(item.get("start", -1))
            count = int(item.get("count", -1))
            expected = expected_start * SHARD_CYCLES
            if start != expected or count < 1 or count > SHARD_CYCLES:
                raise FullCycleCorruptionError(
                    "full-cycle shard %d has an invalid range" % expected_start)
            body = self._read_shard(item)
            chunk = body.get("words")
            if not isinstance(chunk, list) or len(chunk) != count:
                raise FullCycleCorruptionError(
                    "full-cycle shard %d is truncated" % expected_start)
            for word in chunk:
                if isinstance(word, bool) or not isinstance(word, int):
                    raise FullCycleCorruptionError(
                        "full-cycle shard %d contains a non-integer word"
                        % expected_start)
                if not WORD_MIN <= word <= WORD_MAX:
                    raise FullCycleCorruptionError(
                        "full-cycle shard %d contains an out-of-range word"
                        % expected_start)
                words.append(int(word))
        if len(words) != FULL_BASIS_CYCLES:
            raise FullCycleCorruptionError(
                "full basis is truncated: expected %d words, found %d"
                % (FULL_BASIS_CYCLES, len(words)))
        digest = _basis_digest(words, root_value)
        if verify and digest != manifest.get("basis_digest"):
            raise FullCycleCorruptionError(
                "full basis failed its content digest")
        if verify and digest != anchor.get("basis_digest"):
            raise FullCycleCorruptionError(
                "full basis digest disagrees with its anchor")
        state = self._state()
        basis = [
            {"index": index, "i": pair[0], "j": pair[1], "word": words[index]}
            for index, pair in enumerate(_iter_basis_pairs(root_value))
        ]
        record = FullBasisRecord({
            "schema": BASIS_SCHEMA,
            "kind": "full-cycle-basis",
            "root": root_value,
            "head_count": HEAD_COUNT,
            "basis_count": FULL_BASIS_CYCLES,
            "index_count": FULL_BASIS_CYCLES,
            "word_count": FULL_BASIS_CYCLES,
            "count": FULL_BASIS_CYCLES,
            "complete": True,
            "words": words,
            "coefficients": list(words),
            "basis": basis,
            "cycles": basis,
            "basis_digest": digest,
            "content_digest": digest,
            "manifest_digest": anchor.get("manifest_digest"),
            "version": int(anchor.get("version", 0)),
            "generation": state["generation"],
            "digest": state["digest"],
            "state_digest": state["digest"],
            "commit_generation": anchor.get("commit_generation"),
            "data_generation": anchor.get("data_generation"),
            "manifest_leaf_id": anchor.get("manifest_leaf_id"),
            "anchor_leaf_id": anchor.get("anchor_leaf_id"),
            "source": manifest.get("source", "unspecified"),
            "metadata": manifest.get("metadata", {}),
            "shard_count": len(shards),
            "read_only": True,
            "no_floats": True,
            "no_raw_pointers": True,
            "canonical_commit": False,
            "canonical_state_modified": False,
        })
        return record

    def audit(self, *, full: bool = True) -> dict[str, Any]:
        """Verify the latest full basis without changing the universe."""
        before = self._state()
        problems: list[str] = []
        codes: list[str] = []
        anchor = self._latest_anchor()
        loaded = None
        if anchor is None:
            return {
                "schema": AUDIT_SCHEMA,
                "ok": False,
                "accepted": False,
                "basis_count": 0,
                "expected_count": FULL_BASIS_CYCLES,
                "complete": False,
                "problems": ["no full-cycle basis is persisted"],
                "problem_codes": ["missing_basis"],
                "generation": before["generation"],
                "digest": before["digest"],
                "read_only_main": True,
                "canonical_commit": False,
                "canonical_state_modified": False,
            }
        try:
            loaded = self.load_full_basis(verify=True)
        except FullCycleError as exc:
            problems.append(str(exc))
            codes.append("basis_corruption")
        after = self._state()
        if before["generation"] != after["generation"] or \
                before["digest"] != after["digest"]:
            raise FullCycleInvariantError(
                "full-cycle audit must be read-only")
        count = 0 if loaded is None else len(loaded)
        if loaded is not None and count != FULL_BASIS_CYCLES:
            problems.append(
                "full basis count is %d, expected %d"
                % (count, FULL_BASIS_CYCLES))
            codes.append("basis_truncated")
        return {
            "schema": AUDIT_SCHEMA,
            "ok": not problems,
            "accepted": not problems,
            "basis_count": count,
            "expected_count": FULL_BASIS_CYCLES,
            "complete": count == FULL_BASIS_CYCLES,
            "version": int(anchor.get("version", 0)),
            "root": int(anchor.get("root", DEFAULT_ROOT)),
            "basis_digest": anchor.get("basis_digest"),
            "manifest_digest": anchor.get("manifest_digest"),
            "shard_count": 0 if loaded is None else loaded.get("shard_count", 0),
            "checked": count,
            "problems": problems[:MAX_REASONS],
            "problem_codes": codes[:MAX_REASONS],
            "generation": after["generation"],
            "digest": after["digest"],
            "state_digest": after["digest"],
            "read_only_main": True,
            "no_floats": True,
            "no_raw_pointers": True,
            "canonical_commit": False,
            "canonical_state_modified": False,
        }

    def status(self) -> dict[str, Any]:
        """Return a small read-only description of the persisted basis."""
        state = self._state()
        anchor = self._latest_anchor()
        return {
            "schema": RECEIPT_SCHEMA,
            "generation": state["generation"],
            "digest": state["digest"],
            "state_digest": state["digest"],
            "basis_count": 0 if anchor is None
            else int(anchor.get("basis_count", 0)),
            "expected_count": FULL_BASIS_CYCLES,
            "complete": anchor is not None and int(
                anchor.get("basis_count", 0)) == FULL_BASIS_CYCLES,
            "version": None if anchor is None
            else int(anchor.get("version", 0)),
            "root": None if anchor is None
            else int(anchor.get("root", DEFAULT_ROOT)),
            "basis_digest": None if anchor is None
            else anchor.get("basis_digest"),
            "read_only_main": True,
            "canonical_commit": False,
            "canonical_state_modified": False,
        }

    def checkpoint(self, directory: Any = None) -> Any:
        """Delegate a checkpoint to the underlying universe, if supported."""
        checkpoint = getattr(self.universe, "checkpoint", None)
        if not callable(checkpoint):
            raise FullCyclePersistenceError(
                "the universe does not expose checkpoint()")
        return checkpoint(directory)

    save_basis = save_full_basis
    save = save_full_basis
    save_full_cycle_memory = save_full_basis
    load_basis = load_full_basis
    load = load_full_basis
    load_full_cycle_memory = load_full_basis

    # --------------------------------------------------------------- internals
    def _state(self) -> dict[str, Any]:
        snapshot: Mapping[str, Any] = {}
        getter = getattr(self.universe, "snapshot", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if isinstance(value, Mapping):
                snapshot = value
        generation = snapshot.get("generation", 0)
        digest = snapshot.get("digest") or snapshot.get("state_digest") or ""
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 0
        return {"generation": max(0, generation), "digest": str(digest)}

    @staticmethod
    def _generation(result: Any) -> int:
        if isinstance(result, Mapping):
            try:
                return int(result.get("generation", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def _ingest_leaf(self, leaf_id: str, text: str, *, head: int,
                     label: str) -> Any:
        record = {
            "type": "leaf",
            "schema": UNIVERSE_SCHEMA,
            "leaf_id": leaf_id,
            "kind": "data",
            "head": int(head),
            "text": text,
            "features": [],
            "no_feature": True,
            "source_digest": _sha256_text(text),
            "full_cycle_persistence": True,
            "full_cycle_label": label,
        }
        result = self.universe.ingest(record)
        if not isinstance(result, Mapping):
            raise FullCyclePersistenceError(
                "universe.ingest() returned a non-mapping receipt")
        return result

    def _get_leaf(self, leaf_id: str) -> Mapping[str, Any] | None:
        try:
            leaf = self.universe.get_leaf(leaf_id)
        except (KeyError, LookupError):
            return None
        return leaf if isinstance(leaf, Mapping) else None

    def _leaf_text(self, leaf: Mapping[str, Any], *, max_bytes: int,
                   label: str) -> str:
        raw = leaf.get("text")
        if raw is None:
            raw = leaf.get("payload")
        if not isinstance(raw, str):
            raise FullCycleCorruptionError(
                "%s is not a JSON text leaf" % label)
        if len(raw.encode("utf-8")) > max_bytes:
            raise FullCycleBoundsError(
                "%s exceeds %d bytes" % (label, max_bytes))
        return raw

    @staticmethod
    def _parse_object(text: str, *, label: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullCycleCorruptionError(
                "%s is not valid JSON" % label) from exc
        if not isinstance(value, dict):
            raise FullCycleCorruptionError(
                "%s is not a JSON object" % label)
        return value

    def _read_anchor(self, version: int) -> dict[str, Any] | None:
        leaf_id = _anchor_id(version)
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            return None
        text = self._leaf_text(leaf, max_bytes=MAX_ANCHOR_BYTES,
                               label="full-cycle anchor")
        anchor = self._parse_object(text, label="full-cycle anchor %s"
                                    % leaf_id)
        if anchor.get("schema") != ANCHOR_SCHEMA \
                or anchor.get("kind") != "full-cycle-anchor":
            raise FullCycleCorruptionError(
                "full-cycle anchor %s has an unexpected schema/kind"
                % leaf_id)
        if int(anchor.get("version", -1)) != int(version):
            raise FullCycleCorruptionError(
                "full-cycle anchor %s has the wrong version" % leaf_id)
        if int(anchor.get("basis_count", -1)) != FULL_BASIS_CYCLES:
            raise FullCycleCorruptionError(
                "full-cycle anchor %s does not declare %d cycles"
                % (leaf_id, FULL_BASIS_CYCLES))
        if _valid_digest(anchor.get("basis_digest")) is None \
                or _valid_digest(anchor.get("manifest_digest")) is None:
            raise FullCycleCorruptionError(
                "full-cycle anchor %s has invalid digests" % leaf_id)
        manifest_leaf_id = anchor.get("manifest_leaf_id")
        if not isinstance(manifest_leaf_id, str) or \
                not manifest_leaf_id.startswith("full-cycle-manifest:"):
            raise FullCycleCorruptionError(
                "full-cycle anchor %s has an invalid manifest reference"
                % leaf_id)
        anchor["anchor_leaf_id"] = leaf_id
        anchor["version"] = int(version)
        anchor["root"] = _input_root(anchor.get("root", DEFAULT_ROOT))
        return anchor

    def _latest_anchor(self) -> dict[str, Any] | None:
        for version in range(MAX_VERSIONS - 1, -1, -1):
            anchor = self._read_anchor(version)
            if anchor is not None:
                return anchor
        return None

    def _read_manifest(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        leaf_id = str(anchor.get("manifest_leaf_id"))
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            raise FullCycleCorruptionError(
                "full-cycle manifest %s is missing" % leaf_id)
        text = self._leaf_text(leaf, max_bytes=MAX_MANIFEST_BYTES,
                               label="full-cycle manifest")
        digest = _sha256_text(text)
        if digest != anchor.get("manifest_digest") or \
                _manifest_id(digest) != leaf_id:
            raise FullCycleCorruptionError(
                "full-cycle manifest %s failed its content digest" % leaf_id)
        manifest = self._parse_object(text, label="full-cycle manifest")
        if manifest.get("schema") != MANIFEST_SCHEMA \
                or manifest.get("kind") != "full-cycle-manifest":
            raise FullCycleCorruptionError(
                "full-cycle manifest %s has an unexpected schema/kind"
                % leaf_id)
        if int(manifest.get("basis_count", -1)) != FULL_BASIS_CYCLES:
            raise FullCycleCorruptionError(
                "full-cycle manifest %s does not declare %d cycles"
                % (leaf_id, FULL_BASIS_CYCLES))
        if int(manifest.get("root", -1)) != int(anchor.get("root", -2)):
            raise FullCycleCorruptionError(
                "full-cycle manifest root disagrees with its anchor")
        if manifest.get("basis_digest") != anchor.get("basis_digest"):
            raise FullCycleCorruptionError(
                "full-cycle manifest digest disagrees with its anchor")
        return manifest

    def _read_shard(self, item: Mapping[str, Any]) -> dict[str, Any]:
        leaf_id = str(item.get("leaf_id"))
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            raise FullCycleCorruptionError(
                "full-cycle shard %s is missing" % leaf_id)
        text = self._leaf_text(leaf, max_bytes=MAX_SHARD_BYTES,
                               label="full-cycle shard")
        digest = _sha256_text(text)
        if digest != item.get("digest") or _shard_id(digest) != leaf_id:
            raise FullCycleCorruptionError(
                "full-cycle shard %s failed its content digest" % leaf_id)
        body = self._parse_object(text, label="full-cycle shard")
        if body.get("schema") != SHARD_SCHEMA \
                or body.get("kind") != "full-cycle-shard":
            raise FullCycleCorruptionError(
                "full-cycle shard %s has an unexpected schema/kind" % leaf_id)
        if int(body.get("start", -1)) != int(item.get("start", -2)) or \
                int(body.get("count", -1)) != int(item.get("count", -2)):
            raise FullCycleCorruptionError(
                "full-cycle shard %s range disagrees with its manifest"
                % leaf_id)
        return body

    def _receipt(self, *, status: str, accepted: bool, created: bool,
                 duplicate: bool, approved: bool | None,
                 root: int = DEFAULT_ROOT, basis_count: int = 0,
                 basis_digest: str | None = None,
                 **extra: Any) -> FullCycleReceipt:
        state = self._state()
        result: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": status,
            "accepted": bool(accepted),
            "created": bool(created),
            "duplicate": bool(duplicate),
            "persisted": bool(created),
            "approved": approved,
            "kind": "full-cycle-basis",
            "root": int(root),
            "basis_count": int(basis_count),
            "index_count": int(basis_count),
            "word_count": int(basis_count),
            "expected_count": FULL_BASIS_CYCLES,
            "complete": int(basis_count) == FULL_BASIS_CYCLES,
            "basis_digest": basis_digest,
            "content_digest": basis_digest,
            "generation": state["generation"],
            "digest": state["digest"],
            "state_digest": state["digest"],
            "generation_before": extra.pop("generation_before", None),
            "main_modified": extra.pop("main_modified", False),
            "main_cycle_modified": extra.pop("main_cycle_modified", False),
            "read_only_main": not bool(created),
            "canonical_commit": False,
            "canonical_state_modified": False,
            "no_floats": True,
            "no_raw_pointers": True,
        }
        result.update(extra)
        return FullCycleReceipt(result)

    def _denial(self, status: str, reason: str, *, approved: bool,
                **extra: Any) -> FullCycleReceipt:
        return self._receipt(
            status=status, accepted=False, created=False, duplicate=False,
            approved=approved, reasons=[reason], **extra)


# ---------------------------------------------------------------------------
# backend invocation / result validation
# ---------------------------------------------------------------------------
def _backend_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullCycleBackendError("backend %s must return an object" % name)
    return value


def _backend_int(value: Any, name: str, minimum: int = WORD_MIN,
                 maximum: int = WORD_MAX, default: int | None = None
                 ) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleBackendError("backend %s must be an integer" % name)
    if not minimum <= value <= maximum:
        raise FullCycleBackendError(
            "backend %s is outside [%d, %d]" % (name, minimum, maximum))
    return int(value)


def _backend_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullCycleBackendError("backend %s must be a number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise FullCycleBackendError("backend %s must be finite" % name)
    return round(result, 8)


def _backend_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise FullCycleBackendError("backend %s must be a boolean" % name)
    return value


def _backend_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (bytes, bytearray)) or \
            not isinstance(value, Sequence):
        raise FullCycleBackendError("backend reasons must be a sequence")
    if len(value) > MAX_REASONS:
        raise FullCycleBackendError(
            "backend reasons exceed %d entries" % MAX_REASONS)
    return [
        _bounded_string(item, "reason", MAX_REASON_BYTES,
                        FullCycleBackendError, allow_empty=True)
        for item in value
    ]


def _backend_digest(value: Any, name: str = "digest") -> str | None:
    if value is None:
        return None
    text = _bounded_string(value, name, 128, FullCycleBackendError,
                           allow_empty=True)
    return text or None


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


def _call(backend: Any, names: Sequence[str], *args: Any,
          expected_generation: Any = MISSING, **kwargs: Any) -> Any:
    if backend is None:
        raise FullCycleValidationError("backend is required")
    found = _find_method(backend, names)
    if found is None:
        raise FullCycleBackendError("backend must provide %s()" % names[0])
    method_name, function = found
    call_kwargs: dict[str, Any] = {}
    if expected_generation is not MISSING and \
            _accepts_keyword(function, "expected_generation"):
        call_kwargs["expected_generation"] = expected_generation
    for key, value in kwargs.items():
        if _accepts_keyword(function, key):
            call_kwargs[key] = value
    try:
        return function(*args, **call_kwargs)
    except FullCycleError:
        raise
    except Exception as exc:
        raise FullCycleBackendError(
            "backend %s failed: %s" % (method_name, type(exc).__name__)) \
            from exc


def _snapshot(backend: Any) -> Mapping[str, Any]:
    return _backend_mapping(_call(backend, ("snapshot",)), "snapshot()")


def _snapshot_after(backend: Any) -> Mapping[str, Any] | None:
    try:
        return _snapshot(backend)
    except Exception:
        return None


def _generation_from(value: Any, name: str = "generation") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullCycleBackendError("backend %s must be an integer" % name)
    if not 0 <= value <= MAX_GENERATION:
        raise FullCycleBackendError(
            "backend %s is outside [0, %d]" % (name, MAX_GENERATION))
    return int(value)


def _nested_mapping(snapshot: Mapping[str, Any],
                    name: str) -> Mapping[str, Any]:
    value = snapshot.get(name)
    return value if isinstance(value, Mapping) else {}


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
    for key in ("digest", "state_digest", "basis_digest", "field_digest"):
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
        raise FullCycleBackendError("backend %s must be an object" % name)
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
        raise FullCycleBackendError("backend events must be a sequence")
    if len(value) > MAX_EVENTS:
        raise FullCycleBackendError(
            "backend events exceed %d entries" % MAX_EVENTS)
    events: list[str] = []
    for item in value:
        if not isinstance(item, str) or EVENT_RE.fullmatch(item) is None:
            raise FullCycleBackendError(
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
        raise FullCycleInvariantError(
            "scratch write must be reported before the main write")
    if not accepted and (main_positions or
                         (main is not None and main.get("written"))):
        raise FullCycleInvariantError(
            "denied write must not report a main write")
    if main is not None and main.get("written") and scratch is not None \
            and not scratch.get("written"):
        raise FullCycleInvariantError(
            "main write requires a completed scratch write")
    if scratch is not None and main is not None:
        scratch_generation = scratch.get("generation")
        main_generation = main.get("generation")
        if scratch_generation is not None and main_generation is not None \
                and main_generation < scratch_generation:
            raise FullCycleInvariantError(
                "main generation must not precede scratch generation")
    reported = value.get("scratch_before_main")
    if reported is not None:
        if not isinstance(reported, bool):
            raise FullCycleBackendError(
                "backend scratch_before_main must be a boolean")
        if not reported and scratch is not None and main is not None \
                and scratch.get("written") and main.get("written"):
            raise FullCycleInvariantError(
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


def _assert_no_mutation(before: Mapping[str, Any],
                        after: Mapping[str, Any], operation: str) -> None:
    before_generation = _generation_from_snapshot(before)
    after_generation = _generation_from_snapshot(after)
    if before_generation is not None and after_generation is not None \
            and before_generation != after_generation:
        raise FullCycleInvariantError(
            "failed/denied %s changed main generation" % operation)
    before_digest = _digest_from_snapshot(before)
    after_digest = _digest_from_snapshot(after)
    if before_digest is not None and after_digest is not None \
            and before_digest != after_digest:
        raise FullCycleInvariantError(
            "failed/denied %s changed the main digest" % operation)


def _read_only_guard(before: Mapping[str, Any],
                     after: Mapping[str, Any], operation: str) -> None:
    _assert_no_mutation(before, after, operation)
    # A read-only operation must not advance the generation either.
    before_generation = _generation_from_snapshot(before)
    after_generation = _generation_from_snapshot(after)
    if before_generation is not None and after_generation is not None \
            and before_generation != after_generation:
        raise FullCycleInvariantError("%s must be read-only" % operation)


def _check_generation_transition(previous: int | None, current: int | None,
                                 accepted: bool, operation: str) -> None:
    if previous is None or current is None:
        return
    if accepted:
        if current <= previous:
            raise FullCycleInvariantError(
                "accepted %s must advance main generation" % operation)
    elif current != previous:
        raise FullCycleInvariantError(
            "denied %s must not change main generation" % operation)


# ---------------------------------------------------------------------------
# snapshot projections
# ---------------------------------------------------------------------------
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
        value = _nested_mapping(snapshot, "full_basis").get("root")
    if value is None:
        value = _nested_mapping(snapshot, "basis").get("root")
    if value is None:
        return DEFAULT_ROOT
    root = _backend_int(value, "root", 0, nodes - 1)
    return int(root)


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
        value = snapshot.get(name)
        if value is not None:
            result[name] = _phase_record(value, name)
    return result


def _first_value(sources: Sequence[Mapping[str, Any]],
                 names: Sequence[str]) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return None


def _basis_meta(snapshot: Mapping[str, Any], *,
                root_hint: int | None = None) -> dict[str, Any]:
    """Project bounded full-basis metadata out of a backend snapshot."""
    sources: list[Mapping[str, Any]] = [snapshot]
    for name in ("full_basis", "basis", "cycle_memory", "cycle_basis",
                 "field"):
        value = snapshot.get(name)
        if isinstance(value, Mapping):
            sources.append(value)
    root_raw = _first_value(sources, ("root",))
    try:
        root = DEFAULT_ROOT if root_hint is None else _input_root(root_hint)
        if root_raw is not None and root_hint is None:
            root = _input_root(root_raw)
    except FullCycleValidationError as exc:
        raise FullCycleBackendError(str(exc)) from exc
    count_raw = _first_value(
        sources, ("basis_count", "cycle_count", "full_count", "index_count",
                  "word_count", "coefficient_count", "count"))
    count: int | None
    if count_raw is None:
        count = None
    else:
        count = _backend_int(count_raw, "basis_count", 0, FULL_BASIS_CYCLES)
    digest = _first_value(
        sources, ("basis_digest", "full_basis_digest", "cycle_digest"))
    if digest is None:
        digest = _first_value(sources, ("digest", "state_digest"))
    digest = _backend_digest(digest, "basis_digest")
    generation = _first_value(
        sources, ("basis_generation", "cycle_generation", "generation"))
    generation = _generation_from(generation, "basis_generation") \
        if generation is not None else None
    features_raw = _first_value(
        sources, ("features", "feature_map", "cycles", "basis", "words",
                  "coefficients", "entries", "full_basis"))
    if isinstance(features_raw, Mapping):
        # A nested metadata object is not a feature map.  Look one level in
        # for the actual entries before deciding to ignore it.
        inner = _first_value(
            [features_raw], ("words", "basis", "cycles", "coefficients",
                             "entries", "features", "feature_map"))
        if inner is not None:
            features_raw = inner
        elif any(key in features_raw for key in (
                "count", "basis_count", "word_count", "digest",
                "basis_digest", "generation", "root", "complete")):
            features_raw = None
    entries: list[dict[str, int]] = []
    entries_error: str | None = None
    if features_raw is not None:
        try:
            entries = _feature_entries(
                features_raw, root=root, limit=MAX_FEATURES)
        except FullCycleError as exc:
            entries_error = str(exc)
    feature_count_raw = _first_value(
        sources, ("feature_count", "feature_entries", "features_count"))
    if feature_count_raw is None:
        feature_count = len(entries) if features_raw is not None else 0
    else:
        feature_count = _backend_int(
            feature_count_raw, "feature_count", 0, MAX_FEATURES)
        feature_count = int(feature_count or 0)
    if count is None:
        count = len(entries) if features_raw is not None else FULL_BASIS_CYCLES
    if digest is None and entries and len(entries) == FULL_BASIS_CYCLES:
        digest = _entries_digest(entries, root)
    feature_digest = _entries_digest(entries, root) if entries else None
    return {
        "root": root,
        "count": int(count),
        "expected_count": FULL_BASIS_CYCLES,
        "complete": int(count) == FULL_BASIS_CYCLES,
        "digest": digest,
        "generation": generation,
        "feature_count": int(feature_count),
        "feature_digest": feature_digest,
        "features": entries,
        "entries_error": entries_error,
    }


def _words_digest(entries: Sequence[Mapping[str, int]],
                  root: int = DEFAULT_ROOT) -> str:
    material = json.dumps(
        {"root": int(root),
         "words": [[int(item["index"]), int(item["word"])]
                   for item in entries]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _bounded_words(basis: Mapping[str, Any], limit: int
                   ) -> tuple[list[dict[str, int]], int, bool]:
    entries = list(basis.get("features", ()))
    count = int(basis.get("count", len(entries)))
    return entries[:limit], count, count > limit or len(entries) > limit


def _read_word(backend: Any, first: int, second: int) -> tuple[int, Any]:
    value = _call(backend, ("read_word", "get_word", "read_edge", "word"),
                  first, second)
    metadata = value if isinstance(value, Mapping) else None
    if isinstance(value, Mapping):
        value = value.get("word", value.get("value"))
    word = _backend_int(value, "read_word")
    if word is None:
        raise FullCycleBackendError("backend read_word() returned no word")
    return int(word), metadata


def _exact_cycle(backend: Any, first: int, second: int,
                 root: int) -> tuple[int, dict[str, int]]:
    root_i, _ = _read_word(backend, root, first)
    first_second, _ = _read_word(backend, first, second)
    root_second, _ = _read_word(backend, root, second)
    raw = {"root_i": root_i, "i_j": first_second, "root_j": root_second}
    return root_i + first_second - root_second, raw


def _normalize_prediction(item: Any, position: int) -> dict[str, Any] | None:
    if isinstance(item, int) and not isinstance(item, bool):
        pair = basis_pair(position, DEFAULT_ROOT)
        word = _input_word(item, "prediction word")
        return {"index": position, "i": pair[0], "j": pair[1], "word": word}
    if not isinstance(item, Mapping):
        raise FullCycleBackendError("predictions must be objects or integers")
    hit: dict[str, Any] = {}
    index = item.get("index")
    if index is not None:
        value = _input_int(index, "prediction index", 0,
                           FULL_BASIS_CYCLES - 1)
        hit["index"] = value
        pair = basis_pair(value, DEFAULT_ROOT)
        hit.setdefault("i", pair[0])
        hit.setdefault("j", pair[1])
    for key in ("i", "j", "root"):
        if item.get(key) is not None:
            hit[key] = _input_node(item[key], "prediction %s" % key)
    for key in ("word", "value", "cycle"):
        if item.get(key) is not None:
            hit["word"] = _input_word(item[key], "prediction word")
            break
    if item.get("score") is not None:
        hit["score"] = _backend_float(item["score"], "prediction score")
    for key in ("text", "kind", "leaf_id", "feature_id"):
        if item.get(key) is not None:
            text = _bounded_string(
                item[key], "prediction %s" % key, MAX_HIT_TEXT_BYTES,
                FullCycleBackendError, allow_empty=True)
            if text:
                hit[key] = text
    return hit or None


def _normalize_predictions(value: Any, limit: int
                           ) -> tuple[list[dict[str, Any]], int, bool,
                                      dict[str, Any]]:
    base: dict[str, Any] = {}
    raw: Any
    if isinstance(value, Mapping):
        for key in ("mode", "text"):
            if value.get(key) is not None:
                base[key] = _bounded_string(
                    value[key], "prediction %s" % key, MAX_TEXT_BYTES,
                    FullCycleBackendError, allow_empty=True)
        raw = value.get("predictions")
        if raw is None:
            raw = value.get("hits")
        if raw is None:
            raw = value.get("cycles")
        if raw is None:
            raw = value.get("results")
        if raw is None and (value.get("word") is not None
                            or value.get("value") is not None):
            raw = [value]
        if raw is None:
            raw = []
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        raw = value
    elif isinstance(value, int) and not isinstance(value, bool):
        raw = [value]
    else:
        raise FullCycleBackendError(
            "backend predict() must return an object, sequence or integer")
    if isinstance(raw, (str, bytes, bytearray)) or \
            not isinstance(raw, Sequence):
        raise FullCycleBackendError(
            "backend predict() predictions must be a sequence")
    if len(raw) > MAX_PREDICTIONS:
        raise FullCycleBackendError(
            "backend predictions exceed %d entries" % MAX_PREDICTIONS)
    hits: list[dict[str, Any]] = []
    for position, item in enumerate(list(raw)[:limit]):
        hit = _normalize_prediction(item, position)
        if hit is not None:
            hits.append(hit)
    return hits, len(raw), len(raw) > limit, base


def _normalize_metrics(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    base = _backend_mapping(value, "metrics")
    if len(base) > MAX_METRICS:
        raise FullCycleBackendError(
            "backend metrics exceed %d entries" % MAX_METRICS)
    result: dict[str, Any] = {}
    for key, item in base.items():
        if not isinstance(key, str) or METRIC_KEY_RE.fullmatch(key) is None:
            raise FullCycleBackendError("backend metric names must be safe")
        if _compact_key(key) in _POINTER_KEYS:
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
                item, "metric %s" % key, MAX_METRIC_TEXT_BYTES
                if "MAX_METRIC_TEXT_BYTES" in globals() else 240,
                FullCycleBackendError, allow_empty=True)
        else:
            raise FullCycleBackendError(
                "backend metric %s must be scalar" % key)
    return result


def _normalize_hit(item: Any, *,
                   root: int = DEFAULT_ROOT) -> dict[str, Any] | None:
    if isinstance(item, int) and not isinstance(item, bool):
        return {"word": _input_word(item, "hit word")}
    if not isinstance(item, Mapping):
        raise FullCycleBackendError("query hits must be objects")
    hit: dict[str, Any] = {}
    index = item.get("index")
    if index is not None:
        value = _input_int(index, "hit index", 0, FULL_BASIS_CYCLES - 1)
        hit["index"] = value
        pair = basis_pair(value, root)
        hit.setdefault("i", pair[0])
        hit.setdefault("j", pair[1])
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
                FullCycleBackendError, allow_empty=True)
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
                FullCycleBackendError, allow_empty=False)
        if value.get("text") is not None:
            base["text"] = _bounded_string(
                value["text"], "query text", MAX_TEXT_BYTES,
                FullCycleBackendError, allow_empty=True)
        if value.get("word") is not None or value.get("value") is not None:
            base["word"] = _input_word(
                value.get("word", value.get("value")), "query word")
        raw = value.get("hits")
        if raw is None:
            raw = value.get("results")
        if raw is None and "word" not in base:
            raw = []
        elif raw is None:
            raw = [value]
    elif isinstance(value, Sequence) and \
            not isinstance(value, (str, bytes, bytearray)):
        base = {}
        raw = value
    else:
        raise FullCycleBackendError(
            "backend query() must return an object, sequence or integer")
    if isinstance(raw, (str, bytes, bytearray)) or \
            not isinstance(raw, Sequence):
        raise FullCycleBackendError(
            "backend query() hits must be a sequence")
    if len(raw) > MAX_FEATURES:
        raise FullCycleBackendError(
            "backend query hits exceed %d entries" % MAX_FEATURES)
    hits = []
    for item in list(raw)[:limit]:
        hit = _normalize_hit(item)
        if hit is not None:
            hits.append(hit)
    base["hits"] = hits
    base["count"] = len(hits)
    return base


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
        "canonical_commit": False,
        "canonical_state_modified": False,
        "promotion_enabled": False,
    }


def _json_safe(value: dict[str, Any], operation: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False,
                             separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise FullCycleBackendError(
            "%s produced non-JSON-safe output" % operation) from exc
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise FullCycleBackendError(
            "%s output exceeds %d bytes" % (operation, MAX_OUTPUT_BYTES))
    return value


def _full_basis_public(basis: Mapping[str, Any],
                       *, include_features: bool = False) -> dict[str, Any]:
    result = {
        "root": basis["root"],
        "count": basis["count"],
        "expected_count": basis["expected_count"],
        "complete": basis["complete"],
        "digest": basis["digest"],
        "generation": basis["generation"],
        "feature_count": basis["feature_count"],
        "feature_digest": basis["feature_digest"],
        "bounded": True,
        "raw_words": True,
    }
    if include_features:
        result["features"] = list(basis["features"])
    return result


def _geometry(nodes: int) -> dict[str, int]:
    edges = nodes * (nodes - 1) // 2
    return {
        "nodes": nodes,
        "edges": edges,
        "cycle_dimension": edges - (nodes - 1),
        "triangles": nodes * (nodes - 1) * (nodes - 2) // 6,
        "max_node": nodes - 1,
        "full_basis_cycles": FULL_BASIS_CYCLES,
    }


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------
def status(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return a bounded read-only summary of the full 32,385-cycle memory."""
    args = _params(params, frozenset({"limit", "root"}))
    limit = _input_limit(args.get("limit", DEFAULT_STATUS_WORDS),
                         "limit", MAX_SNAPSHOT_WORDS)
    root_hint = None
    if "root" in args:
        root_hint = _input_root(args["root"])
    snapshot = _snapshot(backend)
    meta = _state_meta(snapshot)
    basis = _basis_meta(snapshot, root_hint=root_hint)
    words, word_count, truncated = _bounded_words(basis, limit)
    nodes = meta["nodes"]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "status",
        **meta,
        "geometry": _geometry(nodes),
        "full_basis": _full_basis_public(basis),
        "word_count": word_count,
        "words": words,
        "truncated": truncated,
        "raw_words": True,
        "bounded": True,
        "read_only": True,
        "writes_main": False,
    }
    if result["digest"] is None:
        result["digest"] = basis["digest"] or _words_digest(
            basis["features"], basis["root"])
    if basis["entries_error"]:
        result["full_basis"]["problem"] = basis["entries_error"]
    result.update(_policy_flags())
    return _json_safe(result, "status")


def snapshot(params: Any, *, backend: Any) -> dict[str, Any]:
    """Return a bounded raw-word snapshot plus full-basis metadata/digest."""
    args = _params(params, frozenset({"limit", "root"}))
    limit = _input_limit(args.get("limit", DEFAULT_SNAPSHOT_WORDS),
                         "limit", MAX_SNAPSHOT_WORDS)
    root_hint = None
    if "root" in args:
        root_hint = _input_root(args["root"])
    state = _snapshot(backend)
    meta = _state_meta(state)
    basis = _basis_meta(state, root_hint=root_hint)
    words, word_count, truncated = _bounded_words(basis, limit)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "snapshot",
        **meta,
        "geometry": _geometry(meta["nodes"]),
        "full_basis": _full_basis_public(basis),
        "word_count": word_count,
        "words": words,
        "truncated": truncated,
        "raw_words": True,
        "bounded": True,
        "read_only": True,
        "writes_main": False,
    }
    if result["digest"] is None:
        result["digest"] = basis["digest"] or _words_digest(
            basis["features"], basis["root"])
    result.update(_policy_flags())
    return _json_safe(result, "snapshot")


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


def _resolve_request_pair(args: Mapping[str, Any], *,
                          require_pair: bool = True
                          ) -> tuple[int, int, int, int | None]:
    root = _input_root(args["root"]) if "root" in args else DEFAULT_ROOT
    index: int | None = None
    if "index" in args:
        index = _input_int(args["index"], "index", 0, FULL_BASIS_CYCLES - 1)
        first, second = basis_pair(index, root)
        if "i" in args or "j" in args:
            if "i" not in args or "j" not in args:
                raise FullCycleValidationError(
                    "index with i/j requires both i and j")
            requested_i = _input_node(args["i"], "i")
            requested_j = _input_node(args["j"], "j")
            if (requested_i, requested_j) not in (
                    (first, second), (second, first)):
                raise FullCycleValidationError(
                    "index does not match the requested i/j pair")
    else:
        if require_pair:
            first = _input_node(_required(args, "i"), "i")
            second = _input_node(_required(args, "j"), "j")
        else:
            first = _input_node(args["i"], "i") \
                if "i" in args else DEFAULT_ROOT
            second = _input_node(args["j"], "j") \
                if "j" in args else DEFAULT_ROOT
        if first == second:
            raise FullCycleValidationError("i and j must be distinct")
        if "root" in args and root in (first, second):
            raise FullCycleValidationError(
                "root must be distinct from i and j")
        if "root" not in args and "index" in args:
            pass
    return first, second, root, index


def write(params: Any, *, backend: Any) -> dict[str, Any]:
    """Write one raw cycle word after explicit approval and CAS check."""
    args = _params(params, frozenset({
        "approved", "i", "j", "index", "root", "word", "raw_word",
        "expected_generation", "source"}))
    _require_approved(args)
    raw_word = args.get("word", args.get("raw_word"))
    if raw_word is None:
        raise FullCycleValidationError("word is required")
    word = _input_word(raw_word)
    first, second, root, index = _resolve_request_pair(args)
    expected = None if "expected_generation" not in args else \
        _input_generation(args["expected_generation"])
    source = _input_source(args.get("source"))
    if index is None:
        index = basis_index(first, second, root)

    before = _snapshot(backend)
    previous_generation = _generation_from_snapshot(before)
    fields = {
        "i": first, "j": second, "root": root, "index": index,
        "word": word, "word_hex": _word_hex(word), "source": source,
    }
    if expected is not None:
        if previous_generation is None:
            raise FullCycleValidationError(
                "backend must expose generation for expected_generation")
        if expected != previous_generation:
            return _write_denied(
                "write", fields, ["stale_generation"],
                previous_generation, expected)

    basis_method = None
    word_method = _find_method(
        backend, ("write_word", "set_word", "set_edge", "write", "set"))
    if index is not None:
        basis_method = _find_method(
            backend, ("write_basis", "set_basis", "write_cycle",
                      "set_cycle"))
    try:
        if basis_method is not None:
            value = _backend_mapping(
                _call(backend, (basis_method[0],), index, word,
                      expected_generation=(MISSING if expected is None
                                           else expected)),
                "%s()" % basis_method[0])
        elif word_method is not None:
            value = _backend_mapping(
                _call(backend, (word_method[0],), first, second, word,
                      expected_generation=(MISSING if expected is None
                                           else expected)),
                "%s()" % word_method[0])
        else:
            raise FullCycleBackendError(
                "backend must provide write_word() or write_basis()")
    except FullCycleError as exc:
        after = _snapshot_after(backend)
        if after is not None:
            _assert_no_mutation(before, after, "write")
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
            _assert_no_mutation(before, after, "write")
        elif previous_generation is not None and after_generation is not None \
                and after_generation <= previous_generation:
            raise FullCycleInvariantError(
                "accepted write must advance main generation")
    _check_generation_transition(
        previous_generation, generation, accepted, "write")

    stored_word = value.get("word", value.get("value", word))
    if isinstance(stored_word, bool) or not isinstance(stored_word, int):
        raise FullCycleBackendError(
            "backend stored word must be an integer")
    if not WORD_MIN <= stored_word <= WORD_MAX:
        raise FullCycleBackendError(
            "backend stored word is outside the 64-bit range")

    receipt = {
        "operation": "write",
        "accepted": accepted,
        "i": first,
        "j": second,
        "root": root,
        "index": index,
        "word": word,
        "stored_word": int(stored_word),
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        **phases,
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "write",
        "accepted": accepted,
        "denied": not accepted,
        "i": first,
        "j": second,
        "root": root,
        "index": index,
        "word": word,
        "word_hex": _word_hex(word),
        "stored_word": int(stored_word),
        "previous_generation": previous_generation,
        "generation": generation,
        "expected_generation": expected,
        "reasons": reasons,
        "scratch": phases["scratch"],
        "main": phases["main"],
        "events": phases["events"],
        "scratch_before_main": phases["scratch_before_main"],
        "receipt": receipt,
        "raw_words": True,
        "read_only": not accepted,
        "writes_main": accepted,
    }
    result.update(_policy_flags())
    return _json_safe(result, "write")


def predict(params: Any, *, backend: Any) -> dict[str, Any]:
    """Predict bounded cycle coefficients without changing the field."""
    args = _params(params, frozenset({
        "text", "features", "feature_map", "limit", "i", "j", "index",
        "root"}))
    has_text = "text" in args
    has_edge = any(key in args for key in ("i", "j", "index"))
    if has_text and has_edge:
        raise FullCycleValidationError(
            "predict accepts either text or i/j/index, not both")
    if "limit" in args and not has_text:
        raise FullCycleValidationError("limit requires text")
    feature_source = args.get("features", args.get("feature_map"))
    features = None
    if feature_source is not None:
        features = _feature_entries(
            feature_source,
            root=args.get("root", DEFAULT_ROOT),
            limit=MAX_FEATURES)
    limit = _input_limit(
        args.get("limit", DEFAULT_PREDICTIONS), "limit", MAX_PREDICTIONS)

    before = _snapshot(backend)
    if has_edge:
        first, second, root, index = _resolve_request_pair(args)
        word, metadata = _read_word(backend, first, second)
        payload: dict[str, Any] = {
            "mode": "word",
            "i": first,
            "j": second,
            "root": root,
            "index": index if index is not None else None,
            "word": word,
            "word_hex": _word_hex(word),
            "value": word,
        }
        if metadata is not None:
            digest = _backend_digest(metadata.get("digest"), "word digest")
            if digest is not None:
                payload["word_digest"] = digest
        cycle_value, raw = _exact_cycle(backend, first, second, root)
        payload["cycle"] = cycle_value
        payload["raw_edges"] = raw
        predictions = [{
            "index": index if index is not None else basis_index(
                first, second, root),
            "i": first, "j": second, "root": root, "word": cycle_value,
        }]
        total = len(predictions)
        truncated = False
        base = {"mode": "word"}
    else:
        text = _input_text(_required(args, "text"))
        function = _find_method(backend, ("predict",))
        if function is None:
            raise FullCycleBackendError(
                "backend must provide predict() for text predictions")
        kwargs: dict[str, Any] = {}
        if features is not None:
            kwargs["features"] = features
        kwargs["limit"] = limit
        value = _call(backend, ("predict",), text, **kwargs)
        predictions, total, truncated, base = _normalize_predictions(
            value, limit)
        payload = {"mode": "text", "text": text, "top_k": limit}
        if isinstance(value, Mapping):
            raw_features = value.get("features", value.get("feature_map"))
            if raw_features is not None:
                entries = _feature_entries(
                    raw_features, root=args.get("root", DEFAULT_ROOT),
                    limit=MAX_FEATURES)
                features = entries
            if value.get("feature_count") is not None:
                payload["feature_count"] = _backend_int(
                    value["feature_count"], "feature_count", 0,
                    MAX_FEATURES)

    after = _snapshot(backend)
    _read_only_guard(before, after, "predict")
    basis = _basis_meta(after if after is not None else before,
                        root_hint=args.get("root", DEFAULT_ROOT)
                        if "root" in args else None)
    feature_entries = features or []
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "predict",
        **payload,
        "predictions": predictions,
        "prediction_count": total,
        "truncated": truncated,
        "feature_count": int(payload.get("feature_count",
                                         len(feature_entries))),
        "features": feature_entries[:limit],
        "feature_digest": _entries_digest(feature_entries, basis["root"])
        if feature_entries else None,
        "full_basis": _full_basis_public(basis),
        "generation": _generation_from_snapshot(
            after if after is not None else before),
        "digest": _digest_from_snapshot(
            after if after is not None else before)
        or basis["digest"],
        "raw_words": True,
        "bounded": True,
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "predict")


def _call_train(backend: Any, text: str, batch: int, steps: int,
                expected: int | None, features: Any) -> Any:
    found = _find_method(backend, ("train",))
    names = _positional_names(found[1]) if found is not None else []
    arguments: tuple[Any, ...]
    if names[:3] == ["batch", "steps", "text"]:
        arguments = (batch, steps, text)
    else:
        arguments = (text, batch, steps)
    kwargs: dict[str, Any] = {}
    if features is not None:
        kwargs["features"] = features
    return _call(backend, ("train",), *arguments,
                 expected_generation=(MISSING if expected is None
                                      else expected),
                 **kwargs)


def train(params: Any, *, backend: Any) -> dict[str, Any]:
    """Run a bounded, explicitly approved full-basis training step."""
    args = _params(params, frozenset({
        "approved", "text", "batch", "steps", "expected_generation",
        "features", "feature_map"}))
    _require_approved(args)
    text = _input_text(_required(args, "text"))
    batch = _input_int(args.get("batch", DEFAULT_BATCH),
                       "batch", 1, MAX_BATCH)
    steps = _input_int(args.get("steps", DEFAULT_STEPS),
                       "steps", 1, MAX_STEPS)
    expected = None if "expected_generation" not in args else \
        _input_generation(args["expected_generation"])
    feature_source = args.get("features", args.get("feature_map"))
    features = None
    if feature_source is not None:
        features = _feature_entries(
            feature_source, root=args.get("root", DEFAULT_ROOT),
            limit=MAX_FEATURES)
    # ``root`` is intentionally not an approved train property; the map keys
    # already define their coordinates under the default root.

    before = _snapshot(backend)
    previous_generation = _generation_from_snapshot(before)
    fields = {"text": text, "batch": batch, "steps": steps,
              "feature_count": len(features or ())}
    if expected is not None:
        if previous_generation is None:
            raise FullCycleValidationError(
                "backend must expose generation for expected_generation")
        if expected != previous_generation:
            return _write_denied(
                "train", fields, ["stale_generation"],
                previous_generation, expected)

    try:
        value = _backend_mapping(
            _call_train(backend, text, batch, steps, expected, features),
            "train()")
    except FullCycleError as exc:
        after = _snapshot_after(backend)
        if after is not None:
            _assert_no_mutation(before, after, "train")
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
            _assert_no_mutation(before, after, "train")
        elif previous_generation is not None and after_generation is not None \
                and after_generation <= previous_generation:
            raise FullCycleInvariantError(
                "accepted train must advance main generation")
    _check_generation_transition(
        previous_generation, generation, accepted, "train")

    receipt = {
        "operation": "train",
        "accepted": accepted,
        "text": text,
        "batch": batch,
        "steps": steps,
        "feature_count": len(features or ()),
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
        "feature_count": len(features or ()),
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
        "raw_words": True,
        "read_only": not accepted,
        "writes_main": accepted,
    }
    result.update(_policy_flags())
    return _json_safe(result, "train")


def audit(params: Any, *, backend: Any) -> dict[str, Any]:
    """Audit full-basis count, digest and words without mutation."""
    args = _params(params, frozenset({"full", "limit"}))
    full = _backend_bool(args.get("full"), "full", True)
    limit = _input_limit(args.get("limit", DEFAULT_STATUS_WORDS),
                         "limit", MAX_SNAPSHOT_WORDS)
    before = _snapshot(backend)
    problems: list[str] = []
    codes: list[str] = []
    reported: Mapping[str, Any] | None = None
    method = _find_method(backend, ("audit",))
    if method is not None:
        value = _call(backend, ("audit",))
        reported = _backend_mapping(value, "audit()")
        raw_problems = reported.get("problems")
        if raw_problems is not None:
            if isinstance(raw_problems, str):
                raw_problems = [raw_problems]
            if isinstance(raw_problems, (bytes, bytearray)) or \
                    not isinstance(raw_problems, Sequence):
                raise FullCycleBackendError(
                    "backend audit problems must be a sequence")
            for item in list(raw_problems)[:MAX_REASONS]:
                problems.append(_bounded_string(
                    item, "audit problem", MAX_REASON_BYTES,
                    FullCycleBackendError, allow_empty=True))
        raw_codes = reported.get("problem_codes")
        if isinstance(raw_codes, Sequence) and \
                not isinstance(raw_codes, (str, bytes, bytearray)):
            for item in list(raw_codes)[:MAX_REASONS]:
                if isinstance(item, str) and EVENT_RE.fullmatch(item):
                    codes.append(item)
        if reported.get("ok") is False and not problems:
            problems.append("backend audit reported a failure")

    basis = _basis_meta(before)
    if basis["entries_error"]:
        problems.append(basis["entries_error"])
        codes.append("basis_entries")
    count = int(basis["count"])
    if count != FULL_BASIS_CYCLES:
        problems.append(
            "full basis is truncated: %d of %d cycles"
            % (count, FULL_BASIS_CYCLES))
        codes.append("basis_truncated")
    if not basis["complete"] and count == FULL_BASIS_CYCLES:
        problems.append("full basis metadata is incomplete")
        codes.append("basis_incomplete")
    if basis["digest"] is None:
        problems.append("full basis digest is missing")
        codes.append("basis_digest")
    if basis["feature_count"] > MAX_FEATURES:
        problems.append("feature map exceeds %d entries" % MAX_FEATURES)
        codes.append("feature_overflow")
    words, _word_count, _truncated = _bounded_words(basis, limit)
    for item in words:
        try:
            _input_word(item["word"], "audit word")
        except FullCycleValidationError:
            problems.append("audit word is outside the word range")
            codes.append("word_range")
            break

    after = _snapshot(backend)
    _read_only_guard(before, after, "audit")
    ok = not problems
    result: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "operation": "audit",
        "ok": ok,
        "accepted": ok,
        "checked": count,
        "basis_count": count,
        "expected_count": FULL_BASIS_CYCLES,
        "complete": count == FULL_BASIS_CYCLES,
        "full_basis": _full_basis_public(basis),
        "feature_count": basis["feature_count"],
        "features": words,
        "truncated": _truncated,
        "problems": problems[:MAX_REASONS],
        "problem_codes": codes[:MAX_REASONS],
        "generation": _generation_from_snapshot(
            after if after is not None else before),
        "digest": _digest_from_snapshot(
            after if after is not None else before)
        or basis["digest"],
        "read_only": True,
        "writes_main": False,
        "no_floats": True,
        "no_raw_pointers": True,
    }
    if reported is not None:
        for key in ("generation", "digest"):
            if reported.get(key) is not None:
                try:
                    result[key] = _generation_from(
                        reported[key], key) if key == "generation" else \
                        _backend_digest(reported[key], key)
                except FullCycleError:
                    pass
    result.update(_policy_flags())
    return _json_safe(result, "audit")


def query(params: Any, *, backend: Any) -> dict[str, Any]:
    """Read one cycle word or run a bounded text query without mutation."""
    args = _params(params, frozenset({
        "i", "j", "index", "root", "text", "limit"}))
    has_text = "text" in args
    has_edge = any(key in args for key in ("i", "j", "index"))
    if has_text and (has_edge or "root" in args):
        raise FullCycleValidationError(
            "query accepts either text or i/j/index/root, not both")
    if "limit" in args and not has_text:
        raise FullCycleValidationError("limit requires text")

    before = _snapshot(backend)
    if has_edge:
        if "index" not in args and ("i" not in args or "j" not in args):
            raise FullCycleValidationError(
                "query requires both i and j or an index")
        first, second, root, index = _resolve_request_pair(args)
        word, metadata = _read_word(backend, first, second)
        payload: dict[str, Any] = {
            "mode": "word",
            "i": first,
            "j": second,
            "root": root,
            "index": index if index is not None else basis_index(
                first, second, root),
            "word": word,
            "word_hex": _word_hex(word),
            "value": word,
        }
        if metadata is not None:
            digest = _backend_digest(metadata.get("digest"), "word digest")
            if digest is not None:
                payload["word_digest"] = digest
        if "root" in args:
            cycle_value, raw = _exact_cycle(backend, first, second, root)
            payload["cycle"] = cycle_value
            payload["raw_edges"] = raw
    elif has_text:
        text = _input_text(args["text"])
        limit = _input_limit(args.get("limit", DEFAULT_QUERY_HITS),
                             "limit", MAX_QUERY_HITS)
        method = _find_method(backend, ("query", "search", "resonance"))
        if method is None:
            raise FullCycleBackendError(
                "backend must provide query() for text queries")
        function = method[1]
        if _accepts_keyword(function, "limit"):
            value = _call(backend, (method[0],), text, limit=limit)
        else:
            value = _call(backend, (method[0],), text)
        payload = _normalize_query(value, limit)
        payload.setdefault("mode", "text")
        payload["text"] = text
        payload["top_k"] = limit
    else:
        raise FullCycleValidationError(
            "query requires text or an edge/index")

    after = _snapshot(backend)
    _read_only_guard(before, after, "query")
    basis = _basis_meta(after if after is not None else before)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "query",
        "generation": _generation_from_snapshot(
            after if after is not None else before),
        "digest": _digest_from_snapshot(
            after if after is not None else before) or basis["digest"],
        **payload,
        "full_basis": _full_basis_public(basis),
        "raw_words": True,
        "read_only": True,
        "writes_main": False,
    }
    result.update(_policy_flags())
    return _json_safe(result, "query")


HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "status": status,
    "write": write,
    "predict": predict,
    "train": train,
    "snapshot": snapshot,
    "audit": audit,
    "query": query,
}


def build_handlers(backend: Any) -> dict[str, Callable[[Any],
                                                       dict[str, Any]]]:
    """Bind every handler to one backend for plugin registration."""
    return {name: partial(handler, backend=backend)
            for name, handler in HANDLERS.items()}


def handle(action: Any, params: Any = None, *, backend: Any = None
           ) -> dict[str, Any]:
    """Dispatch one named action through the bounded handler table."""
    if not isinstance(action, str) or action not in HANDLERS:
        raise FullCycleValidationError(
            "action must be one of: %s" % ", ".join(sorted(HANDLERS)))
    return HANDLERS[action](params, backend=backend)


# Aliases used by callers that mirror the lower-level cycle tools.
set_word = write
set_cycle = write
predict_cycles = predict
train_full_basis = train


# ---------------------------------------------------------------------------
# optional deterministic in-memory backend
# ---------------------------------------------------------------------------
class FullCycleMemory:
    """A bounded in-memory backend holding all 32,385 cycle coefficients.

    This is a reference/test backend, not a language model.  It implements the
    handler contract with integer words, explicit generations, stale-CAS
    denial and read-only snapshot copies.  ``FULL_BASIS_COEFFICIENTS`` can be
    supplied through ``coefficients``; otherwise the memory starts at zero.
    """

    def __init__(self, coefficients: Any = None,
                 root: Any = DEFAULT_ROOT) -> None:
        self.root = _input_root(root)
        if coefficients is None:
            self._words = [0] * FULL_BASIS_CYCLES
        else:
            words, _metadata = _normalize_full_basis(
                coefficients, root=self.root)
            self._words = words
        self._generation = 0
        self._digest = _basis_digest(self._words, self.root)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def coefficients(self) -> list[int]:
        return list(self._words)

    @property
    def basis_count(self) -> int:
        return FULL_BASIS_CYCLES

    def _word(self, first: int, second: int) -> int:
        first = _input_node(first, "i")
        second = _input_node(second, "j")
        if first == second:
            raise FullCycleValidationError(
                "a cycle edge requires two distinct heads")
        if first == self.root or second == self.root:
            # The star gauge fixes the root edges at zero; they are not
            # independent full-basis coefficients.
            return 0
        return self._words[basis_index(first, second, self.root)]

    def _advance(self) -> None:
        self._generation += 1
        self._digest = _basis_digest(self._words, self.root)

    def snapshot(self) -> dict[str, Any]:
        entries = []
        for index, pair in enumerate(_iter_basis_pairs(self.root)):
            word = self._words[index]
            if word:
                entries.append({"index": index, "i": pair[0], "j": pair[1],
                                "word": word})
        return {
            "opened": True,
            "nodes": NODE_COUNT,
            "root": self.root,
            "generation": self._generation,
            "digest": self._digest,
            "basis_count": FULL_BASIS_CYCLES,
            "full_basis": {
                "root": self.root,
                "basis_count": FULL_BASIS_CYCLES,
                "basis_digest": self._digest,
                "generation": self._generation,
            },
            "feature_count": len(entries),
            "features": entries,
            "words": entries,
        }

    def read_word(self, first: int, second: int) -> dict[str, Any]:
        word = self._word(first, second)
        return {"word": word, "generation": self._generation,
                "digest": self._digest}

    def write_word(self, first: int, second: int, word: Any,
                   expected_generation: Any = None) -> dict[str, Any]:
        word_value = _input_word(word)
        first = _input_node(first, "i")
        second = _input_node(second, "j")
        if first == second:
            raise FullCycleValidationError(
                "a cycle edge requires two distinct heads")
        if expected_generation is not None:
            expected = _input_generation(expected_generation)
            if expected != self._generation:
                return {"accepted": False, "reasons": ["stale_generation"],
                        "generation": self._generation}
        if first == self.root or second == self.root:
            return {
                "accepted": False,
                "reasons": ["gauge_edge_not_in_basis"],
                "generation": self._generation,
            }
        index = basis_index(first, second, self.root)
        previous = self._generation
        self._words[index] = word_value
        self._advance()
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "word": word_value,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self._digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
        }

    def predict(self, text: Any, *, features: Any = None,
                limit: Any = DEFAULT_PREDICTIONS) -> dict[str, Any]:
        text_value = _input_text(text)
        cap = _input_limit(limit, "limit", MAX_PREDICTIONS)
        if features is not None:
            entries = _feature_entries(features, root=self.root,
                                       limit=MAX_FEATURES)
        else:
            seed = int(hashlib.sha256(text_value.encode("utf-8"))
                       .hexdigest()[:16], 16)
            entries = []
            seen: set[int] = set()
            for position in range(min(cap, FULL_BASIS_CYCLES)):
                index = (seed + position * 7919) % FULL_BASIS_CYCLES
                if index in seen:
                    continue
                seen.add(index)
                pair = basis_pair(index, self.root)
                entries.append({"index": index, "i": pair[0], "j": pair[1],
                                "word": self._words[index]})
        return {
            "mode": "text",
            "text": text_value,
            "predictions": entries[:cap],
            "features": entries,
            "feature_count": len(entries),
            "generation": self._generation,
            "digest": self._digest,
        }

    def train(self, text: Any, batch: Any = 1, steps: Any = 1,
              *, features: Any = None, expected_generation: Any = None
              ) -> dict[str, Any]:
        text_value = _input_text(text)
        batch_value = _input_int(batch, "batch", 1, MAX_BATCH)
        step_value = _input_int(steps, "steps", 1, MAX_STEPS)
        if expected_generation is not None:
            expected = _input_generation(expected_generation)
            if expected != self._generation:
                return {"accepted": False, "reasons": ["stale_generation"],
                        "generation": self._generation}
        if features is not None:
            entries = _feature_entries(features, root=self.root,
                                       limit=MAX_FEATURES)
        else:
            seed = int(hashlib.sha256(text_value.encode("utf-8"))
                       .hexdigest()[:16], 16)
            count = min(FULL_BASIS_CYCLES, batch_value * step_value)
            entries = []
            for position in range(count):
                index = (seed + position * 104729) % FULL_BASIS_CYCLES
                pair = basis_pair(index, self.root)
                entries.append({"index": index, "i": pair[0], "j": pair[1],
                                "word": 0})
        previous = self._generation
        for item in entries[:batch_value * step_value]:
            index = int(item["index"])
            self._words[index] = int(self._words[index]) + step_value
        self._advance()
        return {
            "accepted": True,
            "generation": self._generation,
            "previous_generation": previous,
            "scratch": {"written": True, "generation": previous},
            "main": {"written": True, "generation": self._generation,
                     "digest": self._digest},
            "events": ["scratch", "main"],
            "scratch_before_main": True,
            "metrics": {"steps": step_value, "batch": batch_value,
                        "updates": len(entries[:batch_value * step_value])},
        }

    def query(self, text: Any, limit: Any = DEFAULT_QUERY_HITS
              ) -> dict[str, Any]:
        text_value = _input_text(text)
        cap = _input_limit(limit, "limit", MAX_QUERY_HITS)
        seed = int(hashlib.sha256(text_value.encode("utf-8"))
                   .hexdigest()[:16], 16)
        hits = []
        for position in range(cap):
            index = (seed + position * 4099) % FULL_BASIS_CYCLES
            pair = basis_pair(index, self.root)
            hits.append({
                "index": index, "i": pair[0], "j": pair[1],
                "word": self._words[index],
                "score": round(1.0 / (position + 1), 8),
                "text": text_value[:64], "kind": "full_cycle",
            })
        return {"mode": "text", "text": text_value, "hits": hits,
                "generation": self._generation, "digest": self._digest}

    def audit(self) -> dict[str, Any]:
        problems = []
        if len(self._words) != FULL_BASIS_CYCLES:
            problems.append("coefficient count is not 32,385")
        if _basis_digest(self._words, self.root) != self._digest:
            problems.append("basis digest mismatch")
        return {
            "ok": not problems,
            "checked": len(self._words),
            "basis_count": len(self._words),
            "basis_digest": self._digest,
            "problems": problems,
            "generation": self._generation,
            "digest": self._digest,
        }


FullBasisMemory = FullCycleMemory
FullCycleMemoryPersistence = FullCyclePersistence
FullBasisPersistence = FullCyclePersistence

# Additional constant spellings used by callers that describe the same bound
# in different words.
MAX_CYCLES = FULL_BASIS_CYCLES
MAX_CYCLE_COUNT = FULL_BASIS_CYCLES
MAX_BASIS = FULL_BASIS_CYCLES
MAX_BASIS_COUNT = FULL_BASIS_CYCLES


def save_full_basis(universe: Any, basis: Any, **kwargs: Any
                    ) -> FullCycleReceipt:
    """Convenience wrapper for ``FullCyclePersistence.save_full_basis``."""
    return FullCyclePersistence(universe).save_full_basis(basis, **kwargs)


def load_full_basis(universe: Any, version: Any = None, **kwargs: Any
                    ) -> FullBasisRecord | None:
    """Convenience wrapper for ``FullCyclePersistence.load_full_basis``."""
    return FullCyclePersistence(universe).load_full_basis(
        version, **kwargs)


def full_basis_audit(universe: Any, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for ``FullCyclePersistence.audit``."""
    return FullCyclePersistence(universe).audit(**kwargs)


__all__ = (
    "SCHEMA", "PERSISTENCE_SCHEMA", "BASIS_SCHEMA", "MANIFEST_SCHEMA",
    "SHARD_SCHEMA", "ANCHOR_SCHEMA", "AUDIT_SCHEMA", "RECEIPT_SCHEMA",
    "HEAD_COUNT", "NODE_COUNT", "MAX_NODE", "PAIR_COUNT", "EDGE_COUNT",
    "CYCLE_DIMENSION", "FULL_BASIS_CYCLES", "FULL_CYCLE_COUNT",
    "ROOT_CYCLE_COUNT", "BASIS_COUNT", "TRIANGLE_COUNT", "DEFAULT_ROOT",
    "WORD_BITS", "WORD_MODULUS", "WORD_MASK", "WORD_MIN", "WORD_MAX",
    "MIN_WORD", "MAX_WORD", "U32_MASK", "MIN_CYCLE_WORD", "MAX_CYCLE_WORD",
    "MAX_FEATURES", "MAX_FEATURE_ENTRIES", "MAX_BASIS_ENTRIES",
    "MAX_SNAPSHOT_WORDS", "DEFAULT_SNAPSHOT_WORDS", "DEFAULT_STATUS_WORDS",
    "MAX_PREDICTIONS", "DEFAULT_PREDICTIONS", "MAX_QUERY_HITS",
    "DEFAULT_QUERY_HITS", "MAX_TEXT_BYTES", "MAX_BATCH", "MAX_STEPS",
    "MAX_TRAIN_STEPS", "MAX_GENERATION", "MAX_OUTPUT_BYTES", "SHARD_CYCLES",
    "MAX_SHARD_BYTES", "MAX_MANIFEST_BYTES", "MAX_VERSIONS",
    "FullCycleError", "FullCycleValidationError", "FullCycleApprovalError",
    "FullCycleBackendError", "FullCycleConflictError",
    "FullCycleInvariantError", "FullCyclePersistenceError",
    "FullCycleBoundsError", "FullCycleCorruptionError",
    "FullCyclePathError", "FullCycleSecretError", "FullCyclePointerError",
    "FullCycleReceipt", "FullBasisRecord", "FullCyclePersistence",
    "FullCycleMemory", "FullBasisMemory", "FullCycleMemoryPersistence",
    "FullBasisPersistence", "MAX_CYCLES", "MAX_CYCLE_COUNT", "MAX_BASIS",
    "MAX_BASIS_COUNT", "save_full_basis", "load_full_basis",
    "full_basis_audit", "full_basis_pairs", "basis_index",
    "basis_pair", "status", "write", "predict", "train", "snapshot", "audit",
    "query", "handle", "HANDLERS", "build_handlers", "set_word",
    "set_cycle", "predict_cycles", "train_full_basis",
)
