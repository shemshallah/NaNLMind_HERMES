"""Persist raw cycle observables and their metadata in a resident universe.

The resident universe deliberately exposes only a small Python boundary:
``ingest`` appends a JSON-compatible leaf and ``get_leaf`` reads it back.  This
module adds a bounded, content-addressed cycle layer on top of that facade
without changing ``universe.py`` and without opening a second store.

A cycle observable is one oriented edge ``(i, j)`` of the 256-head graph.  It
carries a *raw integer word* (a 64-bit machine word by default) and a small
metadata record.  The reverse edge ``(j, i)`` is its mirror and must carry the
word's two's-complement negation.  That is the same reverse antisymmetry used by
the Round 029 mirror-pair experiments; an arbitrary vertex gauge cancels when
the forward and reverse lanes are read together.

Storage layout
--------------

``cycle-payload:<hash>``
    Immutable, content-addressed payload.  ``hash`` is SHA-256 over the
    canonical JSON text.  It contains the pair, the raw words, the source and
    the bounded metadata.

``cycle-anchor:<hash>``
    Immutable, content-addressed anchor.  It names the payload digest and the
    pair, and its own id is derived from a SHA-256 over its canonical body.

``cycle-slot:<a>:<b>``
    Tiny deterministic directory record for the unordered pair.  It is written
    once and points at the content-addressed anchor/payload.  Slots are the only
    non-content-addressed leaves; they exist so a cycle can be found after a
    reopen without scanning the whole universe.

``cycle-scratch:<hash>`` / ``cycle-scratch-anchor:<version>``
``cycle-snapshot:<hash>`` / ``cycle-snapshot-anchor:<version>``
    Scratch batches and snapshots.  They are immutable payloads with
    version-ordered anchors.  They never replace or modify the main cycle
    records and do not require ``approved=True``.

All integers are serialized as JSON integers or as ``0x`` hexadecimal strings.
Floats, NaN/Infinity, Python objects, raw pointers and filesystem paths are
never accepted or emitted.  The raw word is kept as an integer end-to-end; it is
never converted through ``float``, so values above ``2**53`` round-trip
exactly.

Security and failure policy
---------------------------

Main cycle writes require the literal ``approved=True``.  Without it a falsey
receipt is returned and the universe is not touched.  Scratch batches and
snapshots are non-main writes and deliberately need no approval.

Payloads that exceed their bounds, contain secret-shaped material, look like
filesystem paths or contain pointer-shaped keys are rejected before the
universe is touched.  A missing cycle loads as ``None``.  An anchor or payload
that fails its content address, or whose mirror word is inconsistent, is
reported as corruption rather than silently returning forged data.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

try:  # Keep the module usable with a duck-typed universe in partial installs.
    from . import HEAD_COUNT
except Exception:  # pragma: no cover - defensive fallback
    HEAD_COUNT = 256

UNIVERSE_SCHEMA = "nanlm.resident-universe/1"
CYCLE_SCHEMA = "nanlm.resident-cycle/1"
ANCHOR_SCHEMA = "nanlm.resident-cycle-anchor/1"
SLOT_SCHEMA = "nanlm.resident-cycle-slot/1"
SCRATCH_SCHEMA = "nanlm.resident-cycle-scratch/1"
SNAPSHOT_SCHEMA = "nanlm.resident-cycle-snapshot/1"
SCRATCH_ANCHOR_SCHEMA = "nanlm.resident-cycle-scratch-anchor/1"
SNAPSHOT_ANCHOR_SCHEMA = "nanlm.resident-cycle-snapshot-anchor/1"
RECEIPT_SCHEMA = "nanlm.resident-cycle-receipt/1"
AUDIT_SCHEMA = "nanlm.resident-cycle-audit/1"

# Descriptive aliases for callers that spell the whole layer in the name.
CYCLE_PAYLOAD_SCHEMA = CYCLE_SCHEMA
CYCLE_ANCHOR_SCHEMA = ANCHOR_SCHEMA
CYCLE_SLOT_SCHEMA = SLOT_SCHEMA
CYCLE_SCRATCH_SCHEMA = SCRATCH_SCHEMA
CYCLE_SNAPSHOT_SCHEMA = SNAPSHOT_SCHEMA
CYCLE_RECEIPT_SCHEMA = RECEIPT_SCHEMA
CYCLE_AUDIT_SCHEMA = AUDIT_SCHEMA

WORD_BITS = 64
WORD_MODULUS = 1 << WORD_BITS
WORD_MASK = WORD_MODULUS - 1
WORD_MIN = -(1 << (WORD_BITS - 1))
WORD_MAX = WORD_MASK
MAX_WORD = WORD_MAX
MAX_CYCLE_BYTES = 32 * 1024
MAX_ANCHOR_BYTES = 8 * 1024
MAX_SLOT_BYTES = 2 * 1024
MAX_SCRATCH_BYTES = 256 * 1024
MAX_SNAPSHOT_BYTES = 512 * 1024
MAX_SOURCE_BYTES = 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 4096
MAX_BATCH = 2048
MAX_SNAPSHOT_CYCLES = 4096
MAX_SCRATCH_VERSIONS = 256
MAX_SNAPSHOT_VERSIONS = 256
VERSION_GAP = 8
MAX_STRING_BYTES = 4096


class CyclePersistenceError(ValueError):
    """A cycle record cannot be represented or recovered safely."""


class CyclePersistenceBoundsError(CyclePersistenceError):
    """A payload or integer exceeds the bounded JSON envelope."""


class CyclePersistenceSecretError(CyclePersistenceError):
    """A payload contains secret-shaped keys or values."""


class CyclePersistencePathError(CyclePersistenceError):
    """A payload contains filesystem-path-shaped data."""


class CyclePersistencePointerError(CyclePersistenceError):
    """A payload contains a raw-pointer-shaped key."""


class CyclePersistenceCorruptionError(CyclePersistenceError):
    """A stored cycle, anchor or payload failed its integrity check."""


class CyclePersistenceMirrorError(CyclePersistenceCorruptionError):
    """A stored cycle's forward/reverse words do not satisfy the mirror."""


class CycleReceipt(dict):
    """Receipt mapping whose truth value follows ``accepted``."""

    def __bool__(self) -> bool:
        return bool(self.get("accepted", False))


class CycleRecord(dict):
    """Loaded cycle record.

    The record behaves like the JSON mapping that was stored, but it also
    converts to the signed raw word and compares equal to either the signed or
    the unsigned representation of that word.  This keeps call sites readable
    without hiding the metadata.
    """

    def __int__(self) -> int:
        return int(self.get("raw_word", 0))

    def __index__(self) -> int:
        return int(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return False
        if isinstance(other, int):
            word = self.get("word")
            if isinstance(word, int) and not isinstance(word, bool):
                return int(word) == (int(other) & WORD_MASK)
            return False
        if isinstance(other, Mapping):
            # A caller may compare against a small expected projection.
            try:
                return all(
                    key in self and self[key] == value
                    for key, value in other.items())
            except Exception:
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash((tuple(self.get("pair", ())),
                     self.get("word"), self.get("mirror")))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.items()}


_SECRET_KEY_EXACT = frozenset({
    "apikey", "accesstoken", "authtoken", "authorization", "bearer",
    "clientsecret", "cookie", "credential", "credentials", "password",
    "passwd", "passphrase", "privatekey", "refreshtoken", "secret",
    "secretkey", "session", "sessionid", "signingkey", "token",
})
_SECRET_KEY_SUFFIXES = (
    "apikey", "accesskey", "privatekey", "secretkey", "password", "passwd",
    "secret", "token", "credential", "authorization", "cookie",
)
_SECRET_KEY_PREFIXES = (
    "secret", "password", "privatekey", "apikey", "accesskey", "authtoken",
    "authorization", "credential",
)
_SECRET_KEY_RE = re.compile(r"[^a-z0-9]+")
_POINTER_KEY_EXACT = frozenset({
    "addr", "address", "buffer", "bufferaddress", "memaddress",
    "memoryaddress", "offset", "pointer", "ptr", "rawpointer", "rawptr",
})
_PATH_KEY_EXACT = frozenset({
    "cwd", "dir", "directory", "file", "filename", "filepath", "folder",
    "location", "path", "root", "uri", "url",
})
_PATH_KEY_SUFFIXES = (
    "path", "filename", "filepath", "directory", "folder",
)
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_RELATIVE_TRAVERSAL_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_HEX_WORD_RE = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
_DEC_WORD_RE = re.compile(r"^[+-]?[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]{16,}=*\b", re.IGNORECASE),
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)"
        r"\s*[:=]\s*\S+"
    ),
    re.compile(
        r"^[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}$"
    ),
)


def _compact_key(key: str) -> str:
    return _SECRET_KEY_RE.sub("", key.casefold())


def _secret_key(key: str) -> bool:
    compact = _compact_key(key)
    if compact in _SECRET_KEY_EXACT:
        return True
    if any(compact.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES):
        return True
    return any(compact.startswith(prefix) for prefix in _SECRET_KEY_PREFIXES)


def _path_key(key: str) -> bool:
    compact = _compact_key(key)
    return compact in _PATH_KEY_EXACT or any(
        compact.endswith(suffix) for suffix in _PATH_KEY_SUFFIXES)


def _pointer_key(key: str) -> bool:
    return _compact_key(key) in _POINTER_KEY_EXACT


def _path_key_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.casefold()
    if lowered.startswith(("http://", "https://", "urn:", "mailto:")):
        return False
    return True


def _path_like(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if "\x00" in text:
        return True
    if text.startswith(("/", "~/", "./", ".\\", "\\\\")):
        return True
    if _DRIVE_PATH_RE.match(text):
        return True
    if text.casefold().startswith("file:"):
        return True
    return bool(_RELATIVE_TRAVERSAL_RE.search(text))


_SCHEMA_VERSION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*/[0-9]+$")
_FILE_LIKE_RE = re.compile(r"\.[A-Za-z0-9]{1,12}$")
_SUSPICIOUS_PATH_SEGMENTS = frozenset({
    "credentials", "etc", "data", "file", "files", "filename", "home",
    "passwd", "path", "paths", "private", "root", "secret", "secrets",
    "shadow", "storage", "tmp", "usr", "var",
})


def _slash_path(value: str) -> bool:
    """Reject relative path strings such as ``notes/private.txt``.

    Plain URLs, URNs, schema identifiers such as
    ``nanlm.resident-cycle/1`` and namespace labels such as
    ``experiments/round029`` remain legal.  A slash alone is not a denial:
    the traversal, absolute-path and file-extension checks above carry the
    security decision.
    """
    text = value.strip()
    if not text:
        return False
    lowered = text.casefold()
    if lowered.startswith(("http://", "https://", "urn:", "mailto:")):
        return False
    if _SCHEMA_VERSION_RE.match(text):
        return False
    if "/" not in text and "\\" not in text:
        return False
    if _FILE_LIKE_RE.search(text):
        return True
    parts = re.split(r"[\\/]+", text)
    if ".." in parts or (parts and parts[0] in (".", "~")):
        return True
    return any(part.casefold() in _SUSPICIOUS_PATH_SEGMENTS
               for part in parts if part)


def _check_string(value: str, *, label: str,
                  key: str | None = None) -> None:
    if "\x00" in value:
        raise CyclePersistencePathError("%s contains a NUL byte" % label)
    if _path_like(value) or _slash_path(value):
        raise CyclePersistencePathError(
            "%s contains a filesystem-path-like value" % label)
    if key is not None and _path_key(key) and _path_key_value(value):
        raise CyclePersistencePathError(
            "%s key %r contains a path-like value" % (label, key))
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(value):
            raise CyclePersistenceSecretError(
                "%s contains secret-shaped data" % label)


def _validate_json(value: Any, *, label: str, depth: int,
                   counter: list[int], key: str | None = None) -> Any:
    """Return a detached JSON-only value or raise a bounded error.

    Floats are rejected outright: cycle observables are integer words, and a
    silently rounded float would change the raw bits being persisted.
    """
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise CyclePersistenceBoundsError(
            "%s exceeds %d JSON nodes" % (label, MAX_JSON_NODES))
    if depth > MAX_JSON_DEPTH:
        raise CyclePersistenceBoundsError(
            "%s exceeds JSON depth %d" % (label, MAX_JSON_DEPTH))
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise CyclePersistenceBoundsError(
                "%s contains an oversized integer" % label)
        return value
    if isinstance(value, float):
        raise CyclePersistenceBoundsError(
            "%s contains a float; cycle payloads require integers only"
            % label)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_STRING_BYTES:
            raise CyclePersistenceBoundsError(
                "%s contains an oversized string" % label)
        _check_string(value, label=label, key=key)
        return value
    if isinstance(value, (list, tuple)):
        return [
            _validate_json(item, label=label, depth=depth + 1,
                           counter=counter, key=key)
            for item in value
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CyclePersistenceError(
                    "%s object keys must be strings" % label)
            if not key:
                raise CyclePersistenceError(
                    "%s contains an empty object key" % label)
            if len(key.encode("utf-8")) > 256:
                raise CyclePersistenceBoundsError(
                    "%s contains an oversized object key" % label)
            _check_string(key, label="%s key" % label)
            if _secret_key(key):
                raise CyclePersistenceSecretError(
                    "%s contains a secret-shaped key %r" % (label, key))
            if _pointer_key(key):
                raise CyclePersistencePointerError(
                    "%s contains a pointer-shaped key %r" % (label, key))
            result[key] = _validate_json(
                item, label="%s.%s" % (label, key), depth=depth + 1,
                counter=counter, key=key)
        return result
    raise CyclePersistenceError(
        "%s must contain only JSON objects, arrays, strings, integers, "
        "booleans or null; got %s" % (label, type(value).__name__))


def _canonical_text(value: Any, *, label: str,
                    max_bytes: int) -> tuple[str, Any]:
    counter = [0]
    clean = _validate_json(value, label=label, depth=0, counter=counter)
    try:
        raw = json.dumps(
            clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CyclePersistenceError(
            "%s cannot be encoded as canonical JSON" % label) from exc
    if len(raw) > max_bytes:
        raise CyclePersistenceBoundsError(
            "%s is %d bytes; the limit is %d bytes"
            % (label, len(raw), max_bytes))
    return raw.decode("utf-8"), clean


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text if _SHA256_RE.match(text) else None


def _signed_word(word: int) -> int:
    return word - WORD_MODULUS if word >= (1 << (WORD_BITS - 1)) else word


def _mirror_word(word: int) -> int:
    return (-word) & WORD_MASK


def _word_hex(word: int) -> str:
    return "0x%x" % word


def _normalize_word(value: Any) -> int:
    """Return an unsigned 64-bit word without ever using ``float``."""
    if isinstance(value, bool):
        raise CyclePersistenceBoundsError(
            "raw_word must be an integer, not a boolean")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise CyclePersistenceError("raw_word must not be empty")
        if _HEX_WORD_RE.match(text):
            try:
                number = int(text, 16)
            except ValueError as exc:
                raise CyclePersistenceError(
                    "raw_word is not a hexadecimal integer") from exc
        elif _DEC_WORD_RE.match(text):
            try:
                number = int(text, 10)
            except ValueError as exc:
                raise CyclePersistenceError(
                    "raw_word is not a decimal integer") from exc
        else:
            raise CyclePersistenceBoundsError(
                "raw_word must be a decimal or 0x-hexadecimal integer")
    elif isinstance(value, float):
        raise CyclePersistenceBoundsError(
            "raw_word must be an integer; floats are not permitted")
    else:
        raise CyclePersistenceBoundsError(
            "raw_word must be an integer; got %s" % type(value).__name__)
    if number < WORD_MIN or number > WORD_MAX:
        raise CyclePersistenceBoundsError(
            "raw_word is outside the signed/unsigned 64-bit range")
    return number & WORD_MASK


def _validate_pair(i: Any, j: Any) -> tuple[int, int]:
    for name, value in (("i", i), ("j", j)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CyclePersistenceError(
                "%s must be an integer in [0, %d)" % (name, HEAD_COUNT - 1))
        if value < 0 or value >= HEAD_COUNT:
            raise CyclePersistenceError(
                "%s is outside [0, %d)" % (name, HEAD_COUNT))
    first = int(i)
    second = int(j)
    if first == second:
        raise CyclePersistenceError(
            "a cycle edge requires two distinct head indices")
    return (first, second) if first < second else (second, first)


def _validate_source(source: Any) -> str:
    if source is None:
        return "unspecified"
    if not isinstance(source, str):
        raise CyclePersistenceError("source must be a string or null")
    text = source.strip()
    if not text:
        return "unspecified"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise CyclePersistenceBoundsError(
            "source exceeds %d bytes" % MAX_SOURCE_BYTES)
    _check_string(source, label="cycle source")
    return source


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CyclePersistenceError("metadata must be an object or null")
    _text, clean = _canonical_text(
        dict(metadata), label="cycle metadata", max_bytes=MAX_METADATA_BYTES)
    return clean


def _source_digest(source: str) -> str:
    digest = _valid_digest(source)
    return digest if digest is not None else _sha256_text(source)


def _payload_id(digest: str) -> str:
    return "cycle-payload:" + digest[:32]


def _anchor_id(digest: str) -> str:
    return "cycle-anchor:" + digest[:32]


def _scratch_id(digest: str) -> str:
    return "cycle-scratch:" + digest[:32]


def _snapshot_id(digest: str) -> str:
    return "cycle-snapshot:" + digest[:32]


def _slot_id(pair: tuple[int, int]) -> str:
    return "cycle-slot:%d:%d" % pair


def _scratch_anchor_id(version: int) -> str:
    return "cycle-scratch-anchor:%d" % int(version)


def _snapshot_anchor_id(version: int) -> str:
    return "cycle-snapshot-anchor:%d" % int(version)


def _anchor_digest(anchor: Mapping[str, Any]) -> str:
    body = {key: value for key, value in anchor.items()
            if key != "anchor_digest"}
    text, _clean = _canonical_text(
        body, label="cycle anchor digest", max_bytes=MAX_ANCHOR_BYTES)
    return _sha256_text(text)


def _slot_digest(slot: Mapping[str, Any]) -> str:
    body = {key: value for key, value in slot.items() if key != "slot_digest"}
    text, _clean = _canonical_text(
        body, label="cycle slot digest", max_bytes=MAX_SLOT_BYTES)
    return _sha256_text(text)


def _build_payload(pair: tuple[int, int], word: int, source: str,
                   metadata: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    forward = int(word)
    mirror = _mirror_word(forward)
    payload = {
        "schema": CYCLE_SCHEMA,
        "kind": "cycle",
        "i": int(pair[0]),
        "j": int(pair[1]),
        "pair": [int(pair[0]), int(pair[1])],
        "word": forward,
        "word_hex": _word_hex(forward),
        "raw_word": _signed_word(forward),
        "mirror": mirror,
        "mirror_hex": _word_hex(mirror),
        "mirror_word": _signed_word(mirror),
        "source": source,
        "source_digest": _source_digest(source),
        "metadata": dict(metadata),
        "read_only": True,
        "word_bits": WORD_BITS,
    }
    text, clean = _canonical_text(
        payload, label="cycle payload", max_bytes=MAX_CYCLE_BYTES)
    return text, clean, _sha256_text(text)


def _build_anchor(pair: tuple[int, int], payload: Mapping[str, Any],
                  payload_digest: str, payload_leaf_id: str,
                  data_generation: int) -> tuple[str, dict[str, Any], str]:
    anchor = {
        "schema": ANCHOR_SCHEMA,
        "kind": "cycle",
        "i": int(pair[0]),
        "j": int(pair[1]),
        "pair": [int(pair[0]), int(pair[1])],
        "payload_leaf_id": payload_leaf_id,
        "data_leaf_id": payload_leaf_id,
        "content_digest": payload_digest,
        "payload_digest": payload_digest,
        "word": int(payload["word"]),
        "word_hex": str(payload["word_hex"]),
        "mirror": int(payload["mirror"]),
        "mirror_hex": str(payload["mirror_hex"]),
        "source": str(payload["source"]),
        "source_digest": str(payload["source_digest"]),
        "data_generation": int(data_generation),
        "commit_generation": int(data_generation),
        "read_only": True,
        "word_bits": WORD_BITS,
    }
    digest = _anchor_digest(anchor)
    anchor["anchor_digest"] = digest
    text, clean = _canonical_text(
        anchor, label="cycle anchor", max_bytes=MAX_ANCHOR_BYTES)
    return text, clean, digest


def _build_slot(pair: tuple[int, int], anchor_id: str, anchor_digest: str,
                payload_digest: str, data_generation: int) -> tuple[str, dict[str, Any]]:
    slot = {
        "schema": SLOT_SCHEMA,
        "kind": "cycle-slot",
        "i": int(pair[0]),
        "j": int(pair[1]),
        "pair": [int(pair[0]), int(pair[1])],
        "anchor_leaf_id": anchor_id,
        "anchor_digest": anchor_digest,
        "payload_digest": payload_digest,
        "data_generation": int(data_generation),
        "commit_generation": int(data_generation),
        "read_only": True,
    }
    slot["slot_digest"] = _slot_digest(slot)
    text, clean = _canonical_text(
        slot, label="cycle slot", max_bytes=MAX_SLOT_BYTES)
    return text, clean


def _walk_json(value: Any, *, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise CyclePersistenceCorruptionError(
            "stored cycle JSON exceeds the depth bound")
    if isinstance(value, float):
        raise CyclePersistenceCorruptionError(
            "stored cycle payload contains a float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CyclePersistenceCorruptionError(
                    "stored cycle object key is not a string")
            if _pointer_key(key):
                raise CyclePersistenceCorruptionError(
                    "stored cycle payload contains a pointer-shaped key")
            _walk_json(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk_json(item, depth=depth + 1)
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    raise CyclePersistenceCorruptionError(
        "stored cycle payload contains a non-JSON value")


def _batch_items(cycles: Any) -> list[dict[str, Any]]:
    """Normalize the accepted batch shapes into item mappings."""
    if isinstance(cycles, Mapping):
        if isinstance(cycles.get("cycles"), (list, tuple)):
            cycles = cycles["cycles"]
        else:
            expanded = []
            for key, value in cycles.items():
                if not isinstance(key, (list, tuple)) or len(key) != 2:
                    raise CyclePersistenceError(
                        "cycle batch mappings must be keyed by (i, j)")
                expanded.append({"i": key[0], "j": key[1], "raw_word": value})
            cycles = expanded
    if isinstance(cycles, (str, bytes)) or not isinstance(
            cycles, (list, tuple, Sequence)):
        raise CyclePersistenceError(
            "cycle batch must be a sequence of items")
    items = list(cycles)
    if len(items) > MAX_BATCH:
        raise CyclePersistenceBoundsError(
            "cycle batch exceeds %d items" % MAX_BATCH)
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            if "i" in item:
                first = item.get("i")
            else:
                first = item.get("a")
            if "j" in item:
                second = item.get("j")
            else:
                second = item.get("b")
            word = item.get("raw_word", item.get("word"))
            if word is None:
                raise CyclePersistenceError(
                    "cycle batch item %d is missing raw_word" % index)
            entry = {
                "i": first, "j": second, "raw_word": word,
                "source": item.get("source"),
                "metadata": item.get("metadata"),
            }
            if "mirror" in item:
                entry["mirror"] = item.get("mirror")
            result.append(entry)
        elif isinstance(item, (list, tuple)):
            if len(item) == 3:
                result.append({
                    "i": item[0], "j": item[1], "raw_word": item[2],
                    "source": None, "metadata": None,
                })
            elif len(item) == 2:
                pair = item[0]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise CyclePersistenceError(
                        "cycle batch item %d does not contain a pair" % index)
                result.append({
                    "i": pair[0], "j": pair[1], "raw_word": item[1],
                    "source": None, "metadata": None,
                })
            else:
                raise CyclePersistenceError(
                    "cycle batch item %d must be (i, j, word)" % index)
        else:
            raise CyclePersistenceError(
                "cycle batch item %d has an unsupported shape" % index)
    return result


class CyclePersistence:
    """Bounded persistence adapter over ``ResidentUniverse`` or a fake facade.

    Only the public facade is required: ``ingest`` and ``get_leaf``.  A
    ``snapshot`` method is used for generation/digest receipts when available,
    and ``checkpoint`` is delegated when the caller asks for one.
    """

    def __init__(self, universe: Any):
        for name in ("ingest", "get_leaf"):
            if not callable(getattr(universe, name, None)):
                raise TypeError("universe must expose %s()" % name)
        self.universe = universe
        self._known_pairs: set[tuple[int, int]] = set()

    # ------------------------------------------------------------- public API
    def save_cycle(self, i: int, j: int, raw_word: Any, *, source: Any = None,
                   approved: bool = False,
                   metadata: Any = None,
                   mirror: Any = None) -> CycleReceipt:
        """Commit one main cycle edge.

        ``approved`` must be the literal ``True``.  The first approved write
        for an unordered pair is immutable.  Repeating the same direction and
        word is an idempotent duplicate.  Writing the reverse direction is a
        duplicate only when it carries the two's-complement mirror word; any
        other reverse word is denied as ``mirror_mismatch``.  A different word
        on the original direction is denied as read-only.
        """
        pair = _validate_pair(i, j)
        word = _normalize_word(raw_word)
        source_text = _validate_source(source)
        clean_metadata = _validate_metadata(metadata)
        forward = (int(i) == pair[0])
        canonical_word = word if forward else _mirror_word(word)
        payload_text, payload, payload_digest = _build_payload(
            pair, canonical_word, source_text, clean_metadata)
        payload_leaf_id = _payload_id(payload_digest)

        if mirror is not None:
            supplied = _normalize_word(mirror)
            if supplied != _mirror_word(word):
                return self._denial(
                    "mirror_mismatch",
                    "the supplied mirror word is not the two's-complement "
                    "negation of the requested word",
                    approved=approved is True,
                    i=int(i), j=int(j), pair=pair,
                    raw_word=_signed_word(word), word=word,
                    word_hex=_word_hex(word),
                    expected_mirror=_signed_word(_mirror_word(word)),
                    supplied_mirror=_signed_word(supplied),
                    leaf_id=payload_leaf_id,
                    content_digest=payload_digest,
                )

        if approved is not True:
            return self._denial(
                "approval_required",
                "explicit approved=True is required for a main cycle commit",
                approved=False,
                i=int(i), j=int(j), pair=pair,
                raw_word=_signed_word(word), word=word,
                word_hex=_word_hex(word),
                mirror=_signed_word(_mirror_word(word)),
                leaf_id=payload_leaf_id,
                content_digest=payload_digest,
            )

        existing = self._read_cycle(pair)
        if existing is not None:
            return self._existing_receipt(
                existing, requested_i=int(i), requested_j=int(j),
                requested_word=word, approved=True)

        before = self._state()
        payload_result = self._ingest_leaf(
            payload_leaf_id, payload_text, head=pair[0],
            label="cycle payload")
        payload_generation = self._generation(payload_result)
        payload_resident = self._resident_id(payload_result)

        anchor_text, anchor, anchor_digest = _build_anchor(
            pair, payload, payload_digest, payload_leaf_id, payload_generation)
        anchor_leaf_id = _anchor_id(anchor_digest)
        anchor_result = self._ingest_leaf(
            anchor_leaf_id, anchor_text, head=pair[0],
            label="cycle anchor")
        anchor_generation = self._generation(anchor_result)
        anchor_resident = self._resident_id(anchor_result)

        slot_text, slot = _build_slot(
            pair, anchor_leaf_id, anchor_digest, payload_digest,
            anchor_generation)
        slot_leaf_id = _slot_id(pair)
        slot_result = self._ingest_leaf(
            slot_leaf_id, slot_text, head=pair[0], label="cycle slot")
        slot_generation = self._generation(slot_result)
        slot_resident = self._resident_id(slot_result)

        stored = self._read_cycle(pair)
        if stored is None or stored["payload"].get("word") != canonical_word:
            raise CyclePersistenceCorruptionError(
                "cycle slot %s did not commit the requested word"
                % slot_leaf_id)
        self._known_pairs.add(pair)
        return self._receipt(
            status="committed", accepted=True, created=True, duplicate=False,
            approved=True, kind="cycle", i=int(i), j=int(j), pair=pair,
            version=0,
            direction="forward" if forward else "mirror",
            raw_word=_signed_word(word), word=word,
            word_hex=_word_hex(word),
            mirror=_signed_word(_mirror_word(word)),
            mirror_hex=_word_hex(_mirror_word(word)),
            forward_word=_signed_word(canonical_word),
            forward_hex=_word_hex(canonical_word),
            reverse_word=_signed_word(_mirror_word(canonical_word)),
            reverse_hex=_word_hex(_mirror_word(canonical_word)),
            source=source_text,
            source_digest=_source_digest(source_text),
            metadata=clean_metadata,
            leaf_id=payload_leaf_id,
            payload_leaf_id=payload_leaf_id,
            resident_leaf_id=payload_resident,
            anchor_leaf_id=anchor_leaf_id,
            anchor_resident_leaf_id=anchor_resident,
            slot_leaf_id=slot_leaf_id,
            slot_resident_leaf_id=slot_resident,
            content_digest=payload_digest,
            payload_digest=payload_digest,
            anchor_digest=anchor_digest,
            slot_digest=slot.get("slot_digest"),
            data_generation=payload_generation,
            anchor_generation=anchor_generation,
            slot_generation=slot_generation,
            generation_before=before["generation"],
            main_modified=True, main_cycle_modified=True,
        )

    def load_cycle(self, i: int, j: int,
                   default: Any = None) -> CycleRecord | Any:
        """Return one cycle in the requested direction, or ``default``."""
        pair = _validate_pair(i, j)
        stored = self._read_cycle(pair)
        if stored is None:
            return default
        return self._record_view(stored, int(i), int(j))

    def load_word(self, i: int, j: int, default: Any = None) -> Any:
        """Return only the requested direction's signed raw word."""
        record = self.load_cycle(i, j, None)
        return default if record is None else int(record)

    def save_cycle_batch(self, cycles: Any, *, source: Any = None,
                         approved: bool = False, metadata: Any = None,
                         scratch: bool | None = None) -> CycleReceipt:
        """Save many cycle edges.

        With ``approved=True`` the batch is a main write and is preflighted
        atomically: if any item conflicts, no item is written.  Without
        approval the batch is written to the scratch area as a content-addressed
        leaf and the main cycle records are left untouched.  ``scratch=False``
        forces main semantics; ``scratch=True`` forces scratch semantics.
        """
        raw_items = _batch_items(cycles)
        if not raw_items:
            return self._receipt(
                status="empty", accepted=True, created=False, duplicate=False,
                approved=None, kind="batch", items=[], main_modified=False,
                main_cycle_modified=False, committed=0, duplicates=0,
                cycle_count=0,
            )
        items: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            pair = _validate_pair(item["i"], item["j"])
            word = _normalize_word(item["raw_word"])
            item_source = item.get("source")
            item_metadata = item.get("metadata")
            source_text = _validate_source(
                source if item_source is None else item_source)
            clean_metadata = _validate_metadata(
                metadata if item_metadata is None else item_metadata)
            supplied_mirror = item.get("mirror")
            if supplied_mirror is not None:
                supplied = _normalize_word(supplied_mirror)
                if supplied != _mirror_word(word):
                    if scratch is False or approved is True:
                        return self._denial(
                            "mirror_mismatch",
                            "batch item %d supplies a mismatched mirror" % index,
                            approved=approved is True, kind="batch",
                            item_index=index, i=int(item["i"]),
                            j=int(item["j"]), pair=pair,
                            raw_word=_signed_word(word), word=word,
                            expected_mirror=_signed_word(_mirror_word(word)),
                            supplied_mirror=_signed_word(supplied),
                        )
            items.append({
                "i": int(item["i"]), "j": int(item["j"]), "pair": pair,
                "word": word, "source": source_text,
                "metadata": clean_metadata,
            })

        main_mode = approved is True
        if scratch is True:
            main_mode = False
        elif scratch is False:
            main_mode = True

        if main_mode:
            if approved is not True:
                return self._denial(
                    "approval_required",
                    "explicit approved=True is required for a main batch",
                    approved=False, kind="batch", items=[],
                    requested_count=len(items),
                )
            return self._save_main_batch(items)

        return self._save_scratch_batch(
            items, source=_validate_source(source))

    def snapshot(self, *, source: Any = None,
                 approved: Any = None) -> CycleReceipt:
        """Write an immutable snapshot of the main cycle records.

        Snapshots are non-main writes: no approval is required.  The returned
        receipt also contains the snapshot itself (``"cycles"``,
        ``"cycle_count"``, ``"main_generation"`` and ``"main_digest"``).
        """
        _ = approved  # accepted for call-site symmetry; never needed
        source_text = _validate_source(
            "snapshot" if source is None else source)
        records = self._all_main_records()
        state = self._main_state(records)
        if len(records) > MAX_SNAPSHOT_CYCLES:
            raise CyclePersistenceBoundsError(
                "cycle snapshot exceeds %d records" % MAX_SNAPSHOT_CYCLES)
        cycles: list[dict[str, Any]] = []
        for stored in records:
            payload = stored["payload"]
            cycles.append({
                "i": int(payload["i"]),
                "j": int(payload["j"]),
                "pair": [int(payload["i"]), int(payload["j"])],
                "word": int(payload["word"]),
                "word_hex": str(payload["word_hex"]),
                "raw_word": int(payload["raw_word"]),
                "mirror": int(payload["mirror"]),
                "mirror_hex": str(payload["mirror_hex"]),
                "mirror_word": int(payload["mirror_word"]),
                "source": str(payload["source"]),
                "source_digest": str(payload["source_digest"]),
                "payload_digest": str(stored["payload_digest"]),
                "anchor_digest": str(stored["anchor_digest"]),
                "metadata": dict(payload["metadata"]),
            })
        cycles.sort(key=lambda item: (item["i"], item["j"]))
        snapshot = {
            "schema": SNAPSHOT_SCHEMA,
            "kind": "snapshot",
            "cycles": cycles,
            "cycle_count": len(cycles),
            "words": {"%d:%d" % (item["i"], item["j"]): item["word"]
                      for item in cycles},
            "source": source_text,
            "source_digest": _source_digest(source_text),
            "main_generation": state["generation"],
            "main_digest": state["digest"],
        }
        text, clean = _canonical_text(
            snapshot, label="cycle snapshot", max_bytes=MAX_SNAPSHOT_BYTES)
        digest = _sha256_text(text)
        leaf_id = _snapshot_id(digest)
        existing = self._find_versioned_by_digest(
            "snapshot", digest, MAX_SNAPSHOT_VERSIONS)
        if existing is not None:
            return self._snapshot_receipt(
                existing, digest, text, clean, duplicate=True,
                source_text=source_text)
        version = self._next_version("snapshot", MAX_SNAPSHOT_VERSIONS)
        before = self._state()
        payload_result = self._ingest_leaf(
            leaf_id, text, head=0, label="cycle snapshot")
        payload_generation = self._generation(payload_result)
        anchor_text, anchor = self._versioned_anchor(
            "snapshot", version, leaf_id, digest, len(cycles),
            payload_generation, state)
        anchor_leaf_id = _snapshot_anchor_id(version)
        self._ingest_leaf(
            anchor_leaf_id, anchor_text, head=0, label="cycle snapshot anchor")
        anchor_record = self._read_versioned_anchor(
            "snapshot", version, expected_digest=digest)
        return self._snapshot_receipt(
            anchor_record, digest, text, clean, duplicate=False,
            source_text=source_text, version=version,
            payload_generation=payload_generation,
            generation_before=before["generation"],
        )

    def load_snapshot(self, version: int | None = None) -> dict[str, Any] | None:
        """Load a snapshot payload, latest by default."""
        if version is None:
            anchor = self._latest_versioned("snapshot", MAX_SNAPSHOT_VERSIONS)
        else:
            anchor = self._read_versioned_anchor("snapshot", int(version))
        if anchor is None:
            return None
        return self._read_versioned_payload(
            anchor, "snapshot", max_bytes=MAX_SNAPSHOT_BYTES)

    def load_scratch_batch(self, version: int | None = None
                           ) -> dict[str, Any] | None:
        """Load a scratch batch payload, latest by default."""
        if version is None:
            anchor = self._latest_versioned(
                "scratch", MAX_SCRATCH_VERSIONS)
        else:
            anchor = self._read_versioned_anchor("scratch", int(version))
        if anchor is None:
            return None
        return self._read_versioned_payload(
            anchor, "scratch", max_bytes=MAX_SCRATCH_BYTES)

    def audit(self, *, full: bool = False) -> dict[str, Any]:
        """Verify every discoverable main, scratch and snapshot record.

        The audit is read-only.  It recomputes the payload and anchor digests,
        rechecks the mirror invariant and rejects floats, pointer-shaped keys,
        paths and secret-shaped material in stored payloads.
        """
        state = self._state()
        problems: list[str] = []
        problem_codes: list[str] = []
        summaries: list[dict[str, Any]] = []

        entries = self._discover_slot_entries(
            full=bool(full))
        seen: dict[tuple[int, int], list[str]] = {}
        for entry in entries:
            pair = entry.get("pair")
            leaf_id = entry.get("leaf_id")
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or isinstance(pair[0], bool) or isinstance(pair[1], bool)
                    or not isinstance(pair[0], int)
                    or not isinstance(pair[1], int)):
                problems.append("cycle slot %r has a malformed pair" % leaf_id)
                problem_codes.append("slot_pair")
                continue
            tuple_pair = (int(pair[0]), int(pair[1]))
            seen.setdefault(tuple_pair, []).append(str(leaf_id))
        for pair in self._known_pairs:
            seen.setdefault(pair, [])
        for pair, leaf_ids in sorted(seen.items()):
            if len(leaf_ids) > 1:
                problems.append(
                    "pair %d:%d is referenced by multiple slots: %s"
                    % (pair[0], pair[1], ", ".join(sorted(leaf_ids))))
                problem_codes.append("duplicate_slot")
            try:
                stored = self._read_cycle(pair)
            except CyclePersistenceError as exc:
                code = ("mirror_mismatch"
                        if "mirror" in str(exc).casefold() else "corruption")
                problems.append("pair %d:%d: %s" % (pair[0], pair[1], exc))
                problem_codes.append(code)
                continue
            if stored is None:
                problems.append("pair %d:%d is missing its slot" % pair)
                problem_codes.append("missing_slot")
                continue
            try:
                self._check_stored_record(stored)
            except CyclePersistenceError as exc:
                code = ("mirror_mismatch"
                        if "mirror" in str(exc).casefold() else "corruption")
                problems.append("pair %d:%d: %s" % (pair[0], pair[1], exc))
                problem_codes.append(code)
                continue
            payload = stored["payload"]
            summaries.append({
                "i": int(payload["i"]),
                "j": int(payload["j"]),
                "pair": [int(payload["i"]), int(payload["j"])],
                "word": int(payload["word"]),
                "word_hex": str(payload["word_hex"]),
                "mirror": int(payload["mirror"]),
                "mirror_hex": str(payload["mirror_hex"]),
                "source_digest": str(payload["source_digest"]),
                "payload_digest": str(stored["payload_digest"]),
                "anchor_digest": str(stored["anchor_digest"]),
            })

        scratch_checked = 0
        for version in self._version_numbers(
                "scratch", MAX_SCRATCH_VERSIONS):
            try:
                anchor = self._read_versioned_anchor("scratch", version)
                if anchor is None:
                    continue
                self._read_versioned_payload(
                    anchor, "scratch", max_bytes=MAX_SCRATCH_BYTES)
                scratch_checked += 1
            except CyclePersistenceError as exc:
                problems.append(
                    "scratch version %d: %s" % (version, exc))
                problem_codes.append("scratch_corruption")

        snapshot_checked = 0
        for version in self._version_numbers(
                "snapshot", MAX_SNAPSHOT_VERSIONS):
            try:
                anchor = self._read_versioned_anchor("snapshot", version)
                if anchor is None:
                    continue
                self._read_versioned_payload(
                    anchor, "snapshot", max_bytes=MAX_SNAPSHOT_BYTES)
                snapshot_checked += 1
            except CyclePersistenceError as exc:
                problems.append(
                    "snapshot version %d: %s" % (version, exc))
                problem_codes.append("snapshot_corruption")

        ok = not problems
        checked = len(summaries) + scratch_checked + snapshot_checked
        return {
            "schema": AUDIT_SCHEMA,
            "status": "ok" if ok else "corrupt",
            "ok": ok,
            "valid": ok,
            "clean": ok,
            "accepted": ok,
            "checked": checked,
            "audited": checked,
            "cycles": len(summaries),
            "cycle_count": len(summaries),
            "cycle_records": summaries,
            "records": summaries,
            "entries": summaries,
            "mirror_checked": len(summaries),
            "scratch_checked": scratch_checked,
            "snapshot_checked": snapshot_checked,
            "problems": problems,
            "problem_codes": problem_codes,
            "generation": state["generation"],
            "digest": state["digest"],
            "state_digest": state["digest"],
            "main_generation": state["generation"],
            "main_digest": state["digest"],
            "read_only_main": True,
            "no_floats": not any("float" in item for item in problems),
            "no_raw_pointers": not any(
                "pointer" in item for item in problems),
            "canonical_commit": False,
            "canonical_state_modified": False,
        }

    def checkpoint(self, directory: Any = None) -> Any:
        """Delegate a checkpoint to the underlying universe, if supported."""
        checkpoint = getattr(self.universe, "checkpoint", None)
        if not callable(checkpoint):
            raise CyclePersistenceError(
                "the universe does not expose checkpoint()")
        return checkpoint(directory)

    def status(self) -> dict[str, Any]:
        """Return a small, read-only description of the persisted layer."""
        records = self._all_main_records()
        scratch = self._latest_versioned("scratch", MAX_SCRATCH_VERSIONS)
        snapshot = self._latest_versioned("snapshot", MAX_SNAPSHOT_VERSIONS)
        state = self._state()
        return {
            "schema": RECEIPT_SCHEMA,
            "generation": state["generation"],
            "digest": state["digest"],
            "cycle_count": len(records),
            "cycles": len(records),
            "scratch_version": None if scratch is None
            else int(scratch.get("version", 0)),
            "snapshot_version": None if snapshot is None
            else int(snapshot.get("version", 0)),
            "read_only_main": True,
            "canonical_commit": False,
            "canonical_state_modified": False,
        }

    # --------------------------------------------------------------- internals
    def _state(self, fallback: Mapping[str, Any] | None = None
               ) -> dict[str, Any]:
        snapshot: Mapping[str, Any] = {}
        getter = getattr(self.universe, "snapshot", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if isinstance(value, Mapping):
                snapshot = value
        fallback = fallback if isinstance(fallback, Mapping) else {}
        generation = snapshot.get("generation")
        if generation is None:
            generation = fallback.get("generation")
        digest = snapshot.get("digest") or snapshot.get("state_digest") \
            or fallback.get("digest") or fallback.get("state_digest") or ""
        if generation is None:
            generation = getattr(self.universe, "generation", 0)
        if not digest:
            digest = getattr(self.universe, "digest", "") or ""
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 0
        if not digest:
            # A duck-typed facade need not expose a digest.  Keep the receipt
            # shape deterministic and content-addressed rather than emitting
            # an empty string from a successful write.
            digest = _sha256_text("cycle-universe-state:%d" % generation)
        return {"generation": generation, "digest": str(digest)}

    def _main_state(self, records: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any]:
        """Return a stable identity for the main cycle layer.

        Universe generations also advance for scratch and snapshot leaves, so
        they cannot be used as the identity of the main cycle content.  The
        newest main commit generation and a digest over the sorted cycle
        payload digests are stable across non-main writes and across a reopen.
        """
        if records is None:
            records = self._all_main_records()
        generation = 0
        material: list[str] = []
        for stored in records:
            anchor = stored.get("anchor", {})
            try:
                commit = int(anchor.get("commit_generation", 0))
            except (TypeError, ValueError):
                commit = 0
            generation = max(generation, commit)
            payload = stored.get("payload", {})
            material.append("%d:%d:%s" % (
                int(payload.get("i", 0)), int(payload.get("j", 0)),
                stored.get("payload_digest", "")))
        return {
            "generation": generation,
            "digest": _sha256_text("\n".join(sorted(material))),
            "count": len(records),
        }

    @staticmethod
    def _generation(result: Any) -> int:
        if isinstance(result, Mapping):
            try:
                return int(result.get("generation", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _resident_id(result: Any) -> str | None:
        if not isinstance(result, Mapping):
            return None
        value = result.get("leaf_id")
        if isinstance(value, str):
            return value
        ids = result.get("leaf_ids")
        if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], str):
            return ids[0]
        return None

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
            "cycle_persistence": True,
            "cycle_label": label,
        }
        result = self.universe.ingest(record)
        if not isinstance(result, Mapping):
            raise CyclePersistenceError(
                "universe.ingest() returned a non-mapping receipt")
        return result

    def _get_leaf(self, leaf_id: str) -> Mapping[str, Any] | None:
        try:
            leaf = self.universe.get_leaf(leaf_id)
        except (KeyError, LookupError):
            return None
        return leaf if isinstance(leaf, Mapping) else None

    def _leaf_text(self, leaf: Mapping[str, Any], *,
                   max_bytes: int, label: str) -> str:
        raw = leaf.get("text")
        if raw is None:
            raw = leaf.get("payload")
        if isinstance(raw, (Mapping, list, tuple)):
            try:
                raw = json.dumps(
                    raw, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CyclePersistenceCorruptionError(
                    "%s is not canonical JSON" % label) from exc
        if not isinstance(raw, str):
            raise CyclePersistenceCorruptionError(
                "%s is not a JSON text leaf" % label)
        size = len(raw.encode("utf-8"))
        if size > max_bytes:
            raise CyclePersistenceBoundsError(
                "%s is %d bytes; the limit is %d bytes"
                % (label, size, max_bytes))
        return raw

    @staticmethod
    def _parse_object(text: str, *, label: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CyclePersistenceCorruptionError(
                "%s is not valid JSON" % label) from exc
        if not isinstance(value, dict):
            raise CyclePersistenceCorruptionError(
                "%s is not a JSON object" % label)
        return value

    def _read_slot(self, pair: tuple[int, int]) -> dict[str, Any] | None:
        leaf_id = _slot_id(pair)
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            return None
        text = self._leaf_text(
            leaf, max_bytes=MAX_SLOT_BYTES, label="cycle slot")
        slot = self._parse_object(text, label="cycle slot %s" % leaf_id)
        if slot.get("schema") != SLOT_SCHEMA \
                or slot.get("kind") != "cycle-slot":
            raise CyclePersistenceCorruptionError(
                "cycle slot %s has an unexpected schema/kind" % leaf_id)
        if slot.get("pair") != [int(pair[0]), int(pair[1])]:
            raise CyclePersistenceCorruptionError(
                "cycle slot %s belongs to another pair" % leaf_id)
        digest = slot.get("slot_digest")
        if not isinstance(digest, str) or _slot_digest(slot) != digest:
            raise CyclePersistenceCorruptionError(
                "cycle slot %s failed its content digest" % leaf_id)
        anchor_leaf_id = slot.get("anchor_leaf_id")
        if not isinstance(anchor_leaf_id, str) \
                or not anchor_leaf_id.startswith("cycle-anchor:"):
            raise CyclePersistenceCorruptionError(
                "cycle slot %s has an invalid anchor reference" % leaf_id)
        for field in ("anchor_digest", "payload_digest"):
            value = slot.get(field)
            if not isinstance(value, str) or _valid_digest(value) is None:
                raise CyclePersistenceCorruptionError(
                    "cycle slot %s is missing %s" % (leaf_id, field))
        return slot

    def _read_anchor(self, anchor_leaf_id: str) -> dict[str, Any]:
        leaf = self._get_leaf(anchor_leaf_id)
        if leaf is None:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s is missing" % anchor_leaf_id)
        text = self._leaf_text(
            leaf, max_bytes=MAX_ANCHOR_BYTES, label="cycle anchor")
        anchor = self._parse_object(
            text, label="cycle anchor %s" % anchor_leaf_id)
        if anchor.get("schema") != ANCHOR_SCHEMA \
                or anchor.get("kind") != "cycle":
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an unexpected schema/kind"
                % anchor_leaf_id)
        digest = anchor.get("anchor_digest")
        if not isinstance(digest, str) or _valid_digest(digest) is None:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s is missing its digest" % anchor_leaf_id)
        if _anchor_digest(anchor) != digest:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s failed its content digest" % anchor_leaf_id)
        if anchor_leaf_id != _anchor_id(digest):
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s does not match its content address"
                % anchor_leaf_id)
        pair = anchor.get("pair")
        if not isinstance(pair, list) or len(pair) != 2 \
                or anchor.get("i") != pair[0] or anchor.get("j") != pair[1]:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an inconsistent pair" % anchor_leaf_id)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in pair) or pair[0] == pair[1] \
                or not (0 <= pair[0] < HEAD_COUNT) \
                or not (0 <= pair[1] < HEAD_COUNT):
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an out-of-range pair" % anchor_leaf_id)
        word = anchor.get("word")
        mirror = anchor.get("mirror")
        if isinstance(word, bool) or not isinstance(word, int) \
                or word < 0 or word > WORD_MASK:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an invalid word" % anchor_leaf_id)
        if mirror != _mirror_word(word):
            raise CyclePersistenceMirrorError(
                "cycle anchor %s has a mismatched mirror word"
                % anchor_leaf_id)
        if anchor.get("word_hex") != _word_hex(word) \
                or anchor.get("mirror_hex") != _word_hex(mirror):
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has inconsistent hexadecimal words"
                % anchor_leaf_id)
        if anchor.get("word_bits") != WORD_BITS:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an unexpected word width"
                % anchor_leaf_id)
        payload_leaf_id = anchor.get("payload_leaf_id")
        if not isinstance(payload_leaf_id, str) \
                or not payload_leaf_id.startswith("cycle-payload:"):
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has an invalid payload reference"
                % anchor_leaf_id)
        content_digest = anchor.get("content_digest")
        if not isinstance(content_digest, str) \
                or _valid_digest(content_digest) is None:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s is missing its payload digest"
                % anchor_leaf_id)
        if anchor.get("payload_digest") != content_digest:
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s has conflicting payload digests"
                % anchor_leaf_id)
        if payload_leaf_id != _payload_id(content_digest):
            raise CyclePersistenceCorruptionError(
                "cycle anchor %s does not match its payload address"
                % anchor_leaf_id)
        return anchor

    def _read_payload(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        leaf_id = str(anchor.get("payload_leaf_id"))
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            raise CyclePersistenceCorruptionError(
                "cycle payload %s is missing" % leaf_id)
        text = self._leaf_text(
            leaf, max_bytes=MAX_CYCLE_BYTES, label="cycle payload")
        payload = self._parse_object(
            text, label="cycle payload %s" % leaf_id)
        if payload.get("schema") != CYCLE_SCHEMA \
                or payload.get("kind") != "cycle":
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has an unexpected schema/kind" % leaf_id)
        digest = _sha256_text(text)
        if digest != anchor.get("content_digest"):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s does not match its content address"
                % leaf_id)
        if leaf_id != _payload_id(digest):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s does not match its digest" % leaf_id)
        if payload.get("pair") != anchor.get("pair"):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s belongs to another pair" % leaf_id)
        if payload.get("i") != anchor.get("i") \
                or payload.get("j") != anchor.get("j"):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has inconsistent indices" % leaf_id)
        pair = payload.get("pair")
        if not isinstance(pair, list) or len(pair) != 2 \
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in pair) or pair[0] == pair[1] \
                or not (0 <= pair[0] < HEAD_COUNT) \
                or not (0 <= pair[1] < HEAD_COUNT):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has an out-of-range pair" % leaf_id)
        word = payload.get("word")
        mirror = payload.get("mirror")
        if isinstance(word, bool) or not isinstance(word, int) \
                or word < 0 or word > WORD_MASK:
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has an invalid word" % leaf_id)
        if mirror != _mirror_word(word):
            raise CyclePersistenceMirrorError(
                "cycle payload %s has a mismatched mirror word" % leaf_id)
        if payload.get("word_hex") != _word_hex(word) \
                or payload.get("mirror_hex") != _word_hex(mirror):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has inconsistent hexadecimal words"
                % leaf_id)
        if payload.get("word_bits") != WORD_BITS:
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has an unexpected word width" % leaf_id)
        if payload.get("raw_word") != _signed_word(word) \
                or payload.get("mirror_word") != _signed_word(mirror):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has inconsistent signed words" % leaf_id)
        if payload.get("word") != anchor.get("word") \
                or payload.get("mirror") != anchor.get("mirror"):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s disagrees with its anchor" % leaf_id)
        if payload.get("source") != anchor.get("source") \
                or payload.get("source_digest") != anchor.get("source_digest"):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s disagrees with its source" % leaf_id)
        if payload.get("source_digest") != _source_digest(
                str(payload.get("source", ""))):
            raise CyclePersistenceCorruptionError(
                "cycle payload %s has a forged source digest" % leaf_id)
        _walk_json(payload)
        # Re-run the normal validators so a tampered payload cannot smuggle a
        # path, secret or pointer past a later read.
        _validate_source(payload.get("source"))
        _validate_metadata(payload.get("metadata"))
        return payload

    def _read_cycle(self, pair: tuple[int, int]
                    ) -> dict[str, Any] | None:
        slot = self._read_slot(pair)
        if slot is None:
            return None
        anchor = self._read_anchor(str(slot["anchor_leaf_id"]))
        if anchor.get("anchor_digest") != slot.get("anchor_digest"):
            raise CyclePersistenceCorruptionError(
                "cycle slot %s does not match its anchor digest" % _slot_id(pair))
        if anchor.get("pair") != slot.get("pair"):
            raise CyclePersistenceCorruptionError(
                "cycle slot %s and its anchor disagree on the pair"
                % _slot_id(pair))
        payload = self._read_payload(anchor)
        if anchor.get("content_digest") != slot.get("payload_digest"):
            raise CyclePersistenceCorruptionError(
                "cycle slot %s does not match its payload digest"
                % _slot_id(pair))
        return {
            "pair": pair,
            "slot": slot,
            "anchor": anchor,
            "payload": payload,
            "slot_leaf_id": _slot_id(pair),
            "anchor_leaf_id": str(slot["anchor_leaf_id"]),
            "payload_leaf_id": str(anchor["payload_leaf_id"]),
            "payload_digest": str(anchor["content_digest"]),
            "anchor_digest": str(anchor["anchor_digest"]),
            "slot_digest": str(slot["slot_digest"]),
        }

    def _check_stored_record(self, stored: Mapping[str, Any]) -> None:
        payload = stored["payload"]
        anchor = stored["anchor"]
        slot = stored["slot"]
        if int(payload["i"]) != int(slot["i"]) \
                or int(payload["j"]) != int(slot["j"]):
            raise CyclePersistenceCorruptionError(
                "stored cycle indices disagree")
        if payload["mirror"] != _mirror_word(int(payload["word"])):
            raise CyclePersistenceMirrorError(
                "stored cycle mirror word does not match")
        if anchor["mirror"] != _mirror_word(int(anchor["word"])):
            raise CyclePersistenceMirrorError(
                "stored cycle anchor mirror word does not match")
        for field in ("payload_digest", "anchor_digest", "slot_digest"):
            if _valid_digest(stored.get(field)) is None:
                raise CyclePersistenceCorruptionError(
                    "stored cycle %s is malformed" % field)

    def _record_view(self, stored: Mapping[str, Any],
                     requested_i: int, requested_j: int) -> CycleRecord:
        payload = stored["payload"]
        first, second = int(payload["i"]), int(payload["j"])
        forward = (requested_i == first)
        forward_word = int(payload["word"])
        reverse_word = int(payload["mirror"])
        word = forward_word if forward else reverse_word
        mirror = reverse_word if forward else forward_word
        record = CycleRecord({
            "schema": CYCLE_SCHEMA,
            "kind": "cycle",
            "i": int(requested_i),
            "j": int(requested_j),
            "pair": [first, second],
            "a": first,
            "b": second,
            "direction": "forward" if forward else "mirror",
            "word": word,
            "word_hex": _word_hex(word),
            "raw_word": _signed_word(word),
            "mirror": mirror,
            "mirror_hex": _word_hex(mirror),
            "mirror_word": _signed_word(mirror),
            "forward_word": _signed_word(forward_word),
            "forward_hex": _word_hex(forward_word),
            "reverse_word": _signed_word(reverse_word),
            "reverse_hex": _word_hex(reverse_word),
            "source": str(payload["source"]),
            "source_digest": str(payload["source_digest"]),
            "metadata": dict(payload["metadata"]),
            "read_only": True,
            "word_bits": WORD_BITS,
            "payload_leaf_id": stored["payload_leaf_id"],
            "leaf_id": stored["payload_leaf_id"],
            "anchor_leaf_id": stored["anchor_leaf_id"],
            "slot_leaf_id": stored["slot_leaf_id"],
            "content_digest": stored["payload_digest"],
            "payload_digest": stored["payload_digest"],
            "anchor_digest": stored["anchor_digest"],
            "slot_digest": stored["slot_digest"],
            "data_generation": int(
                stored["anchor"].get("data_generation", 0)),
            "commit_generation": int(
                stored["anchor"].get("commit_generation", 0)),
            "canonical_commit": False,
            "canonical_state_modified": False,
        })
        return record

    def _existing_receipt(self, stored: Mapping[str, Any], *,
                          requested_i: int, requested_j: int,
                          requested_word: int,
                          approved: bool) -> CycleReceipt:
        pair = stored["pair"]
        payload = stored["payload"]
        expected = (int(payload["word"]) if requested_i == pair[0]
                    else int(payload["mirror"]))
        if requested_word != expected:
            if requested_i != pair[0]:
                return self._denial(
                    "mirror_mismatch",
                    "reverse cycle write does not match the stored mirror word",
                    approved=approved, kind="cycle", i=requested_i,
                    j=requested_j, pair=pair,
                    raw_word=_signed_word(requested_word),
                    word=requested_word,
                    expected_mirror=_signed_word(expected),
                    stored_word=_signed_word(int(payload["word"])),
                    content_digest=stored["payload_digest"],
                    anchor_leaf_id=stored["anchor_leaf_id"],
                    leaf_id=stored["payload_leaf_id"],
                )
            return self._denial(
                "read_only",
                "the main cycle edge is committed read-only",
                approved=approved, kind="cycle", i=requested_i,
                j=requested_j, pair=pair,
                raw_word=_signed_word(requested_word),
                word=requested_word,
                existing_word=_signed_word(int(payload["word"])),
                content_digest=stored["payload_digest"],
                anchor_leaf_id=stored["anchor_leaf_id"],
                leaf_id=stored["payload_leaf_id"],
            )
        self._known_pairs.add(pair)
        return self._receipt(
            status="duplicate", accepted=True, created=False, duplicate=True,
            approved=approved, kind="cycle", i=requested_i, j=requested_j,
            pair=pair, version=0,
            direction="forward" if requested_i == pair[0] else "mirror",
            raw_word=_signed_word(requested_word), word=requested_word,
            word_hex=_word_hex(requested_word),
            mirror=_signed_word(_mirror_word(requested_word)),
            mirror_hex=_word_hex(_mirror_word(requested_word)),
            forward_word=_signed_word(int(payload["word"])),
            forward_hex=str(payload["word_hex"]),
            reverse_word=_signed_word(int(payload["mirror"])),
            reverse_hex=str(payload["mirror_hex"]),
            source=str(payload["source"]),
            source_digest=str(payload["source_digest"]),
            metadata=dict(payload["metadata"]),
            leaf_id=stored["payload_leaf_id"],
            payload_leaf_id=stored["payload_leaf_id"],
            resident_leaf_id=None,
            anchor_leaf_id=stored["anchor_leaf_id"],
            slot_leaf_id=stored["slot_leaf_id"],
            content_digest=stored["payload_digest"],
            payload_digest=stored["payload_digest"],
            anchor_digest=stored["anchor_digest"],
            slot_digest=stored["slot_digest"],
            data_generation=int(stored["anchor"].get("data_generation", 0)),
            anchor_generation=int(
                stored["anchor"].get("commit_generation", 0)),
            slot_generation=int(
                stored["slot"].get("commit_generation", 0)),
            generation_before=None,
            main_modified=False, main_cycle_modified=False,
        )

    def _save_main_batch(self, items: list[dict[str, Any]]) -> CycleReceipt:
        before = self._state()
        planned: dict[tuple[int, int], int] = {}
        duplicates = 0
        for index, item in enumerate(items):
            pair = item["pair"]
            forward = (item["i"] == pair[0])
            requested = int(item["word"])
            if pair in planned:
                expected = (planned[pair] if forward
                            else _mirror_word(planned[pair]))
                if requested != expected:
                    return self._batch_denial(
                        "mirror_mismatch" if not forward else "read_only",
                        "batch item %d conflicts with another item for pair "
                        "%d:%d" % (index, pair[0], pair[1]),
                        items, index)
                duplicates += 1
                continue
            existing = self._read_cycle(pair)
            if existing is None:
                planned[pair] = (requested if forward
                                 else _mirror_word(requested))
                continue
            payload = existing["payload"]
            expected = (int(payload["word"]) if forward
                        else int(payload["mirror"]))
            if requested != expected:
                return self._batch_denial(
                    "mirror_mismatch" if not forward else "read_only",
                    "batch item %d conflicts with the stored edge for pair "
                    "%d:%d" % (index, pair[0], pair[1]),
                    items, index)
            duplicates += 1

        receipts: list[CycleReceipt] = []
        created = 0
        for item in items:
            receipt = self.save_cycle(
                item["i"], item["j"], item["word"],
                source=item["source"], approved=True,
                metadata=item["metadata"])
            receipts.append(receipt)
            if receipt.get("created"):
                created += 1
                self._known_pairs.add(item["pair"])
        state = self._state()
        return self._receipt(
            status="committed" if created else "duplicate",
            accepted=True, created=bool(created), duplicate=not created,
            approved=True, kind="batch", items=receipts,
            receipts=receipts,
            committed=created, duplicates=len(items) - created,
            cycle_count=len({item["pair"] for item in items}),
            leaf_ids=[receipt.get("leaf_id") for receipt in receipts],
            words={"%d:%d" % item["pair"]: _signed_word(item["word"])
                   for item in items},
            generation_before=before["generation"],
            main_modified=bool(created), main_cycle_modified=bool(created),
            batch_size=len(items), count=len(items),
        )

    def _batch_denial(self, status: str, reason: str,
                      items: list[dict[str, Any]],
                      item_index: int) -> CycleReceipt:
        item = items[item_index]
        return self._denial(
            status, reason, approved=True, kind="batch", items=[],
            item_index=item_index,
            i=item["i"], j=item["j"], pair=item["pair"],
            raw_word=_signed_word(item["word"]),
            word=item["word"], word_hex=_word_hex(item["word"]),
            requested_count=len(items),
        )

    def _save_scratch_batch(self, items: list[dict[str, Any]],
                            *, source: str) -> CycleReceipt:
        canonical: dict[tuple[int, int], dict[str, Any]] = {}
        for index, item in enumerate(items):
            pair = item["pair"]
            forward = (item["i"] == pair[0])
            word = int(item["word"])
            forward_word = word if forward else _mirror_word(word)
            previous = canonical.get(pair)
            if previous is not None:
                if previous["word"] != forward_word:
                    return self._denial(
                        "mirror_mismatch",
                        "scratch batch item %d conflicts with another item "
                        "for pair %d:%d" % (index, pair[0], pair[1]),
                        approved=None, kind="batch", items=[],
                        item_index=index, i=item["i"], j=item["j"],
                        pair=pair, raw_word=_signed_word(word),
                        word=word, word_hex=_word_hex(word),
                    )
                if previous["source"] == item["source"] \
                        and previous["metadata"] == item["metadata"]:
                    continue
            canonical[pair] = {
                "i": pair[0], "j": pair[1], "pair": [pair[0], pair[1]],
                "word": forward_word, "word_hex": _word_hex(forward_word),
                "raw_word": _signed_word(forward_word),
                "mirror": _mirror_word(forward_word),
                "mirror_hex": _word_hex(_mirror_word(forward_word)),
                "mirror_word": _signed_word(_mirror_word(forward_word)),
                "source": item["source"],
                "source_digest": _source_digest(item["source"]),
                "metadata": dict(item["metadata"]),
            }
        entries = [canonical[pair] for pair in sorted(canonical)]
        state = self._main_state()
        payload = {
            "schema": SCRATCH_SCHEMA,
            "kind": "scratch",
            "cycles": entries,
            "cycle_count": len(entries),
            "source": source,
            "source_digest": _source_digest(source),
            "main_generation": state["generation"],
            "main_digest": state["digest"],
        }
        text, clean = _canonical_text(
            payload, label="cycle scratch batch", max_bytes=MAX_SCRATCH_BYTES)
        digest = _sha256_text(text)
        leaf_id = _scratch_id(digest)
        existing = self._find_versioned_by_digest(
            "scratch", digest, MAX_SCRATCH_VERSIONS)
        if existing is not None:
            return self._scratch_receipt(
                existing, digest, clean, duplicate=True, source=source)
        version = self._next_version("scratch", MAX_SCRATCH_VERSIONS)
        before = self._state()
        result = self._ingest_leaf(
            leaf_id, text, head=0, label="cycle scratch batch")
        payload_generation = self._generation(result)
        anchor_text, anchor = self._versioned_anchor(
            "scratch", version, leaf_id, digest, len(entries),
            payload_generation, state)
        anchor_leaf_id = _scratch_anchor_id(version)
        self._ingest_leaf(
            anchor_leaf_id, anchor_text, head=0, label="cycle scratch anchor")
        anchor_record = self._read_versioned_anchor(
            "scratch", version, expected_digest=digest)
        return self._scratch_receipt(
            anchor_record, digest, clean, duplicate=False, source=source,
            version=version, payload_generation=payload_generation,
            generation_before=before["generation"],
        )

    def _scratch_receipt(self, anchor: Mapping[str, Any], digest: str,
                         payload: Mapping[str, Any], *, duplicate: bool,
                         source: str, version: int | None = None,
                         payload_generation: int | None = None,
                         generation_before: int | None = None
                         ) -> CycleReceipt:
        if version is None:
            version = int(anchor.get("version", 0))
        return self._receipt(
            status="duplicate" if duplicate else "committed",
            accepted=True, created=not duplicate, duplicate=duplicate,
            approved=None, kind="scratch", version=version,
            cycle_count=int(payload.get("cycle_count", 0)),
            cycles=list(payload.get("cycles", [])),
            main_generation=payload.get("main_generation"),
            main_digest=payload.get("main_digest"),
            source=source,
            leaf_id=anchor.get("data_leaf_id"),
            payload_leaf_id=anchor.get("data_leaf_id"),
            anchor_leaf_id=_scratch_anchor_id(version),
            content_digest=digest, payload_digest=digest,
            anchor_digest=anchor.get("anchor_digest"),
            data_generation=anchor.get("data_generation"),
            anchor_generation=anchor.get("commit_generation"),
            generation_before=generation_before,
            main_modified=False, main_cycle_modified=False,
        )

    def _snapshot_receipt(self, anchor: Mapping[str, Any], digest: str,
                          text: str, payload: Mapping[str, Any], *,
                          duplicate: bool, source_text: str,
                          version: int | None = None,
                          payload_generation: int | None = None,
                          generation_before: int | None = None
                          ) -> CycleReceipt:
        if version is None:
            version = int(anchor.get("version", 0))
        receipt = self._receipt(
            status="duplicate" if duplicate else "committed",
            accepted=True, created=not duplicate, duplicate=duplicate,
            approved=None, kind="snapshot", version=version,
            cycle_count=int(payload.get("cycle_count", 0)),
            count=int(payload.get("cycle_count", 0)),
            cycles=list(payload.get("cycles", [])),
            cycle_records=list(payload.get("cycles", [])),
            records=list(payload.get("cycles", [])),
            words=dict(payload.get("words", {})),
            main_generation=payload.get("main_generation"),
            main_digest=payload.get("main_digest"),
            source=source_text,
            leaf_id=anchor.get("data_leaf_id"),
            payload_leaf_id=anchor.get("data_leaf_id"),
            snapshot_leaf_id=anchor.get("data_leaf_id"),
            anchor_leaf_id=_snapshot_anchor_id(version),
            content_digest=digest, payload_digest=digest,
            snapshot_digest=digest,
            anchor_digest=anchor.get("anchor_digest"),
            data_generation=anchor.get("data_generation"),
            anchor_generation=anchor.get("commit_generation"),
            generation_before=generation_before,
            main_modified=False, main_cycle_modified=False,
        )
        return receipt

    def _versioned_anchor(self, kind: str, version: int, leaf_id: str,
                          digest: str, cycle_count: int,
                          data_generation: int,
                          main_state: Mapping[str, Any]
                          ) -> tuple[str, dict[str, Any]]:
        schema = (SCRATCH_ANCHOR_SCHEMA if kind == "scratch"
                  else SNAPSHOT_ANCHOR_SCHEMA)
        anchor = {
            "schema": schema,
            "kind": "%s-anchor" % kind,
            "version": int(version),
            "data_leaf_id": leaf_id,
            "payload_leaf_id": leaf_id,
            "content_digest": digest,
            "payload_digest": digest,
            "cycle_count": int(cycle_count),
            "main_generation": int(main_state.get("generation", 0)),
            "main_digest": str(main_state.get("digest", "")),
            "data_generation": int(data_generation),
            "commit_generation": int(data_generation),
            "read_only": True,
        }
        body_text, _clean = _canonical_text(
            anchor, label="cycle %s anchor digest" % kind,
            max_bytes=MAX_ANCHOR_BYTES)
        anchor["anchor_digest"] = _sha256_text(body_text)
        text, _clean = _canonical_text(
            anchor, label="cycle %s anchor" % kind,
            max_bytes=MAX_ANCHOR_BYTES)
        return text, anchor

    def _read_versioned_anchor(self, kind: str, version: int,
                               expected_digest: str | None = None
                               ) -> dict[str, Any] | None:
        if isinstance(version, bool) or not isinstance(version, int) \
                or version < 0:
            raise CyclePersistenceError(
                "cycle %s version must be a non-negative integer" % kind)
        limit = (MAX_SCRATCH_VERSIONS if kind == "scratch"
                 else MAX_SNAPSHOT_VERSIONS)
        if version >= limit:
            raise CyclePersistenceError(
                "cycle %s version is outside [0, %d)" % (kind, limit))
        leaf_id = (_scratch_anchor_id(version) if kind == "scratch"
                   else _snapshot_anchor_id(version))
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            return None
        text = self._leaf_text(
            leaf, max_bytes=MAX_ANCHOR_BYTES,
            label="cycle %s anchor" % kind)
        anchor = self._parse_object(
            text, label="cycle %s anchor %s" % (kind, leaf_id))
        expected_schema = (SCRATCH_ANCHOR_SCHEMA if kind == "scratch"
                           else SNAPSHOT_ANCHOR_SCHEMA)
        if anchor.get("schema") != expected_schema \
                or anchor.get("kind") != "%s-anchor" % kind \
                or int(anchor.get("version", -1)) != version:
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s has an unexpected schema/version"
                % (kind, leaf_id))
        digest = anchor.get("anchor_digest")
        if not isinstance(digest, str) or _valid_digest(digest) is None:
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s is missing its digest" % (kind, leaf_id))
        body = {key: value for key, value in anchor.items()
                if key != "anchor_digest"}
        body_text, _clean = _canonical_text(
            body, label="cycle %s anchor digest" % kind,
            max_bytes=MAX_ANCHOR_BYTES)
        if _sha256_text(body_text) != digest:
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s failed its content digest"
                % (kind, leaf_id))
        content_digest = anchor.get("content_digest")
        if not isinstance(content_digest, str) \
                or _valid_digest(content_digest) is None:
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s is missing its payload digest"
                % (kind, leaf_id))
        if expected_digest is not None and content_digest != expected_digest:
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s points at another payload"
                % (kind, leaf_id))
        data_leaf_id = anchor.get("data_leaf_id")
        if not isinstance(data_leaf_id, str) or not data_leaf_id.startswith(
                ("cycle-scratch:" if kind == "scratch" else "cycle-snapshot:")):
            raise CyclePersistenceCorruptionError(
                "cycle %s anchor %s has an invalid payload reference"
                % (kind, leaf_id))
        anchor["version"] = version
        return anchor

    def _read_versioned_payload(self, anchor: Mapping[str, Any], kind: str, *,
                                max_bytes: int) -> dict[str, Any]:
        leaf_id = str(anchor.get("data_leaf_id"))
        leaf = self._get_leaf(leaf_id)
        if leaf is None:
            raise CyclePersistenceCorruptionError(
                "cycle %s payload %s is missing" % (kind, leaf_id))
        text = self._leaf_text(
            leaf, max_bytes=max_bytes, label="cycle %s payload" % kind)
        payload = self._parse_object(
            text, label="cycle %s payload %s" % (kind, leaf_id))
        expected_schema = (SCRATCH_SCHEMA if kind == "scratch"
                           else SNAPSHOT_SCHEMA)
        expected_kind = "scratch" if kind == "scratch" else "snapshot"
        if payload.get("schema") != expected_schema \
                or payload.get("kind") != expected_kind:
            raise CyclePersistenceCorruptionError(
                "cycle %s payload %s has an unexpected schema/kind"
                % (kind, leaf_id))
        digest = _sha256_text(text)
        if digest != anchor.get("content_digest"):
            raise CyclePersistenceCorruptionError(
                "cycle %s payload %s does not match its content address"
                % (kind, leaf_id))
        if leaf_id != (_scratch_id(digest) if kind == "scratch"
                       else _snapshot_id(digest)):
            raise CyclePersistenceCorruptionError(
                "cycle %s payload %s does not match its digest"
                % (kind, leaf_id))
        _walk_json(payload)
        source_text = _validate_source(payload.get("source"))
        if payload.get("source_digest") != _source_digest(source_text):
            raise CyclePersistenceCorruptionError(
                "cycle %s payload has a forged source digest" % kind)
        cycles = payload.get("cycles")
        if not isinstance(cycles, list):
            raise CyclePersistenceCorruptionError(
                "cycle %s payload has no cycle list" % kind)
        if int(payload.get("cycle_count", -1)) != len(cycles):
            raise CyclePersistenceCorruptionError(
                "cycle %s payload has an inconsistent cycle count" % kind)
        for item in cycles:
            if not isinstance(item, Mapping):
                raise CyclePersistenceCorruptionError(
                    "cycle %s payload contains a malformed cycle" % kind)
            word = item.get("word")
            mirror = item.get("mirror")
            if isinstance(word, bool) or not isinstance(word, int) \
                    or word < 0 or word > WORD_MASK:
                raise CyclePersistenceCorruptionError(
                    "cycle %s payload contains an invalid word" % kind)
            if mirror != _mirror_word(word):
                raise CyclePersistenceMirrorError(
                    "cycle %s payload contains a mismatched mirror" % kind)
            if item.get("word_hex") != _word_hex(word) \
                    or item.get("mirror_hex") != _word_hex(mirror):
                raise CyclePersistenceCorruptionError(
                    "cycle %s payload has inconsistent words" % kind)
            item_source = _validate_source(item.get("source"))
            if item.get("source_digest") != _source_digest(item_source):
                raise CyclePersistenceCorruptionError(
                    "cycle %s payload has a forged cycle source digest"
                    % kind)
            _validate_metadata(item.get("metadata"))
        if kind == "snapshot":
            words = payload.get("words")
            expected_words = {
                "%d:%d" % (int(item["i"]), int(item["j"])): int(item["word"])
                for item in cycles
            }
            if not isinstance(words, Mapping) or dict(words) != expected_words:
                raise CyclePersistenceCorruptionError(
                    "cycle snapshot payload has an inconsistent word map")
        return payload

    def _version_numbers(self, kind: str, limit: int) -> list[int]:
        versions: list[int] = []
        misses = 0
        for version in range(limit):
            leaf = self._get_leaf(
                _scratch_anchor_id(version) if kind == "scratch"
                else _snapshot_anchor_id(version))
            if leaf is None:
                misses += 1
                if misses >= VERSION_GAP:
                    break
                continue
            misses = 0
            versions.append(version)
        return versions

    def _latest_versioned(self, kind: str, limit: int
                          ) -> dict[str, Any] | None:
        versions = self._version_numbers(kind, limit)
        if not versions:
            return None
        return self._read_versioned_anchor(kind, versions[-1])

    def _next_version(self, kind: str, limit: int) -> int:
        versions = self._version_numbers(kind, limit)
        if not versions:
            return 0
        version = versions[-1] + 1
        if version >= limit:
            raise CyclePersistenceBoundsError(
                "cycle %s version limit (%d) reached" % (kind, limit))
        return version

    def _find_versioned_by_digest(self, kind: str, digest: str,
                                  limit: int) -> dict[str, Any] | None:
        for version in self._version_numbers(kind, limit):
            anchor = self._read_versioned_anchor(kind, version)
            if anchor is not None and anchor.get("content_digest") == digest:
                return anchor
        return None

    def _discover_slot_entries(self, *, full: bool = False
                               ) -> list[dict[str, Any]]:
        """Find slot leaves without trusting an in-memory cache.

        Real universes expose ``delta_records`` and can enumerate all leaves in
        one bounded pass.  A fake facade may expose its leaf mapping or leaf
        records.  The final fallback probes every unordered pair.
        """
        if not full:
            delta = getattr(self.universe, "delta_records", None)
            if callable(delta):
                try:
                    records = delta(0)
                except Exception:
                    records = None
                if isinstance(records, Mapping) \
                        and "leaves" in records:
                    # The real facade enumerates every leaf here, so an empty
                    # list is authoritative rather than a signal to fall back
                    # to the 32,640-pair probe.
                    return self._entries_from_texts(records.get("leaves", []))
        leaves = getattr(self.universe, "leaves", None)
        if isinstance(leaves, Mapping):
            entries = []
            for key, value in leaves.items():
                text = value.get("text") if isinstance(value, Mapping) else None
                if not isinstance(text, str):
                    continue
                parsed = self._safe_object(text)
                if parsed is not None and parsed.get("schema") == SLOT_SCHEMA:
                    pair = parsed.get("pair")
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        entries.append({
                            "pair": [pair[0], pair[1]],
                            "leaf_id": str(key),
                            "text": text,
                        })
            if entries:
                return entries
        records_method = getattr(leaves, "records", None)
        if callable(records_method):
            entries = []
            try:
                records = list(records_method())
            except Exception:
                records = []
            if len(records) <= 8192:
                for record in records:
                    reference = getattr(record, "leaf_id", None)
                    if isinstance(reference, bytes):
                        reference = reference.hex()
                    if not isinstance(reference, str):
                        continue
                    leaf = self._get_leaf(reference)
                    if leaf is None:
                        continue
                    text = leaf.get("text")
                    if not isinstance(text, str):
                        continue
                    parsed = self._safe_object(text)
                    if parsed is not None and parsed.get("schema") == SLOT_SCHEMA:
                        pair = parsed.get("pair")
                        if isinstance(pair, (list, tuple)) and len(pair) == 2:
                            entries.append({
                                "pair": [pair[0], pair[1]],
                                "leaf_id": reference,
                                "text": text,
                            })
            if entries:
                return entries
        entries = []
        for first in range(HEAD_COUNT):
            for second in range(first + 1, HEAD_COUNT):
                leaf = self._get_leaf(_slot_id((first, second)))
                if leaf is None:
                    continue
                entries.append({
                    "pair": [first, second],
                    "leaf_id": _slot_id((first, second)),
                    "text": leaf.get("text"),
                })
        return entries

    @staticmethod
    def _safe_object(text: Any) -> dict[str, Any] | None:
        if not isinstance(text, str):
            return None
        try:
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _entries_from_texts(self, leaves: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not isinstance(leaves, (list, tuple)):
            return entries
        for item in leaves:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            parsed = self._safe_object(text)
            if parsed is None or parsed.get("schema") != SLOT_SCHEMA:
                continue
            pair = parsed.get("pair")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            entries.append({
                "pair": [pair[0], pair[1]],
                "leaf_id": str(item.get("id") or _slot_id(
                    (int(pair[0]), int(pair[1])))),
                "text": text,
            })
        return entries

    def _all_main_records(self) -> list[dict[str, Any]]:
        entries = self._discover_slot_entries()
        pairs: set[tuple[int, int]] = set(self._known_pairs)
        for entry in entries:
            pair = entry.get("pair")
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pairs.add((int(pair[0]), int(pair[1])))
        records = []
        for pair in sorted(pairs):
            stored = self._read_cycle(pair)
            if stored is not None:
                records.append(stored)
        return records

    def _receipt(self, *, status: str, accepted: bool, created: bool,
                 duplicate: bool, approved: bool | None, kind: str,
                 **extra: Any) -> CycleReceipt:
        state = self._state()
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": status,
            "accepted": bool(accepted),
            "created": bool(created),
            "duplicate": bool(duplicate),
            "persisted": bool(created),
            "approved": approved,
            "kind": kind,
            "leaf_id": None,
            "resident_leaf_id": None,
            "anchor_leaf_id": None,
            "slot_leaf_id": None,
            "content_digest": None,
            "payload_digest": None,
            "anchor_digest": None,
            "generation": state["generation"],
            "digest": state["digest"],
            "state_digest": state["digest"],
            "canonical_commit": False,
            "canonical_state_modified": False,
            "main_cycle_modified": bool(
                extra.pop("main_cycle_modified", False)),
        }
        receipt.update(extra)
        if "main_modified" not in receipt:
            receipt["main_modified"] = False
        receipt["main_field_modified"] = bool(
            receipt.get("main_cycle_modified", False))
        pair = receipt.get("pair")
        if isinstance(pair, tuple):
            receipt["pair"] = list(pair)
        return CycleReceipt(receipt)

    def _denial(self, status: str, reason: str, *, approved: bool,
                kind: str = "cycle", **extra: Any) -> CycleReceipt:
        extra.setdefault("reasons", [reason])
        return self._receipt(
            status=status, accepted=False, created=False, duplicate=False,
            approved=approved, kind=kind, **extra)


__all__ = (
    "UNIVERSE_SCHEMA", "CYCLE_SCHEMA", "ANCHOR_SCHEMA", "SLOT_SCHEMA",
    "SCRATCH_SCHEMA", "SNAPSHOT_SCHEMA", "SCRATCH_ANCHOR_SCHEMA",
    "SNAPSHOT_ANCHOR_SCHEMA", "RECEIPT_SCHEMA", "AUDIT_SCHEMA",
    "CYCLE_PAYLOAD_SCHEMA", "CYCLE_ANCHOR_SCHEMA", "CYCLE_SLOT_SCHEMA",
    "CYCLE_SCRATCH_SCHEMA", "CYCLE_SNAPSHOT_SCHEMA",
    "CYCLE_RECEIPT_SCHEMA", "CYCLE_AUDIT_SCHEMA",
    "WORD_BITS", "WORD_MODULUS", "WORD_MASK", "WORD_MIN", "WORD_MAX",
    "MAX_WORD", "MAX_CYCLE_BYTES", "MAX_ANCHOR_BYTES", "MAX_SLOT_BYTES",
    "MAX_SCRATCH_BYTES", "MAX_SNAPSHOT_BYTES", "MAX_SOURCE_BYTES",
    "MAX_METADATA_BYTES", "MAX_BATCH", "MAX_SNAPSHOT_CYCLES",
    "MAX_SCRATCH_VERSIONS", "MAX_SNAPSHOT_VERSIONS",
    "CyclePersistence", "CyclePersistenceError",
    "CyclePersistenceBoundsError", "CyclePersistenceSecretError",
    "CyclePersistencePathError", "CyclePersistencePointerError",
    "CyclePersistenceCorruptionError", "CyclePersistenceMirrorError",
    "CycleReceipt", "CycleRecord",
)
