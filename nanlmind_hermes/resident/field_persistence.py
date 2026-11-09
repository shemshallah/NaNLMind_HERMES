"""Persist head-field and scratch-layer metadata in a resident universe.

The resident universe is intentionally small at its Python boundary: callers
can append JSON-compatible leaves and read them back through the facade.  This
module adds the missing *field* layer without changing ``universe.py`` and
without introducing a second store:

* a main head-field snapshot is content-addressed, committed once and treated
  as read-only afterwards;
* scratch layers are versioned per ``(head, layer)`` and never replace the
  main field;
* every write returns a bounded receipt with the universe generation/state
  digest and the content digest;
* payloads are ordinary canonical JSON (never pickle, marshal or repr) and are
  bounded by size, depth and node count before they reach the universe.

Storage layout
--------------

Two kinds of leaves are used:

``field:<hash>`` / ``scratch:<head>:<layer>:<hash>``
    Immutable payload leaves.  The hash is derived from the validated JSON
    payload (plus the scratch route), so equal content has an equal address.

``field-anchor:main`` / ``scratch-anchor:<head>:<layer>:<version>``
    Tiny append-only pointer records.  The main-field anchor is written once
    with the first approved commit; scratch anchors give a deterministic,
    version-ordered recovery path without scanning the whole universe.

Anchors are slot-addressed because their whole purpose is to make a
content-addressed payload discoverable after a reopen.  The payloads they point
at remain content-addressed and immutable.

Security and failure policy
---------------------------

Payloads that exceed the configured bounds, contain secret-shaped material or
look like filesystem paths are rejected with :class:`FieldPersistenceError`
before the universe is touched.  A missing field/scratch record loads as
``None``.  An anchor that points at a missing or corrupt payload is reported as
a persistence error rather than silently returning forged data.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

try:  # Keep the module usable with a duck-typed universe in partial installs.
    from . import HEAD_COUNT
except Exception:  # pragma: no cover - defensive fallback
    HEAD_COUNT = 256

UNIVERSE_SCHEMA = "nanlm.resident-universe/1"
FIELD_SCHEMA = "nanlm.hermes-head-field/1"
SCRATCH_SCHEMA = "nanlm.hermes-scratch-layer/1"
ANCHOR_SCHEMA = "nanlm.hermes-field-anchor/1"
RECEIPT_SCHEMA = "nanlm.hermes-field-persistence-receipt/1"

FIELD_ANCHOR_ID = "field-anchor:main"
MAX_FIELD_BYTES = 1024 * 1024
MAX_SCRATCH_BYTES = 1024 * 1024
MAX_ANCHOR_BYTES = 8 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 65_536
MAX_LAYER = 9
MAX_SCRATCH_VERSIONS = 1024
SCRATCH_VERSION_GAP = 8


class FieldPersistenceError(ValueError):
    """A field/scratch record cannot be represented or recovered safely."""


class FieldPersistenceBoundsError(FieldPersistenceError):
    """A payload exceeds the bounded JSON envelope."""


class FieldPersistenceSecretError(FieldPersistenceError):
    """A payload contains secret-shaped keys or values."""


class FieldPersistencePathError(FieldPersistenceError):
    """A payload contains filesystem-path-shaped data."""


class FieldPersistenceCorruptionError(FieldPersistenceError):
    """A stored anchor or payload failed its integrity check."""


class FieldReceipt(dict):
    """Receipt mapping whose truth value follows ``accepted``.

    A caller may inspect ``receipt["accepted"]``/``receipt["status"]`` or use
    the receipt directly in a boolean context.  Both are common in the resident
    subsystem, so supporting both avoids ambiguous call sites.
    """

    def __bool__(self) -> bool:
        return bool(self.get("accepted", False))


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
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_RELATIVE_TRAVERSAL_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_PATH_KEY_EXACT = frozenset({
    "cwd", "dir", "directory", "file", "filename", "filepath", "folder",
    "location", "path", "root", "uri", "url",
})
_PATH_KEY_SUFFIXES = (
    "path", "filename", "filepath", "directory", "folder",
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


def _check_string(value: str, *, label: str, key: str | None = None) -> None:
    if "\x00" in value:
        raise FieldPersistencePathError(
            "%s contains a NUL byte" % label)
    if _path_like(value):
        raise FieldPersistencePathError(
            "%s contains a filesystem-path-like value" % label)
    if key is not None and _path_key(key) and _path_key_value(value):
        raise FieldPersistencePathError(
            "%s key %r contains a path-like value" % (label, key))
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(value):
            raise FieldPersistenceSecretError(
                "%s contains secret-shaped data" % label)


def _validate_json(value: Any, *, label: str, depth: int,
                   counter: list[int], key: str | None = None) -> Any:
    """Return a detached JSON-only value or raise a bounded error.

    The function deliberately does not call ``repr``/``pickle`` on unknown
    objects.  Only JSON primitives, arrays and string-keyed objects are
    accepted, which keeps the on-disk representation language-neutral.
    """
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise FieldPersistenceBoundsError(
            "%s exceeds %d JSON nodes" % (label, MAX_JSON_NODES))
    if depth > MAX_JSON_DEPTH:
        raise FieldPersistenceBoundsError(
            "%s exceeds JSON depth %d" % (label, MAX_JSON_DEPTH))
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise FieldPersistenceBoundsError(
                "%s contains an oversized integer" % label)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FieldPersistenceBoundsError(
                "%s contains a non-finite number" % label)
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        _check_string(value, label=label, key=key)
        return value
    if isinstance(value, list):
        return [
            _validate_json(item, label=label, depth=depth + 1,
                           counter=counter, key=key)
            for item in value
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FieldPersistenceError(
                    "%s object keys must be strings" % label)
            if not key:
                raise FieldPersistenceError(
                    "%s contains an empty object key" % label)
            _check_string(key, label="%s key" % label)
            if _secret_key(key):
                raise FieldPersistenceSecretError(
                    "%s contains a secret-shaped key %r" % (label, key))
            result[key] = _validate_json(
                item, label="%s.%s" % (label, key), depth=depth + 1,
                counter=counter, key=key)
        return result
    raise FieldPersistenceError(
        "%s must contain only JSON objects, arrays, strings, finite numbers, "
        "booleans or null; got %s" % (label, type(value).__name__))


def _canonical_text(value: Any, *, label: str, max_bytes: int) -> tuple[str, Any]:
    counter = [0]
    clean = _validate_json(value, label=label, depth=0, counter=counter)
    try:
        raw = json.dumps(
            clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FieldPersistenceError(
            "%s cannot be encoded as canonical JSON" % label) from exc
    if len(raw) > max_bytes:
        raise FieldPersistenceBoundsError(
            "%s is %d bytes; the limit is %d bytes"
            % (label, len(raw), max_bytes))
    return raw.decode("utf-8"), clean


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_digest(kind: str, text: str, *, head: int | None = None,
                    layer: int | None = None) -> str:
    material = [ANCHOR_SCHEMA, kind]
    if head is not None:
        material.append(str(int(head)))
    if layer is not None:
        material.append(str(int(layer)))
    material.append(text)
    return _sha256_text("\x1f".join(material))


def _field_data_id(digest: str) -> str:
    return "field:" + digest[:32]


def _scratch_data_id(head: int, layer: int, digest: str) -> str:
    return "scratch:%d:%d:%s" % (head, layer, digest[:32])


def _scratch_anchor_id(head: int, layer: int, version: int) -> str:
    return "scratch-anchor:%d:%d:%d" % (head, layer, version)


def _validate_head(head: Any) -> int:
    if isinstance(head, bool) or not isinstance(head, int):
        raise FieldPersistenceError("head must be an integer in [0, %d)"
                                    % (HEAD_COUNT - 1))
    if head < 0 or head >= HEAD_COUNT:
        raise FieldPersistenceError("head is outside [0, %d)" % HEAD_COUNT)
    return head


def _validate_layer(layer: Any) -> int:
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise FieldPersistenceError("layer must be an integer in [0, %d]"
                                    % MAX_LAYER)
    if layer < 0 or layer > MAX_LAYER:
        raise FieldPersistenceError("layer is outside [0, %d]" % MAX_LAYER)
    return layer


class FieldPersistence:
    """Bounded persistence adapter over ``ResidentUniverse`` or a fake facade.

    Only the public facade is required: ``ingest``, ``get_leaf``, ``snapshot``
    and optionally ``checkpoint``.  ``get_relation`` is intentionally not
    needed for these records; fakes may expose it for other tests.
    """

    def __init__(self, universe: Any):
        for name in ("ingest", "get_leaf"):
            if not callable(getattr(universe, name, None)):
                raise TypeError("universe must expose %s()" % name)
        self.universe = universe

    # ------------------------------------------------------------- public API
    def save_field(self, snapshot: Any, *, approved: bool = False
                   ) -> FieldReceipt:
        """Commit the main head-field snapshot exactly once.

        Approval must be the literal ``True``.  Without it a falsey receipt is
        returned and the universe is not touched.  Once a field is committed,
        a different snapshot is denied as read-only; the same snapshot is an
        idempotent duplicate.
        """
        text, _clean = _canonical_text(
            snapshot, label="head-field snapshot", max_bytes=MAX_FIELD_BYTES)
        digest = _content_digest("field", text)
        data_id = _field_data_id(digest)

        if approved is not True:
            return self._denial(
                "approval_required",
                "explicit approved=True is required for a main-field commit",
                approved=False,
                requested_digest=digest,
                leaf_id=data_id,
                content_digest=digest,
            )

        existing = self._read_anchor(FIELD_ANCHOR_ID, "field")
        if existing is not None:
            if existing.get("content_digest") == digest:
                return self._receipt(
                    status="duplicate", accepted=True, created=False,
                    duplicate=True, approved=True, kind="field",
                    leaf_id=existing.get("data_leaf_id", data_id),
                    resident_leaf_id=existing.get("resident_leaf_id"),
                    content_digest=digest,
                    anchor_leaf_id=FIELD_ANCHOR_ID,
                    anchor_generation=existing.get("commit_generation"),
                    data_generation=existing.get("data_generation"),
                )
            return self._denial(
                "read_only",
                "the main head-field is committed read-only; "
                "approval cannot replace it",
                approved=True, requested_digest=digest,
                existing_digest=existing.get("content_digest"),
                anchor_leaf_id=FIELD_ANCHOR_ID,
            )

        before = self._state()
        data_result = self._ingest_leaf(data_id, text, head=0)
        anchor = {
            "schema": ANCHOR_SCHEMA,
            "kind": "field",
            "version": 0,
            "data_leaf_id": data_id,
            "resident_leaf_id": self._resident_id(data_result),
            "content_digest": digest,
            "data_generation": self._generation(data_result),
            "commit_generation": self._generation(data_result),
            "read_only": True,
        }
        self._write_anchor(FIELD_ANCHOR_ID, anchor, head=0)
        # The anchor text is immutable; its data generation is sufficient for
        # recovery.  The receipt reports the post-commit state separately.
        stored = self._read_anchor(FIELD_ANCHOR_ID, "field")
        if stored is None or stored.get("content_digest") != digest:
            return self._denial(
                "commit_conflict",
                "another writer committed a different main field first",
                approved=True, requested_digest=digest,
                existing_digest=(stored or {}).get("content_digest"),
            )
        return self._receipt(
            status="committed", accepted=True, created=True, duplicate=False,
            approved=True, kind="field", leaf_id=data_id,
            resident_leaf_id=self._resident_id(data_result),
            content_digest=digest, anchor_leaf_id=FIELD_ANCHOR_ID,
            anchor_generation=stored.get("commit_generation"),
            data_generation=self._generation(data_result),
            generation_before=before["generation"],
        )

    def load_field(self) -> Any:
        """Return the committed head-field snapshot or ``None`` if absent."""
        anchor = self._read_anchor(FIELD_ANCHOR_ID, "field")
        if anchor is None:
            return None
        payload = self._read_payload(
            anchor, expected_kind="field", max_bytes=MAX_FIELD_BYTES,
            head=None, layer=None)
        return payload

    def save_scratch(self, head: int, layer: int, payload: Any) -> FieldReceipt:
        """Append one bounded scratch-layer payload for ``(head, layer)``.

        Scratch writes never touch the main-field anchor or its payload.  Each
        distinct payload becomes a new immutable version; repeating the current
        payload is an idempotent duplicate.
        """
        head = _validate_head(head)
        layer = _validate_layer(layer)
        text, _clean = _canonical_text(
            payload, label="scratch payload", max_bytes=MAX_SCRATCH_BYTES)
        digest = _content_digest("scratch", text, head=head, layer=layer)
        data_id = _scratch_data_id(head, layer, digest)

        latest = self._latest_scratch_anchor(head, layer)
        if latest is not None and latest.get("content_digest") == digest:
            return self._receipt(
                status="duplicate", accepted=True, created=False,
                duplicate=True, approved=None, kind="scratch",
                head=head, layer=layer,
                version=int(latest.get("version", 0)),
                leaf_id=latest.get("data_leaf_id", data_id),
                resident_leaf_id=latest.get("resident_leaf_id"),
                content_digest=digest,
                anchor_leaf_id=_scratch_anchor_id(
                    head, layer, int(latest.get("version", 0))),
                anchor_generation=latest.get("commit_generation"),
                data_generation=latest.get("data_generation"),
                main_field_modified=False,
            )

        version = 0 if latest is None else int(latest.get("version", 0)) + 1
        while version < MAX_SCRATCH_VERSIONS and self._read_anchor(
                _scratch_anchor_id(head, layer, version), "scratch",
                head=head, layer=layer) is not None:
            version += 1
        if version >= MAX_SCRATCH_VERSIONS:
            raise FieldPersistenceBoundsError(
                "scratch layer version limit (%d) reached for head %d layer %d"
                % (MAX_SCRATCH_VERSIONS, head, layer))

        before = self._state()
        data_result = self._ingest_leaf(data_id, text, head=head)
        anchor = {
            "schema": ANCHOR_SCHEMA,
            "kind": "scratch",
            "head": head,
            "layer": layer,
            "version": version,
            "data_leaf_id": data_id,
            "resident_leaf_id": self._resident_id(data_result),
            "content_digest": digest,
            "data_generation": self._generation(data_result),
            "commit_generation": self._generation(data_result),
        }
        anchor_id = _scratch_anchor_id(head, layer, version)
        self._write_anchor(anchor_id, anchor, head=head)
        stored = self._read_anchor(anchor_id, "scratch", head=head,
                                   layer=layer)
        if stored is None or stored.get("content_digest") != digest:
            raise FieldPersistenceCorruptionError(
                "scratch anchor %s did not commit the requested payload"
                % anchor_id)
        return self._receipt(
            status="committed", accepted=True, created=True, duplicate=False,
            approved=None, kind="scratch", head=head, layer=layer,
            version=version, leaf_id=data_id,
            resident_leaf_id=self._resident_id(data_result),
            content_digest=digest, anchor_leaf_id=anchor_id,
            anchor_generation=stored.get("commit_generation"),
            data_generation=self._generation(data_result),
            generation_before=before["generation"],
            main_field_modified=False,
        )

    def load_scratch(self, head: int, layer: int,
                     version: int | None = None) -> Any:
        """Return a scratch payload, latest by default, or ``None``."""
        head = _validate_head(head)
        layer = _validate_layer(layer)
        if version is None:
            anchor = self._latest_scratch_anchor(head, layer)
        else:
            if isinstance(version, bool) or not isinstance(version, int) \
                    or version < 0 or version >= MAX_SCRATCH_VERSIONS:
                raise FieldPersistenceError(
                    "scratch version is outside [0, %d)"
                    % MAX_SCRATCH_VERSIONS)
            anchor = self._read_anchor(
                _scratch_anchor_id(head, layer, version), "scratch",
                head=head, layer=layer)
        if anchor is None:
            return None
        return self._read_payload(
            anchor, expected_kind="scratch", max_bytes=MAX_SCRATCH_BYTES,
            head=head, layer=layer)

    def checkpoint(self, directory: Any = None) -> Any:
        """Delegate a checkpoint to the underlying universe, if supported."""
        checkpoint = getattr(self.universe, "checkpoint", None)
        if not callable(checkpoint):
            raise FieldPersistenceError(
                "the universe does not expose checkpoint()")
        return checkpoint(directory)

    def status(self) -> dict[str, Any]:
        """Return a small, read-only description of the persisted metadata."""
        field = self._read_anchor(FIELD_ANCHOR_ID, "field")
        scratch = []
        for head in range(HEAD_COUNT):
            for layer in range(MAX_LAYER + 1):
                latest = self._latest_scratch_anchor(head, layer)
                if latest is not None:
                    scratch.append({
                        "head": head, "layer": layer,
                        "version": int(latest.get("version", 0)),
                        "digest": latest.get("content_digest"),
                    })
        return {
            "schema": RECEIPT_SCHEMA,
            "field": None if field is None else {
                "digest": field.get("content_digest"),
                "generation": field.get("commit_generation"),
                "read_only": True,
            },
            "scratch": scratch,
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
        generation = snapshot.get("generation", fallback.get("generation", 0))
        digest = snapshot.get("digest") or snapshot.get("state_digest") \
            or fallback.get("digest") or fallback.get("state_digest") or ""
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 0
        return {"generation": generation, "digest": str(digest)}

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

    def _ingest_leaf(self, leaf_id: str, text: str, *, head: int) -> Any:
        record = {
            "type": "leaf",
            "schema": UNIVERSE_SCHEMA,
            "leaf_id": leaf_id,
            "kind": "data",
            "head": head,
            "text": text,
            "features": [],
            "no_feature": True,
            "source_digest": _sha256_text(text),
            "field_persistence": True,
        }
        result = self.universe.ingest(record)
        if not isinstance(result, Mapping):
            raise FieldPersistenceError(
                "universe.ingest() returned a non-mapping receipt")
        return result

    def _write_anchor(self, anchor_id: str, anchor: Mapping[str, Any],
                      *, head: int) -> Any:
        text, _clean = _canonical_text(
            dict(anchor), label="field anchor", max_bytes=MAX_ANCHOR_BYTES)
        result = self._ingest_leaf(anchor_id, text, head=head)
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
        if isinstance(raw, Mapping) or isinstance(raw, list):
            # A duck-typed fake may return the parsed object directly.
            try:
                raw = json.dumps(
                    raw, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise FieldPersistenceCorruptionError(
                    "%s is not canonical JSON" % label) from exc
        if not isinstance(raw, str):
            raise FieldPersistenceCorruptionError(
                "%s is not a JSON text leaf" % label)
        size = len(raw.encode("utf-8"))
        if size > max_bytes:
            raise FieldPersistenceBoundsError(
                "%s is %d bytes; the limit is %d bytes"
                % (label, size, max_bytes))
        return raw

    def _read_anchor(self, anchor_id: str, kind: str, *,
                     head: int | None = None,
                     layer: int | None = None) -> dict[str, Any] | None:
        leaf = self._get_leaf(anchor_id)
        if leaf is None:
            return None
        text = self._leaf_text(
            leaf, max_bytes=MAX_ANCHOR_BYTES, label="field anchor")
        try:
            anchor = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FieldPersistenceCorruptionError(
                "field anchor %s is not valid JSON" % anchor_id) from exc
        if not isinstance(anchor, dict):
            raise FieldPersistenceCorruptionError(
                "field anchor %s is not an object" % anchor_id)
        if anchor.get("schema") != ANCHOR_SCHEMA or anchor.get("kind") != kind:
            raise FieldPersistenceCorruptionError(
                "field anchor %s has an unexpected schema/kind" % anchor_id)
        if head is not None and anchor.get("head") != head:
            raise FieldPersistenceCorruptionError(
                "field anchor %s belongs to another head" % anchor_id)
        if layer is not None and anchor.get("layer") != layer:
            raise FieldPersistenceCorruptionError(
                "field anchor %s belongs to another layer" % anchor_id)
        digest = anchor.get("content_digest")
        data_id = anchor.get("data_leaf_id")
        if not isinstance(digest, str) or len(digest) != 64 \
                or not isinstance(data_id, str) or not data_id:
            raise FieldPersistenceCorruptionError(
                "field anchor %s is missing its content address" % anchor_id)
        return anchor

    def _read_payload(self, anchor: Mapping[str, Any], *, expected_kind: str,
                      max_bytes: int, head: int | None,
                      layer: int | None) -> Any:
        leaf = self._get_leaf(str(anchor.get("data_leaf_id")))
        if leaf is None:
            resident = anchor.get("resident_leaf_id")
            if isinstance(resident, str):
                leaf = self._get_leaf(resident)
        if leaf is None:
            raise FieldPersistenceCorruptionError(
                "the %s payload referenced by its anchor is missing"
                % expected_kind)
        text = self._leaf_text(
            leaf, max_bytes=max_bytes, label="%s payload" % expected_kind)
        try:
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FieldPersistenceCorruptionError(
                "the %s payload is not valid JSON" % expected_kind) from exc
        expected = _content_digest(
            expected_kind, text, head=head, layer=layer)
        if expected != anchor.get("content_digest"):
            raise FieldPersistenceCorruptionError(
                "the %s payload does not match its content address"
                % expected_kind)
        return payload

    def _latest_scratch_anchor(self, head: int, layer: int
                               ) -> dict[str, Any] | None:
        latest = None
        misses = 0
        for version in range(MAX_SCRATCH_VERSIONS):
            anchor = self._read_anchor(
                _scratch_anchor_id(head, layer, version), "scratch",
                head=head, layer=layer)
            if anchor is None:
                misses += 1
                if misses >= SCRATCH_VERSION_GAP:
                    break
                continue
            latest = anchor
            misses = 0
        return latest

    def _receipt(self, *, status: str, accepted: bool, created: bool,
                 duplicate: bool, approved: bool | None, kind: str,
                 leaf_id: str | None = None,
                 resident_leaf_id: str | None = None,
                 content_digest: str | None = None,
                 anchor_leaf_id: str | None = None,
                 anchor_generation: Any = None,
                 data_generation: Any = None,
                 generation_before: Any = None,
                 head: int | None = None,
                 layer: int | None = None,
                 version: int | None = None,
                 main_field_modified: bool | None = None,
                 **extra: Any) -> FieldReceipt:
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
            "leaf_id": leaf_id,
            "resident_leaf_id": resident_leaf_id,
            "anchor_leaf_id": anchor_leaf_id,
            "content_digest": content_digest,
            "payload_digest": content_digest,
            "digest": state["digest"],
            "state_digest": state["digest"],
            "generation": state["generation"],
            "generation_before": generation_before,
            "anchor_generation": anchor_generation,
            "data_generation": data_generation,
            "head": head,
            "layer": layer,
            "version": version,
            "canonical_commit": False,
            "canonical_state_modified": False,
            "main_field_modified": (
                kind == "field" and bool(created)
                if main_field_modified is None else main_field_modified),
        }
        receipt.update(extra)
        return FieldReceipt(receipt)

    def _denial(self, status: str, reason: str, *, approved: bool,
                **extra: Any) -> FieldReceipt:
        return self._receipt(
            status=status, accepted=False, created=False, duplicate=False,
            approved=approved, kind=extra.pop("kind", "field"),
            reasons=[reason], **extra)


__all__ = (
    "FIELD_SCHEMA", "SCRATCH_SCHEMA", "ANCHOR_SCHEMA", "RECEIPT_SCHEMA",
    "FIELD_ANCHOR_ID", "MAX_FIELD_BYTES", "MAX_SCRATCH_BYTES",
    "MAX_ANCHOR_BYTES", "MAX_JSON_DEPTH", "MAX_JSON_NODES", "MAX_LAYER",
    "MAX_SCRATCH_VERSIONS", "FieldPersistence", "FieldPersistenceError",
    "FieldPersistenceBoundsError", "FieldPersistenceSecretError",
    "FieldPersistencePathError", "FieldPersistenceCorruptionError",
    "FieldReceipt",
)
