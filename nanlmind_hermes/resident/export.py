"""Bounded candidate export, validation and intake for resident universes.

The resident universe is append-only and generation bound. This module is the
only bridge between two universes and it is deliberately conservative:

* :func:`export_candidate` writes a small, self-describing candidate bundle
  containing new relations, leaves, pairs and hyperedges. The full resident
  file is **not** copied unless the caller explicitly passes
  ``include_full=True``.
* :func:`validate_candidate` verifies the bundle without mutating anything. It
  rejects digest mismatches, stale/impossible generations, oversized bundles,
  duplicate ids, dangling relation/leaf references and path/secret leakage.
* :func:`intake_candidate` stages an explicitly approved bundle into a target
  universe using a generation compare-and-swap. It returns a receipt and never
  promotes canonical state or claims canonical authority.
* :class:`MergeGate` keeps canonical promotion disabled by default. An external
  gate object must authorize promotion before that can change. Integrating a
  candidate locally as training corpus or evidence is a separate explicit
  operation (:func:`integrate_local_corpus`).

This module performs no network I/O. A produced bundle is suitable for an
external, explicitly initiated upload; nothing is exfiltrated in the background.

Bundle layout (a directory)::

    manifest.json              canonical manifest, including manifest_digest
    manifest.sha256            SHA-256 of the manifest body
    delta/relations.jsonl      compact canonical JSON records, one per line
    delta/leaves.jsonl
    delta/pairs.jsonl
    delta/hyperedges.jsonl
    full/snapshot.bin          only when include_full=True

Every record is a JSON object with a stable ``id``. Records are sorted by id and
written as compact UTF-8 JSON. Section digests are SHA-256 over the exact section
file bytes, so a receiver does not need to trust the exporter's ordering.

Digests provide corruption/tamper evidence relative to the signed-off manifest;
they are not authentication. Explicit approval is the trust boundary for intake,
and the raw opt-in full snapshot is validated by bounds and digest, not by the
manifest/delta content scanner.

Universe protocol (duck typed)
-----------------------------
``universe_id``, ``generation`` and ``digest`` are required; they may be
properties or zero-argument callables. ``digest_at(generation)`` is used for
the candidate's parent digest. Delta records can be supplied by
``delta_records(since_generation=...)``, ``iter_delta(...)``,
``export_delta(...)`` or ``records_since(...)``. As a fallback the exporter
reads ``iter_relations``/``relations`` (and the same pattern for the other three
sections). ``source_digests`` supplies provenance digests, optionally
``external_references`` lists base ids that a delta is allowed to reference,
and ``snapshot_bytes()`` / ``export_snapshot(path)`` / ``path`` can supply the
opt-in full snapshot.

A target universe implements ``apply_candidate(records, manifest, *,
approved=True, expected_generation=..., target_generation=...,
candidate_digest=...)`` and should expose ``has_candidate(digest)`` (or an
``applied_candidate_digests`` collection) so duplicate intake is idempotent.

If a source universe cannot expose a digest for the requested base generation,
export records the current state digest as a best-effort parent digest rather
than inventing history.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA = "nanlmind.resident.candidate/1"
EXPORT_SCHEMA = "nanlmind.resident.candidate-export/1"
RECEIPT_SCHEMA = "nanlmind.resident.intake-receipt/1"
LOCAL_RECEIPT_SCHEMA = "nanlmind.resident.local-integration-receipt/1"
MERGE_GATE_SCHEMA = "nanlmind.resident.merge-gate/1"

MANIFEST_NAME = "manifest.json"
MANIFEST_DIGEST_NAME = "manifest.sha256"
DELTA_DIR = "delta"
FULL_DIR = "full"
FULL_SNAPSHOT_NAME = "snapshot.bin"
DELTA_FORMAT = "jsonl/1"
SECTIONS = ("relations", "leaves", "pairs", "hyperedges")
HEAD_INDEX_MAX = 255

DEFAULT_MAX_CANDIDATE_BYTES = 32 * 1024 * 1024
FULL_CANDIDATE_BYTES = 1024 * 1024 * 1024 + 32 * 1024 * 1024
MAX_FULL_SNAPSHOT_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SIDECAR_BYTES = 4096
MAX_RECORD_BYTES = 64 * 1024
MAX_STRING_CHARS = 16 * 1024
MAX_RECORDS_PER_SECTION = 100_000
MAX_TOTAL_RECORDS = 200_000
MAX_SOURCE_DIGESTS = 4096
MAX_REFERENCE_IDS = 10_000
MAX_JSON_DEPTH = 32
MAX_REASONS = 100
MAX_GENERATION = 2 ** 63 - 1

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_VALUE_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_SECRET_KEY_RE = re.compile(
    r"^(api[_-]?key|apikey|client[_-]?secret|secret|password|passwd|pwd|"
    r"private[_-]?key|master[_-]?key|access[_-]?token|auth[_-]?token|"
    r"refresh[_-]?token|authorization|credential|credentials|cookie|token)$",
    re.IGNORECASE,
)
_SECRET_KEY_SUFFIX_RE = re.compile(
    r"(^|_)(api[_-]?key|secret|password|passwd|access[_-]?token|"
    r"auth[_-]?token|private[_-]?key|master[_-]?key)$",
    re.IGNORECASE,
)
_PATH_KEY_RE = re.compile(
    r"^(path|file|filename|filepath|file_path|file_name|local_path|source_path|"
    r"abs_path|abspath|directory|dir|uri|url)$",
    re.IGNORECASE,
)
_PATH_KEY_SUFFIX_RE = re.compile(
    r"(^|_)(path|filename|filepath|file_path|directory)$", re.IGNORECASE
)
_SECRET_VALUE_RES = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
)

_MISSING = object()


# ---------------------------------------------------------------------------
# Canonical JSON and small helpers
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON encoding used by every digest."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical manifest body (without ``manifest_digest``)."""
    body = {key: value for key, value in manifest.items()
            if key != "manifest_digest"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON number: %s" % value)


def _no_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: %s" % key)
        result[key] = value
    return result


def _strict_json_loads(raw: bytes | str) -> Any:
    text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
    return json.loads(
        text,
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX_RE.fullmatch(value) is not None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _add(problems: list[str], message: str) -> None:
    if len(problems) < MAX_REASONS:
        problems.append(message)


def _identifier_error(value: Any, label: str) -> str | None:
    if not isinstance(value, str) or not value:
        return "%s must be a non-empty string" % label
    if len(value) > 256:
        return "%s exceeds 256 characters" % label
    if value != value.strip():
        return "%s has leading or trailing whitespace" % label
    if value in {".", ".."}:
        return "%s is a traversal token" % label
    if "/" in value or "\\" in value:
        return "%s must not contain path separators" % label
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return "%s contains a control character" % label
    if _looks_like_path(value):
        return "%s looks like a filesystem path" % label
    if _looks_like_secret(value):
        return "%s looks secret-like" % label
    return None


def _looks_like_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("/", "~/", "\\\\")):
        return True
    if _WINDOWS_ABS_RE.match(stripped):
        return True
    if stripped.lower().startswith("file://"):
        return True
    if _PATH_VALUE_RE.search(stripped):
        return True
    return False


def _looks_like_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_VALUE_RES)


def _check_key_name(key: str, where: str, problems: list[str]) -> None:
    if _SECRET_KEY_RE.match(key) or _SECRET_KEY_SUFFIX_RE.search(key):
        _add(problems, "secret-like field name at %s" % where)
    if _PATH_KEY_RE.match(key) or _PATH_KEY_SUFFIX_RE.search(key):
        _add(problems, "path-like field name at %s" % where)


def _scan_sensitive(value: Any, where: str, problems: list[str],
                    depth: int = 0) -> None:
    """Reject path-like, secret-like and non-finite content recursively."""
    if depth > MAX_JSON_DEPTH:
        _add(problems, "JSON nesting exceeds %d levels at %s"
             % (MAX_JSON_DEPTH, where))
        return
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            _add(problems, "string exceeds %d characters at %s"
                 % (MAX_STRING_CHARS, where))
        if _looks_like_path(value):
            _add(problems, "path-like string at %s" % where)
        if _looks_like_secret(value):
            _add(problems, "secret-like value at %s" % where)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _add(problems, "non-string object key at %s" % where)
                continue
            _check_key_name(key, "%s.%s" % (where, key), problems)
            _scan_sensitive(item, "%s.%s" % (where, key), problems, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_sensitive(item, "%s[%d]" % (where, index), problems, depth + 1)
        return
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _add(problems, "non-finite number at %s" % where)
        return
    _add(problems, "unsupported value type at %s" % where)


def _read_value(target: Any, names: Iterable[str], default: Any = _MISSING) -> Any:
    for name in names:
        try:
            value = getattr(target, name)
        except AttributeError:
            continue
        if callable(value):
            value = value()
        return value
    if default is not _MISSING:
        return default
    raise ValueError("universe does not expose any of: %s" % ", ".join(names))


def _read_optional(target: Any, names: Iterable[str]) -> Any:
    return _read_value(target, names, default=None)


def _read_generation(universe: Any, label: str = "universe") -> int:
    try:
        value = _read_value(universe, ("generation",))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("%s generation is unavailable: %s" % (label, exc))
    if not _is_int(value) or value < 0:
        raise ValueError("%s generation must be a non-negative integer" % label)
    return value


def _read_digest(universe: Any, label: str = "universe") -> str:
    try:
        value = _read_value(
            universe, ("digest", "state_digest", "content_digest", "sha256"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("%s digest is unavailable: %s" % (label, exc))
    if not isinstance(value, str) or not _is_digest(value.lower()):
        raise ValueError("%s digest must be a SHA-256 hex string" % label)
    return value.lower()


def _digest_at(universe: Any, generation: int) -> str | None:
    for name in ("digest_at", "generation_digest"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if not callable(method):
            continue
        try:
            value = method(generation)
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if isinstance(value, str) and _is_digest(value.lower()):
            return value.lower()
        return None
    return None


def _call_with_since(method: Any, since_generation: int) -> Any:
    """Call a delta method with whichever ``since`` parameter it exposes."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(since_generation)
    parameters = signature.parameters
    if "since_generation" in parameters:
        return method(since_generation=since_generation)
    if "since" in parameters:
        return method(since=since_generation)
    positional = [
        parameter for parameter in parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(parameter.kind == parameter.VAR_POSITIONAL
                   for parameter in parameters.values())
    if positional or variadic:
        return method(since_generation)
    return method()


def _call_filtered(method: Any, kwargs: Mapping[str, Any]) -> Any:
    """Call ``method`` with only the keyword arguments it accepts."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**dict(kwargs))
    parameters = signature.parameters
    if any(parameter.kind == parameter.VAR_KEYWORD
           for parameter in parameters.values()):
        return method(**dict(kwargs))
    accepted = {
        name: value for name, value in kwargs.items()
        if name in parameters
        and parameters[name].kind in (parameters[name].POSITIONAL_OR_KEYWORD,
                                      parameters[name].KEYWORD_ONLY)
    }
    return method(**accepted)


# ---------------------------------------------------------------------------
# Merge gate and validation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeGate:
    """Policy boundary around canonical promotion.

    Canonical merge is disabled by default. Even when a caller deliberately
    sets ``canonical_merge_enabled=True``, promotion also requires a real
    external gate object exposing ``authorize_canonical(candidate_digest, *,
    approved=True)``. Local candidate intake and local corpus/evidence
    integration do not grant canonical authority.
    """

    canonical_merge_enabled: bool = False
    external_gate: Any | None = None
    local_integration_requires_approval: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_merge_enabled, bool):
            raise ValueError("canonical_merge_enabled must be a boolean")
        if not isinstance(self.local_integration_requires_approval, bool):
            raise ValueError("local_integration_requires_approval must be a boolean")

    @property
    def canonical_promotion_allowed(self) -> bool:
        authorize = getattr(self.external_gate, "authorize_canonical", None)
        return bool(
            self.canonical_merge_enabled
            and self.external_gate is not None
            and callable(authorize)
        )

    def status(self) -> dict[str, Any]:
        return {
            "schema": MERGE_GATE_SCHEMA,
            "canonical_merge_enabled": bool(self.canonical_merge_enabled),
            "canonical_promotion_allowed": self.canonical_promotion_allowed,
            "external_gate_required": True,
            "external_gate_present": self.external_gate is not None,
            "local_integration_requires_approval": bool(
                self.local_integration_requires_approval),
            "local_integration_is_canonical": False,
            "note": (
                "canonical promotion stays blocked until an external merge "
                "gate authorizes it; this package never promotes canonical "
                "state by itself"
            ),
        }

    def require_canonical(self, candidate_digest: str,
                          *, approved: bool = False) -> dict[str, Any]:
        """Raise ``PermissionError`` unless an external gate authorizes promotion."""
        if not self.canonical_merge_enabled:
            raise PermissionError(
                "canonical merge is disabled (canonical_merge_enabled=False)")
        if self.external_gate is None:
            raise PermissionError(
                "canonical merge requires an external merge gate")
        authorize = getattr(self.external_gate, "authorize_canonical", None)
        if not callable(authorize):
            raise PermissionError(
                "external merge gate does not expose authorize_canonical")
        if approved is not True:
            raise PermissionError(
                "canonical promotion requires explicit approved=True")
        decision = _call_filtered(
            authorize, {"candidate_digest": candidate_digest, "approved": True})
        if decision is True:
            return {"authorized": True}
        if isinstance(decision, Mapping) and decision.get("authorized") is True:
            return dict(decision)
        raise PermissionError("external merge gate did not authorize promotion")


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_candidate`."""

    accepted: bool
    reasons: tuple[str, ...]
    bundle: str | None = None
    manifest_digest: str | None = None
    counts: Mapping[str, int] | None = None
    records: Mapping[str, tuple[dict[str, Any], ...]] | None = None
    manifest: Mapping[str, Any] | None = None
    bytes: int = 0
    include_full: bool = False

    def __bool__(self) -> bool:
        return self.accepted

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "bundle": self.bundle,
            "manifest_digest": self.manifest_digest,
            "counts": dict(self.counts or {}),
            "bytes": self.bytes,
            "include_full": self.include_full,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_MANIFEST_KEYS = {
    "schema", "universe_id", "generation", "parent_generation",
    "parent_digest", "source_digests", "counts", "sections",
    "section_digests", "delta", "references", "include_full",
    "full_snapshot", "manifest_digest", "no_canonical_claim",
    "merge_enabled", "canonical_merge_enabled",
}
_SECTION_META_KEYS = {"file", "count", "bytes", "sha256"}
_FULL_META_KEYS = {"file", "bytes", "sha256"}
_DELTA_KEYS = {"format", "base_generation", "record_count"}
_REFERENCE_KEYS = {"leaves", "relations", "records"}


def _bundle_inventory(root: Path) -> tuple[dict[str, Path], int, list[str]]:
    """Collect regular files and reject symlinks, special files and extras."""
    files: dict[str, Path] = {}
    problems: list[str] = []
    total = 0
    allowed = {MANIFEST_NAME, MANIFEST_DIGEST_NAME}
    allowed.update("%s/%s.jsonl" % (DELTA_DIR, section) for section in SECTIONS)
    allowed.add("%s/%s" % (FULL_DIR, FULL_SNAPSHOT_NAME))
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        for name in list(dirnames):
            child = directory / name
            if child.is_symlink():
                _add(problems, "symlink directory not allowed: %s"
                     % child.relative_to(root).as_posix())
                dirnames.remove(name)
        for name in filenames:
            child = directory / name
            relative = child.relative_to(root).as_posix()
            if child.is_symlink():
                _add(problems, "symlink file not allowed: %s" % relative)
                continue
            if not child.is_file():
                _add(problems, "non-regular bundle entry: %s" % relative)
                continue
            if relative not in allowed:
                _add(problems, "unexpected bundle file: %s" % relative)
            try:
                total += child.stat().st_size
            except OSError as exc:
                _add(problems, "cannot stat bundle file %s: %s" % (relative, exc))
                continue
            files[relative] = child
    return files, total, problems


def _parse_manifest(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if path.is_symlink():
        _add(problems, "manifest.json must not be a symlink")
        return None
    try:
        size = path.stat().st_size
    except OSError as exc:
        _add(problems, "manifest.json is unreadable: %s" % exc)
        return None
    if size > MAX_MANIFEST_BYTES:
        _add(problems, "manifest.json exceeds %d bytes" % MAX_MANIFEST_BYTES)
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _add(problems, "manifest.json is unreadable: %s" % exc)
        return None
    try:
        manifest = _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _add(problems, "manifest.json is not valid strict JSON: %s" % exc)
        return None
    if not isinstance(manifest, dict):
        _add(problems, "manifest.json must contain a JSON object")
        return None
    return manifest


def _validate_manifest_shape(manifest: Mapping[str, Any],
                             problems: list[str]) -> None:
    missing = _MANIFEST_KEYS - set(manifest)
    if missing:
        _add(problems, "manifest is missing required fields: %s"
             % ", ".join(sorted(missing)))
    unknown = set(manifest) - _MANIFEST_KEYS
    if unknown:
        _add(problems, "manifest has unknown fields: %s"
             % ", ".join(sorted(unknown)))

    if manifest.get("schema") != SCHEMA:
        _add(problems, "unsupported candidate schema: %r" % manifest.get("schema"))

    if manifest.get("no_canonical_claim") is not True:
        _add(problems, "no_canonical_claim must be true")
    if manifest.get("merge_enabled") is not False:
        _add(problems, "merge_enabled must be false")
    if manifest.get("canonical_merge_enabled") is not False:
        _add(problems, "canonical_merge_enabled must be false")

    universe_error = _identifier_error(manifest.get("universe_id"), "universe_id")
    if universe_error:
        _add(problems, universe_error)

    for field in ("generation", "parent_generation"):
        value = manifest.get(field)
        if not _is_int(value) or value < 0 or value > MAX_GENERATION:
            _add(problems, "%s must be an integer in [0, %d]"
                 % (field, MAX_GENERATION))
    generation = manifest.get("generation")
    parent = manifest.get("parent_generation")
    if _is_int(generation) and _is_int(parent) and generation < parent:
        _add(problems, "generation must not precede parent_generation")

    if not _is_digest(manifest.get("parent_digest")):
        _add(problems, "parent_digest must be a lowercase SHA-256 hex string")

    sources = manifest.get("source_digests")
    if not isinstance(sources, Mapping):
        _add(problems, "source_digests must be an object")
    elif len(sources) > MAX_SOURCE_DIGESTS:
        _add(problems, "source_digests exceeds %d entries" % MAX_SOURCE_DIGESTS)
    else:
        for key, value in sources.items():
            _check_key_name(str(key), "source_digests.%s" % key, problems)
            error = _identifier_error(key, "source digest id")
            if error:
                _add(problems, error)
            if not _is_digest(value):
                _add(problems, "source digest for %r must be SHA-256 hex" % key)

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        _add(problems, "counts must be an object")
    else:
        expected = set(SECTIONS) | {"total"}
        if set(counts) != expected:
            _add(problems, "counts must contain exactly: %s"
                 % ", ".join(sorted(expected)))
        for key in expected & set(counts):
            value = counts.get(key)
            if not _is_int(value) or value < 0 or value > MAX_TOTAL_RECORDS:
                _add(problems, "count %s is outside [0, %d]"
                     % (key, MAX_TOTAL_RECORDS))

    sections = manifest.get("sections")
    if not isinstance(sections, Mapping):
        _add(problems, "sections must be an object")
    else:
        if set(sections) != set(SECTIONS):
            _add(problems, "sections must contain exactly: %s"
                 % ", ".join(SECTIONS))
        for section in SECTIONS:
            meta = sections.get(section)
            if not isinstance(meta, Mapping):
                _add(problems, "sections.%s must be an object" % section)
                continue
            if set(meta) != _SECTION_META_KEYS:
                _add(problems, "sections.%s must contain exactly: %s"
                     % (section, ", ".join(sorted(_SECTION_META_KEYS))))
            expected_file = "%s/%s.jsonl" % (DELTA_DIR, section)
            if meta.get("file") != expected_file:
                _add(problems, "sections.%s.file must be %s"
                     % (section, expected_file))
            value = meta.get("count")
            if not _is_int(value) or value < 0 or value > MAX_RECORDS_PER_SECTION:
                _add(problems, "sections.%s.count is outside [0, %d]"
                     % (section, MAX_RECORDS_PER_SECTION))
            value = meta.get("bytes")
            if not _is_int(value) or value < 0:
                _add(problems, "sections.%s.bytes must be a non-negative integer"
                     % section)
            if not _is_digest(meta.get("sha256")):
                _add(problems, "sections.%s.sha256 must be SHA-256 hex" % section)

    section_digests = manifest.get("section_digests")
    if not isinstance(section_digests, Mapping):
        _add(problems, "section_digests must be an object")
    else:
        if set(section_digests) != set(SECTIONS):
            _add(problems, "section_digests must contain exactly: %s"
                 % ", ".join(SECTIONS))
        for section in SECTIONS:
            if not _is_digest(section_digests.get(section)):
                _add(problems, "section_digests.%s must be SHA-256 hex" % section)

    delta = manifest.get("delta")
    if not isinstance(delta, Mapping):
        _add(problems, "delta must be an object")
    else:
        if set(delta) != _DELTA_KEYS:
            _add(problems, "delta must contain exactly: %s"
                 % ", ".join(sorted(_DELTA_KEYS)))
        if delta.get("format") != DELTA_FORMAT:
            _add(problems, "delta.format must be %r" % DELTA_FORMAT)
        if delta.get("base_generation") != parent:
            _add(problems, "delta.base_generation must equal parent_generation")
        value = delta.get("record_count")
        if not _is_int(value) or value < 0 or value > MAX_TOTAL_RECORDS:
            _add(problems, "delta.record_count is outside [0, %d]"
                 % MAX_TOTAL_RECORDS)

    references = manifest.get("references", {})
    if not isinstance(references, Mapping):
        _add(problems, "references must be an object")
    else:
        unknown = set(references) - _REFERENCE_KEYS
        if unknown:
            _add(problems, "references has unknown fields: %s"
                 % ", ".join(sorted(unknown)))
        for key in _REFERENCE_KEYS & set(references):
            value = references.get(key)
            if not isinstance(value, list):
                _add(problems, "references.%s must be a list" % key)
                continue
            if len(value) > MAX_REFERENCE_IDS:
                _add(problems, "references.%s exceeds %d ids"
                     % (key, MAX_REFERENCE_IDS))
                continue
            for reference in value:
                error = _identifier_error(reference, "references.%s id" % key)
                if error:
                    _add(problems, error)

    include_full = manifest.get("include_full")
    if not isinstance(include_full, bool):
        _add(problems, "include_full must be a boolean")
    full = manifest.get("full_snapshot")
    if include_full is False:
        if full is not None:
            _add(problems, "full_snapshot must be null when include_full is false")
    elif include_full is True:
        if not isinstance(full, Mapping):
            _add(problems, "full_snapshot is required when include_full is true")
        else:
            if set(full) != _FULL_META_KEYS:
                _add(problems, "full_snapshot must contain exactly: %s"
                     % ", ".join(sorted(_FULL_META_KEYS)))
            expected_file = "%s/%s" % (FULL_DIR, FULL_SNAPSHOT_NAME)
            if full.get("file") != expected_file:
                _add(problems, "full_snapshot.file must be %s" % expected_file)
            value = full.get("bytes")
            if not _is_int(value) or value < 0:
                _add(problems, "full_snapshot.bytes must be a non-negative integer")
            elif value > MAX_FULL_SNAPSHOT_BYTES:
                _add(problems, "full_snapshot exceeds the 1 GiB resident ceiling")
            if not _is_digest(full.get("sha256")):
                _add(problems, "full_snapshot.sha256 must be SHA-256 hex")

    if not _is_digest(manifest.get("manifest_digest")):
        _add(problems, "manifest_digest must be a lowercase SHA-256 hex string")


def _read_section(path: Path, section: str, meta: Mapping[str, Any],
                  problems: list[str]) -> tuple[bytes, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    if path.is_symlink():
        _add(problems, "section file must not be a symlink: %s" % section)
        return b"", records
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _add(problems, "cannot read section %s: %s" % (section, exc))
        return b"", records
    actual_digest = _digest_bytes(raw)
    declared_digest = meta.get("sha256")
    if declared_digest != actual_digest:
        _add(problems, "section digest mismatch: %s" % section)
    declared_bytes = meta.get("bytes")
    if _is_int(declared_bytes) and declared_bytes != len(raw):
        _add(problems, "section byte count mismatch: %s" % section)
    if raw and not raw.endswith(b"\n"):
        _add(problems, "section file must end with a newline: %s" % section)
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) > MAX_RECORDS_PER_SECTION:
        _add(problems, "section %s exceeds %d records"
             % (section, MAX_RECORDS_PER_SECTION))
        lines = lines[:MAX_RECORDS_PER_SECTION]
    for index, line in enumerate(lines):
        if not line:
            _add(problems, "blank line in section %s at line %d"
                 % (section, index + 1))
            continue
        if len(line) > MAX_RECORD_BYTES:
            _add(problems, "record exceeds %d bytes in %s at line %d"
                 % (MAX_RECORD_BYTES, section, index + 1))
            continue
        try:
            record = _strict_json_loads(line)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            _add(problems, "invalid JSON record in %s at line %d: %s"
                 % (section, index + 1, exc))
            continue
        if not isinstance(record, dict):
            _add(problems, "record must be a JSON object in %s at line %d"
                 % (section, index + 1))
            continue
        records.append(record)
    declared_count = meta.get("count")
    if _is_int(declared_count) and declared_count != len(lines):
        _add(problems, "section record count mismatch: %s" % section)
    return raw, records


def _nonempty_text(record: Mapping[str, Any], field: str, section: str,
                   index: int, problems: list[str]) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        _add(problems, "%s record %d requires non-empty %s"
             % (section, index + 1, field))
    elif len(value) > MAX_STRING_CHARS:
        _add(problems, "%s record %d %s is too long"
             % (section, index + 1, field))


def _reference_list(record: Mapping[str, Any], field: str, section: str,
                    index: int, problems: list[str]) -> list[str]:
    value = record.get(field)
    if value is None:
        return []
    if not isinstance(value, list):
        _add(problems, "%s record %d %s must be a list"
             % (section, index + 1, field))
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        error = _identifier_error(item, "%s record %d %s id"
                                  % (section, index + 1, field))
        if error:
            _add(problems, error)
            continue
        if item in seen:
            _add(problems, "duplicate reference in %s record %d %s"
                 % (section, index + 1, field))
            continue
        seen.add(item)
        result.append(item)
    return result


def _validate_record(record: Mapping[str, Any], section: str, index: int,
                     source_digest_values: set[str],
                     problems: list[str]) -> str | None:
    ident = record.get("id")
    error = _identifier_error(ident, "%s record %d id" % (section, index + 1))
    if error:
        _add(problems, error)
        ident = None

    if "source_digest" in record:
        value = record.get("source_digest")
        if not _is_digest(value):
            _add(problems, "%s record %d source_digest is not SHA-256 hex"
                 % (section, index + 1))
        elif value not in source_digest_values:
            _add(problems, "%s record %d cites an undeclared source digest"
                 % (section, index + 1))

    if "confidence" in record:
        confidence = record.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            _add(problems, "%s record %d confidence must be in [0, 1]"
                 % (section, index + 1))

    if section == "relations":
        for field in ("subject", "predicate", "object"):
            _nonempty_text(record, field, section, index, problems)
    elif section == "pairs":
        for field in ("left", "right"):
            value = record.get(field)
            if isinstance(value, bool):
                _add(problems, "pair record %d %s is invalid"
                     % (index + 1, field))
            elif isinstance(value, int):
                if not 0 <= value <= HEAD_INDEX_MAX:
                    _add(problems, "pair record %d %s head index is out of range"
                         % (index + 1, field))
            elif isinstance(value, str):
                if not value.strip():
                    _add(problems, "pair record %d %s must be non-empty"
                         % (index + 1, field))
            else:
                _add(problems, "pair record %d requires %s"
                     % (index + 1, field))
    elif section == "hyperedges":
        members = record.get("members")
        if members is None:
            members = record.get("nodes")
        if members is None:
            members = record.get("member_ids")
        if not isinstance(members, list) or not members:
            _add(problems, "hyperedge record %d requires a non-empty members list"
                 % (index + 1))
        else:
            seen: set[Any] = set()
            for member in members:
                if isinstance(member, bool):
                    _add(problems, "hyperedge record %d has an invalid member"
                         % (index + 1))
                    continue
                if isinstance(member, int):
                    if not 0 <= member <= HEAD_INDEX_MAX:
                        _add(problems, "hyperedge record %d member index is out of range"
                             % (index + 1))
                    key: Any = member
                elif isinstance(member, str):
                    if not member.strip():
                        _add(problems, "hyperedge record %d has an empty member"
                             % (index + 1))
                    key = member
                else:
                    _add(problems, "hyperedge record %d has an invalid member type"
                         % (index + 1))
                    continue
                if key in seen:
                    _add(problems, "hyperedge record %d has a duplicate member"
                         % (index + 1))
                seen.add(key)
    return ident


def _validate_references(records: Mapping[str, list[dict[str, Any]]],
                         references: Mapping[str, Any],
                         problems: list[str]) -> None:
    external_leaves: set[str] = set()
    external_relations: set[str] = set()
    external_any: set[str] = set()
    all_external: set[str] = set()
    for key in ("leaves", "relations", "records"):
        values = references.get(key, []) if isinstance(references, Mapping) else []
        if not isinstance(values, list):
            continue
        target = (external_leaves if key == "leaves"
                  else external_relations if key == "relations"
                  else external_any)
        for value in values:
            if not isinstance(value, str):
                continue
            if value in all_external:
                _add(problems, "duplicate external reference id: %s" % value)
                continue
            all_external.add(value)
            target.add(value)

    candidate_ids: set[str] = set()
    for section in SECTIONS:
        for record in records.get(section, []):
            ident = record.get("id")
            if isinstance(ident, str):
                candidate_ids.add(ident)
    colliding = candidate_ids & all_external
    for ident in sorted(colliding):
        _add(problems, "external reference duplicates candidate id: %s" % ident)

    known_any = candidate_ids | all_external
    known_leaves = {record.get("id") for record in records.get("leaves", [])
                    if isinstance(record.get("id"), str)} | external_leaves
    known_relations = {
        record.get("id") for record in records.get("relations", [])
        if isinstance(record.get("id"), str)
    } | external_relations

    for section in ("relations", "leaves", "pairs", "hyperedges"):
        for index, record in enumerate(records.get(section, [])):
            ident = record.get("id")
            label = ident if isinstance(ident, str) else "#%d" % (index + 1)
            if section == "relations":
                for field in ("leaf_id", "evidence_leaf_id", "source_leaf_id"):
                    reference = record.get(field)
                    if reference is None:
                        continue
                    if not isinstance(reference, str):
                        _add(problems, "relation %s %s must be a string"
                             % (label, field))
                    elif reference not in known_leaves:
                        _add(problems, "relation %s references unknown leaf: %s"
                             % (label, reference))
                for field in ("leaf_ids", "evidence_leaf_ids"):
                    for reference in _reference_list(
                            record, field, section, index, problems):
                        if reference not in known_leaves:
                            _add(problems, "relation %s references unknown leaf: %s"
                                 % (label, reference))
            elif section == "leaves":
                for field in ("relation_id", "source_relation_id"):
                    reference = record.get(field)
                    if reference is None:
                        continue
                    if not isinstance(reference, str):
                        _add(problems, "leaf %s %s must be a string"
                             % (label, field))
                    elif reference not in known_relations:
                        _add(problems, "leaf %s references unknown relation: %s"
                             % (label, reference))
                for field in ("relation_ids", "source_relation_ids"):
                    for reference in _reference_list(
                            record, field, section, index, problems):
                        if reference not in known_relations:
                            _add(problems, "leaf %s references unknown relation: %s"
                                 % (label, reference))
            elif section == "pairs":
                for field in ("left", "right"):
                    reference = record.get(field)
                    if isinstance(reference, str) and reference not in known_leaves:
                        _add(problems, "pair %s references unknown leaf: %s"
                             % (label, reference))
                for reference in _reference_list(
                        record, "leaf_ids", section, index, problems):
                    if reference not in known_leaves:
                        _add(problems, "pair %s references unknown leaf: %s"
                             % (label, reference))
            else:
                members = record.get("members")
                if members is None:
                    members = record.get("nodes")
                if members is None:
                    members = record.get("member_ids")
                if isinstance(members, list):
                    for member in members:
                        if isinstance(member, str) and member not in known_any:
                            _add(problems, "hyperedge %s references unknown id: %s"
                                 % (label, member))


def validate_candidate(
    bundle: str | os.PathLike[str],
    max_bytes: int = DEFAULT_MAX_CANDIDATE_BYTES,
) -> ValidationResult:
    """Validate a candidate bundle without mutating any universe.

    ``max_bytes`` bounds the whole bundle (manifest, sidecar, delta sections and
    opt-in full snapshot). The default is deliberately small: validating a full
    snapshot requires the caller to raise the bound explicitly.

    Content scanning rejects known path-like and secret-like field names and
    values. It is a guardrail, not a complete data-loss-prevention scanner.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    problems: list[str] = []
    try:
        path = Path(bundle).expanduser()
    except TypeError:
        return ValidationResult(False, ("bundle must be a filesystem path",))
    if not path.exists():
        return ValidationResult(False, ("bundle does not exist: %s" % path,),
                                bundle=str(path))
    if path.is_symlink():
        return ValidationResult(False, ("bundle path must not be a symlink",),
                                bundle=str(path))
    if path.is_dir():
        root = path
        manifest_path = root / MANIFEST_NAME
    elif path.is_file():
        if path.name != MANIFEST_NAME:
            return ValidationResult(
                False, ("bundle file must be named %s" % MANIFEST_NAME,),
                bundle=str(path))
        root = path.parent
        manifest_path = path
    else:
        return ValidationResult(False, ("bundle is not a file or directory",),
                                bundle=str(path))
    if not manifest_path.is_file():
        return ValidationResult(False, ("bundle is missing manifest.json",),
                                bundle=str(root))

    files, total_bytes, inventory_problems = _bundle_inventory(root)
    for problem in inventory_problems:
        _add(problems, problem)

    manifest = _parse_manifest(manifest_path, problems)
    if manifest is None:
        return ValidationResult(
            False, tuple(problems[:MAX_REASONS]), bundle=str(root), bytes=total_bytes)

    _validate_manifest_shape(manifest, problems)
    declared_digest = manifest.get("manifest_digest")
    if _is_digest(declared_digest):
        try:
            computed = compute_manifest_digest(manifest)
        except (TypeError, ValueError) as exc:
            _add(problems, "manifest cannot be canonicalized: %s" % exc)
        else:
            if computed != declared_digest:
                _add(problems, "manifest digest mismatch")

    sidecar = files.get(MANIFEST_DIGEST_NAME)
    if sidecar is not None:
        if sidecar.stat().st_size > MAX_SIDECAR_BYTES:
            _add(problems, "manifest.sha256 is unreasonably large")
        else:
            try:
                tokens = sidecar.read_text(encoding="utf-8").split()
            except (OSError, UnicodeDecodeError) as exc:
                _add(problems, "manifest.sha256 is unreadable: %s" % exc)
            else:
                if not tokens or tokens[0] != declared_digest:
                    _add(problems, "manifest.sha256 digest mismatch")

    counts = manifest.get("counts")
    records: dict[str, list[dict[str, Any]]] = {section: [] for section in SECTIONS}
    sections = manifest.get("sections")
    section_digests = manifest.get("section_digests")
    if total_bytes > max_bytes:
        _add(problems, "bundle exceeds max_bytes: %d > %d"
             % (total_bytes, max_bytes))
        return ValidationResult(
            False,
            tuple(problems[:MAX_REASONS]),
            bundle=str(root),
            manifest_digest=declared_digest if _is_digest(declared_digest) else None,
            counts=counts if isinstance(counts, Mapping) else None,
            records={section: tuple() for section in SECTIONS},
            manifest=manifest,
            bytes=total_bytes,
            include_full=bool(manifest.get("include_full") is True),
        )

    if isinstance(sections, Mapping):
        for section in SECTIONS:
            meta = sections.get(section)
            if not isinstance(meta, Mapping):
                continue
            expected_file = "%s/%s.jsonl" % (DELTA_DIR, section)
            if meta.get("file") != expected_file:
                continue
            section_path = files.get(expected_file)
            if section_path is None:
                _add(problems, "bundle is missing section file: %s" % expected_file)
                continue
            raw, section_records = _read_section(
                section_path, section, meta, problems)
            records[section] = section_records
            actual_digest = _digest_bytes(raw)
            if isinstance(section_digests, Mapping):
                declared = section_digests.get(section)
                if declared != actual_digest:
                    _add(problems, "section digest map mismatch: %s" % section)
            if len(raw) > max_bytes:
                _add(problems, "section %s exceeds max_bytes" % section)

    actual_counts = {section: len(records.get(section, [])) for section in SECTIONS}
    actual_total = sum(actual_counts.values())
    if isinstance(counts, Mapping):
        for section in SECTIONS:
            value = counts.get(section)
            if _is_int(value) and value != actual_counts[section]:
                _add(problems, "count mismatch for %s: manifest=%d actual=%d"
                     % (section, value, actual_counts[section]))
        value = counts.get("total")
        if _is_int(value) and value != actual_total:
            _add(problems, "total count mismatch: manifest=%d actual=%d"
                 % (value, actual_total))
    if actual_total > MAX_TOTAL_RECORDS:
        _add(problems, "candidate exceeds %d total records" % MAX_TOTAL_RECORDS)

    delta = manifest.get("delta")
    if isinstance(delta, Mapping):
        value = delta.get("record_count")
        if _is_int(value) and value != actual_total:
            _add(problems, "delta.record_count mismatch: manifest=%d actual=%d"
                 % (value, actual_total))

    references = manifest.get("references", {})
    if not isinstance(references, Mapping):
        references = {}
    source_digests = manifest.get("source_digests")
    if not isinstance(source_digests, Mapping):
        source_digests = {}
    source_digest_values = {
        value for value in source_digests.values() if _is_digest(value)}

    seen_ids: dict[str, str] = {}
    for section in SECTIONS:
        for index, record in enumerate(records.get(section, [])):
            _scan_sensitive(record, "%s[%d]" % (section, index + 1), problems)
            ident = _validate_record(
                record, section, index, source_digest_values, problems)
            if ident is None:
                continue
            if ident in seen_ids:
                _add(problems, "duplicate id %s in %s (already in %s)"
                     % (ident, section, seen_ids[ident]))
            else:
                seen_ids[ident] = section

    _validate_references(records, references, problems)

    full = manifest.get("full_snapshot")
    full_path = files.get("%s/%s" % (FULL_DIR, FULL_SNAPSHOT_NAME))
    if manifest.get("include_full") is True:
        if isinstance(full, Mapping):
            if full_path is None:
                _add(problems, "bundle is missing full/snapshot.bin")
            elif full_path.is_symlink():
                _add(problems, "full snapshot must not be a symlink")
            else:
                try:
                    size = full_path.stat().st_size
                except OSError as exc:
                    _add(problems, "cannot stat full snapshot: %s" % exc)
                else:
                    if _is_int(full.get("bytes")) and full.get("bytes") != size:
                        _add(problems, "full snapshot byte count mismatch")
                    if size > MAX_FULL_SNAPSHOT_BYTES:
                        _add(problems, "full snapshot exceeds the 1 GiB resident ceiling")
                    try:
                        actual_digest = _digest_file(full_path)
                    except OSError as exc:
                        _add(problems, "cannot hash full snapshot: %s" % exc)
                    else:
                        if full.get("sha256") != actual_digest:
                            _add(problems, "full snapshot digest mismatch")
    elif full_path is not None:
        _add(problems, "full snapshot present but include_full is false")

    accepted = not problems
    reason_tuple = tuple(problems[:MAX_REASONS])
    if len(problems) > MAX_REASONS:
        reason_tuple = reason_tuple + ("additional problems omitted",)
    return ValidationResult(
        accepted=accepted,
        reasons=reason_tuple,
        bundle=str(root),
        manifest_digest=declared_digest if _is_digest(declared_digest) else None,
        counts={**actual_counts, "total": actual_total},
        records={section: tuple(records.get(section, ())) for section in SECTIONS},
        manifest=manifest,
        bytes=total_bytes,
        include_full=bool(manifest.get("include_full") is True),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _normalize_section_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "id" in value:
            return [value]
        return list(value.values())
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("section records must be a collection of objects")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError("section records are not iterable: %s" % exc)


def _normalize_delta_result(result: Any) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {section: [] for section in SECTIONS}
    if result is None:
        return groups
    if isinstance(result, Mapping):
        if "records" in result and not any(section in result for section in SECTIONS):
            return _normalize_delta_result(result["records"])
        if "sections" in result and not any(section in result for section in SECTIONS):
            return _normalize_delta_result(result["sections"])
        unknown = set(result) - set(SECTIONS)
        if unknown:
            raise ValueError("delta result has unknown sections: %s"
                             % ", ".join(sorted(map(str, unknown))))
        for section in SECTIONS:
            groups[section].extend(_normalize_section_value(result.get(section)))
        return groups
    if isinstance(result, (str, bytes, bytearray)):
        raise ValueError("delta result must be a mapping or record iterable")
    for item in result:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[0], str) and item[0] in SECTIONS):
            groups[item[0]].append(item[1])
            continue
        if isinstance(item, Mapping) and item.get("section") in SECTIONS:
            section = str(item["section"])
            record = dict(item)
            record.pop("section", None)
            groups[section].append(record)
            continue
        raise ValueError(
            "delta iterable items must be (section, record) pairs or "
            "records carrying a valid section")
    return groups


def _collect_delta(universe: Any, since_generation: int) -> dict[str, list[Any]]:
    for name in ("delta_records", "iter_delta", "export_delta", "records_since"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if not callable(method):
            continue
        return _normalize_delta_result(_call_with_since(method, since_generation))

    groups: dict[str, list[Any]] = {section: [] for section in SECTIONS}
    records_container = getattr(universe, "records", None)
    for section in SECTIONS:
        value: Any = None
        found = False
        for name in ("iter_%s" % section, "%s_records" % section,
                     "pending_%s" % section):
            if not hasattr(universe, name):
                continue
            method = getattr(universe, name)
            if callable(method):
                try:
                    value = _call_with_since(method, since_generation)
                except TypeError:
                    value = method()
                found = True
                break
        if not found:
            candidate = getattr(universe, section, _MISSING)
            if candidate is not _MISSING:
                value = (_call_with_since(candidate, since_generation)
                         if callable(candidate) else candidate)
                found = True
        if not found and records_container is not None:
            if callable(records_container):
                try:
                    value = _call_filtered(
                        records_container, {"section": section})
                except TypeError:
                    value = None
                found = value is not None
            elif isinstance(records_container, Mapping) and section in records_container:
                value = records_container[section]
                found = True
        if found:
            groups[section] = _normalize_section_value(value)
    return groups


def _record_identity(record: Mapping[str, Any]) -> str:
    for field in ("id", "record_id", "event_id", "leaf_id", "relation_id",
                  "pair_id", "hyperedge_id", "digest"):
        if field in record and record.get(field) is not None:
            value = record[field]
            if isinstance(value, str):
                return value
            if _is_int(value):
                return str(value)
            raise ValueError("record id field %s is not a string" % field)
    raise ValueError("record is missing a stable id")


def _normalize_source_digests(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if callable(value):
        value = value()
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        items: Iterable[Any] = value.items()
        for key, digest in items:
            key_text = str(key)
            if not isinstance(digest, str) or not _is_digest(digest.lower()):
                raise ValueError("source digest for %r is not SHA-256 hex" % key_text)
            if key_text in result:
                raise ValueError("duplicate source digest id: %s" % key_text)
            result[key_text] = digest.lower()
        return dict(sorted(result.items()))
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("source_digests must be a mapping or collection")
    for index, item in enumerate(value):
        if isinstance(item, str):
            if not _is_digest(item.lower()):
                raise ValueError("source digest is not SHA-256 hex")
            key_text = "source-%d" % index
            result[key_text] = item.lower()
            continue
        if not isinstance(item, Mapping):
            raise ValueError("source digest entries must be strings or objects")
        key_text = (item.get("id") or item.get("source") or item.get("name"))
        digest = item.get("sha256") or item.get("digest")
        if not isinstance(key_text, str) or not key_text:
            raise ValueError("source digest entry is missing an id")
        if not isinstance(digest, str) or not _is_digest(digest.lower()):
            raise ValueError("source digest for %r is not SHA-256 hex" % key_text)
        if key_text in result:
            raise ValueError("duplicate source digest id: %s" % key_text)
        result[key_text] = digest.lower()
    return dict(sorted(result.items()))


def _normalize_external_references(value: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {key: [] for key in _REFERENCE_KEYS}
    if value is None:
        return result
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        unknown = set(value) - _REFERENCE_KEYS
        if unknown:
            raise ValueError("external references have unknown sections: %s"
                             % ", ".join(sorted(map(str, unknown))))
        entries = [(key, value.get(key, [])) for key in _REFERENCE_KEYS]
    elif isinstance(value, (str, bytes, bytearray)):
        raise ValueError("external references must be a mapping")
    else:
        grouped: dict[str, list[str]] = {key: [] for key in _REFERENCE_KEYS}
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("external references must be (section, id) pairs")
            section, ident = item
            if section not in _REFERENCE_KEYS:
                raise ValueError("unknown external reference section: %r" % section)
            grouped[section].append(ident)
        entries = list(grouped.items())

    for key, values in entries:
        if isinstance(values, (str, bytes, bytearray)):
            values = [values]
        try:
            items = list(values)
        except TypeError as exc:
            raise ValueError("external references.%s is not iterable" % key) from exc
        if len(items) > MAX_REFERENCE_IDS:
            raise ValueError("external references.%s exceeds %d ids"
                             % (key, MAX_REFERENCE_IDS))
        seen: set[str] = set()
        for ident in items:
            error = _identifier_error(ident, "external reference id")
            if error:
                raise ValueError(error)
            if ident in seen:
                raise ValueError("duplicate external reference id: %s" % ident)
            seen.add(ident)
            result[key].append(ident)
        result[key].sort()
    return result


def _write_snapshot(universe: Any, root: Path) -> dict[str, Any]:
    relative = "%s/%s" % (FULL_DIR, FULL_SNAPSHOT_NAME)
    destination = root / FULL_DIR / FULL_SNAPSHOT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrote = False

    for name in ("export_snapshot", "write_snapshot", "save_snapshot",
                 "snapshot_to_path"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if not callable(method):
            continue
        _call_filtered(method, {"path": destination, "destination": destination})
        if destination.is_file() and not destination.is_symlink():
            wrote = True
            break

    if not wrote:
        data = _read_optional(
            universe, ("snapshot_bytes", "export_snapshot_bytes", "to_bytes"))
        if isinstance(data, (bytes, bytearray)):
            destination.write_bytes(bytes(data))
            wrote = True

    if not wrote:
        source = _read_optional(
            universe, ("snapshot_path", "path", "file_path"))
        if source is not None:
            source_path = Path(source).expanduser()
            if source_path.is_file() and not source_path.is_symlink():
                with source_path.open("rb") as reader:
                    with destination.open("wb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                wrote = True

    if not wrote or not destination.is_file() or destination.is_symlink():
        raise ValueError(
            "include_full=True requires snapshot_bytes(), export_snapshot(path) "
            "or an existing snapshot path")
    size = destination.stat().st_size
    return {
        "file": relative,
        "bytes": size,
        "sha256": _digest_file(destination),
    }


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if destination.is_symlink():
            raise ValueError("destination must not be a symlink")
        if not destination.is_dir():
            raise FileExistsError("destination exists and is not a directory: %s"
                                  % destination)
        try:
            next(destination.iterdir())
        except StopIteration:
            return
        raise FileExistsError("destination is not empty: %s" % destination)


def export_candidate(
    universe: Any,
    destination: str | os.PathLike[str],
    *,
    include_full: bool = False,
    since_generation: int = 0,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Write a bounded candidate bundle for ``universe``.

    ``include_full`` is opt-in. By default only the compact delta sections are
    written, so a multi-hundred-megabyte resident file is never uploaded
    implicitly. ``since_generation`` selects the delta base (default 0, i.e. a
    compact candidate built from the universe's supplied record history).
    """
    if not isinstance(include_full, bool):
        raise ValueError("include_full must be a boolean")
    if isinstance(since_generation, bool) or not isinstance(since_generation, int):
        raise ValueError("since_generation must be a non-negative integer")
    if since_generation < 0:
        raise ValueError("since_generation must be a non-negative integer")
    if max_bytes is None:
        max_bytes = FULL_CANDIDATE_BYTES if include_full else DEFAULT_MAX_CANDIDATE_BYTES
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    generation = _read_generation(universe)
    if since_generation > generation:
        raise ValueError("since_generation must not exceed universe generation")
    universe_id = _read_value(universe, ("universe_id", "id"))
    if callable(universe_id):
        universe_id = universe_id()
    identifier_error = _identifier_error(universe_id, "universe_id")
    if identifier_error:
        raise ValueError(identifier_error)
    current_digest = _read_digest(universe)

    parent_digest = _digest_at(universe, since_generation)
    if parent_digest is None:
        if since_generation == generation:
            parent_digest = current_digest
        else:
            parent_digest = _read_optional(
                universe, ("parent_digest", "base_digest"))
            if (parent_digest is None
                    or not isinstance(parent_digest, str)
                    or not _is_digest(parent_digest.lower())):
                genesis = _read_optional(universe, ("genesis_digest",))
                if (since_generation == 0 and isinstance(genesis, str)
                        and _is_digest(genesis.lower())):
                    parent_digest = genesis.lower()
                else:
                    parent_digest = current_digest
            else:
                parent_digest = parent_digest.lower()

    source_digests = _normalize_source_digests(
        _read_optional(universe, ("source_digests", "provenance_digests")))
    external_references = _normalize_external_references(
        _read_optional(universe, ("external_references", "external_refs")))

    groups = _collect_delta(universe, since_generation)
    normalized: dict[str, list[tuple[str, bytes]]] = {
        section: [] for section in SECTIONS}
    seen_ids: dict[str, str] = {}
    total_records = 0
    for section in SECTIONS:
        for item in groups.get(section, []):
            if not isinstance(item, Mapping):
                raise ValueError("%s record must be a JSON object" % section)
            record = dict(item)
            if "section" in record:
                tagged = record.pop("section")
                if tagged != section:
                    raise ValueError("record section tag does not match %s" % section)
            ident = _record_identity(record)
            record["id"] = ident
            if ident in seen_ids:
                raise ValueError("duplicate record id %s in %s and %s"
                                 % (ident, seen_ids[ident], section))
            seen_ids[ident] = section
            source_digest = record.get("source_digest")
            if source_digest is not None:
                if not isinstance(source_digest, str) or not _is_digest(
                        source_digest.lower()):
                    raise ValueError("%s record %s source_digest is not SHA-256 hex"
                                     % (section, ident))
                source_digest = source_digest.lower()
                record["source_digest"] = source_digest
                if source_digest not in source_digests.values():
                    source_digests["record-source:%s" % source_digest] = source_digest
            try:
                encoded = canonical_json(record)
            except (TypeError, ValueError) as exc:
                raise ValueError("%s record %s is not canonical JSON: %s"
                                 % (section, ident, exc)) from exc
            if len(encoded) > MAX_RECORD_BYTES:
                raise ValueError("%s record %s exceeds %d bytes"
                                 % (section, ident, MAX_RECORD_BYTES))
            normalized[section].append((ident, encoded))

    source_digests = dict(sorted(source_digests.items()))
    section_meta: dict[str, dict[str, Any]] = {}
    section_digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    payloads: dict[str, bytes] = {}
    for section in SECTIONS:
        entries = sorted(normalized[section], key=lambda item: item[0])
        if len(entries) > MAX_RECORDS_PER_SECTION:
            raise ValueError("section %s exceeds %d records"
                             % (section, MAX_RECORDS_PER_SECTION))
        payload = b"".join(line + b"\n" for _, line in entries)
        digest = _digest_bytes(payload)
        relative = "%s/%s.jsonl" % (DELTA_DIR, section)
        payloads[relative] = payload
        section_meta[section] = {
            "file": relative,
            "count": len(entries),
            "bytes": len(payload),
            "sha256": digest,
        }
        section_digests[section] = digest
        counts[section] = len(entries)
        total_records += len(entries)
    counts["total"] = total_records
    if total_records > MAX_TOTAL_RECORDS:
        raise ValueError("candidate exceeds %d total records" % MAX_TOTAL_RECORDS)

    destination_path = Path(destination).expanduser()
    _prepare_destination(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=".resident-candidate-", dir=str(destination_path.parent)))
    committed = False
    try:
        for relative, payload in payloads.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        full_metadata = _write_snapshot(universe, temporary) if include_full else None
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "universe_id": universe_id,
            "generation": generation,
            "parent_generation": since_generation,
            "parent_digest": parent_digest,
            "source_digests": source_digests,
            "counts": counts,
            "sections": section_meta,
            "section_digests": section_digests,
            "delta": {
                "format": DELTA_FORMAT,
                "base_generation": since_generation,
                "record_count": total_records,
            },
            "references": external_references,
            "include_full": include_full,
            "full_snapshot": full_metadata,
            "no_canonical_claim": True,
            "merge_enabled": False,
            "canonical_merge_enabled": False,
        }
        manifest["manifest_digest"] = compute_manifest_digest(manifest)
        manifest_bytes = canonical_json(manifest) + b"\n"
        (temporary / MANIFEST_NAME).write_bytes(manifest_bytes)
        (temporary / MANIFEST_DIGEST_NAME).write_text(
            manifest["manifest_digest"] + "\n", encoding="utf-8")

        validation = validate_candidate(temporary, max_bytes=max_bytes)
        if not validation.accepted:
            raise ValueError(
                "candidate export failed validation: %s"
                % "; ".join(validation.reasons))
        if destination_path.exists():
            destination_path.rmdir()
        os.replace(temporary, destination_path)
        committed = True

        return {
            "schema": EXPORT_SCHEMA,
            "bundle": str(destination_path),
            "manifest_digest": manifest["manifest_digest"],
            "universe_id": universe_id,
            "generation": generation,
            "parent_generation": since_generation,
            "parent_digest": parent_digest,
            "counts": counts,
            "bytes": validation.bytes,
            "include_full": include_full,
            "canonical": False,
            "canonical_authority": False,
            "canonical_commit": False,
            "merge_enabled": False,
            "note": (
                "candidate bundle only; no canonical state is promoted and the "
                "full resident file is included only on explicit opt-in"
            ),
        }
    finally:
        if not committed:
            shutil.rmtree(temporary, ignore_errors=True)


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


def _candidate_already_applied(universe: Any, candidate_digest: str) -> bool:
    for name in ("has_candidate", "candidate_applied", "has_applied_candidate"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if not callable(method):
            continue
        try:
            return bool(method(candidate_digest))
        except TypeError:
            continue
    for name in ("applied_candidate_digests", "applied_candidates",
                 "candidate_digests"):
        if not hasattr(universe, name):
            continue
        value = getattr(universe, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, Mapping):
            if candidate_digest in value:
                return True
        elif isinstance(value, (set, frozenset, list, tuple)):
            if candidate_digest in value:
                return True
    last = _read_optional(universe, ("last_candidate_digest",))
    return last == candidate_digest


def _apply_method(universe: Any) -> Any:
    for name in ("apply_candidate", "ingest_candidate", "stage_candidate",
                 "apply_candidate_delta"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if callable(method):
            return method
    return None


def _local_integration_method(universe: Any) -> Any:
    for name in ("integrate_local_corpus", "integrate_local", "stage_corpus",
                 "ingest_evidence"):
        if not hasattr(universe, name):
            continue
        method = getattr(universe, name)
        if callable(method):
            return method
    return None


def _receipt(
    *,
    status: str,
    accepted: bool,
    applied: bool,
    approved: bool,
    gate: MergeGate,
    reasons: Iterable[str] = (),
    validation: ValidationResult | None = None,
    target_universe_id: str | None = None,
    target_generation_before: int | None = None,
    target_generation_after: int | None = None,
    applied_records: int = 0,
    duplicate: bool = False,
) -> dict[str, Any]:
    manifest = validation.manifest if validation is not None else None
    manifest = manifest if isinstance(manifest, Mapping) else {}
    digest = (validation.manifest_digest if validation is not None else None)
    parent_generation = manifest.get("parent_generation")
    generation = manifest.get("generation")
    counts = dict(validation.counts or {}) if validation is not None else {}
    expected = parent_generation if _is_int(parent_generation) else None
    observed = target_generation_before
    swap_ok = (observed is not None and expected is not None and observed == expected)
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "accepted": bool(accepted),
        "applied": bool(applied),
        "duplicate": bool(duplicate),
        "already_applied": bool(duplicate),
        "approved": approved is True,
        "candidate_digest": digest,
        "manifest_digest": digest,
        "universe_id": manifest.get("universe_id"),
        "target_universe_id": target_universe_id,
        "generation": generation,
        "parent_generation": parent_generation,
        "target_generation_before": target_generation_before,
        "target_generation_after": target_generation_after,
        "generation_compare_and_swap": {
            "expected": expected,
            "observed": observed,
            "result_generation": target_generation_after,
            "ok": swap_ok,
        },
        "counts": counts,
        "applied_records": int(applied_records),
        "reasons": list(reasons),
        "canonical": False,
        "canonical_merge": False,
        "canonical_authority": False,
        "canonical_merge_enabled": bool(gate.canonical_merge_enabled),
        "merge_enabled": False,
        "promotion_allowed": False,
        "canonical_commit": False,
        "canonical_state_modified": False,
        "six_runner_barrier": "absent",
        "mode": "local_candidate_staging",
        "external_gate_required": True,
        "merge_gate": gate.status(),
        "note": (
            "intake stages a candidate locally; it does not merge canonical "
            "state and does not claim canonical authority"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def intake_candidate(
    target_universe: Any,
    bundle: str | os.PathLike[str],
    approved: bool = False,
    *,
    gate: MergeGate | None = None,
    max_bytes: int = DEFAULT_MAX_CANDIDATE_BYTES,
) -> dict[str, Any]:
    """Stage an approved candidate into ``target_universe``.

    Approval must be the literal ``True``. The target must be at the candidate's
    ``parent_generation`` (compare-and-swap) and must expose an application
    method such as ``apply_candidate``. Replayed bundles are reported as
    duplicates without writing records. The receipt is always non-canonical.
    """
    if gate is None:
        gate = MergeGate()
    if not isinstance(gate, MergeGate):
        raise TypeError("gate must be a MergeGate instance")
    if approved is not True:
        return _receipt(
            status="approval_required",
            accepted=False,
            applied=False,
            approved=False,
            gate=gate,
            reasons=("explicit approved=True is required",),
        )

    validation = validate_candidate(bundle, max_bytes=max_bytes)
    if not validation.accepted:
        return _receipt(
            status="rejected",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=validation.reasons or ("candidate validation failed",),
            validation=validation,
        )

    target_universe_id = _read_optional(
        target_universe, ("universe_id", "id"))
    if callable(target_universe_id):
        target_universe_id = target_universe_id()
    if not isinstance(target_universe_id, str):
        target_universe_id = None
    try:
        target_generation = _read_generation(target_universe, "target universe")
    except ValueError as exc:
        return _receipt(
            status="unsupported_target",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=(str(exc),),
            validation=validation,
            target_universe_id=target_universe_id,
        )

    candidate_digest = validation.manifest_digest
    if candidate_digest is None:
        return _receipt(
            status="rejected",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=("validated candidate has no manifest digest",),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
        )

    if _candidate_already_applied(target_universe, candidate_digest):
        return _receipt(
            status="duplicate",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=(
                "candidate digest already applied to target; no records written",
            ),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
            target_generation_after=target_generation,
            duplicate=True,
        )

    manifest = validation.manifest or {}
    expected_generation = manifest.get("parent_generation")
    target_generation_expected = manifest.get("generation")
    if not _is_int(expected_generation) or not _is_int(target_generation_expected):
        return _receipt(
            status="rejected",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=("candidate generation fields are invalid",),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
        )
    if target_generation != expected_generation:
        return _receipt(
            status="stale_generation",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=(
                "generation compare-and-swap failed: target is at generation "
                "%d, candidate expects %d"
                % (target_generation, expected_generation),
            ),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
            target_generation_after=target_generation,
        )

    apply = _apply_method(target_universe)
    if apply is None:
        return _receipt(
            status="unsupported_target",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=(
                "target universe does not expose apply_candidate/ingest_candidate",
            ),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
        )

    records = {
        section: tuple(validation.records.get(section, ()))
        for section in SECTIONS
    }
    try:
        result = _call_filtered(apply, {
            "records": records,
            "manifest": manifest,
            "approved": True,
            "expected_generation": expected_generation,
            "target_generation": target_generation_expected,
            "candidate_digest": candidate_digest,
        })
    except Exception as exc:  # target application is a separate subsystem
        return _receipt(
            status="target_error",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=("%s: %s" % (type(exc).__name__, str(exc)[:300]),),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
        )

    result_map = result if isinstance(result, Mapping) else {}
    result_status = result_map.get("status")
    if result_map.get("duplicate") is True or result_status == "duplicate":
        reasons = result_map.get("reasons") or result_map.get("reason") or (
            "target reports duplicate candidate",)
        if isinstance(reasons, str):
            reasons = (reasons,)
        return _receipt(
            status="duplicate",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=tuple(str(item) for item in reasons),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
            target_generation_after=target_generation,
            duplicate=True,
        )
    if (result is False or result_map.get("accepted") is False
            or result_status in {"rejected", "stale_generation", "conflict",
                                 "unsupported"}):
        reasons = result_map.get("reasons") or result_map.get("reason") or (
            "target rejected the candidate",)
        if isinstance(reasons, str):
            reasons = (reasons,)
        return _receipt(
            status="target_rejected",
            accepted=False,
            applied=False,
            approved=True,
            gate=gate,
            reasons=tuple(str(item) for item in reasons),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
        )

    applied_records = result_map.get("applied")
    if isinstance(applied_records, bool) or not _is_int(applied_records):
        applied_records = int(validation.counts.get("total", 0)) if validation.counts else 0

    try:
        target_after = _read_generation(target_universe, "target universe")
    except ValueError as exc:
        return _receipt(
            status="generation_unverified",
            accepted=False,
            applied=True,
            approved=True,
            gate=gate,
            reasons=(str(exc),),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
            applied_records=applied_records,
        )
    if target_after != target_generation_expected:
        return _receipt(
            status="generation_mismatch",
            accepted=False,
            applied=True,
            approved=True,
            gate=gate,
            reasons=(
                "target generation after intake is %d, expected %d"
                % (target_after, target_generation_expected),
            ),
            validation=validation,
            target_universe_id=target_universe_id,
            target_generation_before=target_generation,
            target_generation_after=target_after,
            applied_records=applied_records,
        )

    return _receipt(
        status="accepted",
        accepted=True,
        applied=True,
        approved=True,
        gate=gate,
        validation=validation,
        target_universe_id=target_universe_id,
        target_generation_before=target_generation,
        target_generation_after=target_after,
        applied_records=applied_records,
    )


LOCAL_PURPOSES = ("training_corpus", "evidence")


def integrate_local_corpus(
    universe: Any,
    bundle: str | os.PathLike[str],
    *,
    purpose: str = "training_corpus",
    approved: bool = False,
    gate: MergeGate | None = None,
    max_bytes: int = DEFAULT_MAX_CANDIDATE_BYTES,
) -> dict[str, Any]:
    """Explicit local corpus/evidence integration, separate from intake.

    This does not touch resident state generations or canonical authority. It
    only calls a target method whose name makes the local purpose explicit.
    """
    if gate is None:
        gate = MergeGate()
    if not isinstance(gate, MergeGate):
        raise TypeError("gate must be a MergeGate instance")
    if purpose not in LOCAL_PURPOSES:
        raise ValueError("purpose must be one of: %s" % ", ".join(LOCAL_PURPOSES))
    if approved is not True:
        return {
            "schema": LOCAL_RECEIPT_SCHEMA,
            "status": "approval_required",
            "accepted": False,
            "approved": False,
            "purpose": purpose,
            "reasons": ["explicit approved=True is required"],
            "local_only": True,
            "canonical": False,
            "canonical_authority": False,
            "merge_gate": gate.status(),
        }

    validation = validate_candidate(bundle, max_bytes=max_bytes)
    if not validation.accepted:
        return {
            "schema": LOCAL_RECEIPT_SCHEMA,
            "status": "rejected",
            "accepted": False,
            "approved": True,
            "purpose": purpose,
            "candidate_digest": validation.manifest_digest,
            "reasons": list(validation.reasons),
            "local_only": True,
            "canonical": False,
            "canonical_authority": False,
            "merge_gate": gate.status(),
        }

    method = _local_integration_method(universe)
    if method is None:
        return {
            "schema": LOCAL_RECEIPT_SCHEMA,
            "status": "unsupported_target",
            "accepted": False,
            "approved": True,
            "purpose": purpose,
            "candidate_digest": validation.manifest_digest,
            "reasons": [
                "target universe does not expose a local corpus/evidence method",
            ],
            "local_only": True,
            "canonical": False,
            "canonical_authority": False,
            "merge_gate": gate.status(),
        }

    records = {
        section: tuple(validation.records.get(section, ()))
        for section in SECTIONS
    }
    try:
        result = _call_filtered(method, {
            "records": records,
            "manifest": validation.manifest,
            "approved": True,
            "purpose": purpose,
            "candidate_digest": validation.manifest_digest,
        })
    except Exception as exc:
        return {
            "schema": LOCAL_RECEIPT_SCHEMA,
            "status": "target_error",
            "accepted": False,
            "approved": True,
            "purpose": purpose,
            "candidate_digest": validation.manifest_digest,
            "reasons": ["%s: %s" % (type(exc).__name__, str(exc)[:300])],
            "local_only": True,
            "canonical": False,
            "canonical_authority": False,
            "merge_gate": gate.status(),
        }

    result_map = result if isinstance(result, Mapping) else {}
    accepted = result is not False and result_map.get("accepted") is not False
    reasons = result_map.get("reasons") or ()
    if isinstance(reasons, str):
        reasons = (reasons,)
    return {
        "schema": LOCAL_RECEIPT_SCHEMA,
        "status": "accepted" if accepted else "target_rejected",
        "accepted": accepted,
        "approved": True,
        "purpose": purpose,
        "candidate_digest": validation.manifest_digest,
        "universe_id": (validation.manifest or {}).get("universe_id"),
        "counts": dict(validation.counts or {}),
        "reasons": [str(item) for item in reasons],
        "local_only": True,
        "canonical": False,
        "canonical_authority": False,
        "merge_gate": gate.status(),
        "note": (
            "local training/evidence integration only; generation and canonical "
            "state are untouched"
        ),
    }


__all__ = (
    "SCHEMA",
    "EXPORT_SCHEMA",
    "RECEIPT_SCHEMA",
    "LOCAL_RECEIPT_SCHEMA",
    "MERGE_GATE_SCHEMA",
    "SECTIONS",
    "DEFAULT_MAX_CANDIDATE_BYTES",
    "FULL_CANDIDATE_BYTES",
    "MAX_FULL_SNAPSHOT_BYTES",
    "MergeGate",
    "ValidationResult",
    "canonical_json",
    "compute_manifest_digest",
    "export_candidate",
    "validate_candidate",
    "intake_candidate",
    "integrate_local_corpus",
)
