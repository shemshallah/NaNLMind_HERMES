"""Typed retrieval hypergraph for Round 032 facts.

Round 032 stores WordNet hypernym facts as ``(source, relation, destination)``
records and routes the source/destination concept ids onto mirror-pair cells of
the resident 256-head graph.  This module persists those facts as *typed
hyperedges* without duplicating the resident pair cells or the full fact text:

* one deterministic fact leaf holds the canonical fact payload exactly once;
* one deterministic concept leaf is interned for the source concept and one for
  the destination concept (both are references, never copies of the fact);
* one relation record is appended with the concept leaf references, the typed
  relation digest and the Round 032 source-anchored mirror-pair membership;
* one shared cycle-membership leaf binds the source anchor to its canonical
  pair address, mirror and fundamental cycle;
* one hyperedge binds the fact leaf, both concept leaves, the relation type, the
  pair reference, the mirror reference, the cycle membership and the provenance
  digest.  The directed source/destination concept pair is retained separately
  in the fact payload for mirror-safe reverse lookup.

Identity is content-addressed and mirror-safe.  Re-adding the same fact is
idempotent.  A second provenance for the same fact is reported as a duplicate
with ``source_mismatch=True`` and never overwrites the original payload.  An id
collision that names different members raises a conflict instead of merging.

All reads are bounded: exact lookup uses deterministic ids, traversal follows
the resident relation chains and snapshot/audit accept explicit limits.  No
canonical merge or promotion is performed anywhere.

The module makes no quantum claim.  Pair/cycle membership is classical graph
incidence; the raw pair references are resident locations, not raw pointers.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Iterator

from .cells import (
    address_of,
    canonical_address,
    mirror_address,
    pair_index as cell_pair_index,
)
from .leaves import (
    HEADER_BYTES as LEAF_HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    RECORD_BYTES as LEAF_RECORD_BYTES,
    LeafStore,
)
from .relations import (
    MEMBER_MIRROR,
    MEMBER_PAIR,
    RelationBoundsError,
    RelationConflictError,
    RelationStore,
)

SCHEMA = "nanlm.resident-retrieval-hypergraph/1"
FACT_SCHEMA = "nanlm.resident-retrieval-fact/1"
CONCEPT_SCHEMA = "nanlm.resident-retrieval-concept/1"
CYCLE_SCHEMA = "nanlm.resident-retrieval-cycle-membership/1"

DEFAULT_NODES = 256
DEFAULT_CAPACITY = 256
DEFAULT_SNAPSHOT_LIMIT = 64
MAX_SNAPSHOT_LIMIT = 512
DEFAULT_AUDIT_LIMIT = 4096
MAX_AUDIT_LIMIT = 65_536
DEFAULT_NEIGHBOR_LIMIT = 64
MAX_NEIGHBOR_LIMIT = 512
DEFAULT_TWO_HOP_LIMIT = 64
MAX_TWO_HOP_LIMIT = 512
MAX_BATCH = 2048
MAX_LEN_SCAN = 8192
DEFAULT_SCAN_EXTRA = 8
DEFAULT_SCAN_FACTOR = 4
ZERO_ID = b"\x00" * 32
_ID_BYTES = 32

# Member kinds.  Codes 1-8 are already used by relations/cycle_hypergraph; the
# retrieval layer keeps using pair/mirror/cycle/vertex and adds its own typed
# codes for the fact and its concept/relation/provenance references.
MEMBER_SOURCE = 9
MEMBER_DESTINATION = 10
MEMBER_RELATION_TYPE = 11
MEMBER_FACT = 12
MEMBER_PROVENANCE = 13
MEMBER_ORIENTATION = 14
MEMBER_PAIR_ADDRESS = 15
MEMBER_CYCLE = 5
MEMBER_VERTEX = 7

MEMBER_NAMES = {
    MEMBER_SOURCE: "source",
    MEMBER_DESTINATION: "destination",
    MEMBER_RELATION_TYPE: "relation_type",
    MEMBER_FACT: "fact",
    MEMBER_PROVENANCE: "provenance",
    MEMBER_ORIENTATION: "orientation",
    MEMBER_PAIR_ADDRESS: "pair_address",
    MEMBER_CYCLE: "cycle",
    MEMBER_VERTEX: "vertex",
    MEMBER_PAIR: "pair",
    MEMBER_MIRROR: "mirror",
}
MEMBER_CODES = {name: code for code, name in MEMBER_NAMES.items()}

_POINTER_KEYS = frozenset({
    "addr", "address", "buffer", "bufferaddress", "memaddress",
    "memoryaddress", "offset", "pointer", "ptr", "rawpointer", "rawptr",
})
_PATH_KEYS = frozenset({
    "cwd", "dir", "directory", "file", "filename", "filepath", "folder",
    "location", "path", "root", "uri", "url",
})
_SECRET_KEYS = frozenset({
    "apikey", "accesstoken", "authtoken", "authorization", "bearer",
    "clientsecret", "cookie", "credential", "credentials", "password",
    "passwd", "passphrase", "privatekey", "refreshtoken", "secret",
    "secretkey", "session", "sessionid", "signingkey", "token",
})
_ABS_PATH_RE = re.compile(r"^(?:/|~/|\./|\.\\|\\\\|[A-Za-z]:[\\/])")
_TRAVERSAL_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_NUL_RE = re.compile("\x00")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)"
               r"\s*[:=]\s*\S+"),
)


class RetrievalHypergraphError(ValueError):
    """Base error for the resident retrieval hypergraph."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


class RetrievalBoundsError(RetrievalHypergraphError):
    """A capacity, batch, limit or integer bound was exceeded."""


class RetrievalProvenanceError(RetrievalHypergraphError):
    """A fact was supplied without usable provenance."""


class RetrievalConflictError(RetrievalHypergraphError):
    """An id already names different immutable members."""


class RetrievalCorruptionError(RetrievalHypergraphError):
    """Persisted retrieval members do not agree with their identity."""


class RetrievalNotFound(RetrievalHypergraphError, KeyError):
    """A requested retrieval fact is missing."""


class RetrievalDenied(RetrievalHypergraphError):
    """A structurally forbidden operation (canonical merge) was requested."""


# ---------------------------------------------------------------------------
# value types
# ---------------------------------------------------------------------------
class RetrievalFact(dict):
    """JSON-friendly fact record with attribute and mapping access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.items()}

    as_dict = to_dict

    def __bool__(self) -> bool:
        return not bool(self.get("missing", False))


class RetrievalNeighbors(list):
    """Bounded neighbour list with traversal metadata."""

    def __init__(self, facts: Iterable[Mapping[str, Any]] = (),
                 **metadata: Any):
        super().__init__(facts)
        for name, value in metadata.items():
            setattr(self, name, value)

    @property
    def facts(self) -> list[dict[str, Any]]:
        return list(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "facts":
                return list(self)
            return getattr(self, key)
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        if key == "facts":
            return list(self)
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept": getattr(self, "concept", None),
            "relation": getattr(self, "relation", None),
            "direction": getattr(self, "direction", None),
            "limit": getattr(self, "limit", None),
            "facts": list(self),
            "neighbors": list(self),
            "truncated": getattr(self, "truncated", False),
            "missing": getattr(self, "missing", False),
            "bounded": True,
        }


class RetrievalPaths(list):
    """Bounded two-hop path list with traversal metadata."""

    def __init__(self, paths: Iterable[Mapping[str, Any]] = (),
                 **metadata: Any):
        super().__init__(paths)
        for name, value in metadata.items():
            setattr(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": getattr(self, "source", None),
            "relation": getattr(self, "relation", None),
            "second_relation": getattr(self, "second_relation", None),
            "limit": getattr(self, "limit", None),
            "paths": list(self),
            "count": len(self),
            "truncated": getattr(self, "truncated", False),
            "missing": getattr(self, "missing", False),
            "bounded": True,
        }


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _field(record: Any, name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _field_any(record: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        value = _field(record, name, None)
        if value is not None:
            return value
    return default


def _host_attr(host: Any, name: str, default: Any = None) -> Any:
    if host is None:
        return default
    if isinstance(host, Mapping):
        return host.get(name, default)
    return getattr(host, name, default)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _check_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrievalHypergraphError("%s must be an integer" % label,
                                       code="invalid_%s" % label)
    return value


def _bounded(value: Any, label: str, minimum: int, maximum: int) -> int:
    number = _check_int(value, label)
    if number < minimum or number > maximum:
        raise RetrievalBoundsError(
            "%s must be in [%d, %d]" % (label, minimum, maximum),
            code="%s_out_of_bounds" % label,
            details={label: number, "minimum": minimum, "maximum": maximum})
    return number


def _bytes32(value: Any, *, field: str = "id") -> bytes:
    if isinstance(value, bool):
        raise RetrievalHypergraphError("%s must not be a boolean" % field,
                                       code="invalid_%s" % field)
    if isinstance(value, int):
        if value < 0 or value >= (1 << 256):
            raise RetrievalBoundsError("%s is outside the resident id range"
                                       % field, code="id_out_of_bounds")
        return value.to_bytes(32, "big")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise RetrievalHypergraphError("%s must not be empty" % field,
                                           code="empty_%s" % field)
        if len(raw) == 32:
            return raw
        return raw.rjust(32, b"\x00") if len(raw) < 32 \
            else hashlib.sha256(raw).digest()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RetrievalHypergraphError("%s must not be empty" % field,
                                           code="empty_%s" % field)
        if len(text) == 64:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        return hashlib.sha256(text.encode("utf-8")).digest()
    raise RetrievalHypergraphError("%s must be text, bytes or an integer"
                                   % field, code="invalid_%s" % field)


def _int32(value: int) -> bytes:
    number = int(value)
    if number < 0:
        raise RetrievalBoundsError("member integer must be non-negative",
                                   code="member_out_of_bounds")
    return number.to_bytes(32, "big")


def _leaf16(value: bytes) -> bytes:
    raw = bytes(value)
    if len(raw) >= 16:
        return raw[-16:]
    return raw.rjust(16, b"\x00")


def _pad_leaf(leaf_id: bytes) -> bytes:
    return _bytes32(_leaf16(leaf_id), field="leaf")


def _leaf_ref(leaf_id: bytes) -> bytes:
    return hashlib.sha256(
        b"leaf-ref:" + _leaf16(leaf_id).hex().encode("ascii")).digest()


def _digest(prefix: bytes, material: bytes) -> bytes:
    return hashlib.sha256(prefix + material).digest()


def _digest_hex(prefix: bytes, material: bytes) -> str:
    return _digest(prefix, material).hex()


def _check_text(text: str, *, label: str) -> None:
    if _NUL_RE.search(text):
        raise RetrievalHypergraphError("%s contains a NUL byte" % label,
                                       code="invalid_text")
    if _ABS_PATH_RE.match(text) or _TRAVERSAL_RE.search(text):
        raise RetrievalHypergraphError(
            "%s contains a filesystem-path-like value" % label,
            code="path_denied")
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(text):
            raise RetrievalProvenanceError(
                "%s contains secret-shaped data" % label,
                code="secret_denied")


def _clean_json(value: Any, *, label: str, depth: int = 0,
                counter: list[int] | None = None,
                key: str | None = None) -> Any:
    """Validate and detach a bounded JSON-only value."""
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > 4096:
        raise RetrievalBoundsError("%s exceeds the JSON node bound" % label,
                                   code="json_too_large")
    if depth > 16:
        raise RetrievalBoundsError("%s exceeds the JSON depth bound" % label,
                                   code="json_too_deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise RetrievalBoundsError("%s contains an oversized integer"
                                       % label, code="integer_out_of_bounds")
        return value
    if isinstance(value, float):
        raise RetrievalHypergraphError(
            "%s contains a float; retrieval payloads are integer/string only"
            % label, code="float_denied")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 4096:
            raise RetrievalBoundsError("%s contains an oversized string"
                                       % label, code="string_too_large")
        _check_text(value, label=label)
        return value
    if isinstance(value, (list, tuple)):
        return [_clean_json(item, label=label, depth=depth + 1,
                            counter=counter, key=key) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for item_key, item in value.items():
            if not isinstance(item_key, str) or not item_key:
                raise RetrievalHypergraphError(
                    "%s object keys must be non-empty strings" % label,
                    code="invalid_object_key")
            if len(item_key.encode("utf-8")) > 256:
                raise RetrievalBoundsError("%s contains an oversized key"
                                           % label, code="key_too_large")
            _check_text(item_key, label="%s key" % label)
            compact = re.sub(r"[^a-z0-9]+", "", item_key.casefold())
            if compact in _POINTER_KEYS:
                raise RetrievalHypergraphError(
                    "%s contains a pointer-shaped key %r"
                    % (label, item_key), code="pointer_denied")
            if compact in _PATH_KEYS:
                raise RetrievalHypergraphError(
                    "%s contains a path-shaped key %r"
                    % (label, item_key), code="path_denied")
            if compact in _SECRET_KEYS:
                raise RetrievalProvenanceError(
                    "%s contains a secret-shaped key %r"
                    % (label, item_key), code="secret_denied")
            result[item_key] = _clean_json(
                item, label="%s.%s" % (label, item_key), depth=depth + 1,
                counter=counter, key=item_key)
        return result
    raise RetrievalHypergraphError(
        "%s must contain only JSON values; got %s"
        % (label, type(value).__name__), code="invalid_json_value")


def _identifier_text(value: Any, label: str = "identifier") -> str:
    if value is None:
        raise RetrievalHypergraphError("%s is required" % label,
                                       code="%s_required" % label)
    if isinstance(value, bool):
        raise RetrievalHypergraphError("%s must not be a boolean" % label,
                                       code="invalid_%s" % label)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RetrievalHypergraphError("%s must not be empty" % label,
                                           code="empty_%s" % label)
        _check_text(text, label=label)
        return text
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise RetrievalHypergraphError("%s must not be empty" % label,
                                           code="empty_%s" % label)
        return raw.hex()
    if isinstance(value, (Mapping, Sequence)):
        clean = _clean_json(value, label=label)
        return _canonical_json(clean)
    raise RetrievalHypergraphError(
        "%s must be text, bytes, an integer, a mapping or a sequence" % label,
        code="invalid_%s" % label)


def _normalize_provenance(value: Any) -> Any:
    if value is None:
        raise RetrievalProvenanceError(
            "fact provenance is required", code="provenance_required")
    if isinstance(value, bool):
        raise RetrievalProvenanceError(
            "fact provenance must not be a boolean",
            code="invalid_provenance")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RetrievalProvenanceError(
                "fact provenance must not be empty", code="empty_provenance")
        _check_text(text, label="fact provenance")
        return text
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise RetrievalProvenanceError(
                "fact provenance must not be empty", code="empty_provenance")
        return raw.hex()
    return _clean_json(value, label="fact provenance")


def provenance_digest(value: Any) -> bytes:
    """Stable 32-byte provenance digest using the resident convention.

    A 64-character hexadecimal provenance is treated as an already-computed
    digest, matching ``cycle_hypergraph``/``cycle_persistence``.  Text is
    hashed directly, integers are left-padded and structured values are hashed
    over canonical JSON.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise RetrievalProvenanceError(
                "fact provenance must not be empty", code="empty_provenance")
        if len(raw) == 32:
            return raw
        if len(raw) < 32:
            return raw.rjust(32, b"\x00")
        return hashlib.sha256(raw).digest()
    clean = _normalize_provenance(value)
    if isinstance(clean, str):
        text = clean.strip()
        if len(text) == 64:
            try:
                return bytes.fromhex(text)
            except ValueError:
                pass
        return hashlib.sha256(text.encode("utf-8")).digest()
    if isinstance(clean, int):
        return _bytes32(clean, field="provenance")
    return hashlib.sha256(_canonical_json(clean).encode("utf-8")).digest()


# ---------------------------------------------------------------------------
# deterministic identity and mirror-pair membership
# ---------------------------------------------------------------------------
def concept_text(value: Any) -> str:
    """Canonical concept identifier (``1`` and ``"1"`` name the same leaf)."""
    return _identifier_text(value, "concept id")


def relation_text(value: Any) -> str:
    """Canonical relation type identifier."""
    return _identifier_text(value, "relation")


def concept_leaf_id(value: Any) -> bytes:
    """Deterministic 128-bit leaf id for a source/destination concept."""
    return _digest(b"NLMR-RETRIEVAL-CONCEPT\0",
                   concept_text(value).encode("utf-8"))[:16]


def relation_type_id(value: Any) -> bytes:
    """Deterministic 32-byte typed relation id."""
    return _digest(b"NLMR-RETRIEVAL-REL-TYPE\0",
                   relation_text(value).encode("utf-8"))


def _identity_text(source: Any, relation: Any, destination: Any) -> str:
    return _canonical_json([_identifier_text(source, "source_id"),
                            _identifier_text(relation, "relation"),
                            _identifier_text(destination, "destination_id")])


def fact_identity(source: Any, relation: Any, destination: Any) -> str:
    """Canonical fact identity text shared by all deterministic ids."""
    return _identity_text(source, relation, destination)


def fact_id(source: Any, relation: Any, destination: Any) -> str:
    """Stable 256-bit fact id as lowercase hex."""
    return _digest_hex(b"NLMR-RETRIEVAL-FACT\0",
                       _identity_text(source, relation, destination)
                       .encode("utf-8"))


def hyperedge_id(source: Any, relation: Any, destination: Any) -> bytes:
    """Stable 32-byte typed-hyperedge id for a fact."""
    return _digest(b"NLMR-RETRIEVAL-EDGE\0",
                   _identity_text(source, relation, destination)
                   .encode("utf-8"))


def relation_id(source: Any, relation: Any, destination: Any) -> bytes:
    """Stable 32-byte resident relation id for a fact."""
    return _digest(b"NLMR-RETRIEVAL-REL\0",
                   _identity_text(source, relation, destination)
                   .encode("utf-8"))


def fact_leaf_id(source: Any, relation: Any, destination: Any) -> bytes:
    """Deterministic 128-bit leaf id holding the canonical fact payload."""
    return _digest(b"NLMR-RETRIEVAL-FACT-LEAF\0",
                   _identity_text(source, relation, destination)
                   .encode("utf-8"))[:16]


_MASK64 = (1 << 64) - 1
_DECIMAL_RE = re.compile(r"^[0-9]+$")


def _hash64(value: int) -> int:
    """SplitMix64 used by the Round 032 native anchor lane."""
    number = value & _MASK64
    number ^= number >> 30
    number = (number * 0xBF58476D1CE4E5B9) & _MASK64
    number ^= number >> 27
    number = (number * 0x94D049BB133111EB) & _MASK64
    number ^= number >> 31
    return number & _MASK64


def concept_head(value: Any, nodes: int = DEFAULT_NODES) -> int:
    """Deterministic non-zero head in ``[1, nodes)`` for a concept."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
        raise RetrievalBoundsError("nodes must be an integer of at least 3",
                                   code="invalid_nodes")
    raw = _digest(b"NLMR-RETRIEVAL-HEAD\0",
                  concept_text(value).encode("utf-8"))
    return int.from_bytes(raw[:8], "big") % (nodes - 1) + 1


def anchor_pair(concept_id: Any,
                nodes: int = DEFAULT_NODES) -> tuple[int, int]:
    """Round 032-style source anchor: a sorted distinct head pair.

    Integer concept ids use the same SplitMix64 mix as the native experiment and
    :mod:`nanlmind_hermes.resident.mirror_retrieval`, so the pair membership is
    directly comparable.  Textual concepts use a SHA-256 mix in the same
    ``[1, nodes)`` range.
    """
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
        raise RetrievalBoundsError("nodes must be an integer of at least 3",
                                   code="invalid_nodes")
    text = concept_text(concept_id)
    span = nodes - 1
    if _DECIMAL_RE.match(text) and int(text) <= _MASK64:
        mixed = _hash64(int(text) + 0x9E3779B97F4A7C15)
        first = 1 + mixed % span
        second = 1 + (mixed >> 15) % span
    else:
        raw = _digest(b"NLMR-RETRIEVAL-ANCHOR\0", text.encode("utf-8"))
        first = 1 + int.from_bytes(raw[:8], "big") % span
        second = 1 + int.from_bytes(raw[8:16], "big") % span
    if first == second:
        second = second % span + 1
    if first > second:
        first, second = second, first
    return int(first), int(second)


def pair_heads(source: Any, destination: Any,
               nodes: int = DEFAULT_NODES) -> tuple[int, int]:
    """Deterministic distinct heads for a source/destination concept pair."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
        raise RetrievalBoundsError("nodes must be an integer of at least 3",
                                   code="invalid_nodes")
    first = concept_head(source, nodes)
    second = concept_head(destination, nodes)
    if first == second:
        # Keep the adjustment unordered so (A, B) and (B, A) still share one
        # mirror-pair/cycle membership when both concepts hash to one head.
        material = "\x1f".join(sorted((concept_text(source),
                                       concept_text(destination)))).encode(
            "utf-8")
        second = int.from_bytes(
            _digest(b"NLMR-RETRIEVAL-PARTNER\0", material)[:8], "big") \
            % (nodes - 1) + 1
        if second == first:
            second = second % (nodes - 1) + 1
    return first, second


def cycle_leaf_id(first: int, second: int) -> bytes:
    """Cycle leaf id matching the resident fundamental-triangle convention."""
    cycle = (0,) + tuple(sorted((int(first), int(second))))
    key = ("%d:%d:%d" % cycle).encode("ascii")
    return _digest(b"NLMR-CYCLE-LEAF\0", key)[:16]


def pair_membership(source: Any, destination: Any,
                    nodes: int = DEFAULT_NODES) -> dict[str, Any]:
    """Return the mirror-pair and fundamental-cycle membership of a fact.

    The pair is unordered and mirror-safe: ``(a, b)`` and ``(b, a)`` map to the
    same canonical address/pair index.  The direction is retained separately as
    ``orientation`` so the typed hyperedge still knows which concept is the
    source.
    """
    # Primary membership is the Round 032 source anchor.  Facts with the same
    # source share one mirror-pair posting/membership even when their
    # destinations differ.
    anchor_first, anchor_second = anchor_pair(source, nodes)
    canonical = canonical_address(anchor_first, anchor_second)
    mirror = mirror_address(canonical)
    cycle = (0, int(anchor_first), int(anchor_second))
    leaf = cycle_leaf_id(anchor_first, anchor_second)

    # The directed concept pair is retained separately for traversal metadata.
    ordered_first, ordered_second = pair_heads(source, destination, nodes)
    concept_first, concept_second = sorted((ordered_first, ordered_second))
    concept_canonical = canonical_address(concept_first, concept_second)
    concept_cycle = (0, int(concept_first), int(concept_second))
    orientation = "forward" \
        if address_of(ordered_first, ordered_second) == concept_canonical \
        else "mirror"
    return {
        "source_head": int(ordered_first),
        "destination_head": int(ordered_second),
        "anchor_heads": (int(anchor_first), int(anchor_second)),
        "pair": (int(anchor_first), int(anchor_second)),
        "pair_ref": int(cell_pair_index(anchor_first, anchor_second)),
        "canonical_pair": int(canonical),
        "mirror": int(mirror),
        "cycle": cycle,
        "cycle_id": "%d:%d:%d" % cycle,
        "cycle_leaf": leaf,
        "orientation": orientation,
        "concept_pair": (int(concept_first), int(concept_second)),
        "concept_pair_ref": int(cell_pair_index(concept_first,
                                                concept_second)),
        "concept_canonical_pair": int(concept_canonical),
        "concept_mirror": int(mirror_address(concept_canonical)),
        "concept_cycle": concept_cycle,
        "concept_cycle_id": "%d:%d:%d" % concept_cycle,
        "concept_cycle_leaf": cycle_leaf_id(concept_first, concept_second),
    }


# ---------------------------------------------------------------------------
# store resolution and record adapters
# ---------------------------------------------------------------------------
def _default_stores(capacity: int = DEFAULT_CAPACITY):
    relations = RelationStore.allocate(
        relation_capacity=max(8, capacity),
        hyperedge_capacity=max(8, capacity),
        arena_bytes=max(8192, capacity * 16 * 40))
    leaf_capacity = max(32, capacity * 5)
    index_capacity = _next_power_of_two(max(16, leaf_capacity * 2))
    buffer = bytearray(
        LEAF_HEADER_BYTES + leaf_capacity * LEAF_RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)
    leaves = LeafStore.create(buffer, capacity=leaf_capacity)
    return relations, leaves, None


def _resolve_stores(universe_or_stores: Any = None, *,
                    relations: Any = None, leaves: Any = None,
                    cells: Any = None):
    """Resolve a universe, a mapping, a tuple or direct stores."""
    host = None
    source = universe_or_stores
    if source is None:
        pass
    elif isinstance(source, Mapping) and (
            "relations" in source or "leaves" in source or "cells" in source):
        relations = relations if relations is not None else source.get("relations")
        leaves = leaves if leaves is not None else source.get("leaves")
        cells = cells if cells is not None else source.get("cells")
        host = source.get("host")
    elif isinstance(source, (tuple, list)):
        values = list(source)
        if len(values) >= 1 and relations is None:
            relations = values[0]
        if len(values) >= 2 and leaves is None:
            leaves = values[1]
        if len(values) >= 3 and cells is None:
            cells = values[2]
    elif _host_attr(source, "relations", None) is not None:
        host = source
        relations = relations if relations is not None else source.relations
        leaves = leaves if leaves is not None else _host_attr(source, "leaves")
        cells = cells if cells is not None else _host_attr(source, "cells")
    elif relations is None and (
            callable(_host_attr(source, "get_hyperedge"))
            or callable(_host_attr(source, "append_relation"))):
        relations = source
    elif leaves is None and callable(_host_attr(source, "insert")):
        leaves = source
    if relations is None or leaves is None:
        default_relations, default_leaves, _ = _default_stores()
        if relations is None:
            relations = default_relations
        if leaves is None:
            leaves = default_leaves
    return relations, leaves, cells, host


def _member_pairs(view: Any) -> list[tuple[int, bytes]]:
    """Normalize a hyperedge view into ``(kind_code, member_id)`` pairs."""
    members = _field_any(view, ("members", "member_ids"))
    kinds = _field_any(view, ("member_kinds",))
    out: list[tuple[int, bytes]] = []
    if members is None:
        return out
    for index, member in enumerate(members):
        if isinstance(member, Mapping):
            code = member.get("kind_code")
            if code is None:
                name = member.get("kind")
                if isinstance(name, str) and name in MEMBER_CODES:
                    code = MEMBER_CODES[name]
            identifier = member.get("id", member.get("member_id"))
            if identifier is None and "id_int" in member:
                identifier = member["id_int"]
        elif isinstance(member, (tuple, list)) and len(member) == 2:
            kind, identifier = member
            if isinstance(kind, int) and not isinstance(kind, bool):
                code = kind
            else:
                code = MEMBER_CODES.get(str(kind), 0)
        else:
            if kinds is not None and index < len(kinds):
                kind = kinds[index]
                code = kind if isinstance(kind, int) and not isinstance(
                    kind, bool) else MEMBER_CODES.get(str(kind), 0)
            else:
                code = 0
            identifier = member
        if isinstance(code, bool) or not isinstance(code, int):
            code = 0
        out.append((code, _bytes32(identifier, field="member id")))
    return out


def _member_details(pairs: Iterable[tuple[int, bytes]]) -> list[dict[str, Any]]:
    return [{
        "kind": MEMBER_NAMES.get(code, "kind:%d" % code),
        "kind_code": int(code),
        "id": raw.hex(),
    } for code, raw in pairs]


def _core_pairs(pairs: Iterable[tuple[int, bytes]]
                ) -> list[tuple[int, bytes]]:
    return [(code, raw) for code, raw in pairs
            if code != MEMBER_PROVENANCE]


def _scan_limit(limit: int, scan_limit: int | None,
                maximum: int = MAX_AUDIT_LIMIT) -> int:
    if scan_limit is None:
        scan_limit = min(maximum, max(limit * DEFAULT_SCAN_FACTOR
                                      + DEFAULT_SCAN_EXTRA, limit))
    return _bounded(scan_limit, "scan_limit", 1, maximum)


class RetrievalHypergraph:
    """Bounded typed-hyperedge layer over resident or fake stores."""

    def __init__(self, universe_or_stores: Any = None, *,
                 relations: Any = None, leaves: Any = None, cells: Any = None,
                 nodes: int = DEFAULT_NODES,
                 commit: Any = None, flush: Any = None):
        if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 3:
            raise RetrievalBoundsError(
                "nodes must be an integer of at least 3", code="invalid_nodes")
        self.nodes = nodes
        self.relations, self.leaves, self.cells, self.host = _resolve_stores(
            universe_or_stores, relations=relations, leaves=leaves,
            cells=cells)
        self._commit = commit if commit is not None else (
            _host_attr(self.host, "_commit") or _host_attr(self.host, "commit"))
        self._flush = flush if flush is not None else _host_attr(
            self.host, "flush")
        # Direct LeafStore slots do not own a payload arena.  Keeping the
        # bounded canonical text next to the in-process adapter lets a caller
        # use direct stores for traversal; the resident universe/fake facade
        # remains the durable source of the same text.
        self._leaf_cache: dict[bytes, str] = {}

    # -- metadata -------------------------------------------------------
    @property
    def canonical_merge(self) -> bool:
        return False

    def _relation_status(self) -> dict[str, Any]:
        status = _host_attr(self.relations, "status", None)
        if callable(status):
            try:
                value = status()
            except Exception:
                return {}
            return dict(value) if isinstance(value, Mapping) else {}
        if isinstance(status, Mapping):
            return dict(status)
        return {}

    def _relation_count(self) -> int | None:
        status = self._relation_status()
        for key in ("relations", "relation_count"):
            value = status.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = _host_attr(self.relations, "relation_count", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        records = self._relation_sequence()
        if records is not None:
            return len(records)
        return None

    def _hyperedge_count(self) -> int | None:
        status = self._relation_status()
        for key in ("hyperedges", "hyperedge_count"):
            value = status.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        records = _host_attr(self.relations, "hyperedges", None)
        if isinstance(records, Mapping):
            return len(records)
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            return len(records)
        return None

    def _leaf_store_attr(self, name: str) -> Any:
        owner = self.host if self.host is not None else self.leaves
        value = _host_attr(self.leaves, name, None)
        if value is None:
            value = _host_attr(owner, name, None)
        return value

    def _leaf_count(self) -> int | None:
        if isinstance(self.leaves, Mapping):
            return len(self.leaves)
        value = self._leaf_store_attr("count")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        status = _host_attr(self.host, "status", None) if self.host else None
        if callable(status):
            try:
                counts = status().get("counts", {})
                value = counts.get("leaves")
            except Exception:
                value = None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _leaf_capacity(self) -> int | None:
        value = self._leaf_store_attr("capacity")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        status = _host_attr(self.host, "status", None) if self.host else None
        if callable(status):
            try:
                counts = status().get("counts", {})
                value = counts.get("leaf_capacity")
            except Exception:
                value = None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _relation_sequence(self) -> Any:
        records = _host_attr(self.relations, "records", None)
        if records is not None:
            return records
        value = _host_attr(self.relations, "relations", None)
        if value is not None and not callable(value):
            return value
        return None

    # -- relation/hyperedge access -------------------------------------
    def _get_relation(self, key: Any) -> Any:
        get = _host_attr(self.relations, "get_relation", None)
        if callable(get):
            keys = [key]
            if isinstance(key, (bytes, bytearray, memoryview)):
                keys.append(bytes(key).hex())
            for candidate in keys:
                try:
                    value = get(candidate)
                except Exception:
                    value = None
                if value is not None:
                    return value
        records = self._relation_sequence()
        if isinstance(records, Mapping):
            for candidate in (key, key.hex()
                              if isinstance(key, (bytes, bytearray, memoryview))
                              else key):
                if candidate in records:
                    return records[candidate]
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            wanted = key.hex() if isinstance(
                key, (bytes, bytearray, memoryview)) else str(key)
            for record in records:
                stored = _field(record, "relation_id")
                if stored == wanted or stored == key:
                    return record
        return None

    def _iter_relations(self, limit: int) -> Iterator[Any]:
        if limit <= 0:
            return
        iterator = _host_attr(self.relations, "iter_relations", None)
        if callable(iterator):
            try:
                stream = iterator(limit)
            except TypeError:
                stream = iterator()
            for index, item in enumerate(stream):
                if index >= limit:
                    break
                yield item
            return
        lister = _host_attr(self.relations, "relations", None)
        if callable(lister):
            try:
                stream = lister(limit)
            except TypeError:
                stream = lister()
            for index, item in enumerate(stream):
                if index >= limit:
                    break
                yield item
            return
        records = self._relation_sequence()
        if isinstance(records, Mapping):
            for index, item in enumerate(records.values()):
                if index >= limit:
                    break
                yield item
            return
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            for index, item in enumerate(records):
                if index >= limit:
                    break
                yield item
            return
        count = self._relation_count()
        get = _host_attr(self.relations, "get_relation", None)
        if callable(get) and count is not None:
            for index in range(min(count, limit)):
                item = self._get_relation(index)
                if item is not None:
                    yield item

    def _get_hyperedge(self, identifier: Any) -> Any:
        if isinstance(identifier, Mapping):
            return identifier
        get = _host_attr(self.relations, "get_hyperedge", None)
        if callable(get):
            keys = [identifier]
            if isinstance(identifier, (bytes, bytearray, memoryview)):
                keys.append(bytes(identifier).hex())
            for candidate in keys:
                try:
                    value = get(candidate)
                except Exception:
                    value = None
                if value is not None:
                    return value
        records = _host_attr(self.relations, "hyperedges", None)
        if isinstance(records, Mapping):
            for candidate in (
                    identifier, identifier.hex()
                    if isinstance(identifier, (bytes, bytearray, memoryview))
                    else identifier):
                if candidate in records:
                    return records[candidate]
        if isinstance(records, Sequence) and not isinstance(
                records, (str, bytes, bytearray)):
            wanted = identifier.hex() if isinstance(
                identifier, (bytes, bytearray, memoryview)) else str(identifier)
            for record in records:
                stored = _field(record, "hyperedge_id")
                if stored == wanted or stored == identifier:
                    return record
        return None

    # -- leaf access ----------------------------------------------------
    def _find_leaf(self, leaf_id: bytes) -> Any:
        if self.host is not None:
            getter = _host_attr(self.host, "get_leaf", None)
            if callable(getter):
                for key in (leaf_id, leaf_id.hex()):
                    try:
                        value = getter(key)
                    except Exception:
                        value = None
                    if value is not None:
                        return value
                return None
        if isinstance(self.leaves, Mapping):
            for key in (leaf_id, leaf_id.hex()):
                if key in self.leaves:
                    return self.leaves[key]
            return None
        find = _host_attr(self.leaves, "find", None)
        if callable(find):
            try:
                return find(leaf_id)
            except Exception:
                try:
                    return find(leaf_id.hex())
                except Exception:
                    return None
        get = _host_attr(self.leaves, "get", None)
        if callable(get):
            for key in (leaf_id, leaf_id.hex()):
                try:
                    return get(key)
                except Exception:
                    continue
        return None

    def _leaf_text(self, record: Any) -> str | None:
        if record is None:
            return None
        if isinstance(record, str):
            return record
        if isinstance(record, Mapping):
            text = record.get("text")
            if isinstance(text, str):
                return text
            if isinstance(record.get("payload"), str):
                return record["payload"]
            return None
        length = _field(record, "feature_length", 0)
        if isinstance(length, int) and length > 0:
            reader = _host_attr(self.leaves, "feature_bytes", None)
            if callable(reader):
                try:
                    raw = reader(record)
                    return raw.decode("utf-8") if isinstance(raw, bytes) else None
                except Exception:
                    return None
        return None

    def _leaf_digest(self, record: Any) -> bytes | None:
        value = _field_any(record, ("payload_digest", "digest"))
        if value is None:
            return None
        try:
            return _bytes32(value, field="payload_digest")
        except RetrievalHypergraphError:
            return None

    def _leaf_text_payload(self, leaf_id: bytes) -> dict[str, Any] | None:
        record = self._find_leaf(leaf_id)
        if record is None:
            return None
        text = self._leaf_text(record) or self._leaf_cache.get(leaf_id)
        if not text:
            return None
        try:
            value = json.loads(text)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _verify_leaf_content(self, leaf_id: bytes, content_text: str,
                             expected: Mapping[str, Any]) -> Any:
        existing = self._find_leaf(leaf_id)
        if existing is None:
            return None
        payload = self._leaf_text_payload(leaf_id)
        if payload is None and self._leaf_cache.get(leaf_id):
            try:
                cached = json.loads(self._leaf_cache[leaf_id])
            except (TypeError, ValueError):
                cached = None
            payload = cached if isinstance(cached, dict) else None
        if payload is not None:
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise RetrievalConflictError(
                        "leaf %s already names different content"
                        % leaf_id.hex(),
                        code="leaf_conflict",
                        details={"leaf_id": leaf_id.hex(), "field": key})
        else:
            digest = self._leaf_digest(existing)
            expected_digest = hashlib.sha256(
                content_text.encode("utf-8")).digest()
            if digest is not None and digest != expected_digest:
                raise RetrievalConflictError(
                    "leaf %s already names a different payload"
                    % leaf_id.hex(),
                    code="leaf_payload_conflict",
                    details={"leaf_id": leaf_id.hex()})
        return existing

    def _check_prepared_leaves(self, prepared: Mapping[str, Any]) -> None:
        membership = prepared["membership"]
        self._verify_leaf_content(
            prepared["source_leaf"], prepared["source_text_payload"],
            {"concept": prepared["source_text"]})
        self._verify_leaf_content(
            prepared["destination_leaf"],
            prepared["destination_text_payload"],
            {"concept": prepared["destination_text"]})
        self._verify_leaf_content(
            membership["cycle_leaf"], prepared["cycle_text_payload"],
            {"cycle_id": membership["cycle_id"],
             "pair_ref": int(membership["pair_ref"])})
        self._verify_leaf_content(
            prepared["fact_leaf"], prepared["fact_text"],
            {"fact_id": prepared["fact_id"],
             "source": prepared["source_text"],
             "relation": prepared["relation_text"],
             "destination": prepared["destination_text"]})

    def _ensure_leaf(self, leaf_id: bytes, kind: str, content_text: str,
                     expected: Mapping[str, Any], head: int,
                     source_digest: bytes,
                     pair_id: int | None = None) -> tuple[Any, bool]:
        existing = self._verify_leaf_content(leaf_id, content_text, expected)
        if existing is not None:
            return existing, False
        append = _host_attr(self.host, "append_leaf", None) if self.host else None
        if not callable(append):
            append = _host_attr(self.host, "ingest_leaf", None) \
                if self.host else None
        if callable(append):
            record = {
                "type": "leaf",
                "leaf_id": leaf_id.hex(),
                "kind": kind,
                "head": int(head),
                "text": content_text,
                "source_digest": source_digest.hex(),
            }
            if pair_id is not None:
                record["pair_id"] = int(pair_id)
            result = append(record)
            self._leaf_cache[leaf_id] = content_text
            if isinstance(result, Mapping) and result.get("duplicate"):
                repaired = self._find_leaf(leaf_id)
                if repaired is None:
                    raise RetrievalCorruptionError(
                        "leaf append reported a duplicate that cannot be read",
                        code="leaf_missing", details={"leaf_id": leaf_id.hex()})
                return repaired, False
            return result, True
        if isinstance(self.leaves, Mapping):
            record = {
                "leaf_id": leaf_id.hex(),
                "kind": kind,
                "head": int(head),
                "text": content_text,
                "payload_digest": hashlib.sha256(
                    content_text.encode("utf-8")).hexdigest(),
            }
            if pair_id is not None:
                record["pair_id"] = int(pair_id)
            self.leaves[leaf_id] = record
            self._leaf_cache[leaf_id] = content_text
            return record, True
        insert = _host_attr(self.leaves, "insert", None)
        if callable(insert):
            raw = content_text.encode("utf-8")
            result = insert(
                kind, leaf_id=leaf_id, content=raw,
                payload_digest=hashlib.sha256(raw).digest(),
                head=int(head), pair_id=pair_id)
            self._leaf_cache[leaf_id] = content_text
            return result, True
        raise RetrievalHypergraphError(
            "leaf store does not support insert", code="unsupported_store")

    # -- payload/member construction ------------------------------------
    def _fact_payload(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        membership = prepared["membership"]
        return {
            "schema": FACT_SCHEMA,
            "kind": "retrieval-fact",
            "source": prepared["source_text"],
            "relation": prepared["relation_text"],
            "destination": prepared["destination_text"],
            "provenance": prepared["provenance_value"],
            "fact_id": prepared["fact_id"],
            "fact_leaf": prepared["fact_leaf"].hex(),
            "source_leaf": prepared["source_leaf"].hex(),
            "destination_leaf": prepared["destination_leaf"].hex(),
            "relation_type": prepared["relation_type"].hex(),
            "anchor_pair": list(membership["anchor_heads"]),
            "pair": list(membership["pair"]),
            "pair_ref": int(membership["pair_ref"]),
            "canonical_pair": int(membership["canonical_pair"]),
            "mirror": int(membership["mirror"]),
            "cycle": list(membership["cycle"]),
            "cycle_id": str(membership["cycle_id"]),
            "cycle_leaf": membership["cycle_leaf"].hex(),
            "concept_pair": list(membership["concept_pair"]),
            "concept_pair_ref": int(membership["concept_pair_ref"]),
            "concept_canonical_pair": int(
                membership["concept_canonical_pair"]),
            "concept_mirror": int(membership["concept_mirror"]),
            "concept_cycle": list(membership["concept_cycle"]),
            "concept_cycle_id": str(membership["concept_cycle_id"]),
            "concept_cycle_leaf": membership["concept_cycle_leaf"].hex(),
            "orientation": str(membership["orientation"]),
            "provenance_digest": prepared["provenance_digest"].hex(),
        }

    def _fact_members(self, prepared: Mapping[str, Any]
                      ) -> list[tuple[int, bytes]]:
        membership = prepared["membership"]
        first, second = membership["pair"]
        provenance = prepared["provenance_digest"]
        return [
            (MEMBER_FACT, _pad_leaf(prepared["fact_leaf"])),
            (MEMBER_SOURCE, _pad_leaf(prepared["source_leaf"])),
            (MEMBER_RELATION_TYPE, prepared["relation_type"]),
            (MEMBER_DESTINATION, _pad_leaf(prepared["destination_leaf"])),
            (MEMBER_PAIR, _int32(membership["pair_ref"])),
            (MEMBER_PAIR_ADDRESS, _int32(membership["canonical_pair"])),
            (MEMBER_MIRROR, _int32(membership["mirror"])),
            (MEMBER_CYCLE, _pad_leaf(membership["cycle_leaf"])),
            (MEMBER_VERTEX, _int32(0)),
            (MEMBER_VERTEX, _int32(first)),
            (MEMBER_VERTEX, _int32(second)),
            (MEMBER_PROVENANCE, provenance),
            (MEMBER_ORIENTATION,
             _int32(1 if membership["orientation"] == "forward" else 0)),
        ]

    def _prepare_fact(self, source_id: Any, relation: Any,
                      destination_id: Any, provenance: Any) -> dict[str, Any]:
        source_text = _identifier_text(source_id, "source_id")
        relation_value = relation_text(relation)
        destination_text = _identifier_text(destination_id, "destination_id")
        provenance_value = _normalize_provenance(provenance)
        identity = _identity_text(source_text, relation_value,
                                  destination_text)
        membership = pair_membership(source_text, destination_text,
                                     self.nodes)
        prepared = {
            "source_text": source_text,
            "relation_text": relation_value,
            "destination_text": destination_text,
            "provenance_value": provenance_value,
            "provenance_digest": provenance_digest(provenance_value),
            "identity": identity,
            "fact_id": _digest_hex(b"NLMR-RETRIEVAL-FACT\0",
                                   identity.encode("utf-8")),
            "relation_key": _digest(b"NLMR-RETRIEVAL-REL\0",
                                    identity.encode("utf-8")),
            "edge_key": _digest(b"NLMR-RETRIEVAL-EDGE\0",
                                identity.encode("utf-8")),
            "fact_leaf": _digest(b"NLMR-RETRIEVAL-FACT-LEAF\0",
                                 identity.encode("utf-8"))[:16],
            "source_leaf": concept_leaf_id(source_text),
            "destination_leaf": concept_leaf_id(destination_text),
            "relation_type": relation_type_id(relation_value),
            "membership": membership,
            "source_head": concept_head(source_text, self.nodes),
            "destination_concept_head": concept_head(destination_text,
                                                     self.nodes),
            "subject_ref": _leaf_ref(concept_leaf_id(source_text)),
            "predicate_ref": relation_type_id(relation_value),
            "object_ref": _leaf_ref(concept_leaf_id(destination_text)),
        }
        prepared["members"] = self._fact_members(prepared)
        prepared["fact_payload"] = self._fact_payload(prepared)
        prepared["fact_text"] = _canonical_json(prepared["fact_payload"])
        prepared["source_text_payload"] = _canonical_json({
            "schema": CONCEPT_SCHEMA,
            "kind": "concept",
            "concept": source_text,
        })
        prepared["destination_text_payload"] = _canonical_json({
            "schema": CONCEPT_SCHEMA,
            "kind": "concept",
            "concept": destination_text,
        })
        prepared["cycle_text_payload"] = _canonical_json({
            "schema": CYCLE_SCHEMA,
            "kind": "cycle-membership",
            "cycle": list(membership["cycle"]),
            "cycle_id": str(membership["cycle_id"]),
            "pair": list(membership["pair"]),
            "pair_ref": int(membership["pair_ref"]),
            "canonical_pair": int(membership["canonical_pair"]),
            "mirror": int(membership["mirror"]),
        })
        return prepared

    def _parse_fact_view(self, view: Any) -> dict[str, Any] | None:
        if view is None:
            return None
        pairs = _member_pairs(view)
        codes = [code for code, _ in pairs]
        if MEMBER_FACT not in codes:
            return None
        fact_leaf = next((raw for code, raw in pairs
                          if code == MEMBER_FACT), None)
        source_leaf = next((raw for code, raw in pairs
                            if code == MEMBER_SOURCE), None)
        destination_leaf = next((raw for code, raw in pairs
                                 if code == MEMBER_DESTINATION), None)
        relation_type = next((raw for code, raw in pairs
                              if code == MEMBER_RELATION_TYPE), None)
        pair_refs = [int.from_bytes(raw, "big") for code, raw in pairs
                     if code == MEMBER_PAIR]
        canonical_values = [int.from_bytes(raw, "big") for code, raw in pairs
                            if code == MEMBER_PAIR_ADDRESS]
        mirrors = [int.from_bytes(raw, "big") for code, raw in pairs
                   if code == MEMBER_MIRROR]
        cycle_values = [raw for code, raw in pairs
                        if code == MEMBER_CYCLE]
        vertices = [int.from_bytes(raw, "big") for code, raw in pairs
                    if code == MEMBER_VERTEX]
        provenance = next((raw for code, raw in pairs
                           if code == MEMBER_PROVENANCE), None)
        orientations = [int.from_bytes(raw, "big") for code, raw in pairs
                        if code == MEMBER_ORIENTATION]
        complete = bool(
            fact_leaf is not None and source_leaf is not None
            and destination_leaf is not None and relation_type is not None
            and len(pair_refs) == 1 and len(canonical_values) == 1
            and len(mirrors) == 1 and len(cycle_values) == 1
            and len(vertices) == 3 and provenance is not None
            and len(orientations) == 1)
        pair = None
        if len(vertices) == 3:
            pair = (int(vertices[1]), int(vertices[2]))
        return {
            "pairs": pairs,
            "member_details": _member_details(pairs),
            "arity": int(_field(view, "arity", len(pairs)) or len(pairs)),
            "hyperedge_id": str(_field(view, "hyperedge_id", "") or ""),
            "fact_leaf": _leaf16(fact_leaf) if fact_leaf is not None else None,
            "source_leaf": _leaf16(source_leaf)
            if source_leaf is not None else None,
            "destination_leaf": _leaf16(destination_leaf)
            if destination_leaf is not None else None,
            "relation_type": relation_type,
            "pair_ref": pair_refs[0] if len(pair_refs) == 1 else None,
            "canonical_pair": canonical_values[0]
            if len(canonical_values) == 1 else None,
            "mirror": mirrors[0] if len(mirrors) == 1 else None,
            "cycle_leaf": _leaf16(cycle_values[0])
            if len(cycle_values) == 1 else None,
            "vertices": tuple(vertices),
            "pair": pair,
            "provenance": provenance,
            "orientation": orientations[0] if len(orientations) == 1 else 0,
            "complete": complete,
        }

    def _looks_like_retrieval(self, view: Any) -> bool:
        codes = {code for code, _ in _member_pairs(view)}
        return bool(codes & {MEMBER_FACT, MEMBER_SOURCE, MEMBER_DESTINATION,
                             MEMBER_RELATION_TYPE, MEMBER_PROVENANCE,
                             MEMBER_ORIENTATION, MEMBER_PAIR_ADDRESS})

    def _record_from_parsed(self, parsed: Mapping[str, Any], *,
                            relation_record: Any = None,
                            payload: Mapping[str, Any] | None = None,
                            source_value: Any = None,
                            relation_value: Any = None,
                            destination_value: Any = None,
                            provenance_value: Any = None
                            ) -> RetrievalFact:
        pair = parsed.get("pair") or (0, 0)
        payload = payload or {}
        source = payload.get("source", source_value)
        relation = payload.get("relation", relation_value)
        destination = payload.get("destination", destination_value)
        provenance = payload.get("provenance", provenance_value)
        resolved = bool(payload) or all(
            value is not None for value in
            (source_value, relation_value, destination_value))
        relation_ref = _field_any(relation_record, ("relation_id",)) \
            if relation_record is not None else None
        fact_value = payload.get("fact_id")
        if not isinstance(fact_value, str):
            if source is not None and relation is not None \
                    and destination is not None:
                try:
                    fact_value = fact_id(source, relation, destination)
                except Exception:
                    fact_value = None
            else:
                fact_value = str(relation_ref) if relation_ref else None
        provenance_bytes = parsed.get("provenance")
        provenance_hex = provenance_bytes.hex() \
            if isinstance(provenance_bytes, (bytes, bytearray)) else None
        concept_membership = None
        if source is not None and destination is not None:
            try:
                concept_membership = pair_membership(source, destination,
                                                     self.nodes)
            except Exception:
                concept_membership = None
        anchor_value = payload.get("anchor_pair")
        if not isinstance(anchor_value, (list, tuple)):
            anchor_value = [int(pair[0]), int(pair[1])]
        concept_value = payload.get("concept_pair")
        if not isinstance(concept_value, (list, tuple)) \
                and concept_membership is not None:
            concept_value = list(concept_membership["concept_pair"])
        concept_cycle = payload.get("concept_cycle")
        if not isinstance(concept_cycle, (list, tuple)) \
                and concept_membership is not None:
            concept_cycle = list(concept_membership["concept_cycle"])
        record = RetrievalFact({
            "schema": SCHEMA,
            "kind": "retrieval-fact",
            "source": source,
            "relation": relation,
            "destination": destination,
            "provenance": provenance,
            "source_id": source,
            "destination_id": destination,
            "fact_id": fact_value,
            "relation_ref": str(relation_ref) if relation_ref else None,
            "hyperedge_id": parsed.get("hyperedge_id") or None,
            "fact_leaf_id": parsed["fact_leaf"].hex()
            if parsed.get("fact_leaf") else None,
            "source_leaf_id": parsed["source_leaf"].hex()
            if parsed.get("source_leaf") else None,
            "destination_leaf_id": parsed["destination_leaf"].hex()
            if parsed.get("destination_leaf") else None,
            "relation_type_id": parsed["relation_type"].hex()
            if parsed.get("relation_type") else None,
            "cycle_leaf_id": parsed["cycle_leaf"].hex()
            if parsed.get("cycle_leaf") else None,
            "anchor_pair": [int(anchor_value[0]), int(anchor_value[1])],
            "pair": [int(pair[0]), int(pair[1])],
            "pair_ref": parsed.get("pair_ref"),
            "pair_index": parsed.get("pair_ref"),
            "canonical_pair": parsed.get("canonical_pair"),
            "mirror": parsed.get("mirror"),
            "mirror_pair": parsed.get("mirror"),
            "cycle": [0, int(pair[0]), int(pair[1])],
            "cycle_id": "0:%d:%d" % (int(pair[0]), int(pair[1])),
            "concept_pair": list(concept_value)
            if isinstance(concept_value, (list, tuple)) else None,
            "concept_pair_ref": payload.get(
                "concept_pair_ref",
                concept_membership["concept_pair_ref"]
                if concept_membership is not None else None),
            "concept_canonical_pair": payload.get(
                "concept_canonical_pair",
                concept_membership["concept_canonical_pair"]
                if concept_membership is not None else None),
            "concept_mirror": payload.get(
                "concept_mirror",
                concept_membership["concept_mirror"]
                if concept_membership is not None else None),
            "concept_cycle": list(concept_cycle)
            if isinstance(concept_cycle, (list, tuple)) else None,
            "concept_cycle_id": payload.get("concept_cycle_id",
                                            concept_membership["concept_cycle_id"]
                                            if concept_membership is not None
                                            else None),
            "concept_cycle_leaf_id": payload.get(
                "concept_cycle_leaf",
                concept_membership["concept_cycle_leaf"].hex()
                if concept_membership is not None else None),
            "orientation": "forward" if parsed.get("orientation") else "mirror",
            "provenance_digest": provenance_hex,
            "source_digest": provenance_hex,
            "members": list(parsed.get("member_details", ())),
            "resolved": resolved,
            "missing": False,
            "canonical_merge": False,
        })
        return record

    # -- capacity checks ------------------------------------------------
    def _check_capacity(self, *, new_relations: int, new_hyperedges: int,
                        members_per_edge: int = 13, new_leaves: int = 0,
                        new_chain: int = 0) -> None:
        status = self._relation_status()
        for count_key, capacity_key in (
                ("relations", "relation_capacity"),
                ("hyperedges", "hyperedge_capacity")):
            count = status.get(count_key)
            capacity = status.get(capacity_key)
            if isinstance(count, int) and isinstance(capacity, int):
                needed = new_relations if count_key == "relations" \
                    else new_hyperedges
                if count + needed > capacity:
                    raise RetrievalBoundsError(
                        "%s capacity exhausted" % count_key,
                        code="%s_capacity" % count_key,
                        details={"count": count, "capacity": capacity,
                                 "requested": needed})
        arena_used = status.get("arena_used")
        arena_bytes = status.get("arena_bytes")
        if isinstance(arena_used, int) and isinstance(arena_bytes, int):
            needed = new_hyperedges * _align8(
                members_per_edge * (_ID_BYTES + 1))
            if arena_used + needed > arena_bytes:
                raise RetrievalBoundsError(
                    "hyperedge member arena exhausted",
                    code="arena_capacity",
                    details={"used": arena_used, "required": needed,
                             "capacity": arena_bytes})
        chain_used = status.get("chain_used")
        chain_capacity = status.get("chain_capacity")
        if isinstance(chain_used, int) and isinstance(chain_capacity, int):
            if chain_used + new_chain > chain_capacity:
                raise RetrievalBoundsError(
                    "leaf relation chain capacity exhausted",
                    code="chain_capacity",
                    details={"used": chain_used,
                             "required": new_chain,
                             "capacity": chain_capacity})
        if new_leaves:
            count = self._leaf_count()
            capacity = self._leaf_capacity()
            if isinstance(count, int) and isinstance(capacity, int):
                if count + new_leaves > capacity:
                    raise RetrievalBoundsError(
                        "leaf capacity exhausted", code="leaf_capacity",
                        details={"count": count, "capacity": capacity,
                                 "requested": new_leaves})

    def _missing_leaves(self, prepared: Mapping[str, Any]) -> list[bytes]:
        leaf_ids = [prepared["source_leaf"], prepared["destination_leaf"],
                    prepared["membership"]["cycle_leaf"],
                    prepared["fact_leaf"]]
        return [leaf_id for leaf_id in dict.fromkeys(leaf_ids)
                if self._find_leaf(leaf_id) is None]

    def _ensure_prepared_leaves(self, prepared: Mapping[str, Any]) -> None:
        source_digest = prepared["provenance_digest"]
        membership = prepared["membership"]
        pair_ref = int(membership["pair_ref"])
        self._ensure_leaf(
            prepared["source_leaf"], "concept",
            prepared["source_text_payload"],
            {"concept": prepared["source_text"]},
            prepared["source_head"], source_digest, pair_ref)
        self._ensure_leaf(
            prepared["destination_leaf"], "concept",
            prepared["destination_text_payload"],
            {"concept": prepared["destination_text"]},
            prepared["destination_concept_head"], source_digest, pair_ref)
        self._ensure_leaf(
            membership["cycle_leaf"], "hyperedge",
            prepared["cycle_text_payload"],
            {"cycle_id": membership["cycle_id"],
             "pair_ref": pair_ref,
             "canonical_pair": int(membership["canonical_pair"]),
             "mirror": int(membership["mirror"])},
            prepared["source_head"], source_digest, pair_ref)
        self._ensure_leaf(
            prepared["fact_leaf"], "hyperedge", prepared["fact_text"],
            {"fact_id": prepared["fact_id"],
             "source": prepared["source_text"],
             "relation": prepared["relation_text"],
             "destination": prepared["destination_text"]},
            prepared["source_head"], source_digest, pair_ref)

    def _existing_fact(self, prepared: Mapping[str, Any]):
        record = self._get_relation(prepared["relation_key"])
        if record is None:
            return None, None, None
        edge_identifier = _field_any(
            record, ("hyperedge_id", "hyperedge"))
        if not edge_identifier:
            raise RetrievalCorruptionError(
                "retrieval relation %s has no hyperedge"
                % prepared["fact_id"], code="missing_hyperedge",
                details={"fact_id": prepared["fact_id"]})
        view = self._get_hyperedge(edge_identifier)
        parsed = self._parse_fact_view(view)
        return record, view, parsed

    def _duplicate_receipt(self, prepared: Mapping[str, Any], record: Any,
                           parsed: Mapping[str, Any]) -> dict[str, Any]:
        if parsed is None or not parsed.get("complete"):
            raise RetrievalCorruptionError(
                "retrieval relation %s has a malformed hyperedge"
                % prepared["fact_id"], code="malformed_hyperedge",
                details={"fact_id": prepared["fact_id"]})
        if _core_pairs(prepared["members"]) != _core_pairs(parsed["pairs"]):
            raise RetrievalConflictError(
                "retrieval id already names different members",
                code="fact_collision",
                details={"fact_id": prepared["fact_id"]})
        if (_field(record, "subject") != prepared["subject_ref"].hex()
                or _field(record, "predicate") != prepared["predicate_ref"].hex()
                or _field(record, "object") != prepared["object_ref"].hex()):
            raise RetrievalConflictError(
                "retrieval relation id already names a different triple",
                code="relation_collision",
                details={"fact_id": prepared["fact_id"]})
        payload = self._leaf_text_payload(parsed["fact_leaf"])
        fact = self._record_from_parsed(
            parsed, relation_record=record, payload=payload,
            source_value=prepared["source_text"],
            relation_value=prepared["relation_text"],
            destination_value=prepared["destination_text"],
            provenance_value=prepared["provenance_value"])
        out = fact.to_dict()
        out.update({
            "status": "duplicate",
            "created": False,
            "duplicate": True,
            "conflict": False,
            "source_mismatch": parsed["provenance"]
            != prepared["provenance_digest"],
            "requested_provenance": prepared["provenance_value"],
            "canonical_merge": False,
        })
        return out

    def _append_prepared(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_prepared_leaves(prepared)
        append = _host_attr(self.relations, "append_relation", None)
        if not callable(append):
            append = _host_attr(self.relations, "append_hyperedge", None)
        if not callable(append):
            raise RetrievalHypergraphError(
                "relation store does not support append_relation",
                code="unsupported_store")
        try:
            result = append(
                subject=prepared["subject_ref"],
                predicate=prepared["predicate_ref"],
                object=prepared["object_ref"],
                polarity="supports",
                confidence=1.0,
                pair_id=prepared["membership"]["pair_ref"],
                mirror_id=prepared["membership"]["mirror"],
                source_digest=prepared["provenance_digest"],
                relation_id=prepared["relation_key"],
                hyperedge_id=prepared["edge_key"],
                hyperedge_members=prepared["members"],
            )
        except RelationConflictError as exc:
            raise RetrievalConflictError(
                "retrieval hyperedge id already names different members",
                code="hyperedge_collision",
                details={"fact_id": prepared["fact_id"]}) from exc
        except RelationBoundsError as exc:
            raise RetrievalBoundsError(
                "retrieval hyperedge could not be appended: %s" % exc,
                code="hyperedge_capacity",
                details={"fact_id": prepared["fact_id"]}) from exc
        except TypeError:
            # Very small duck-typed stores may expose only a positional
            # append_hyperedge(hyperedge_id, source, members) method.
            if not callable(_host_attr(self.relations, "append_hyperedge")):
                raise
            result = self.relations.append_hyperedge(
                prepared["edge_key"], prepared["provenance_digest"],
                prepared["members"])
        record, view, parsed = self._existing_fact(prepared)
        if record is None or view is None or parsed is None \
                or not parsed.get("complete"):
            raise RetrievalCorruptionError(
                "retrieval fact was not persisted",
                code="missing_fact", details={"fact_id": prepared["fact_id"]})
        if _core_pairs(prepared["members"]) != _core_pairs(parsed["pairs"]):
            raise RetrievalCorruptionError(
                "persisted retrieval members do not match the receipt",
                code="member_readback_mismatch",
                details={"fact_id": prepared["fact_id"]})
        if parsed["provenance"] != prepared["provenance_digest"]:
            raise RetrievalCorruptionError(
                "persisted provenance does not match the receipt",
                code="provenance_readback_mismatch",
                details={"fact_id": prepared["fact_id"]})
        payload = self._leaf_text_payload(parsed["fact_leaf"])
        fact = self._record_from_parsed(
            parsed, relation_record=record, payload=payload,
            source_value=prepared["source_text"],
            relation_value=prepared["relation_text"],
            destination_value=prepared["destination_text"],
            provenance_value=prepared["provenance_value"])
        self._maybe_commit("retrieval-fact", prepared["fact_id"])
        out = fact.to_dict()
        out.update({
            "status": "appended",
            "created": True,
            "duplicate": False,
            "conflict": False,
            "source_mismatch": False,
            "requested_provenance": prepared["provenance_value"],
            "canonical_merge": False,
        })
        return out

    # -- commit hook ----------------------------------------------------
    def _maybe_commit(self, kind: str, identity: str) -> None:
        callback = self._commit
        if callback is None:
            return
        try:
            signature = inspect.signature(callback)
            positional = [
                parameter for parameter in signature.parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD)]
            varargs = any(parameter.kind == parameter.VAR_POSITIONAL
                          for parameter in signature.parameters.values())
            if varargs or len(positional) >= 2:
                callback(kind, identity)
            elif len(positional) == 1:
                callback(kind)
            else:
                callback()
        except (TypeError, ValueError):
            try:
                callback(kind, identity)
            except TypeError:
                callback()
        flush = self._flush
        if flush is not None and flush is not callback:
            try:
                flush()
            except Exception:
                pass

    # -- public writes --------------------------------------------------
    def add_fact(self, source_id: Any, relation: Any, destination_id: Any,
                 provenance: Any) -> dict[str, Any]:
        """Persist one typed fact and return a receipt.

        A repeated identical fact is an idempotent duplicate.  A second
        provenance for the same fact is also a duplicate
        (``source_mismatch=True``) and never overwrites the original payload.
        """
        prepared = self._prepare_fact(
            source_id, relation, destination_id, provenance)
        record, _view, parsed = self._existing_fact(prepared)
        if record is not None:
            if parsed is None:
                raise RetrievalCorruptionError(
                    "retrieval relation exists without a readable hyperedge",
                    code="missing_hyperedge",
                    details={"fact_id": prepared["fact_id"]})
            return self._duplicate_receipt(prepared, record, parsed)
        self._check_prepared_leaves(prepared)
        missing = self._missing_leaves(prepared)
        self._check_capacity(
            new_relations=1, new_hyperedges=1,
            members_per_edge=len(prepared["members"]),
            new_leaves=len(missing), new_chain=3)
        return self._append_prepared(prepared)

    add = add_fact
    put = add_fact
    save_fact = add_fact

    def add_facts(self, facts: Iterable[Any], provenance: Any = None, *,
                  limit: int | None = None) -> dict[str, Any]:
        """Persist a bounded batch of facts with an all-or-nothing preflight."""
        items = _batch_items(facts, provenance)
        maximum = MAX_BATCH if limit is None else _bounded(
            limit, "limit", 1, MAX_BATCH)
        if len(items) > maximum:
            raise RetrievalBoundsError(
                "fact batch exceeds the bounded batch size",
                code="batch_too_large",
                details={"items": len(items), "limit": maximum})
        if not items:
            return {"status": "empty", "accepted": 0, "created": 0,
                    "duplicates": 0, "count": 0, "facts": [],
                    "bounded": True, "canonical_merge": False}
        prepared_items: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        duplicates = 0
        for item in items:
            prepared = self._prepare_fact(
                item["source_id"], item["relation"],
                item["destination_id"], item["provenance"])
            identity = prepared["identity"]
            if identity in seen:
                duplicates += 1
                continue
            seen[identity] = prepared
            prepared_items.append(prepared)
        fresh: list[dict[str, Any]] = []
        missing_leaves: dict[bytes, None] = {}
        for prepared in prepared_items:
            record, _view, parsed = self._existing_fact(prepared)
            if record is not None:
                if parsed is None or not parsed.get("complete"):
                    raise RetrievalCorruptionError(
                        "retrieval relation exists without a readable hyperedge",
                        code="missing_hyperedge",
                        details={"fact_id": prepared["fact_id"]})
                if _core_pairs(prepared["members"]) != _core_pairs(
                        parsed["pairs"]):
                    raise RetrievalConflictError(
                        "retrieval id already names different members",
                        code="fact_collision",
                        details={"fact_id": prepared["fact_id"]})
                duplicates += 1
                continue
            self._check_prepared_leaves(prepared)
            fresh.append(prepared)
            for leaf_id in self._missing_leaves(prepared):
                missing_leaves[leaf_id] = None
        if fresh:
            self._check_capacity(
                new_relations=len(fresh), new_hyperedges=len(fresh),
                members_per_edge=len(fresh[0]["members"]),
                new_leaves=len(missing_leaves), new_chain=3 * len(fresh))
        receipts: list[dict[str, Any]] = []
        for prepared in fresh:
            record, _view, parsed = self._existing_fact(prepared)
            if record is not None:
                receipts.append(self._duplicate_receipt(
                    prepared, record, parsed))
            else:
                receipts.append(self._append_prepared(prepared))
        accepted = sum(1 for receipt in receipts
                       if receipt.get("created"))
        duplicates += sum(1 for receipt in receipts
                          if receipt.get("duplicate"))
        return {
            "status": "committed" if accepted else "duplicate",
            "accepted": accepted,
            "created": accepted,
            "duplicates": duplicates,
            "count": len(receipts),
            "facts": receipts,
            "bounded": True,
            "limit": maximum,
            "canonical_merge": False,
        }

    add_fact_batch = add_facts
    add_batch = add_facts
    batch_add = add_facts
    add_many = add_facts
    persist = add_facts

    # -- public reads ---------------------------------------------------
    def get(self, source_id: Any, relation: Any = None,
            destination_id: Any = None, default: Any = None) -> Any:
        """Exact lookup by ``(source, relation, destination)``."""
        source_id, relation, destination_id = _unpack_fact_args(
            source_id, relation, destination_id)
        prepared = self._prepare_fact(
            source_id, relation, destination_id, "lookup")
        record, _view, parsed = self._existing_fact(prepared)
        if record is None:
            return default
        if parsed is None or not parsed.get("complete"):
            raise RetrievalCorruptionError(
                "retrieval relation exists without a readable hyperedge",
                code="malformed_hyperedge",
                details={"fact_id": prepared["fact_id"]})
        if _core_pairs(prepared["members"]) != _core_pairs(parsed["pairs"]):
            raise RetrievalCorruptionError(
                "stored retrieval members do not match the requested fact",
                code="member_mismatch",
                details={"fact_id": prepared["fact_id"]})
        payload = self._leaf_text_payload(parsed["fact_leaf"])
        return self._record_from_parsed(
            parsed, relation_record=record, payload=payload,
            source_value=prepared["source_text"],
            relation_value=prepared["relation_text"],
            destination_value=prepared["destination_text"])

    lookup = get
    read = get
    load = get

    def __contains__(self, fact: Any) -> bool:
        try:
            if isinstance(fact, Mapping):
                source = fact.get("source", fact.get("source_id"))
                relation = fact.get("relation")
                destination = fact.get("destination",
                                       fact.get("destination_id"))
            elif isinstance(fact, (tuple, list)) and len(fact) == 3:
                source, relation, destination = fact
            else:
                return False
            return self.get(source, relation, destination) is not None
        except Exception:
            return False

    def neighbors(self, concept_id: Any, relation: Any = None,
                  direction: str = "out", limit: int = DEFAULT_NEIGHBOR_LIMIT,
                  *, scan_limit: int | None = None) -> RetrievalNeighbors:
        """Bounded typed traversal of facts incident to one concept."""
        concept = _identifier_text(concept_id, "concept id")
        if limit is None:
            limit = DEFAULT_NEIGHBOR_LIMIT
        if isinstance(direction, bool):
            raise RetrievalHypergraphError(
                "direction must be 'out', 'in' or 'both'",
                code="invalid_direction")
        if isinstance(direction, int):
            # Accept neighbors(concept, relation, limit) as a convenience.
            if limit == DEFAULT_NEIGHBOR_LIMIT:
                limit = direction
            direction = "out"
        if direction is None:
            direction = "out"
        limit = _bounded(limit, "limit", 1, MAX_NEIGHBOR_LIMIT)
        if direction not in ("out", "in", "both"):
            raise RetrievalHypergraphError(
                "direction must be 'out', 'in' or 'both'",
                code="invalid_direction")
        relation_value = None if relation is None else relation_text(relation)
        predicate = None if relation_value is None \
            else relation_type_id(relation_value)
        roles = ("subject",) if direction == "out" else \
            ("object",) if direction == "in" else ("subject", "object")
        scan = _scan_limit(limit, scan_limit, MAX_AUDIT_LIMIT)
        results: list[RetrievalFact] = []
        seen: set[str] = set()
        truncated = False
        ref = _leaf_ref(concept_leaf_id(concept))
        for role in roles:
            records = self._walk_role(ref, role, scan)
            for record in records:
                if predicate is not None \
                        and _field(record, "predicate") != predicate.hex():
                    continue
                identifier = _field_any(record, ("relation_id",))
                if identifier is not None and str(identifier) in seen:
                    continue
                edge_identifier = _field_any(record, ("hyperedge_id",))
                if not edge_identifier:
                    continue
                view = self._get_hyperedge(edge_identifier)
                parsed = self._parse_fact_view(view)
                if parsed is None or not parsed.get("complete"):
                    continue
                key = str(identifier) if identifier is not None \
                    else parsed["fact_leaf"].hex()
                if key in seen:
                    continue
                if len(results) >= limit:
                    truncated = True
                    break
                payload = self._leaf_text_payload(parsed["fact_leaf"])
                results.append(self._record_from_parsed(
                    parsed, relation_record=record, payload=payload,
                    relation_value=relation_value))
                seen.add(key)
            if truncated:
                break
        return RetrievalNeighbors(
            results, concept=concept, relation=relation_value,
            direction=direction, limit=limit, truncated=truncated,
            missing=not results, bounded=True, scanned=scan)

    neighbours = neighbors

    def _walk_role(self, ref: bytes, role: str, limit: int) -> list[Any]:
        walker = _host_attr(self.relations, "chain_walk", None)
        if callable(walker):
            try:
                result = walker(ref, role=role, limit=limit)
            except TypeError:
                result = walker(ref, role, limit)
            if isinstance(result, Mapping):
                return list(result.get("relations", ()))
            if isinstance(result, (list, tuple)):
                return list(result)
        fallback: list[Any] = []
        for record in self._iter_relations(limit):
            value = _field(record, role)
            if value == ref.hex() or value == ref:
                fallback.append(record)
            if len(fallback) >= limit:
                break
        return fallback

    def two_hop(self, source_id: Any, relation: Any = None,
                second_relation: Any = None, *,
                via: Any = None, destination: Any = None,
                target: Any = None,
                limit: int = DEFAULT_TWO_HOP_LIMIT,
                max_mid: int | None = None,
                include_cycles: bool = False) -> RetrievalPaths:
        """Bounded two-hop traversal ``source -> via -> destination``."""
        source_text = _identifier_text(source_id, "source_id")
        if limit is None:
            limit = DEFAULT_TWO_HOP_LIMIT
        limit = _bounded(limit, "limit", 1, MAX_TWO_HOP_LIMIT)
        first_relation = relation
        second = second_relation
        if isinstance(relation, (tuple, list)) and not isinstance(
                relation, (str, bytes, bytearray)) and len(relation) == 2:
            first_relation, second = relation[0], relation[1]
        first_text = None if first_relation is None \
            else relation_text(first_relation)
        second_text = None if second is None else relation_text(second)
        mid_limit = min(MAX_NEIGHBOR_LIMIT,
                        max(limit + 1, limit * 2)) \
            if max_mid is None else _bounded(
                max_mid, "max_mid", 1, MAX_NEIGHBOR_LIMIT)
        mid_limit = min(max(mid_limit, limit), MAX_NEIGHBOR_LIMIT)
        via_text = None if via is None else _identifier_text(via, "via")
        target_value = destination if destination is not None else target
        target_text = None if target_value is None \
            else _identifier_text(target_value, "destination")
        firsts = self.neighbors(
            source_text, first_text, "out", limit=mid_limit)
        paths: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        truncated = bool(getattr(firsts, "truncated", False))
        for first in firsts:
            if first.get("destination") is None:
                continue
            if via_text is not None and str(first.get("destination")) != via_text:
                continue
            middles = self.neighbors(
                first["destination"], second_text, "out", limit=mid_limit)
            if getattr(middles, "truncated", False):
                truncated = True
            for second_fact in middles:
                if target_text is not None \
                        and str(second_fact.get("destination")) != target_text:
                    continue
                if not include_cycles \
                        and str(second_fact.get("destination")) == source_text:
                    continue
                key = (str(first.get("fact_id")),
                       str(second_fact.get("fact_id")))
                if key in seen:
                    continue
                if len(paths) >= limit:
                    truncated = True
                    break
                seen.add(key)
                paths.append({
                    "schema": SCHEMA,
                    "kind": "two-hop",
                    "source": source_text,
                    "via": first.get("destination"),
                    "destination": second_fact.get("destination"),
                    "relations": [first.get("relation"),
                                  second_fact.get("relation")],
                    "fact_ids": [first.get("fact_id"),
                                 second_fact.get("fact_id")],
                    "first": first.to_dict(),
                    "second": second_fact.to_dict(),
                    "hops": 2,
                    "canonical_merge": False,
                })
            if truncated:
                break
        return RetrievalPaths(
            paths, source=source_text, relation=first_text,
            second_relation=second_text, limit=limit,
            truncated=truncated, missing=not firsts, bounded=True)

    twohop = two_hop

    def snapshot(self, limit: int = DEFAULT_SNAPSHOT_LIMIT, *,
                 scan_limit: int | None = None,
                 full: bool = False) -> dict[str, Any]:
        """Return a bounded JSON projection of stored retrieval facts."""
        _ = full  # accepted for call-site symmetry; bounds still apply
        if limit is None:
            limit = DEFAULT_SNAPSHOT_LIMIT
        limit = _bounded(limit, "limit", 1, MAX_SNAPSHOT_LIMIT)
        scan = _scan_limit(limit, scan_limit, MAX_AUDIT_LIMIT)
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        scanned = 0
        truncated = False
        for record in self._iter_relations(scan):
            scanned += 1
            edge_identifier = _field_any(record, ("hyperedge_id",))
            if not edge_identifier:
                continue
            view = self._get_hyperedge(edge_identifier)
            parsed = self._parse_fact_view(view)
            if parsed is None or not parsed.get("complete"):
                continue
            key = str(_field_any(record, ("relation_id",))
                      or parsed["fact_leaf"].hex())
            if key in seen:
                continue
            if len(facts) >= limit:
                truncated = True
                break
            seen.add(key)
            payload = self._leaf_text_payload(parsed["fact_leaf"])
            facts.append(self._record_from_parsed(
                parsed, relation_record=record, payload=payload).to_dict())
        relation_count = self._relation_count()
        if relation_count is not None and scanned < relation_count:
            truncated = True
        return {
            "schema": SCHEMA,
            "ok": True,
            "valid": True,
            "facts": facts,
            "fact_count": len(facts),
            "count": len(facts),
            "returned": len(facts),
            "limit": limit,
            "scan_limit": scan,
            "scanned_relations": scanned,
            "relation_count": relation_count,
            "hyperedge_count": self._hyperedge_count(),
            "leaf_count": self._leaf_count(),
            "truncated": truncated,
            "bounded": True,
            "canonical_merge": False,
            "generation": self._generation(),
            "nodes": self.nodes,
        }

    def audit(self, limit: int = DEFAULT_AUDIT_LIMIT, *,
              scan_limit: int | None = None,
              full: bool = False) -> dict[str, Any]:
        """Read-only bounded check of identities, membership and provenance."""
        _ = full  # accepted for call-site symmetry; bounds still apply
        if limit is None:
            limit = DEFAULT_AUDIT_LIMIT
        limit = _bounded(limit, "limit", 1, MAX_AUDIT_LIMIT)
        scan = _bounded(scan_limit if scan_limit is not None else limit,
                        "scan_limit", 1, MAX_AUDIT_LIMIT)
        problems: list[str] = []
        checked = 0
        scanned = 0
        seen: set[str] = set()
        for record in self._iter_relations(scan):
            scanned += 1
            edge_identifier = _field_any(record, ("hyperedge_id",))
            if not edge_identifier:
                continue
            view = self._get_hyperedge(edge_identifier)
            if view is None:
                if self._looks_like_retrieval_candidate(record):
                    problems.append(
                        "retrieval relation %r is missing its hyperedge"
                        % _field_any(record, ("relation_id",)))
                continue
            parsed = self._parse_fact_view(view)
            if parsed is None:
                if self._looks_like_retrieval(view):
                    problems.append(
                        "malformed retrieval hyperedge %r"
                        % _field_any(view, ("hyperedge_id",)))
                continue
            if not parsed.get("complete"):
                problems.append(
                    "incomplete retrieval hyperedge %r"
                    % _field_any(view, ("hyperedge_id",)))
                continue
            checked += 1
            label = str(_field_any(record, ("relation_id",))
                        or _field_any(view, ("hyperedge_id",)))
            if label in seen:
                problems.append("duplicate retrieval relation %s" % label)
            seen.add(label)
            self._audit_one(record, view, parsed, problems, label)
        relation_count = self._relation_count()
        truncated = relation_count is not None and scanned < relation_count
        ok = not problems
        return {
            "schema": SCHEMA,
            "ok": ok,
            "valid": ok,
            "accepted": ok,
            "problems": problems,
            "issues": list(problems),
            "facts_checked": checked,
            "checked": checked,
            "scanned_relations": scanned,
            "relation_count": relation_count,
            "hyperedge_count": self._hyperedge_count(),
            "leaf_count": self._leaf_count(),
            "limit": limit,
            "scan_limit": scan,
            "truncated": truncated,
            "bounded": True,
            "canonical_merge": False,
            "no_raw_pointers": not any("pointer" in item.casefold()
                                       for item in problems),
            "no_floats": not any("float" in item.casefold()
                                 for item in problems),
            "generation": self._generation(),
            "nodes": self.nodes,
            "stores": {
                "relations": type(self.relations).__name__,
                "leaves": type(self.leaves).__name__,
                "cells": None if self.cells is None
                else type(self.cells).__name__,
            },
        }

    def _looks_like_retrieval_candidate(self, record: Any) -> bool:
        predicate = _field(record, "predicate")
        if not isinstance(predicate, str):
            return False
        try:
            return predicate == relation_type_id("1").hex()
        except Exception:
            return False

    def _audit_one(self, record: Any, view: Any, parsed: Mapping[str, Any],
                   problems: list[str], label: str) -> None:
        if parsed["provenance"] in (None, ZERO_ID):
            problems.append("missing provenance for %s" % label)
        if parsed["fact_leaf"] is None:
            problems.append("missing fact leaf for %s" % label)
        else:
            if self._find_leaf(parsed["fact_leaf"]) is None:
                problems.append("dangling fact leaf for %s" % label)
        for name in ("source_leaf", "destination_leaf", "cycle_leaf"):
            leaf = parsed.get(name)
            if leaf is None:
                problems.append("missing %s for %s" % (name, label))
            elif self._find_leaf(leaf) is None:
                problems.append("dangling %s for %s" % (name, label))
        vertices = parsed.get("vertices") or ()
        if len(vertices) != 3 or vertices[0] != 0 \
                or len(set(vertices)) != 3:
            problems.append("invalid cycle vertices for %s" % label)
            return
        first, second = int(vertices[1]), int(vertices[2])
        expected_canonical = canonical_address(first, second)
        expected_mirror = mirror_address(expected_canonical)
        expected_pair_ref = cell_pair_index(first, second)
        if parsed["pair_ref"] != expected_pair_ref:
            problems.append("mismatched pair reference for %s" % label)
        if parsed["canonical_pair"] != expected_canonical:
            problems.append("mismatched canonical pair for %s" % label)
        if parsed["mirror"] != expected_mirror:
            problems.append("mismatched mirror pair for %s" % label)
        if parsed["cycle_leaf"] != cycle_leaf_id(first, second):
            problems.append("mismatched cycle leaf for %s" % label)
        if parsed["orientation"] not in (0, 1):
            problems.append("invalid mirror orientation for %s" % label)
        cycle_payload = self._leaf_text_payload(parsed["cycle_leaf"]) \
            if parsed.get("cycle_leaf") else None
        if cycle_payload is not None:
            if cycle_payload.get("cycle_id") != "0:%d:%d" % (first, second):
                problems.append("cycle leaf payload mismatch for %s" % label)
            if cycle_payload.get("pair_ref") != expected_pair_ref:
                problems.append(
                    "cycle leaf pair reference mismatch for %s" % label)
        payload = self._leaf_text_payload(parsed["fact_leaf"])
        if payload is None:
            return
        unsafe = _unsafe_payload_paths(payload)
        for location in unsafe:
            problems.append("pointer-shaped payload key %s in %s"
                            % (location, label))
        if payload.get("schema") != FACT_SCHEMA:
            problems.append("unexpected fact schema for %s" % label)
            return
        try:
            source_text = _identifier_text(payload.get("source"), "source")
            relation_value = relation_text(payload.get("relation"))
            destination_text = _identifier_text(
                payload.get("destination"), "destination")
            provenance_value = payload.get("provenance")
            provenance_bytes = provenance_digest(provenance_value)
        except RetrievalHypergraphError as exc:
            problems.append("invalid fact payload for %s: %s" % (label, exc))
            return
        membership = pair_membership(source_text, destination_text,
                                     self.nodes)
        expected_orientation = 1 if membership["orientation"] == "forward" \
            else 0
        if parsed["orientation"] != expected_orientation:
            problems.append("mismatched mirror orientation for %s" % label)
        expected_fact_id = fact_id(source_text, relation_value,
                                   destination_text)
        if payload.get("fact_id") != expected_fact_id:
            problems.append("non-deterministic fact id for %s" % label)
        if parsed["provenance"] != provenance_bytes:
            problems.append("provenance digest mismatch for %s" % label)
        if parsed["source_leaf"] != concept_leaf_id(source_text):
            problems.append("source concept leaf mismatch for %s" % label)
        if parsed["destination_leaf"] != concept_leaf_id(destination_text):
            problems.append("destination concept leaf mismatch for %s"
                            % label)
        if parsed["relation_type"] != relation_type_id(relation_value):
            problems.append("relation type mismatch for %s" % label)
        if tuple(parsed["vertices"]) != tuple(membership["cycle"]):
            problems.append("cycle membership mismatch for %s" % label)
        if parsed["pair_ref"] != membership["pair_ref"]:
            problems.append("pair membership mismatch for %s" % label)
        if tuple(payload.get("anchor_pair", ())) != tuple(
                membership["anchor_heads"]):
            problems.append("anchor pair payload mismatch for %s" % label)
        if tuple(payload.get("concept_pair", ())) != tuple(
                membership["concept_pair"]):
            problems.append("concept pair payload mismatch for %s" % label)
        if _field(record, "subject") != _leaf_ref(
                concept_leaf_id(source_text)).hex():
            problems.append("relation subject mismatch for %s" % label)
        if _field(record, "predicate") != relation_type_id(
                relation_value).hex():
            problems.append("relation predicate mismatch for %s" % label)
        if _field(record, "object") != _leaf_ref(
                concept_leaf_id(destination_text)).hex():
            problems.append("relation object mismatch for %s" % label)
        record_source = _field(record, "source_digest")
        if not record_source or record_source == ZERO_ID.hex():
            problems.append("missing relation provenance for %s" % label)
        elif record_source != parsed["provenance"].hex():
            problems.append("relation provenance mismatch for %s" % label)
        for field_name, expected_value in (
                ("pair_id", membership["pair_ref"]),
                ("mirror_id", membership["mirror"])):
            stored = _field(record, field_name)
            if stored is None:
                continue
            try:
                stored_int = int.from_bytes(bytes.fromhex(str(stored)), "big")
            except (TypeError, ValueError):
                problems.append("malformed %s for %s" % (field_name, label))
                continue
            if stored_int != int(expected_value):
                problems.append("%s mismatch for %s" % (field_name, label))
        expected = self._prepare_fact(
            source_text, relation_value, destination_text,
            provenance_value)["members"]
        if _core_pairs(expected) != _core_pairs(parsed["pairs"]):
            problems.append("member content mismatch for %s" % label)
        expected_identifier = _field(record, "relation_id")
        if expected_identifier not in (None, relation_id(
                source_text, relation_value, destination_text).hex()):
            problems.append("non-deterministic relation id for %s" % label)
        expected_edge = hyperedge_id(
            source_text, relation_value, destination_text).hex()
        if parsed.get("hyperedge_id") not in (None, expected_edge):
            problems.append("non-deterministic hyperedge id for %s" % label)

    def __len__(self) -> int:
        seen: set[str] = set()
        for record in self._iter_relations(MAX_LEN_SCAN):
            edge_identifier = _field_any(record, ("hyperedge_id",))
            if not edge_identifier:
                continue
            view = self._get_hyperedge(edge_identifier)
            parsed = self._parse_fact_view(view)
            if parsed is not None and parsed.get("complete"):
                key = str(_field_any(record, ("relation_id",))
                          or parsed["fact_leaf"].hex())
                seen.add(key)
        return len(seen)

    def status(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "facts": len(self),
            "relation_count": self._relation_count(),
            "hyperedge_count": self._hyperedge_count(),
            "leaf_count": self._leaf_count(),
            "leaf_capacity": self._leaf_capacity(),
            "nodes": self.nodes,
            "bounded": True,
            "canonical_merge": False,
            "generation": self._generation(),
        }

    def _generation(self) -> int | None:
        for owner in (self.host, self.relations):
            value = _host_attr(owner, "generation", None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        status = self._relation_status()
        value = status.get("generation")
        return value if isinstance(value, int) else None

    # -- reopen and denial ---------------------------------------------
    def reopen(self) -> "RetrievalHypergraph":
        """Return a fresh adapter over the reopened host or same stores."""
        host = self.host
        reopen = _host_attr(host, "reopen", None) if host is not None else None
        if callable(reopen):
            try:
                reopened = reopen()
            except TypeError:
                reopened = None
            if reopened is not None:
                host = reopened
        if host is not None and host is not self.host:
            return RetrievalHypergraph(host, nodes=self.nodes)
        return RetrievalHypergraph(
            host if host is not None else self.relations,
            relations=self.relations, leaves=self.leaves, cells=self.cells,
            nodes=self.nodes)

    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Canonical merge is deliberately disabled."""
        raise RetrievalDenied(
            "canonical retrieval merge is disabled",
            code="canonical_merge_disabled",
            details={"canonical_merge": False})

    promote = merge


# ---------------------------------------------------------------------------
# batch and argument helpers
# ---------------------------------------------------------------------------
def _batch_items(facts: Iterable[Any], provenance: Any = None
                 ) -> list[dict[str, Any]]:
    if isinstance(facts, (str, bytes, bytearray, Mapping)):
        raise RetrievalHypergraphError(
            "fact batch must be an iterable of items", code="invalid_batch")
    try:
        items = list(facts)
    except TypeError as exc:
        raise RetrievalHypergraphError(
            "fact batch must be an iterable of items",
            code="invalid_batch") from exc
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            source = item.get("source_id", item.get("source"))
            relation = item.get("relation", item.get("predicate"))
            destination = item.get("destination_id",
                                   item.get("destination",
                                            item.get("object")))
            item_provenance = item.get("provenance")
            if item_provenance is None:
                item_provenance = provenance
        elif isinstance(item, (tuple, list)):
            if len(item) == 3:
                source, relation, destination = item
                item_provenance = provenance
            elif len(item) == 4:
                source, relation, destination, item_provenance = item
            else:
                raise RetrievalHypergraphError(
                    "batch item %d must be a 3- or 4-tuple" % index,
                    code="invalid_batch_item")
        else:
            raise RetrievalHypergraphError(
                "batch item %d has an unsupported shape" % index,
                code="invalid_batch_item")
        if source is None or relation is None or destination is None:
            raise RetrievalHypergraphError(
                "batch item %d is missing source, relation or destination"
                % index, code="invalid_batch_item")
        if item_provenance is None:
            raise RetrievalProvenanceError(
                "batch item %d has no provenance" % index,
                code="provenance_required")
        result.append({
            "source_id": source,
            "relation": relation,
            "destination_id": destination,
            "provenance": item_provenance,
        })
    return result


def _unpack_fact_args(source: Any, relation: Any, destination: Any):
    if isinstance(source, Mapping) and relation is None and destination is None:
        return (source.get("source_id", source.get("source")),
                source.get("relation", source.get("predicate")),
                source.get("destination_id", source.get("destination",
                                                        source.get("object"))))
    if isinstance(source, (tuple, list)) and not isinstance(
            source, (str, bytes, bytearray)) and len(source) == 3 \
            and relation is None and destination is None:
        return source[0], source[1], source[2]
    return source, relation, destination


def _unsafe_payload_paths(value: Any, *, prefix: str = "",
                          depth: int = 0) -> list[str]:
    problems: list[str] = []
    if depth > 16:
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append("%s<non-string-key>" % (prefix + "."))
                continue
            compact = re.sub(r"[^a-z0-9]+", "", key.casefold())
            location = "%s.%s" % (prefix, key) if prefix else key
            if compact in _POINTER_KEYS or compact in _PATH_KEYS:
                problems.append(location)
            problems.extend(_unsafe_payload_paths(
                item, prefix=location, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            problems.extend(_unsafe_payload_paths(
                item, prefix="%s[%d]" % (prefix, index), depth=depth + 1))
    elif isinstance(value, float):
        problems.append("%s<float>" % (prefix + "."))
    return problems


__all__ = (
    "SCHEMA", "FACT_SCHEMA", "CONCEPT_SCHEMA", "CYCLE_SCHEMA",
    "DEFAULT_NODES",
    "DEFAULT_CAPACITY", "DEFAULT_SNAPSHOT_LIMIT", "MAX_SNAPSHOT_LIMIT",
    "DEFAULT_AUDIT_LIMIT", "MAX_AUDIT_LIMIT", "DEFAULT_NEIGHBOR_LIMIT",
    "MAX_NEIGHBOR_LIMIT", "DEFAULT_TWO_HOP_LIMIT", "MAX_TWO_HOP_LIMIT",
    "MAX_BATCH", "MEMBER_SOURCE", "MEMBER_DESTINATION",
    "MEMBER_RELATION_TYPE", "MEMBER_FACT", "MEMBER_PROVENANCE",
    "MEMBER_ORIENTATION", "MEMBER_PAIR_ADDRESS", "MEMBER_CYCLE",
    "MEMBER_VERTEX", "MEMBER_PAIR", "MEMBER_MIRROR", "MEMBER_NAMES",
    "MEMBER_CODES", "RetrievalHypergraphError", "RetrievalBoundsError",
    "RetrievalProvenanceError", "RetrievalConflictError",
    "RetrievalCorruptionError", "RetrievalNotFound", "RetrievalDenied",
    "RetrievalFact", "RetrievalNeighbors", "RetrievalPaths",
    "concept_text", "relation_text", "concept_leaf_id", "relation_type_id",
    "fact_identity", "fact_id", "hyperedge_id", "relation_id",
    "fact_leaf_id", "concept_head", "pair_heads", "anchor_pair",
    "cycle_leaf_id",
    "pair_membership", "provenance_digest", "RetrievalHypergraph",
)
