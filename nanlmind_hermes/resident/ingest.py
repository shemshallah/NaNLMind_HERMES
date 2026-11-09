"""Bounded adapter from plugin evidence into the resident universe.

The adapter accepts the two evidence shapes that already exist in Hermes:

* plugin memory rows (``kind``, ``subject``, ``predicate``, ``object``,
  ``expression``, ``provenance``, ``polarity``, ``confidence``, ``head``,
  ``layer``), optionally wrapped as ``{"item": ..., "source_digest": ...}``;
* ``nanlm.trainable.v1`` JSONL records.

Prepared records are plain JSON-compatible dictionaries.  Text-bearing rows
become immutable ``leaf`` records with a deterministic id and a bounded,
quantized feature projection.  Complete subject/predicate/object rows also
produce a ``relation`` record that references the three leaves.  Observations
only produce leaves; they never create relation records.

Nothing here performs canonical merges, network access, expression evaluation or
unbounded writes.  ``prepare_records`` is pure and is the fallback when no
``ResidentUniverse`` implementation is importable.  When one is available,
``ingest_rows``/``apply_records`` append the prepared records through its
bounded append API.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

INGEST_SCHEMA = "nanlm.resident-ingest/1"
FEATURE_SCHEMA = "nanlm.resident-features/1"
TRAINABLE_SCHEMA = "nanlm.trainable.v1"

MAX_BATCH = 4096
FEATURE_DIMS = (128, 256)
DEFAULT_FEATURE_DIM = 256
FEATURE_DIM = DEFAULT_FEATURE_DIM
FEATURE_DTYPES = ("int8", "int16")
DEFAULT_FEATURE_DTYPE = "int16"
FEATURE_DTYPE = DEFAULT_FEATURE_DTYPE
BATCH_LIMIT = MAX_BATCH
_QUANT_MAX = {"int8": 127, "int16": 32767}

# Lazily populated by ``discover_resident_universe_class``.  Keeping the name
# module-level makes the adapter easy to inject in tests without importing an
# optional implementation at module import time.
ResidentUniverse: type | None = None

MAX_HEAD = 255
MAX_LAYER = 9
DEFAULT_CONFIDENCE = 0.5
MAX_FEATURE_TEXT = 65536
MAX_FEATURE_TOKENS = 4096
MAX_FEATURE_TOKEN_CHARS = 96
MAX_EXPRESSION_DEPTH = 32

VALID_POLARITIES = ("supports", "refutes", "neutral")
OBSERVATION_KINDS = frozenset({"observation"})
DEFAULT_LAYER_BY_KIND = {
    "observation": 0,
    "entity": 1,
    "relation": 2,
    "operator": 3,
    "equation": 4,
    "constraint": 5,
    "hypothesis": 5,
    "counterexample": 9,
}

_POLARITY_ALIASES = {
    "support": "supports",
    "supports": "supports",
    "supported": "supports",
    "true": "supports",
    "yes": "supports",
    "positive": "supports",
    "refute": "refutes",
    "refutes": "refutes",
    "refuted": "refutes",
    "false": "refutes",
    "no": "refutes",
    "negative": "refutes",
    "contradicts": "refutes",
    "neutral": "neutral",
    "unknown": "neutral",
    "": "neutral",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*", re.UNICODE)


class ResidentIngestError(ValueError):
    """Raised when a row cannot be represented without losing its identity."""


# ---------------------------------------------------------------------------
# Canonical JSON, digests and normalization
# ---------------------------------------------------------------------------

def _sanitize(value: Any, depth: int = 0) -> Any:
    """Return deterministic JSON-safe data without evaluating expressions."""
    if depth > MAX_EXPRESSION_DEPTH:
        return {"truncated": True}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item, depth + 1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_sanitize(item, depth + 1) for item in value),
                      key=lambda item: canonical_json(item))
    return str(value)


def canonical_json(value: Any) -> str:
    """Canonical, stable JSON used for every identity digest."""
    return json.dumps(
        _sanitize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest_value(value: Any) -> str:
    """SHA-256 of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    """NFKC-normalize, trim and collapse whitespace without changing case."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (Mapping, list, tuple)):
        text = canonical_json(value)
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text).strip()
    return _WHITESPACE_RE.sub(" ", text)


def tokens(text: Any) -> list[str]:
    """Deterministic Unicode word tokens (case-folded)."""
    return _TOKEN_RE.findall(normalize_text(text).casefold())


def _identity_text(value: Any) -> str:
    return normalize_text(value).casefold()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def normalize_source_digest(value: Any = None, *,
                            default: Any = None,
                            derive_from: Any = None) -> str:
    """Validate/preserve a source digest, optionally deriving one locally."""
    if value is not None:
        if not _is_digest(value):
            raise ValueError("source_digest must be a SHA-256 hex digest")
        return value.lower()
    if default is not None:
        if not _is_digest(default):
            raise ValueError("default source_digest must be a SHA-256 hex digest")
        return default.lower()
    if derive_from is None:
        raise ValueError("source_digest is required")
    # Derived sources are explicit local digests, not a canonical merge.
    return digest_value({"schema": INGEST_SCHEMA, "derived_source": derive_from})


def derive_source_digest(value: Any) -> str:
    """Public helper for deterministic source identity of local evidence."""
    return digest_value({"schema": INGEST_SCHEMA, "derived_source": value})


# ---------------------------------------------------------------------------
# Deterministic ids
# ---------------------------------------------------------------------------

def _text_digest(text: Any) -> str:
    return digest_value({"text": _identity_text(text)})


def leaf_id_for_text(text: Any) -> str:
    """Return the stable leaf id for a text/expression leaf."""
    identity = _identity_text(text)
    if not identity:
        raise ResidentIngestError("leaf text must not be empty")
    return "leaf:" + _text_digest(identity)[:32]


def relation_hash_for_triple(subject: Any, predicate: Any, obj: Any) -> str:
    """Full SHA-256 identity of a subject/predicate/object relation."""
    parts = [_identity_text(subject), _identity_text(predicate),
             _identity_text(obj)]
    if not all(parts):
        raise ResidentIngestError("relation requires subject, predicate and object")
    return digest_value({"relation": {
        "subject": parts[0], "predicate": parts[1], "object": parts[2],
    }})


def relation_id_for_triple(subject: Any, predicate: Any, obj: Any) -> str:
    """Compact stable relation id used by the resident index."""
    return "rel:" + relation_hash_for_triple(subject, predicate, obj)[:32]


# ---------------------------------------------------------------------------
# Quantized feature projection
# ---------------------------------------------------------------------------

def _validate_feature_options(dim: Any, dtype: Any) -> tuple[int, str]:
    if isinstance(dim, bool) or not isinstance(dim, int) or dim not in FEATURE_DIMS:
        raise ValueError("feature_dim must be 128 or 256")
    if not isinstance(dtype, str):
        raise ValueError("feature_dtype must be int8 or int16")
    value = dtype.strip().casefold()
    if value not in FEATURE_DTYPES:
        raise ValueError("feature_dtype must be int8 or int16")
    return dim, value


def _add_projection(vector: list[float], feature: str, weight: float,
                    dim: int) -> None:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    vector[index] += sign * weight


def _quantize(vector: Sequence[float], dtype: str) -> list[int]:
    norm = math.sqrt(sum(value * value for value in vector))
    limit = _QUANT_MAX[dtype]
    if norm <= 0.0:
        return [0] * len(vector)
    scale = limit / norm
    quantized = []
    for value in vector:
        scaled = abs(value) * scale
        magnitude = int(math.floor(scaled + 0.5))
        magnitude = min(limit, magnitude)
        quantized.append(-magnitude if value < 0 else magnitude)
    return quantized


def project_features(text: Any, *, dim: int = DEFAULT_FEATURE_DIM,
                     dtype: str = DEFAULT_FEATURE_DTYPE) -> list[int]:
    """Project text/expression into a deterministic quantized feature vector.

    The projection hashes word unigrams and adjacent bigrams into a bounded
    dimension (128 or 256), accumulates signed counts, L2-normalizes, then
    quantizes to int8 or int16.  It is deliberately a compact lexical feature
    map, not a learned or pretrained encoder.
    """
    dim, dtype = _validate_feature_options(dim, dtype)
    normalized = normalize_text(text)[:MAX_FEATURE_TEXT]
    words = tokens(normalized)[:MAX_FEATURE_TOKENS]
    vector = [0.0] * dim
    previous = None
    for word in words:
        clipped = word[:MAX_FEATURE_TOKEN_CHARS]
        _add_projection(vector, "w:" + clipped, 1.0, dim)
        if previous is not None:
            _add_projection(vector, "b:" + previous + "\x1f" + clipped,
                            0.45, dim)
        previous = clipped
    return _quantize(vector, dtype)


def encode_features(text: Any, *, dim: int = DEFAULT_FEATURE_DIM,
                    dtype: str = DEFAULT_FEATURE_DTYPE) -> dict[str, Any]:
    """Return feature values plus their dimension, quantization and digest."""
    dim, dtype = _validate_feature_options(dim, dtype)
    values = project_features(text, dim=dim, dtype=dtype)
    digest = digest_value({
        "schema": FEATURE_SCHEMA, "dim": dim, "dtype": dtype,
        "values": values,
    })
    return {
        "schema": FEATURE_SCHEMA,
        "dim": dim,
        "dtype": dtype,
        "values": values,
        "digest": digest,
    }


# Public aliases kept intentionally small and descriptive.
feature_vector = project_features
derive_features = encode_features


# ---------------------------------------------------------------------------
# Row normalization
# ---------------------------------------------------------------------------

def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError:
            return None
    return None


def _coerce_head(value: Any, fallback: int) -> int:
    parsed = _coerce_int(value)
    if parsed is None:
        return fallback
    return max(0, min(MAX_HEAD, parsed))


def _coerce_layer(value: Any, fallback: int) -> int:
    parsed = _coerce_int(value)
    if parsed is None:
        return fallback
    return max(0, min(MAX_LAYER, parsed))


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return DEFAULT_CONFIDENCE
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    if not math.isfinite(confidence):
        return DEFAULT_CONFIDENCE
    return max(0.0, min(1.0, confidence))


def _coerce_polarity(value: Any) -> str:
    if value is None:
        return "neutral"
    return _POLARITY_ALIASES.get(str(value).strip().casefold(), "neutral")


def _default_layer(kind: str) -> int:
    return DEFAULT_LAYER_BY_KIND.get(kind, DEFAULT_LAYER_BY_KIND["observation"])


def _derived_head(*parts: Any) -> int:
    digest = digest_value({"schema": INGEST_SCHEMA, "head": list(parts)})
    return int(digest[:8], 16) % (MAX_HEAD + 1)


def _expression_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    return canonical_json(value)


def _unwrap_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(row, Mapping):
        raise ResidentIngestError("each row must be an object")
    outer: Mapping[str, Any] = row
    item: Mapping[str, Any] = row
    for key in ("item", "record"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            item = nested
            break
    else:
        nested = row.get("payload")
        looks_like_evidence = any(
            field in row for field in
            ("kind", "subject", "predicate", "object", "schema", "text"))
        if isinstance(nested, Mapping) and not looks_like_evidence:
            item = nested
    result = dict(item)
    for key in ("source_digest", "head", "layer", "polarity", "confidence"):
        if result.get(key) is None and outer.get(key) is not None:
            result[key] = outer[key]
    return result, outer


def _provenance(item: Mapping[str, Any]) -> dict[str, Any]:
    value = item.get("provenance")
    if not isinstance(value, Mapping):
        return {}
    sanitized = _sanitize(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _source_for(item: Mapping[str, Any], default: Any,
                identity: Any) -> str:
    raw = item.get("source_digest")
    if raw is None:
        return normalize_source_digest(default=default, derive_from=identity)
    return normalize_source_digest(raw)


def _feature_fields(text: Any, dim: int, dtype: str) -> dict[str, Any]:
    encoded = encode_features(text, dim=dim, dtype=dtype)
    return {
        "features": encoded["values"],
        "feature_dim": encoded["dim"],
        "feature_dtype": encoded["dtype"],
        "feature_digest": encoded["digest"],
    }


def _leaf_record(text: Any, *, kind: str, role: str, source_digest: str,
                 head: int, layer: int, provenance: Mapping[str, Any],
                 dim: int, dtype: str,
                 extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        raise ResidentIngestError("leaf text must not be empty")
    identity = _identity_text(normalized)
    full_digest = digest_value({"text": identity})
    record: dict[str, Any] = {
        "type": "leaf",
        "schema": INGEST_SCHEMA,
        "leaf_id": "leaf:" + full_digest[:32],
        "kind": kind,
        "role": role,
        "text": normalized,
        "text_digest": full_digest,
        "source_digest": source_digest,
        "head": head,
        "layer": layer,
        "provenance": dict(provenance),
        "immutable": True,
    }
    record.update(_feature_fields(normalized, dim, dtype))
    if extra:
        record.update(_sanitize(dict(extra)))
    return record


def _prepare_trainable(item: Mapping[str, Any], *, default_source: Any,
                       dim: int, dtype: str) -> list[dict[str, Any]]:
    text = normalize_text(item.get("text"))
    if not text:
        raise ResidentIngestError("trainable record text must not be empty")
    provenance = _provenance(item)
    record_id = item.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        record_id = digest_value({
            "schema": TRAINABLE_SCHEMA, "text": _identity_text(text),
        })
    else:
        record_id = record_id.strip()
    identity = {
        "schema": TRAINABLE_SCHEMA,
        "record_id": record_id,
        "source": item.get("source"),
        "text": _identity_text(text),
    }
    source_digest = _source_for(item, default_source, identity)
    head = _coerce_head(item.get("head"),
                        _derived_head("trainable", record_id, text))
    layer = _coerce_layer(item.get("layer"), _default_layer("observation"))
    observation_id = "obs:" + digest_value({
        "record_id": record_id, "source_digest": source_digest,
    })[:32]
    extra = {
        "observation": True,
        "observation_id": observation_id,
        "record_id": record_id,
        "origin": TRAINABLE_SCHEMA,
        "stage": item.get("stage"),
        "source": item.get("source"),
        "split": item.get("split"),
    }
    return [_leaf_record(
        text, kind="observation", role="observation",
        source_digest=source_digest, head=head, layer=layer,
        provenance=provenance, dim=dim, dtype=dtype, extra=extra,
    )]


def _observation_text(item: Mapping[str, Any]) -> tuple[str, str]:
    explicit = normalize_text(item.get("text"))
    expression = _expression_text(item.get("expression"))
    subject = normalize_text(item.get("subject"))
    predicate = normalize_text(item.get("predicate") or item.get("role"))
    obj = normalize_text(item.get("object"))
    content = explicit or expression or obj or subject or predicate
    feature_text = " ".join(part for part in (
        subject, predicate, obj, expression, explicit,
    ) if part)
    return content, feature_text or content


def _prepare_observation(item: Mapping[str, Any], *, kind: str,
                         default_source: Any, dim: int,
                         dtype: str) -> list[dict[str, Any]]:
    content, feature_text = _observation_text(item)
    if not content:
        raise ResidentIngestError("observation row has no text or expression")
    provenance = _provenance(item)
    source_kind = kind if kind in DEFAULT_LAYER_BY_KIND else "observation"
    identity = {
        "kind": source_kind,
        "subject": _identity_text(item.get("subject")),
        "predicate": _identity_text(item.get("predicate") or item.get("role")),
        "object": _identity_text(item.get("object")),
        "expression": _sanitize(item.get("expression")),
        "text": _identity_text(content),
    }
    source_digest = _source_for(item, default_source, identity)
    head = _coerce_head(item.get("head"),
                        _derived_head(source_kind, content))
    layer = _coerce_layer(item.get("layer"), _default_layer(source_kind))
    record_id = item.get("record_id") or item.get("event_id")
    if not isinstance(record_id, str) or not record_id.strip():
        record_id = "obs:" + digest_value({
            "content": _identity_text(content),
            "source_digest": source_digest,
            "kind": source_kind,
        })[:32]
    extra = {
        "observation": True,
        "observation_id": record_id,
        "record_id": record_id,
        "origin": "plugin-memory",
        "surface_kind": source_kind,
        "expression": _sanitize(item.get("expression")),
    }
    # Preserve the original fields for query compatibility; none of them is
    # promoted to a relation record for an observation.
    for key in ("subject", "predicate", "object"):
        if item.get(key) is not None:
            extra[key] = normalize_text(item.get(key))
    leaf = _leaf_record(
        content, kind=source_kind, role="observation",
        source_digest=source_digest, head=head, layer=layer,
        provenance=provenance, dim=dim, dtype=dtype, extra=extra,
    )
    # The full row (expression, predicate, etc.) contributes to features even
    # when the observation content itself is only a short surface string.
    if normalize_text(feature_text) != normalize_text(content):
        leaf.update(_feature_fields(feature_text, dim, dtype))
    return [leaf]


def _prepare_relation(item: Mapping[str, Any], *, kind: str,
                      default_source: Any, dim: int,
                      dtype: str) -> list[dict[str, Any]]:
    subject = normalize_text(item.get("subject"))
    predicate = normalize_text(item.get("predicate") or item.get("role"))
    obj = normalize_text(item.get("object"))
    if not (subject and predicate and obj):
        return _prepare_observation(
            item, kind=kind, default_source=default_source,
            dim=dim, dtype=dtype,
        )
    provenance = _provenance(item)
    expression = _sanitize(item.get("expression"))
    identity = {
        "subject": _identity_text(subject),
        "predicate": _identity_text(predicate),
        "object": _identity_text(obj),
    }
    triple_hash = digest_value({"relation": identity})
    triple_id = "rel:" + triple_hash[:32]
    source_digest = _source_for(item, default_source, {
        "kind": kind, "relation": identity, "expression": expression,
    })
    head = _coerce_head(item.get("head"),
                        _derived_head("relation", triple_hash))
    layer = _coerce_layer(item.get("layer"), _default_layer("relation"))
    polarity = _coerce_polarity(item.get("polarity"))
    confidence = _coerce_confidence(item.get("confidence"))
    # The resident relation store keeps polarity/confidence in its relation
    # identity so a refutation is never mistaken for the support it opposes.
    semantics = {
        "subject": identity["subject"],
        "predicate": identity["predicate"],
        "object": identity["object"],
        "polarity": polarity,
        "confidence": confidence,
    }
    relation_hash = digest_value({"relation": semantics})
    relation_id = "rel:" + relation_hash[:32]
    subject_digest = _text_digest(subject)
    predicate_digest = _text_digest(predicate)
    object_digest = _text_digest(obj)
    leaves = [
        _leaf_record(
            subject, kind="entity", role="subject",
            source_digest=source_digest, head=head, layer=layer,
            provenance=provenance, dim=dim, dtype=dtype,
            extra={"part_digest": subject_digest},
        ),
        _leaf_record(
            predicate, kind="entity", role="predicate",
            source_digest=source_digest, head=head, layer=layer,
            provenance=provenance, dim=dim, dtype=dtype,
            extra={"part_digest": predicate_digest},
        ),
        _leaf_record(
            obj, kind="entity", role="object",
            source_digest=source_digest, head=head, layer=layer,
            provenance=provenance, dim=dim, dtype=dtype,
            extra={"part_digest": object_digest},
        ),
    ]
    feature_text = " ".join(part for part in (
        subject, predicate, obj, _expression_text(expression),
    ) if part)
    evidence = {
        "schema": INGEST_SCHEMA,
        "relation_id": relation_id,
        "subject": identity["subject"],
        "predicate": identity["predicate"],
        "object": identity["object"],
        "polarity": polarity,
        "confidence": confidence,
        "expression": expression,
        "source_digest": source_digest,
        "provenance": provenance,
    }
    evidence_id = "relrec:" + digest_value(evidence)[:32]
    relation = {
        "type": "relation",
        "schema": INGEST_SCHEMA,
        "record_id": evidence_id,
        "evidence_id": evidence_id,
        "relation_id": relation_id,
        "relation_hash": relation_hash,
        "triple_id": triple_id,
        "triple_hash": triple_hash,
        "kind": "relation",
        "surface_kind": kind,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "subject_digest": subject_digest,
        "predicate_digest": predicate_digest,
        "object_digest": object_digest,
        "subject_leaf_id": leaves[0]["leaf_id"],
        "predicate_leaf_id": leaves[1]["leaf_id"],
        "object_leaf_id": leaves[2]["leaf_id"],
        "subject_id": leaves[0]["leaf_id"],
        "predicate_id": leaves[1]["leaf_id"],
        "object_id": leaves[2]["leaf_id"],
        "polarity": polarity,
        "confidence": confidence,
        "head": head,
        "layer": layer,
        "expression": expression,
        "provenance": provenance,
        "source_digest": source_digest,
        "observation": False,
        "immutable": True,
    }
    relation.update(_feature_fields(feature_text, dim, dtype))
    return [*leaves, relation]


def _prepare_row(row: Mapping[str, Any], *, default_source: Any,
                 dim: int, dtype: str) -> list[dict[str, Any]]:
    item, _outer = _unwrap_row(row)
    if item.get("schema") == TRAINABLE_SCHEMA:
        return _prepare_trainable(item, default_source=default_source,
                                  dim=dim, dtype=dtype)
    kind = normalize_text(item.get("kind") or "observation").casefold()
    if not kind:
        kind = "observation"
    subject = normalize_text(item.get("subject"))
    predicate = normalize_text(item.get("predicate") or item.get("role"))
    obj = normalize_text(item.get("object"))
    complete_relation = bool(subject and predicate and obj)
    if complete_relation and kind not in OBSERVATION_KINDS:
        return _prepare_relation(item, kind=kind,
                                 default_source=default_source,
                                 dim=dim, dtype=dtype)
    return _prepare_observation(item, kind=kind,
                                default_source=default_source,
                                dim=dim, dtype=dtype)


# ---------------------------------------------------------------------------
# Pure preparation and batching
# ---------------------------------------------------------------------------

def _validate_batch_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_batch must be an integer")
    if not 1 <= value <= MAX_BATCH:
        raise ValueError("max_batch must be in [1, %d]" % MAX_BATCH)
    return value


def _as_rows(rows: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return [rows]
    if isinstance(rows, (str, bytes, bytearray)):
        raise ResidentIngestError("rows must be an iterable of objects")
    try:
        values = list(rows)
    except TypeError as exc:
        raise ResidentIngestError("rows must be an iterable of objects") from exc
    for value in values:
        if not isinstance(value, Mapping):
            raise ResidentIngestError("each row must be an object")
    return values


def _prepared_key(record: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(record.get("type"))
    if kind == "leaf":
        return kind, "%s|%s" % (record.get("leaf_id"),
                                record.get("source_digest"))
    return kind, str(record.get("record_id") or record.get("relation_id"))


def prepare_records(rows: Iterable[Mapping[str, Any]], *,
                    source_digest: str | None = None,
                    feature_dim: int = DEFAULT_FEATURE_DIM,
                    feature_dtype: str = DEFAULT_FEATURE_DTYPE,
                    max_batch: int = MAX_BATCH,
                    deduplicate: bool = True) -> list[dict[str, Any]]:
    """Return canonical leaf/relation ingest dictionaries.

    Records are ordered leaves-first for each row.  Duplicate records (same
    identity and source digest) are removed by default.  This function performs
    no I/O, no network access and no canonical merge.
    """
    dim, dtype = _validate_feature_options(feature_dim, feature_dtype)
    limit = _validate_batch_limit(max_batch)
    values = _as_rows(rows)
    if len(values) > limit:
        raise ValueError("batch exceeds %d rows" % limit)
    if source_digest is not None and not _is_digest(source_digest):
        raise ValueError("source_digest must be a SHA-256 hex digest")
    seen: set[tuple[str, str]] = set()
    prepared: list[dict[str, Any]] = []
    for row in values:
        for record in _prepare_row(row, default_source=source_digest,
                                   dim=dim, dtype=dtype):
            key = _prepared_key(record)
            if deduplicate and key in seen:
                continue
            seen.add(key)
            prepared.append(record)
    return prepared


def prepare_batch(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Prepare one bounded batch and return records plus compact counts."""
    records = prepare_records(rows, **kwargs)
    leaves = [record for record in records if record.get("type") == "leaf"]
    relations = [record for record in records if record.get("type") == "relation"]
    observations = [record for record in leaves if record.get("observation")]
    return {
        "schema": INGEST_SCHEMA,
        "records": records,
        "counts": {
            "records": len(records),
            "leaves": len(leaves),
            "relations": len(relations),
            "observations": len(observations),
        },
        "source_digests": sorted({str(record.get("source_digest"))
                                  for record in records
                                  if record.get("source_digest")}),
    }


def _iter_row_batches(rows: Iterable[Mapping[str, Any]],
                      limit: int) -> Iterator[list[Mapping[str, Any]]]:
    batch: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResidentIngestError("each row must be an object")
        batch.append(row)
        if len(batch) >= limit:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_batches(rows: Iterable[Mapping[str, Any]], *,
                 max_batch: int = MAX_BATCH) -> Iterator[list[Mapping[str, Any]]]:
    """Yield row batches of at most ``max_batch`` elements (hard max 4096)."""
    limit = _validate_batch_limit(max_batch)
    return _iter_row_batches(rows, limit)


# ---------------------------------------------------------------------------
# ResidentUniverse binding
# ---------------------------------------------------------------------------

def _candidate_modules() -> tuple[str, ...]:
    return (
        "nanlmind_hermes.resident.universe",
        "nanlmind_hermes.resident.core",
        "nanlmind_hermes.resident",
        "nanlmind_hermes.resident_universe",
    )


def discover_resident_universe_class() -> type | None:
    """Lazily import a ``ResidentUniverse`` class when one exists."""
    global ResidentUniverse
    if isinstance(ResidentUniverse, type):
        return ResidentUniverse
    for name in _candidate_modules():
        try:
            module = importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            continue
        candidate = getattr(module, "ResidentUniverse", None)
        if isinstance(candidate, type):
            ResidentUniverse = candidate
            return candidate
    return None


def _context_state_dir(context: Any) -> Any | None:
    if context is None:
        return None
    if isinstance(context, (str, Path)):
        return context
    for name in ("state_dir", "resident_path", "universe_path", "path"):
        value = getattr(context, name, None)
        if isinstance(value, (str, Path)):
            return value
    getter = getattr(context, "get_config", None)
    if callable(getter):
        for name in ("state_dir", "resident_path", "universe_path"):
            try:
                value = getter(name, None)
            except Exception:
                continue
            if isinstance(value, (str, Path)):
                return value
    if isinstance(context, Mapping):
        for name in ("state_dir", "resident_path", "universe_path", "path"):
            value = context.get(name)
            if isinstance(value, (str, Path)):
                return value
    return None


def _call_factory(factory: Any, *, context: Any = None,
                  state_dir: Any = None) -> Any:
    if context is None and state_dir is None:
        return factory()
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(state_dir if state_dir is not None else context)
    parameters = signature.parameters
    if context is not None and "context" in parameters:
        return factory(context=context)
    target = state_dir if state_dir is not None else context
    if state_dir is not None and "state_dir" in parameters:
        return factory(state_dir=state_dir)
    if "path" in parameters:
        return factory(path=target)
    if "directory" in parameters:
        return factory(directory=target)
    return factory(target)


def load_resident_universe(*, context: Any = None,
                           state_dir: Any = None) -> Any | None:
    """Open or construct a resident universe, or return ``None`` if absent."""
    universe_class = discover_resident_universe_class()
    if universe_class is None:
        return None
    if state_dir is None:
        state_dir = _context_state_dir(context)
    if context is None and state_dir is None:
        # No path/context: stay on the pure preparation fallback rather than
        # guessing a location for a persistent resident store.
        return None
    opener = getattr(universe_class, "open", None)
    if callable(opener):
        return _call_factory(opener, context=context, state_dir=state_dir)
    return _call_factory(universe_class, context=context, state_dir=state_dir)


def _method(universe: Any, *names: str) -> Any | None:
    for name in names:
        method = getattr(universe, name, None)
        if callable(method):
            return method
    return None


def _invoke(method: Any, value: Any, source_digest: str | None) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(value)
    parameters = list(signature.parameters.values())
    names = {parameter.name for parameter in parameters}
    accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD
                         for parameter in parameters)
    if "source_digest" in names or accepts_kwargs:
        return method(value, source_digest=source_digest)
    positional = [parameter for parameter in parameters
                  if parameter.kind in (parameter.POSITIONAL_ONLY,
                                        parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2 and positional[1].name in {
            "source_digest", "source", "digest"}:
        return method(value, source_digest)
    return method(value)


def _result_duplicate(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    for key in ("duplicate", "duplicated", "existing", "idempotent"):
        if result.get(key) is True:
            return True
    status = str(result.get("status") or "").casefold()
    if status in {"duplicate", "existing", "unchanged", "already-present"}:
        return True
    for key in ("inserted", "created", "appended"):
        if result.get(key) is False and any(
                result.get(identity) for identity in
                ("leaf_id", "relation_id", "record_id")):
            return True
    return False


def _result_counts(result: Any, fallback_records: int) -> tuple[int, int]:
    if isinstance(result, Mapping):
        accepted = result.get("accepted")
        if not isinstance(accepted, int) or isinstance(accepted, bool):
            accepted = result.get("created_records")
        duplicates = result.get("duplicates")
        if not isinstance(duplicates, int) or isinstance(duplicates, bool):
            duplicates = result.get("duplicate_records")
        if isinstance(accepted, int) and not isinstance(accepted, bool):
            if isinstance(duplicates, int) and not isinstance(duplicates, bool):
                return accepted, duplicates
            return accepted, max(0, fallback_records - accepted)
    return fallback_records, 0


def apply_records(universe: Any, records: Iterable[Mapping[str, Any]], *,
                  source_digest: str | None = None) -> dict[str, Any]:
    """Append prepared records through a resident universe's API."""
    if universe is None:
        raise ValueError("universe is required")
    if isinstance(records, Mapping):
        values = [dict(records)]
    else:
        values = [dict(record) for record in records]
    if len(values) > MAX_BATCH:
        raise ValueError("batch exceeds %d records" % MAX_BATCH)
    if not values:
        return {"accepted": 0, "duplicates": 0, "receipts": []}
    batch_method = _method(universe, "ingest_records", "ingest_batch",
                           "append_records")
    if batch_method is not None:
        result = _invoke(batch_method, values, source_digest)
        accepted, duplicates = _result_counts(result, len(values))
        return {"accepted": accepted, "duplicates": duplicates,
                "receipts": result if isinstance(result, list) else [result]}
    leaf_method = _method(universe, "append_leaf", "intern_leaf",
                          "ingest_leaf", "create_leaf", "store_leaf",
                          "add_leaf")
    relation_method = _method(universe, "append_relation", "ingest_relation",
                              "add_relation")
    record_method = _method(universe, "append_record", "ingest_record",
                            "ingest")
    accepted = duplicates = 0
    receipts: list[Any] = []
    for record in values:
        record_type = record.get("type")
        if record_type == "relation":
            method = relation_method or record_method
        else:
            method = leaf_method or record_method
        if method is None:
            raise TypeError(
                "ResidentUniverse does not expose a leaf/relation append API")
        result = _invoke(method, record, record.get("source_digest")
                         or source_digest)
        if _result_duplicate(result):
            duplicates += 1
        else:
            accepted += 1
        receipts.append(result)
    return {"accepted": accepted, "duplicates": duplicates,
            "receipts": receipts}


def _empty_summary() -> dict[str, Any]:
    return {
        "schema": INGEST_SCHEMA,
        "applied": False,
        "available": False,
        "rows": 0,
        "batches": 0,
        "record_batches": 0,
        "records": 0,
        "leaves": 0,
        "relations": 0,
        "observations": 0,
        "accepted": 0,
        "duplicates": 0,
        "source_digests": [],
    }


def ingest_rows(rows: Iterable[Mapping[str, Any]], *,
                universe: Any = None,
                context: Any = None,
                state_dir: Any = None,
                source_digest: str | None = None,
                feature_dim: int = DEFAULT_FEATURE_DIM,
                feature_dtype: str = DEFAULT_FEATURE_DTYPE,
                max_batch: int = MAX_BATCH,
                loader: Any = None,
                include_records: bool | None = None) -> dict[str, Any]:
    """Prepare and, when available, append bounded evidence batches.

    If no universe is supplied and none can be imported, this is a pure
    preparation call: the returned summary includes ``records`` and reports
    ``applied=False``/``available=False``.  No network path exists here.
    """
    dim, dtype = _validate_feature_options(feature_dim, feature_dtype)
    limit = _validate_batch_limit(max_batch)
    if source_digest is not None and not _is_digest(source_digest):
        raise ValueError("source_digest must be a SHA-256 hex digest")
    values = _as_rows(rows)
    if not values:
        summary = _empty_summary()
        if include_records:
            summary["records"] = []
        return summary
    resolved = universe
    if resolved is None:
        load = loader or load_resident_universe
        resolved = load(context=context, state_dir=state_dir)
    applied = resolved is not None
    if include_records is None:
        include_records = not applied
    summary = _empty_summary()
    summary["applied"] = applied
    summary["available"] = applied
    summary["rows"] = len(values)
    all_records: list[dict[str, Any]] = []
    source_digests: set[str] = set()
    accepted = duplicates = total = 0
    batches = record_batches = 0
    for chunk in iter_batches(values, max_batch=limit):
        records = prepare_records(
            chunk, source_digest=source_digest,
            feature_dim=dim, feature_dtype=dtype,
            max_batch=limit,
        )
        batches += 1
        total += len(records)
        for record in records:
            summary["leaves"] += record.get("type") == "leaf"
            summary["relations"] += record.get("type") == "relation"
            summary["observations"] += bool(
                record.get("type") == "leaf" and record.get("observation"))
            if record.get("source_digest"):
                source_digests.add(str(record["source_digest"]))
        if applied:
            for start in range(0, len(records), MAX_BATCH):
                response = apply_records(
                    resolved, records[start:start + MAX_BATCH],
                    source_digest=source_digest)
                accepted += int(response.get("accepted") or 0)
                duplicates += int(response.get("duplicates") or 0)
                record_batches += 1
        if include_records:
            all_records.extend(records)
    summary["batches"] = batches
    summary["record_batches"] = (record_batches if applied else
                                 (total + MAX_BATCH - 1) // MAX_BATCH)
    summary["records"] = total
    summary["accepted"] = accepted if applied else total
    summary["duplicates"] = duplicates
    summary["source_digests"] = sorted(source_digests)
    if include_records:
        summary["prepared_records"] = all_records
    return summary


# Backwards/readability aliases for callers that describe the operation as
# loading evidence rather than rows.
ingest = ingest_rows
load_evidence = ingest_rows


# ---------------------------------------------------------------------------
# JSONL input (local files only)
# ---------------------------------------------------------------------------

def _iter_jsonl_line(line: str,
                     line_number: int) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    position = 0
    length = len(line)
    while position < length:
        while position < length and line[position] in " \t\r\n":
            position += 1
        if position >= length:
            break
        try:
            value, position = decoder.raw_decode(line, position)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "invalid JSON on line %d" % line_number) from exc
        if not isinstance(value, dict):
            raise ValueError(
                "JSONL records must be objects (line %d)" % line_number)
        yield value


def _iter_jsonl_text(text: str) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(text.splitlines(), 1):
        yield from _iter_jsonl_line(line, line_number)


def parse_jsonl(text: str | bytes) -> list[dict[str, Any]]:
    """Parse one or more JSON objects per line (local text only)."""
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8")
    if not isinstance(text, str):
        raise ValueError("JSONL content must be text")
    return list(_iter_jsonl_text(text))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records from a local path without network access."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            yield from _iter_jsonl_line(line, line_number)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a local JSONL file as a list of records."""
    return list(iter_jsonl(path))


read_jsonl = load_jsonl


def ingest_jsonl(source: str | Path, *, limit: int = MAX_BATCH,
                 universe: Any = None, context: Any = None,
                 state_dir: Any = None, source_digest: str | None = None,
                 feature_dim: int = DEFAULT_FEATURE_DIM,
                 feature_dtype: str = DEFAULT_FEATURE_DTYPE,
                 max_batch: int = MAX_BATCH,
                 loader: Any = None,
                 include_records: bool | None = None) -> dict[str, Any]:
    """Load a local JSONL file and feed it through bounded ingestion.

    ``limit`` is clamped to :data:`MAX_BATCH`; at most one bounded batch is
    selected per call.  There is no network or remote-source path.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    selected_limit = min(limit, MAX_BATCH)
    selected = []
    for row in iter_jsonl(source):
        if len(selected) >= selected_limit:
            break
        selected.append(row)
    summary = ingest_rows(
        selected, universe=universe, context=context, state_dir=state_dir,
        source_digest=source_digest, feature_dim=feature_dim,
        feature_dtype=feature_dtype, max_batch=max_batch, loader=loader,
        include_records=include_records,
    )
    summary["source"] = str(source)
    summary["limit"] = selected_limit
    summary["requested_limit"] = limit
    summary["ingested"] = (summary["accepted"] if summary["applied"]
                           else summary["records"])
    summary["limit_clamped"] = selected_limit != limit
    return summary


ingest_file = ingest_jsonl
ingest_jsonl_file = ingest_jsonl


__all__ = (
    "BATCH_LIMIT", "DEFAULT_CONFIDENCE", "DEFAULT_FEATURE_DIM",
    "DEFAULT_FEATURE_DTYPE", "FEATURE_DIM", "FEATURE_DIMS", "FEATURE_DTYPE",
    "FEATURE_DTYPES", "FEATURE_SCHEMA", "INGEST_SCHEMA", "MAX_BATCH",
    "MAX_FEATURE_TEXT", "MAX_HEAD", "MAX_LAYER", "OBSERVATION_KINDS",
    "ResidentIngestError", "ResidentUniverse", "TRAINABLE_SCHEMA",
    "VALID_POLARITIES", "apply_records", "canonical_json", "derive_features",
    "derive_source_digest", "digest_value", "discover_resident_universe_class",
    "encode_features", "feature_vector", "ingest", "ingest_file", "ingest_jsonl",
    "ingest_jsonl_file", "ingest_rows", "iter_batches", "iter_jsonl",
    "leaf_id_for_text", "load_evidence", "load_jsonl", "load_resident_universe",
    "normalize_source_digest", "normalize_text", "parse_jsonl", "prepare_batch",
    "prepare_records", "project_features", "read_jsonl",
    "relation_hash_for_triple", "relation_id_for_triple", "tokens",
)
