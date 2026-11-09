"""Content-addressable retrieval over typed facts.

This module is a bounded Python counterpart of the Round 032 native retrieval
experiment (``experiments/round032/native_recursive.cpp``).  It keeps two
indexes over exactly the same immutable facts:

* a **mirror-pair posting list** keyed by the anchored pair selected from the
  source concept id.  The pair is a deterministic hash-to-pair mapping on the
  256-head resident chart.  Different concepts can select the same pair, so a
  posting lookup resolves collisions by comparing the exact ``(source,
  relation)`` key; and
* an **exact hash comparator** ``dict[(source, relation)]`` over the same facts.

The exact dict comparator is a plain reference baseline.  This module makes no
claim that mirror-pair postings are faster than it; ``audit()`` and
``snapshot()`` report the comparator alongside the posting statistics.

Facts carry monotonically increasing integer ids.  Nothing here exposes a raw
pointer or a mutable resident address: returned projections are dictionaries of
integers, floats, strings and tuples, suitable for JSON snapshots.  Adding a
fact stores an immutable record; a conflicting re-add is rejected rather than
merged, so canonical merge is never performed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, NamedTuple

from .cells import (
    PAIR_COUNT,
    REGION_BYTES,
    canonical_address as _canonical_address,
    mirror_address as _mirror_address,
    pair_index as _cell_pair_index,
)

SCHEMA = "nanlm.resident-mirror-retrieval/1"
SNAPSHOT_SCHEMA = "nanlm.resident-mirror-retrieval-snapshot/1"

UINT64_MAX = (1 << 64) - 1
ANCHOR_HEADS = 256
ANCHOR_MIN = 1
ANCHOR_MAX = 255

DEFAULT_CAPACITY = 65_536
MAX_CAPACITY = 1_000_000
DEFAULT_MAX_POSTING_SCAN = 4_096
DEFAULT_MAX_NEIGHBORS = 256
DEFAULT_MAX_NEIGHBOR_SCAN = 4_096
DEFAULT_MAX_EXPANSIONS = 4_096
DEFAULT_SNAPSHOT_LIMIT = 1_024
DEFAULT_AUDIT_SCAN = 65_536
MAX_BOUND = 1_000_000

DEFAULT_MAX_LINE_BYTES = 16_384
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ROWS = 1_000_000
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024

_MASK64 = UINT64_MAX


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class MirrorRetrievalError(ValueError):
    """Base error for mirror retrieval operations."""

    def __init__(self, message: str, *, code: str = "invalid",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"status": "error", "code": self.code, "error": str(self),
                "details": self.details}


class MirrorBoundsError(MirrorRetrievalError):
    """A fact capacity, scan bound, line bound or expansion bound was hit."""


class MirrorConflictError(MirrorRetrievalError):
    """An existing ``(source, relation)`` key names different content."""


class MirrorRowError(MirrorRetrievalError):
    """A TSV row is malformed or outside the stable integer range."""


class MirrorCorruptionError(MirrorRetrievalError):
    """A persisted snapshot or internal index fails structural validation."""


# ---------------------------------------------------------------------------
# Value types and small validation helpers
# ---------------------------------------------------------------------------
class Fact(NamedTuple):
    """One immutable typed fact.  All fields are JSON-friendly values."""

    fact_id: int
    source: int
    relation: int
    destination: int
    weight: float
    source_digest: str
    anchor_first: int
    anchor_second: int
    pair_index: int
    canonical_address: int
    mirror_address: int


class QueryRow(NamedTuple):
    """One ``source relation destination`` query row."""

    line: int
    source: int
    relation: int
    destination: int


def _require_uint64(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MirrorRetrievalError("%s must be an integer" % field,
                                   code="invalid_integer",
                                   details={"field": field, "value": value})
    if value < 0 or value > UINT64_MAX:
        raise MirrorRetrievalError(
            "%s is outside the stable 64-bit integer range" % field,
            code="integer_out_of_bounds",
            details={"field": field, "value": value})
    return value


def _require_bounded(value: Any, field: str, minimum: int,
                     maximum: int = MAX_BOUND) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MirrorBoundsError("%s must be an integer" % field,
                                code="invalid_bound",
                                details={"field": field, "value": value})
    if value < minimum or value > maximum:
        raise MirrorBoundsError(
            "%s must be in [%d, %d]" % (field, minimum, maximum),
            code="bound_out_of_range",
            details={"field": field, "value": value,
                     "minimum": minimum, "maximum": maximum})
    return value


def _require_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MirrorRetrievalError("weight must be numeric",
                                   code="invalid_weight",
                                   details={"value": value})
    result = float(value)
    if not math.isfinite(result):
        raise MirrorRetrievalError("weight must be finite",
                                   code="invalid_weight",
                                   details={"value": value})
    return result


def _normalize_digest(value: Any) -> str:
    """Normalize a source digest to lowercase 64-character SHA-256 hex.

    A 32-byte value or a 64-character hex string is accepted verbatim.  Any
    other non-empty string/bytes value is hashed with SHA-256 so callers can
    pass a short source label without losing stability.
    """
    if isinstance(value, bool) or value is None:
        raise MirrorRetrievalError("source_digest is required",
                                   code="missing_source_digest")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise MirrorRetrievalError("source_digest must not be empty",
                                       code="missing_source_digest")
        if len(raw) == 32:
            return raw.hex()
        return hashlib.sha256(raw).hexdigest()
    if isinstance(value, int):
        _require_uint64(value, "source_digest")
        return value.to_bytes(32, "big").hex()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise MirrorRetrievalError("source_digest must not be empty",
                                       code="missing_source_digest")
        if len(text) == 64:
            try:
                raw = bytes.fromhex(text)
            except ValueError:
                raw = b""
            if len(raw) == 32:
                return raw.hex()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    raise MirrorRetrievalError("source_digest must be a string or bytes",
                               code="invalid_source_digest")


# ---------------------------------------------------------------------------
# Anchoring: hash of a concept id -> a canonical mirror pair
# ---------------------------------------------------------------------------
def hash64(value: int) -> int:
    """The 64-bit mixer used by the Round 032 native implementation."""
    value = _require_uint64(value, "concept_id") & _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def anchor_pair(concept_id: int) -> tuple[int, int]:
    """Deterministically select an ordered, distinct head pair for a concept.

    Heads are in ``[1, 255]``; head 0 is deliberately not used, matching the
    experimental anchor lane.  The pair is sorted so the mirror orientation is
    canonical and stable.
    """
    _require_uint64(concept_id, "concept_id")
    mixed = hash64(concept_id + 0x9E3779B97F4A7C15) & _MASK64
    first = 1 + mixed % 255
    second = 1 + (mixed >> 15) % 255
    if first == second:
        second = second % 255 + 1
    if first > second:
        first, second = second, first
    return int(first), int(second)


def anchor_addresses(concept_id: int) -> dict[str, int]:
    """Return the anchored heads, pair index and mirror addresses."""
    first, second = anchor_pair(concept_id)
    canonical = _canonical_address(first, second)
    return {
        "first": first,
        "second": second,
        "pair_index": _cell_pair_index(first, second),
        "canonical_address": canonical,
        "mirror_address": _mirror_address(canonical),
    }


# ---------------------------------------------------------------------------
# MirrorRetrieval
# ---------------------------------------------------------------------------
class MirrorRetrieval:
    """Bounded, dual-index retrieval over typed facts.

    ``cells_or_fake`` may be a real :class:`ResidentCells` region, any object
    with a compatible ``pair_index(a, b)`` method, or ``None``.  When it also
    exposes ``observe_pair(a, b, weight, relation_id=...)`` each newly stored
    fact is reflected into the resident pair cell; the posting list itself is
    kept in bounded Python storage.
    """

    canonical_merge = False

    def __init__(self, cells_or_fake: Any = None, *, capacity: int = DEFAULT_CAPACITY,
                 max_posting_scan: int | None = None,
                 max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
                 max_neighbor_scan: int = DEFAULT_MAX_NEIGHBOR_SCAN,
                 max_expansions: int = DEFAULT_MAX_EXPANSIONS,
                 max_snapshot: int = DEFAULT_SNAPSHOT_LIMIT,
                 max_audit: int = DEFAULT_AUDIT_SCAN,
                 unique_keys: bool = False):
        self.cells = cells_or_fake
        self.unique_keys = bool(unique_keys)
        self.capacity = _require_bounded(capacity, "capacity", 1, MAX_CAPACITY)
        if max_posting_scan is None:
            max_posting_scan = min(DEFAULT_MAX_POSTING_SCAN, self.capacity)
        self.max_posting_scan = _require_bounded(
            max_posting_scan, "max_posting_scan", 1, MAX_CAPACITY)
        self.max_neighbors = _require_bounded(
            max_neighbors, "max_neighbors", 1, MAX_BOUND)
        self.max_neighbor_scan = _require_bounded(
            max_neighbor_scan, "max_neighbor_scan", 1, MAX_BOUND)
        self.max_expansions = _require_bounded(
            max_expansions, "max_expansions", 1, MAX_BOUND)
        self.max_snapshot = _require_bounded(
            max_snapshot, "max_snapshot", 1, MAX_BOUND)
        self.max_audit = _require_bounded(
            max_audit, "max_audit", 1, MAX_BOUND)

        # Index 0 is a reserved sentinel; fact ids start at 1 and never change.
        self._facts: list[Fact | None] = [None]
        # The comparator is a real exact hash map.  A key may name more than
        # one typed fact (WordNet synsets have multiple hypernyms), so each
        # bucket is an id list with the newest id last.
        self._exact: dict[tuple[int, int], list[int]] = {}
        self._postings: dict[int, list[int]] = {}
        self._outgoing: dict[int, list[int]] = {}
        self._pair_anchors: dict[int, tuple[int, int]] = {}
        self.generation = 0
        self.last_load: dict[str, Any] | None = None
        self._stats: dict[str, int] = {
            "facts_added": 0,
            "duplicate_facts": 0,
            "conflicts": 0,
            "posting_lookups": 0,
            "posting_steps": 0,
            "hash_lookups": 0,
            "collisions_resolved": 0,
            "observations": 0,
            "neighbor_calls": 0,
            "neighbor_steps": 0,
            "two_hop_calls": 0,
            "two_hop_steps": 0,
            "bound_hits": 0,
        }

    # -- adapters -------------------------------------------------------
    def _pair_index(self, first: int, second: int) -> int:
        """Resolve the dense canonical pair index through cells or fallback."""
        resolver = getattr(self.cells, "pair_index", None) \
            if self.cells is not None else None
        if callable(resolver):
            value = resolver(first, second)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or not 0 <= value < PAIR_COUNT:
                raise MirrorRetrievalError(
                    "cells pair_index returned an invalid pair index",
                    code="invalid_pair_index",
                    details={"value": value, "pair_count": PAIR_COUNT})
            return value
        return int(_cell_pair_index(first, second))

    def _observe(self, first: int, second: int, weight: float,
                 fact_id: int) -> None:
        observer = getattr(self.cells, "observe_pair", None) \
            if self.cells is not None else None
        if not callable(observer):
            return
        relation_id = fact_id if fact_id <= 0xFFFFFFFF else 0
        try:
            observer(first, second, weight, relation_id=relation_id)
        except TypeError:
            # Small fakes may expose ``observe_pair(a, b, weight)`` only.
            observer(first, second, weight)
        self._stats["observations"] += 1

    # -- construction / persistence ------------------------------------
    @classmethod
    def from_snapshot(cls, snapshot: Any, cells_or_fake: Any = None,
                      **kwargs: Any) -> "MirrorRetrieval":
        """Rebuild a retrieval instance from a bounded JSON snapshot.

        Fact ids are preserved by replaying rows in ascending id order.  A
        truncated snapshot is rejected unless ``allow_truncated=True`` is
        passed, because silently restoring a prefix would make subsequent ids
        look complete when they are not.
        """
        if not isinstance(snapshot, dict):
            raise MirrorCorruptionError("snapshot must be a JSON object",
                                        code="invalid_snapshot")
        if snapshot.get("schema") != SNAPSHOT_SCHEMA:
            raise MirrorCorruptionError("unknown mirror retrieval snapshot schema",
                                        code="invalid_snapshot_schema",
                                        details={"schema": snapshot.get("schema")})
        allow_truncated = bool(kwargs.pop("allow_truncated", False))
        if snapshot.get("truncated") and not allow_truncated:
            raise MirrorBoundsError(
                "snapshot is truncated; refusing to restore a partial index",
                code="truncated_snapshot",
                details={"returned": snapshot.get("returned"),
                         "fact_count": snapshot.get("fact_count")})
        rows = snapshot.get("facts")
        if not isinstance(rows, list):
            raise MirrorCorruptionError("snapshot facts must be a list",
                                        code="invalid_snapshot")
        if len(rows) > MAX_CAPACITY:
            raise MirrorBoundsError("snapshot contains too many facts",
                                    code="snapshot_capacity")
        stated = snapshot.get("fact_count")
        if stated is not None:
            if isinstance(stated, bool) or not isinstance(stated, int) \
                    or stated < len(rows):
                raise MirrorCorruptionError("snapshot fact_count is invalid",
                                            code="invalid_snapshot")
            if stated != len(rows) and not allow_truncated:
                raise MirrorBoundsError("snapshot fact_count does not match rows",
                                        code="truncated_snapshot")
        prepared: list[Fact] = []
        seen: set[int] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise MirrorCorruptionError("snapshot fact row must be an object",
                                            code="invalid_snapshot")
            try:
                fact_id = row.get("fact_id", row.get("id"))
                source = row["source"]
                relation = row["relation"]
                destination = row.get("destination", row.get("object"))
                weight = row.get("weight", 1.0)
                digest = row["source_digest"]
            except KeyError as exc:
                raise MirrorCorruptionError(
                    "snapshot fact row is missing %s" % exc.args[0],
                    code="invalid_snapshot") from exc
            fact_id = _require_uint64(fact_id, "fact_id")
            if fact_id in seen:
                raise MirrorCorruptionError("duplicate fact id in snapshot",
                                            code="invalid_snapshot")
            seen.add(fact_id)
            prepared.append(Fact(
                fact_id=fact_id,
                source=_require_uint64(source, "source"),
                relation=_require_uint64(relation, "relation"),
                destination=_require_uint64(destination, "destination"),
                weight=_require_weight(weight),
                source_digest=_normalize_digest(digest),
                anchor_first=0,
                anchor_second=0,
                pair_index=0,
                canonical_address=0,
                mirror_address=0,
            ))
        prepared.sort(key=lambda fact: fact.fact_id)
        expected = list(range(1, len(prepared) + 1))
        if [fact.fact_id for fact in prepared] != expected:
            raise MirrorCorruptionError(
                "snapshot fact ids must be contiguous from 1",
                code="invalid_snapshot")
        capacity = kwargs.pop("capacity", None)
        if capacity is None:
            capacity = max(DEFAULT_CAPACITY, len(prepared) + 1)
        else:
            capacity = _require_bounded(capacity, "capacity", 1, MAX_CAPACITY)
            if capacity < len(prepared) + 1:
                raise MirrorBoundsError("snapshot exceeds requested capacity",
                                        code="snapshot_capacity")
        restored = cls(cells_or_fake, capacity=capacity, **kwargs)
        for row in prepared:
            fact_id = restored.add_fact(
                row.source, row.relation, row.destination, row.weight,
                row.source_digest)
            if fact_id != row.fact_id:
                raise MirrorCorruptionError(
                    "restored fact id diverged from snapshot",
                    code="invalid_snapshot",
                    details={"expected": row.fact_id, "actual": fact_id})
        restored.last_load = {
            "restored_facts": len(prepared),
            "source": "snapshot",
        }
        return restored

    # -- facts ----------------------------------------------------------
    def add_fact(self, source: int, relation: int, destination: int,
                 weight: float = 1.0, source_digest: Any = None) -> int:
        """Add one fact and return its stable integer id.

        An identical re-add returns the existing id.  By default a key may
        name several facts (for example, a synset with multiple hypernyms);
        each is stored separately and neither index merges them.  Set
        ``unique_keys=True`` to reject a second distinct fact for one
        ``(source, relation)`` with :class:`MirrorConflictError`.
        """
        source = _require_uint64(source, "source")
        relation = _require_uint64(relation, "relation")
        destination = _require_uint64(destination, "destination")
        weight_value = _require_weight(weight)
        digest = _normalize_digest(source_digest)
        key = (source, relation)

        existing_ids = self._exact.get(key)
        if existing_ids:
            for existing_id in reversed(existing_ids):
                existing = self._facts[existing_id]
                assert existing is not None
                if (existing.destination == destination
                        and existing.weight == weight_value
                        and existing.source_digest == digest):
                    self._stats["duplicate_facts"] += 1
                    return existing_id
            if self.unique_keys:
                existing = self._facts[existing_ids[-1]]
                assert existing is not None
                self._stats["conflicts"] += 1
                raise MirrorConflictError(
                    "fact key already names different immutable content",
                    code="fact_conflict",
                    details={
                        "fact_id": existing_ids[-1],
                        "source": source,
                        "relation": relation,
                        "stored_destination": existing.destination,
                        "requested_destination": destination,
                        "canonical_merge": False,
                    })

        if len(self._facts) - 1 >= self.capacity:
            self._stats["bound_hits"] += 1
            raise MirrorBoundsError(
                "mirror retrieval fact capacity exhausted",
                code="fact_capacity",
                details={"capacity": self.capacity,
                         "facts": len(self._facts) - 1})

        first, second = anchor_pair(source)
        pair_index = self._pair_index(first, second)
        canonical = _canonical_address(first, second)
        fact_id = len(self._facts)

        # The pair observation happens before the immutable record is
        # committed, so a rejected cell write cannot leave a fact without its
        # resident pair update.
        self._observe(first, second, weight_value, fact_id)

        fact = Fact(
            fact_id=fact_id,
            source=source,
            relation=relation,
            destination=destination,
            weight=weight_value,
            source_digest=digest,
            anchor_first=first,
            anchor_second=second,
            pair_index=pair_index,
            canonical_address=canonical,
            mirror_address=_mirror_address(canonical),
        )
        self._facts.append(fact)
        self._exact.setdefault(key, []).append(fact_id)
        self._postings.setdefault(pair_index, []).append(fact_id)
        self._outgoing.setdefault(source, []).append(fact_id)
        self._pair_anchors.setdefault(pair_index, (first, second))
        self.generation += 1
        self._stats["facts_added"] += 1
        return fact_id

    # Backwards-friendly spelling for callers that treat relations as edges.
    add = add_fact

    def _fact_view(self, fact: Fact) -> dict[str, Any]:
        return {
            "fact_id": fact.fact_id,
            "id": fact.fact_id,
            "source": fact.source,
            "relation": fact.relation,
            "destination": fact.destination,
            "object": fact.destination,
            "dst": fact.destination,
            "weight": fact.weight,
            "source_digest": fact.source_digest,
            "anchor": (fact.anchor_first, fact.anchor_second),
            "anchor_pair": (fact.anchor_first, fact.anchor_second),
            "anchor_first": fact.anchor_first,
            "anchor_second": fact.anchor_second,
            "pair_index": fact.pair_index,
            "canonical_address": fact.canonical_address,
            "mirror_address": fact.mirror_address,
        }

    def get(self, source: int, relation: int) -> dict[str, Any] | None:
        """Return the newest fact for an exact key via the mirror posting list."""
        source = _require_uint64(source, "source")
        relation = _require_uint64(relation, "relation")
        first, second = anchor_pair(source)
        pair_index = self._pair_index(first, second)
        self._stats["posting_lookups"] += 1
        posting = self._postings.get(pair_index)
        if not posting:
            return None
        scanned = 0
        for fact_id in reversed(posting):
            if scanned >= self.max_posting_scan:
                self._stats["bound_hits"] += 1
                raise MirrorBoundsError(
                    "mirror posting scan bound exceeded",
                    code="posting_scan_limit",
                    details={"pair_index": pair_index,
                             "posting_length": len(posting),
                             "max_posting_scan": self.max_posting_scan})
            scanned += 1
            self._stats["posting_steps"] += 1
            fact = self._facts[fact_id]
            assert fact is not None
            if fact.source == source and fact.relation == relation:
                if scanned > 1:
                    self._stats["collisions_resolved"] += 1
                return self._fact_view(fact)
        return None

    def get_hash(self, source: int, relation: int) -> dict[str, Any] | None:
        """Return the newest fact through the exact ``(source, relation)`` map."""
        source = _require_uint64(source, "source")
        relation = _require_uint64(relation, "relation")
        self._stats["hash_lookups"] += 1
        fact_ids = self._exact.get((source, relation))
        if not fact_ids:
            return None
        fact = self._facts[fact_ids[-1]]
        assert fact is not None
        return self._fact_view(fact)

    def get_all(self, source: int, relation: int, *,
                limit: int | None = None) -> list[dict[str, Any]]:
        """Return every fact for an exact key, newest first, bounded by limit."""
        source = _require_uint64(source, "source")
        relation = _require_uint64(relation, "relation")
        if limit is None:
            limit = self.max_neighbors
        else:
            limit = _require_bounded(limit, "limit", 1, self.max_neighbors)
        facts: list[dict[str, Any]] = []
        for fact_id in reversed(self._exact.get((source, relation), ())):
            if len(facts) >= limit:
                break
            fact = self._facts[fact_id]
            assert fact is not None
            facts.append(self._fact_view(fact))
        return facts

    # Convenient explicit name for the comparison baseline.
    get_comparator = get_hash

    # -- traversal ------------------------------------------------------
    def neighbors_detail(self, source: int, relation: int | None = None,
                         limit: int | None = None) -> dict[str, Any]:
        """Bounded outgoing facts for ``source`` with a truncation report."""
        source = _require_uint64(source, "source")
        if relation is not None:
            relation = _require_uint64(relation, "relation")
        if limit is None:
            limit = self.max_neighbors
        else:
            limit = _require_bounded(limit, "limit", 1, self.max_neighbors)
        self._stats["neighbor_calls"] += 1
        facts: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        for fact_id in reversed(self._outgoing.get(source, ())):
            if scanned >= self.max_neighbor_scan:
                self._stats["bound_hits"] += 1
                raise MirrorBoundsError(
                    "neighbor scan bound exceeded",
                    code="neighbor_scan_limit",
                    details={"source": source,
                             "scanned": scanned,
                             "max_neighbor_scan": self.max_neighbor_scan})
            scanned += 1
            self._stats["neighbor_steps"] += 1
            fact = self._facts[fact_id]
            assert fact is not None
            if relation is not None and fact.relation != relation:
                continue
            if len(facts) >= limit:
                truncated = True
                break
            facts.append(self._fact_view(fact))
        return {
            "facts": facts,
            "neighbors": facts,
            "returned": len(facts),
            "limit": limit,
            "scanned": scanned,
            "truncated": truncated,
            "bounded": True,
        }

    def neighbors(self, source: int, relation: int | None = None,
                  limit: int | None = None) -> list[dict[str, Any]]:
        """Return bounded outgoing facts, newest first."""
        return self.neighbors_detail(
            source, relation=relation, limit=limit)["facts"]

    def two_hop(self, source: int, goal: int, relation: int, *,
                limit: int | None = None) -> bool:
        """Return whether a directed two-edge typed path exists.

        Exactly two edges are required: ``source -relation-> middle
        -relation-> goal``.  The direct one-edge case is deliberately not
        accepted, matching the Round 032 ``two_hop`` comparison.  ``limit``
        bounds the number of fact visits.
        """
        source = _require_uint64(source, "source")
        goal = _require_uint64(goal, "goal")
        relation = _require_uint64(relation, "relation")
        if limit is None:
            budget = self.max_expansions
        else:
            budget = _require_bounded(limit, "limit", 1, self.max_expansions)
        self._stats["two_hop_calls"] += 1
        steps = 0
        seen_middles: set[int] = set()
        for fact_id in reversed(self._outgoing.get(source, ())):
            if steps >= budget:
                self._stats["two_hop_steps"] += steps
                self._stats["bound_hits"] += 1
                raise MirrorBoundsError(
                    "two-hop expansion bound exceeded",
                    code="two_hop_limit",
                    details={"source": source, "goal": goal,
                             "max_expansions": budget})
            steps += 1
            first = self._facts[fact_id]
            assert first is not None
            if first.relation != relation:
                continue
            middle = first.destination
            if middle in seen_middles:
                continue
            seen_middles.add(middle)
            for second_id in reversed(self._outgoing.get(middle, ())):
                if steps >= budget:
                    self._stats["two_hop_steps"] += steps
                    self._stats["bound_hits"] += 1
                    raise MirrorBoundsError(
                        "two-hop expansion bound exceeded",
                        code="two_hop_limit",
                        details={"source": source, "goal": goal,
                                 "max_expansions": budget})
                steps += 1
                second = self._facts[second_id]
                assert second is not None
                if second.relation == relation and second.destination == goal:
                    self._stats["two_hop_steps"] += steps
                    return True
        self._stats["two_hop_steps"] += steps
        return False

    # -- introspection --------------------------------------------------
    def _collision_summary(self) -> dict[str, Any]:
        colliding_pair_lists = 0
        colliding_keys = 0
        max_length = 0
        entries = 0
        for posting in self._postings.values():
            entries += len(posting)
            max_length = max(max_length, len(posting))
            keys = set()
            for fact_id in posting:
                fact = self._facts[fact_id]
                assert fact is not None
                keys.add((fact.source, fact.relation))
            if len(keys) > 1:
                colliding_pair_lists += 1
                colliding_keys += len(keys)
        return {
            "posting_lists": len(self._postings),
            "posting_entries": entries,
            "collision_lists": colliding_pair_lists,
            "collision_keys": colliding_keys,
            "max_posting_length": max_length,
        }

    def comparator_view(self) -> dict[str, Any]:
        """Describe the conventional comparator without a speed claim."""
        fact_count = len(self._facts) - 1
        indexed = sum(len(ids) for ids in self._exact.values())
        return {
            "kind": "exact_hash_map",
            "key": "(source, relation)",
            "indexed_facts": indexed,
            "same_facts": indexed == fact_count,
            "performance_claim": "none",
            "note": ("Exact dict comparator indexed over the same facts; "
                     "mirror-posting speed is not measured or asserted."),
        }

    def status(self) -> dict[str, Any]:
        """Return counters and index sizes (read-only)."""
        collision = self._collision_summary()
        return {
            "schema": SCHEMA,
            "facts": len(self._facts) - 1,
            "capacity": self.capacity,
            "generation": self.generation,
            "exact_index_size": sum(len(ids) for ids in self._exact.values()),
            "exact_key_count": len(self._exact),
            "pair_anchors": len(self._pair_anchors),
            "stats": dict(self._stats),
            "bounds": {
                "max_posting_scan": self.max_posting_scan,
                "max_neighbors": self.max_neighbors,
                "max_neighbor_scan": self.max_neighbor_scan,
                "max_expansions": self.max_expansions,
                "max_snapshot": self.max_snapshot,
                "max_audit": self.max_audit,
            },
            "collision": collision,
            "comparator": self.comparator_view(),
            "canonical_merge": False,
            "bounded": True,
        }

    def snapshot(self, limit: int | None = None) -> dict[str, Any]:
        """Return a bounded, JSON-friendly projection of facts and anchors."""
        if limit is None:
            limit = self.max_snapshot
        else:
            limit = _require_bounded(limit, "limit", 1, self.max_snapshot)
        fact_count = len(self._facts) - 1
        returned = min(fact_count, limit)
        facts = []
        for fact_id in range(1, returned + 1):
            fact = self._facts[fact_id]
            assert fact is not None
            facts.append(self._fact_view(fact))
        anchors = []
        for pair_index in sorted(self._postings):
            if len(anchors) >= limit:
                break
            posting = self._postings[pair_index]
            first, second = self._pair_anchors[pair_index]
            canonical = _canonical_address(first, second)
            anchors.append({
                "pair_index": pair_index,
                "anchor": (first, second),
                "canonical_address": canonical,
                "mirror_address": _mirror_address(canonical),
                "postings": len(posting),
            })
        collision = self._collision_summary()
        return {
            "schema": SNAPSHOT_SCHEMA,
            "facts": facts,
            "fact_count": fact_count,
            "returned": len(facts),
            "limit": limit,
            "truncated": returned < fact_count,
            "anchors": anchors,
            "anchor_count": len(self._postings),
            "anchors_truncated": len(anchors) < len(self._postings),
            "collision": collision,
            "exact_index_size": sum(len(ids) for ids in self._exact.values()),
            "exact_key_count": len(self._exact),
            "comparator": self.comparator_view(),
            "canonical_merge": False,
            "bounded": True,
            "generation": self.generation,
            "stats": dict(self._stats),
        }

    def audit(self, limit: int | None = None) -> dict[str, Any]:
        """Check mirror postings, exact comparator and stable ids for mismatches.

        The audit is read-only and scans at most ``limit`` facts.  It returns
        ``ok=False`` with concrete ``problems`` when the two indexes disagree or
        an anchor/provenance field is inconsistent.
        """
        if limit is None:
            limit = self.max_audit
        else:
            limit = _require_bounded(limit, "limit", 1, self.max_audit)
        fact_count = len(self._facts) - 1
        checked = min(fact_count, limit)
        problems: list[str] = []
        for fact_id in range(1, checked + 1):
            fact = self._facts[fact_id]
            if fact is None:
                problems.append("missing immutable fact row %d" % fact_id)
                continue
            if fact.fact_id != fact_id:
                problems.append("fact id mismatch at row %d" % fact_id)
            for field_name, value in (
                    ("source", fact.source), ("relation", fact.relation),
                    ("destination", fact.destination)):
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value < 0 or value > UINT64_MAX:
                    problems.append("invalid %s for fact %d"
                                    % (field_name, fact_id))
            if not math.isfinite(fact.weight):
                problems.append("non-finite weight for fact %d" % fact_id)
            if len(fact.source_digest) != 64:
                problems.append("invalid source digest for fact %d" % fact_id)
            else:
                try:
                    bytes.fromhex(fact.source_digest)
                except ValueError:
                    problems.append("non-hex source digest for fact %d" % fact_id)
            first, second = anchor_pair(fact.source)
            if (first, second) != (fact.anchor_first, fact.anchor_second):
                problems.append("anchor mismatch for fact %d" % fact_id)
            expected_index = self._pair_index(first, second)
            if expected_index != fact.pair_index:
                problems.append("pair index mismatch for fact %d" % fact_id)
            canonical = _canonical_address(first, second)
            if canonical != fact.canonical_address:
                problems.append("canonical address mismatch for fact %d"
                                % fact_id)
            if _mirror_address(canonical) != fact.mirror_address:
                problems.append("mirror address mismatch for fact %d" % fact_id)
            if fact_id not in self._exact.get(
                    (fact.source, fact.relation), ()):
                problems.append("exact comparator mismatch for fact %d"
                                % fact_id)
            posting = self._postings.get(fact.pair_index, ())
            if fact_id not in posting:
                problems.append("fact %d missing from mirror posting" % fact_id)
            outgoing = self._outgoing.get(fact.source, ())
            if fact_id not in outgoing:
                problems.append("fact %d missing from outgoing index" % fact_id)

        exact_entries = sum(len(ids) for ids in self._exact.values())
        if exact_entries != fact_count:
            problems.append("exact comparator size %d != fact count %d"
                            % (exact_entries, fact_count))
        collision = self._collision_summary()
        if collision["posting_entries"] != fact_count:
            problems.append("posting entry count %d != fact count %d"
                            % (collision["posting_entries"], fact_count))
        outgoing_entries = sum(len(ids) for ids in self._outgoing.values())
        if outgoing_entries != fact_count:
            problems.append("outgoing entry count %d != fact count %d"
                            % (outgoing_entries, fact_count))
        if checked == fact_count:
            seen_ids: set[int] = set()
            for posting in self._postings.values():
                for fact_id in posting:
                    if fact_id in seen_ids:
                        problems.append("duplicate posting for fact %d" % fact_id)
                    seen_ids.add(fact_id)
            if len(seen_ids) != fact_count:
                problems.append("posting coverage %d != fact count %d"
                                % (len(seen_ids), fact_count))
        truncated = checked < fact_count
        return {
            "schema": SCHEMA,
            "ok": not problems,
            "problems": problems,
            "issues": list(problems),
            "facts_checked": checked,
            "fact_count": fact_count,
            "scanned": checked,
            "truncated": truncated,
            "bounded": True,
            "limit": limit,
            "exact_index_size": sum(len(ids) for ids in self._exact.values()),
            "exact_key_count": len(self._exact),
            "pair_anchors": len(self._pair_anchors),
            "posting_lists": collision["posting_lists"],
            "posting_entries": collision["posting_entries"],
            "collision_lists": collision["collision_lists"],
            "collision_keys": collision["collision_keys"],
            "max_posting_length": collision["max_posting_length"],
            "comparator": self.comparator_view(),
            "canonical_merge": False,
            "generation": self.generation,
            "stats": dict(self._stats),
        }

    def merge(self, *args: Any, **kwargs: Any) -> None:
        """Canonical merge is deliberately disabled in this layer."""
        raise MirrorRetrievalError(
            "canonical mirror retrieval merge is disabled",
            code="canonical_merge_disabled",
            details={"canonical_merge": False})

    promote = merge

    def __len__(self) -> int:
        return len(self._facts) - 1

    def __contains__(self, key: Any) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        try:
            return self.get_hash(key[0], key[1]) is not None
        except MirrorRetrievalError:
            return False


# ---------------------------------------------------------------------------
# Bounded TSV loading
# ---------------------------------------------------------------------------
def _iter_lines(path: Any, *, max_line_bytes: int, max_bytes: int,
                max_rows: int) -> Iterator[tuple[int, str]]:
    """Yield non-empty, non-comment TSV rows with streaming bounds."""
    if isinstance(max_line_bytes, bool) or not isinstance(max_line_bytes, int) \
            or max_line_bytes < 1:
        raise MirrorBoundsError("max_line_bytes must be positive",
                                code="invalid_bound")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) \
            or max_bytes < 1:
        raise MirrorBoundsError("max_bytes must be positive",
                                code="invalid_bound")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) \
            or max_rows < 1:
        raise MirrorBoundsError("max_rows must be positive",
                                code="invalid_bound")
    total = 0
    rows = 0
    with open(Path(path), "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            encoded_length = len(line.encode("utf-8"))
            total += encoded_length
            if total > max_bytes:
                raise MirrorBoundsError(
                    "file exceeds the streaming byte bound",
                    code="file_size_limit",
                    details={"max_bytes": max_bytes, "line": line_number})
            if encoded_length > max_line_bytes:
                raise MirrorBoundsError(
                    "line exceeds the streaming line bound",
                    code="line_too_long",
                    details={"line": line_number,
                             "length": encoded_length,
                             "max_line_bytes": max_line_bytes})
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows += 1
            if rows > max_rows:
                raise MirrorBoundsError(
                    "file exceeds the row bound",
                    code="row_limit",
                    details={"max_rows": max_rows, "line": line_number})
            yield line_number, text


def _parse_tsv_row(text: str, line_number: int, label: str
                   ) -> tuple[int, int, int]:
    parts = text.split()
    if len(parts) != 3:
        raise MirrorRowError(
            "%s row must have exactly three integer columns" % label,
            code="malformed_row",
            details={"line": line_number, "columns": len(parts)})
    values = []
    for name, token in zip(("source", "relation", "destination"), parts):
        try:
            value = int(token, 10)
        except ValueError as exc:
            raise MirrorRowError(
                "%s row contains a non-integer %s" % (label, name),
                code="malformed_row",
                details={"line": line_number, "column": name,
                         "value": token}) from exc
        if value < 0 or value > UINT64_MAX:
            raise MirrorRowError(
                "%s row %s is outside the stable integer range" % (label, name),
                code="integer_out_of_bounds",
                details={"line": line_number, "column": name, "value": value})
        values.append(value)
    return values[0], values[1], values[2]


def iter_fact_rows(path: Any, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
                   max_bytes: int = DEFAULT_MAX_FILE_BYTES,
                   max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[dict[str, int]]:
    """Stream ``source relation destination`` fact rows from a TSV file."""
    for line_number, text in _iter_lines(
            path, max_line_bytes=max_line_bytes, max_bytes=max_bytes,
            max_rows=max_rows):
        source, relation, destination = _parse_tsv_row(text, line_number, "fact")
        yield {"line": line_number, "source": source, "relation": relation,
               "destination": destination}


def iter_query_rows(path: Any, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
                    max_bytes: int = DEFAULT_MAX_FILE_BYTES,
                    max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[QueryRow]:
    """Stream ``source relation destination`` query rows from a TSV file."""
    for line_number, text in _iter_lines(
            path, max_line_bytes=max_line_bytes, max_bytes=max_bytes,
            max_rows=max_rows):
        source, relation, destination = _parse_tsv_row(text, line_number, "query")
        yield QueryRow(line_number, source, relation, destination)


def _file_digest(path: Any, max_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with open(Path(path), "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise MirrorBoundsError(
                    "file exceeds the streaming byte bound",
                    code="file_size_limit", details={"max_bytes": max_bytes})
            digest.update(chunk)
    return digest.hexdigest()


def load_facts(path: Any, retrieval: MirrorRetrieval | None = None, *,
               limit: int | None = None, cells_or_fake: Any = None,
               capacity: int = DEFAULT_CAPACITY,
               weight: float = 1.0, source_digest: Any = None,
               max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
               max_bytes: int = DEFAULT_MAX_FILE_BYTES,
               max_rows: int = DEFAULT_MAX_ROWS,
               skip_malformed: bool = False) -> MirrorRetrieval:
    """Stream facts.tsv into a :class:`MirrorRetrieval` and return it.

    The loader never reads the whole file into memory.  ``limit`` stops after
    that many valid facts (a conservative ``truncated`` flag is recorded);
    ``max_line_bytes``, ``max_bytes`` and ``max_rows`` are hard bounds.
    Malformed rows raise :class:`MirrorRowError` unless ``skip_malformed`` is
    true, in which case they are counted and skipped.  ``source_digest``
    defaults to the SHA-256 of the file contents.
    """
    if retrieval is not None and not isinstance(retrieval, MirrorRetrieval):
        raise MirrorRetrievalError("retrieval must be a MirrorRetrieval",
                                   code="invalid_retrieval")
    if limit is not None:
        limit = _require_bounded(limit, "limit", 1, MAX_CAPACITY)
    weight_value = _require_weight(weight)
    if source_digest is None:
        digest = _file_digest(path, max_bytes)
    else:
        digest = _normalize_digest(source_digest)
    if retrieval is None:
        retrieval = MirrorRetrieval(cells_or_fake, capacity=capacity)
    fact_ids: list[int] = []
    rows_read = 0
    skipped = 0
    duplicates = 0
    lines_seen = 0
    truncated = False
    iterator = _iter_lines(Path(path), max_line_bytes=max_line_bytes,
                           max_bytes=max_bytes, max_rows=max_rows)
    try:
        for line_number, text in iterator:
            lines_seen = line_number
            try:
                source, relation, destination = _parse_tsv_row(
                    text, line_number, "fact")
            except MirrorRowError:
                if skip_malformed:
                    skipped += 1
                    continue
                raise
            rows_read += 1
            before = len(retrieval)
            fact_id = retrieval.add_fact(
                source, relation, destination, weight_value, digest)
            if len(retrieval) == before:
                duplicates += 1
            else:
                fact_ids.append(fact_id)
            if limit is not None and rows_read >= limit:
                truncated = True
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    report = {
        "path": str(Path(path)),
        "source_digest": digest,
        "limit": limit,
        "facts_added": len(fact_ids),
        "duplicates": duplicates,
        "rows_read": rows_read,
        "skipped_malformed": skipped,
        "line_count": lines_seen,
        "truncated": truncated,
        "fact_ids": fact_ids,
        "bounds": {
            "max_line_bytes": max_line_bytes,
            "max_bytes": max_bytes,
            "max_rows": max_rows,
        },
        "bounded": True,
        "canonical_merge": False,
    }
    retrieval.last_load = report
    retrieval.load_report = report
    return retrieval


class QueryRows(list):
    """A list of :class:`QueryRow` with the loader's bounded report attached."""

    def __init__(self, rows: list[QueryRow], report: dict[str, Any]):
        super().__init__(rows)
        self.report = report

    @property
    def last_load(self) -> dict[str, Any]:
        return self.report


def load_queries(path: Any, *, limit: int | None = None,
                 max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
                 max_bytes: int = DEFAULT_MAX_FILE_BYTES,
                 max_rows: int = DEFAULT_MAX_ROWS,
                 skip_malformed: bool = False) -> QueryRows:
    """Stream queries.tsv and return bounded :class:`QueryRow` values."""
    if limit is not None:
        limit = _require_bounded(limit, "limit", 1, MAX_CAPACITY)
    rows: list[QueryRow] = []
    skipped = 0
    lines_seen = 0
    truncated = False
    iterator = _iter_lines(Path(path), max_line_bytes=max_line_bytes,
                           max_bytes=max_bytes, max_rows=max_rows)
    try:
        for line_number, text in iterator:
            lines_seen = line_number
            try:
                source, relation, destination = _parse_tsv_row(
                    text, line_number, "query")
            except MirrorRowError:
                if skip_malformed:
                    skipped += 1
                    continue
                raise
            rows.append(QueryRow(line_number, source, relation, destination))
            if limit is not None and len(rows) >= limit:
                truncated = True
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    report = {
        "path": str(Path(path)),
        "limit": limit,
        "queries": len(rows),
        "skipped_malformed": skipped,
        "line_count": lines_seen,
        "truncated": truncated,
        "bounds": {"max_line_bytes": max_line_bytes,
                   "max_bytes": max_bytes, "max_rows": max_rows},
        "bounded": True,
    }
    return QueryRows(rows, report)


# ---------------------------------------------------------------------------
# Snapshot persistence (stdlib JSON, bounded)
# ---------------------------------------------------------------------------
def dump_snapshot(path: Any, retrieval: MirrorRetrieval, *,
                  limit: int | None = None) -> dict[str, Any]:
    """Atomically write a bounded JSON snapshot of ``retrieval``."""
    if not isinstance(retrieval, MirrorRetrieval):
        raise MirrorRetrievalError("retrieval must be a MirrorRetrieval",
                                   code="invalid_retrieval")
    snapshot = retrieval.snapshot(limit=limit)
    try:
        data = json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MirrorCorruptionError("snapshot is not JSON serialisable",
                                    code="invalid_snapshot") from exc
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise MirrorBoundsError("snapshot exceeds the byte bound",
                                code="snapshot_bytes",
                                details={"bytes": len(data),
                                         "max_bytes": MAX_SNAPSHOT_BYTES})
    target = Path(path)
    parent = target.parent
    fd, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "path": str(target),
        "bytes": len(data),
        "digest": hashlib.sha256(data).hexdigest(),
        "facts": snapshot["fact_count"],
    }


def load_snapshot(path: Any, cells_or_fake: Any = None, *,
                  max_bytes: int = MAX_SNAPSHOT_BYTES,
                  **kwargs: Any) -> MirrorRetrieval:
    """Read a bounded JSON snapshot and restore a :class:`MirrorRetrieval`."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) \
            or max_bytes < 1:
        raise MirrorBoundsError("max_bytes must be positive",
                                code="invalid_bound")
    target = Path(path)
    size = target.stat().st_size
    if size > max_bytes:
        raise MirrorBoundsError("snapshot exceeds the byte bound",
                                code="snapshot_bytes",
                                details={"bytes": size, "max_bytes": max_bytes})
    data = target.read_bytes()
    try:
        snapshot = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MirrorCorruptionError("snapshot is not valid JSON",
                                    code="invalid_snapshot") from exc
    return MirrorRetrieval.from_snapshot(
        snapshot, cells_or_fake, **kwargs)


__all__ = (
    "SCHEMA", "SNAPSHOT_SCHEMA", "UINT64_MAX", "ANCHOR_HEADS", "ANCHOR_MIN",
    "ANCHOR_MAX", "DEFAULT_CAPACITY", "MAX_CAPACITY", "DEFAULT_MAX_POSTING_SCAN",
    "DEFAULT_MAX_NEIGHBORS", "DEFAULT_MAX_NEIGHBOR_SCAN",
    "DEFAULT_MAX_EXPANSIONS", "DEFAULT_SNAPSHOT_LIMIT", "DEFAULT_AUDIT_SCAN",
    "MAX_BOUND", "DEFAULT_MAX_LINE_BYTES", "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_ROWS", "MAX_SNAPSHOT_BYTES", "MirrorRetrievalError",
    "MirrorBoundsError", "MirrorConflictError", "MirrorRowError",
    "MirrorCorruptionError", "Fact", "QueryRow", "MirrorRetrieval", "hash64",
    "anchor_pair", "anchor_addresses", "iter_fact_rows", "iter_query_rows",
    "load_facts", "load_queries", "QueryRows", "dump_snapshot", "load_snapshot",
    "REGION_BYTES",
)
