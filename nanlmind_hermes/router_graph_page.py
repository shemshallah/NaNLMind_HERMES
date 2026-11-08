"""Read-only cross-repository verifier for NaNROUTER L0-L9 graph pages.

This never creates a second graph, writes the 8 MiB chart, or authenticates
a sender. It checks topology, actual local chart parent/epoch, whole-page
commitment formatting and streams all pages to verify the router's graph
state hash without retaining the full state in HERMES memory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

from .mirror_topology import CANONICAL_CELLS, MirrorTopology, TOPOLOGY_DIGEST
from .universe_chart import Chart8MiB
from .universe_root import read_universe_root

SCHEMA = "nanrouter.nan-layer-master-commitment/1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DOMAIN_STATE = b"nanrouter.nan-layer-state/1\0"
_DOMAIN_COMMIT = b"nanrouter.nan-layer-commitment/1\0"
_EDGE_TYPES = frozenset((
    "isa", "part_of", "defines", "supports", "refutes",
    "equivalent", "transforms", "causes", "depends_on",
    "similar", "contradicts", "observes", "routes_to",
))
_HEADER_KEYS = (
    "schema", "topology_sha256", "parent_root_sha256",
    "chart_generation", "total_edges", "candidate_edges",
    "retracted_edges", "layer_counts", "state_sha256", "history_sha256",
)
_RECORD_KEYS = (
    "edge_id", "src", "dst", "layer", "relation", "evidence_id",
    "evidence_version", "evidence_digest", "source_head",
    "generation", "tick", "status",
)


class RouterGraphPageError(ValueError):
    pass


def _integer(v: Any, lo: int = 0, hi: int | None = None) -> bool:
    return type(v) is int and v >= lo and (hi is None or v <= hi)


def _sha(v: Any) -> bool:
    return type(v) is str and _SHA.fullmatch(v) is not None


def _route(cell: int) -> str:
    return "HW0:" + str(cell) if cell < 256 else "HW0:MP" + str(cell - 256)


def _heads(cell: int) -> tuple[int, ...]:
    if cell < 256:
        return (cell,) if cell else ()
    return tuple(MirrorTopology.mp(cell - 256).as_dict()["writable_heads"])


def _check_record(item: Any) -> bytes:
    if type(item) is not dict or any(k not in item for k in _RECORD_KEYS):
        raise RouterGraphPageError("missing or malformed graph entry")
    record = {key: item[key] for key in _RECORD_KEYS}
    if (not _integer(record["src"], 0, CANONICAL_CELLS - 1)
            or not _integer(record["dst"], 0, CANONICAL_CELLS - 1)
            or not _integer(record["layer"], 0, 9)
            or not _integer(record["source_head"], 1, 255)
            or record["source_head"] not in
                (_heads(record["src"]) + _heads(record["dst"]))
            or not _integer(record["generation"], 1)
            or not _integer(record["tick"])
            or not _integer(record["evidence_version"], 1)
            or not _sha(record["evidence_digest"])
            or record["relation"] not in _EDGE_TYPES
            or record["status"] not in ("candidate", "retracted")
            or type(record["edge_id"]) is not str
            or re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", record["edge_id"]) is None
            or type(record["evidence_id"]) is not str
            or not record["evidence_id"]
            or item.get("src_route") != _route(record["src"])
            or item.get("dst_route") != _route(record["dst"])):
        raise RouterGraphPageError("graph entry violates canonical topology")
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


class RouterGraphPageVerifier:
    """Stream one pinned snapshot, rejecting reorders, stale or mixed epochs."""

    def __init__(self, chart: Chart8MiB):
        if not isinstance(chart, Chart8MiB):
            raise RouterGraphPageError("requires actual local HERMES 8 MiB chart")
        root = read_universe_root(chart)
        if not root.get("mounted") or root.get("topology_sha256") != TOPOLOGY_DIGEST:
            raise RouterGraphPageError("missing or incompatible chart root")
        self.chart = chart
        self.parent = chart.root_hash
        self.generation = chart.snapshot_generation
        if not _sha(self.parent) or not _integer(self.generation):
            raise RouterGraphPageError("invalid chart parent/epoch")
        self._state = hashlib.sha256(_DOMAIN_STATE)
        self._next = 0
        self._last_id = None
        self._header = None
        self._commitment = None
        self._counts = [0] * 10
        self._active = self._retracted = 0
        self._failed = False
        self._complete = False

    def add_page(self, page: dict) -> dict:
        if self._failed or self._complete:
            raise RouterGraphPageError("verification already failed or completed")
        try:
            return self._add_page(page)
        except BaseException:
            self._failed = True
            raise

    def _add_page(self, page: dict) -> dict:
        root = read_universe_root(self.chart)
        if (not root.get("mounted") or self.chart.root_hash != self.parent
                or self.chart.snapshot_generation != self.generation):
            raise RouterGraphPageError("chart root or generation changed")
        if type(page) is not dict or not all(k in page for k in _HEADER_KEYS):
            raise RouterGraphPageError("missing graph commitment fields")
        header = {k:page[k] for k in _HEADER_KEYS}
        if (header["schema"] != SCHEMA
                or header["topology_sha256"] != TOPOLOGY_DIGEST
                or header["parent_root_sha256"] != self.parent
                or not _integer(header["chart_generation"])
                or header["chart_generation"] != self.generation
                or not all(_integer(header[k]) for k in (
                    "total_edges","candidate_edges","retracted_edges"))
                or header["candidate_edges"] + header["retracted_edges"]
                    != header["total_edges"]
                or type(header["layer_counts"]) is not list
                or len(header["layer_counts"]) != 10
                or not all(_integer(x) for x in header["layer_counts"])
                or sum(header["layer_counts"]) != header["total_edges"]
                or not _sha(header["state_sha256"])
                or not _sha(header["history_sha256"])
                or not _sha(page.get("graph_commitment_sha256"))
                or page.get("signed_by_router") is not False
                or page.get("native_mth_receipt") is not False
                or page.get("mathematical_truth_promoted") is not False
                or page.get("snapshot_consistent") is not True):
            raise RouterGraphPageError("invalid or overclaimed graph header")
        canonical = json.dumps(header,sort_keys=True,separators=(",",":")).encode()
        digest = hashlib.sha256(_DOMAIN_COMMIT + canonical).hexdigest()
        if not hmac.compare_digest(page["graph_commitment_sha256"], digest):
            raise RouterGraphPageError("graph commitment checksum mismatch")
        if self._header is None:
            self._header = header
            self._commitment = digest
        elif header != self._header or digest != self._commitment:
            raise RouterGraphPageError("mixed graph snapshot revisions")
        if not _integer(page.get("offset")) or page["offset"] != self._next:
            raise RouterGraphPageError("graph pagination gap or replay")
        entries = page.get("entries")
        if type(entries) is not list or len(entries) > 100:
            raise RouterGraphPageError("oversized or malformed graph page")
        total = header["total_edges"]
        if (self._next + len(entries) > total
                or (self._next < total and not entries)):
            raise RouterGraphPageError("missing graph page records")
        for entry in entries:
            raw = _check_record(entry)
            eid = entry["edge_id"]
            if self._last_id is not None and eid <= self._last_id:
                raise RouterGraphPageError("unordered or duplicate graph edge")
            self._state.update(len(raw).to_bytes(8, "big") + raw)
            self._last_id = eid
            self._counts[entry["layer"]] += 1
            if entry["status"] == "candidate":
                self._active += 1
            else:
                self._retracted += 1
        self._next += len(entries)
        expected_next = self._next if self._next < total else None
        if page.get("next_offset") != expected_next:
            raise RouterGraphPageError("invalid graph continuation cursor")
        if self._next == total:
            if (self._state.hexdigest() != header["state_sha256"]
                    or self._counts != header["layer_counts"]
                    or self._active != header["candidate_edges"]
                    or self._retracted != header["retracted_edges"]):
                raise RouterGraphPageError("graph contents do not match commitment")
            self._complete = True
        return self.status()

    def status(self) -> dict:
        return {
            "schema": "nanlm.router-graph-page-verifier/1",
            "verified_entries": self._next,
            "total_edges": None if self._header is None else self._header["total_edges"],
            "graph_commitment_sha256": self._commitment,
            "state_verified": self._complete and not self._failed,
            "chart_root_bound": self._complete and not self._failed,
            "history_digest_declared_not_independently_replayed": True,
            "sender_authenticated": False,
            "native_mth_executed": False,
            "failed": self._failed,
        }
