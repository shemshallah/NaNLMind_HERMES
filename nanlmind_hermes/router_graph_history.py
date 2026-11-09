"""Independent read-only replay of NaNROUTER's committed graph revisions.

State projections are verified by RouterGraphPageVerifier. This companion
replays the ordered history records from the SAME router graph commitment,
checks valid edge transitions, and checks each latest historical revision
against the independently streamed present-state page. It does not verify a
sender signature, prove evidence truth, execute MTH, or admit model updates.
"""
from __future__ import annotations

import hashlib
import hmac
import json

from .router_graph_page import (
    RouterGraphPageError, RouterGraphPageVerifier,
    _HEADER_KEYS, _RECORD_KEYS, _check_record, _integer, _route,
)
from .universe_root import read_universe_root

_HISTORY_DOMAIN = b"nanrouter.nan-layer-history/1\0"
_MAX_PAGE = 100
_MAX_PAYLOAD_BYTES = 65536
_IMMUTABLE = ("src", "dst", "source_head")
_RETRACTION_PRESERVED = (
    "src", "dst", "layer", "relation", "evidence_id",
    "evidence_version", "evidence_digest", "source_head",
)


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


class RouterGraphHistoryVerifier:
    """Read-only two-phase graph-state and full revision-chain verifier.

    State pages must arrive first, then history pages. Memory holds current
    state record hashes only (not a new graph, chart, or mutable model).
    Each finished edge is removed from the verification map.
    """

    def __init__(self, chart):
        self.state = RouterGraphPageVerifier(chart)
        self._records = {}
        self._history = hashlib.sha256(_HISTORY_DOMAIN)
        self._next = 0
        self._total = None
        self._last_id = None
        self._last_record = None
        self._complete = False
        self._failed = False
        self._started = False

    def add_state_page(self, page):
        if self._failed or self._started or self.state.status()["state_verified"]:
            raise RouterGraphPageError("graph state phase closed")
        try:
            result = self.state.add_page(page)
            for entry in page["entries"]:
                raw = _check_record(entry)
                if entry["edge_id"] in self._records:
                    raise RouterGraphPageError("duplicate state edge")
                self._records[entry["edge_id"]] = raw
            return self.status()
        except BaseException:
            self._failed = True
            raise

    def _finish_edge(self):
        if self._last_record is None:
            return
        expected = self._records.pop(self._last_id, None)
        if expected is None or not hmac.compare_digest(
                expected, _canonical(self._last_record)):
            raise RouterGraphPageError("latest history revision differs from graph state")

    def add_history_page(self, page):
        if (self._failed or self._complete
                or not self.state.status()["state_verified"]):
            raise RouterGraphPageError("history phase unavailable")
        self._started = True
        try:
            self._add_history_page(page)
            return self.status()
        except BaseException:
            self._failed = True
            raise

    def _add_history_page(self, page):
        chart = self.state.chart
        root = read_universe_root(chart)
        if (not root.get("mounted")
                or chart.root_hash != self.state.parent
                or chart.snapshot_generation != self.state.generation):
            raise RouterGraphPageError("chart root or generation changed")
        header = self.state._header
        if (type(page) is not dict
                or any(page.get(k) != header[k] for k in _HEADER_KEYS)
                or page.get("graph_commitment_sha256") != self.state._commitment
                or page.get("snapshot_consistent") is not True
                or page.get("signed_by_router") is not False
                or page.get("native_mth_receipt") is not False
                or page.get("mathematical_truth_promoted") is not False):
            raise RouterGraphPageError("history page not bound to verified state")
        total = page.get("history_total")
        offset = page.get("history_offset")
        entries = page.get("history_entries")
        if (not _integer(total) or total < header["total_edges"]
                or (self._total is not None and total != self._total)
                or not _integer(offset) or offset != self._next
                or type(entries) is not list or len(entries) > _MAX_PAGE
                or self._next + len(entries) > total
                or (self._next < total and not entries)):
            raise RouterGraphPageError("invalid history pagination")
        self._total = total
        for item in entries:
            if (type(item) is not dict
                    or set(item) != {"edge_id", "generation", "payload"}
                    or type(item["payload"]) is not str
                    or len(item["payload"].encode("utf-8")) > _MAX_PAYLOAD_BYTES):
                raise RouterGraphPageError("invalid history revision")
            payload = item["payload"]
            try:
                record = json.loads(payload)
            except (TypeError, ValueError) as exc:
                raise RouterGraphPageError("invalid history JSON") from exc
            if (type(record) is not dict or set(record) != set(_RECORD_KEYS)
                    or _canonical(record).decode() != payload
                    or item["edge_id"] != record["edge_id"]
                    or type(item["generation"]) is not int
                    or item["generation"] != record["generation"]):
                raise RouterGraphPageError("noncanonical history revision")
            _check_record({
                **record,
                "src_route": _route(record["src"]),
                "dst_route": _route(record["dst"]),
            })
            eid, generation = record["edge_id"], record["generation"]
            if self._last_id is None or eid != self._last_id:
                if self._last_id is not None and eid <= self._last_id:
                    raise RouterGraphPageError("unordered history edge IDs")
                self._finish_edge()
                if generation != 1 or record["status"] != "candidate":
                    raise RouterGraphPageError("history must begin at candidate generation 1")
            else:
                if (generation != self._last_record["generation"] + 1
                        or record["tick"] < self._last_record["tick"]
                        or any(record[k] != self._last_record[k]
                               for k in _IMMUTABLE)):
                    raise RouterGraphPageError("invalid history generation or immutable endpoints")
                if record["status"] == "retracted":
                    if (self._last_record["status"] != "candidate"
                            or any(record[k] != self._last_record[k]
                                   for k in _RETRACTION_PRESERVED)):
                        raise RouterGraphPageError("invalid tombstone transition")
            raw = _canonical([eid, generation, payload])
            self._history.update(len(raw).to_bytes(8, "big") + raw)
            self._last_id, self._last_record = eid, record
        self._next += len(entries)
        expected_next = self._next if self._next < total else None
        if page.get("history_next_offset") != expected_next:
            raise RouterGraphPageError("history cursor gap or replay")
        if self._next == total:
            self._finish_edge()
            if self._records:
                raise RouterGraphPageError("orphan graph-state edge without history")
            if not hmac.compare_digest(
                    self._history.hexdigest(), header["history_sha256"]):
                raise RouterGraphPageError("history digest mismatch")
            self._complete = True

    def status(self):
        state = self.state.status()
        return {
            "schema": "nanlm.router-graph-history-verifier/1",
            "state_verified": state["state_verified"] and not self._failed,
            "history_verified": self._complete and not self._failed,
            "verified_history_revisions": self._next,
            "expected_history_revisions": self._total,
            "graph_commitment_sha256": state["graph_commitment_sha256"],
            "chart_root_bound": state["chart_root_bound"] and not self._failed,
            "sender_authenticated": False,
            "native_mth_executed": False,
            "learning_admitted": False,
            "mathematical_truth_promoted": False,
            "failed": self._failed or state["failed"],
        }


__all__ = ("RouterGraphHistoryVerifier",)
