"""Persistent additive mathematical memory for the Hermes universe.

This is a sidecar ledger, not a neural-weight checkpoint. Discoveries are immutable,
content-addressed events; the current graph is a deterministic projection of that log.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "nanlm-hermes-memory/1"
VALID_KINDS = {"entity", "relation", "operator", "equation", "constraint", "observation", "hypothesis", "counterexample"}
VALID_POLARITIES = {"supports", "refutes", "neutral"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class MemoryStore:
    """Append-only discovery log plus rebuildable head/layer projections."""

    def __init__(self, path: str | Path, heads: int = 256, layers: int = 7):
        if isinstance(heads, bool) or not 1 <= heads <= 256:
            raise ValueError("heads must be in [1, 256]")
        if isinstance(layers, bool) or not 1 <= layers <= 32:
            raise ValueError("layers must be in [1, 32]")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heads, self.layers = heads, layers
        with self._db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS events(
                    event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    source_digest TEXT NOT NULL, created_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projections(
                    projection_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id),
                    head INTEGER NOT NULL, layer INTEGER NOT NULL, role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS projection_route ON projections(head, layer, active);
                CREATE TABLE IF NOT EXISTS edges(
                    edge_id TEXT PRIMARY KEY, src TEXT NOT NULL, relation TEXT NOT NULL,
                    dst TEXT NOT NULL, event_id TEXT NOT NULL REFERENCES events(event_id)
                );
                CREATE INDEX IF NOT EXISTS edge_src ON edges(src);
                CREATE INDEX IF NOT EXISTS edge_dst ON edges(dst);
                CREATE TABLE IF NOT EXISTS hyperedges(
                    hyperedge_id TEXT PRIMARY KEY, relation TEXT NOT NULL,
                    members TEXT NOT NULL, event_id TEXT NOT NULL REFERENCES events(event_id)
                );
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema',?)", (SCHEMA,))
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('heads',?)", (str(heads),))
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('layers',?)", (str(layers),))

    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def ingest(self, item: dict[str, Any], *, source_digest: str,
               head: int | None = None, layer: int = 0) -> dict[str, Any]:
        """Validate and append one mathematical claim without overwriting prior claims."""
        if not isinstance(item, dict):
            raise ValueError("discovery must be an object")
        kind = item.get("kind")
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
        if not isinstance(source_digest, str) or len(source_digest) != 64:
            raise ValueError("source_digest must be a SHA-256 hex digest")
        try:
            int(source_digest, 16)
        except ValueError as exc:
            raise ValueError("source_digest must be hexadecimal") from exc
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < self.layers:
            raise ValueError("layer is outside configured universe")
        if head is None:
            head = int(_digest(item)[:8], 16) % self.heads
        if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < self.heads:
            raise ValueError("head is outside configured universe")
        polarity = item.get("polarity", "neutral")
        if polarity not in VALID_POLARITIES:
            raise ValueError("invalid polarity")
        confidence = item.get("confidence", 0.5)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        # Preserve a typed expression tree; do not execute supplied expressions.
        if "expression" in item and not isinstance(item["expression"], (dict, list, str, int, float)):
            raise ValueError("expression must be inert structured data")
        payload = {
            "schema": SCHEMA, "kind": kind, "subject": item.get("subject"),
            "predicate": item.get("predicate"), "object": item.get("object"),
            "expression": item.get("expression"), "units": item.get("units"),
            "conditions": item.get("conditions", []), "participants": item.get("participants", []),
            "polarity": polarity, "confidence": float(confidence),
            "evidence": item.get("evidence", {}), "provenance": item.get("provenance", {}),
            "tags": sorted(set(item.get("tags", []))),
        }
        if not payload["subject"] and kind not in {"observation", "counterexample"}:
            raise ValueError("subject is required for this discovery kind")
        event_id = _digest({"payload": payload, "source_digest": source_digest})
        now = time.time_ns()
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
                       (event_id, _canonical(payload), source_digest, now))
            db.execute("INSERT OR IGNORE INTO projections(projection_id,event_id,head,layer,role) VALUES(?,?,?,?,?)",
                       (_digest([event_id, head, layer]), event_id, head, layer, "discovery"))
            if payload["subject"] and payload["object"] is not None and payload["predicate"]:
                edge = {"s": payload["subject"], "p": payload["predicate"], "o": payload["object"]}
                db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?,?,?)",
                           (_digest(edge), str(edge["s"]), str(edge["p"]), str(edge["o"]), event_id))
            members = payload["participants"]
            if members:
                if not isinstance(members, list) or len(members) < 2 or not all(isinstance(x, (str, int)) for x in members):
                    raise ValueError("participants must be a list of at least two string/integer identifiers")
                relation = str(payload["predicate"] or payload["kind"])
                hyperedge = {"relation": relation, "members": members}
                db.execute("INSERT OR IGNORE INTO hyperedges VALUES(?,?,?,?)",
                           (_digest(hyperedge), relation, _canonical(members), event_id))
        return {"event_id": event_id, "head": head, "layer": layer, "duplicate_safe": True}

    def merge(self, records: Iterable[dict[str, Any]]) -> dict[str, int]:
        """Idempotently merge remote discovery records; no last-writer-wins replacement."""
        accepted = duplicates = 0
        for record in records:
            result = self.ingest(record["item"], source_digest=record["source_digest"],
                                 head=record.get("head"), layer=record.get("layer", 0))
            with self._db() as db:
                count = db.execute("SELECT COUNT(*) FROM projections WHERE event_id=?",
                                   (result["event_id"],)).fetchone()[0]
            if count > 1:
                duplicates += 1
            else:
                accepted += 1
        return {"accepted_or_present": accepted, "duplicate_events": duplicates}

    def reorganize(self) -> dict[str, int]:
        """Recompute routing projections from immutable events, deterministically."""
        with self._db() as db:
            rows = db.execute("SELECT event_id,payload FROM events ORDER BY event_id").fetchall()
            db.execute("UPDATE projections SET active=0")
            for row in rows:
                payload = json.loads(row["payload"])
                # Route related claims together; layer follows abstraction level.
                key = payload.get("subject") or payload.get("kind")
                head = int(_digest({"route": key})[:8], 16) % self.heads
                layer = min(self.layers - 1, {
                    "observation": 0, "entity": 1, "relation": 2,
                    "operator": 3, "equation": 4, "constraint": 5,
                    "hypothesis": 5, "counterexample": self.layers - 1,
                }[payload["kind"]])
                pid = _digest([row["event_id"], head, layer])
                db.execute("INSERT OR IGNORE INTO projections(projection_id,event_id,head,layer,role,active) VALUES(?,?,?,?,?,1)",
                           (pid, row["event_id"], head, layer, "reorganized"))
                db.execute("UPDATE projections SET active=1 WHERE projection_id=?", (pid,))
            return {"events": len(rows), "active_routes": db.execute(
                "SELECT COUNT(*) FROM projections WHERE active=1").fetchone()[0]}

    def query(self, subject: str | None = None, head: int | None = None,
              layer: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1,1000]")
        sql = """SELECT e.event_id,e.payload,p.head,p.layer FROM events e
                 JOIN projections p ON p.event_id=e.event_id WHERE p.active=1"""
        args: list[Any] = []
        if subject is not None:
            sql += " AND json_extract(e.payload,'$.subject')=?"; args.append(subject)
        if head is not None:
            sql += " AND p.head=?"; args.append(head)
        if layer is not None:
            sql += " AND p.layer=?"; args.append(layer)
        sql += " ORDER BY e.event_id LIMIT ?"; args.append(limit)
        with self._db() as db:
            return [{"event_id": r["event_id"], "head": r["head"], "layer": r["layer"],
                     **json.loads(r["payload"])} for r in db.execute(sql, args)]

    def status(self) -> dict[str, Any]:
        with self._db() as db:
            events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            edges = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            hyperedges = db.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0]
            routes = db.execute("SELECT COUNT(*) FROM projections WHERE active=1").fetchone()[0]
        return {"schema": SCHEMA, "path": str(self.path), "heads": self.heads,
                "layers": self.layers, "events": events, "graph_edges": edges,
                "hyperedges": hyperedges,
                "active_routes": routes, "append_only": True,
                "weights_checkpoint": False}
