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

SCHEMA = "nanlm-hermes-memory/2"\nRESOLUTION_INDEX = {\n    "bit": 0, "nibble": 1, "byte": 2, "phonetic": 3, "morpheme": 4,\n    "lexical": 5, "word_completion": 6, "sentence": 7,\n    "paragraph": 8, "narrative": 9,\n}
VALID_KINDS = {"entity", "relation", "operator", "equation", "constraint", "observation", "hypothesis", "counterexample"}
VALID_POLARITIES = {"supports", "refutes", "neutral"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class MemoryStore:
    """Append-only discovery log plus rebuildable head/layer projections."""

    def __init__(self, path: str | Path, heads: int = 256, layers: int = 10):
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
                CREATE TABLE IF NOT EXISTS assessments(
                    assessment_id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                    status TEXT NOT NULL, support_weight REAL NOT NULL,
                    refute_weight REAL NOT NULL, evidence_count INTEGER NOT NULL,
                    source_event_ids TEXT NOT NULL, assessor_version TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assessment_subject ON assessments(subject);
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
        provenance = item.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be an object")
        provenance = dict(provenance)
        provenance.setdefault("origin_head", head)
        provenance.setdefault("origin_layer", layer)
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")
        payload = {
            "schema": SCHEMA, "kind": kind, "subject": item.get("subject"),
            "predicate": item.get("predicate"), "object": item.get("object"),
            "expression": item.get("expression"), "units": item.get("units"),
            "conditions": item.get("conditions", []), "participants": item.get("participants", []),
            "polarity": polarity, "confidence": float(confidence),
            "evidence": evidence, "provenance": provenance,
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
            with self._db() as db:
                before = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            self.ingest(record["item"], source_digest=record["source_digest"],
                        head=record.get("head"), layer=record.get("layer", 0))
            with self._db() as db:
                after = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            if after > before:
                accepted += 1
            else:
                duplicates += 1
        return {"accepted": accepted, "duplicates": duplicates}

    def reorganize(self) -> dict[str, int]:
        """Rebuild routes and evidence assessments from the immutable event set."""
        assessor_version = "evidence-review/1"
        with self._db() as db:
            rows = db.execute("SELECT event_id,payload FROM events ORDER BY event_id").fetchall()
            grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
            for row in rows:
                payload = json.loads(row["payload"])
                subject = str(payload.get("subject") or payload["kind"])
                grouped.setdefault(subject, []).append((row["event_id"], payload))

            assessments: dict[str, dict[str, Any]] = {}
            contested: set[str] = set()
            for subject, claims in grouped.items():
                support = refute = 0.0
                for _, payload in claims:
                    polarity = payload["polarity"]
                    evidence = payload.get("evidence", {})
                    samples = evidence.get("test_samples", evidence.get("samples", 1))
                    if isinstance(samples, bool) or not isinstance(samples, (int, float)) or samples < 0:
                        samples = 1
                    weight = float(payload["confidence"]) * max(1.0, float(samples))
                    if polarity == "supports":
                        support += weight
                    elif polarity == "refutes":
                        refute += weight
                if support > 0 and refute > 0:
                    status = "contested"
                    contested.add(subject)
                elif support > refute:
                    status = "supported"
                elif refute > support:
                    status = "refuted"
                else:
                    status = "unresolved"
                event_ids = sorted(event_id for event_id, _ in claims)
                assessment_id = _digest([assessor_version, subject, status, support, refute, event_ids])
                assessments[subject] = {
                    "assessment_id": assessment_id, "subject": subject, "status": status,
                    "support_weight": support, "refute_weight": refute,
                    "evidence_count": len(event_ids), "source_event_ids": event_ids,
                    "assessor_version": assessor_version,
                }

            db.execute("UPDATE projections SET active=0")
            for row in rows:
                payload = json.loads(row["payload"])
                provenance = payload.get("provenance", {})
                origin_head = provenance.get("native_head", provenance.get("origin_head"))
                if isinstance(origin_head, bool) or not isinstance(origin_head, int) or not 0 <= origin_head < self.heads:
                    origin_head = int(_digest({"route": payload.get("subject") or payload["kind"]})[:8], 16) % self.heads
                subject = str(payload.get("subject") or payload["kind"])
                layer = min(self.layers - 1, {
                    "observation": 0, "entity": 1, "relation": 2,
                    "operator": 3, "equation": 4, "constraint": 5,
                    "hypothesis": 5, "counterexample": self.layers - 1,
                }[payload["kind"]])
                # Linguistic resolution is a first-class routing signal shared by
                # conversational reflection and native runner discoveries.  It
                # refines (rather than replaces) the semantic kind default.
                resolution = provenance.get("resolution")
                if provenance.get("cross_resolution") is True:
                    layer = min(self.layers - 1, 9)
                elif resolution in RESOLUTION_INDEX:
                    layer = min(self.layers - 1, RESOLUTION_INDEX[resolution])
                else:
                    # Native runner discoveries carry an explicit layer from
                    # the 256-head topology. Preserve that route when rebuilding
                    # the shared projection; chat/extracted claims without a
                    # native head continue to use semantic-kind defaults.
                    native_head = provenance.get("native_head")
                    origin_layer = provenance.get("origin_layer")
                    if (isinstance(native_head, int) and not isinstance(native_head, bool)
                            and 0 <= native_head < self.heads
                            and isinstance(origin_layer, int) and not isinstance(origin_layer, bool)
                            and 0 <= origin_layer < self.layers):
                        layer = origin_layer
                # Contested claims move to the highest review layer without
                # erasing their origin head or changing the source event.
                if subject in contested:
                    layer = self.layers - 1
                pid = _digest([row["event_id"], origin_head, layer, assessor_version])
                db.execute("INSERT OR IGNORE INTO projections(projection_id,event_id,head,layer,role,active) VALUES(?,?,?,?,?,1)",
                           (pid, row["event_id"], origin_head, layer, "evidence-review"))
                db.execute("UPDATE projections SET active=1 WHERE projection_id=?", (pid,))

            db.execute("DELETE FROM assessments")
            for assessment in assessments.values():
                db.execute(
                    "INSERT INTO assessments VALUES(?,?,?,?,?,?,?,?)",
                    (assessment["assessment_id"], assessment["subject"], assessment["status"],
                     assessment["support_weight"], assessment["refute_weight"],
                     assessment["evidence_count"], _canonical(assessment["source_event_ids"]),
                     assessment["assessor_version"]),
                )
            return {"events": len(rows),
                    "active_routes": db.execute("SELECT COUNT(*) FROM projections WHERE active=1").fetchone()[0],
                    "assessments": len(assessments), "contested_subjects": len(contested)}

    def query_assessments(self, subject: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1,1000]")
        sql = "SELECT * FROM assessments"
        args: list[Any] = []
        if subject is not None:
            sql += " WHERE subject=?"
            args.append(subject)
        sql += " ORDER BY subject LIMIT ?"
        args.append(limit)
        with self._db() as db:
            return [{
                "assessment_id": row["assessment_id"], "subject": row["subject"],
                "status": row["status"], "support_weight": row["support_weight"],
                "refute_weight": row["refute_weight"], "evidence_count": row["evidence_count"],
                "source_event_ids": json.loads(row["source_event_ids"]),
                "assessor_version": row["assessor_version"],
            } for row in db.execute(sql, args)]

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
            assessments = db.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
            contested = db.execute("SELECT COUNT(*) FROM assessments WHERE status='contested'").fetchone()[0]
            routes = db.execute("SELECT COUNT(*) FROM projections WHERE active=1").fetchone()[0]
        return {"schema": SCHEMA, "path": str(self.path), "heads": self.heads,
                "layers": self.layers, "events": events, "graph_edges": edges,
                "hyperedges": hyperedges, "assessments": assessments,
                "contested_subjects": contested,
                "active_routes": routes, "append_only": True,
                "weights_checkpoint": False}
