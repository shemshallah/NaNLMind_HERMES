"""HERMES isolated 8 MiB experimental register with inspectable hierarchical leaves.

The SQLite index and byte payloads live in a bounded 8 MiB file; no canonical
NaNLMind head, BS0 or MC0 state is modified by this module. A native NaN16DE
execution claim requires a separately attested guest operation. This module
only stages candidate data for native tool experiments and gated merges.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path

MAX_REGISTER_BYTES = 8 * 1024 * 1024
MAX_LEAF_BYTES = 64 * 1024
MAX_LEAVES = 4096
_NAME = re.compile(r"^(?:vector|hyperedge|matrix|transform|math|relation|word|logic)[0-9]{1,6}$")
_SEGMENT = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_LOCK = threading.RLock()


class ScratchError(ValueError):
    pass


def _root(ctx):
    return Path(str(ctx.get_config("state_dir", "~/.local/share/nanlmind-hermes"))).expanduser() / "experimental-register"


def _db(ctx):
    path = _root(ctx)
    path.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path / "hermes-8mib.sqlite", timeout=10)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA secure_delete=ON")
    database.execute("PRAGMA max_page_count=2048")  # 2048 x 4096 bytes
    database.execute("""CREATE TABLE IF NOT EXISTS leaves(
        head INTEGER NOT NULL CHECK(head BETWEEN 1 AND 255),
        branch TEXT NOT NULL, leaf TEXT NOT NULL,
        payload BLOB NOT NULL, digest TEXT NOT NULL,
        revision INTEGER NOT NULL,
        PRIMARY KEY(head,branch,leaf))""")
    database.execute("""CREATE TABLE IF NOT EXISTS audit(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, head INTEGER NOT NULL,
        branch TEXT NOT NULL, leaf TEXT NOT NULL, digest TEXT NOT NULL,
        revision INTEGER NOT NULL, operation TEXT NOT NULL)""")
    return database


def _address(head, branch=None, leaf=None):
    if type(head) is not int or not 1 <= head <= 255:
        raise ScratchError("HW0:0 is the root; scratch head must be HW0:1..255")
    if branch is not None and not _NAME.fullmatch(branch):
        raise ScratchError("branch must be a numbered mathematical category")
    if leaf is not None and (branch is None or not _SEGMENT.fullmatch(leaf)):
        raise ScratchError("leaf must be a bounded lowercase name under a branch")
    return f"HW0:{head}" + (f"::{branch}" if branch else "") + (f"::{leaf}" if leaf else "")


def inspect(ctx, head=None, branch=None, leaf=None, limit=128):
    """Root -> head -> category -> leaf. Return digests, never unbounded blobs."""
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ScratchError("limit out of bounds")
    if head is not None:
        _address(head, branch, leaf)
    elif branch is not None or leaf is not None:
        raise ScratchError("branch requires head")
    with _LOCK, _db(ctx) as db:
        if head is None:
            rows = db.execute("SELECT head,COUNT(*) AS leaves FROM leaves GROUP BY head ORDER BY head LIMIT ?", (limit,)).fetchall()
            return {"root": "BS0:0/HW0:0", "layer": "head", "heads": 255,
                    "populated": [dict(row) for row in rows],
                    "merge_gates": {"BS0": "required", "MC0": "required"},
                    "scratch_only": True, "capacity_bytes": MAX_REGISTER_BYTES}
        if branch is None:
            rows = db.execute("SELECT branch,COUNT(*) AS leaves FROM leaves WHERE head=? GROUP BY branch ORDER BY branch LIMIT ?", (head,limit)).fetchall()
            return {"address": _address(head), "layer": "branch",
                    "branches": [dict(row) for row in rows], "scratch_only": True}
        if leaf is None:
            rows = db.execute("SELECT leaf,digest,revision,length(payload) AS bytes FROM leaves WHERE head=? AND branch=? ORDER BY leaf LIMIT ?", (head,branch,limit)).fetchall()
            return {"address": _address(head,branch), "layer": "leaf",
                    "leaves": [dict(row) for row in rows], "scratch_only": True}
        row = db.execute("SELECT payload,digest,revision FROM leaves WHERE head=? AND branch=? AND leaf=?", (head,branch,leaf)).fetchone()
        if row is None:
            return {"address": _address(head,branch,leaf), "found": False}
        return {"address": _address(head,branch,leaf), "found": True,
                "data_hex": bytes(row["payload"]).hex(), "digest": row["digest"],
                "revision": row["revision"], "scratch_only": True}


def mutate(ctx, head, branch, leaf, data_hex, expected_revision=0):
    """Optimistic revisioned, digest-logged experiment; no implicit merge."""
    address = _address(head, branch, leaf)
    if (not isinstance(data_hex, str) or len(data_hex) > MAX_LEAF_BYTES * 2
            or len(data_hex) % 2):
        raise ScratchError("payload must be <=64KiB canonical hex")
    try:
        raw = bytes.fromhex(data_hex)
    except ValueError as exc:
        raise ScratchError("payload must be hexadecimal") from exc
    if raw.hex() != data_hex.lower():
        raise ScratchError("payload must be canonical hexadecimal")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ScratchError("invalid optimistic revision")
    digest = hashlib.sha256(raw).hexdigest()
    with _LOCK, _db(ctx) as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT revision FROM leaves WHERE head=? AND branch=? AND leaf=?", (head,branch,leaf)).fetchone()
        current = old["revision"] if old else 0
        if current != expected_revision:
            raise ScratchError("stale scratch revision; re-inspect before writing")
        count = db.execute("SELECT COUNT(*) FROM leaves").fetchone()[0]
        if old is None and count >= MAX_LEAVES:
            raise ScratchError("scratch leaf limit reached")
        used = db.execute("SELECT COALESCE(SUM(length(payload)),0) FROM leaves").fetchone()[0]
        old_size = db.execute("SELECT length(payload) FROM leaves WHERE head=? AND branch=? AND leaf=?", (head,branch,leaf)).fetchone()
        if used + len(raw) - (old_size[0] if old_size else 0) > MAX_REGISTER_BYTES // 2:
            raise ScratchError("scratch payload quota exceeded")
        revision = current + 1
        db.execute("""INSERT INTO leaves(head,branch,leaf,payload,digest,revision)
                      VALUES(?,?,?,?,?,?)
                      ON CONFLICT(head,branch,leaf) DO UPDATE SET
                      payload=excluded.payload,digest=excluded.digest,revision=excluded.revision""",
                   (head,branch,leaf,raw,digest,revision))
        db.execute("INSERT INTO audit(head,branch,leaf,digest,revision,operation) VALUES(?,?,?,?,?,'scratch-mutate')",
                   (head,branch,leaf,digest,revision))
    return {"address": address, "revision": revision, "sha256": digest,
            "bytes": len(raw), "canonical_merged": False, "native_nan16_execution": False}


def audit(ctx, limit=64):
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ScratchError("invalid audit limit")
    with _LOCK, _db(ctx) as db:
        rows = db.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return {"operations": [dict(row) for row in rows], "scratch_only": True}
