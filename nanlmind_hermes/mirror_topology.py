"""Resident 256-head / 32,640-MP / 65,536-address mirror topology.

HERMES needs the same logical addressing tree as NaNROUTER without copying the
physical state.  The 128 KiB forward map, 65,792-byte inverse map and the
versioned leaf bindings are **logical topology**, not a second 8 MiB chart.
The single physical 8 MiB chart remains owned by `Chart8MiB`/the native guest.

The rank/unrank transform and topology digest are pinned to NaNROUTER's
`runner_riddle/mirror_point_register.py` (PR #74).  A mismatch fails closed.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

HEADS = 256
NANNET_HEAD = 0
FIXED_POINTS = HEADS
PAIR_COUNT = HEADS * (HEADS - 1) // 2
CANONICAL_CELLS = FIXED_POINTS + PAIR_COUNT
RAW_ADDRESSES = HEADS * HEADS
LOOKUP_BYTES = RAW_ADDRESSES * 2
INVERSE_BYTES = CANONICAL_CELLS * 2
TOPOLOGY_SCHEMA = "nanlm.mirror-topology/1"
EXPECTED_TOPOLOGY_DIGEST = "13b02d7d45fa00c1a5df9586e630f44033ea49de21b12a14f5350c1f01f90016"
ROOT_ROUTE = "nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0"
LAYERS = 10
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ROUTE = re.compile(
    r"^nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0"
    r"(?:::(?:HW0:(?P<head>\d{1,3})|HW0:MP(?P<mp>\d{1,5})))"
    r"(?:::(?:HW0:MP(?P<target_mp>\d{1,5})))?"
    r"::L(?P<layer>[0-9])$"
)


class MirrorTopologyError(ValueError):
    pass


def _point(value: Any, name: str = "head") -> int:
    if type(value) is not int or not 0 <= value < HEADS:
        raise MirrorTopologyError(f"{name} must be an integer 0..255")
    return value


def _ordinal(value: Any) -> int:
    if type(value) is not int or not 0 <= value < PAIR_COUNT:
        raise MirrorTopologyError("MP ordinal must be 0..32639")
    return value


def reverse8(value: int) -> int:
    value = _point(value, "byte")
    value = (value >> 4) | ((value & 15) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    return ((value & 0xAA) >> 1) | ((value & 0x55) << 1)


def reverse16(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise MirrorTopologyError("16-bit address required")
    return (reverse8(value & 255) << 8) | reverse8(value >> 8)


def address(a: int, b: int) -> int:
    return (_point(a) << 8) | reverse8(_point(b))


def pair_number(a: int, b: int) -> int:
    a, b = _point(a), _point(b)
    if a == b:
        raise MirrorTopologyError("palindromic head is not an MP leaf")
    if a > b:
        a, b = b, a
    return a * (2 * HEADS - a - 1) // 2 + b - a - 1


_START = tuple(a * (2 * HEADS - a - 1) // 2 for a in range(HEADS))


def pair_heads(mp: int) -> tuple[int, int]:
    mp = _ordinal(mp)
    a = bisect_right(_START, mp) - 1
    return a, a + 1 + mp - _START[a]


def _materialize() -> tuple[bytes, bytes]:
    raw_to_cell = bytearray(LOOKUP_BYTES)
    cell_to_raw = bytearray(INVERSE_BYTES)
    for head in range(HEADS):
        fixed = address(head, head)
        struct.pack_into("<H", raw_to_cell, 2 * fixed, head)
        struct.pack_into("<H", cell_to_raw, 2 * head, fixed)
    for a in range(HEADS):
        for b in range(a + 1, HEADS):
            n = pair_number(a, b)
            cell = FIXED_POINTS + n
            forward, mirror = address(a, b), address(b, a)
            struct.pack_into("<H", raw_to_cell, 2 * forward, cell)
            struct.pack_into("<H", raw_to_cell, 2 * mirror, cell)
            struct.pack_into("<H", cell_to_raw, 2 * cell, forward)
    return bytes(raw_to_cell), bytes(cell_to_raw)


RAW_TO_CELL, CELL_TO_RAW = _materialize()
TOPOLOGY_DIGEST = hashlib.sha256(
    b"nanrouter.mirror-point-register/1\0" + RAW_TO_CELL + CELL_TO_RAW
).hexdigest()
if TOPOLOGY_DIGEST != EXPECTED_TOPOLOGY_DIGEST:
    raise MirrorTopologyError("mirror topology digest does not match NaNROUTER")


@dataclass(frozen=True)
class Microtable:
    kind: str
    ordinal: int
    a: int
    b: int
    forward_address: int
    mirror_address: int
    canonical_cell: int
    canonical_address: int
    topology_sha256: str = TOPOLOGY_DIGEST

    def as_dict(self, source: int | None = None, target: int | None = None) -> dict[str, Any]:
        if self.kind == "head":
            route = f"HW0:{self.ordinal}"
            heads = [self.ordinal]
            orientation = "fixed"
        else:
            route = f"HW0:MP{self.ordinal}"
            heads = [self.a, self.b]
            orientation = ("forward" if source is None or target is None or source < target
                           else "mirror")
        return {
            "kind": self.kind,
            "route": route,
            "ordinal": self.ordinal,
            "heads": heads,
            "writable_heads": sorted(h for h in heads if h != NANNET_HEAD),
            "addresses": [self.forward_address, self.mirror_address],
            "canonical_address": self.canonical_address,
            "canonical_cell": self.canonical_cell,
            "orientation": orientation,
            "topology_sha256": self.topology_sha256,
        }


class MirrorTopology:
    @staticmethod
    def head(head: int) -> Microtable:
        head = _point(head)
        raw = address(head, head)
        return Microtable("head", head, head, head, raw, raw, head, raw)

    @staticmethod
    def mp(ordinal: int) -> Microtable:
        ordinal = _ordinal(ordinal)
        a, b = pair_heads(ordinal)
        forward, mirror = address(a, b), address(b, a)
        cell = struct.unpack_from("<H", RAW_TO_CELL, forward * 2)[0]
        return Microtable("mp", ordinal, a, b, forward, mirror, cell, forward)

    @classmethod
    def from_heads(cls, a: int, b: int) -> dict[str, Any]:
        a, b = _point(a), _point(b)
        table = cls.head(a) if a == b else cls.mp(pair_number(a, b))
        return table.as_dict(a, b)

    @classmethod
    def from_raw(cls, raw: int) -> dict[str, Any]:
        if type(raw) is not int or not 0 <= raw < RAW_ADDRESSES:
            raise MirrorTopologyError("16-bit raw address required")
        a, b = raw >> 8, reverse8(raw & 255)
        result = cls.from_heads(a, b)
        if result["canonical_cell"] != struct.unpack_from("<H", RAW_TO_CELL, 2 * raw)[0]:
            raise MirrorTopologyError("forward map disagreement")
        return result

    @staticmethod
    def status() -> dict[str, Any]:
        return {
            "schema": TOPOLOGY_SCHEMA,
            "root": ROOT_ROUTE,
            "heads": FIXED_POINTS,
            "mirror_points": PAIR_COUNT,
            "canonical_cells": CANONICAL_CELLS,
            "raw_addresses": RAW_ADDRESSES,
            "lookup_bytes": LOOKUP_BYTES,
            "inverse_bytes": INVERSE_BYTES,
            "topology_sha256": TOPOLOGY_DIGEST,
            "nannet0_writable": False,
            "physical_chart_duplicated": False,
            "layers": list(range(LAYERS)),
        }

    @classmethod
    def route(cls, *, head: int | None = None, mp: int | None = None,
              target_mp: int | None = None, layer: int = 5) -> str:
        if type(layer) is not int or not 0 <= layer < LAYERS:
            raise MirrorTopologyError("layer must be 0..9")
        if mp is not None:
            _ordinal(mp)
            prefix = f"{ROOT_ROUTE}::HW0:MP{mp}"
        elif head is not None:
            _head = _point(head)
            prefix = f"{ROOT_ROUTE}::HW0:{_head}"
        else:
            raise MirrorTopologyError("head or MP required")
        if target_mp is not None:
            _ordinal(target_mp)
            prefix += f"::HW0:MP{target_mp}"
        return f"{prefix}::L{layer}"

    @classmethod
    def parse_route(cls, value: str) -> dict[str, Any]:
        match = _ROUTE.fullmatch(str(value))
        if match is None:
            raise MirrorTopologyError("invalid recursive nan:// route")
        head = int(match.group("head")) if match.group("head") is not None else None
        mp = int(match.group("mp")) if match.group("mp") is not None else None
        target = int(match.group("target_mp")) if match.group("target_mp") is not None else None
        if head is not None:
            table = cls.head(head).as_dict()
        else:
            table = cls.mp(mp).as_dict()
        if target is not None:
            if table["kind"] == "mp" and target != table["ordinal"]:
                raise MirrorTopologyError("endpoint-constrained route must name its own MP")
            if table["kind"] == "head" and head not in pair_heads(target):
                raise MirrorTopologyError("head is not an endpoint of the target MP")
        return {
            "schema": TOPOLOGY_SCHEMA,
            "route": str(value),
            "head": head,
            "mp": mp,
            "target_mp": target,
            "layer": int(match.group("layer")),
            "microtable": table,
        }


class MirrorLeafBindings:
    """Versioned MP -> immutable leaf pointer in the existing HERMES SQLite DB."""

    SCHEMA = "nanlm.hermes-mirror-leaf-bindings/1"

    def __init__(self, database: str | Path):
        self.database = str(database)
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS hermes_mp_binding (
                    mp INTEGER PRIMARY KEY CHECK(mp BETWEEN 0 AND 32639),
                    leaf_id TEXT NOT NULL,
                    leaf_version INTEGER NOT NULL CHECK(leaf_version >= 1),
                    leaf_digest TEXT NOT NULL,
                    generation INTEGER NOT NULL CHECK(generation >= 1),
                    source_head INTEGER NOT NULL CHECK(source_head BETWEEN 1 AND 255),
                    tick INTEGER NOT NULL CHECK(tick >= 0),
                    status TEXT NOT NULL DEFAULT 'active'
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS hermes_mp_binding_history (
                    mp INTEGER NOT NULL,
                    generation INTEGER NOT NULL CHECK(generation >= 1),
                    leaf_id TEXT NOT NULL,
                    leaf_version INTEGER NOT NULL,
                    leaf_digest TEXT NOT NULL,
                    source_head INTEGER NOT NULL,
                    tick INTEGER NOT NULL,
                    PRIMARY KEY(mp, generation)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        return db

    def bind(
        self,
        *,
        mp: int,
        leaf_id: str,
        leaf_version: int,
        leaf_digest: str,
        source_head: int,
        expected_generation: int,
        tick: int = 0,
    ) -> dict[str, Any]:
        mp = _ordinal(mp)
        _point(source_head, "source head")
        if source_head == NANNET_HEAD or source_head not in pair_heads(mp):
            raise MirrorTopologyError("source head is not a writable endpoint of this MP")
        if not isinstance(leaf_id, str) or not 1 <= len(leaf_id) <= 256:
            raise MirrorTopologyError("leaf_id must be 1..256 characters")
        if type(leaf_version) is not int or leaf_version < 1:
            raise MirrorTopologyError("leaf_version must be a positive integer")
        if not isinstance(leaf_digest, str) or _HEX.fullmatch(leaf_digest) is None:
            raise MirrorTopologyError("leaf_digest must be lowercase SHA-256")
        if type(expected_generation) is not int or expected_generation < 0:
            raise MirrorTopologyError("expected_generation must be >= 0")
        if type(tick) is not int or tick < 0:
            raise MirrorTopologyError("tick must be >= 0")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM hermes_mp_binding WHERE mp=?", (mp,)).fetchone()
            if row is None:
                if expected_generation != 0:
                    raise MirrorTopologyError("new binding must expect generation zero")
                generation = 1
                db.execute(
                    "INSERT INTO hermes_mp_binding(mp,leaf_id,leaf_version,leaf_digest,"
                    "generation,source_head,tick,status) VALUES(?,?,?,?,?,?,?,'active')",
                    (mp, leaf_id, leaf_version, leaf_digest, generation, source_head, tick),
                )
            else:
                if row["generation"] != expected_generation:
                    raise MirrorTopologyError("stale mirror leaf generation")
                generation = int(row["generation"]) + 1
                db.execute(
                    "UPDATE hermes_mp_binding SET leaf_id=?,leaf_version=?,leaf_digest=?,"
                    "generation=?,source_head=?,tick=? WHERE mp=?",
                    (leaf_id, leaf_version, leaf_digest, generation, source_head, tick, mp),
                )
            db.execute(
                "INSERT OR IGNORE INTO hermes_mp_binding_history"
                "(mp,generation,leaf_id,leaf_version,leaf_digest,source_head,tick)"
                " VALUES(?,?,?,?,?,?,?)",
                (mp, generation, leaf_id, leaf_version, leaf_digest, source_head, tick),
            )
            db.execute("COMMIT")
        return {
            "schema": self.SCHEMA,
            "mp": mp,
            "heads": list(pair_heads(mp)),
            "leaf_id": leaf_id,
            "leaf_version": leaf_version,
            "leaf_digest": leaf_digest,
            "generation": generation,
            "source_head": source_head,
            "tick": tick,
            "canonical_cell": MirrorTopology.mp(mp).canonical_cell,
        }

    def resolve(self, mp: int) -> dict[str, Any] | None:
        mp = _ordinal(mp)
        with self._connect() as db:
            row = db.execute("SELECT * FROM hermes_mp_binding WHERE mp=?", (mp,)).fetchone()
        if row is None:
            return None
        return {
            "schema": self.SCHEMA,
            "mp": mp,
            "heads": list(pair_heads(mp)),
            **{key: row[key] for key in
               ("leaf_id", "leaf_version", "leaf_digest", "generation",
                "source_head", "tick", "status")},
        }

    def history(self, mp: int) -> list[dict[str, Any]]:
        mp = _ordinal(mp)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM hermes_mp_binding_history WHERE mp=? ORDER BY generation",
                (mp,),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._connect() as db:
            bindings = db.execute("SELECT COUNT(*) AS n FROM hermes_mp_binding").fetchone()["n"]
            history = db.execute(
                "SELECT COUNT(*) AS n FROM hermes_mp_binding_history"
            ).fetchone()["n"]
        return {"schema": self.SCHEMA, "bindings": bindings, "history": history}


__all__ = [
    "CANONICAL_CELLS",
    "CELL_TO_RAW",
    "EXPECTED_TOPOLOGY_DIGEST",
    "FIXED_POINTS",
    "HEADS",
    "INVERSE_BYTES",
    "LAYERS",
    "LOOKUP_BYTES",
    "Microtable",
    "MirrorLeafBindings",
    "MirrorTopology",
    "MirrorTopologyError",
    "NANNET_HEAD",
    "PAIR_COUNT",
    "RAW_ADDRESSES",
    "RAW_TO_CELL",
    "ROOT_ROUTE",
    "TOPOLOGY_DIGEST",
    "TOPOLOGY_SCHEMA",
    "address",
    "pair_heads",
    "pair_number",
    "reverse16",
    "reverse8",
]
