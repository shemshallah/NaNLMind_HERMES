"""Append-only, hash-chained HERMES scratch shared by CLOCKED and NaNLMind."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LANES = {
    "user", "hermes_plan", "clocked_request", "clocked_result",
    "evidence_atoms", "conflict_map", "nan16_package", "head_response",
    "hermes_synthesis", "head_validation", "final", "learning_proposal",
}
ZERO = "0" * 64


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class ScratchError(ValueError):
    pass


class RevisionConflict(ScratchError):
    pass


class Scratch:
    """A local durable scratch journal. Each append is atomic and parent-bound."""

    def __init__(self, root: str | Path, scratch_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", scratch_id or ""):
            raise ScratchError("invalid scratch_id")
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / (scratch_id + ".jsonl")
        self.lock_path = self.root / (scratch_id + ".lock")
        self.scratch_id = scratch_id

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        previous = ZERO
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScratchError(f"corrupt scratch journal at line {number}") from exc
            claimed = row.pop("revision", None)
            if row.get("parent_revision") != previous or digest(row) != claimed:
                raise ScratchError(f"scratch hash chain mismatch at line {number}")
            row["revision"] = claimed
            rows.append(row)
            previous = claimed
        return rows

    def snapshot(self) -> dict:
        rows = self._read()
        return {
            "scratch_id": self.scratch_id,
            "revision": rows[-1]["revision"] if rows else ZERO,
            "entries": rows,
            "entry_count": len(rows),
        }

    def append(self, lane: str, writer: str, payload: Any,
               expected_revision: str | None = None) -> dict:
        if lane not in LANES:
            raise ScratchError("unknown scratch lane")
        if writer not in {"user", "hermes", "clocked", "nanlmind", "policy"}:
            raise ScratchError("unknown scratch writer")
        if len(canonical(payload)) > 1_048_576:
            raise ScratchError("scratch payload exceeds 1 MiB")
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            rows = self._read()
            parent = rows[-1]["revision"] if rows else ZERO
            if expected_revision is not None and expected_revision != parent:
                raise RevisionConflict("scratch revision changed; refresh before appending")
            row = {
                "schema": "nanlm.hermes-scratch-entry/1",
                "scratch_id": self.scratch_id,
                "entry_id": uuid.uuid4().hex,
                "sequence": len(rows) + 1,
                "lane": lane,
                "writer": writer,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parent_revision": parent,
                "payload": payload,
            }
            row["revision"] = digest(row)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(row).decode("utf-8") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return row

    def lane(self, lane: str) -> list[dict]:
        if lane not in LANES:
            raise ScratchError("unknown scratch lane")
        return [row for row in self._read() if row["lane"] == lane]
