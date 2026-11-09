"""Independent support/refute evidence lanes over the full cycle basis.

Round 042's four-state answer surface is not a single signed score.  Evidence
that supports a statement and evidence that refutes it are retained as
separate records; a query then distinguishes ``unknown``, ``supports``,
``refutes`` and ``both`` without cancelling one lane against the other.

Each record occupies a short run of the 32,385 raw cycle coefficients.  The
bytes are written directly through :class:`FullCycleMemory`; the 255 root
edges remain zero.  Mutations accept an optional generation check so a stale
caller cannot silently write after another writer committed.

This is a symbolic evidence ledger, not learned propagation and not a neural
speaker.  :class:`ConventionalEvidence` is an indexed control with the same
four-state semantics; the geometric and conventional paths tie on clean
records by design.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

from .full_cycle_memory import FullCycleMemory, cycle_pair

SCHEMA = "nanlm.resident-evidence-lanes/1"
CONVENTIONAL_SCHEMA = "nanlm.resident-evidence-index/1"
MAGIC = 0xA5
POLARITY_SUPPORTS = 1
POLARITY_REFUTES = 2
POLARITIES = {
    "supports": POLARITY_SUPPORTS,
    "refutes": POLARITY_REFUTES,
}
POLARITY_NAMES = {value: key for key, value in POLARITIES.items()}
RECORD_CYCLES = 48
MAX_BODY_BYTES = RECORD_CYCLES - 2
EMPTY_RECORD = bytes(RECORD_CYCLES)
STATUSES = ("unknown", "supports", "refutes", "both")


class EvidenceError(ValueError):
    """Malformed lane, record or snapshot."""


class EvidenceGenerationError(EvidenceError):
    """The caller's expected generation is stale."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("%s must be a non-empty string" % name)
    if "\x00" in value or ">" in value:
        raise EvidenceError("%s must not contain NUL or '>'" % name)
    return value


def _polarity(value: Any) -> str:
    if value not in POLARITIES:
        raise EvidenceError("polarity must be 'supports' or 'refutes'")
    return value


def _record_digest(polarity: str, subject: str, relation: str,
                   destination: str) -> str:
    body = ("%s>%s>%s" % (subject, relation, destination)).encode("utf-8")
    return hashlib.sha256(bytes([POLARITIES[polarity]]) + body).hexdigest()


def encode_record(polarity: str, subject: str, relation: str,
                  destination: str) -> bytes:
    """Encode one padded record, or raise if it exceeds its cycle run."""
    lane = _polarity(polarity)
    body = ("%s>%s>%s" % (
        _text(subject, "subject"), _text(relation, "relation"),
        _text(destination, "destination"))).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise EvidenceError(
            "record body is %d bytes; maximum is %d"
            % (len(body), MAX_BODY_BYTES))
    raw = bytes([MAGIC, POLARITIES[lane]]) + body
    return raw.ljust(RECORD_CYCLES, b"\x00")


def decode_record(raw: bytes) -> tuple[str, str, str, str] | None:
    """Decode a byte run, returning ``(polarity, subject, relation, dest)``."""
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != RECORD_CYCLES:
        raise EvidenceError("record must be exactly %d bytes" % RECORD_CYCLES)
    raw = bytes(raw)
    if raw == EMPTY_RECORD or raw[0] != MAGIC:
        return None
    lane = POLARITY_NAMES.get(raw[1])
    if lane is None:
        return None
    body = raw[2:].rstrip(b"\x00")
    if not body:
        return None
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    parts = text.split(">")
    if len(parts) != 3 or not all(parts):
        return None
    return lane, parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class EvidenceRecord:
    polarity: str
    subject: str
    relation: str
    destination: str
    slot: int | None = None

    def __post_init__(self):
        _polarity(self.polarity)
        _text(self.subject, "subject")
        _text(self.relation, "relation")
        _text(self.destination, "destination")
        if self.slot is not None and (
                isinstance(self.slot, bool) or not isinstance(self.slot, int)
                or self.slot < 0):
            raise EvidenceError("slot must be a non-negative integer or None")

    @property
    def digest(self) -> str:
        return _record_digest(self.polarity, self.subject, self.relation,
                              self.destination)

    def as_dict(self) -> dict[str, Any]:
        return {
            "polarity": self.polarity,
            "subject": self.subject,
            "relation": self.relation,
            "destination": self.destination,
            "slot": self.slot,
            "digest": self.digest,
        }

    def key(self) -> tuple[str, str, str, str]:
        return (self.polarity, self.subject, self.relation, self.destination)


@dataclass(frozen=True)
class EvidenceState:
    subject: str
    relation: str
    destination: str
    status: str
    supports: tuple[EvidenceRecord, ...]
    refutes: tuple[EvidenceRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "subject": self.subject,
            "relation": self.relation,
            "destination": self.destination,
            "status": self.status,
            "supports": [record.as_dict() for record in self.supports],
            "refutes": [record.as_dict() for record in self.refutes],
            "canonical_merge": False,
        }


@dataclass(frozen=True)
class EvidenceReceipt:
    action: str
    status: str
    generation: int
    expected_generation: int | None
    record: EvidenceRecord | None = None
    deleted: int = 0
    restored: int = 0
    digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "action": self.action,
            "status": self.status,
            "generation": self.generation,
            "expected_generation": self.expected_generation,
            "record": None if self.record is None else self.record.as_dict(),
            "deleted": self.deleted,
            "restored": self.restored,
            "digest": self.digest,
            "canonical_merge": False,
        }


class _LaneBase:
    schema = SCHEMA

    def _check_expected(self, expected: int | None) -> int:
        current = self.generation
        if expected is not None:
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise EvidenceError("expected_generation must be an integer")
            if expected != current:
                raise EvidenceGenerationError(
                    "expected generation %d, current generation is %d"
                    % (expected, current))
        return current

    def add(self, subject: str, relation: str, destination: str,
            polarity: str = "supports", *,
            expected_generation: int | None = None) -> EvidenceReceipt:
        record = EvidenceRecord(polarity, subject, relation, destination)
        expected = self._check_expected(expected_generation)
        if self.contains(record):
            return EvidenceReceipt(
                "add", "duplicate", self.generation, expected, record,
                digest=record.digest)
        slot, generation = self._insert(record)
        record = EvidenceRecord(record.polarity, record.subject,
                                record.relation, record.destination, slot)
        return EvidenceReceipt(
            "add", "appended", generation, expected, record,
            digest=record.digest)

    def delete(self, subject: str, relation: str, destination: str,
               polarity: str | None = None, *,
               expected_generation: int | None = None) -> EvidenceReceipt:
        subject = _text(subject, "subject")
        relation = _text(relation, "relation")
        destination = _text(destination, "destination")
        if polarity is not None:
            polarity = _polarity(polarity)
        expected = self._check_expected(expected_generation)
        removed = self._remove(subject, relation, destination, polarity)
        if not removed:
            return EvidenceReceipt(
                "delete", "missing", self.generation, expected)
        digest = hashlib.sha256(
            "".join(record.digest for record in removed).encode("ascii")
        ).hexdigest()
        return EvidenceReceipt(
            "delete", "deleted", self.generation, expected,
            deleted=len(removed), digest=digest)

    def state(self, subject: str, relation: str,
              destination: str) -> EvidenceState:
        subject = _text(subject, "subject")
        relation = _text(relation, "relation")
        destination = _text(destination, "destination")
        supports = tuple(
            record for record in self.records()
            if record.polarity == "supports"
            and record.subject == subject
            and record.relation == relation
            and record.destination == destination)
        refutes = tuple(
            record for record in self.records()
            if record.polarity == "refutes"
            and record.subject == subject
            and record.relation == relation
            and record.destination == destination)
        if supports and refutes:
            status = "both"
        elif supports:
            status = "supports"
        elif refutes:
            status = "refutes"
        else:
            status = "unknown"
        return EvidenceState(subject, relation, destination, status,
                             supports, refutes)

    query = state

    def restore(self, snapshot: Mapping[str, Any] | Iterable[Mapping[str, Any]],
                *, expected_generation: int | None = None) -> EvidenceReceipt:
        rows = self._snapshot_rows(snapshot)
        expected = self._check_expected(expected_generation)
        restored = 0
        digest = hashlib.sha256()
        for row in rows:
            record = EvidenceRecord(
                row["polarity"], row["subject"], row["relation"],
                row["destination"])
            digest.update(record.digest.encode("ascii"))
            if self.contains(record):
                continue
            self._insert(record)
            restored += 1
        return EvidenceReceipt(
            "restore", "restored" if restored else "duplicate",
            self.generation, expected, restored=restored,
            digest=digest.hexdigest())

    @staticmethod
    def _snapshot_rows(snapshot):
        if isinstance(snapshot, Mapping):
            if snapshot.get("schema") not in (SCHEMA, CONVENTIONAL_SCHEMA):
                raise EvidenceError("snapshot schema is not an evidence schema")
            rows = snapshot.get("records")
        else:
            rows = snapshot
        if not isinstance(rows, (list, tuple)):
            raise EvidenceError("snapshot records must be a list")
        clean = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise EvidenceError("snapshot record %d is not an object" % index)
            try:
                clean.append({
                    "polarity": _polarity(row["polarity"]),
                    "subject": _text(row["subject"], "subject"),
                    "relation": _text(row["relation"], "relation"),
                    "destination": _text(row["destination"], "destination"),
                })
            except KeyError as error:
                raise EvidenceError(
                    "snapshot record %d is missing %s"
                    % (index, error.args[0])) from error
        return clean


class EvidenceLanes(_LaneBase):
    """Support/refute records stored in raw full-cycle coefficients."""

    def __init__(self, memory: FullCycleMemory, *,
                 record_cycles: int = RECORD_CYCLES):
        if not isinstance(memory, FullCycleMemory):
            raise EvidenceError("memory must be a FullCycleMemory")
        if record_cycles != RECORD_CYCLES:
            raise EvidenceError(
                "record_cycles is fixed at %d for this schema" % RECORD_CYCLES)
        self.memory = memory
        self.record_cycles = RECORD_CYCLES
        self.max_records = len(memory) // RECORD_CYCLES

    @property
    def generation(self) -> int:
        return int(self.memory.generation)

    def _indices(self, slot: int) -> range:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise EvidenceError("slot must be an integer")
        if not 0 <= slot < self.max_records:
            raise EvidenceError("slot must be in [0, %d): %r"
                                % (self.max_records, slot))
        start = slot * RECORD_CYCLES
        return range(start, start + RECORD_CYCLES)

    def _read_raw(self, slot: int) -> bytes:
        raw = bytearray()
        for index in self._indices(slot):
            word = self.memory[index]
            if word & ~0xFF:
                return EMPTY_RECORD
            raw.append(word)
        return bytes(raw)

    def _write_raw(self, slot: int, raw: bytes) -> None:
        if not isinstance(raw, (bytes, bytearray)) or len(raw) != RECORD_CYCLES:
            raise EvidenceError("record must be exactly %d bytes"
                                % RECORD_CYCLES)
        if any(value > 0xFF for value in raw):
            raise EvidenceError("record byte is outside 0..255")
        for index, value in zip(self._indices(slot), raw):
            first, second = cycle_pair(index)
            self.memory.set_word(first, second, value)

    def _slot_record(self, slot: int) -> EvidenceRecord | None:
        decoded = decode_record(self._read_raw(slot))
        if decoded is None:
            return None
        return EvidenceRecord(*decoded, slot=slot)

    def records(self) -> tuple[EvidenceRecord, ...]:
        result = []
        for slot in range(self.max_records):
            record = self._slot_record(slot)
            if record is not None:
                result.append(record)
        return tuple(result)

    def contains(self, record: EvidenceRecord) -> bool:
        key = record.key()
        return any(item.key() == key for item in self.records())

    def _free_slot(self) -> int:
        for slot in range(self.max_records):
            if self._read_raw(slot) == EMPTY_RECORD:
                return slot
        raise EvidenceError("evidence lanes are full (%d records)"
                            % self.max_records)

    def _insert(self, record: EvidenceRecord) -> tuple[int, int]:
        slot = self._free_slot()
        self._write_raw(slot, encode_record(
            record.polarity, record.subject, record.relation,
            record.destination))
        return slot, self.generation

    def _remove(self, subject: str, relation: str, destination: str,
                polarity: str | None) -> tuple[EvidenceRecord, ...]:
        removed = []
        for slot in range(self.max_records):
            record = self._slot_record(slot)
            if record is None:
                continue
            if (record.subject == subject and record.relation == relation
                    and record.destination == destination
                    and (polarity is None or record.polarity == polarity)):
                removed.append(record)
                self._write_raw(slot, EMPTY_RECORD)
        return tuple(removed)

    def snapshot(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema": SCHEMA,
            "dimension": len(self.memory),
            "record_cycles": RECORD_CYCLES,
            "max_records": self.max_records,
            "records": [record.as_dict() for record in records],
            "generation": self.generation,
            "canonical_merge": False,
        }

    def audit(self) -> dict[str, Any]:
        problems = []
        records = []
        for slot in range(self.max_records):
            try:
                record = self._slot_record(slot)
            except EvidenceError as error:
                problems.append("slot %d: %s" % (slot, error))
                continue
            if record is not None:
                records.append(record)
        if len(self.memory) != 32385:
            problems.append("full basis is %d, expected 32385"
                            % len(self.memory))
        try:
            self.memory.validate_root_edges()
        except Exception as error:  # noqa: BLE001 - backend boundary
            problems.append("root edges: %s" % error)
        counts = {state: 0 for state in ("supports", "refutes")}
        for record in records:
            counts[record.polarity] += 1
        return {
            "schema": SCHEMA,
            "ok": not problems,
            "problems": problems,
            "dimension": len(self.memory),
            "record_cycles": RECORD_CYCLES,
            "max_records": self.max_records,
            "records": len(records),
            "counts": counts,
            "generation": self.generation,
            "canonical_merge": False,
        }


class ConventionalEvidence(_LaneBase):
    """Indexed control with the same four-state semantics, no raw geometry."""

    def __init__(self):
        self._records: list[EvidenceRecord] = []
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def contains(self, record: EvidenceRecord) -> bool:
        key = record.key()
        return any(item.key() == key for item in self._records)

    def _insert(self, record: EvidenceRecord) -> tuple[int, int]:
        slot = len(self._records)
        self._records.append(EvidenceRecord(
            record.polarity, record.subject, record.relation,
            record.destination, slot=slot))
        self._generation += 1
        return slot, self._generation

    def _remove(self, subject: str, relation: str, destination: str,
                polarity: str | None) -> tuple[EvidenceRecord, ...]:
        keep = []
        removed = []
        for record in self._records:
            if (record.subject == subject and record.relation == relation
                    and record.destination == destination
                    and (polarity is None or record.polarity == polarity)):
                removed.append(record)
            else:
                keep.append(record)
        if removed:
            self._records = [
                EvidenceRecord(item.polarity, item.subject, item.relation,
                               item.destination, slot=slot)
                for slot, item in enumerate(keep)]
            self._generation += 1
        return tuple(removed)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": CONVENTIONAL_SCHEMA,
            "records": [record.as_dict() for record in self._records],
            "generation": self._generation,
            "canonical_merge": False,
        }


__all__ = (
    "SCHEMA", "CONVENTIONAL_SCHEMA", "RECORD_CYCLES", "MAX_BODY_BYTES",
    "POLARITIES", "STATUSES", "EvidenceError", "EvidenceGenerationError",
    "EvidenceRecord", "EvidenceState", "EvidenceReceipt", "EvidenceLanes",
    "ConventionalEvidence", "encode_record", "decode_record",
)
