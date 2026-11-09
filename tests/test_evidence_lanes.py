"""Independent support/refute lanes and four-state evidence status."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.evidence_lanes import (
    CONVENTIONAL_SCHEMA,
    RECORD_CYCLES,
    SCHEMA,
    ConventionalEvidence,
    EvidenceError,
    EvidenceGenerationError,
    EvidenceLanes,
    decode_record,
    encode_record,
)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory
from nanlmind_hermes.resident.universe import ResidentUniverse


def _memory():
    cells = ResidentCells.format_region(bytearray(REGION_BYTES))
    return FullCycleMemory(cells)


class EncodingTests(unittest.TestCase):
    def test_round_trip_and_size_limit(self):
        raw = encode_record("supports", "quokka", "is a", "animal")
        self.assertEqual(len(raw), RECORD_CYCLES)
        self.assertEqual(decode_record(raw),
                         ("supports", "quokka", "is a", "animal"))
        with self.assertRaises(EvidenceError):
            encode_record("supports", "x" * 80, "is a", "animal")
        with self.assertRaises(EvidenceError):
            encode_record("maybe", "a", "is a", "b")

    def test_empty_and_malformed_runs(self):
        self.assertIsNone(decode_record(bytes(RECORD_CYCLES)))
        bad = bytearray(encode_record("refutes", "a", "is a", "b"))
        bad[0] = 0
        self.assertIsNone(decode_record(bytes(bad)))
        with self.assertRaises(EvidenceError):
            decode_record(b"short")


class EvidenceLaneTests(unittest.TestCase):
    def setUp(self):
        self.memory = _memory()
        self.lanes = EvidenceLanes(self.memory)

    def _add(self, polarity, subject="quokka", relation="is a",
             destination="animal"):
        return self.lanes.add(subject, relation, destination, polarity)

    def test_four_state_statuses_without_cancellation(self):
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "unknown")
        support = self._add("supports")
        self.assertEqual(support.status, "appended")
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "supports")
        refute = self._add("refutes")
        state = self.lanes.state("quokka", "is a", "animal")
        self.assertEqual(state.status, "both")
        self.assertEqual(len(state.supports), 1)
        self.assertEqual(len(state.refutes), 1)
        self.assertEqual(self.lanes.delete("quokka", "is a", "animal",
                                           "supports").deleted, 1)
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "refutes")
        self.assertEqual(self.lanes.delete("quokka", "is a", "animal").deleted,
                         1)
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "unknown")

    def test_duplicate_is_idempotent(self):
        first = self._add("supports")
        second = self._add("supports")
        self.assertEqual(second.status, "duplicate")
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(self.lanes.records()), 1)

    def test_generation_cas_rejects_stale_writer(self):
        first = self._add("supports")
        with self.assertRaises(EvidenceGenerationError):
            self.lanes.add("quokka", "is a", "animal", "refutes",
                           expected_generation=first.generation - 1)
        second = self._add("refutes")
        self.assertEqual(self.lanes.delete(
            "quokka", "is a", "animal", "supports",
            expected_generation=second.generation).status, "deleted")
        with self.assertRaises(EvidenceGenerationError):
            self.lanes.add("quokka", "is a", "animal", "supports",
                           expected_generation=first.generation)

    def test_snapshot_delete_and_restore(self):
        self._add("supports")
        self._add("refutes")
        snapshot = self.lanes.snapshot()
        json.dumps(snapshot)
        self.assertEqual(len(snapshot["records"]), 2)
        deleted = self.lanes.delete("quokka", "is a", "animal")
        self.assertEqual(deleted.deleted, 2)
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "unknown")
        restored = self.lanes.restore(
            snapshot, expected_generation=deleted.generation)
        self.assertEqual(restored.restored, 2)
        self.assertEqual(self.lanes.state("quokka", "is a", "animal").status,
                         "both")

    def test_restore_into_fresh_memory(self):
        self._add("supports")
        snapshot = self.lanes.snapshot()
        memory = _memory()
        lanes = EvidenceLanes(memory)
        receipt = lanes.restore(snapshot, expected_generation=memory.generation)
        self.assertEqual(receipt.restored, 1)
        self.assertEqual(lanes.state("quokka", "is a", "animal").status,
                         "supports")

    def test_audit_and_full_dimension(self):
        self._add("supports")
        report = self.lanes.audit()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["dimension"], 32385)
        self.assertEqual(report["record_cycles"], RECORD_CYCLES)
        self.assertEqual(report["counts"]["supports"], 1)
        self.assertEqual(report["counts"]["refutes"], 0)
        self.assertFalse(report["canonical_merge"])
        self.assertEqual(self.memory.validate_root_edges(), 510)

    def test_conventional_control_ties(self):
        conventional = ConventionalEvidence()
        operations = [
            ("supports", "quokka", "is a", "animal"),
            ("refutes", "quokka", "is a", "animal"),
            ("supports", "quokka", "is a", "marsupial"),
        ]
        for polarity, subject, relation, destination in operations:
            self.lanes.add(subject, relation, destination, polarity)
            conventional.add(subject, relation, destination, polarity)
        for subject, relation, destination in (
                ("quokka", "is a", "animal"),
                ("quokka", "is a", "marsupial"),
                ("quokka", "is a", "plant")):
            self.assertEqual(
                self.lanes.state(subject, relation, destination).status,
                conventional.state(subject, relation, destination).status)
        self.assertEqual(
            [record.key() for record in self.lanes.records()],
            [record.key() for record in conventional.records()])
        self.assertEqual(self.lanes.snapshot()["schema"], SCHEMA)
        self.assertEqual(conventional.snapshot()["schema"],
                         CONVENTIONAL_SCHEMA)

    def test_snapshot_validation(self):
        with self.assertRaises(EvidenceError):
            self.lanes.restore({"schema": "wrong", "records": []})
        with self.assertRaises(EvidenceError):
            self.lanes.restore({"schema": SCHEMA, "records": [{"polarity": "x"}]})


class ResidentFacadeTests(unittest.TestCase):
    def test_universe_accessor_persists_lane_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.nlm"
            with ResidentUniverse.open(path, 32 * 1024 * 1024) as universe:
                lanes = universe.evidence_lanes()
                self.assertIs(lanes, universe.evidence_lanes())
                receipt = lanes.add(
                    "quokka", "is a", "animal", "supports")
                self.assertEqual(receipt.status, "appended")
                self.assertEqual(
                    lanes.state("quokka", "is a", "animal").status,
                    "supports")
            with ResidentUniverse.open(path, 32 * 1024 * 1024) as reopened:
                lanes = reopened.evidence_lanes()
                self.assertEqual(
                    lanes.state("quokka", "is a", "animal").status,
                    "supports")
                self.assertTrue(lanes.audit()["ok"])



if __name__ == "__main__":
    unittest.main()
