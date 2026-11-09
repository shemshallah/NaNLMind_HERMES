"""Tests for bounded raw-cycle persistence over the resident universe.

The suite exercises a real 16 MiB ``ResidentUniverse`` when its sibling
modules are available and a small duck-typed facade for the fallback cases.
No existing resident file is modified by these tests.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cycle_persistence import (
    ANCHOR_SCHEMA,
    CYCLE_SCHEMA,
    SLOT_SCHEMA,
    WORD_MASK,
    CyclePersistence,
    CyclePersistenceBoundsError,
    CyclePersistenceCorruptionError,
    CyclePersistenceError,
    CyclePersistencePathError,
    CyclePersistencePointerError,
    CyclePersistenceSecretError,
    CycleReceipt,
    CycleRecord,
    _anchor_digest,
    _anchor_id,
    _build_slot,
    _canonical_text,
    _mirror_word,
    _payload_id,
    _slot_id,
    _word_hex,
)

try:  # The real facade is the primary target; fakes cover partial installs.
    from nanlmind_hermes.resident.universe import ResidentUniverse
except Exception:  # pragma: no cover - only on a partial installation
    ResidentUniverse = None  # type: ignore[assignment]

MIB = 1024 * 1024
SIZE_16MIB = 16 * MIB
RAW = 0xDEADBEEF
SOURCE = "round029:cycle-test"


def _walk_no_float(testcase, value):
    if isinstance(value, float):
        testcase.fail("stored cycle JSON contains a float: %r" % value)
    if isinstance(value, dict):
        for key, item in value.items():
            testcase.assertIsInstance(key, str)
            if key.casefold() in {
                    "pointer", "ptr", "address", "addr", "offset",
                    "raw_pointer", "memory_address"}:
                testcase.fail("stored cycle JSON contains a raw pointer key")
            _walk_no_float(testcase, item)
    elif isinstance(value, list):
        for item in value:
            _walk_no_float(testcase, item)


class FakeUniverse:
    """Minimal duck-typed ResidentUniverse facade used by fallback tests."""

    def __init__(self):
        self.leaves: dict[str, dict] = {}
        self.generation = 0
        self.digest = hashlib.sha256(b"fake-cycle-genesis").hexdigest()
        self.checkpoint_calls = 0

    def ingest(self, record):
        record = dict(record)
        leaf_id = record.get("leaf_id")
        if not isinstance(leaf_id, str) or not leaf_id:
            leaf_id = "leaf:" + hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()).hexdigest()[:32]
        if leaf_id in self.leaves:
            return {
                "status": "duplicate", "created": False, "duplicate": True,
                "leaf_id": leaf_id, "leaf_ids": [leaf_id],
                "generation": self.generation, "digest": self.digest,
            }
        text = record.get("text")
        if not isinstance(text, str):
            text = json.dumps(
                record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False)
        self.generation += 1
        self.digest = hashlib.sha256(
            (self.digest + leaf_id + text).encode("utf-8")).hexdigest()
        self.leaves[leaf_id] = {
            "leaf_id": leaf_id,
            "prepared_id": record.get("prepared_id") or leaf_id,
            "kind": record.get("kind"),
            "head": record.get("head"),
            "text": text,
            "source_digest": record.get("source_digest"),
        }
        return {
            "status": "appended", "created": True, "duplicate": False,
            "leaf_id": leaf_id, "leaf_ids": [leaf_id],
            "generation": self.generation, "digest": self.digest,
        }

    def get_leaf(self, leaf_id):
        leaf = self.leaves.get(str(leaf_id))
        return dict(leaf) if leaf is not None else None

    def snapshot(self):
        return {
            "schema": "nanlm.resident-universe/1",
            "generation": self.generation,
            "digest": self.digest,
            "state_digest": self.digest,
            "counts": {"leaves": len(self.leaves), "relations": 0},
        }

    def checkpoint(self, directory=None):
        self.checkpoint_calls += 1
        return {
            "schema": "nanlm.resident-universe-checkpoint/1",
            "directory": str(directory) if directory is not None else None,
            "generation": self.generation,
            "digest": self.digest,
            "canonical_commit": False,
        }


class RealUniverseTestCase(unittest.TestCase):
    def setUp(self):
        if ResidentUniverse is None:
            self.skipTest("resident universe siblings are unavailable")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "universe.nlm"
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)

    def tearDown(self):
        if getattr(self, "universe", None) is not None:
            self.universe.close()
        self._tmp.cleanup()

    def reopen(self):
        self.universe.close()
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)
        return self.universe


class MainCycleTests(RealUniverseTestCase):
    def test_round_trip_requires_approval_and_returns_generation_receipt(self):
        persistence = CyclePersistence(self.universe)
        self.assertIsNone(persistence.load_cycle(0, 1))

        before = self.universe.snapshot()
        denial = persistence.save_cycle(0, 1, RAW, source=SOURCE)
        self.assertIsInstance(denial, CycleReceipt)
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "approval_required")
        self.assertTrue(denial["reasons"])
        self.assertEqual(self.universe.snapshot()["generation"],
                         before["generation"])
        self.assertIsNone(persistence.load_cycle(0, 1))

        receipt = persistence.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True,
            metadata={"round": 29, "lane": "forward"})
        self.assertTrue(receipt)
        self.assertEqual(receipt["status"], "committed")
        self.assertTrue(receipt["created"])
        self.assertTrue(receipt["approved"])
        self.assertEqual(receipt["kind"], "cycle")
        self.assertEqual(receipt["pair"], [0, 1])
        self.assertEqual(receipt["word"], RAW)
        self.assertEqual(receipt["word_hex"], _word_hex(RAW))
        self.assertEqual(receipt["content_digest"],
                         receipt["payload_digest"])
        self.assertEqual(len(receipt["content_digest"]), 64)
        self.assertEqual(len(receipt["anchor_digest"]), 64)
        self.assertEqual(len(receipt["slot_digest"]), 64)
        self.assertTrue(receipt["leaf_id"].startswith("cycle-payload:"))
        self.assertTrue(receipt["anchor_leaf_id"].startswith("cycle-anchor:"))
        self.assertEqual(receipt["slot_leaf_id"], "cycle-slot:0:1")
        self.assertTrue(receipt["main_cycle_modified"])
        self.assertFalse(receipt["canonical_commit"])
        self.assertFalse(receipt["canonical_state_modified"])

        after = self.universe.snapshot()
        self.assertEqual(receipt["generation"], after["generation"])
        self.assertEqual(receipt["digest"], after["digest"])
        self.assertEqual(receipt["state_digest"], after["digest"])
        self.assertGreater(receipt["generation_before"], before["generation"]
                           - 1)
        self.assertGreater(receipt["generation"], before["generation"])

        leaf = self.universe.get_leaf(receipt["leaf_id"])
        self.assertIsNotNone(leaf)
        payload = json.loads(leaf["text"])
        self.assertEqual(payload["schema"], CYCLE_SCHEMA)
        self.assertEqual(payload["word"], RAW)
        self.assertEqual(payload["raw_word"], RAW)
        self.assertEqual(payload["mirror"], (-RAW) & WORD_MASK)
        self.assertEqual(payload["mirror_word"], -RAW)
        self.assertEqual(payload["metadata"], {"round": 29, "lane": "forward"})
        _walk_no_float(self, payload)

        loaded = persistence.load_cycle(0, 1)
        self.assertIsInstance(loaded, CycleRecord)
        self.assertEqual(loaded, RAW)
        self.assertEqual(int(loaded), RAW)
        self.assertEqual(loaded["raw_word"], RAW)
        self.assertEqual(loaded["word"], RAW)
        self.assertEqual(loaded["source"], SOURCE)
        mirror = persistence.load_cycle(1, 0)
        self.assertEqual(mirror, (-RAW) & WORD_MASK)
        self.assertEqual(mirror["raw_word"], -RAW)
        self.assertEqual(mirror["word"], (-RAW) & WORD_MASK)
        self.assertEqual(mirror["mirror_word"], RAW)
        self.assertEqual(mirror["direction"], "mirror")

        duplicate = persistence.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True)
        self.assertTrue(duplicate)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["generation"], receipt["generation"])
        self.assertEqual(duplicate["content_digest"],
                         receipt["content_digest"])
        self.assertEqual(duplicate["anchor_leaf_id"],
                         receipt["anchor_leaf_id"])

        reverse_duplicate = persistence.save_cycle(
            1, 0, -RAW, source=SOURCE, approved=True)
        self.assertEqual(reverse_duplicate["status"], "duplicate")
        self.assertEqual(self.universe.snapshot()["generation"],
                         receipt["generation"])

        changed = persistence.save_cycle(
            0, 1, RAW + 1, source=SOURCE, approved=True)
        self.assertFalse(changed)
        self.assertEqual(changed["status"], "read_only")
        self.assertEqual(changed["existing_word"], RAW)

        mismatch = persistence.save_cycle(
            1, 0, RAW + 1, source=SOURCE, approved=True)
        self.assertFalse(mismatch)
        self.assertEqual(mismatch["status"], "mirror_mismatch")
        self.assertEqual(mismatch["expected_mirror"], -RAW)
        self.assertEqual(self.universe.snapshot()["generation"],
                         receipt["generation"])

        report = persistence.audit()
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["cycles"], 1)
        self.assertEqual(report["mirror_checked"], 1)
        self.assertEqual(report["generation"], after["generation"])

    def test_batch_scratch_snapshot_and_reopen(self):
        persistence = CyclePersistence(self.universe)
        batch = persistence.save_cycle_batch(
            [(0, 2, 6), (3, 4, 7)], source=SOURCE, approved=True)
        self.assertTrue(batch)
        self.assertEqual(batch["status"], "committed")
        self.assertEqual(batch["created"], True)
        self.assertEqual(batch["committed"], 2)
        self.assertEqual(batch["duplicates"], 0)
        self.assertEqual(batch["cycle_count"], 2)
        self.assertEqual(len(batch["items"]), 2)
        self.assertEqual(persistence.load_cycle(0, 2), 6)
        self.assertEqual(persistence.load_cycle(4, 3), (-7) & WORD_MASK)

        duplicate = persistence.save_cycle_batch(
            [(0, 2, 6), (3, 4, 7)], source=SOURCE, approved=True)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertFalse(duplicate["created"])

        # A conflicting batch is preflighted atomically: the new pair is not
        # written when the reverse item carries a mismatched mirror.
        conflict = persistence.save_cycle_batch(
            [(5, 6, 9), (6, 5, 9)], source=SOURCE, approved=True)
        self.assertFalse(conflict)
        self.assertEqual(conflict["status"], "mirror_mismatch")
        self.assertIsNone(persistence.load_cycle(5, 6))

        main_count = len(persistence.audit()["cycle_records"])
        scratch = persistence.save_cycle_batch(
            [(10, 11, 123), (12, 13, 456)], source="scratch-source")
        self.assertTrue(scratch)
        self.assertEqual(scratch["status"], "committed")
        self.assertEqual(scratch["kind"], "scratch")
        self.assertEqual(scratch["version"], 0)
        self.assertEqual(scratch["cycle_count"], 2)
        self.assertFalse(scratch["main_modified"])
        self.assertIsNone(persistence.load_cycle(10, 11))
        self.assertIsNone(persistence.load_cycle(12, 13))
        scratch_payload = persistence.load_scratch_batch()
        self.assertEqual(scratch_payload["cycle_count"], 2)
        self.assertEqual(
            len(persistence.audit()["cycle_records"]), main_count)

        scratch_duplicate = persistence.save_cycle_batch(
            [(10, 11, 123), (12, 13, 456)], source="scratch-source")
        self.assertEqual(scratch_duplicate["status"], "duplicate")
        self.assertEqual(scratch_duplicate["version"], 0)

        snapshot = persistence.snapshot(source="snapshot-source")
        self.assertTrue(snapshot)
        self.assertEqual(snapshot["status"], "committed")
        self.assertEqual(snapshot["kind"], "snapshot")
        self.assertEqual(snapshot["version"], 0)
        self.assertEqual(snapshot["cycle_count"], 2)
        self.assertFalse(snapshot["main_modified"])
        self.assertEqual(len(snapshot["cycles"]), 2)
        main_generation = max(
            persistence.load_cycle(0, 2)["commit_generation"],
            persistence.load_cycle(3, 4)["commit_generation"])
        self.assertEqual(snapshot["main_generation"], main_generation)
        self.assertEqual(snapshot["snapshot_digest"],
                         snapshot["content_digest"])
        loaded_snapshot = persistence.load_snapshot()
        self.assertEqual(loaded_snapshot["cycle_count"], 2)
        self.assertEqual(loaded_snapshot["schema"],
                         "nanlm.resident-cycle-snapshot/1")

        snapshot_duplicate = persistence.snapshot(source="snapshot-source")
        self.assertEqual(snapshot_duplicate["status"], "duplicate")
        self.assertEqual(snapshot_duplicate["version"], 0)

        report = persistence.audit()
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["cycles"], 2)
        self.assertEqual(report["scratch_checked"], 1)
        self.assertEqual(report["snapshot_checked"], 1)
        self.assertEqual(report["checked"], 4)

        receipt_generation = snapshot["generation"]
        snapshot_digest = snapshot["snapshot_digest"]
        snapshot_main_digest = snapshot["main_digest"]
        self.universe.close()
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)
        reopened = CyclePersistence(self.universe)
        self.assertEqual(reopened.load_cycle(0, 2), 6)
        self.assertEqual(reopened.load_cycle(3, 4), 7)
        self.assertEqual(reopened.load_scratch_batch()["cycle_count"], 2)
        self.assertEqual(reopened.load_snapshot()["cycle_count"], 2)
        self.assertEqual(reopened.load_snapshot()["main_digest"],
                         snapshot_main_digest)
        self.assertEqual(
            reopened.snapshot(source="snapshot-source")["status"], "duplicate")
        self.assertEqual(
            reopened.snapshot(source="snapshot-source")["snapshot_digest"],
            snapshot_digest)
        reopened_report = reopened.audit()
        self.assertTrue(reopened_report["ok"], reopened_report["problems"])
        self.assertEqual(reopened_report["cycles"], 2)
        self.assertEqual(reopened_report["scratch_checked"], 1)
        self.assertEqual(reopened_report["snapshot_checked"], 1)
        self.assertEqual(reopened_report["generation"], receipt_generation)

        checkpoint = reopened.checkpoint(self.root / "checkpoint")
        self.assertEqual(checkpoint["generation"],
                         self.universe.snapshot()["generation"])
        self.assertEqual(len(checkpoint["digest"]), 64)


class DenialTests(RealUniverseTestCase):
    def test_bounds_secret_path_pointer_and_pair_validation(self):
        persistence = CyclePersistence(self.universe)
        before = self.universe.snapshot()

        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, 1 << 64, source=SOURCE, approved=True)
        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, -(1 << 63) - 1, source=SOURCE, approved=True)
        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, 1.25, source=SOURCE, approved=True)
        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, True, source=SOURCE, approved=True)
        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, RAW, source="x" * 4096, approved=True)

        with self.assertRaises(CyclePersistenceSecretError):
            persistence.save_cycle(
                0, 1, RAW, source="sk-" + "a" * 32, approved=True)
        with self.assertRaises(CyclePersistenceSecretError):
            persistence.save_cycle(
                0, 1, RAW, source=SOURCE, approved=True,
                metadata={"note": "-----BEGIN " "PRIVATE KEY-----"})

        with self.assertRaises(CyclePersistencePathError):
            persistence.save_cycle(
                0, 1, RAW, source="/etc/passwd", approved=True)
        with self.assertRaises(CyclePersistencePathError):
            persistence.save_cycle(
                0, 1, RAW, source="notes/private.txt", approved=True)
        with self.assertRaises(CyclePersistencePathError):
            persistence.save_cycle(
                0, 1, RAW, source="../../escape", approved=True)
        with self.assertRaises(CyclePersistencePathError):
            persistence.save_cycle(
                0, 1, RAW, source=SOURCE, approved=True,
                metadata={"path": "notes.txt"})

        with self.assertRaises(CyclePersistencePointerError):
            persistence.save_cycle(
                0, 1, RAW, source=SOURCE, approved=True,
                metadata={"pointer": 0x1234})
        with self.assertRaises(CyclePersistenceBoundsError):
            persistence.save_cycle(
                0, 1, RAW, source=SOURCE, approved=True,
                metadata={"confidence": 0.5})

        with self.assertRaises(CyclePersistenceError):
            persistence.save_cycle(0, 0, RAW, source=SOURCE, approved=True)
        with self.assertRaises(CyclePersistenceError):
            persistence.save_cycle(0, 256, RAW, source=SOURCE, approved=True)
        mirror_denial = persistence.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True, mirror=RAW)
        self.assertFalse(mirror_denial)
        self.assertEqual(mirror_denial["status"], "mirror_mismatch")

        after = self.universe.snapshot()
        self.assertEqual(after["generation"], before["generation"])
        self.assertEqual(after["counts"], before["counts"])
        self.assertIsNone(persistence.load_cycle(0, 1))

    def test_missing_records_and_dangling_slot(self):
        persistence = CyclePersistence(self.universe)
        self.assertIsNone(persistence.load_cycle(0, 1))

        slot_text, _slot = _build_slot(
            (0, 1), _anchor_id("0" * 64), "0" * 64, "0" * 64, 0)
        self.universe.ingest({
            "type": "leaf",
            "schema": "nanlm.resident-universe/1",
            "leaf_id": _slot_id((0, 1)),
            "kind": "data",
            "head": 0,
            "text": slot_text,
            "features": [],
        })
        with self.assertRaises(CyclePersistenceCorruptionError):
            persistence.load_cycle(0, 1)
        report = persistence.audit()
        self.assertFalse(report["ok"])
        self.assertTrue(report["problems"])


class FakeFacadeTests(unittest.TestCase):
    def test_fake_round_trip_reopen_and_audit(self):
        universe = FakeUniverse()
        persistence = CyclePersistence(universe)
        self.assertIsNone(persistence.load_cycle(0, 1))

        denial = persistence.save_cycle(0, 1, RAW, source=SOURCE)
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "approval_required")
        self.assertEqual(universe.generation, 0)

        receipt = persistence.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True,
            metadata={"iteration": 7})
        self.assertTrue(receipt)
        self.assertEqual(receipt["generation"], universe.generation)
        self.assertEqual(receipt["digest"], universe.digest)
        self.assertEqual(persistence.load_cycle(0, 1), RAW)
        self.assertEqual(persistence.load_cycle(1, 0), (-RAW) & WORD_MASK)

        duplicate = persistence.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(universe.generation, receipt["generation"])

        scratch = persistence.save_cycle_batch(
            [(0, 2, 9)], source=SOURCE)
        self.assertEqual(scratch["kind"], "scratch")
        self.assertEqual(scratch["generation"], universe.generation)
        self.assertIsNone(persistence.load_cycle(0, 2))
        snapshot = persistence.snapshot()
        self.assertEqual(snapshot["cycle_count"], 1)
        self.assertFalse(snapshot["main_modified"])

        reopened = CyclePersistence(universe)
        self.assertEqual(reopened.load_cycle(0, 1), RAW)
        self.assertEqual(reopened.load_cycle(1, 0)["raw_word"], -RAW)
        self.assertEqual(reopened.load_scratch_batch()["cycle_count"], 1)
        self.assertEqual(reopened.load_snapshot()["cycle_count"], 1)
        self.assertEqual(reopened.save_cycle(
            0, 1, RAW, source=SOURCE, approved=True)["status"], "duplicate")
        report = reopened.audit()
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["cycles"], 1)
        self.assertEqual(report["scratch_checked"], 1)
        self.assertEqual(report["snapshot_checked"], 1)
        self.assertEqual(reopened.checkpoint("checkpoints")["generation"],
                         universe.generation)

    def test_fake_audit_detects_forged_mirror(self):
        universe = FakeUniverse()
        persistence = CyclePersistence(universe)
        pair = (10, 11)
        word = 77
        payload_text, _payload = _canonical_text(
            {
                "schema": CYCLE_SCHEMA,
                "kind": "cycle",
                "i": pair[0],
                "j": pair[1],
                "pair": list(pair),
                "word": word,
                "word_hex": _word_hex(word),
                "raw_word": word,
                # Deliberately wrong: the mirror must be -word.
                "mirror": word + 1,
                "mirror_hex": _word_hex(word + 1),
                "mirror_word": word + 1,
                "source": SOURCE,
                "source_digest": hashlib.sha256(
                    SOURCE.encode("utf-8")).hexdigest(),
                "metadata": {},
                "read_only": True,
                "no_float": True,
                "no_raw_pointers": True,
                "word_bits": 64,
            },
            label="forged cycle payload", max_bytes=32 * 1024,
        )
        digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        payload_leaf_id = _payload_id(digest)
        universe.ingest({
            "type": "leaf", "schema": "nanlm.resident-universe/1",
            "leaf_id": payload_leaf_id, "kind": "data", "head": pair[0],
            "text": payload_text, "features": [],
        })
        anchor = {
            "schema": ANCHOR_SCHEMA,
            "kind": "cycle",
            "i": pair[0],
            "j": pair[1],
            "pair": list(pair),
            "payload_leaf_id": payload_leaf_id,
            "data_leaf_id": payload_leaf_id,
            "content_digest": digest,
            "payload_digest": digest,
            "word": word,
            "word_hex": _word_hex(word),
            "mirror": word + 1,
            "mirror_hex": _word_hex(word + 1),
            "source": SOURCE,
            "source_digest": hashlib.sha256(
                SOURCE.encode("utf-8")).hexdigest(),
            "data_generation": 1,
            "commit_generation": 1,
            "read_only": True,
            "word_bits": 64,
        }
        anchor_digest = _anchor_digest(anchor)
        anchor["anchor_digest"] = anchor_digest
        anchor_text, _clean = _canonical_text(
            anchor, label="forged cycle anchor", max_bytes=8 * 1024)
        anchor_leaf_id = _anchor_id(anchor_digest)
        universe.ingest({
            "type": "leaf", "schema": "nanlm.resident-universe/1",
            "leaf_id": anchor_leaf_id, "kind": "data", "head": pair[0],
            "text": anchor_text, "features": [],
        })
        slot_text, _slot = _build_slot(
            pair, anchor_leaf_id, anchor_digest, digest, 2)
        universe.ingest({
            "type": "leaf", "schema": "nanlm.resident-universe/1",
            "leaf_id": _slot_id(pair), "kind": "data", "head": pair[0],
            "text": slot_text, "features": [],
        })

        with self.assertRaises(CyclePersistenceError):
            persistence.load_cycle(*pair)
        report = persistence.audit()
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(len(report["problems"]), 1)
        self.assertIn("mirror_mismatch", report["problem_codes"])
        self.assertTrue(report["no_floats"])
        self.assertTrue(report["no_raw_pointers"])

    def test_constructor_requires_the_minimal_facade(self):
        with self.assertRaises(TypeError):
            CyclePersistence(object())
        with self.assertRaises(TypeError):
            CyclePersistence({"ingest": lambda record: None})
        fake = FakeUniverse()
        fake.ingest = None
        with self.assertRaises(TypeError):
            CyclePersistence(fake)


if __name__ == "__main__":
    unittest.main()
