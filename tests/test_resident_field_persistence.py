"""Tests for bounded head-field and scratch-layer persistence.

The suite exercises the real 16 MiB resident universe when its sibling modules
are available and uses a small duck-typed facade for the fallback cases.  No
existing resident file is modified by these tests.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.field_persistence import (
    ANCHOR_SCHEMA,
    FIELD_ANCHOR_ID,
    FIELD_SCHEMA,
    MAX_FIELD_BYTES,
    MAX_JSON_DEPTH,
    MAX_SCRATCH_BYTES,
    FieldPersistence,
    FieldPersistenceBoundsError,
    FieldPersistenceError,
    FieldPersistencePathError,
    FieldPersistenceSecretError,
    FieldReceipt,
)

try:  # The real facade is the primary target; fakes cover partial installs.
    from nanlmind_hermes.resident.universe import ResidentUniverse
except Exception:  # pragma: no cover - only on a partial installation
    ResidentUniverse = None  # type: ignore[assignment]

MIB = 1024 * 1024
SIZE_16MIB = 16 * MIB


def field_snapshot(**overrides):
    snapshot = {
        "schema": "nanlm.hermes-head-field/1",
        "field_id": "phone-head-field",
        "heads": 256,
        "layers": {
            "0": {"energy": 0.25, "active": 12},
            "1": {"energy": 0.5, "active": 7},
        },
        "routes": [
            {"head": 0, "layer": 0, "weight": 1},
            {"head": 17, "layer": 2, "weight": 0.75},
        ],
        "note": "bounded local-derived metadata",
    }
    snapshot.update(overrides)
    return snapshot


def scratch_payload(label="first", **overrides):
    payload = {
        "label": label,
        "values": [1, 2, 3],
        "confidence": 0.5,
        "note": "scratch layer metadata",
    }
    payload.update(overrides)
    return payload


class FakeUniverse:
    """Minimal duck-typed ResidentUniverse facade used by fallback tests."""

    def __init__(self):
        self.leaves: dict[str, dict] = {}
        self.relations: dict[str, dict] = {}
        self.generation = 0
        self.digest = hashlib.sha256(b"fake-genesis").hexdigest()
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
        stored = {
            "leaf_id": leaf_id,
            "prepared_id": record.get("prepared_id") or leaf_id,
            "kind": record.get("kind"),
            "head": record.get("head"),
            "text": text,
            "source_digest": record.get("source_digest"),
        }
        self.leaves[leaf_id] = stored
        return {
            "status": "appended", "created": True, "duplicate": False,
            "leaf_id": leaf_id, "leaf_ids": [leaf_id],
            "generation": self.generation, "digest": self.digest,
        }

    def get_leaf(self, leaf_id):
        leaf = self.leaves.get(str(leaf_id))
        return dict(leaf) if leaf is not None else None

    def get_relation(self, relation_id):
        relation = self.relations.get(str(relation_id))
        return dict(relation) if relation is not None else None

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


class FieldRoundTripTests(RealUniverseTestCase):
    def test_field_requires_approval_and_returns_generation_digest_receipt(self):
        persistence = FieldPersistence(self.universe)
        self.assertIsNone(persistence.load_field())

        before = self.universe.snapshot()
        denial = persistence.save_field(field_snapshot())
        self.assertIsInstance(denial, FieldReceipt)
        self.assertFalse(denial)
        self.assertFalse(denial["accepted"])
        self.assertEqual(denial["status"], "approval_required")
        self.assertTrue(denial["reasons"])
        self.assertEqual(self.universe.snapshot()["generation"],
                         before["generation"])
        self.assertIsNone(persistence.load_field())

        receipt = persistence.save_field(field_snapshot(), approved=True)
        self.assertTrue(receipt)
        self.assertTrue(receipt["accepted"])
        self.assertEqual(receipt["status"], "committed")
        self.assertTrue(receipt["created"])
        self.assertFalse(receipt["duplicate"])
        self.assertTrue(receipt["approved"])
        self.assertEqual(receipt["anchor_leaf_id"], FIELD_ANCHOR_ID)
        self.assertTrue(receipt["leaf_id"].startswith("field:"))
        self.assertEqual(len(receipt["content_digest"]), 64)
        self.assertEqual(len(receipt["digest"]), 64)
        self.assertEqual(len(receipt["state_digest"]), 64)
        self.assertFalse(receipt["canonical_commit"])
        self.assertFalse(receipt["canonical_state_modified"])
        self.assertTrue(receipt["main_field_modified"])

        after = self.universe.snapshot()
        self.assertEqual(receipt["generation"], after["generation"])
        self.assertEqual(receipt["digest"], after["digest"])
        self.assertEqual(receipt["state_digest"], after["digest"])
        self.assertGreater(receipt["generation"], before["generation"])

        leaf = self.universe.get_leaf(receipt["leaf_id"])
        self.assertIsNotNone(leaf)
        self.assertEqual(json.loads(leaf["text"]), field_snapshot())
        self.assertEqual(persistence.load_field(), field_snapshot())

        duplicate = persistence.save_field(field_snapshot(), approved=True)
        self.assertTrue(duplicate)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["generation"], after["generation"])
        self.assertEqual(duplicate["content_digest"],
                         receipt["content_digest"])

    def test_committed_main_field_is_read_only(self):
        persistence = FieldPersistence(self.universe)
        first = persistence.save_field(field_snapshot(), approved=True)
        generation = self.universe.snapshot()["generation"]

        changed = field_snapshot(heads=128, note="trying to replace")
        denial = persistence.save_field(changed, approved=True)
        self.assertFalse(denial)
        self.assertFalse(denial["accepted"])
        self.assertEqual(denial["status"], "read_only")
        self.assertTrue(denial["reasons"])
        self.assertEqual(denial["existing_digest"], first["content_digest"])
        self.assertNotEqual(denial["requested_digest"],
                            first["content_digest"])
        self.assertEqual(self.universe.snapshot()["generation"], generation)
        self.assertEqual(persistence.load_field(), field_snapshot())

        no_approval = persistence.save_field(changed)
        self.assertFalse(no_approval)
        self.assertEqual(no_approval["status"], "approval_required")


class ScratchLayerTests(RealUniverseTestCase):
    def test_scratch_versions_are_isolated_and_never_replace_main_field(self):
        persistence = FieldPersistence(self.universe)
        field_receipt = persistence.save_field(
            field_snapshot(), approved=True)
        field_leaf_before = self.universe.get_leaf(
            field_receipt["leaf_id"])["text"]

        first = persistence.save_scratch(5, 3, scratch_payload("first"))
        self.assertTrue(first)
        self.assertEqual(first["status"], "committed")
        self.assertEqual(first["kind"], "scratch")
        self.assertEqual(first["head"], 5)
        self.assertEqual(first["layer"], 3)
        self.assertEqual(first["version"], 0)
        self.assertTrue(first["leaf_id"].startswith("scratch:5:3:"))
        self.assertFalse(first["main_field_modified"])
        self.assertEqual(persistence.load_scratch(5, 3),
                         scratch_payload("first"))
        self.assertEqual(persistence.load_field(), field_snapshot())

        field_leaf_after = self.universe.get_leaf(
            field_receipt["leaf_id"])["text"]
        self.assertEqual(field_leaf_after, field_leaf_before)

        duplicate = persistence.save_scratch(5, 3, scratch_payload("first"))
        self.assertTrue(duplicate)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["generation"], first["generation"])

        second = persistence.save_scratch(5, 3, scratch_payload("second"))
        self.assertTrue(second)
        self.assertEqual(second["version"], 1)
        self.assertEqual(persistence.load_scratch(5, 3),
                         scratch_payload("second"))
        self.assertEqual(persistence.load_scratch(5, 3, version=0),
                         scratch_payload("first"))

        other_route = persistence.save_scratch(
            5, 4, scratch_payload("other layer"))
        self.assertEqual(other_route["version"], 0)
        self.assertEqual(persistence.load_scratch(5, 4),
                         scratch_payload("other layer"))
        self.assertEqual(persistence.load_scratch(5, 3),
                         scratch_payload("second"))
        self.assertIsNone(persistence.load_scratch(6, 3))
        self.assertEqual(persistence.load_field(), field_snapshot())

    def test_reopen_recovers_field_and_latest_scratch_deterministically(self):
        persistence = FieldPersistence(self.universe)
        receipt = persistence.save_field(field_snapshot(), approved=True)
        persistence.save_scratch(12, 2, scratch_payload("old"))
        latest = persistence.save_scratch(12, 2, scratch_payload(
            "new", values=[9, 8, 7]))
        self.assertEqual(self.universe.checkpoint(
            self.root / "checkpoints")["generation"],
            self.universe.snapshot()["generation"])

        self.reopen()
        reopened = FieldPersistence(self.universe)
        self.assertEqual(reopened.load_field(), field_snapshot())
        self.assertEqual(reopened.load_scratch(12, 2),
                         scratch_payload("new", values=[9, 8, 7]))
        self.assertEqual(reopened.load_scratch(12, 2, version=0),
                         scratch_payload("old"))
        self.assertIsNone(reopened.load_scratch(12, 3))

        duplicate = reopened.save_field(field_snapshot(), approved=True)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(duplicate["content_digest"],
                         receipt["content_digest"])
        self.assertEqual(duplicate["anchor_generation"],
                         receipt["anchor_generation"])
        self.assertEqual(duplicate["generation"], latest["generation"])

        denied = reopened.save_field(field_snapshot(heads=1), approved=True)
        self.assertEqual(denied["status"], "read_only")
        self.assertEqual(reopened.load_field(), field_snapshot())

    def test_checkpoint_delegates_to_universe(self):
        persistence = FieldPersistence(self.universe)
        persistence.save_field(field_snapshot(), approved=True)
        result = persistence.checkpoint(self.root / "checkpoint")
        self.assertEqual(result["generation"],
                         self.universe.snapshot()["generation"])
        self.assertEqual(len(result["digest"]), 64)


class DenialAndBoundsTests(RealUniverseTestCase):
    def test_oversized_secret_path_and_non_json_payloads_are_denied(self):
        persistence = FieldPersistence(self.universe)
        before = self.universe.snapshot()

        with self.assertRaises(FieldPersistenceBoundsError):
            persistence.save_field(
                {"blob": "x" * (MAX_FIELD_BYTES + 1)}, approved=True)
        with self.assertRaises(FieldPersistenceBoundsError):
            persistence.save_scratch(
                1, 0, {"blob": "x" * (MAX_SCRATCH_BYTES + 1)})

        with self.assertRaises(FieldPersistenceSecretError):
            persistence.save_field(
                {"api_key": "sk-" + "a" * 32}, approved=True)
        with self.assertRaises(FieldPersistenceSecretError):
            persistence.save_scratch(
                1, 0, {"note": "-----BEGIN " "PRIVATE KEY-----"})

        with self.assertRaises(FieldPersistencePathError):
            persistence.save_field({"path": "/etc/passwd"}, approved=True)
        with self.assertRaises(FieldPersistencePathError):
            persistence.save_scratch(1, 0, {"note": "../../escape"})
        with self.assertRaises(FieldPersistencePathError):
            persistence.save_scratch(1, 0, {"file": "relative/secret.txt"})
        with self.assertRaises(FieldPersistencePathError):
            persistence.save_scratch(1, 0, {"path": "notes.txt"})

        with self.assertRaises(FieldPersistenceError):
            persistence.save_scratch(1, 0, {"object": object()})
        with self.assertRaises(FieldPersistenceBoundsError):
            persistence.save_scratch(1, 0, {"value": float("nan")})

        nested = {"value": "leaf"}
        for _ in range(MAX_JSON_DEPTH + 2):
            nested = {"nested": nested}
        with self.assertRaises(FieldPersistenceBoundsError):
            persistence.save_scratch(1, 0, nested)

        with self.assertRaises(FieldPersistenceError):
            persistence.save_scratch(256, 0, {"ok": True})
        with self.assertRaises(FieldPersistenceError):
            persistence.save_scratch(0, 10, {"ok": True})

        after = self.universe.snapshot()
        self.assertEqual(after["generation"], before["generation"])
        self.assertEqual(after["counts"], before["counts"])
        self.assertIsNone(persistence.load_field())
        self.assertIsNone(persistence.load_scratch(1, 0))

    def test_missing_records_and_dangling_anchor(self):
        persistence = FieldPersistence(self.universe)
        self.assertIsNone(persistence.load_field())
        self.assertIsNone(persistence.load_scratch(0, 0))

        # A missing payload behind an existing anchor is reported as corruption,
        # not as forged or partially recovered metadata.
        anchor = {
            "schema": ANCHOR_SCHEMA,
            "kind": "field",
            "version": 0,
            "data_leaf_id": "field:" + "0" * 32,
            "content_digest": "0" * 64,
        }
        self.universe.ingest({
            "type": "leaf",
            "schema": "nanlm.resident-universe/1",
            "leaf_id": FIELD_ANCHOR_ID,
            "kind": "data",
            "head": 0,
            "text": json.dumps(anchor, sort_keys=True,
                               separators=(",", ":")),
            "features": [],
        })
        with self.assertRaises(FieldPersistenceError):
            persistence.load_field()


class FakeFacadeTests(unittest.TestCase):
    def test_fake_round_trip_approval_receipt_and_reopen(self):
        universe = FakeUniverse()
        persistence = FieldPersistence(universe)
        self.assertIsNone(persistence.load_field())

        denial = persistence.save_field(field_snapshot())
        self.assertFalse(denial)
        self.assertEqual(denial["status"], "approval_required")
        self.assertEqual(universe.generation, 0)

        receipt = persistence.save_field(field_snapshot(), approved=True)
        self.assertTrue(receipt)
        self.assertEqual(receipt["generation"], universe.generation)
        self.assertEqual(receipt["digest"], universe.digest)
        self.assertEqual(receipt["content_digest"],
                         receipt["payload_digest"])
        self.assertEqual(persistence.load_field(), field_snapshot())

        duplicate = persistence.save_field(field_snapshot(), approved=True)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(universe.generation, receipt["generation"])
        read_only = persistence.save_field(
            field_snapshot(heads=8), approved=True)
        self.assertEqual(read_only["status"], "read_only")
        self.assertEqual(persistence.load_field(), field_snapshot())

        scratch = persistence.save_scratch(4, 1, scratch_payload("fake"))
        self.assertEqual(scratch["generation"], universe.generation)
        self.assertEqual(persistence.load_scratch(4, 1),
                         scratch_payload("fake"))
        self.assertEqual(persistence.load_field(), field_snapshot())
        self.assertIsNone(persistence.load_scratch(4, 2))

        # A fresh adapter over the same duck-typed store has no in-memory cache.
        reopened = FieldPersistence(universe)
        self.assertEqual(reopened.load_field(), field_snapshot())
        self.assertEqual(reopened.load_scratch(4, 1),
                         scratch_payload("fake"))
        self.assertEqual(reopened.save_field(
            field_snapshot(), approved=True)["status"], "duplicate")
        self.assertEqual(reopened.checkpoint("checkpoints")["generation"],
                         universe.generation)

    def test_fake_scratch_does_not_create_or_change_main_field(self):
        universe = FakeUniverse()
        persistence = FieldPersistence(universe)
        self.assertIsNone(persistence.load_field())
        persistence.save_scratch(0, 0, scratch_payload("only scratch"))
        self.assertIsNone(persistence.load_field())
        self.assertNotIn("field:" + "0" * 32, universe.leaves)
        self.assertNotIn(FIELD_ANCHOR_ID, universe.leaves)

        persistence.save_field(field_snapshot(), approved=True)
        field_before = persistence.load_field()
        field_count = len([
            leaf for leaf in universe.leaves.values()
            if leaf.get("text") and '"kind":"field"' in leaf["text"]
        ])
        persistence.save_scratch(0, 0, scratch_payload("second"))
        persistence.save_scratch(0, 0, scratch_payload("third"))
        self.assertEqual(persistence.load_field(), field_before)
        field_after = len([
            leaf for leaf in universe.leaves.values()
            if leaf.get("text") and '"kind":"field"' in leaf["text"]
        ])
        self.assertEqual(field_after, field_count)
        self.assertEqual(persistence.load_scratch(0, 0)["label"], "third")

    def test_constructor_requires_the_minimal_facade(self):
        with self.assertRaises(TypeError):
            FieldPersistence(object())
        with self.assertRaises(TypeError):
            FieldPersistence({"ingest": lambda record: None})
        fake = FakeUniverse()
        fake.ingest = None
        with self.assertRaises(TypeError):
            FieldPersistence(fake)


if __name__ == "__main__":
    unittest.main()
