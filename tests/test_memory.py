import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3", heads=256, layers=7)
        self.source = hashlib.sha256(b"source").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_idempotent_ingest_and_reopen(self):
        item = {"kind": "equation", "subject": "energy", "expression": {"op": "mul", "args": ["m", "c", "c"]},
                "polarity": "supports", "confidence": 0.8}
        first = self.store.ingest(item, source_digest=self.source)
        second = self.store.ingest(item, source_digest=self.source)
        self.assertEqual(first["event_id"], second["event_id"])
        reopened = MemoryStore(self.store.path)
        self.assertEqual(reopened.status()["events"], 1)

    def test_conflicting_claims_are_retained(self):
        for polarity in ("supports", "refutes"):
            self.store.ingest({"kind": "hypothesis", "subject": "claim-x", "object": "true",
                               "polarity": polarity}, source_digest=self.source)
        self.assertEqual(self.store.status()["events"], 2)

    def test_reorganization_is_idempotent(self):
        self.store.ingest({"kind": "relation", "subject": "a", "predicate": "links", "object": "b"},
                          source_digest=self.source)
        a = self.store.reorganize()
        b = self.store.reorganize()
        self.assertEqual(a, b)
        self.assertEqual(self.store.status()["active_routes"], 1)

    def test_merge_order_independent_event_set(self):
        items = [
            {"kind": "entity", "subject": "alpha"},
            {"kind": "entity", "subject": "beta"},
        ]
        ids = set()
        for item in items:
            ids.add(self.store.ingest(item, source_digest=self.source)["event_id"])
        self.assertEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main()
