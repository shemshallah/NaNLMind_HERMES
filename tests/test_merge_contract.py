"""Algebraic contract tests for additive Hermes memory merges.

These tests compare independently built universes, not merely repeated writes to
one database, so they catch order-sensitive event identity or projection bugs.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.memory import MemoryStore


class MergeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.left = MemoryStore(root / "left.sqlite3")
        self.right = MemoryStore(root / "right.sqlite3")
        self.source_a = hashlib.sha256(b"source-a").hexdigest()
        self.source_b = hashlib.sha256(b"source-b").hexdigest()
        self.records = [
            {
                "item": {
                    "kind": "hypothesis", "subject": "claim:water-boils",
                    "predicate": "has_property", "object": "boils-at-100C",
                    "polarity": "supports", "confidence": 0.8,
                    "evidence": {"samples": 12},
                    "provenance": {"experiment": "A"},
                },
                "source_digest": self.source_a, "head": 17, "layer": 5,
            },
            {
                "item": {
                    "kind": "hypothesis", "subject": "claim:water-boils",
                    "predicate": "has_property", "object": "boils-at-100C",
                    "polarity": "refutes", "confidence": 0.7,
                    "evidence": {"samples": 8},
                    "provenance": {"experiment": "B"},
                },
                "source_digest": self.source_b, "head": 91, "layer": 5,
            },
            {
                "item": {
                    "kind": "equation", "subject": "equation:e=mc2",
                    "expression": {"op": "equals", "left": "e", "right": {"op": "mul", "args": ["m", "c", "c"]}},
                    "polarity": "neutral", "confidence": 1.0,
                    "provenance": {"source": "symbolic-corpus"},
                },
                "source_digest": self.source_a, "head": 203, "layer": 4,
            },
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _event_set(self, store):
        return {row["event_id"] for row in store.query(limit=1000)}

    def test_permutation_and_replay_converge_to_same_event_set(self):
        first = self.left.merge(self.records)
        second = self.right.merge(list(reversed(self.records)))
        self.assertEqual(first, {"accepted": 3, "duplicates": 0})
        self.assertEqual(second, {"accepted": 3, "duplicates": 0})
        self.assertEqual(self._event_set(self.left), self._event_set(self.right))

        replay = self.right.merge(self.records)
        self.assertEqual(replay, {"accepted": 0, "duplicates": 3})
        self.assertEqual(self.right.status()["events"], 3)

    def test_reorganization_converges_and_preserves_conflicting_evidence(self):
        self.left.merge(self.records)
        self.right.merge(list(reversed(self.records)))
        left_projection = self.left.reorganize()
        right_projection = self.right.reorganize()
        self.assertEqual(left_projection, right_projection)
        self.assertEqual(left_projection["contested_subjects"], 1)

        left_assessment = self.left.query_assessments("claim:water-boils")[0]
        right_assessment = self.right.query_assessments("claim:water-boils")[0]
        self.assertEqual(left_assessment, right_assessment)
        self.assertEqual(left_assessment["status"], "contested")
        self.assertEqual(left_assessment["evidence_count"], 2)
        self.assertEqual(len(left_assessment["source_event_ids"]), 2)

        routes = self.left.query(subject="claim:water-boils", limit=10)
        self.assertEqual({row["head"] for row in routes}, {17, 91})
        self.assertEqual({row["layer"] for row in routes}, {9})

    def test_reopen_then_merge_is_idempotent(self):
        self.left.merge(self.records)
        reopened = MemoryStore(self.left.path)
        result = reopened.merge(list(reversed(self.records)))
        self.assertEqual(result, {"accepted": 0, "duplicates": 3})
        self.assertEqual(reopened.status()["events"], 3)


if __name__ == "__main__":
    unittest.main()
