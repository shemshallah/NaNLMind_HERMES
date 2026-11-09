"""Unified resident-universe facade over the real sibling modules.

The tests use the actual layout, cells, leaves, relations, resonance, ingest
and persistence implementations.  Synthetic plugin rows are prepared by the
real ingest adapter before they enter the universe.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes.resident import (
    DEFAULT_BYTES,
    MAX_BYTES,
    MIN_BYTES,
    SECTION_ARENA,
    SECTION_CHART,
    SECTION_FEATURES,
    SECTION_HEADER,
    SECTION_HEADS,
    SECTION_HYPEREDGES,
    SECTION_INDEX,
    SECTION_JOURNAL,
    SECTION_LEAVES,
    SECTION_PAIRS,
    SECTION_RELATIONS,
    SECTION_TOKENS,
    HEAD_COUNT,
    PAIR_COUNT,
    ingest,
    persistence,
)
from nanlmind_hermes.resident import universe as universe_module
from nanlmind_hermes.resident.universe import (
    FEATURE_DIMENSIONS,
    KNOWN_SECTIONS,
    ResidentIntegrationError,
    ResidentUniverse,
    ResidentUniverseError,
)

MIB = 1024 * 1024
SIZE_32MIB = 32 * MIB
SIZE_16MIB = 16 * MIB
SOURCE = hashlib.sha256(b"resident-universe-test-source").hexdigest()

REQUIRED_SECTIONS = {
    SECTION_CHART,
    SECTION_HEADS,
    SECTION_PAIRS,
    SECTION_LEAVES,
    SECTION_RELATIONS,
    SECTION_HYPEREDGES,
    SECTION_FEATURES,
    SECTION_TOKENS,
    SECTION_INDEX,
    SECTION_ARENA,
    SECTION_JOURNAL,
}


def relation_row(subject="river", predicate="flows_to", object_="sea",
                 **overrides):
    row = {
        "kind": "relation",
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "expression": {"op": "relation",
                       "args": [subject, predicate, object_]},
        "provenance": {"source": "resident-universe-test", "native_head": 17},
        "polarity": "supports",
        "confidence": 0.8,
        "head": 17,
        "layer": 2,
    }
    row.update(overrides)
    return row


def observation_row(text="The river carries water.", **overrides):
    row = {
        "kind": "observation",
        "subject": "river",
        "predicate": "observed",
        "object": "water",
        "expression": text,
        "provenance": {"source": "resident-universe-test"},
        "polarity": "neutral",
        "confidence": 0.6,
        "head": 21,
        "layer": 0,
    }
    row.update(overrides)
    return row


class UniverseTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.path = self.root / "universe.nlm"

    def tearDown(self):
        self._tmp.cleanup()

    def open(self, size=SIZE_32MIB):
        return ResidentUniverse.open(self.path, size)

    def ingest_relation(self, universe, **overrides):
        row = relation_row(**overrides)
        return universe.ingest(row), row


class OpenAndBudgetTests(UniverseTestCase):
    def test_open_initialises_all_sections_in_one_memory_mapped_file(self):
        universe = self.open()
        try:
            self.assertEqual(universe.path, self.path)
            self.assertEqual(self.path.stat().st_size, SIZE_32MIB)
            self.assertEqual(universe.file_size, SIZE_32MIB)
            self.assertEqual(sorted(self.root.iterdir()), [self.path])

            snapshot = universe.snapshot()
            self.assertEqual(snapshot["schema"], "nanlm.resident-universe/1")
            self.assertEqual(snapshot["generation"], 0)
            self.assertEqual(len(snapshot["digest"]), 64)
            self.assertEqual(snapshot["counts"]["leaves"], 0)
            self.assertEqual(snapshot["counts"]["relations"], 0)
            self.assertEqual(snapshot["counts"]["features"], 0)
            self.assertEqual(snapshot["counts"]["heads"], HEAD_COUNT)
            self.assertEqual(snapshot["counts"]["pair_cells"], PAIR_COUNT)
            self.assertEqual(set(int(key) for key in snapshot["sections"]),
                             set(range(1, 13)))
            for section_id in REQUIRED_SECTIONS:
                self.assertGreater(
                    snapshot["sections"][str(section_id)]["size"], 0)
            self.assertEqual(KNOWN_SECTIONS, tuple(range(1, 13)))
            self.assertLessEqual(snapshot["size"], MAX_BYTES)
            self.assertEqual(universe.cells.read_head(17).head_id, 17)
            self.assertEqual(len(universe.features), 0)
        finally:
            universe.close()

    def test_default_budget_and_hard_one_gib_ceiling(self):
        self.assertEqual(DEFAULT_BYTES, 256 * MIB)
        self.assertEqual(MAX_BYTES, 1024 * MIB)
        oversized = self.root / "too-big.nlm"
        with self.assertRaises(ResidentUniverseError):
            ResidentUniverse.open(oversized, MAX_BYTES + 1)
        self.assertFalse(oversized.exists())
        with self.assertRaises((ResidentUniverseError, ValueError)):
            ResidentUniverse.open(self.root / "tiny.nlm", MIN_BYTES - 1)
        with self.assertRaises((ResidentUniverseError, ValueError)):
            ResidentUniverse.open(self.root / "not-int.nlm", 32.5)
        with self.assertRaises(ResidentUniverseError):
            ResidentUniverse.open(self.root / "two-gib.nlm", 2 * MAX_BYTES)
        self.assertFalse((self.root / "two-gib.nlm").exists())

        with ResidentUniverse.open(self.root / "default.nlm") as universe:
            self.assertEqual(universe.file_size, DEFAULT_BYTES)
            self.assertLessEqual(universe.file_size, MAX_BYTES)
            self.assertEqual(len(universe.snapshot()["sections"]), 12)

    def test_missing_sibling_imports_fail_with_clear_integration_error(self):
        missing = ModuleNotFoundError(
            "No module named 'nanlmind_hermes.resident.relations'",
            name="nanlmind_hermes.resident.relations")
        with mock.patch.object(universe_module, "_IMPORT_ERROR", missing):
            with self.assertRaisesRegex(
                    ResidentIntegrationError,
                    r"resident sibling modules.*relations"):
                ResidentUniverse.open(self.root / "missing.nlm", SIZE_16MIB)
        self.assertFalse((self.root / "missing.nlm").exists())


class IngestTests(UniverseTestCase):
    def test_relation_ingest_updates_leaf_relation_pair_and_feature_state(self):
        with self.open() as universe:
            receipt, row = self.ingest_relation(universe)
            self.assertEqual(receipt["schema"], "nanlm.resident-receipt/1")
            self.assertEqual(receipt["status"], "appended")
            self.assertTrue(receipt["created"])
            self.assertFalse(receipt["duplicate"])
            self.assertGreaterEqual(receipt["generation"], 1)
            self.assertGreaterEqual(receipt["leaves_added"], 3)
            self.assertEqual(receipt["relations_added"], 1)
            self.assertGreaterEqual(receipt["features_added"], 3)
            self.assertEqual(receipt["pair_updates"], 1)
            self.assertEqual(len(receipt["source_digest"]), 64)
            self.assertEqual(len(receipt["relation_id"]), 64)
            self.assertEqual(len(receipt["hyperedge_id"]), 64)
            self.assertEqual(len(receipt["digest"]), 64)
            self.assertIsNotNone(receipt["relation_leaf_id"])

            counts = universe.snapshot()["counts"]
            self.assertGreaterEqual(counts["leaves"], 4)
            self.assertEqual(counts["relations"], 1)
            self.assertEqual(counts["hyperedges"], 1)
            self.assertGreaterEqual(counts["features"], 3)
            self.assertGreaterEqual(counts["occupied_pairs"], 1)

            relation = universe.get_relation(receipt["relation_id"])
            self.assertIsNotNone(relation)
            self.assertEqual(relation["polarity"], "supports")
            self.assertAlmostEqual(relation["confidence"], 0.8)
            self.assertEqual(relation["hyperedge_id"],
                             receipt["hyperedge_id"])
            self.assertEqual(relation["source_digest"],
                             receipt["source_digest"])

            hyperedge = universe.get_hyperedge(receipt["hyperedge_id"])
            self.assertIsNotNone(hyperedge)
            self.assertGreaterEqual(hyperedge["arity"], 6)
            kinds = {member["kind"] for member in hyperedge["members"]}
            self.assertTrue({"leaf", "pair", "mirror", "relation"} <= kinds)

            self.assertGreaterEqual(len(receipt["leaf_ids"]), 3)
            for leaf_id in receipt["leaf_ids"]:
                leaf = universe.get_leaf(leaf_id)
                self.assertIsNotNone(leaf)
                self.assertEqual(leaf["head"], 17)
            relation_leaf = universe.get_leaf(receipt["relation_leaf_id"])
            self.assertIsNotNone(relation_leaf)
        with ResidentUniverse.open(
                self.root / "duplicate.nlm", SIZE_32MIB) as universe:
            before = universe.snapshot()
            first, _ = self.ingest_relation(universe)
            after_first = universe.snapshot()
            duplicate, _ = self.ingest_relation(universe)
            after_duplicate = universe.snapshot()

            self.assertTrue(first["created"])
            self.assertTrue(duplicate["duplicate"])
            self.assertFalse(duplicate["created"])
            self.assertEqual(duplicate["generation"], first["generation"])
            self.assertEqual(after_duplicate["digest"],
                             after_first["digest"])
            self.assertEqual(after_duplicate["generation"],
                             after_first["generation"])
            self.assertEqual(after_duplicate["counts"],
                             after_first["counts"])
            self.assertNotEqual(after_first["digest"], before["digest"])

    def test_observation_is_a_leaf_and_never_a_relation(self):
        with self.open() as universe:
            receipt = universe.ingest(observation_row())
            self.assertTrue(receipt["created"])
            self.assertEqual(receipt["relations_added"], 0)
            self.assertGreaterEqual(receipt["features_added"], 1)
            counts = universe.snapshot()["counts"]
            self.assertGreaterEqual(counts["leaves"], 1)
            self.assertEqual(counts["relations"], 0)
            self.assertEqual(counts["hyperedges"], 0)
            leaf = universe.get_leaf(receipt["leaf_ids"][0])
            self.assertIsNotNone(leaf)
            self.assertNotEqual(leaf["kind"], "relation")

    def test_mixed_batch_and_generation_bound_receipts(self):
        with self.open() as universe:
            first, _ = self.ingest_relation(universe)
            second = universe.ingest(observation_row(
                text="Mountain peaks are high.", subject="mountain",
                object="peak", head=42))
            self.assertGreater(second["generation"],
                               first["generation"])
            self.assertEqual(universe.snapshot()["counts"]["relations"], 1)
            batch = universe.ingest([
                relation_row(subject="sun", predicate="warms",
                             object_="stone", head=33),
                observation_row(text="Stone is heavy.", subject="stone",
                                object="weight", head=34),
            ])
            self.assertEqual(batch["records"], 5)
            self.assertTrue(batch["created"])
            self.assertGreater(batch["generation"], second["generation"])
            self.assertEqual(universe.snapshot()["counts"]["relations"], 2)


class QueryAndLinkTests(UniverseTestCase):
    def test_resonance_query_returns_ranked_resident_hits(self):
        with self.open() as universe:
            river, _ = self.ingest_relation(universe)
            mountain, mountain_row = self.ingest_relation(
                universe, subject="mountain", predicate="reaches",
                object_="peak", head=42)
            result = universe.query("river flows_to sea", 5)
            self.assertEqual(result["top_k"], 5)
            hits = result["hits"]
            self.assertEqual(list(result), hits)
            self.assertLessEqual(len(hits), 5)
            scores = [hit["score"] for hit in hits]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(result["generation"] >= 1 for _ in hits))
            self.assertTrue(any(hit["kind"] == "relation" for hit in hits))
            self.assertNotEqual(hits[0]["feature_id"],
                                mountain["relation_id"])

            prepared = ingest.prepare_records(
                [relation_row()], source_digest=SOURCE)
            relation = [record for record in prepared
                        if record["type"] == "relation"][0]
            exact = universe.query(relation["features"], top_k=1)
            self.assertEqual(len(exact), 1)
            self.assertEqual(exact[0]["feature_id"], relation["relation_id"])
            self.assertAlmostEqual(exact[0]["score"], 1.0, places=5)
            self.assertEqual(exact[0]["kind"], "relation")

            text_hit = universe.query(text="mountain reaches peak", top_k=3)
            self.assertTrue(any(hit.get("kind") == "relation"
                                for hit in text_hit))
            self.assertEqual(len(universe.query(
                mountain_row["expression"], top_k=1)), 1)
            self.assertEqual(len(universe.query("x", top_k=0)), 1)

    def test_link_relinks_child_and_persists_tree_shape(self):
        with self.open() as universe:
            parent, _ = self.ingest_relation(universe)
            child, _ = self.ingest_relation(
                universe, subject="delta", predicate="joins",
                object_="river", head=55)
            parent_id = parent["leaf_ids"][0]
            child_id = child["leaf_ids"][0]
            linked = universe.link(parent_id, child_id)
            self.assertTrue(linked["created"])
            self.assertFalse(linked["duplicate"])
            self.assertEqual(linked["parent"],
                             universe.get_leaf(parent_id)["leaf_id"])
            self.assertEqual(linked["child"],
                             universe.get_leaf(child_id)["leaf_id"])
            self.assertGreaterEqual(linked["depth"], 1)
            duplicate = universe.link(parent_id, child_id)
            self.assertTrue(duplicate["duplicate"])
            self.assertFalse(duplicate["created"])
            self.assertEqual(duplicate["generation"], linked["generation"])

            before = universe.snapshot()
            universe.close()
            with self.open() as reopened:
                after = reopened.snapshot()
                self.assertEqual(after["generation"], before["generation"])
                self.assertEqual(after["digest"], before["digest"])
                self.assertEqual(reopened.get_leaf(child_id)["parent"],
                                 reopened.get_leaf(parent_id)["slot"])


class ReopenAndCheckpointTests(UniverseTestCase):
    def test_reopen_preserves_counts_digest_relation_and_features(self):
        with self.open() as universe:
            receipt, _ = self.ingest_relation(universe)
            universe.ingest(observation_row())
            before = universe.snapshot()
            hits_before = universe.query("river flows_to sea", 4)["hits"]
            relation_before = universe.get_relation(receipt["relation_id"])
        self.assertTrue(self.path.exists())

        with self.open() as reopened:
            after = reopened.snapshot()
            self.assertEqual(after["generation"], before["generation"])
            self.assertEqual(after["digest"], before["digest"])
            self.assertEqual(after["counts"], before["counts"])
            self.assertEqual(reopened.get_relation(receipt["relation_id"]),
                             relation_before)
            hits_after = reopened.query("river flows_to sea", 4)["hits"]
            self.assertEqual(
                [(hit["feature_id"], round(hit["score"], 8))
                 for hit in hits_after],
                [(hit["feature_id"], round(hit["score"], 8))
                 for hit in hits_before],
            )
            self.assertIsNotNone(reopened.get_leaf(receipt["leaf_ids"][0]))

    def test_checkpoint_writes_bounded_manifest_and_recovers(self):
        with self.open() as universe:
            receipt, _ = self.ingest_relation(universe)
            snapshot = universe.snapshot()
            manifest = universe.checkpoint()
            self.assertEqual(manifest["generation"], snapshot["generation"])
            self.assertEqual(manifest["digest"], snapshot["digest"])
            self.assertEqual(len(manifest["manifest_digest"]), 64)
            self.assertEqual(manifest["manifest"]["generation"],
                             snapshot["generation"])
            self.assertEqual(manifest["manifest"]["counts"],
                             snapshot["counts"])
            self.assertEqual(manifest["manifest"]["files"][0]["path"],
                             str(self.path.absolute()))
            self.assertTrue(
                manifest["manifest"]["files"][0]["regions"])
            directory = Path(manifest["directory"])
            self.assertTrue(directory.is_dir())

            recovered = persistence.recover(directory)
            self.assertEqual(recovered.generation, snapshot["generation"])
            self.assertEqual(recovered.manifest["digest"],
                             snapshot["digest"])
            self.assertLess(
                sum(path.stat().st_size for path in directory.iterdir()),
                universe.file_size,
            )

            repeated = universe.checkpoint()
            self.assertEqual(repeated["generation"], manifest["generation"])
            self.assertEqual(repeated["manifest_digest"],
                             manifest["manifest_digest"])

            universe.ingest(observation_row(text="A second observation."))
            second = universe.checkpoint()
            self.assertGreater(second["generation"],
                               manifest["generation"])
            self.assertNotEqual(second["digest"], manifest["digest"])
            newest = persistence.recover(directory)
            self.assertEqual(newest.generation, second["generation"])
            self.assertEqual(newest.manifest["counts"]["leaves"],
                             second["snapshot"]["counts"]["leaves"])
            self.assertEqual(receipt["relations_added"], 1)


if __name__ == "__main__":
    unittest.main()
