"""Resident typed retrieval-hypergraph tests.

The primary paths use the real relation/leaf stores and, when available, a real
``ResidentUniverse``.  A duck-typed fake facade covers partial installations.
The published Round 032 corpus is used only as a small bounded sample; its
50k-scale line count is checked without inserting the whole corpus.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cells import (
    canonical_address,
    mirror_address,
    pair_index as cell_pair_index,
)
from nanlmind_hermes.resident.leaves import (
    HEADER_BYTES as LEAF_HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    RECORD_BYTES,
    LeafStore,
)
from nanlmind_hermes.resident.relations import RelationStore
from nanlmind_hermes.resident.retrieval_hypergraph import (
    MAX_BATCH,
    MEMBER_CYCLE,
    MEMBER_DESTINATION,
    MEMBER_FACT,
    MEMBER_MIRROR,
    MEMBER_ORIENTATION,
    MEMBER_PAIR,
    MEMBER_PAIR_ADDRESS,
    MEMBER_PROVENANCE,
    MEMBER_RELATION_TYPE,
    MEMBER_SOURCE,
    MEMBER_VERTEX,
    RetrievalBoundsError,
    RetrievalConflictError,
    RetrievalDenied,
    RetrievalHypergraph,
    RetrievalProvenanceError,
)

try:  # The real facade is the primary target; fakes cover partial installs.
    from nanlmind_hermes.resident.universe import ResidentUniverse
except Exception:  # pragma: no cover - only on a partial installation
    ResidentUniverse = None  # type: ignore[assignment]


MIB = 1024 * 1024
SIZE_16MIB = 16 * MIB
ROUND032_FACTS = (Path(__file__).resolve().parents[1]
                  / "experiments" / "round032" / "data" / "facts.tsv")
POINTER_KEYS = frozenset({
    "addr", "address", "buffer", "bufferaddress", "memaddress",
    "memoryaddress", "offset", "pointer", "ptr", "rawpointer", "rawptr",
})


def make_leaf_store(capacity=128):
    index_capacity = 1
    while index_capacity < max(16, capacity * 2):
        index_capacity <<= 1
    buffer = bytearray(
        LEAF_HEADER_BYTES + capacity * RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)
    return buffer, LeafStore.create(buffer, capacity=capacity)


def assert_json_safe(testcase, value, path="<root>"):
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        testcase.fail("non-integer float at %s: %r" % (path, value))
    if isinstance(value, dict):
        for key, item in value.items():
            testcase.assertIsInstance(key, str)
            testcase.assertNotIn(
                key.casefold().replace("_", ""), POINTER_KEYS,
                "raw-pointer-shaped key at %s" % path)
            assert_json_safe(testcase, item, "%s.%s" % (path, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_safe(testcase, item, "%s[%d]" % (path, index))
        return
    testcase.fail("non-JSON value at %s: %r" % (path, type(value)))


def read_round032_sample(limit=24):
    if not ROUND032_FACTS.exists():
        return None
    facts = []
    with ROUND032_FACTS.open(encoding="utf-8") as handle:
        for line in handle:
            source, relation, destination = (int(part) for part in line.split())
            facts.append((source, relation, destination))
            if len(facts) >= limit:
                break
    return facts


class FakeUniverse:
    """Small duck-typed facade with real relation/leaf stores and leaf text."""

    def __init__(self):
        self.relations = RelationStore.allocate(
            relation_capacity=64, hyperedge_capacity=64, arena_bytes=65536)
        self._buffer, self.leaves = make_leaf_store(capacity=256)
        self.texts: dict[str, str] = {}
        self.generation = 0
        self.digest = hashlib.sha256(b"fake-retrieval-genesis").hexdigest()

    def append_leaf(self, record):
        leaf_id = str(record["leaf_id"])
        existing = self.leaves.find(bytes.fromhex(leaf_id))
        if existing is not None:
            return {"status": "duplicate", "created": False,
                    "duplicate": True, "leaf_id": leaf_id,
                    "generation": self.generation}
        text = str(record["text"])
        self.leaves.insert(
            record.get("kind", "data"), leaf_id=bytes.fromhex(leaf_id),
            content=text.encode("utf-8"),
            payload_digest=hashlib.sha256(text.encode("utf-8")).digest(),
            head=int(record.get("head", 0)))
        self.texts[leaf_id] = text
        self.generation += 1
        self.digest = hashlib.sha256(
            (self.digest + leaf_id).encode()).hexdigest()
        return {"status": "appended", "created": True, "duplicate": False,
                "leaf_id": leaf_id, "generation": self.generation}

    def get_leaf(self, leaf):
        leaf_id = str(leaf)
        try:
            record = self.leaves.find(bytes.fromhex(leaf_id))
        except ValueError:
            record = None
        if record is None:
            return None
        return {
            "leaf_id": record.leaf_id.hex(),
            "kind": record.kind,
            "head": record.head,
            "text": self.texts.get(record.leaf_id.hex()),
            "payload_digest": record.payload_digest.hex(),
        }

    def snapshot(self):
        return {"generation": self.generation, "digest": self.digest,
                "counts": {"leaves": self.leaves.count,
                           "relations": self.relations.status()["relations"]}}

    def reopen(self):
        return self


class DirectStoreTests(unittest.TestCase):
    def setUp(self):
        self.relations = RelationStore.allocate(
            relation_capacity=64, hyperedge_capacity=64, arena_bytes=65536)
        self.leaf_buffer, self.leaves = make_leaf_store(capacity=256)
        self.hg = RetrievalHypergraph(
            relations=self.relations, leaves=self.leaves)

    def add(self, source=1, relation=1, destination=2,
            provenance="round032:facts.tsv"):
        return self.hg.add_fact(source, relation, destination, provenance)

    def test_add_get_membership_and_typed_hyperedge(self):
        receipt = self.add()
        self.assertEqual(receipt["status"], "appended")
        self.assertTrue(receipt["created"])
        self.assertFalse(receipt["duplicate"])
        self.assertEqual(receipt["source"], "1")
        self.assertEqual(receipt["relation"], "1")
        self.assertEqual(receipt["destination"], "2")
        self.assertEqual(receipt["provenance"], "round032:facts.tsv")
        self.assertEqual(len(receipt["fact_id"]), 64)
        self.assertEqual(len(receipt["hyperedge_id"]), 64)

        membership = self.hg._prepare_fact(1, 1, 2, "prov")["membership"]
        self.assertEqual(receipt["pair"], list(membership["pair"]))
        self.assertEqual(receipt["pair_ref"], membership["pair_ref"])
        self.assertEqual(receipt["canonical_pair"],
                         membership["canonical_pair"])
        self.assertEqual(receipt["mirror"], membership["mirror"])
        self.assertEqual(receipt["cycle"], list(membership["cycle"]))
        self.assertEqual(
            receipt["cycle_leaf_id"], membership["cycle_leaf"].hex())

        status = self.relations.status()
        self.assertEqual(status["relations"], 1)
        self.assertEqual(status["hyperedges"], 1)

        edge = self.relations.get_hyperedge(receipt["hyperedge_id"])
        self.assertIsNotNone(edge)
        codes = [member["kind_code"] for member in edge["members"]]
        self.assertEqual(edge["arity"], 13)
        self.assertEqual(codes.count(MEMBER_FACT), 1)
        self.assertEqual(codes.count(MEMBER_SOURCE), 1)
        self.assertEqual(codes.count(MEMBER_DESTINATION), 1)
        self.assertEqual(codes.count(MEMBER_RELATION_TYPE), 1)
        self.assertEqual(codes.count(MEMBER_PAIR), 1)
        self.assertEqual(codes.count(MEMBER_PAIR_ADDRESS), 1)
        self.assertEqual(codes.count(MEMBER_MIRROR), 1)
        self.assertEqual(codes.count(MEMBER_CYCLE), 1)
        self.assertEqual(codes.count(MEMBER_VERTEX), 3)
        self.assertEqual(codes.count(MEMBER_PROVENANCE), 1)
        self.assertEqual(codes.count(MEMBER_ORIENTATION), 1)

        # Source and destination are real leaves; the fact text is stored once
        # in its own leaf and is only referenced by the hyperedge.
        source_leaf = self.leaves.find(
            bytes.fromhex(receipt["source_leaf_id"]))
        destination_leaf = self.leaves.find(
            bytes.fromhex(receipt["destination_leaf_id"]))
        cycle_leaf = self.leaves.find(bytes.fromhex(receipt["cycle_leaf_id"]))
        fact_leaf = self.leaves.find(bytes.fromhex(receipt["fact_leaf_id"]))
        self.assertIsNotNone(source_leaf)
        self.assertIsNotNone(destination_leaf)
        self.assertIsNotNone(cycle_leaf)
        self.assertIsNotNone(fact_leaf)
        self.assertEqual(source_leaf.kind, "concept")
        self.assertEqual(destination_leaf.kind, "concept")
        self.assertEqual(source_leaf.pair_id,
                         cell_pair_index(*membership["pair"]))
        self.assertEqual(self.leaves.count, 4)

        fact = self.hg.get(1, 1, 2)
        self.assertIsNotNone(fact)
        self.assertEqual(fact["source"], "1")
        self.assertEqual(fact["relation"], "1")
        self.assertEqual(fact["destination"], "2")
        self.assertEqual(fact["provenance"], "round032:facts.tsv")
        self.assertTrue(fact["resolved"])
        self.assertEqual(fact["fact_id"], receipt["fact_id"])
        self.assertEqual(fact["pair"], receipt["pair"])
        self.assertEqual(fact["canonical_pair"],
                         canonical_address(*membership["pair"]))
        self.assertEqual(fact["mirror"],
                         mirror_address(canonical_address(*membership["pair"])))
        self.assertEqual(len(fact["members"]), 13)
        self.assertTrue(self.hg.audit()["ok"])

    def test_idempotent_and_second_provenance_never_overwrites(self):
        first = self.add()
        before_status = self.relations.status()
        before_bytes = self.relations.relation_bytes(0)
        before_leaf_count = self.leaves.count

        duplicate = self.add()
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertFalse(duplicate["source_mismatch"])
        self.assertEqual(duplicate["fact_id"], first["fact_id"])
        self.assertEqual(self.relations.status(), before_status)
        self.assertEqual(self.relations.relation_bytes(0), before_bytes)
        self.assertEqual(self.leaves.count, before_leaf_count)

        other = self.add(provenance="round032:other.tsv")
        self.assertEqual(other["status"], "duplicate")
        self.assertTrue(other["duplicate"])
        self.assertTrue(other["source_mismatch"])
        self.assertEqual(other["provenance"], "round032:facts.tsv")
        self.assertEqual(other["requested_provenance"],
                         "round032:other.tsv")
        self.assertEqual(self.relations.status(), before_status)
        self.assertEqual(self.leaves.count, before_leaf_count)
        self.assertEqual(self.hg.get(1, 1, 2)["provenance"],
                         "round032:facts.tsv")
        self.assertEqual(len(self.hg), 1)

    def test_source_anchor_and_reverse_concept_membership(self):
        first = self.add(1, 1, 2)
        same_source = self.add(1, 1, 3)
        reverse = self.add(2, 1, 1)
        # Round 032 anchors the posting/membership on the source concept, so
        # two destinations for head 1 share one pair and one cycle.
        self.assertEqual(first["pair"], same_source["pair"])
        self.assertEqual(first["pair_ref"], same_source["pair_ref"])
        self.assertEqual(first["canonical_pair"],
                         same_source["canonical_pair"])
        self.assertEqual(first["cycle"], same_source["cycle"])
        # The directed source/destination pair remains available for the
        # reverse fact, but the primary source anchors differ.
        self.assertNotEqual(first["pair"], reverse["pair"])
        self.assertEqual(first["concept_pair"], reverse["concept_pair"])
        self.assertEqual(first["concept_pair_ref"],
                         reverse["concept_pair_ref"])
        self.assertEqual(first["concept_canonical_pair"],
                         reverse["concept_canonical_pair"])
        self.assertEqual(first["concept_mirror"],
                         reverse["concept_mirror"])
        self.assertEqual(first["concept_cycle"], reverse["concept_cycle"])
        self.assertNotEqual(first["orientation"], reverse["orientation"])
        self.assertEqual(self.hg.get(1, 1, 2)["destination"], "2")
        self.assertEqual(self.hg.get(2, 1, 1)["destination"], "1")
        self.assertEqual(self.relations.status()["relations"], 3)

    def test_collision_is_rejected_without_overwrite(self):
        prepared = self.hg._prepare_fact("x", "rel", "y", "prov")
        self.relations.append_relation(
            subject=b"\x01" * 32,
            predicate=prepared["predicate_ref"],
            object=prepared["object_ref"],
            source_digest=prepared["provenance_digest"],
            relation_id=prepared["relation_key"],
            hyperedge_id=prepared["edge_key"],
            hyperedge_members=prepared["members"],
        )
        before = self.relations.relation_bytes(0)
        with self.assertRaises(RetrievalConflictError):
            self.hg.add_fact("x", "rel", "y", "prov")
        self.assertEqual(self.relations.relation_bytes(0), before)
        self.assertEqual(self.relations.status()["relations"], 1)

    def test_neighbors_and_two_hop_are_typed_and_bounded(self):
        self.add("a", "hypernym", "b")
        self.add("a", "hypernym", "c")
        self.add("b", "hypernym", "d")
        self.add("c", "hypernym", "e")
        self.add("b", "related", "x")

        out = self.hg.neighbors("a")
        self.assertEqual({fact["destination"] for fact in out}, {"b", "c"})
        self.assertEqual({fact["relation"] for fact in out}, {"hypernym"})
        self.assertEqual(len(self.hg.neighbors("a", "hypernym")), 2)
        self.assertEqual(len(self.hg.neighbors("a", "related")), 0)
        incoming = self.hg.neighbors("d", direction="in")
        self.assertEqual([fact["source"] for fact in incoming], ["b"])
        both = self.hg.neighbors("b", direction="both")
        self.assertEqual({fact["destination"] for fact in both},
                         {"b", "d", "x"})
        limited = self.hg.neighbors("a", limit=1)
        self.assertEqual(len(limited), 1)
        self.assertTrue(limited.truncated)
        missing = self.hg.neighbors("no-such-concept")
        self.assertEqual(len(missing), 0)
        self.assertTrue(missing.missing)

        paths = self.hg.two_hop("a", ("hypernym", "hypernym"))
        triples = {(path["source"], path["via"], path["destination"])
                   for path in paths}
        self.assertEqual(triples,
                         {("a", "b", "d"), ("a", "c", "e")})
        via_b = self.hg.two_hop("a", ("hypernym", "hypernym"), via="b")
        self.assertEqual({path["destination"] for path in via_b}, {"d"})
        limited_paths = self.hg.two_hop("a", "hypernym", limit=1)
        self.assertEqual(len(limited_paths), 1)
        self.assertTrue(limited_paths.truncated)

    def test_snapshot_audit_batch_and_denial_are_bounded(self):
        receipt = self.hg.add_facts(
            [("a", "hypernym", "b"), ("b", "hypernym", "c"),
             ("a", "hypernym", "b")],
            provenance="round032:batch")
        self.assertEqual(receipt["accepted"], 2)
        self.assertEqual(receipt["duplicates"], 1)
        self.assertEqual(receipt["count"], 2)
        self.assertEqual(len(self.hg), 2)

        snapshot = self.hg.snapshot(limit=1)
        self.assertEqual(snapshot["fact_count"], 1)
        self.assertEqual(snapshot["limit"], 1)
        self.assertTrue(snapshot["truncated"])
        self.assertTrue(snapshot["bounded"])
        self.assertFalse(snapshot["canonical_merge"])
        self.assertLessEqual(snapshot["scanned_relations"],
                             snapshot["scan_limit"])
        assert_json_safe(self, snapshot)

        full = self.hg.snapshot(limit=16)
        self.assertEqual(full["fact_count"], 2)
        assert_json_safe(self, full)
        audit = self.hg.audit(limit=1)
        self.assertEqual(audit["scanned_relations"], 1)
        self.assertTrue(audit["truncated"])
        self.assertTrue(self.hg.audit(limit=16)["ok"])
        assert_json_safe(self, audit)

        with self.assertRaises(RetrievalDenied):
            self.hg.merge()
        with self.assertRaises(RetrievalDenied):
            self.hg.promote()
        with self.assertRaises(RetrievalProvenanceError):
            self.hg.add_fact("a", "hypernym", "z", None)
        with self.assertRaises(RetrievalProvenanceError):
            self.hg.add_fact("a", "hypernym", "z", "")
        with self.assertRaises(RetrievalBoundsError):
            self.hg.add_facts(
                [("x", "r", "y")] * (MAX_BATCH + 1),
                provenance="round032:too-large")
        with self.assertRaises(RetrievalBoundsError):
            self.hg.snapshot(limit=0)
        with self.assertRaises(RetrievalBoundsError):
            self.hg.neighbors("a", limit=0)
        with self.assertRaises(RetrievalBoundsError):
            self.hg.two_hop("a", limit=0)

    def test_batch_limit_rejects_before_writing(self):
        before = self.relations.status()
        with self.assertRaises(RetrievalBoundsError):
            self.hg.add_facts(
                [("x", "r", "y")] * (MAX_BATCH + 1),
                provenance="round032:too-large")
        self.assertEqual(self.relations.status(), before)
        self.assertEqual(self.leaves.count, 0)

    def test_leaf_collision_is_denied_before_partial_writes(self):
        prepared = self.hg._prepare_fact("x", "rel", "y", "prov")
        conflicting = (
            b'{"kind":"concept","concept":"different","schema":'
            b'"nanlm.resident-retrieval-concept/1"}')
        self.leaves.insert(
            "concept", leaf_id=prepared["source_leaf"],
            content=conflicting,
            payload_digest=hashlib.sha256(conflicting).digest())
        before = self.relations.status()
        before_leaves = self.leaves.count
        with self.assertRaises(RetrievalConflictError):
            self.hg.add_fact("x", "rel", "y", "prov")
        self.assertEqual(self.relations.status(), before)
        self.assertEqual(self.leaves.count, before_leaves)
        self.assertEqual(len(self.hg), 0)

    def test_capacity_exhaustion_is_denied_before_partial_writes(self):
        relations = RelationStore.allocate(
            relation_capacity=1, hyperedge_capacity=1, arena_bytes=8192)
        _, leaves = make_leaf_store(capacity=32)
        hg = RetrievalHypergraph(relations=relations, leaves=leaves)
        hg.add_fact(1, 1, 2, "prov")
        before_status = relations.status()
        before_leaves = leaves.count
        with self.assertRaises(RetrievalBoundsError):
            hg.add_fact(3, 1, 4, "prov")
        self.assertEqual(relations.status(), before_status)
        self.assertEqual(leaves.count, before_leaves)

    def test_audit_detects_malformed_retrieval_hyperedge(self):
        self.add()
        # Append a foreign edge that carries the retrieval member kind but is
        # structurally incomplete.  The audit must report it instead of
        # silently treating it as a valid fact.
        self.relations.append_relation(
            subject=b"\x01" * 32,
            predicate=self.hg._prepare_fact(1, 1, 2, "p")["predicate_ref"],
            object=b"\x02" * 32,
            source_digest=hashlib.sha256(b"foreign").digest(),
            relation_id=hashlib.sha256(b"foreign-relation").digest(),
            hyperedge_id=hashlib.sha256(b"foreign-edge").digest(),
            hyperedge_members=[(MEMBER_FACT, b"\x03" * 32)],
        )
        audit = self.hg.audit(limit=8)
        self.assertFalse(audit["ok"])
        self.assertTrue(any("incomplete" in item
                            or "malformed" in item
                            for item in audit["problems"]))
        assert_json_safe(self, audit)


class FakeUniverseTests(unittest.TestCase):
    def test_fake_facade_persists_and_reopens(self):
        universe = FakeUniverse()
        hg = RetrievalHypergraph(universe)
        receipt = hg.add_fact("a", "hypernym", "b", "fake:source")
        self.assertTrue(receipt["created"])
        self.assertEqual(hg.get("a", "hypernym", "b")["provenance"],
                         "fake:source")
        self.assertTrue(hg.audit()["ok"])

        reopened = hg.reopen()
        self.assertIsInstance(reopened, RetrievalHypergraph)
        fact = reopened.get("a", "hypernym", "b")
        self.assertEqual(fact["provenance"], "fake:source")
        self.assertEqual(fact["source"], "a")
        self.assertEqual(fact["destination"], "b")
        self.assertEqual(reopened.snapshot(limit=4)["fact_count"], 1)
        self.assertTrue(reopened.audit()["ok"])
        self.assertEqual(universe.texts, {
            key: value for key, value in universe.texts.items()})
        self.assertTrue(all(isinstance(value, str)
                            for value in universe.texts.values()))


class RealUniverseTests(unittest.TestCase):
    def setUp(self):
        if ResidentUniverse is None:
            self.skipTest("resident universe siblings are unavailable")
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "universe.nlm"
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)
        self.hg = RetrievalHypergraph(self.universe)

    def tearDown(self):
        if getattr(self, "universe", None) is not None:
            self.universe.close()
        self._tmp.cleanup()

    def reopen(self):
        self.universe.close()
        self.universe = ResidentUniverse.open(self.path, SIZE_16MIB)
        self.hg = RetrievalHypergraph(self.universe)
        return self.hg

    def test_round032_sample_persists_and_reopens(self):
        sample = read_round032_sample(limit=12)
        if sample is None:
            self.skipTest("Round 032 facts sample is unavailable")
        provenance = "experiments/round032/data/facts.tsv"
        receipt = self.hg.add_facts(sample, provenance=provenance)
        self.assertEqual(receipt["accepted"], len(sample))
        self.assertEqual(receipt["duplicates"], 0)
        self.hg.add_fact("chain:a", 1, "chain:b", provenance)
        self.hg.add_fact("chain:b", 1, "chain:c", provenance)

        first = sample[0]
        fact = self.hg.get(*first)
        self.assertIsNotNone(fact)
        self.assertEqual(fact["provenance"], provenance)
        self.assertEqual(fact["source"], str(first[0]))
        self.assertEqual(fact["relation"], str(first[1]))
        self.assertEqual(fact["destination"], str(first[2]))
        self.assertTrue(fact["resolved"])

        snapshot = self.hg.snapshot(limit=5)
        self.assertEqual(snapshot["fact_count"], 5)
        self.assertTrue(snapshot["truncated"])
        self.assertLessEqual(snapshot["scanned_relations"],
                             snapshot["scan_limit"])
        assert_json_safe(self, snapshot)
        self.assertTrue(self.hg.audit(limit=64)["ok"])

        reopened = self.reopen()
        self.assertEqual(reopened.get(*first)["provenance"], provenance)
        self.assertEqual(reopened.get("chain:a", 1, "chain:b")["destination"],
                         "chain:b")
        paths = reopened.two_hop("chain:a", (1, 1))
        self.assertEqual([(path["via"], path["destination"])
                          for path in paths], [("chain:b", "chain:c")])
        self.assertEqual(reopened.snapshot(limit=2)["fact_count"], 2)
        self.assertTrue(reopened.audit(limit=64)["ok"])

    def test_50k_corpus_is_bounded_with_a_small_sample(self):
        if not ROUND032_FACTS.exists():
            self.skipTest("Round 032 facts corpus is unavailable")
        with ROUND032_FACTS.open(encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        self.assertGreater(line_count, 50_000)
        sample = read_round032_sample(limit=10)
        receipt = self.hg.add_facts(
            sample, provenance="experiments/round032/data/facts.tsv")
        self.assertEqual(receipt["accepted"], len(sample))

        snapshot = self.hg.snapshot(limit=4)
        self.assertEqual(snapshot["fact_count"], 4)
        self.assertEqual(snapshot["relation_count"], len(sample))
        self.assertTrue(snapshot["truncated"])
        self.assertLessEqual(snapshot["scanned_relations"],
                             snapshot["scan_limit"])
        self.assertLess(snapshot["scanned_relations"], line_count)
        self.assertLessEqual(snapshot["scan_limit"],
                             4 * 4 + 8)
        audit = self.hg.audit(limit=5)
        self.assertLessEqual(audit["scanned_relations"], 5)
        self.assertTrue(audit["truncated"])
        self.assertEqual(audit["relation_count"], len(sample))


if __name__ == "__main__":
    unittest.main()
