"""Tests for the complete 32,385-cycle typed hypergraph.

The primary path uses real resident relation, leaf and cell stores.  A small
duck-typed fake pair is also exercised so the adapter remains usable with
partial resident installations.
"""
from __future__ import annotations

import hashlib
import unittest

from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.full_cycle_hypergraph import (
    FULL_CYCLE_COUNT,
    MEMBER_CONCEPT,
    MEMBER_CYCLE,
    MEMBER_HEAD,
    MEMBER_PAIR,
    MEMBER_PARENT,
    MEMBER_PAYLOAD,
    MEMBER_VALUE,
    Composition,
    FullCycleBoundsError,
    FullCycleConflictError,
    FullCycleDenied,
    FullCycleHypergraph,
    FullCycleNotFound,
    FullCycleParentCycleError,
    FullCycleProvenanceError,
    TypedMembers,
    concept_leaf_id,
    cycle_edges,
    cycle_from_index,
    cycle_hyperedge_id,
    cycle_id,
    cycle_index,
    full_cycle_ids,
    normalize_cycle,
    pair_from_index,
    pair_reference,
    root_cycle_count,
    root_cycle_pairs,
)
from nanlmind_hermes.resident.leaves import (
    HEADER_BYTES as LEAF_HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    RECORD_BYTES,
    LeafStore,
)
from nanlmind_hermes.resident.relations import RelationStore


def make_leaf_store(capacity=64, base_offset=0):
    index_capacity = 1
    while index_capacity < max(16, capacity * 2):
        index_capacity <<= 1
    buffer = bytearray(
        base_offset + LEAF_HEADER_BYTES + capacity * RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)
    return buffer, LeafStore.create(
        buffer, base_offset=base_offset, capacity=capacity)


class CycleIdentityTests(unittest.TestCase):
    def test_full_cycle_count_is_32385_not_256(self):
        self.assertEqual(FULL_CYCLE_COUNT, 32385)
        self.assertEqual(root_cycle_count(256), 32385)
        self.assertEqual(root_cycle_count(8), 21)
        relations = RelationStore.allocate(
            relation_capacity=16, hyperedge_capacity=16, arena_bytes=8192)
        _, leaves = make_leaf_store(8)
        hg = FullCycleHypergraph(relations=relations, leaves=leaves)
        self.assertEqual(hg.full_cycle_count, 32385)
        self.assertEqual(hg.cycle_capacity, 32385)
        self.assertNotEqual(hg.full_cycle_count, 256)
        self.assertFalse(hg.canonical_merge)
        self.assertEqual(len(root_cycle_pairs(limit=5)), 5)
        self.assertEqual(root_cycle_pairs(limit=5),
                         ((1, 2), (1, 3), (1, 4), (1, 5), (1, 6)))

    def test_exhaustive_cycle_id_mapping(self):
        total = 32385
        seen_ids: set[str] = set()
        seen_edges: set[bytes] = set()
        for index in range(total):
            first, second = pair_from_index(index)
            self.assertLess(first, second)
            self.assertEqual(cycle_index(first, second), index)
            canonical = cycle_from_index(index)
            self.assertEqual(canonical, (0, first, second))
            self.assertEqual(normalize_cycle((second, first)), canonical)
            identifier = cycle_id((first, second))
            self.assertEqual(identifier, "0:%d:%d" % (first, second))
            self.assertEqual(cycle_id(canonical), identifier)
            self.assertNotIn(identifier, seen_ids)
            seen_ids.add(identifier)
            edge = cycle_hyperedge_id(canonical)
            self.assertEqual(len(edge), 32)
            self.assertNotIn(edge, seen_edges)
            seen_edges.add(edge)
            edges = cycle_edges(canonical)
            self.assertEqual(len(edges), 3)
            self.assertEqual(set(edges), {
                (0, first), (first, second), (0, second)})
            self.assertEqual(len({pair_reference(*edge) for edge in edges}),
                             3)
        self.assertEqual(len(seen_ids), total)
        self.assertEqual(len(full_cycle_ids()), total)

    def test_cycle_identity_formula_is_stable(self):
        canonical = (0, 1, 2)
        expected = hashlib.sha256(
            b"NLMR-CYCLE-EDGE\0" + b"0:1:2").digest()
        self.assertEqual(cycle_hyperedge_id(canonical), expected)
        self.assertEqual(cycle_hyperedge_id((2, 1, 0)), expected)
        self.assertEqual(concept_leaf_id("Geometry"),
                         concept_leaf_id("geometry"))


class RealStoreCycleTests(unittest.TestCase):
    def setUp(self):
        self.source = hashlib.sha256(b"full-cycle-test-source").hexdigest()
        self.relations = RelationStore.allocate(
            relation_capacity=256, hyperedge_capacity=256,
            chain_capacity=2048, arena_bytes=262144)
        self.leaf_buffer, self.leaves = make_leaf_store(capacity=64)
        self.hg = FullCycleHypergraph(
            relations=self.relations, leaves=self.leaves)

    def add(self, i=1, j=2, value=7, source=None, payload=None):
        if payload is None:
            payload = {"value": value}
        return self.hg.add_cycle(
            i, j, payload, self.source if source is None else source)

    def test_cycle_has_typed_members_and_no_duplicate_storage(self):
        receipt = self.add(1, 2, 7)
        self.assertEqual(receipt["status"], "appended")
        self.assertTrue(receipt["created"])
        self.assertEqual(receipt["cycle"], (0, 1, 2))
        self.assertEqual(receipt["cycle_id"], "0:1:2")
        self.assertEqual(receipt["value"], 7)
        self.assertEqual(receipt["full_cycle_count"], 32385)
        self.assertEqual(receipt["source_digest"], self.source)
        self.assertEqual(receipt["hyperedge_id"],
                         cycle_hyperedge_id((0, 1, 2)).hex())

        status = self.relations.status()
        self.assertEqual(status["hyperedges"], 1)
        self.assertEqual(status["relations"], 1)
        self.assertEqual(self.leaves.count, 0)

        members = self.hg.members((0, 1, 2))
        self.assertIsInstance(members, TypedMembers)
        self.assertFalse(members.missing)
        self.assertEqual(members.value, 7)
        self.assertEqual(members["value"], 7)
        self.assertEqual(members.cycle, (0, 1, 2))
        self.assertEqual(members.cycle_id, "0:1:2")
        kinds = [member["kind"] for member in members]
        self.assertEqual(kinds.count("cycle"), 1)
        self.assertEqual(kinds.count("pair"), 3)
        self.assertEqual(kinds.count("head"), 3)
        self.assertEqual(kinds.count("value"), 1)
        self.assertEqual(kinds.count("payload"), 1)
        self.assertEqual(len(members), 9)
        self.assertEqual(tuple(member["pair"] for member in members
                               if member["kind"] == "pair"),
                         ((0, 1), (1, 2), (0, 2)))
        self.assertEqual(tuple(member["head"] for member in members
                               if member["kind"] == "head"), (0, 1, 2))
        self.assertEqual(members.pair_refs,
                         tuple(pair_reference(*edge)
                               for edge in cycle_edges((0, 1, 2))))
        codes = [member["kind_code"] for member in members]
        self.assertEqual(codes.count(MEMBER_CYCLE), 1)
        self.assertEqual(codes.count(MEMBER_PAIR), 3)
        self.assertEqual(codes.count(MEMBER_HEAD), 3)
        self.assertEqual(codes.count(MEMBER_VALUE), 1)
        self.assertEqual(codes.count(MEMBER_PAYLOAD), 1)

        # Only references are stored: no pair cell was created or written.
        hyperedge = self.relations.get_hyperedge(
            cycle_hyperedge_id((0, 1, 2)))
        self.assertEqual(hyperedge["arity"], 9)
        self.assertIn("0000000000000000000000000000000000000000000000000000000000000080",
                      [member["id"] for member in hyperedge["members"]])
        self.assertEqual(self.relations.status()["arena_used"] > 0, True)

    def test_real_cells_are_not_modified(self):
        cell_buffer = bytearray(REGION_BYTES)
        cells = ResidentCells.format_region(cell_buffer)
        hg = FullCycleHypergraph(
            relations=self.relations, leaves=self.leaves, cells=cells)
        hg.add_cycle(3, 5, {"value": 42}, self.source)
        self.assertEqual(cells.generation, 0)
        self.assertEqual(cells.read_rail(0, 3).weight, 0)
        self.assertEqual(hg.members((0, 3, 5)).value, 42)

    def test_idempotency_conflict_and_provenance_denial(self):
        first = self.add(1, 2, 7)
        before_status = self.relations.status()
        before_hyperedge = self.relations.get_hyperedge(
            first["hyperedge_id"])
        duplicate = self.add(2, 1, 7)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["cycle"], (0, 1, 2))
        self.assertEqual(duplicate["hyperedge_id"], first["hyperedge_id"])
        self.assertEqual(self.relations.status(), before_status)
        self.assertEqual(
            self.relations.get_hyperedge(first["hyperedge_id"]),
            before_hyperedge)

        other_source = hashlib.sha256(b"other-source").hexdigest()
        sourced = self.add(2, 1, 7, source=other_source)
        self.assertTrue(sourced["duplicate"])
        self.assertTrue(sourced["source_mismatch"])
        self.assertEqual(sourced["source_digest"], self.source)

        for bad_payload in ({"value": 8}, 8):
            with self.assertRaises(FullCycleConflictError):
                self.hg.add_cycle(2, 1, bad_payload, self.source)
        with self.assertRaises(FullCycleConflictError):
            self.hg.add_cycle(
                1, 2, {"value": 7, "label": "different"}, self.source)
        self.assertEqual(self.relations.status(), before_status)
        self.assertTrue(self.hg.audit()["ok"])

        with self.assertRaises(FullCycleProvenanceError):
            self.hg.add_cycle(1, 3, 1, None)
        with self.assertRaises(FullCycleProvenanceError):
            self.hg.add_cycle(1, 3, 1, "")
        with self.assertRaises((FullCycleBoundsError, ValueError)):
            self.hg.add_cycle(1, 1, 1, self.source)
        with self.assertRaises((FullCycleBoundsError, ValueError)):
            self.hg.add_cycle(0, 1, 1, self.source)
        with self.assertRaises((FullCycleBoundsError, ValueError)):
            self.hg.add_cycle(1, 256, 1, self.source)
        with self.assertRaises((FullCycleBoundsError, ValueError)):
            self.hg.add_cycle(True, 2, 1, self.source)

    def test_concepts_links_and_concept_parents(self):
        for index, (i, j) in enumerate(((1, 2), (2, 3), (1, 3)), start=1):
            self.add(i, j, index)
        parent = self.hg.add_concept(
            "shape", {"name": "shape"}, self.source)
        child = self.hg.add_concept(
            "triangle", {"name": "triangle"}, self.source, parent="shape")
        self.assertEqual(parent["status"], "appended")
        self.assertEqual(child["status"], "appended")
        self.assertEqual(child["parent_concept_id"],
                         parent["concept_id"])
        self.assertEqual(self.leaves.count, 2)
        self.assertIsNotNone(self.leaves.find(
            bytes.fromhex(parent["concept_id"])))
        self.assertEqual(
            self.leaves.find(bytes.fromhex(child["concept_id"])).kind,
            "concept")

        link = self.hg.link(
            "triangle", [(0, 1, 2), (0, 1, 3)], self.source)
        self.assertEqual(link["status"], "appended")
        self.assertEqual(link["owner_kind"], "concept")
        self.assertEqual(set(link["cycle_ids"]), {"0:1:2", "0:1:3"})
        members = self.hg.members("triangle")
        self.assertEqual(members.concept_id, child["concept_id"])
        self.assertEqual(set(members.cycle_ids), {
            cycle_hyperedge_id((0, 1, 2)), cycle_hyperedge_id((0, 1, 3))})
        self.assertEqual([member["kind"] for member in members].count(
            "concept"), 2)

        duplicate = self.hg.link(
            "triangle", [(0, 1, 3), (0, 1, 2)], self.source)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["hyperedge_id"], link["hyperedge_id"])

        with self.assertRaises(FullCycleParentCycleError):
            self.hg.add_concept("selfish", {"x": 1}, self.source,
                                parent="selfish")
        with self.assertRaises(FullCycleParentCycleError):
            self.hg.link("triangle", ["shape"], self.source)
        with self.assertRaises(FullCycleParentCycleError):
            self.hg.link("triangle", ["triangle"], self.source)
        with self.assertRaises(FullCycleNotFound):
            self.hg.link("triangle", [(0, 200, 201)], self.source)
        with self.assertRaises(FullCycleConflictError):
            self.hg.add_concept("shape", {"name": "different"}, self.source)
        self.assertTrue(self.hg.audit()["ok"])

    def test_parent_composition_is_recursive_and_never_canonical_merge(self):
        for index, (i, j) in enumerate(
                ((1, 2), (2, 3), (1, 3), (3, 4)), start=1):
            self.add(i, j, index)
        before_cycles = self.hg.status()["attached_cycle_count"]
        before_values = {
            cycle: self.hg.members(cycle).value
            for cycle in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (0, 3, 4))}

        parent = self.hg.compose(
            (0, 1, 2), (0, 2, 3), source=self.source)
        self.assertIsInstance(parent, Composition)
        self.assertEqual(parent["status"], "composed")
        self.assertTrue(parent["created"])
        self.assertEqual(parent["value"], 1 + 2)
        self.assertEqual(int(parent), 3)
        self.assertEqual(set(parent["cycle_ids"]), {"0:1:2", "0:2:3"})
        self.assertFalse(parent["canonical_merge"])
        self.assertTrue(parent["stored"])
        parent_members = self.hg.members(parent["hyperedge_id"])
        self.assertEqual(
            [member["kind"] for member in parent_members],
            ["cycle", "cycle", "value", "payload"])

        nested = self.hg.compose(
            (0, 1, 3), (0, 3, 4), source=self.source,
            parent=parent["hyperedge_id"])
        self.assertEqual(nested["status"], "composed")
        self.assertEqual(nested["parent_id"], parent["hyperedge_id"])
        nested_members = self.hg.members(
            nested["hyperedge_id"], include_parents=True)
        self.assertIn("parent", [member["kind"]
                                 for member in nested_members])
        self.assertIn("cycle", [member["kind"]
                                for member in nested_members])
        self.assertEqual(nested["value"], 3 + 4)

        duplicate = self.hg.compose(
            (0, 2, 3), (0, 1, 2), source=self.source)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["hyperedge_id"], parent["hyperedge_id"])

        idempotent = self.hg.compose((0, 1, 2), source=self.source)
        self.assertEqual(idempotent["status"], "idempotent")
        self.assertFalse(idempotent["stored"])

        with self.assertRaises(FullCycleDenied):
            self.hg.compose((0, 1, 2), (0, 3, 4), source=self.source,
                            require_shared=True)
        with self.assertRaises(FullCycleNotFound):
            self.hg.compose((0, 100, 101), source=self.source)
        with self.assertRaises(FullCycleParentCycleError):
            self.hg.link(
                nested["hyperedge_id"], [parent["hyperedge_id"]],
                self.source, parent_edge=parent["hyperedge_id"])
        with self.assertRaises(FullCycleParentCycleError):
            self.hg.link(
                nested["hyperedge_id"], [parent["hyperedge_id"]],
                self.source, parent_edge=nested["hyperedge_id"])

        # Composition only added parent/link hyperedges.  It never created a
        # merged cycle record and never changed a source value.
        self.assertEqual(self.hg.status()["attached_cycle_count"],
                         before_cycles)
        for cycle, value in before_values.items():
            self.assertEqual(self.hg.members(cycle).value, value)
        self.assertEqual(
            set(parent["cycles"]), {(0, 1, 2), (0, 2, 3)})
        self.assertTrue(self.hg.audit()["ok"])

    def test_bounded_recursive_neighbors(self):
        cycles = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
        for index, (i, j) in enumerate(cycles, start=1):
            self.add(i, j, index)
        self.hg.add_concept("star", {"x": 1}, self.source)
        self.hg.link("star", [(0, 1, 2), (0, 3, 4)], self.source)
        parent = self.hg.compose(
            (0, 1, 2), (0, 2, 3), source=self.source)

        first = self.hg.neighbors((0, 1, 2), hops=1, limit=32)
        self.assertFalse(first.missing)
        self.assertFalse(first.truncated)
        self.assertIn((0, 1, 3), first)
        self.assertIn((0, 1, 4), first)
        self.assertIn((0, 2, 3), first)
        self.assertIn((0, 2, 4), first)
        self.assertIn((0, 3, 4), first)
        self.assertIn(concept_leaf_id("star").hex(), first.concept_ids)
        self.assertIn(parent["hyperedge_id"], first.parent_ids)

        bounded = self.hg.neighbors((0, 1, 2), hops=1, limit=2)
        self.assertEqual(len(bounded), 2)
        self.assertTrue(bounded.truncated)
        self.assertLessEqual(bounded.edges_visited, 32)
        self.assertEqual(len(set(bounded)), len(bounded))

        way = self.hg.neighbors((0, 1, 2), hops=2, limit=32)
        self.assertIn((0, 3, 4), way)
        self.assertEqual(len(set(way)), len(way))
        self.assertLessEqual(len(way), 32)

        missing = self.hg.neighbors((0, 100, 101), hops=1, limit=4)
        self.assertTrue(missing.missing)
        self.assertEqual(list(missing), [])
        for hops, limit in ((0, 1), (9, 1), (1, 0), (1, 513)):
            with self.assertRaises(FullCycleBoundsError):
                self.hg.neighbors((0, 1, 2), hops=hops, limit=limit)

    def test_snapshot_and_audit_report_full_count_without_truncation(self):
        for index, (i, j) in enumerate(((1, 2), (1, 3), (2, 3)), start=1):
            self.add(i, j, index)
        audit = self.hg.audit()
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["full_cycle_count"], 32385)
        self.assertNotEqual(audit["full_cycle_count"], 256)
        self.assertEqual(audit["attached_cycle_count"], 3)
        self.assertEqual(audit["missing_cycle_count"], 32382)
        self.assertFalse(audit["full_coverage"])
        self.assertFalse(audit["truncated"])
        self.assertTrue(audit["bounded"])
        self.assertFalse(audit["canonical_merge"])
        self.assertTrue(audit["no_duplicate_storage"])
        self.assertEqual(audit["scanned_hyperedges"], 3)

        snapshot = self.hg.snapshot()
        self.assertEqual(snapshot["full_cycle_count"], 32385)
        self.assertEqual(snapshot["attached_cycle_count"], 3)
        self.assertEqual(snapshot["returned"], 3)
        self.assertFalse(snapshot["truncated"])
        self.assertTrue(snapshot["bounded"])
        self.assertEqual(snapshot["cycles"][0]["cycle_index"], 0)

        bounded = self.hg.snapshot(limit=2)
        self.assertEqual(bounded["returned"], 2)
        self.assertTrue(bounded["truncated"])
        with self.assertRaises(FullCycleBoundsError):
            self.hg.snapshot(limit=0)
        with self.assertRaises(FullCycleBoundsError):
            self.hg.audit(limit=0)
        partial = self.hg.audit(limit=1)
        self.assertTrue(partial["truncated"])
        self.assertEqual(partial["scanned_hyperedges"], 1)

    def test_capacity_exhaustion_is_denied_before_partial_writes(self):
        relations = RelationStore.allocate(
            relation_capacity=1, hyperedge_capacity=1,
            chain_capacity=256, arena_bytes=8192)
        _, leaves = make_leaf_store(capacity=8)
        hg = FullCycleHypergraph(relations=relations, leaves=leaves)
        hg.add_cycle(1, 2, {"value": 1}, self.source)
        before_status = relations.status()
        before_leaf_count = leaves.count
        with self.assertRaises(FullCycleBoundsError):
            hg.add_cycle(1, 3, {"value": 2}, self.source)
        self.assertEqual(relations.status(), before_status)
        self.assertEqual(leaves.count, before_leaf_count)

    def test_audit_detects_tampered_cycle_reference(self):
        receipt = self.add(1, 2, 7)
        view = self.relations.get_hyperedge(receipt["hyperedge_id"])
        members_offset = view["members_offset"]
        arena_start = self.relations._layout.arena + members_offset
        # The second member is the first pair reference (0,1).
        self.relations._view[arena_start + 32:arena_start + 64] = \
            (0xDEAD).to_bytes(32, "big")
        audit = self.hg.audit()
        self.assertFalse(audit["ok"])
        self.assertTrue(any("pair references" in problem
                            for problem in audit["problems"]))

    def test_reopen_preserves_deterministic_ids_and_full_count(self):
        for index, (i, j) in enumerate(((1, 2), (2, 3), (1, 3)), start=1):
            self.add(i, j, index)
        reopened = self.hg.reopen()
        self.assertEqual(reopened.full_cycle_count, 32385)
        self.assertEqual(reopened.members((0, 2, 3)).value, 2)
        self.assertEqual(reopened.members((0, 1, 3)).value, 3)
        self.assertEqual(reopened.members((0, 2, 3)).hyperedge_id,
                         cycle_hyperedge_id((0, 2, 3)).hex())
        self.assertTrue(reopened.audit()["ok"])

    def test_canonical_merge_is_disabled(self):
        self.add(1, 2, 1)
        with self.assertRaises(FullCycleDenied):
            self.hg.merge((0, 1, 2), (0, 2, 3))
        with self.assertRaises(FullCycleDenied):
            self.hg.promote((0, 1, 2))
        self.assertFalse(self.hg.canonical_merge)


class FakeStoreTests(unittest.TestCase):
    class FakeRelations:
        def __init__(self):
            self.hyperedges = {}
            self.generation = 0

        def append_hyperedge(self, hyperedge_id, source_digest, members):
            key = hyperedge_id.hex() if isinstance(
                hyperedge_id, (bytes, bytearray, memoryview)) \
                else str(hyperedge_id)
            if key in self.hyperedges:
                return {"created": False, "duplicate": True,
                        "hyperedge_id": key}
            self.hyperedges[key] = {
                "hyperedge_id": key,
                "source_digest": source_digest.hex()
                if isinstance(source_digest, (bytes, bytearray, memoryview))
                else str(source_digest),
                "members": [
                    {"kind_code": kind,
                     "id": identifier.hex()
                     if isinstance(identifier,
                                   (bytes, bytearray, memoryview))
                     else str(identifier)}
                    for kind, identifier in members
                ],
            }
            self.generation += 1
            return {"created": True, "duplicate": False,
                    "hyperedge_id": key}

        def get_hyperedge(self, identifier):
            key = identifier.hex() if isinstance(
                identifier, (bytes, bytearray, memoryview)) \
                else str(identifier)
            return self.hyperedges.get(key)

        def status(self):
            return {
                "hyperedges": len(self.hyperedges),
                "hyperedge_capacity": 256,
                "relations": 0,
                "relation_capacity": 256,
                "generation": self.generation,
            }

    class FakeLeafRecord:
        def __init__(self, leaf_id, payload_digest, parent=None):
            self.leaf_id = bytes(leaf_id)
            self.payload_digest = bytes(payload_digest)
            self.kind = "concept"
            self.parent = parent

        def to_dict(self):
            return {"leaf_id": self.leaf_id.hex(),
                    "payload_digest": self.payload_digest.hex(),
                    "kind": self.kind}

    class FakeLeaves:
        def __init__(self):
            self.records = {}

        def find(self, leaf_id):
            key = bytes(leaf_id).hex() if isinstance(
                leaf_id, (bytes, bytearray, memoryview)) else str(leaf_id)
            return self.records.get(key)

        def insert(self, kind, leaf_id=None, payload_digest=None, **kwargs):
            key = bytes(leaf_id).hex()
            record = FakeStoreTests.FakeLeafRecord(
                leaf_id, payload_digest, kwargs.get("parent"))
            self.records[key] = record
            return record

        def __len__(self):
            return len(self.records)

    def setUp(self):
        self.relations = self.FakeRelations()
        self.leaves = self.FakeLeaves()
        self.hg = FullCycleHypergraph({
            "relations": self.relations,
            "leaves": self.leaves,
        })
        self.source = hashlib.sha256(b"fake-source").hexdigest()

    def test_fake_store_round_trip(self):
        first = self.hg.add_cycle(1, 2, {"value": 9}, self.source)
        self.assertTrue(first["created"])
        second = self.hg.add_cycle(2, 3, {"value": 4}, self.source)
        self.assertTrue(second["created"])
        self.assertEqual(self.hg.members((0, 2, 1)).value, 9)
        self.assertEqual(len(self.hg.members((0, 2, 1))), 9)

        concept = self.hg.add_concept("fake", {"x": 1}, self.source)
        self.assertTrue(concept["created"])
        link = self.hg.link("fake", [(0, 1, 2), (0, 2, 3)], self.source)
        self.assertTrue(link["created"])
        self.assertEqual(set(self.hg.members("fake").cycle_ids),
                         {cycle_hyperedge_id((0, 1, 2)),
                          cycle_hyperedge_id((0, 2, 3))})

        composed = self.hg.compose(
            (0, 1, 2), (0, 2, 3), source=self.source)
        self.assertEqual(composed["status"], "composed")
        self.assertEqual(int(composed), 13)
        self.assertEqual(
            set(self.hg.members(composed["hyperedge_id"]).cycle_ids),
            {cycle_hyperedge_id((0, 1, 2)),
             cycle_hyperedge_id((0, 2, 3))})

        neighbors = self.hg.neighbors((0, 1, 2), hops=1, limit=8)
        self.assertIn((0, 2, 3), neighbors)
        self.assertFalse(neighbors.missing)

        duplicate = self.hg.add_cycle(2, 1, {"value": 9}, self.source)
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(FullCycleConflictError):
            self.hg.add_cycle(1, 2, {"value": 10}, self.source)
        self.assertTrue(self.hg.audit()["ok"])
        snapshot = self.hg.snapshot()
        self.assertEqual(snapshot["cycle_count"], 2)
        self.assertEqual(snapshot["full_cycle_count"], 32385)


class FullSpaceAttachTests(unittest.TestCase):
    def test_all_32385_cycles_are_attached_and_verified(self):
        hg = FullCycleHypergraph()
        source = hashlib.sha256(b"full-space-source").hexdigest()
        summary = hg.add_all_cycles(source)
        self.assertEqual(summary["full_cycle_count"], 32385)
        self.assertEqual(summary["requested"], 32385)
        self.assertFalse(summary["truncated"])
        self.assertEqual(summary["added"], 32385)
        self.assertEqual(summary["duplicates"], 0)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["attached_cycle_count"], 32385)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["status"], "complete")

        status = hg.status()
        self.assertEqual(status["full_cycle_count"], 32385)
        self.assertEqual(status["attached_cycle_count"], 32385)
        self.assertEqual(status["missing_cycle_count"], 0)
        self.assertEqual(status["hyperedge_count"], 32385)
        self.assertEqual(status["leaf_count"], 0)

        coverage = hg.verify_full_coverage()
        self.assertTrue(coverage["covered"])
        self.assertEqual(coverage["full_cycle_count"], 32385)
        self.assertEqual(coverage["attached_cycle_count"], 32385)
        self.assertEqual(coverage["missing_cycle_count"], 0)
        self.assertFalse(coverage["truncated"])

        audit = hg.audit(require_full=True)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["problems"], [])
        self.assertEqual(audit["full_cycle_count"], 32385)
        self.assertNotEqual(audit["full_cycle_count"], 256)
        self.assertEqual(audit["attached_cycle_count"], 32385)
        self.assertEqual(audit["cycles_checked"], 32385)
        self.assertTrue(audit["full_coverage"])
        self.assertEqual(audit["missing_cycle_count"], 0)
        self.assertEqual(audit["duplicate_cycle_count"], 0)
        self.assertFalse(audit["truncated"])
        self.assertEqual(audit["scanned_hyperedges"], 32385)

        snapshot = hg.snapshot(limit=32385)
        self.assertEqual(snapshot["returned"], 32385)
        self.assertEqual(snapshot["full_cycle_count"], 32385)
        self.assertEqual(snapshot["attached_cycle_count"], 32385)
        self.assertFalse(snapshot["truncated"])
        self.assertEqual(len(snapshot["cycles"]), 32385)
        self.assertEqual(len({entry["cycle_id"]
                              for entry in snapshot["cycles"]}), 32385)
        self.assertEqual(snapshot["cycles"][0]["cycle"], (0, 1, 2))
        self.assertEqual(snapshot["cycles"][-1]["cycle"], (0, 254, 255))

        # Spot-check the mapping at both ends and in the middle.
        for index in (0, 1, 16192, 32383, 32384):
            cycle = cycle_from_index(index)
            members = hg.members(cycle)
            self.assertFalse(members.missing)
            self.assertEqual(members.cycle, cycle)
            self.assertEqual(members.hyperedge_id,
                             cycle_hyperedge_id(cycle).hex())
            self.assertEqual(members.cycle_index if hasattr(
                members, "cycle_index") else index, index)


if __name__ == "__main__":
    unittest.main()
