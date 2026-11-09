"""Resident cycle-observable hypergraph tests.

The primary path uses the real relation, leaf and cell stores.  A small
duck-typed fake store is also exercised so the layer remains usable with
partial resident installations.
"""
from __future__ import annotations

import hashlib
import unittest

from nanlmind_hermes.resident.cells import (
    REGION_BYTES,
    ResidentCells,
    canonical_address,
    mirror_address,
)
from nanlmind_hermes.resident.cycle_hypergraph import (
    MEMBER_CYCLE,
    MEMBER_PAYLOAD,
    MEMBER_PAIR,
    MEMBER_VALUE,
    MEMBER_VERTEX,
    CycleBoundsError,
    CycleComposition,
    CycleConflictError,
    CycleDenied,
    CycleHypergraph,
    CycleNotFound,
    CycleProvenanceError,
    cycle_edges,
    cycle_hyperedge_id,
    cycle_leaf_id,
    normalize_cycle,
    pair_reference,
)
from nanlmind_hermes.resident.leaves import (
    HEADER_BYTES as LEAF_HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    RECORD_BYTES,
    LeafStore,
)
from nanlmind_hermes.resident.relations import RelationStore


def make_leaf_store(capacity=32, base_offset=0):
    index_capacity = 1
    while index_capacity < max(16, capacity * 2):
        index_capacity <<= 1
    buffer = bytearray(
        base_offset + LEAF_HEADER_BYTES + capacity * RECORD_BYTES
        + index_capacity * INDEX_ENTRY_BYTES)
    return buffer, LeafStore.create(
        buffer, base_offset=base_offset, capacity=capacity)


class RealStoreCycleTests(unittest.TestCase):
    def setUp(self):
        self.source = hashlib.sha256(b"cycle-test-source").hexdigest()
        self.relations = RelationStore.allocate(
            relation_capacity=32, hyperedge_capacity=32, arena_bytes=32768)
        self.leaf_buffer, self.leaves = make_leaf_store(capacity=32)
        self.hg = CycleHypergraph(
            relations=self.relations, leaves=self.leaves)

    def add(self, i=1, j=2, value=7, source=None, payload=None):
        if payload is None:
            payload = value
        return self.hg.add_cycle(
            i, j, payload, self.source if source is None else source)

    def test_incidence_uses_existing_pair_references_and_cycle_leaf(self):
        receipt = self.add(1, 2, 7)
        self.assertEqual(receipt["status"], "appended")
        self.assertTrue(receipt["created"])
        self.assertEqual(receipt["cycle"], (0, 1, 2))
        self.assertEqual(receipt["value"], 7)
        self.assertEqual(receipt["source_digest"], self.source)

        # One relation record and one hyperedge were attached; no pair cell
        # was written or duplicated.
        status = self.relations.status()
        self.assertEqual(status["relations"], 1)
        self.assertEqual(status["hyperedges"], 1)
        leaf = self.leaves.find(bytes.fromhex(receipt["leaf_id"]))
        self.assertIsNotNone(leaf)
        self.assertEqual(leaf.kind, "hyperedge")
        self.assertEqual(leaf.payload_digest,
                         bytes.fromhex(receipt["payload_digest"]))
        self.assertEqual(leaf.leaf_id, cycle_leaf_id((0, 1, 2)))

        members = self.hg.cycle_members((0, 1, 2))
        self.assertEqual(len(members), 3)
        self.assertEqual(tuple(member["pair"] for member in members),
                         ((0, 1), (1, 2), (0, 2)))
        self.assertEqual(tuple(member["id_int"] for member in members),
                         tuple(pair_reference(*edge)
                               for edge in cycle_edges((0, 1, 2))))
        self.assertEqual(members.value, 7)
        self.assertEqual(members["value"], 7)
        self.assertEqual(members["source_digest"], self.source)
        self.assertEqual(members["hyperedge_id"],
                         cycle_hyperedge_id((0, 1, 2)).hex())
        self.assertFalse(members.missing)

        hyperedge = self.relations.get_hyperedge(
            cycle_hyperedge_id((0, 1, 2)))
        self.assertIsNotNone(hyperedge)
        self.assertEqual(hyperedge["arity"], 9)
        codes = [member["kind_code"] for member in hyperedge["members"]]
        self.assertEqual(codes.count(MEMBER_PAIR), 3)
        self.assertEqual(codes.count(MEMBER_CYCLE), 1)
        self.assertEqual(codes.count(MEMBER_VALUE), 1)
        self.assertEqual(codes.count(MEMBER_VERTEX), 3)
        self.assertEqual(codes.count(MEMBER_PAYLOAD), 1)

    def test_real_cells_are_references_only_and_pair_index_is_reported(self):
        self.relations = RelationStore.allocate(
            relation_capacity=8, hyperedge_capacity=8, arena_bytes=8192)
        self.leaf_buffer, self.leaves = make_leaf_store(capacity=16)
        cell_buffer = bytearray(REGION_BYTES)
        cells = ResidentCells.format_region(cell_buffer)
        hg = CycleHypergraph(
            relations=self.relations, leaves=self.leaves, cells=cells)
        receipt = hg.add_cycle(3, 5, 42, self.source)
        self.assertEqual(receipt["value"], 42)
        members = hg.cycle_members((0, 3, 5))
        self.assertEqual(tuple(member["pair"] for member in members),
                         ((0, 3), (3, 5), (0, 5)))
        self.assertEqual(members[0]["pair_index"], cells.pair_index(0, 3))
        self.assertEqual(members[0]["address"], canonical_address(0, 3))
        self.assertEqual(members[0]["mirror"],
                         mirror_address(canonical_address(0, 3)))
        # Attaching a cycle never observes/creates pair-cell state.
        self.assertEqual(cells.generation, 0)
        self.assertEqual(cells.read_rail(0, 3).weight, 0)
        leaf = self.leaves.find(bytes.fromhex(receipt["leaf_id"]))
        self.assertEqual(leaf.pair_id, cells.pair_index(0, 3))

    def test_mirror_safe_idempotency_does_not_rewrite_or_commit(self):
        first = self.add(1, 2, 7)
        before_status = self.relations.status()
        before_bytes = self.relations.relation_bytes(0)
        before_leaf_count = self.leaves.count

        duplicate = self.add(2, 1, 7)
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(duplicate["cycle"], (0, 1, 2))
        self.assertEqual(duplicate["hyperedge_id"], first["hyperedge_id"])
        self.assertEqual(duplicate["leaf_id"], first["leaf_id"])
        self.assertEqual(self.relations.relation_bytes(0), before_bytes)
        self.assertEqual(self.relations.status(), before_status)
        self.assertEqual(self.leaves.count, before_leaf_count)
        self.assertEqual(len(self.hg), 1)
        self.assertEqual(self.hg.cycle_members((0, 2, 1)).value, 7)
        self.assertEqual(self.hg.cycle_members((0, 1, 2)),
                         self.hg.cycle_members((2, 1, 0)))

    def test_conflicting_value_or_payload_is_rejected_and_source_is_not(self):
        first = self.add(1, 2, 7)
        before_status = self.relations.status()
        before_bytes = self.relations.relation_bytes(0)
        before_snapshot = self.hg.snapshot()

        cases = (
            (8, self.source),
            ({"value": 7, "label": "different"}, self.source),
        )
        for payload, source in cases:
            with self.assertRaises(CycleConflictError):
                self.hg.add_cycle(2, 1, payload, source)
            self.assertEqual(self.relations.relation_bytes(0), before_bytes)
            self.assertEqual(self.relations.status(), before_status)
            self.assertEqual(self.hg.snapshot(), before_snapshot)

        # New provenance for the same observable is a duplicate, not a
        # conflict; the original source is preserved.
        other_source = hashlib.sha256(b"other-source").hexdigest()
        duplicate = self.hg.add_cycle(2, 1, 7, other_source)
        self.assertTrue(duplicate["duplicate"])
        self.assertTrue(duplicate["source_mismatch"])
        self.assertEqual(self.relations.relation_bytes(0), before_bytes)
        self.assertEqual(self.relations.status(), before_status)
        members = self.hg.cycle_members((0, 1, 2))
        self.assertEqual(members.value, 7)
        self.assertEqual(members.source_digest, first["source_digest"])
        self.assertFalse(self.hg.audit()["problems"])

    def test_payload_mapping_preserves_raw_integer_value(self):
        receipt = self.add(
            1, 2, payload={"value": 11, "label": "triangle", "tags": ["a"]})
        self.assertEqual(receipt["value"], 11)
        members = self.hg.cycle_members((0, 1, 2))
        self.assertEqual(members.value, 11)
        self.assertEqual(len(members.payload_digest), 64)

    def test_invalid_vertices_provenance_values_and_limits_are_denied(self):
        invalid_cycles = (
            (1, 1),
            (0, 1),
            (-1, 2),
            (256, 2),
            (True, 2),
            (1.5, 2),
        )
        for i, j in invalid_cycles:
            with self.assertRaises((CycleBoundsError, ValueError)):
                self.hg.add_cycle(i, j, 1, self.source)
        with self.assertRaises(CycleProvenanceError):
            self.hg.add_cycle(1, 2, 1, None)
        with self.assertRaises(CycleProvenanceError):
            self.hg.add_cycle(1, 2, 1, "")
        with self.assertRaises(CycleBoundsError):
            self.hg.add_cycle(1, 2, 1 << 300, self.source)
        with self.assertRaises(ValueError):
            self.hg.add_cycle(1, 2, 1.25, self.source)
        with self.assertRaises(ValueError):
            self.hg.add_cycle(1, 2, True, self.source)
        with self.assertRaises(CycleBoundsError):
            self.hg.neighbors((0, 1, 2), hops=0, limit=1)
        with self.assertRaises(CycleBoundsError):
            self.hg.neighbors((0, 1, 2), hops=9, limit=1)
        with self.assertRaises(CycleBoundsError):
            self.hg.neighbors((0, 1, 2), hops=1, limit=0)
        with self.assertRaises(CycleBoundsError):
            self.hg.snapshot(limit=0)
        with self.assertRaises(CycleBoundsError):
            self.hg.audit(limit=0)
        with self.assertRaises(CycleBoundsError):
            normalize_cycle((0, 1, 300), nodes=256)

    def test_capacity_exhaustion_is_denied_before_partial_writes(self):
        relations = RelationStore.allocate(
            relation_capacity=1, hyperedge_capacity=1, arena_bytes=1024)
        _, leaves = make_leaf_store(capacity=8)
        hg = CycleHypergraph(relations=relations, leaves=leaves)
        hg.add_cycle(1, 2, 1, self.source)
        before_status = relations.status()
        before_leaf_count = leaves.count
        before_snapshot = hg.snapshot()
        with self.assertRaises(CycleBoundsError):
            hg.add_cycle(1, 3, 2, self.source)
        self.assertEqual(relations.status(), before_status)
        self.assertEqual(leaves.count, before_leaf_count)
        self.assertEqual(hg.snapshot(), before_snapshot)

    def test_missing_cycle_members_are_empty_and_neighbors_are_empty(self):
        members = self.hg.cycle_members((0, 1, 2))
        self.assertEqual(list(members), [])
        self.assertTrue(members.missing)
        neighbors = self.hg.neighbors((0, 1, 2), hops=1, limit=4)
        self.assertEqual(list(neighbors), [])
        self.assertTrue(neighbors.missing)
        with self.assertRaises(CycleNotFound):
            self.hg.compose((0, 1, 2), (0, 2, 3))

    def test_composition_is_raw_integer_symmetric_difference(self):
        self.add(1, 2, 5)
        self.add(2, 3, 7)
        self.add(4, 5, 1)
        before_status = self.relations.status()
        before_snapshot = self.hg.snapshot()

        composed = self.hg.compose((0, 1, 2), (0, 2, 3))
        self.assertIsInstance(composed, CycleComposition)
        self.assertIsInstance(composed, int)
        self.assertEqual(composed, 12)
        self.assertEqual(composed.value, 12)
        self.assertEqual(composed["value"], 12)
        self.assertEqual(composed.cycle, (0, 1, 2, 3))
        self.assertEqual(composed["cycle"], (0, 1, 2, 3))
        self.assertEqual(composed, (0, 1, 2, 3))
        self.assertEqual(composed.shared,
                         (pair_reference(0, 2),))
        self.assertEqual(len(composed.members), 4)
        self.assertEqual(composed.as_dict()["stored"], False)
        self.assertFalse(composed.as_dict()["created"])

        # Mirror order and argument order do not change the composition.
        mirror = self.hg.compose((0, 3, 2), (0, 2, 1))
        self.assertEqual(int(mirror), 12)
        self.assertEqual(mirror.cycle, (0, 1, 2, 3))
        self.assertEqual(self.relations.status(), before_status)
        self.assertEqual(self.hg.snapshot(), before_snapshot)

        disjoint = self.hg.compose((0, 1, 2), (0, 4, 5))
        self.assertEqual(int(disjoint), 6)
        self.assertEqual(disjoint.status, "union")
        self.assertEqual(disjoint.cycle, (0, 1, 2, 4, 5))
        with self.assertRaises(CycleDenied):
            self.hg.compose((0, 1, 2), (0, 4, 5), require_shared=True)
        same = self.hg.compose((0, 1, 2), (0, 2, 1))
        self.assertEqual(int(same), 10)
        self.assertEqual(same.status, "idempotent")

    def test_neighbors_are_bounded_and_multihop(self):
        cycles = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
        for index, (i, j) in enumerate(cycles):
            self.add(i, j, index + 1)

        first = self.hg.neighbors((0, 1, 2), hops=1, limit=16)
        self.assertFalse(first.missing)
        self.assertFalse(first.truncated)
        self.assertEqual(set(first), {
            (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4)})
        self.assertEqual(first["neighbors"], list(first))
        self.assertEqual(first.hops, 1)

        bounded = self.hg.neighbors((0, 1, 2), hops=1, limit=2)
        self.assertEqual(len(bounded), 2)
        self.assertTrue(bounded.truncated)

        two_hops = self.hg.neighbors((0, 1, 2), hops=2, limit=16)
        self.assertIn((0, 3, 4), two_hops)
        self.assertEqual(len(set(two_hops)), len(two_hops))
        self.assertLessEqual(len(two_hops), 16)

        # A one-hop walk cannot reach a cycle that shares no star edge.
        one_hop = self.hg.neighbors((0, 1, 2), hops=1, limit=16)
        self.assertNotIn((0, 3, 4), one_hop)

    def test_snapshot_audit_are_bounded_and_reopen_preserves_ids(self):
        for index, (i, j) in enumerate(((1, 2), (1, 3), (2, 3)), start=1):
            self.add(i, j, index)
        snapshot = self.hg.snapshot()
        self.assertEqual(snapshot["schema"],
                         "nanlm.resident-cycle-hypergraph/1")
        self.assertEqual(snapshot["cycle_count"], 3)
        self.assertFalse(snapshot["truncated"])
        self.assertTrue(snapshot["bounded"])
        self.assertFalse(snapshot["canonical_merge"])
        bounded = self.hg.snapshot(limit=2)
        self.assertEqual(len(bounded["cycles"]), 2)
        self.assertTrue(bounded["truncated"])

        audit = self.hg.audit()
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["problems"], [])
        self.assertEqual(audit["issues"], [])
        self.assertEqual(audit["cycles_checked"], 3)
        self.assertFalse(audit["canonical_merge"])
        partial = self.hg.audit(limit=1)
        self.assertEqual(partial["cycles_checked"], 1)
        self.assertTrue(partial["truncated"])

        reopened = CycleHypergraph(
            relations=RelationStore.attach(self.relations._buffer),
            leaves=LeafStore.attach(self.leaf_buffer),
            nodes=256)
        members = reopened.cycle_members((0, 1, 2))
        self.assertEqual(members.value, 1)
        self.assertEqual(members.source_digest, self.source)
        self.assertEqual(members.hyperedge_id,
                         cycle_hyperedge_id((0, 1, 2)).hex())
        self.assertTrue(reopened.audit()["ok"])
        replay = reopened.add_cycle(2, 1, 1, self.source)
        self.assertTrue(replay["duplicate"])
        self.assertEqual(reopened.reopen().cycle_members((0, 2, 3)).value, 3)

    def test_audit_detects_a_tampered_pair_reference(self):
        self.add(1, 2, 7)
        # Rewrite one pair member inside the hyperedge arena to a foreign
        # mirror-unsafe address while keeping the record structurally valid.
        view = self.relations.get_hyperedge(cycle_hyperedge_id((0, 1, 2)))
        member = view["members"][1]
        self.assertEqual(member["kind_code"], MEMBER_PAIR)
        # The arena offset is documented as a byte offset into the section;
        # locate the exact 32-byte id through the relation store's view.
        members_offset = view["members_offset"]
        arena_start = self.relations._layout.arena + members_offset
        # Replace the second member id with an unrelated canonical address.
        self.relations._view[arena_start + 32:arena_start + 64] = \
            (0xDEAD).to_bytes(32, "big")
        audit = self.hg.audit()
        self.assertFalse(audit["ok"])
        self.assertTrue(any("pair references" in problem
                            for problem in audit["problems"]))


class FakeStoreCycleTests(unittest.TestCase):
    """A duck-typed fallback keeps the adapter usable on partial installs."""

    class FakeRelations:
        def __init__(self):
            self.hyperedges = {}
            self.generation = 0

        def append_hyperedge(self, hyperedge_id, source_digest, members):
            key = hyperedge_id.hex() if isinstance(hyperedge_id, bytes) \
                else str(hyperedge_id)
            if key in self.hyperedges:
                return {"created": False, "duplicate": True,
                        "hyperedge_id": key}
            self.hyperedges[key] = {
                "hyperedge_id": key,
                "source_digest": source_digest.hex()
                if isinstance(source_digest, bytes) else source_digest,
                "members": [
                    {"kind_code": kind,
                     "id": identifier.hex()
                     if isinstance(identifier, bytes) else identifier}
                    for kind, identifier in members
                ],
            }
            self.generation += 1
            return {"created": True, "duplicate": False,
                    "hyperedge_id": key, "generation": self.generation}

        def get_hyperedge(self, identifier):
            key = identifier.hex() if isinstance(identifier, bytes) \
                else str(identifier)
            return self.hyperedges.get(key)

        def status(self):
            return {"hyperedges": len(self.hyperedges),
                    "hyperedge_capacity": 64,
                    "relations": 0, "relation_capacity": 64,
                    "generation": self.generation}

    class FakeLeafRecord:
        def __init__(self, leaf_id, payload_digest):
            self.leaf_id = bytes(leaf_id)
            self.payload_digest = bytes(payload_digest)
            self.kind = "hyperedge"

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
            record = FakeStoreCycleTests.FakeLeafRecord(
                leaf_id, payload_digest)
            self.records[key] = record
            return record

        def __len__(self):
            return len(self.records)

    def setUp(self):
        self.relations = self.FakeRelations()
        self.leaves = self.FakeLeaves()
        self.hg = CycleHypergraph({
            "relations": self.relations,
            "leaves": self.leaves,
        })

    def test_fake_store_round_trip(self):
        receipt = self.hg.add_cycle(1, 2, 9, "fake-source")
        self.assertTrue(receipt["created"])
        self.assertEqual(receipt["value"], 9)
        members = self.hg.cycle_members((0, 2, 1))
        self.assertEqual(members.value, 9)
        self.assertEqual(len(members), 3)
        self.assertEqual(self.hg.snapshot()["cycle_count"], 1)
        self.assertTrue(self.hg.audit()["ok"])
        self.hg.add_cycle(2, 3, 4, "fake-source")
        self.assertEqual(set(self.hg.neighbors((0, 1, 2), 1, 8)),
                         {(0, 2, 3)})
        composed = self.hg.compose((0, 1, 2), (0, 2, 3))
        self.assertEqual(int(composed), 13)
        duplicate = self.hg.add_cycle(2, 1, 9, "fake-source")
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(CycleConflictError):
            self.hg.add_cycle(1, 2, 10, "fake-source")


if __name__ == "__main__":
    unittest.main()
