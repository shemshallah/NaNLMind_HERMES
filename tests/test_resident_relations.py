import hashlib
import unittest

from nanlmind_hermes.resident.relations import (
    MAX_WALK_LIMIT,
    MEMBER_LEAF,
    MEMBER_MIRROR,
    MEMBER_PAIR,
    MEMBER_RELATION,
    RelationBoundsError,
    RelationStore,
    hyperedge_id_for_members,
)


class ResidentRelationTests(unittest.TestCase):
    def setUp(self):
        self.source = hashlib.sha256(b"resident-relations-source").hexdigest()
        self.store = RelationStore.allocate(
            relation_capacity=8, hyperedge_capacity=8, arena_bytes=4096,
            base_offset=64)

    def append(self, **overrides):
        values = {
            "subject": "leaf:alpha",
            "predicate": "leaf:relates",
            "object": "leaf:beta",
            "polarity": "supports",
            "confidence": 0.75,
            "source_digest": self.source,
        }
        values.update(overrides)
        return self.store.append_relation(**values)

    def test_append_returns_content_addressed_record_and_status(self):
        receipt = self.append(pair_id="pair:alpha-beta", mirror_id="leaf:beta")
        self.assertEqual(receipt["status"], "appended")
        self.assertTrue(receipt["created"])
        self.assertFalse(receipt["duplicate"])
        self.assertEqual(len(receipt["relation_id"]), 64)
        self.assertEqual(receipt["generation"], 1)
        self.assertEqual(len(self.store), 1)

        record = self.store.get_relation(receipt["relation_id"])
        self.assertIsNotNone(record)
        self.assertEqual(record["relation_id"], receipt["relation_id"])
        self.assertEqual(record["source_digest"], self.source)
        self.assertEqual(record["generation"], 1)
        self.assertIsNone(record["next_subject"])
        self.assertEqual(self.store.relation_by_id(record["relation_id"]),
                         record)
        status = self.store.status()
        self.assertEqual(status["relations"], 1)
        self.assertEqual(status["hyperedges"], 1)
        self.assertEqual(status["generation"], 1)
        self.assertEqual(status["relation_record_bytes"], 304)
        self.assertEqual(status["hyperedge_record_bytes"], 96)

    def test_append_links_subject_predicate_and_object_leaf_chains(self):
        first = self.append(subject="leaf:a", predicate="leaf:p",
                            object="leaf:b")
        second = self.append(subject="leaf:a", predicate="leaf:q",
                             object="leaf:c")
        third = self.append(subject="leaf:d", predicate="leaf:p",
                            object="leaf:b")

        subject_chain = self.store.walk_chain("leaf:a", role="subject")
        self.assertEqual([row["relation_id"] for row in subject_chain],
                         [second["relation_id"], first["relation_id"]])
        predicate_chain = self.store.walk_chain("leaf:p", role="predicate")
        self.assertEqual([row["relation_id"] for row in predicate_chain],
                         [third["relation_id"], first["relation_id"]])
        object_chain = self.store.walk_chain("leaf:b", role="object")
        self.assertEqual([row["relation_id"] for row in object_chain],
                         [third["relation_id"], first["relation_id"]])
        self.assertEqual(
            [row["relation_id"] for row in self.store.walk_chain(
                "leaf:d", role="subject")],
            [third["relation_id"]])

        newest = self.store.relation_by_id(second["relation_id"])
        self.assertEqual(newest["next_subject"], first["relation_id"])
        self.assertEqual(newest["next_subject_index"], 0)
        oldest = self.store.relation_by_id(first["relation_id"])
        self.assertIsNone(oldest["next_subject"])

        bounded = self.store.chain_walk("leaf:a", role="subject", limit=1)
        self.assertEqual([row["relation_id"] for row in bounded["relations"]],
                         [second["relation_id"]])
        self.assertTrue(bounded["truncated"])
        complete = self.store.chain_walk("leaf:a", role="subject", limit=8)
        self.assertFalse(complete["truncated"])
        self.assertEqual(len(complete["relations"]), 2)

    def test_hyperedge_membership_reuses_deterministic_edge(self):
        receipt = self.append(pair_id="pair:a-b", mirror_id="leaf:beta")
        edge = self.store.hyperedge_for_relation(receipt["relation_id"])
        self.assertIsNotNone(edge)
        self.assertEqual(edge["hyperedge_id"], receipt["hyperedge_id"])
        self.assertEqual(edge["arity"], 6)
        self.assertEqual([member["kind"] for member in edge["members"]],
                         ["leaf", "leaf", "leaf", "pair", "mirror",
                          "relation"])
        record = self.store.relation_by_id(receipt["relation_id"])
        self.assertEqual([member["id"] for member in edge["members"]], [
            record["subject"], record["predicate"], record["object"],
            record["pair_id"], record["mirror_id"], record["relation_id"],
        ])
        self.assertEqual(
            [member["kind_code"] for member in edge["members"]],
            [MEMBER_LEAF, MEMBER_LEAF, MEMBER_LEAF, MEMBER_PAIR,
             MEMBER_MIRROR, MEMBER_RELATION])
        iterated = list(self.store.iter_hyperedge_members(
            receipt["hyperedge_id"]))
        self.assertEqual(iterated, edge["members"])
        self.assertEqual(self.store.hyperedge_members("missing-edge"), [])

        shared_members = [("leaf", "leaf:x"), ("leaf", "leaf:y"),
                          ("pair", "pair:xy")]
        first = self.append(subject="leaf:x", predicate="leaf:y",
                            object="leaf:z",
                            hyperedge_members=shared_members)
        second = self.append(subject="leaf:z", predicate="leaf:y",
                             object="leaf:x",
                             hyperedge_members=shared_members)
        expected_id = hyperedge_id_for_members(shared_members)
        self.assertEqual(first["hyperedge_id"], expected_id)
        self.assertEqual(second["hyperedge_id"], expected_id)
        self.assertTrue(first["hyperedge_created"])
        self.assertFalse(second["hyperedge_created"])
        reused = self.store.hyperedge_by_id(expected_id)
        self.assertEqual(reused["member_ids"],
                         [member["id"] for member in reused["members"]])
        self.assertEqual(reused["arity"], 3)

    def test_duplicate_relation_id_is_idempotent_and_never_overwrites(self):
        first = self.append(confidence=0.6, pair_id="pair:a-b")
        raw_before = self.store.relation_bytes(0)
        status_before = self.store.status()

        duplicate = self.append(confidence=0.6, pair_id="pair:a-b",
                                source_digest=hashlib.sha256(b"other").hexdigest())
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(duplicate["created"])
        self.assertFalse(duplicate["conflict"])
        self.assertEqual(duplicate["relation_id"], first["relation_id"])
        self.assertEqual(self.store.relation_bytes(0), raw_before)
        self.assertEqual(len(self.store), 1)
        after = self.store.status()
        self.assertEqual(after["generation"], status_before["generation"])
        self.assertEqual(after["hyperedges"], status_before["hyperedges"])
        self.assertEqual(after["arena_used"], status_before["arena_used"])
        self.assertEqual(self.store.relation_by_id(first["relation_id"])[
            "source_digest"], self.source)

        conflict = self.append(confidence=0.9, pair_id="pair:a-b",
                               relation_id=first["relation_id"])
        self.assertTrue(conflict["duplicate"])
        self.assertTrue(conflict["conflict"])
        self.assertEqual(conflict["relation"]["confidence"], 0.6)
        self.assertEqual(self.store.relation_bytes(0), raw_before)

    def test_polarity_and_confidence_are_preserved(self):
        supported = self.append(subject="leaf:s1", polarity="supports",
                                confidence=0.1)
        refuted = self.append(subject="leaf:s2", polarity="refutes",
                              confidence=0.875)
        neutral = self.append(subject="leaf:s3", polarity="neutral",
                              confidence=1.0)
        for receipt, name, code, confidence in (
            (supported, "supports", 1, 0.1),
            (refuted, "refutes", -1, 0.875),
            (neutral, "neutral", 0, 1.0),
        ):
            record = self.store.relation_by_id(receipt["relation_id"])
            self.assertEqual(record["polarity"], name)
            self.assertEqual(record["polarity_code"], code)
            self.assertEqual(record["confidence"], confidence)
        self.assertNotEqual(supported["relation_id"], refuted["relation_id"])

    def test_bounds_failures_do_not_append_or_overwrite(self):
        small = RelationStore.allocate(
            relation_capacity=1, hyperedge_capacity=1, arena_bytes=1024)
        source = hashlib.sha256(b"small").hexdigest()
        first = small.append_relation(
            subject="leaf:a", predicate="leaf:p", object="leaf:b",
            source_digest=source)
        raw_before = small.relation_bytes(0)
        status_before = small.status()
        with self.assertRaises(RelationBoundsError) as raised:
            small.append_relation(
                subject="leaf:c", predicate="leaf:p", object="leaf:d",
                source_digest=source)
        self.assertEqual(raised.exception.code, "relation_capacity")
        self.assertEqual(len(small), 1)
        self.assertEqual(small.status()["hyperedges"],
                         status_before["hyperedges"])
        self.assertEqual(small.status()["arena_used"],
                         status_before["arena_used"])
        self.assertEqual(small.relation_bytes(0), raw_before)
        duplicate = small.append_relation(
            subject="leaf:a", predicate="leaf:p", object="leaf:b",
            source_digest=source)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["relation_id"], first["relation_id"])

        hyperedge_limited = RelationStore.allocate(
            relation_capacity=2, hyperedge_capacity=1, arena_bytes=1024)
        hyperedge_limited.append_relation(
            subject="leaf:a", predicate="leaf:p", object="leaf:b",
            source_digest=source)
        with self.assertRaises(RelationBoundsError) as raised:
            hyperedge_limited.append_relation(
                subject="leaf:c", predicate="leaf:p", object="leaf:d",
                source_digest=source)
        self.assertEqual(raised.exception.code, "hyperedge_capacity")
        self.assertEqual(len(hyperedge_limited), 1)

        with self.assertRaises(RelationBoundsError):
            small.walk_chain("leaf:a", limit=0)
        with self.assertRaises(RelationBoundsError):
            small.walk_chain("leaf:a", limit=MAX_WALK_LIMIT + 1)
        with self.assertRaises(RelationBoundsError):
            RelationStore.attach(bytearray(32))
        with self.assertRaises(RelationBoundsError):
            RelationStore.allocate(relation_capacity=4, base_offset=-1)

    def test_pair_and_mirror_callbacks_update_external_references(self):
        calls = []

        def pair_callback(**values):
            calls.append(("pair", values))

        def mirror_callback(**values):
            calls.append(("mirror", values))

        receipt = self.append(pair_id="pair:a-b", mirror_id="leaf:beta",
                              pair_callback=pair_callback,
                              mirror_callback=mirror_callback)
        self.assertTrue(receipt["pair_linked"])
        self.assertTrue(receipt["mirror_linked"])
        record = receipt["relation"]
        self.assertEqual(calls[0][0], "pair")
        self.assertEqual(calls[0][1]["target_id"], record["pair_id"])
        self.assertEqual(calls[0][1]["relation_id"], record["relation_id"])
        self.assertEqual(calls[0][1]["generation"], 1)
        self.assertEqual(calls[0][1]["source_digest"], self.source)
        self.assertEqual(calls[1][0], "mirror")
        self.assertEqual(calls[1][1]["target_id"], record["mirror_id"])

    def test_records_are_never_rewritten_and_prefix_is_preserved(self):
        buffer = bytearray(RelationStore.required_bytes(
            relation_capacity=4, hyperedge_capacity=4, arena_bytes=1024,
            base_offset=64))
        buffer[:64] = b"P" * 64
        store = RelationStore.create(
            buffer, base_offset=64, relation_capacity=4,
            hyperedge_capacity=4, arena_bytes=1024)
        first = store.append_relation(
            subject="leaf:a", predicate="leaf:p", object="leaf:b",
            source_digest=self.source)
        raw_first = store.relation_bytes(0)
        second = store.append_relation(
            subject="leaf:a", predicate="leaf:p", object="leaf:c",
            source_digest=self.source)
        self.assertEqual(store.relation_bytes(0), raw_first)
        self.assertEqual(store.get_relation(0)["relation_id"],
                         first["relation_id"])
        self.assertEqual(store.get_relation(1)["relation_id"],
                         second["relation_id"])
        self.assertEqual(bytes(buffer[:64]), b"P" * 64)
        reopened = RelationStore.attach(buffer, base_offset=64)
        self.assertEqual(reopened.status()["relations"], 2)
        self.assertEqual(reopened.get_relation(first["relation_id"])["object"],
                         store.get_relation(first["relation_id"])["object"])


if __name__ == "__main__":
    unittest.main()
