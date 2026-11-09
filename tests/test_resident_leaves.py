"""Resident leaf/tree store tests: records, links, bounds and images."""
import hashlib
import json
import unittest

from nanlmind_hermes.resident import HEAD_COUNT, PAIR_COUNT
from nanlmind_hermes.resident.leaves import (
    HEADER_BYTES,
    INDEX_ENTRY_BYTES,
    KINDS,
    RECORD_BYTES,
    LeafBoundsError,
    LeafCycleError,
    LeafKindError,
    LeafNotFound,
    LeafRecord,
    LeafStore,
    record_layout,
)


def make_store(capacity=32, size=8192, base_offset=0, **kwargs):
    buffer = bytearray(size)
    store = LeafStore.create(
        buffer, base_offset=base_offset, capacity=capacity, **kwargs)
    return buffer, store


class RecordLayoutTests(unittest.TestCase):
    def test_layout_is_fixed_and_covers_the_required_fields(self):
        self.assertEqual(HEADER_BYTES, 48)
        self.assertEqual(RECORD_BYTES, 112)
        self.assertEqual(INDEX_ENTRY_BYTES, 24)
        layout = record_layout()
        self.assertEqual({item["name"] for item in layout}, {
            "magic", "version", "kind_code", "flags", "head_index",
            "relation_count", "primary_pair_id", "feature_offset",
            "feature_length", "parent", "first_child", "next_sibling",
            "first_relation", "next_relation", "relation_head", "generation",
            "leaf_id", "payload_digest", "reserved",
        })
        self.assertEqual(layout[0]["offset"], 0)
        self.assertEqual(layout[-1]["offset"] + layout[-1]["size"], RECORD_BYTES)
        self.assertEqual(KINDS,
                         ("word", "concept", "vector", "relation",
                          "hyperedge", "data"))


class InsertTests(unittest.TestCase):
    def test_insert_and_lookup_by_slot_bytes_hex_and_record(self):
        _, store = make_store()
        leaf = store.insert(
            "word", content=b"alpha", head=7, pair_id=123, flags=0x12)
        self.assertIsInstance(leaf, LeafRecord)
        self.assertEqual(leaf.slot, 0)
        self.assertEqual(leaf.kind, "word")
        self.assertEqual(leaf.head, 7)
        self.assertEqual(leaf.pair_id, 123)
        self.assertEqual(leaf.flags, 0x12)
        self.assertEqual(leaf.generation, 1)
        self.assertEqual(len(leaf.leaf_id), 16)
        self.assertEqual(leaf.payload_digest, hashlib.sha256(b"alpha").digest())
        self.assertIsNone(leaf.parent)
        self.assertIsNone(leaf.first_child)
        self.assertIsNone(leaf.first_relation)
        self.assertEqual(leaf, store.get(leaf.slot))
        self.assertEqual(leaf, store.get(leaf.leaf_id))
        self.assertEqual(leaf, store.get(leaf.leaf_id.hex()))
        self.assertEqual(leaf, store.get("leaf:" + leaf.leaf_id.hex()))
        self.assertEqual(leaf, store.get(leaf))
        self.assertIn(leaf.leaf_id, store)
        self.assertIn(leaf.leaf_id.hex(), store)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.stats()["count"], 1)

        other = store.insert("concept", content=b"beta", head=8)
        self.assertEqual(other.slot, 1)
        self.assertEqual(other.generation, 2)
        self.assertEqual([item.slot for item in store], [0, 1])
        self.assertEqual(store.leaf_at(0), leaf)
        self.assertIsNone(store.leaf_at(2))
        self.assertIsNone(store.find(b"\x00" * 16))
        with self.assertRaises(LeafNotFound):
            store.get(b"\x00" * 16)

    def test_explicit_id_and_payload_digest_are_stored_as_bytes(self):
        _, store = make_store(capacity=8)
        leaf_id = bytes(range(16))
        digest = hashlib.sha256(b"payload").digest()
        leaf = store.insert(
            "data", leaf_id=leaf_id.hex(), payload_digest=digest,
            feature_offset=10, feature_length=6)
        self.assertEqual(leaf.leaf_id, leaf_id)
        self.assertEqual(leaf.payload_digest, digest)
        self.assertEqual(leaf.feature_offset, 10)
        self.assertEqual(leaf.feature_length, 6)
        self.assertEqual(store.get(leaf_id), leaf)

    def test_auto_id_is_a_deterministic_128_bit_content_hash(self):
        _, first = make_store(capacity=8)
        _, second = make_store(capacity=8)
        one = first.insert("word", content=b"same", head=1, pair_id=2)
        two = second.insert("word", content=b"same", head=1, pair_id=2)
        self.assertEqual(one.leaf_id, two.leaf_id)
        changed = second.insert("word", content=b"same", head=2, pair_id=2)
        self.assertNotEqual(one.leaf_id, changed.leaf_id)
        self.assertEqual(len(one.leaf_id), 16)

    def test_duplicate_id_is_idempotent_and_does_not_relink(self):
        _, store = make_store(capacity=8)
        parent = store.insert("concept", content=b"parent")
        leaf_id = b"\xAB" * 16
        original = store.insert(
            "word", leaf_id=leaf_id, content=b"first", head=3, pair_id=4)
        before = store.serialize()
        duplicate = store.insert(
            "vector", leaf_id=leaf_id, content=b"different", head=9,
            pair_id=9, parent=parent)
        self.assertEqual(duplicate, original)
        self.assertEqual(duplicate.kind, "word")
        self.assertEqual(duplicate.head, 3)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.children(parent), [])
        self.assertEqual(store.serialize(), before)


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.buffer, self.store = make_store(capacity=64, size=16384)
        self.root = self.store.insert("concept", content=b"root")
        self.a = self.store.insert(
            "word", content=b"a", parent=self.root)
        self.b = self.store.insert(
            "word", content=b"b", parent=self.root)
        self.c = self.store.insert(
            "word", content=b"c", parent=self.root)
        self.d = self.store.insert(
            "vector", content=b"d", parent=self.b)
        self.e = self.store.insert(
            "data", content=b"e", parent=self.d)

    def slots(self, records):
        return [record.slot for record in records]

    def test_children_order_depth_and_iterative_subtree_walk(self):
        store = self.store
        self.assertEqual(self.slots(store.children(self.root)),
                         [self.a.slot, self.b.slot, self.c.slot])
        self.assertEqual(self.slots(store.walk_subtree(self.root)),
                         [self.root.slot, self.a.slot, self.b.slot,
                          self.d.slot, self.e.slot, self.c.slot])
        self.assertEqual(store.depth(self.root), 0)
        self.assertEqual(store.depth(self.a), 1)
        self.assertEqual(store.depth(self.d), 2)
        self.assertEqual(store.depth(self.e), 3)
        self.assertTrue(self.root.is_root)
        self.assertFalse(self.e.is_root)
        self.assertIn(self.d.slot, self.slots(store.subtree(self.b)))

    def test_relink_moves_leaf_and_preserves_its_subtree_and_chain(self):
        store = self.store
        moved = store.move_leaf(self.b, self.c)
        self.assertEqual(moved.parent, self.c.slot)
        self.assertEqual(self.slots(store.children(self.root)),
                         [self.a.slot, self.c.slot])
        self.assertEqual(self.slots(store.children(self.c)),
                         [self.b.slot])
        self.assertEqual(self.slots(store.children(self.b)),
                         [self.d.slot])
        self.assertEqual(self.slots(store.walk_subtree(self.c)),
                         [self.c.slot, self.b.slot, self.d.slot, self.e.slot])
        self.assertEqual(store.get(self.d).parent, self.b.slot)
        self.assertIsNone(store.get(self.c).next_sibling)

        # Move a leaf out of the middle; the remaining chain stays intact.
        store.unlink(self.a)
        self.assertIsNone(store.get(self.a).parent)
        self.assertEqual(self.slots(store.children(self.root)),
                         [self.c.slot])
        self.assertEqual(self.slots(store.walk_subtree(self.a)),
                         [self.a.slot])

        # Relink a leaf with its own children under a new parent.
        store.relink(self.e, self.a)
        self.assertEqual(self.slots(store.children(self.a)), [self.e.slot])
        self.assertEqual(self.slots(store.walk_subtree(self.a)),
                         [self.a.slot, self.e.slot])
        self.assertEqual(self.slots(store.children(self.b)), [self.d.slot])

        store.move_leaf(self.b, self.root, prepend=True)
        self.assertEqual(self.slots(store.children(self.root)),
                         [self.b.slot, self.c.slot])
        self.assertEqual(self.slots(store.walk_subtree(self.root)),
                         [self.root.slot, self.b.slot, self.d.slot, self.c.slot])

    def test_move_to_same_parent_is_idempotent_and_cycles_rejected(self):
        store = self.store
        before = self.slots(store.children(self.root))
        same = store.move_leaf(self.b, self.root)
        self.assertEqual(same, store.get(self.b))
        self.assertEqual(self.slots(store.children(self.root)), before)
        with self.assertRaises(LeafCycleError):
            store.move_leaf(self.root, self.d)
        with self.assertRaises(LeafCycleError):
            store.move_leaf(self.b, self.b)
        with self.assertRaises(LeafCycleError):
            store.move_leaf(self.b, self.e)

    def test_parent_first_link_helper(self):
        store = self.store
        store.unlink(self.a)
        self.assertIsNone(store.get(self.a).parent)
        linked = store.link(self.root, self.a)
        self.assertEqual(linked.parent, self.root.slot)
        self.assertEqual(self.slots(store.children(self.root)),
                         [self.b.slot, self.c.slot, self.a.slot])
        with self.assertRaises(LeafCycleError):
            store.link(self.c, self.root)

    def test_walk_is_iterative_for_a_deep_chain(self):
        buffer = bytearray(65536)
        store = LeafStore.create(buffer, capacity=320)
        root = store.insert("word", content=b"deep-root")
        previous = root
        for index in range(299):
            previous = store.insert(
                "word", content=("deep-%03d" % index).encode(),
                parent=previous)
        walked = list(store.walk_subtree(root))
        self.assertEqual(len(walked), 300)
        self.assertEqual(store.depth(previous), 299)
        limited = list(store.walk_subtree(root, max_depth=10))
        self.assertEqual(len(limited), 11)
        without_root = list(store.walk_subtree(root, include_self=False))
        self.assertEqual(len(without_root), 299)
        self.assertEqual(without_root[0], store.get(walked[1]))


class RelationTests(unittest.TestCase):
    def test_attach_move_and_detach_relation_chain(self):
        _, store = make_store(capacity=32)
        head = store.insert("word", content=b"head")
        other = store.insert("word", content=b"other")
        first = store.insert("relation", content=b"r1")
        second = store.insert("relation", content=b"r2")

        store.attach_relation(head, first)
        store.attach_relation(head, second)
        self.assertEqual(self.ids(store.relations(head)),
                         [second.leaf_id, first.leaf_id])
        self.assertEqual(store.get(head).relation_count, 2)
        self.assertEqual(store.get(head).first_relation, second.slot)
        self.assertEqual(store.get(first).relation_head, head.slot)
        self.assertEqual(store.get(second).next_relation, first.slot)

        # Re-attaching the same relation to the same head is a no-op.
        store.attach_relation(head, first)
        self.assertEqual(self.ids(store.relations(head)),
                         [second.leaf_id, first.leaf_id])
        self.assertEqual(store.get(head).relation_count, 2)

        # Moving a relation between heads does not duplicate it.
        store.attach_relation(other, second)
        self.assertEqual(self.ids(store.relations(head)), [first.leaf_id])
        self.assertEqual(store.get(head).relation_count, 1)
        self.assertEqual(self.ids(store.relations(other)), [second.leaf_id])
        self.assertEqual(store.get(other).relation_count, 1)

        store.detach_relation(first)
        self.assertEqual(store.relations(head), [])
        self.assertEqual(store.get(head).relation_count, 0)
        self.assertIsNone(store.get(first).relation_head)

        with self.assertRaises(LeafKindError):
            store.attach_relation(head, other)
        with self.assertRaises(LeafCycleError):
            store.attach_relation(first, first)

    @staticmethod
    def ids(records):
        return [record.leaf_id for record in records]


class SuperpositionTests(unittest.TestCase):
    def test_leaves_share_pair_and_feature_references(self):
        buffer = bytearray(4096)
        store = LeafStore.create(
            buffer, capacity=16, feature_region=(3000, 128))
        buffer[3008:3016] = b"shared!!"
        common = dict(
            head=9, pair_id=99, feature_offset=3008, feature_length=8)
        word = store.insert("word", content=b"one", **common)
        concept = store.insert("concept", content=b"two", **common)
        vector = store.insert("vector", content=b"three", **common)
        self.assertEqual(len({word.leaf_id, concept.leaf_id, vector.leaf_id}), 3)
        self.assertEqual(len({word.slot, concept.slot, vector.slot}), 3)
        for leaf in (word, concept, vector):
            self.assertEqual(leaf.head, 9)
            self.assertEqual(leaf.pair_id, 99)
            self.assertEqual((leaf.feature_offset, leaf.feature_length),
                             (3008, 8))
            self.assertEqual(store.feature_bytes(leaf), b"shared!!")

        # No one-word-one-cell rule: the same head/pair can hold many words.
        first_word = store.insert(
            "word", content=b"word-a", head=1, pair_id=5)
        second_word = store.insert(
            "word", content=b"word-b", head=1, pair_id=5)
        self.assertNotEqual(first_word.leaf_id, second_word.leaf_id)
        self.assertEqual(first_word.head, second_word.head)
        self.assertEqual(first_word.pair_id, second_word.pair_id)


class BoundsTests(unittest.TestCase):
    def test_buffer_and_record_bounds_are_rejected(self):
        with self.assertRaises(LeafBoundsError):
            LeafStore.create(bytearray(256), capacity=100)
        with self.assertRaises(LeafBoundsError):
            LeafStore.create(bytes(4096), capacity=8)
        with self.assertRaises(LeafBoundsError):
            LeafStore.create(bytearray(128), base_offset=100, capacity=4)
        with self.assertRaises(LeafBoundsError):
            LeafStore.create(bytearray(4096), capacity=0)

        _, store = make_store(capacity=3, size=4096)
        for index in range(3):
            store.insert("word", content=("w%d" % index).encode())
        before = store.serialize()
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"overflow")
        self.assertEqual(len(store), 3)
        self.assertEqual(store.serialize(), before)
        with self.assertRaises(LeafBoundsError):
            store.get(3)
        with self.assertRaises(LeafBoundsError):
            store.leaf_at(3)

    def test_invalid_ids_kinds_heads_pairs_and_flags_are_rejected(self):
        _, store = make_store(capacity=8)
        with self.assertRaises(LeafBoundsError):
            store.insert("word", leaf_id=b"short")
        with self.assertRaises(LeafBoundsError):
            store.insert("word", leaf_id="not hex")
        with self.assertRaises(LeafKindError):
            store.insert("sentence", content=b"nope")
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"head", head=HEAD_COUNT)
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"pair", pair_id=PAIR_COUNT)
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"flags", flags=1 << 16)
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"generation", generation=-1)
        with self.assertRaises(LeafBoundsError):
            store.insert("word", content=b"digest", payload_digest=b"short")

    def test_feature_region_and_shared_feature_bytes_are_bounded(self):
        buffer = bytearray(4096)
        store = LeafStore.create(
            buffer, capacity=8, feature_region=(3000, 128))
        buffer[3000:3004] = b"data"
        leaf = store.insert(
            "data", content=b"feature", feature_offset=3000,
            feature_length=4)
        self.assertEqual(store.feature_bytes(leaf), b"data")
        with self.assertRaises(LeafBoundsError):
            store.insert(
                "data", content=b"outside", feature_offset=3150,
                feature_length=8)
        with self.assertRaises(LeafBoundsError):
            store.insert(
                "data", content=b"before", feature_offset=2999,
                feature_length=1)


class SerializationTests(unittest.TestCase):
    def build(self, capacity=16, size=4096, base_offset=0):
        buffer, store = make_store(
            capacity=capacity, size=size, base_offset=base_offset)
        root = store.insert("concept", content=b"root", flags=1)
        left = store.insert(
            "word", content=b"left", head=2, pair_id=3, parent=root)
        right = store.insert(
            "word", content=b"right", head=4, pair_id=5, parent=root)
        child = store.insert(
            "vector", content=b"child", feature_offset=100,
            feature_length=12, parent=left)
        relation = store.insert("relation", content=b"relation")
        store.attach_relation(left, relation)
        store.move_leaf(right, left)
        return store, root, left, right, child, relation

    def test_identical_operations_produce_identical_images(self):
        first, root, *_ = self.build()
        second, second_root, *_ = self.build()
        self.assertEqual(first.serialize(), first.serialize())
        self.assertEqual(first.serialize(), second.serialize())
        self.assertEqual(
            first.record_bytes(root), second.record_bytes(second_root))
        self.assertEqual(first.stats()["generation"],
                         second.stats()["generation"])

        restored = LeafStore.from_serialized(first.serialize())
        self.assertEqual(len(restored), len(first))
        for record in first.records():
            self.assertEqual(restored.get(record.leaf_id).to_dict(),
                             record.to_dict())
        self.assertEqual(
            [item.leaf_id for item in restored.walk_subtree(root)],
            [item.leaf_id for item in first.walk_subtree(root)])
        self.assertEqual(restored.serialize(), first.serialize())

    def test_serialization_is_json_friendly_and_content_sensitive(self):
        store, root, *_ = self.build()
        payload = json.dumps(
            store.get(root).to_dict(), sort_keys=True, separators=(",", ":"))
        self.assertIn('"leaf_id"', payload)
        changed, _, _, _, _, _ = self.build()
        extra = changed.insert("data", content=b"extra")
        self.assertNotEqual(changed.serialize(), store.serialize())
        self.assertNotEqual(extra.leaf_id, root.leaf_id)

    def test_base_offset_create_attach_round_trip(self):
        buffer = bytearray(8192)
        store = LeafStore.create(
            buffer, base_offset=256, capacity=8,
            feature_region=(6000, 256))
        root = store.insert("word", content=b"offset-root")
        child = store.insert("data", content=b"offset-child", parent=root)
        image = store.serialize()
        self.assertEqual(len(image), store.total_bytes)
        reopened = LeafStore.attach(
            buffer, base_offset=256, feature_region=(6000, 256))
        self.assertEqual(len(reopened), 2)
        self.assertEqual(reopened.get(root.leaf_id), store.get(root.slot))
        self.assertEqual(reopened.get(child.leaf_id), store.get(child.slot))
        self.assertEqual(reopened.get(root.leaf_id).first_child, child.slot)
        self.assertEqual(reopened.serialize(), image)
        self.assertEqual(store.base_offset, 256)


class CallerIndexTests(unittest.TestCase):
    def test_caller_provided_index_callback_replaces_the_builtin_index(self):
        table = {}

        def index_get(leaf_id):
            return table.get(bytes(leaf_id))

        def index_set(leaf_id, slot):
            table[bytes(leaf_id)] = slot

        buffer = bytearray(4096)
        store = LeafStore.create(
            buffer, capacity=8, index_get=index_get, index_set=index_set)
        leaf = store.insert("word", content=b"callback")
        self.assertEqual(table[leaf.leaf_id], leaf.slot)
        self.assertEqual(store.get(leaf.leaf_id), leaf)
        self.assertTrue(store.stats()["external_index"])
        self.assertEqual(store.index_capacity, 0)

        reopened = LeafStore.attach(
            buffer, index_get=index_get, index_set=index_set)
        self.assertEqual(len(reopened), 1)
        self.assertEqual(reopened.get(leaf.leaf_id), leaf)


if __name__ == "__main__":
    unittest.main()
