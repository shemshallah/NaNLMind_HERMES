"""Tests for resident quantized resonance and superposition retrieval."""
import math
import unittest

from nanlmind_hermes.resident.resonance import (
    ArenaFullError,
    CellRef,
    DEFAULT_DIMENSIONS,
    FeatureArena,
    FeatureRef,
    FeatureStore,
    ResonanceHit,
    query,
    superpose,
)


class FeatureArenaTests(unittest.TestCase):
    def test_default_store_geometry_and_stable_offsets(self):
        store = FeatureStore()
        self.assertEqual(store.dimensions, DEFAULT_DIMENSIONS)
        self.assertEqual(store.dimensions, 128)
        self.assertEqual(store.dtype, "int16")
        self.assertEqual(store.capacity, 256)

        first = store.add([0.5] * 128, feature_id=10)
        second = store.add([0.25] * 128, feature_id=11)
        self.assertEqual(first.offset, 0)
        self.assertEqual(second.offset, store.arena.slot_bytes)
        self.assertNotEqual(first.offset, second.offset)
        self.assertEqual(first.offset % 8, 0)
        self.assertEqual(store.arena.slot_for_offset(first.offset), first.cell)
        self.assertEqual(store.arena.offset_for_slot(second.cell), second.offset)
        self.assertEqual(store.arena.read_quantized(first.offset)[0], 32767)
        self.assertEqual(store.arena.read_quantized(second.offset)[0], 32767)
        self.assertAlmostEqual(store.read_feature(11)[0], 0.25, places=6)

    def test_int8_auto_scale_and_quantized_range(self):
        arena = FeatureArena(dimensions=3, capacity=2, dtype="int8")
        store = FeatureStore(arena=arena)
        ref = store.add([0.25, -0.5, 1.0], feature_id="concept")
        self.assertEqual(arena.item_size, 1)
        self.assertEqual(arena.qmax, 127)
        self.assertEqual(max(abs(value) for value in
                             arena.read_quantized(ref.offset)), 127)
        for got, expected in zip(store.read_feature("concept"),
                                 (0.25, -0.5, 1.0)):
            self.assertAlmostEqual(got, expected, delta=1.0 / 127.0 + 1e-12)
        self.assertEqual(store.feature("concept")["offset"], ref.offset)

        saturated = store.add([1000.0, -1000.0, 0.0], feature_id="saturated",
                              scale=0.001)
        self.assertEqual(arena.read_quantized(saturated.offset),
                         [127, -128, 0])

    def test_external_writable_buffer_and_read_only_rejection(self):
        arena = FeatureArena(dimensions=4, capacity=2, dtype="int8",
                             buffer=bytearray(64))
        store = FeatureStore(arena=arena)
        ref = store.add([1.0, 0.0, 0.0, 0.0], feature_id=7)
        self.assertEqual(ref.offset, 0)
        self.assertEqual(store.read_feature(7)[0], 1.0)
        with self.assertRaises(ValueError):
            FeatureArena(dimensions=4, capacity=2, dtype="int8",
                         buffer=bytes(64))
        with self.assertRaises(ValueError):
            FeatureArena(dimensions=4, capacity=2, dtype="int8",
                         buffer=bytearray(8))

    def test_arena_capacity_is_bounded(self):
        arena = FeatureArena(dimensions=2, capacity=1)
        store = FeatureStore(arena=arena)
        store.add([1.0, 0.0], feature_id=1)
        self.assertEqual(store.arena.free, 0)
        with self.assertRaises(ArenaFullError):
            store.add([0.0, 1.0], feature_id=2)
        self.assertEqual(arena.stats()["used"], 1)

    def test_attach_existing_resident_offset(self):
        arena = FeatureArena(dimensions=2, capacity=2)
        offset = arena.allocate()
        arena.write_vector(offset, [1.0, 0.0], scale=1.0)
        self.assertIsInstance(arena.buffer, bytearray)
        self.assertEqual(len(arena.buffer), arena.byte_size)

        store = FeatureStore(arena=arena)
        attached = store.attach("external", offset, leaf_id="leaf",
                                scale=1.0)
        self.assertEqual(attached.cell, 0)
        self.assertEqual(attached.offset, offset)
        self.assertEqual(store.query([1.0, 0.0], top_k=1)[0].feature_id,
                         "external")
        alias = store.attach("alias", offset, scale=1.0)
        self.assertEqual(alias.cell, 0)
        self.assertEqual(store.cell(0).features, ("external", "alias"))
        with self.assertRaises(ValueError):
            store.attach("external", offset, scale=1.0)
        with self.assertRaises(ValueError):
            store.attach("missing", arena.offset_for_slot(1), scale=1.0)


class ResonanceQueryTests(unittest.TestCase):
    def test_nearest_known_vector_from_noisy_query(self):
        store = FeatureStore(dimensions=4)
        vectors = {
            10: [1.0, 0.0, 0.0, 0.0],
            20: [0.0, 1.0, 0.0, 0.0],
            30: [0.0, 0.0, 1.0, 0.0],
            40: [0.15, 0.05, 0.0, 0.0],
        }
        for feature_id in (40, 30, 10, 20):
            store.add(vectors[feature_id], feature_id=feature_id, scale=0.001)
        hits = store.query([0.9, 0.1, 0.0, 0.0], top_k=2)
        self.assertEqual([hit.feature_id for hit in hits], [10, 40])
        self.assertEqual(hits[0].leaf_id, 10)
        self.assertAlmostEqual(hits[0].score, 0.9, places=6)
        self.assertAlmostEqual(hits[1].score, 0.14, places=6)
        self.assertEqual(hits[0].offset, store.feature(10).offset)
        self.assertEqual(hits[0].dimensions, 4)

    def test_empty_store_returns_empty_result(self):
        store = FeatureStore(dimensions=4)
        self.assertEqual(store.query([1.0, 0.0, 0.0, 0.0], top_k=5), [])
        self.assertEqual(len(store), 0)
        self.assertEqual(store.stats()["cells"], 0)

    def test_cosine_normalizes_magnitude(self):
        store = FeatureStore(dimensions=4)
        store.add([3.0, 0.0, 0.0, 0.0], feature_id=1, scale=1.0)
        store.add([10.0, 10.0, 0.0, 0.0], feature_id=2, scale=1.0)
        store.add([0.0, 0.0, 0.0, 0.0], feature_id=3)

        dot = store.query([1.0, 0.0, 0.0, 0.0], top_k=3, mode="dot")
        self.assertEqual([hit.feature_id for hit in dot], [2, 1, 3])
        self.assertAlmostEqual(dot[0].score, 10.0, places=6)
        self.assertAlmostEqual(dot[1].score, 3.0, places=6)
        self.assertEqual(dot[2].score, 0.0)

        cosine = store.query([1.0, 0.0, 0.0, 0.0], top_k=3, mode="cosine")
        self.assertEqual([hit.feature_id for hit in cosine], [1, 2, 3])
        self.assertAlmostEqual(cosine[0].score, 1.0, places=6)
        self.assertAlmostEqual(cosine[1].score, 1.0 / math.sqrt(2.0), places=6)
        self.assertEqual(cosine[2].score, 0.0)

        zero_query = store.query([0.0] * 4, top_k=3, mode="cosine")
        self.assertEqual([hit.score for hit in zero_query], [0.0, 0.0, 0.0])

    def test_superposition_ingest_decay_and_shared_cell(self):
        store = FeatureStore(dimensions=4)
        first = store.add([1.0, 0.0, 0.0, 0.0], feature_id=1,
                          leaf_id="leaf-a")
        updated = store.ingest([0.0, 1.0, 0.0, 0.0], feature_id=1,
                               weight=0.5, decay=0.5)
        self.assertIsInstance(updated, FeatureRef)
        self.assertEqual(updated.offset, first.offset)
        self.assertEqual(updated.cell, first.cell)
        for got, expected in zip(store.read_feature(1), (0.5, 0.5, 0.0, 0.0)):
            self.assertAlmostEqual(got, expected, places=6)

        shared = store.share_cell(2, source_feature_id=1, leaf_id="leaf-b")
        self.assertEqual(shared.offset, first.offset)
        self.assertEqual(shared.cell, first.cell)
        self.assertEqual(store.cell(first.cell).features, (1, 2))

        # Ingest through the alias mutates the single shared cell.
        store.ingest([0.0, 0.0, 1.0, 0.0], feature_id=2, weight=2.0, decay=0.0)
        for feature_id in (1, 2):
            for got, expected in zip(store.read_feature(feature_id),
                                     (0.0, 0.0, 2.0, 0.0)):
                self.assertAlmostEqual(got, expected, places=6)
        hits = store.query([0.0, 0.0, 1.0, 0.0], top_k=2, mode="cosine")
        self.assertEqual([hit.feature_id for hit in hits], [1, 2])
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertEqual(hits[0].cell, hits[1].cell)
        self.assertEqual(hits[0].offset, hits[1].offset)

    def test_weighted_replace_and_phase_metadata(self):
        store = FeatureStore(dimensions=3)
        ref = store.add([2.0, -4.0, 6.0], feature_id="concept", leaf_id=9,
                        weight=0.5, phase=1.5, scale=0.5)
        self.assertEqual(ref.scale, 0.5)
        self.assertEqual(ref.phase, 1.5)
        self.assertEqual(ref.leaf_id, 9)
        for got, expected in zip(store.read_feature("concept"),
                                 (1.0, -2.0, 3.0)):
            self.assertAlmostEqual(got, expected, places=6)

        replaced = store.ingest([1.0, 1.0, 1.0], feature_id="concept",
                                weight=2.0, decay=0.0, phase=0.25)
        self.assertEqual(replaced.phase, 0.25)
        for got in store.read_feature("concept"):
            self.assertAlmostEqual(got, 2.0, places=6)


class PhaseModeTests(unittest.TestCase):
    def test_phase_coherent_orders_by_phase_alignment(self):
        store = FeatureStore(dimensions=4)
        store.add([1.0, 0.0, 0.0, 0.0], feature_id=1, phase=0.0)
        store.add([1.0, 0.0, 0.0, 0.0], feature_id=2, phase=math.pi)
        store.add([1.0, 0.0, 0.0, 0.0], feature_id=3, phase=math.pi / 2.0)

        hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=3, mode="phase")
        self.assertEqual([hit.feature_id for hit in hits], [1, 3, 2])
        self.assertAlmostEqual(hits[0].score, 1.0, places=7)
        self.assertAlmostEqual(hits[1].score, 0.0, places=7)
        self.assertAlmostEqual(hits[2].score, -1.0, places=7)

        rotated = store.query([1.0, 0.0, 0.0, 0.0], top_k=3,
                              mode="phase", phase=math.pi)
        self.assertEqual([hit.feature_id for hit in rotated], [2, 3, 1])
        self.assertAlmostEqual(rotated[0].score, 1.0, places=7)
        self.assertAlmostEqual(rotated[1].score, 0.0, places=7)
        self.assertAlmostEqual(rotated[2].score, -1.0, places=7)

        # Dot and cosine ignore the stored phase, so the tie is by feature id.
        for mode in ("dot", "cosine"):
            tied = store.query([1.0, 0.0, 0.0, 0.0], top_k=3, mode=mode)
            self.assertEqual([hit.feature_id for hit in tied], [1, 2, 3])
            self.assertEqual([round(hit.score, 12) for hit in tied],
                             [1.0, 1.0, 1.0])

    def test_phase_mode_aliases(self):
        store = FeatureStore(dimensions=2)
        store.add([1.0, 0.0], feature_id=1, phase=0.0)
        store.add([1.0, 0.0], feature_id=2, phase=math.pi)
        expected = [hit.feature_id for hit in
                    store.query([1.0, 0.0], top_k=2, mode="phase")]
        for alias in ("phase_coherent", "phase-coherent", "resonance"):
            self.assertEqual([hit.feature_id for hit in
                              store.query([1.0, 0.0], top_k=2, mode=alias)],
                             expected)


class TopKAndDeterminismTests(unittest.TestCase):
    def test_top_k_order_and_bounds(self):
        store = FeatureStore(dimensions=4)
        for feature_id in (4, 2, 5, 1, 3):
            store.add([float(feature_id), 0.0, 0.0, 0.0],
                      feature_id=feature_id, scale=1.0)

        hits = store.query([1.0, 0.0, 0.0, 0.0], top_k=3)
        self.assertEqual([hit.feature_id for hit in hits], [5, 4, 3])
        self.assertEqual([hit.score for hit in hits], [5.0, 4.0, 3.0])

        everything = store.query([1.0, 0.0, 0.0, 0.0], top_k=10)
        self.assertEqual([hit.feature_id for hit in everything], [5, 4, 3, 2, 1])
        self.assertEqual(len(store.query([1.0, 0.0, 0.0, 0.0], top_k=1)), 1)

        with self.assertRaises(ValueError):
            store.query([1.0, 0.0, 0.0, 0.0], top_k=0)
        with self.assertRaises(ValueError):
            store.query([1.0, 0.0, 0.0, 0.0], top_k=-1)
        with self.assertRaises(TypeError):
            store.query([1.0, 0.0, 0.0, 0.0], top_k=1.5)
        with self.assertRaises(TypeError):
            store.query([1.0, 0.0, 0.0, 0.0], top_k=True)
        with self.assertRaises(ValueError):
            store.query([1.0, 0.0, 0.0, 0.0], mode="unknown")
        with self.assertRaises(ValueError):
            store.query([1.0, 0.0, 0.0], top_k=1)
        with self.assertRaises(ValueError):
            store.query([float("nan"), 0.0, 0.0, 0.0], top_k=1)

    def test_tie_break_is_deterministic_across_insertion_order(self):
        store = FeatureStore(dimensions=3)
        for feature_id in (5, 2, 9, 1):
            store.add([1.0, 0.0, 0.0], feature_id=feature_id, scale=1.0)
        expected = [1, 2, 5, 9]
        for _ in range(5):
            hits = store.query([1.0, 0.0, 0.0], top_k=4, mode="cosine")
            self.assertEqual([hit.feature_id for hit in hits], expected)

        # Equal logical vectors with different explicit scales still tie.
        scaled = FeatureStore(dimensions=2)
        scaled.add([1.0, 0.0], feature_id=10, scale=1.0)
        scaled.add([1.0, 0.0], feature_id=11, scale=0.5)
        for mode in ("dot", "cosine"):
            hits = scaled.query([1.0, 0.0], top_k=2, mode=mode)
            self.assertEqual([hit.feature_id for hit in hits], [10, 11])
            self.assertAlmostEqual(hits[0].score, hits[1].score, places=12)

    def test_repeated_queries_are_byte_for_byte_stable(self):
        store = FeatureStore(dimensions=6)
        for index in range(12):
            vector = [((index + 1) * (axis + 3)) % 11 / 11.0 for axis in range(6)]
            store.add(vector, feature_id=index, scale=0.001)
        baseline = [(hit.feature_id, hit.score) for hit in
                    store.query([0.31, 0.47, 0.11, 0.83, 0.29, 0.61],
                                top_k=5, mode="cosine")]
        for _ in range(4):
            measured = [(hit.feature_id, hit.score) for hit in
                        store.query([0.31, 0.47, 0.11, 0.83, 0.29, 0.61],
                                    top_k=5, mode="cosine")]
            self.assertEqual(measured, baseline)


class ErrorAndWrapperTests(unittest.TestCase):
    def test_dimension_and_argument_errors(self):
        store = FeatureStore(dimensions=4)
        store.add([1.0, 0.0, 0.0, 0.0], feature_id=1)
        with self.assertRaises(KeyError):
            store.ingest([0.0, 1.0, 0.0, 0.0], feature_id=99)
        with self.assertRaises(ValueError):
            store.ingest([0.0, 1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            store.ingest([0.0, 1.0, 0.0, 0.0], feature_id=1, cell=0)
        with self.assertRaises(IndexError):
            store.ingest([0.0, 1.0, 0.0, 0.0], cell=3)
        with self.assertRaises(IndexError):
            store.ingest([0.0, 1.0, 0.0, 0.0], cell=-1)
        with self.assertRaises(ValueError):
            store.share_cell(2)
        with self.assertRaises(KeyError):
            store.share_cell(2, source_feature_id=99)
        with self.assertRaises(ValueError):
            store.share_cell(1, source_feature_id=1)
        with self.assertRaises(ValueError):
            store.add([1.0, 0.0, 0.0, 0.0], feature_id=2, phase=float("nan"))
        with self.assertRaises(ValueError):
            store.add([1.0, 0.0, 0.0, 0.0], feature_id=2, scale=0.0)
        with self.assertRaises(ValueError):
            store.add([1.0, 0.0, 0.0, 0.0], feature_id=2, weight=float("inf"))
        with self.assertRaises(ValueError):
            store.ingest([0.0, 1.0, 0.0, 0.0], feature_id=1, decay=-1.0)
        with self.assertRaises(ValueError):
            FeatureArena(dimensions=0)
        with self.assertRaises(ValueError):
            FeatureArena(dimensions=4, dtype="float16")
        with self.assertRaises(ValueError):
            FeatureArena(dimensions=4, capacity=-1)

    def test_module_level_query_and_superpose_wrappers(self):
        store = FeatureStore(dimensions=3)
        superpose(store, [1.0, 0.0, 0.0], feature_id=1, scale=1.0)
        superpose(store, [0.0, 1.0, 0.0], feature_id=2, weight=0.5)
        self.assertEqual(store.ingest([0.0, 0.0, 1.0], cell=0,
                                      weight=2.0, decay=0.0).cell, 0)
        hits = query(store, [0.0, 0.0, 1.0], top_k=2, mode="cosine")
        self.assertEqual([hit.feature_id for hit in hits], [1, 2])
        self.assertIsInstance(hits[0], ResonanceHit)
        with self.assertRaises(TypeError):
            query(object(), [1.0, 0.0, 0.0])

    def test_stats_and_cell_reference(self):
        store = FeatureStore(dimensions=2, capacity=3, dtype="int8")
        store.add([1.0, 0.0], feature_id=1)
        store.add([0.0, 1.0], feature_id=2)
        stats = store.stats()
        self.assertEqual(stats["dimensions"], 2)
        self.assertEqual(stats["dtype"], "int8")
        self.assertEqual(stats["cells"], 2)
        self.assertEqual(stats["features"], 2)
        self.assertEqual(stats["used"], 2)
        self.assertEqual(stats["free"], 1)
        cell = store.cell(0)
        self.assertIsInstance(cell, CellRef)
        self.assertEqual(cell.features, (1,))
        self.assertEqual(store.feature(1)["offset"], cell.offset)
        self.assertEqual(list(store), [store.feature(1), store.feature(2)])


if __name__ == "__main__":
    unittest.main()
