"""Full 32,385-dimensional root-star cycle memory over resident cells.

These tests use the real 2 MiB :class:`ResidentCells` region and a duck-typed
fake.  They exercise the exhaustive C(255, 2) index map, every cycle lane, raw
word bit preservation, mirror/head identity checks, root-edge constraints and
integer-only dot products and gradient steps.
"""
import ast
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from nanlmind_hermes.resident import full_cycle_memory as fcm
from nanlmind_hermes.resident.cells import (
    HEAD_COUNT,
    HEAD_REGION_BYTES,
    PAIR_COUNT,
    RAIL_RECORD_BYTES,
    RAIL_S_PLUS,
    REGION_BYTES,
    ResidentCells,
    address_of,
    identity_address,
    pair_index,
)
from nanlmind_hermes.resident.cycle_observable import (
    PAIR_CANONICALS,
    WORD_MASK,
    heads_from_canonical,
)


def _word(index):
    return (index * 2654435761 + 0x9E3779B9) & WORD_MASK


class FakeRawCells:
    """Duck-typed fake with oriented raw word lanes and permanent heads."""

    head_count = HEAD_COUNT
    pair_count = PAIR_COUNT

    def __init__(self):
        self.generation = 0
        self.words = {}
        self.heads = {
            head: {"head_id": head, "identity": identity_address(head)}
            for head in range(HEAD_COUNT)
        }

    def read_head(self, head):
        return dict(self.heads[head])

    def read_raw_word(self, first, second):
        return self.words.get((first, second), 0)

    def write_raw_word(self, first, second, word):
        self.words[(first, second)] = word & WORD_MASK
        self.generation += 1


class FakeSmallCells(FakeRawCells):
    """A 256-pair imitation that must be rejected as a toy store."""

    pair_count = 256


class IndexMapTests(unittest.TestCase):
    def test_full_count_is_32385_not_256(self):
        self.assertEqual(fcm.FULL_CYCLE_COUNT, 32385)
        self.assertEqual(fcm.FULL_CYCLE_COUNT, 255 * 254 // 2)
        self.assertEqual(fcm.CYCLE_COUNT, fcm.FULL_CYCLE_COUNT)
        self.assertEqual(fcm.DIMENSION, fcm.FULL_CYCLE_COUNT)
        self.assertEqual(fcm.ROOT_CYCLE_COUNT, fcm.FULL_CYCLE_COUNT)
        self.assertNotEqual(fcm.FULL_CYCLE_COUNT, 256)
        self.assertEqual(len(tuple(fcm.iter_cycle_pairs())),
                         fcm.FULL_CYCLE_COUNT)

    def test_exhaustive_forward_and_inverse_mapping(self):
        seen = bytearray(fcm.FULL_CYCLE_COUNT)
        for index in range(fcm.FULL_CYCLE_COUNT):
            first, second = fcm.cycle_pair(index)
            self.assertGreaterEqual(first, 1)
            self.assertLess(first, second)
            self.assertLessEqual(second, 255)
            self.assertEqual(fcm.cycle_index(first, second), index)
            self.assertEqual(fcm.index_of(first, second), index)
            self.assertEqual(fcm.pair_of_index(index), (first, second))
            self.assertEqual(seen[index], 0)
            seen[index] = 1
        self.assertEqual(sum(seen), fcm.FULL_CYCLE_COUNT)

        count = 0
        for first, second in fcm.iter_cycle_pairs():
            index = fcm.cycle_index(first, second)
            self.assertEqual(fcm.cycle_pair(index), (first, second))
            count += 1
        self.assertEqual(count, fcm.FULL_CYCLE_COUNT)

        for first in range(1, 255):
            for second in range(first + 1, 256):
                index = fcm.cycle_index(first, second)
                self.assertEqual(fcm.cycle_pair(index), (first, second))
                self.assertEqual(fcm.cycle_index(second, first), index)

    def test_row_boundaries_and_endpoints(self):
        self.assertEqual(fcm.cycle_row_start(1), 0)
        self.assertEqual(fcm.cycle_row_start(2), 254)
        self.assertEqual(fcm.cycle_row_start(254), 32384)
        self.assertEqual(fcm.cycle_row_start(255), fcm.FULL_CYCLE_COUNT)
        self.assertEqual(fcm.cycle_pair(0), (1, 2))
        self.assertEqual(fcm.cycle_pair(253), (1, 255))
        self.assertEqual(fcm.cycle_pair(254), (2, 3))
        self.assertEqual(fcm.cycle_pair(fcm.FULL_CYCLE_COUNT - 1),
                         (254, 255))

    def test_invalid_indices_and_pairs_are_denied(self):
        for bad in (-1, fcm.FULL_CYCLE_COUNT, True, 1.5, "1", None):
            with self.assertRaises(fcm.FullCycleRangeError):
                fcm.cycle_pair(bad)
        for bad in (-1, 0, 256, True, 1.5, "1", None):
            with self.assertRaises(fcm.FullCycleRangeError):
                fcm.cycle_row_start(bad)
        for args in ((0, 1), (1, 0), (1, 1), (255, 255),
                     (-1, 2), (1, 256), (True, 2), (1, 2.0)):
            with self.assertRaises(fcm.FullCycleError):
                fcm.cycle_index(*args)
        with self.assertRaises(fcm.FullCycleRootError):
            fcm.cycle_index(0, 1)

    def test_signed_word_helpers(self):
        cases = ((0, 0), (1, 1), (0x7FFFFFFF, 0x7FFFFFFF),
                 (0x80000000, -0x80000000),
                 (0xFFFFFFFF, -1), (0xDEADBEEF, -559038737))
        for word, signed in cases:
            self.assertEqual(fcm.signed_word(word), signed)
            self.assertEqual(fcm.word_from_signed(signed), word)
        for bad in (-1, WORD_MASK + 1, True, 1.5, "1"):
            with self.assertRaises(fcm.FullCycleWordError):
                fcm.signed_word(bad)
        for bad in (fcm.I32_MIN - 1, fcm.I32_MAX + 1, True, 1.5):
            with self.assertRaises(fcm.FullCycleError):
                fcm.word_from_signed(bad)


class RealResidentCellsTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.memory = fcm.FullCycleMemory(self.cells)

    def _rail_word(self, first, second):
        state = self.cells.pair_state(first, second)
        chosen = (state.forward
                  if address_of(first, second) == state.canonical
                  else state.reverse)
        return chosen.weight & WORD_MASK

    def test_real_region_is_2_mib_and_full_dimension(self):
        self.assertEqual(REGION_BYTES, 2 * 1024 * 1024)
        self.assertEqual(len(self.memory), fcm.FULL_CYCLE_COUNT)
        self.assertEqual(self.memory.dimension, 32385)
        self.assertEqual(self.memory.cycle_count, 32385)
        self.assertEqual(self.memory.head_count, 256)
        self.assertEqual(self.memory.pair_count, 32640)
        self.assertEqual(self.memory.validate_root_edges(), 510)
        report = self.memory.audit()
        self.assertTrue(report)
        self.assertEqual(report["cycles_checked"], 32385)
        self.assertEqual(report["root_edges_checked"], 510)

    def test_exhaustive_write_and_read_every_cycle(self):
        words = tuple(_word(index) for index in range(fcm.FULL_CYCLE_COUNT))
        self.assertEqual(self.memory.write_all(words), fcm.FULL_CYCLE_COUNT)
        self.assertEqual(self.memory.read_all(), words)
        self.assertEqual(len(self.memory.read_all()), 32385)
        self.assertEqual(int(self.memory.occupancy), 32385)
        self.assertEqual(self.memory.occupancy(), 32385)

        for index, (first, second) in enumerate(fcm.iter_cycle_pairs()):
            self.assertEqual(self.memory.get_word(first, second), words[index])
            self.assertEqual(self.memory.get_word(second, first),
                             (-words[index]) & WORD_MASK)
        for index in (0, 12345, fcm.FULL_CYCLE_COUNT - 1):
            first, second = fcm.cycle_pair(index)
            self.assertEqual(self.memory.cycle(first, second), words[index])

        report = self.memory.audit()
        self.assertTrue(report)
        self.assertEqual(report["cycles_checked"], fcm.FULL_CYCLE_COUNT)
        self.assertEqual(report["mirror_mismatches"], 0)
        self.assertEqual(report["root_conflicts"], 0)

    def test_raw_word_bits_and_metadata_are_preserved(self):
        cases = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF,
                 0xDEADBEEF, 0x7FC00000, 0x12345678)
        heads_before = bytes(self.buffer[:HEAD_REGION_BYTES])
        for first, second in ((1, 2), (200, 5), (17, 136)):
            state_before = self.cells.pair_state(first, second)
            for word in cases:
                self.assertEqual(self.memory.set_word(first, second, word),
                                 word)
                self.assertEqual(self.memory.get_word(first, second), word)
                self.assertEqual(self.memory.get_word(second, first),
                                 (-word) & WORD_MASK)
                self.assertEqual(self._rail_word(first, second), word)
                self.assertEqual(self._rail_word(second, first),
                                 (-word) & WORD_MASK)
                state = self.cells.pair_state(first, second)
                self.assertEqual(state.forward.source,
                                 state_before.forward.source)
                self.assertEqual(state.forward.target,
                                 state_before.forward.target)
                self.assertEqual(state.forward.mirror,
                                 state_before.forward.mirror)
                self.assertEqual(state.reverse.source,
                                 state_before.reverse.source)
                self.assertEqual(state.reverse.target,
                                 state_before.reverse.target)
                self.assertEqual(state.reverse.mirror,
                                 state_before.reverse.mirror)
                self.assertEqual(bytes(self.buffer[:HEAD_REGION_BYTES]),
                                 heads_before)

    def test_root_edges_are_zeroed_and_cycle_checks_them(self):
        for head in (1, 7, 255):
            state = self.cells.pair_state(0, head)
            self.assertEqual(state.forward.weight, 0)
            self.assertEqual(state.reverse.weight, 0)

        self.memory.field.set_oriented(0, 7, 0x1234)
        with self.assertRaises(fcm.FullCycleRootError):
            self.memory.cycle(7, 2)
        with self.assertRaises(fcm.FullCycleRootError):
            self.memory.validate_root_edges()
        report = self.memory.audit(full=False)
        self.assertFalse(report)
        self.assertFalse(report["ok"])
        self.assertEqual(report["root_conflicts"], 1)

        self.assertGreaterEqual(self.memory.zero_root_edges(), 1)
        self.assertTrue(self.memory.audit(full=False))
        self.assertEqual(self.memory.cycle(7, 2), 0)

        # A dirty root in a reused region is cleared by construction.
        self.memory.field.set_oriented(0, 9, 77)
        reused = fcm.FullCycleMemory(self.cells)
        self.assertTrue(reused.audit(full=False))
        self.assertEqual(reused.get_word(9, 10), 0)

    def test_set_word_rejects_root_and_same_head_before_writing(self):
        for i, j in ((0, 1), (1, 0), (0, 255), (255, 0)):
            with self.assertRaises(fcm.FullCycleRootError):
                self.memory.set_word(i, j, 0)
        for i, j in ((1, 1), (255, 255)):
            with self.assertRaises(fcm.FullCycleRangeError):
                self.memory.set_word(i, j, 1)
        self.assertEqual(self.memory.get_word(1, 2), 0)

    def test_predict_sparse_dense_and_signed_dot_product(self):
        self.memory.set_word(1, 2, 3)
        self.memory.set_word(1, 3, 0xFFFFFFFF)   # signed -1
        self.memory.set_word(4, 5, 0x80000000)   # signed -2**31
        self.assertEqual(self.memory.predict({(1, 2): 2}), 6)
        self.assertEqual(self.memory.predict({(1, 2): 2, (1, 3): 1}), 5)
        self.assertEqual(self.memory.predict({0: 2, 1: 1}), 5)
        # A reversed pair addresses the negated coefficient.
        self.assertEqual(self.memory.predict({(2, 1): 2}), -6)
        self.assertEqual(self.memory.predict({(1, 3): 0}), 0)
        self.assertEqual(self.memory.predict([0] * 32385), 0)
        self.assertEqual(
            self.memory.predict({(4, 5): 1}),
            fcm.signed_word(0x80000000))
        self.assertEqual(
            self.memory.predict_word({(1, 3): 1}), WORD_MASK)

        dense = [0] * fcm.FULL_CYCLE_COUNT
        dense[0] = 2
        dense[1] = 1
        self.assertEqual(self.memory.predict(dense), 5)
        self.assertEqual(self.memory.predict(iter(dense)), 5)

    def test_sparse_gradient_step_is_exact_and_sparse(self):
        result = self.memory.update({(1, 2): 2, 1: 1}, 10, 1)
        self.assertEqual(int(result), 2)
        self.assertEqual(result.updated, 2)
        self.assertEqual(result.prediction, 0)
        self.assertEqual(result.target, 10)
        self.assertEqual(result.error, 10)
        self.assertEqual(result.loss, 100)
        self.assertEqual(self.memory.get_word(1, 2), 20)
        self.assertEqual(self.memory.get_word(1, 3), 10)
        self.assertEqual(self.memory.get_word(2, 1), (-20) & WORD_MASK)
        self.assertEqual(self.memory.get_word(2, 4), 0)
        self.assertEqual(self.memory.predict({(1, 2): 2, 1: 1}), 50)

        no_change = self.memory.update({(1, 2): 2, 1: 1}, 50, 1)
        self.assertEqual(int(no_change), 0)
        self.assertEqual(no_change.error, 0)
        self.assertEqual(self.memory.get_word(1, 2), 20)

        zero_lr = self.memory.update({(1, 2): 1}, 100, 0)
        self.assertEqual(int(zero_lr), 0)
        self.assertEqual(self.memory.get_word(1, 2), 20)

    def test_gradient_overflow_is_denied_without_a_write(self):
        self.memory.set_word(1, 2, 0x7FFFFFF0)
        with self.assertRaises(fcm.FullCycleOverflowError):
            self.memory.update({(1, 2): 1}, 0x7FFFFFFF, 2)
        self.assertEqual(self.memory.get_word(1, 2), 0x7FFFFFF0)
        with self.assertRaises(fcm.FullCycleOverflowError):
            self.memory.update({(1, 2): 1}, 1, fcm.I32_MAX + 1)

    def test_write_all_sequence_and_mapping_roundtrip(self):
        pairs = tuple(fcm.iter_cycle_pairs())
        mapping = {pair: _word(index) for index, pair in enumerate(pairs)}
        self.assertEqual(self.memory.write_all(mapping=mapping), 32385)
        expected = tuple(_word(index) for index in range(32385))
        self.assertEqual(self.memory.read_all(), expected)

        reversed_mapping = {
            (second, first): (-word) & WORD_MASK
            for (first, second), word in mapping.items()
        }
        self.assertEqual(self.memory.write_all(reversed_mapping), 32385)
        self.assertEqual(self.memory.read_all(3), expected[:3])
        self.assertEqual(self.memory.get_word(254, 255), expected[-1])

    def test_write_all_denies_subsets_toys_and_unknown_pairs(self):
        self.memory.set_word(1, 2, 0xABCDEF01)
        with self.assertRaises(fcm.FullCycleDimensionError):
            self.memory.write_all([0] * 256)
        with self.assertRaises(fcm.FullCycleDimensionError):
            self.memory.write_all([0] * (fcm.FULL_CYCLE_COUNT - 1))
        with self.assertRaises(fcm.FullCycleDimensionError):
            self.memory.write_all([0] * (fcm.FULL_CYCLE_COUNT + 1))
        self.assertEqual(self.memory.get_word(1, 2), 0xABCDEF01)

        partial = {(1, 2): 1}
        with self.assertRaises(fcm.FullCycleDimensionError):
            self.memory.write_all(mapping=partial)
        self.assertEqual(self.memory.get_word(1, 2), 0xABCDEF01)

        full = {pair: 0 for pair in fcm.iter_cycle_pairs()}
        broken = dict(full)
        del broken[(254, 255)]
        with self.assertRaises(fcm.FullCycleDimensionError):
            self.memory.write_all(mapping=broken)
        broken = dict(full)
        broken[(1, 256)] = 1
        with self.assertRaises(fcm.FullCycleRangeError):
            self.memory.write_all(mapping=broken)
        broken = dict(full)
        broken[(0, 1)] = 1
        with self.assertRaises(fcm.FullCycleRootError):
            self.memory.write_all(mapping=broken)
        self.assertEqual(self.memory.get_word(1, 2), 0xABCDEF01)

    def test_read_all_is_bounded_and_validated(self):
        self.memory.set_word(1, 2, 11)
        self.memory.set_word(1, 3, 22)
        self.assertEqual(self.memory.read_all(2), (11, 22))
        self.assertEqual(self.memory.read_all(bounded=2), (11, 22))
        self.assertEqual(self.memory.read_all(0), ())
        for bad in (-1, fcm.FULL_CYCLE_COUNT + 1, True, 1.5):
            with self.assertRaises(fcm.FullCycleRangeError):
                self.memory.read_all(bad)
        with self.assertRaises(fcm.FullCycleRangeError):
            self.memory.read_all(1, bounded=1)

    def test_invalid_words_features_and_heads_are_denied(self):
        for bad in (-1, WORD_MASK + 1, True, 1.5, "1", None):
            with self.assertRaises(fcm.FullCycleWordError):
                self.memory.set_word(1, 2, bad)
        for args in ((-1, 2), (1, 256), (True, 2), (1, 2.5)):
            with self.assertRaises(fcm.FullCycleRangeError):
                self.memory.get_word(*args)
            with self.assertRaises(fcm.FullCycleRangeError):
                self.memory.set_word(*args, 1)
        for features in (
                {(0, 1): 1}, {(1, 1): 1}, {(1, 256): 1}, {(-1, 2): 1},
                {(1, 2): 1.5}, {(1, 2): True}, [0] * 256,
                [(1, 1, 1)], {"pair": 1}):
            with self.assertRaises(fcm.FullCycleError):
                self.memory.predict(features)
        self.assertEqual(self.memory.get_word(1, 2), 0)

    def test_head_and_pair_identity_denial(self):
        self.memory.set_word(3, 4, 0x11223344)
        offset = self.cells.head_record_offset(1)
        identity_bytes = bytes(self.buffer[offset + 2:offset + 4])
        self.buffer[offset + 2:offset + 4] = b"\x00\x00"
        with self.assertRaises(fcm.FullCycleIdentityError):
            self.memory.set_word(1, 2, 0x55667788)
        self.assertEqual(self.memory.get_word(1, 2), 0)
        self.buffer[offset + 2:offset + 4] = identity_bytes

        index = pair_index(3, 4)
        first, second = heads_from_canonical(PAIR_CANONICALS[index])
        rail_offset = self.cells.rail_record_offset(
            first, second, RAIL_S_PLUS)
        original = self.buffer[rail_offset]
        self.buffer[rail_offset] = (first + 1) % HEAD_COUNT
        pair_before = bytes(self.buffer[
            self.cells.pair_cell_offset(3, 4):
            self.cells.pair_cell_offset(3, 4) + 2 * RAIL_RECORD_BYTES])
        with self.assertRaises(fcm.FullCycleIdentityError):
            self.memory.set_word(3, 4, 0x99AABBCC)
        self.assertEqual(bytes(self.buffer[
            self.cells.pair_cell_offset(3, 4):
            self.cells.pair_cell_offset(3, 4) + 2 * RAIL_RECORD_BYTES]),
            pair_before)
        self.buffer[rail_offset] = original
        self.assertEqual(self.memory.get_word(3, 4), 0x11223344)

    def test_snapshot_is_immutable_and_detached(self):
        self.memory.set_word(1, 2, 0xDEADBEEF)
        snapshot = self.memory.snapshot()
        self.assertIsInstance(snapshot, MappingProxyType)
        self.assertEqual(snapshot["dimension"], 32385)
        self.assertEqual(snapshot["cycle_count"], 32385)
        self.assertEqual(len(snapshot["words"]), 32385)
        self.assertEqual(len(snapshot["cycles"]), 32385)
        self.assertEqual(len(snapshot["pairs"]), 32385)
        self.assertEqual(len(snapshot["word_map"]), 32385)
        self.assertEqual(snapshot["words"][0], 0xDEADBEEF)
        self.assertEqual(snapshot["cycles"][0].pair, (1, 2))
        self.assertEqual(snapshot["cycles"][0].signed, -559038737)
        self.assertEqual(snapshot["word_map"][(1, 2)], 0xDEADBEEF)
        self.assertEqual(snapshot["root_edge_count"], 255)
        self.assertEqual(snapshot["root_edges"][0][2:], (0, 0))
        self.assertEqual(snapshot["root_cycles"], snapshot["cycles"])

        with self.assertRaises(TypeError):
            snapshot["generation"] = 1
        with self.assertRaises(TypeError):
            snapshot["words"][0] = 0
        with self.assertRaises(TypeError):
            snapshot["word_map"][(1, 2)] = 0
        with self.assertRaises(AttributeError):
            snapshot["cycles"][0].word = 0

        self.memory.set_word(1, 2, 0)
        self.assertEqual(snapshot["words"][0], 0xDEADBEEF)
        self.assertEqual(self.memory.get_word(1, 2), 0)

    def test_occupancy_counts_nonzero_cycles_only(self):
        self.assertEqual(self.memory.occupancy, 0)
        self.assertEqual(self.memory.occupancy(), 0)
        self.assertEqual(self.memory.occupancy_report()["free"], 32385)
        self.memory.set_word(1, 2, 1)
        self.memory.set_word(2, 3, 0xFFFFFFFF)
        self.memory.set_word(3, 4, 0)
        self.assertEqual(self.memory.occupancy, 2)
        self.assertEqual(self.memory.occupancy(), 2)
        self.assertEqual(self.memory.occupied, 2)
        self.assertEqual(self.memory.occupancy.total, 32385)
        self.assertEqual(self.memory.occupancy.free, 32383)
        self.assertEqual(self.memory.occupancy_report()["occupied"], 2)

    def test_constructor_rejects_a_256_pair_toy_store(self):
        with self.assertRaises(fcm.FullCycleDimensionError):
            fcm.FullCycleMemory(FakeSmallCells())

    def test_module_uses_no_binary_float_identity(self):
        source = Path(fcm.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(
            [node for node in ast.walk(tree)
             if isinstance(node, ast.Constant)
             and type(node.value) is float], [])
        self.assertEqual(
            [node for node in ast.walk(tree)
             if isinstance(node, ast.Name) and node.id == "float"], [])
        self.assertEqual(
            [node for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and node.attr == "float"], [])

        def forbidden(*_args, **_kwargs):
            raise AssertionError("full cycle memory used a binary float")

        with mock.patch("builtins.float", side_effect=forbidden):
            self.memory.set_word(1, 2, 0x7FC00000)
            self.assertIsInstance(self.memory.get_word(1, 2), int)
            self.assertIsInstance(
                self.memory.cycle(1, 2), int)
            self.assertIsInstance(
                self.memory.predict({(1, 2): 1}), int)
            self.assertIsInstance(
                int(self.memory.update({(1, 2): 1}, 3, 1)), int)
            self.assertEqual(int(self.memory.occupancy), 1)
            self.assertTrue(self.memory.audit(full=False))
            self.assertIsInstance(self.memory.snapshot()["words"], tuple)


class FakeRawCellsTests(unittest.TestCase):
    def test_fake_supports_full_basis_and_raw_words(self):
        fake = FakeRawCells()
        memory = fcm.FullCycleMemory(fake)
        self.assertEqual(len(memory), 32385)
        self.assertEqual(memory.validate_root_edges(), 510)

        memory.set_word(1, 2, 0xFFFFFFFF)
        self.assertEqual(memory.get_word(1, 2), 0xFFFFFFFF)
        self.assertEqual(memory.get_word(2, 1), 1)
        self.assertEqual(fake.words[(1, 2)], 0xFFFFFFFF)
        self.assertEqual(fake.words[(2, 1)], 1)
        self.assertTrue(memory.audit())

        words = tuple(_word(index) for index in range(fcm.FULL_CYCLE_COUNT))
        self.assertEqual(memory.write_all(words), 32385)
        self.assertEqual(memory.read_all(), words)
        self.assertEqual(memory.predict({(1, 2): 1}), fcm.signed_word(words[0]))
        self.assertEqual(int(memory.update({(1, 2): 1}, 0, 1)), 1)
        self.assertEqual(memory.get_word(1, 2),
                         fcm.word_from_signed(fcm.signed_word(words[0])
                                              - fcm.signed_word(words[0])))
        self.assertEqual(int(memory.occupancy), 32384)

        snapshot = memory.snapshot()
        self.assertEqual(len(snapshot["words"]), 32385)
        self.assertEqual(snapshot["words"][-1], _word(32384))
        self.assertTrue(memory.audit())

    def test_fake_head_identity_denial(self):
        fake = FakeRawCells()
        fake.heads[9]["identity"] = 0
        with self.assertRaises(fcm.FullCycleIdentityError):
            fcm.FullCycleMemory(fake)


if __name__ == "__main__":
    unittest.main()
