"""Raw 32-bit cycle observable tests over real resident cells and a fake."""
import itertools
import random
import unittest
from types import MappingProxyType

from nanlmind_hermes.resident.cells import (
    HEAD_COUNT,
    HEAD_REGION_BYTES,
    PAIR_COUNT,
    RAIL_FLAG_INITIALIZED,
    RAIL_RECORD_BYTES,
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    REGION_BYTES,
    ResidentCells,
    address_of,
    canonical_address,
    identity_address,
    pair_index,
    reverse16,
)
from nanlmind_hermes.resident.cycle_observable import (
    DEFAULT_ROOT,
    PAIR_CANONICALS,
    ROOT_CYCLE_COUNT,
    WORD_MASK,
    CycleField,
    CycleFieldIdentityError,
    CycleFieldRangeError,
    CycleFieldWordError,
    CycleSample,
    PairWords,
    heads_from_canonical,
    pair_identity,
    word_distance,
)


def _pair_cell_bytes(cells, a, b):
    offset = cells.pair_cell_offset(a, b)
    return bytes(cells.buffer[offset:offset + RAIL_RECORD_BYTES * 2])


class FakeRawCells:
    """Duck-typed fake with an already raw oriented word store."""

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


class FakeRailCells:
    """Duck-typed fake that exposes the resident rail shape, not its buffer."""

    head_count = HEAD_COUNT
    pair_count = PAIR_COUNT

    def __init__(self):
        self.generation = 0
        self.rails = {}
        self.heads = {
            head: {"head_id": head, "identity": identity_address(head)}
            for head in range(HEAD_COUNT)
        }

    def read_head(self, head):
        return dict(self.heads[head])

    def read_rail(self, first, second, rail):
        index = pair_index(first, second)
        record = self.rails.get((index, rail))
        if record is not None:
            return record
        canonical = PAIR_CANONICALS[index]
        a, b = heads_from_canonical(canonical)
        mirror = reverse16(canonical)
        if rail == RAIL_S_PLUS:
            return self._record(a, b, mirror)
        if rail == RAIL_S_MINUS:
            return self._record(b, a, mirror)
        raise ValueError("bad rail")

    def write_rail(self, first, second, rail, record):
        index = pair_index(first, second)
        self.rails[(index, rail)] = record
        self.generation = max(self.generation, record.generation)

    @staticmethod
    def _record(source, target, mirror):
        from nanlmind_hermes.resident.cells import RailRecord
        return RailRecord(source, target, RAIL_FLAG_INITIALIZED,
                          0, 0, 0, mirror, 0, 0, 0)


class CachedIdentityTests(unittest.TestCase):
    def test_exhaustive_pair_mapping_from_cached_identity(self):
        self.assertEqual(len(PAIR_CANONICALS), PAIR_COUNT)
        self.assertEqual(len(set(PAIR_CANONICALS)), PAIR_COUNT)
        self.assertEqual(ROOT_CYCLE_COUNT, 32385)
        for index in range(PAIR_COUNT):
            identity = pair_identity(index)
            self.assertEqual(identity.index, index)
            self.assertEqual(identity.canonical, PAIR_CANONICALS[index])
            self.assertEqual(identity.mirror,
                             reverse16(identity.canonical))
            self.assertNotEqual(identity.first, identity.second)
            self.assertEqual(
                canonical_address(identity.first, identity.second),
                identity.canonical)
            self.assertEqual(
                pair_index(identity.first, identity.second), index)
            self.assertEqual(
                pair_index(identity.second, identity.first), index)
            self.assertEqual(
                address_of(identity.first, identity.second),
                identity.canonical)
            self.assertEqual(
                address_of(identity.second, identity.first),
                identity.mirror)

    def test_pair_identity_bounds(self):
        for bad in (-1, PAIR_COUNT, True, 1.5, "1"):
            with self.assertRaises(CycleFieldRangeError):
                pair_identity(bad)


class RealResidentCellsTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.field = CycleField(self.cells)

    def test_raw_word_preservation_in_both_orientations(self):
        words = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE,
                 0xFFFFFFFF, 0x12345678)
        for a, b in ((1, 2), (200, 5), (17, 136)):
            for word in words:
                self.assertEqual(
                    self.field.set_oriented(a, b, word), word)
                self.assertEqual(self.field.get_oriented(a, b), word)
                self.assertEqual(self.field.get_oriented(b, a), 0)
                state = self.cells.pair_state(a, b)
                chosen = (state.forward
                          if address_of(a, b) == state.canonical
                          else state.reverse)
                self.assertEqual(chosen.weight & WORD_MASK, word)
                self.assertEqual(chosen.amplitude, 0)
                self.assertEqual(chosen.phase, 0)
                self.assertEqual(chosen.source, a)
                self.assertEqual(chosen.target, b)
                self.assertEqual(chosen.mirror, state.mirror)
                self.assertTrue(chosen.initialized)

    def test_write_preserves_amplitude_phase_relation_and_head_bytes(self):
        identity = pair_identity(pair_index(11, 240))
        a, b = identity.first, identity.second
        self.cells.observe_pair(
            a, b, weight=1.5, phase=0.25, relation_id=0x12345678)
        before = self.cells.pair_state(a, b)
        heads_before = bytes(self.buffer[:HEAD_REGION_BYTES])

        self.field.set_oriented(a, b, 0xDEADBEEF)
        after = self.cells.pair_state(a, b)
        self.assertEqual(after.forward.weight & WORD_MASK, 0xDEADBEEF)
        self.assertEqual(after.forward.amplitude, before.forward.amplitude)
        self.assertEqual(after.forward.phase, before.forward.phase)
        self.assertEqual(
            after.forward.relation_chain, before.forward.relation_chain)
        self.assertEqual(after.forward.source, before.forward.source)
        self.assertEqual(after.forward.target, before.forward.target)
        self.assertEqual(after.forward.mirror, before.forward.mirror)
        self.assertEqual(after.reverse.weight, before.reverse.weight)
        self.assertEqual(bytes(self.buffer[:HEAD_REGION_BYTES]),
                         heads_before)

    def test_mirror_antisymmetry_modulo_2_pow_32(self):
        cases = (
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 0x80000000),
            (200, 5, 0xFFFFFFFF),
            (17, 136, 0xDEADBEEF),
        )
        for a, b, word in cases:
            self.field.set_oriented(a, b, word)
            mirror = self.field.mirror(a, b)
            self.assertEqual(mirror, (-word) & WORD_MASK)
            self.assertEqual(self.field.get_oriented(a, b), word)
            self.assertEqual(self.field.get_oriented(b, a), mirror)
            self.assertEqual(
                (self.field.get_oriented(a, b)
                 + self.field.get_oriented(b, a)) & WORD_MASK, 0)
            # Mirroring the reverse orientation writes the original word back.
            self.assertEqual(self.field.mirror(b, a), word)
            self.assertEqual(self.field.get_oriented(a, b), word)
            self.assertEqual(self.field.get_oriented(b, a),
                             (-word) & WORD_MASK)

    def test_cycle_and_triangle_are_exact_modular_sums(self):
        # Borrow across the modulus: 0xFFFFFFFF + 2 - 1 == 0 mod 2**32.
        self.field.set_oriented(0, 1, 0xFFFFFFFF)
        self.field.set_oriented(1, 2, 2)
        self.field.set_oriented(0, 2, 1)
        self.assertEqual(self.field.cycle(1, 2), 0)
        self.assertEqual(
            self.field.cycle(1, 2),
            (self.field.get_oriented(0, 1)
             + self.field.get_oriented(1, 2)
             - self.field.get_oriented(0, 2)) & WORD_MASK)

        a, b, c = 3, 7, 200
        words = {
            (a, b): 0x80000001,
            (b, c): 0xFFFFFFFE,
            (c, a): 0x7FFFFFFF,
        }
        for (left, right), word in words.items():
            self.field.set_oriented(left, right, word)
        expected = sum(words.values()) & WORD_MASK
        self.assertEqual(self.field.triangle(a, b, c), expected)
        self.assertEqual(
            self.field.triangle(a, b, c),
            (self.field.get_oriented(a, b)
             + self.field.get_oriented(b, c)
             + self.field.get_oriented(c, a)) & WORD_MASK)

    def test_exhaustive_root_cycle_gauge_cancellation(self):
        rng = random.Random(2029)
        for index in range(PAIR_COUNT):
            identity = pair_identity(index)
            self.field.set_oriented(
                identity.first, identity.second, rng.getrandbits(32))

        before = self.field.batch_cycles()
        self.assertEqual(len(before), ROOT_CYCLE_COUNT)
        gauge = [rng.getrandbits(32) for _ in range(HEAD_COUNT)]
        self.assertEqual(self.field.gauge_shift(gauge), 2 * PAIR_COUNT)
        after = self.field.batch_cycles()
        self.assertEqual(after, before)

        # Exact raw arithmetic was maintained: every cycle is a word too.
        self.assertTrue(all(0 <= sample.word <= WORD_MASK
                            for sample in after))
        self.assertEqual(
            max(word_distance(sample.word) for sample in after),
            max(word_distance(sample.word) for sample in before))

    def test_pure_gauge_shift_has_zero_residual_and_zero_cycles(self):
        rng = random.Random(41)
        gauge = [rng.getrandbits(32) for _ in range(HEAD_COUNT)]
        self.assertEqual(self.field.gauge_shift(gauge), 2 * PAIR_COUNT)
        self.assertEqual(self.field.gauge_residual(), 0)
        self.assertTrue(all(sample.word == 0
                            for sample in self.field.batch_cycles(limit=512)))

        # Applying the inverse gauge returns the zero field exactly.
        inverse = [(-value) & WORD_MASK for value in gauge]
        self.field.gauge_shift(inverse)
        self.assertEqual(self.field.gauge_residual(), 0)
        for a, b in ((0, 1), (1, 2), (17, 136), (200, 5)):
            self.assertEqual(self.field.get_oriented(a, b), 0)
            self.assertEqual(self.field.get_oriented(b, a), 0)

    def test_batch_api_is_bounded_and_accepts_selected_subset(self):
        all_cycles = self.field.batch_cycles()
        self.assertEqual(len(all_cycles), ROOT_CYCLE_COUNT)
        self.assertEqual(len({(sample.left, sample.right)
                              for sample in all_cycles}),
                         ROOT_CYCLE_COUNT)
        self.assertTrue(all(sample.root == DEFAULT_ROOT
                            and sample.left != DEFAULT_ROOT
                            and sample.right != DEFAULT_ROOT
                            and sample.left < sample.right
                            for sample in all_cycles))
        self.assertEqual(self.field.batch_cycles(limit=10),
                         all_cycles[:10])
        self.assertEqual(self.field.root_cycle_pairs(limit=3),
                         ((1, 2), (1, 3), (1, 4)))

        subset = ((1, 2), (3, 4), (10, 20))
        self.assertEqual(
            self.field.batch_cycles(subset=iter(subset)),
            tuple(CycleSample(0, i, j, 0) for i, j in subset))

        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(limit=ROOT_CYCLE_COUNT + 1)
        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(limit=-1)
        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(
                subset=itertools.repeat((1, 2), ROOT_CYCLE_COUNT + 1))
        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(subset=((0, 1),))
        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(subset=((1, 1),))
        with self.assertRaises(CycleFieldRangeError):
            self.field.batch_cycles(subset=(("1", "2"),))

    def test_snapshot_is_immutable_and_detached(self):
        self.field.set_oriented(1, 2, 0xDEADBEEF)
        snapshot = self.field.snapshot()
        self.assertIsInstance(snapshot, MappingProxyType)
        self.assertEqual(snapshot["head_count"], HEAD_COUNT)
        self.assertEqual(snapshot["pair_count"], PAIR_COUNT)
        self.assertEqual(snapshot["root_cycle_count"], ROOT_CYCLE_COUNT)
        self.assertEqual(len(snapshot["pairs"]), PAIR_COUNT)
        self.assertEqual(len(snapshot["cycles"]), ROOT_CYCLE_COUNT)
        index = pair_index(1, 2)
        pair = snapshot["pairs"][index]
        self.assertIsInstance(pair, PairWords)
        self.assertEqual(pair.index, index)
        self.assertEqual(pair.forward, 0xDEADBEEF)
        self.assertIsInstance(snapshot["cycles"][0], CycleSample)

        with self.assertRaises(TypeError):
            snapshot["generation"] = 1
        with self.assertRaises(TypeError):
            del snapshot["pairs"]
        with self.assertRaises(TypeError):
            snapshot["pairs"][0] = None
        with self.assertRaises(AttributeError):
            snapshot["pairs"][0].forward = 0
        with self.assertRaises(AttributeError):
            snapshot["cycles"][0].word = 0

        self.field.set_oriented(1, 2, 0)
        self.assertEqual(snapshot["pairs"][index].forward, 0xDEADBEEF)
        self.assertEqual(self.field.get_oriented(1, 2), 0)

    def test_invalid_heads_words_and_gauge_are_rejected_without_writes(self):
        self.field.set_oriented(1, 2, 7)
        pair_before = _pair_cell_bytes(self.cells, 1, 2)
        heads_before = bytes(self.buffer[:HEAD_REGION_BYTES])

        for args in ((1, 1), (-1, 2), (1, 256), (True, 2), (1, 2.0)):
            with self.assertRaises(CycleFieldRangeError):
                self.field.set_oriented(*args, 9)
            with self.assertRaises(CycleFieldRangeError):
                self.field.get_oriented(*args)
            with self.assertRaises(CycleFieldRangeError):
                self.field.mirror(*args)
        for bad_word in (-1, WORD_MASK + 1, True, 1.5, "1", None):
            with self.assertRaises(CycleFieldWordError):
                self.field.set_oriented(1, 2, bad_word)
        for bad_root in (-1, 256, True, 1.5):
            with self.assertRaises(CycleFieldRangeError):
                self.field.cycle(1, 2, root=bad_root)
            with self.assertRaises(CycleFieldRangeError):
                self.field.gauge_residual(root=bad_root)
        with self.assertRaises(CycleFieldRangeError):
            self.field.cycle(1, 2, root=1)
        with self.assertRaises(CycleFieldRangeError):
            self.field.cycle(1, 2, root=2)
        with self.assertRaises(CycleFieldRangeError):
            self.field.triangle(1, 1, 2)

        for bad_gauge in ([0] * (HEAD_COUNT - 1),
                          [0] * (HEAD_COUNT + 1),
                          [0] * (HEAD_COUNT - 1) + [True],
                          [0] * (HEAD_COUNT - 1) + [WORD_MASK + 1]):
            with self.assertRaises((CycleFieldWordError,
                                    CycleFieldRangeError)):
                self.field.gauge_shift(bad_gauge)

        self.assertEqual(self.field.get_oriented(1, 2), 7)
        self.assertEqual(_pair_cell_bytes(self.cells, 1, 2), pair_before)
        self.assertEqual(bytes(self.buffer[:HEAD_REGION_BYTES]), heads_before)

    def test_corrupt_permanent_head_identity_blocks_write(self):
        self.field.set_oriented(1, 2, 123)
        pair_before = _pair_cell_bytes(self.cells, 1, 2)
        offset = self.cells.head_record_offset(1)
        self.buffer[offset + 2:offset + 4] = b"\x00\x00"

        with self.assertRaises(CycleFieldIdentityError):
            self.field.set_oriented(1, 2, 456)
        self.assertEqual(_pair_cell_bytes(self.cells, 1, 2), pair_before)
        self.assertEqual(self.field.get_oriented(1, 2), 123)

    def test_corrupt_pair_rail_identity_is_not_rewritten(self):
        identity = pair_identity(pair_index(3, 4))
        self.field.set_oriented(3, 4, 0x11223344)
        offset = self.cells.rail_record_offset(
            identity.first, identity.second, RAIL_S_PLUS)
        # Corrupt only the rail source while leaving the cell otherwise intact.
        self.buffer[offset] = (identity.first + 1) % HEAD_COUNT
        pair_before = _pair_cell_bytes(self.cells, 3, 4)

        with self.assertRaises(CycleFieldIdentityError):
            self.field.set_oriented(3, 4, 0x55667788)
        self.assertEqual(_pair_cell_bytes(self.cells, 3, 4), pair_before)
        self.assertEqual(self.field.get_oriented(3, 4), 0x11223344)


class FakeCellsTests(unittest.TestCase):
    def test_raw_word_adapter_supports_full_api(self):
        fake = FakeRawCells()
        field = CycleField(fake)
        self.assertIs(field.cells, fake)
        field.set_oriented(1, 2, 0xFFFFFFFF)
        field.set_oriented(0, 1, 5)
        field.set_oriented(0, 2, 3)
        self.assertEqual(field.get_oriented(1, 2), 0xFFFFFFFF)
        self.assertEqual(field.cycle(1, 2), (5 + 0xFFFFFFFF - 3) & WORD_MASK)
        self.assertEqual(field.mirror(1, 2), 1)
        self.assertEqual(field.get_oriented(2, 1), 1)

        gauge = list(range(HEAD_COUNT))
        before = field.batch_cycles(limit=64)
        self.assertEqual(field.gauge_shift(gauge), 2 * PAIR_COUNT)
        self.assertEqual(field.batch_cycles(limit=64), before)
        self.assertGreater(fake.generation, 0)

    def test_fake_head_identity_denial(self):
        fake = FakeRawCells()
        fake.heads[9]["identity"] = 0
        field = CycleField(fake)
        with self.assertRaises(CycleFieldIdentityError):
            field.set_oriented(9, 10, 1)
        self.assertNotIn((9, 10), fake.words)

    def test_fake_rail_adapter_preserves_rails(self):
        fake = FakeRailCells()
        field = CycleField(fake)
        field.set_oriented(5, 200, 0xCAFEBABE)
        self.assertEqual(field.get_oriented(5, 200), 0xCAFEBABE)
        self.assertEqual(field.get_oriented(200, 5), 0)
        self.assertEqual(field.mirror(5, 200), (-0xCAFEBABE) & WORD_MASK)
        self.assertEqual(field.get_oriented(200, 5),
                         (-0xCAFEBABE) & WORD_MASK)
        identity = pair_identity(pair_index(5, 200))
        a, b = identity.first, identity.second
        forward = fake.read_rail(a, b, RAIL_S_PLUS)
        self.assertEqual(forward.weight & WORD_MASK, 0xCAFEBABE)
        self.assertEqual(forward.source, a)
        self.assertEqual(forward.target, b)


if __name__ == "__main__":
    unittest.main()
