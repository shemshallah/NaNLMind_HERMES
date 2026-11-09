"""Resident head and pair-cell region tests.

Covers the permanent head identity, canonical/mirror pair addressing, the two
independent rails, additive superposition-friendly updates and decay, byte-for-
byte deterministic readback (including mmap and a non-zero base offset), and
bounded index rejection.
"""
import mmap
import struct
import tempfile
import unittest

from nanlmind_hermes.resident import cells as cells_module
from nanlmind_hermes.resident.cells import (
    FIXED_ONE,
    HEAD_COUNT,
    HEAD_RECORD_BYTES,
    HEAD_RECORD_SIZE,
    HEAD_REGION_BYTES,
    HEAD_STRUCT,
    RAIL_S_MINUS,
    RAIL_S_PLUS,
    RAIL_RECORD_BYTES,
    RAIL_RECORD_SIZE,
    RAIL_STRUCT,
    PAIR_CELL_BYTES,
    PAIR_CELL_SIZE,
    PAIR_COUNT,
    PAIR_REGION_BYTES,
    PHASE_ONE,
    REGION_BYTES,
    ResidentCells,
    address_of,
    canonical_address,
    format_cells,
    identity_address,
    mirror_address,
    pair_index,
    reverse8,
    reverse16,
    to_fixed,
    to_phase,
)


def forward_heads(a: int, b: int) -> tuple[int, int]:
    """Return the ordered heads whose address is the canonical address."""
    if address_of(a, b) == canonical_address(a, b):
        return a, b
    return b, a


def canonical_heads(canonical: int) -> tuple[int, int]:
    return canonical >> 8, reverse8(canonical & 0xFF)


class RecordSizingTests(unittest.TestCase):
    def test_fixed_record_sizes_and_region_layout(self):
        self.assertEqual(HEAD_RECORD_SIZE, 32)
        self.assertEqual(RAIL_RECORD_SIZE, 32)
        self.assertEqual(PAIR_CELL_SIZE, 64)
        self.assertEqual(HEAD_RECORD_BYTES, HEAD_STRUCT.size)
        self.assertEqual(RAIL_RECORD_BYTES, RAIL_STRUCT.size)
        self.assertEqual(HEAD_REGION_BYTES, 8_192)
        self.assertEqual(PAIR_COUNT, 32_640)
        self.assertEqual(PAIR_REGION_BYTES, 2_088_960)
        self.assertEqual(REGION_BYTES, 2_097_152)
        self.assertEqual(REGION_BYTES, HEAD_REGION_BYTES + PAIR_REGION_BYTES)
        self.assertEqual(HEAD_STRUCT.format, "<BBHiIIQII")
        self.assertEqual(RAIL_STRUCT.format, "<BBHiiiHII2xI")
        layout = ResidentCells.layout()
        self.assertEqual(layout["heads"], 0)
        self.assertEqual(layout["pairs"], HEAD_REGION_BYTES)
        self.assertEqual(layout["region_bytes"], REGION_BYTES)
        self.assertEqual(HEAD_COUNT, 256)

    def test_format_helper_returns_initialized_region(self):
        buffer = bytearray(REGION_BYTES)
        cells = format_cells(buffer)
        self.assertIsInstance(cells, ResidentCells)
        self.assertEqual(cells.read_head(0).head_id, 0)
        self.assertEqual(cells.read_head(255).identity, identity_address(255))


class HeadIdentityTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)

    def test_all_heads_receive_permanent_identity(self):
        identities = set()
        for head in range(HEAD_COUNT):
            record = self.cells.read_head(head)
            self.assertEqual(record.head_id, head)
            self.assertEqual(record.identity, identity_address(head))
            self.assertEqual(record.identity, (head << 8) | reverse8(head))
            self.assertEqual(record.score, 0)
            self.assertEqual(record.state, 0)
            self.assertEqual(record.generation, 0)
            identities.add(record.identity)
        self.assertEqual(len(identities), HEAD_COUNT)

    def test_updates_never_overwrite_head_identity(self):
        head = 17
        offset = self.cells.head_record_offset(head)
        before_id = bytes(self.buffer[offset:offset + 1])
        before_identity = bytes(self.buffer[offset + 2:offset + 4])
        updated = self.cells.update_head(
            head, score_delta=2.5, state_delta=3, flags=0x0A,
            leaf_chain=0x1122334455667788, relation_chain=0xDEADBEEF)
        self.assertEqual(updated.head_id, head)
        self.assertEqual(updated.identity, identity_address(head))
        self.assertEqual(updated.score, to_fixed(2.5))
        self.assertEqual(updated.score_value, 2.5)
        self.assertEqual(updated.state, 3)
        self.assertEqual(updated.flags, 0x0A)
        self.assertEqual(updated.generation, 1)
        self.assertEqual(updated.leaf_chain, 0x1122334455667788)
        self.assertEqual(updated.relation_chain, 0xDEADBEEF)
        self.assertEqual(bytes(self.buffer[offset:offset + 1]), before_id)
        self.assertEqual(bytes(self.buffer[offset + 2:offset + 4]), before_identity)

        raw = HEAD_STRUCT.unpack_from(self.buffer, offset)
        self.assertEqual(tuple(updated), raw)

        again = self.cells.update_head(head, score_delta=-0.5, state_delta=2)
        self.assertEqual(again.head_id, head)
        self.assertEqual(again.identity, identity_address(head))
        self.assertEqual(again.score, to_fixed(2.0))
        self.assertEqual(again.state, 5)
        self.assertEqual(again.flags, 0x0A)
        self.assertEqual(again.leaf_chain, 0x1122334455667788)
        self.assertEqual(again.relation_chain, 0xDEADBEEF)
        self.assertEqual(again.generation, 2)

    def test_uninitialized_head_is_rejected(self):
        raw = bytearray(REGION_BYTES)
        uninitialized = ResidentCells(raw)
        with self.assertRaises(ValueError):
            uninitialized.update_head(17, score_delta=1.0)


class PairMirrorTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.a, self.b = 17, 136

    def test_canonical_and_reverse16_mirror_are_symmetric(self):
        canonical = canonical_address(self.a, self.b)
        mirror = reverse16(canonical)
        self.assertEqual(canonical_address(self.b, self.a), canonical)
        self.assertEqual(mirror_address(canonical), mirror)
        self.assertEqual(mirror_address(mirror), canonical)
        self.assertEqual(pair_index(self.a, self.b), pair_index(self.b, self.a))
        self.assertEqual(
            self.cells.pair_cell_offset(self.a, self.b),
            self.cells.pair_cell_offset(self.b, self.a))

        state = self.cells.pair_state(self.a, self.b)
        self.assertEqual(state.canonical, canonical)
        self.assertEqual(state.mirror, mirror)
        self.assertEqual(state.index, pair_index(self.a, self.b))
        self.assertEqual(state, self.cells.pair_state(self.b, self.a))

    def test_both_rails_carry_coherent_mirror_metadata(self):
        canonical = canonical_address(self.a, self.b)
        mirror = reverse16(canonical)
        first, second = canonical_heads(canonical)
        state = self.cells.pair_state(self.a, self.b)

        self.assertEqual(state.forward.source, first)
        self.assertEqual(state.forward.target, second)
        self.assertEqual(state.reverse.source, second)
        self.assertEqual(state.reverse.target, first)
        for rail in (state.forward, state.reverse):
            self.assertTrue(rail.initialized)
            self.assertEqual(rail.mirror, mirror)
            self.assertEqual(rail.weight, 0)
            self.assertEqual(rail.amplitude, 0)
            self.assertEqual(rail.phase, 0)
            self.assertEqual(rail.generation, 0)

    def test_observing_reverse_orientation_keeps_one_canonical_cell(self):
        x, y = forward_heads(self.a, self.b)
        forward_state = self.cells.observe_pair(
            x, y, weight=1.0, phase=0.25, relation_id=7)
        reverse_state = self.cells.observe_pair(
            y, x, weight=1.0, phase=0.25, relation_id=8)
        self.assertEqual(forward_state.index, reverse_state.index)
        self.assertEqual(forward_state.canonical, reverse_state.canonical)
        self.assertEqual(forward_state.mirror, reverse_state.mirror)
        self.assertEqual(forward_state.forward.mirror, forward_state.mirror)
        self.assertEqual(reverse_state.reverse.mirror, reverse_state.mirror)
        self.assertEqual(
            forward_state.forward.generation,
            forward_state.reverse.generation)


class IndependentRailTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.x, self.y = forward_heads(5, 200)

    def test_forward_and_reverse_accumulate_independently(self):
        self.cells.observe_pair(
            self.x, self.y, weight=1.5, phase=0.25, relation_id=101)
        state = self.cells.pair_state(self.x, self.y)
        self.assertEqual(state.forward.weight, to_fixed(1.5))
        self.assertEqual(state.forward.amplitude, to_fixed(1.5))
        self.assertEqual(state.forward.phase, to_phase(0.25))
        self.assertEqual(state.forward.relation_chain, 101)
        self.assertEqual(state.reverse.weight, 0)
        self.assertEqual(state.reverse.amplitude, 0)
        self.assertEqual(state.reverse.phase, 0)

        state = self.cells.observe_pair(
            self.y, self.x, weight=2.5, phase=-0.25, relation_id=202)
        self.assertEqual(state.forward.weight, to_fixed(1.5))
        self.assertEqual(state.forward.phase, to_phase(0.25))
        self.assertEqual(state.forward.relation_chain, 101)
        self.assertEqual(state.reverse.weight, to_fixed(2.5))
        self.assertEqual(state.reverse.amplitude, to_fixed(2.5))
        self.assertEqual(state.reverse.phase, to_phase(-0.25))
        self.assertEqual(state.reverse.relation_chain, 202)
        self.assertEqual(state.total_weight, to_fixed(4.0))
        self.assertEqual(state.total_amplitude, to_fixed(4.0))
        self.assertEqual(state.forward.generation, state.reverse.generation)
        self.assertEqual(state.forward.generation, 2)

    def test_explicit_rail_override_does_not_touch_other_rail(self):
        self.cells.observe_pair(
            self.y, self.x, weight=0.75, phase=0.5, relation_id=9,
            rail=RAIL_S_PLUS)
        state = self.cells.pair_state(self.x, self.y)
        self.assertEqual(state.forward.weight, to_fixed(0.75))
        self.assertEqual(state.forward.phase, to_phase(0.5))
        self.assertEqual(state.reverse.weight, 0)
        self.assertEqual(state.reverse.phase, 0)


class AdditiveDecayTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)
        self.x, self.y = forward_heads(11, 240)

    def test_weights_and_phases_add_and_decay(self):
        self.cells.observe_pair(
            self.x, self.y, weight=1.5, phase=0.25, relation_id=1)
        state = self.cells.observe_pair(
            self.x, self.y, weight=2.5, phase=0.5, relation_id=2)
        self.assertEqual(state.forward.weight, to_fixed(4.0))
        self.assertEqual(state.forward.amplitude, to_fixed(4.0))
        self.assertEqual(state.forward.phase, to_phase(0.75))
        self.assertEqual(state.forward.relation_chain, 2)

        state = self.cells.decay_pair(self.x, self.y, 0.5)
        self.assertEqual(state.forward.weight, to_fixed(2.0))
        self.assertEqual(state.forward.amplitude, to_fixed(2.0))
        self.assertEqual(state.forward.phase, to_phase(0.375))
        self.assertEqual(state.reverse.weight, 0)
        self.assertEqual(state.forward.generation, state.reverse.generation)
        self.assertEqual(state.forward.generation, 3)

        # phase wraps at one full turn; relation_id 0 preserves the pointer
        state = self.cells.observe_pair(
            self.x, self.y, weight=0.0, phase=0.25, relation_id=0)
        self.assertEqual(state.forward.weight, to_fixed(2.0))
        self.assertEqual(state.forward.phase, to_phase(0.625))
        self.assertEqual(state.forward.relation_chain, 2)
        state = self.cells.observe_pair(
            self.x, self.y, weight=0.0, phase=0.5, relation_id=0)
        self.assertEqual(state.forward.phase, to_phase(0.125))
        self.assertEqual(state.forward.generation, 5)

    def test_opposite_sign_weight_reduces_net_and_keeps_magnitude(self):
        self.cells.observe_pair(
            self.x, self.y, weight=2.0, phase=0.0, relation_id=0)
        state = self.cells.observe_pair(
            self.x, self.y, weight=-0.5, phase=0.125, relation_id=0)
        self.assertEqual(state.forward.weight, to_fixed(1.5))
        self.assertEqual(state.forward.amplitude, to_fixed(2.5))
        self.assertEqual(state.forward.phase, to_phase(0.125))

    def test_single_rail_decay_leaves_the_other_rail_state_alone(self):
        self.cells.observe_pair(self.x, self.y, weight=2.0, phase=0.5)
        self.cells.observe_pair(self.y, self.x, weight=3.0, phase=0.25)
        state = self.cells.decay_pair(
            self.x, self.y, 0.5, rail=RAIL_S_PLUS)
        self.assertEqual(state.forward.weight, to_fixed(1.0))
        self.assertEqual(state.forward.phase, to_phase(0.25))
        self.assertEqual(state.reverse.weight, to_fixed(3.0))
        self.assertEqual(state.reverse.phase, to_phase(0.25))
        self.assertEqual(state.forward.generation, state.reverse.generation)

    def test_decay_all_touches_every_cell(self):
        self.cells.observe_pair(self.x, self.y, weight=2.0, phase=0.5)
        self.assertEqual(self.cells.decay_all(0.5), PAIR_COUNT)
        state = self.cells.pair_state(self.x, self.y)
        self.assertEqual(state.forward.weight, to_fixed(1.0))
        self.assertEqual(state.forward.phase, to_phase(0.25))


class DeterministicReadbackTests(unittest.TestCase):
    def test_second_view_reads_exact_same_records(self):
        buffer = bytearray(REGION_BYTES)
        first = ResidentCells(buffer, initialize=True)
        x, y = forward_heads(9, 42)
        expected = first.observe_pair(
            x, y, weight=1.25, phase=0.125, relation_id=0x12345678)
        first.update_head(9, score_delta=3.25, state_delta=4,
                          flags=1, leaf_chain=0xCAFEBABE, relation_chain=77)

        raw_head = bytes(buffer[first.head_record_offset(9):
                                first.head_record_offset(9) + HEAD_RECORD_BYTES])
        raw_forward = bytes(buffer[first.rail_record_offset(x, y, RAIL_S_PLUS):
                                  first.rail_record_offset(x, y, RAIL_S_PLUS)
                                  + RAIL_RECORD_BYTES])

        second = ResidentCells(buffer)
        self.assertEqual(second.pair_state(x, y), expected)
        self.assertEqual(second.pair_state(y, x), expected)
        self.assertEqual(second.read_head(9), first.read_head(9))
        self.assertEqual(
            bytes(buffer[first.head_record_offset(9):
                         first.head_record_offset(9) + HEAD_RECORD_BYTES]),
            raw_head)
        self.assertEqual(
            bytes(buffer[first.rail_record_offset(x, y, RAIL_S_PLUS):
                         first.rail_record_offset(x, y, RAIL_S_PLUS)
                         + RAIL_RECORD_BYTES]),
            raw_forward)
        self.assertEqual(
            RAIL_STRUCT.unpack_from(
                buffer, first.rail_record_offset(x, y, RAIL_S_PLUS)),
            tuple(expected.forward))
        self.assertEqual(
            HEAD_STRUCT.unpack_from(buffer, first.head_record_offset(9)),
            tuple(first.read_head(9)))

        second.observe_pair(x, y, weight=0.5, phase=0.125, relation_id=5)
        self.assertEqual(first.pair_state(x, y), second.pair_state(x, y))

    def test_non_zero_base_offset_is_isolated(self):
        prefix = b"P" * 64
        suffix = b"S" * 48
        base = 64
        buffer = bytearray(len(prefix) + REGION_BYTES + len(suffix))
        buffer[:len(prefix)] = prefix
        buffer[-len(suffix):] = suffix
        cells = ResidentCells(buffer, base, initialize=True)
        self.assertEqual(cells.heads_offset, base)
        self.assertEqual(cells.pairs_offset, base + HEAD_REGION_BYTES)
        self.assertEqual(bytes(buffer[:len(prefix)]), prefix)
        self.assertEqual(bytes(buffer[-len(suffix):]), suffix)
        x, y = forward_heads(1, 254)
        state = cells.observe_pair(x, y, weight=0.75, phase=0.5, relation_id=3)
        self.assertEqual(bytes(buffer[:len(prefix)]), prefix)
        self.assertEqual(bytes(buffer[-len(suffix):]), suffix)
        reopened = ResidentCells(buffer, base)
        self.assertEqual(reopened.pair_state(x, y), state)

    def test_mmap_backed_region_round_trip(self):
        x, y = forward_heads(11, 22)
        with tempfile.TemporaryFile() as handle:
            handle.truncate(REGION_BYTES)
            mapped = mmap.mmap(handle.fileno(), REGION_BYTES)
            try:
                cells = ResidentCells(mapped, initialize=True)
                expected = cells.observe_pair(
                    x, y, weight=2.25, phase=0.75, relation_id=44)
                cells.update_head(11, score_delta=1.5, state_delta=2)
                mapped.flush()
            finally:
                mapped.close()

            reopened_map = mmap.mmap(handle.fileno(), REGION_BYTES)
            try:
                reopened = ResidentCells(reopened_map)
                self.assertEqual(reopened.pair_state(x, y), expected)
                self.assertEqual(reopened.read_head(11).score, to_fixed(1.5))
            finally:
                reopened_map.close()


class BoundsRejectionTests(unittest.TestCase):
    def setUp(self):
        self.buffer = bytearray(REGION_BYTES)
        self.cells = ResidentCells(self.buffer, initialize=True)

    def test_buffer_and_base_offset_bounds(self):
        with self.assertRaises(ValueError):
            ResidentCells(bytearray(REGION_BYTES - 1))
        with self.assertRaises(ValueError):
            ResidentCells(self.buffer, -1)
        with self.assertRaises(ValueError):
            ResidentCells(self.buffer, True)
        with self.assertRaises(ValueError):
            ResidentCells(self.buffer, len(self.buffer) - REGION_BYTES + 1)

    def test_head_bounds(self):
        for bad in (-1, HEAD_COUNT, 256, 10_000, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                self.cells.read_head(bad)
            with self.assertRaises(ValueError):
                self.cells.head_record_offset(bad)

    def test_pair_and_rail_bounds(self):
        for a, b in ((0, 0), (255, 255), (-1, 0), (0, -1), (256, 1), (1, 256),
                     (True, 1), (1, 2.0)):
            with self.assertRaises(ValueError):
                self.cells.pair_state(a, b)
            with self.assertRaises(ValueError):
                pair_index(a, b)
        for rail in (-1, 2, 3, True, 0.0):
            with self.assertRaises(ValueError):
                self.cells.read_rail(1, 2, rail)
            with self.assertRaises(ValueError):
                self.cells.rail_record_offset(1, 2, rail)

    def test_value_bounds(self):
        with self.assertRaises(ValueError):
            self.cells.observe_pair(1, 2, weight=10 ** 9)
        with self.assertRaises(ValueError):
            self.cells.observe_pair(1, 2, phase=float("nan"))
        with self.assertRaises(ValueError):
            self.cells.observe_pair(1, 2, relation_id=2 ** 32)
        with self.assertRaises(ValueError):
            self.cells.observe_pair(1, 2, relation_id=-1)
        with self.assertRaises(ValueError):
            self.cells.decay_pair(1, 2, 1.5)
        with self.assertRaises(ValueError):
            self.cells.decay_pair(1, 2, float("inf"))
        with self.assertRaises(ValueError):
            self.cells.update_head(0, leaf_chain=-1)
        with self.assertRaises(ValueError):
            self.cells.update_head(0, state_delta=-1)
        with self.assertRaises(ValueError):
            cells_module.to_fixed(True)
        with self.assertRaises(ValueError):
            cells_module.to_phase(float("inf"))


if __name__ == "__main__":
    unittest.main()
