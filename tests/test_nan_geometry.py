"""Tests for raw NaN payload geometry and K256 mirror addressing.

All checks use integer payload units.  No test asks the host to interpret a
NaN payload as a Python float; the module-level AST check enforces the same for
the implementation itself.
"""
import ast
import random
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from nanlmind_hermes import nan_geometry as ng


UINT64_MASK = (1 << 64) - 1


class NanFieldTests(unittest.TestCase):
    def test_canonical_qnan_and_snan_layout(self):
        qnan = ng.pack_nan(0, ng.QUIET_BIT)
        self.assertEqual(qnan, 0x7FF8000000000000)
        fields = ng.unpack_bits(qnan)
        self.assertEqual((fields.sign, fields.exponent, fields.payload),
                         (0, ng.EXPONENT_ALL_ONES, ng.QUIET_BIT))
        self.assertTrue(fields.is_nan)
        self.assertTrue(fields.quiet)
        self.assertTrue(ng.is_quiet_nan(qnan))
        self.assertFalse(ng.is_signaling_nan(qnan))

        snan = ng.pack_nan(1, 1)
        fields = ng.validate_nan(snan)
        self.assertEqual(fields.sign, 1)
        self.assertEqual(fields.payload, 1)
        self.assertFalse(fields.quiet)
        self.assertTrue(ng.is_signaling_nan(snan))
        self.assertEqual(ng.extract_sign(snan), 1)
        self.assertEqual(ng.extract_exponent(snan), ng.EXPONENT_ALL_ONES)
        self.assertEqual(ng.extract_payload(snan), 1)
        self.assertEqual(ng.extract_fraction(snan), 1)
        self.assertEqual(ng.extract_quiet_bit(qnan), 1)
        self.assertEqual(ng.extract_quiet_bit(snan), 0)
        self.assertEqual(ng.canonical_quiet_nan(1), 0xFFF8000000000000)

    def test_all_16_bit_payload_patterns(self):
        # The low 16 payload bits are exhaustively covered for both quiet and
        # signaling NaNs.  This includes every 16-bit boundary value.
        for value in range(1 << 16):
            quiet_payload = ng.QUIET_BIT | value
            quiet = ng.pack_nan(0, quiet_payload, quiet=True)
            fields = ng.unpack_bits(quiet)
            self.assertEqual(fields.payload, quiet_payload)
            self.assertTrue(fields.quiet)
            self.assertEqual(fields.sign, 0)
            self.assertEqual(fields.exponent, ng.EXPONENT_ALL_ONES)

            signaling_payload = value if value else 1
            signaling = ng.pack_nan(1, signaling_payload, quiet=False)
            fields = ng.unpack_bits(signaling)
            self.assertEqual(fields.payload, signaling_payload)
            self.assertFalse(fields.quiet)
            self.assertEqual(fields.sign, 1)
            self.assertTrue(fields.is_nan)

    def test_payload_boundaries(self):
        payloads = [1, 2, ng.QUIET_BIT - 1, ng.QUIET_BIT,
                    ng.QUIET_BIT + 1, ng.PAYLOAD_MASK - 1, ng.PAYLOAD_MASK]
        for sign in (0, 1):
            for payload in payloads:
                bits = ng.pack_nan(sign, payload)
                fields = ng.unpack_bits(bits)
                self.assertEqual(fields.sign, sign)
                self.assertEqual(fields.exponent, ng.EXPONENT_ALL_ONES)
                self.assertEqual(fields.payload, payload)
                self.assertEqual(fields.quiet, bool(payload & ng.QUIET_BIT))
                self.assertEqual(ng.canonicalize_nan(bits),
                                 ng.canonicalize_nan(ng.canonicalize_nan(bits)))

        self.assertEqual(ng.canonicalize_nan(ng.pack_nan(1, ng.QUIET_BIT),
                                             quiet=False),
                         ng.pack_nan(1, 1))
        with self.assertRaises(ng.NonNanError):
            ng.pack_nan(0, 0)
        with self.assertRaises(ng.NaNFieldError):
            ng.pack_nan(0, ng.PAYLOAD_MASK + 1)
        with self.assertRaises(ng.NaNFieldError):
            ng.pack_nan(0, ng.QUIET_BIT, quiet=False)
        with self.assertRaises(ng.NaNFieldError):
            ng.pack_nan(0, 1, quiet=True)
        with self.assertRaises(TypeError):
            ng.pack_nan(True, 1)
        with self.assertRaises(ng.NonNanError):
            ng.canonicalize_nan(0)

    def test_reject_non_nan_and_finite_control(self):
        finite_patterns = [0, 1, (1 << 63), 0x3FF0000000000000,
                           0x0010000000000000]
        for bits in finite_patterns:
            with self.assertRaises(ng.NonNanError):
                ng.validate_nan(bits)
            self.assertFalse(ng.is_nan(bits))
            self.assertTrue(ng.is_finite(bits))
            self.assertFalse(ng.is_infinity(bits))
            self.assertEqual(ng.validate_finite_control(bits).pack(), bits)

        for bits in (0x7FF0000000000000, 0xFFF0000000000000):
            with self.assertRaises(ng.NonNanError):
                ng.validate_nan(bits)
            with self.assertRaises(ng.NonNanError):
                ng.validate_finite_control(bits)
            self.assertTrue(ng.is_infinity(bits))
            self.assertFalse(ng.is_finite(bits))

        for bits in (0x7FF8000000000000, 0x7FF0000000000001):
            with self.assertRaises(ng.NonNanError):
                ng.validate_finite_control(bits)

    def test_canonicalization_is_idempotent_and_preserves_payload(self):
        signaling = ng.pack_nan(0, 0x1234)
        quiet = ng.canonicalize_nan(signaling, quiet=True)
        self.assertEqual(ng.extract_payload(quiet),
                         ng.QUIET_BIT | 0x1234)
        self.assertEqual(ng.canonicalize_nan(quiet, quiet=True), quiet)

        quiet = ng.pack_nan(1, ng.QUIET_BIT | 0x5678)
        signaling = ng.canonicalize_nan(quiet, quiet=False)
        self.assertEqual(ng.extract_payload(signaling), 0x5678)
        self.assertEqual(ng.canonicalize_nan(signaling, quiet=False),
                         signaling)
        self.assertEqual(ng.extract_sign(signaling), 1)

        smallest = ng.pack_nan(0, ng.QUIET_BIT)
        self.assertEqual(ng.extract_payload(ng.canonicalize_nan(
            smallest, quiet=False)), 1)
        self.assertTrue(ng.validate_nan(
            ng.canonicalize_nan(smallest, quiet=False)))

    def test_all_reserved_tags_and_quiet_policy(self):
        values = [0, 1, ng.VALUE_SIGN_BIT - 1, ng.VALUE_SIGN_BIT,
                  ng.VALUE_MASK]
        for tag in range(1 << ng.TAG_BITS):
            for value in values:
                if tag == 0 and value == 0:
                    with self.assertRaises(ng.NonNanError):
                        ng.encode_reserved(tag, value)
                    continue
                for sign in (0, 1):
                    bits = ng.encode_reserved(tag, value, sign=sign)
                    decoded = ng.decode_reserved(bits)
                    self.assertEqual(decoded.sign, sign)
                    self.assertEqual(decoded.tag, tag)
                    self.assertEqual(decoded.value, value)
                    self.assertEqual(decoded.quiet, bool(tag & 0x8))
                    self.assertEqual(ng.is_quiet_nan(bits), bool(tag & 0x8))
                    self.assertEqual(ng.is_signaling_nan(bits),
                                     not bool(tag & 0x8))
                    if tag in ng.QUIET_TAGS:
                        self.assertTrue(ng.is_quiet_nan(bits))
                    else:
                        self.assertTrue(ng.is_signaling_nan(bits))
                    expected_units = (value - (1 << ng.VALUE_BITS)
                                      if value & ng.VALUE_SIGN_BIT else value)
                    self.assertEqual(decoded.units, expected_units)
                    self.assertEqual(decoded.pack(), bits)

        self.assertEqual(ng.split_reserved_payload(
            (ng.TAG_FIXED << ng.TAG_SHIFT) | 0x1234),
            (ng.TAG_FIXED, 0x1234))
        self.assertEqual(ng.join_reserved_payload(ng.TAG_FIXED, 0x1234),
                         (ng.TAG_FIXED << ng.TAG_SHIFT) | 0x1234)
        with self.assertRaises(ng.NaNFieldError):
            ng.encode_reserved(ng.TAG_FIXED, 1, quiet=False)
        with self.assertRaises(ng.NaNFieldError):
            ng.decode_reserved(ng.encode_reserved(ng.TAG_FIXED, 1),
                               expected_tag=0x9)

    def test_fixed_point_boundaries(self):
        units_values = [ng.VALUE_MIN, ng.VALUE_MIN + 1, -65536, -1, 0, 1,
                        65535, 65536, ng.VALUE_MAX - 1, ng.VALUE_MAX]
        for units in units_values:
            bits = ng.encode_fixed_units(units)
            self.assertEqual(ng.decode_fixed_units(bits), units)
            reserved = ng.decode_reserved(bits)
            self.assertEqual(reserved.sign, 0)
            self.assertEqual(reserved.tag, ng.TAG_FIXED)
            self.assertTrue(reserved.quiet)
            self.assertEqual(reserved.value, units & ng.VALUE_MASK)
            self.assertTrue(ng.is_quiet_nan(bits))

        for units in (ng.VALUE_MIN - 1, ng.VALUE_MAX + 1):
            with self.assertRaises(ng.FixedPointOverflow):
                ng.encode_fixed_units(units)
        with self.assertRaises(TypeError):
            ng.encode_fixed_units(True)
        with self.assertRaises(ng.NaNFieldError):
            ng.encode_fixed_units(1, tag=0x1)
        with self.assertRaises(ng.NaNFieldError):
            ng.encode_fixed_units(1, sign=1)
        with self.assertRaises(ng.NaNFieldError):
            ng.decode_fixed_units(ng.encode_reserved(ng.TAG_FIXED, 1,
                                                     sign=1))
        with self.assertRaises(ng.NaNFieldError):
            ng.decode_fixed_units(ng.encode_reserved(0x9, 1))
        with self.assertRaises(ng.NonNanError):
            ng.decode_fixed_units(0)

    def test_all_16_bit_signed_values_roundtrip(self):
        for units in range(-(1 << 15), 1 << 15):
            bits = ng.encode_fixed_units(units)
            self.assertEqual(ng.decode_fixed_units(bits), units)
            self.assertEqual(ng.decode_reserved(bits).value,
                             units & ng.VALUE_MASK)

    def test_fixed_fractions_are_exact(self):
        cases = [
            (Fraction(0), 0),
            (Fraction(1, 2), 4),
            (Fraction(-3, 4), 4),
            (Fraction(12345, 16), 4),
            (Fraction(-7, 8), 3),
            (Fraction(123456789), 12),
        ]
        for value, scale_bits in cases:
            bits = ng.encode_fixed_fraction(value, scale_bits=scale_bits)
            self.assertEqual(ng.decode_fixed_fraction(
                bits, scale_bits=scale_bits), value)
            self.assertEqual(ng.decode_fixed_units(bits),
                             int(value * (1 << scale_bits)))
        with self.assertRaises(ng.NaNFieldError):
            ng.encode_fixed_fraction(Fraction(1, 3), scale_bits=2)
        with self.assertRaises(TypeError):
            ng.encode_fixed_fraction(1.5)
        with self.assertRaises(ng.NaNFieldError):
            ng.decode_fixed_fraction(ng.encode_fixed_units(1),
                                     scale_bits=256)

    def test_fixed_arithmetic_is_exact_and_saturating(self):
        cases = [(0, 0), (1, -1), (123456, 654321),
                 (-1, -1), (ng.VALUE_MAX, 0), (ng.VALUE_MIN, 0)]
        for first, second in cases:
            bits_first = ng.encode_fixed_units(first)
            bits_second = ng.encode_fixed_units(second)
            self.assertEqual(ng.decode_fixed_units(
                ng.fixed_add(bits_first, bits_second)), first + second)
            self.assertEqual(ng.decode_fixed_units(
                ng.fixed_sub(bits_first, bits_second)), first - second)
            if first == ng.VALUE_MIN:
                with self.assertRaises(ng.FixedPointOverflow):
                    ng.fixed_negate(bits_first)
                self.assertEqual(ng.decode_fixed_units(ng.fixed_negate(
                    bits_first, overflow=ng.OVERFLOW_SATURATE)),
                    ng.VALUE_MAX)
            else:
                self.assertEqual(ng.decode_fixed_units(
                    ng.fixed_negate(bits_first)), -first)

        one = ng.encode_fixed_units(1)
        maximum = ng.encode_fixed_units(ng.VALUE_MAX)
        minimum = ng.encode_fixed_units(ng.VALUE_MIN)
        with self.assertRaises(ng.FixedPointOverflow):
            ng.fixed_add(maximum, one)
        with self.assertRaises(ng.FixedPointOverflow):
            ng.fixed_sub(minimum, one)
        with self.assertRaises(ng.FixedPointOverflow):
            ng.fixed_negate(minimum)
        self.assertEqual(ng.decode_fixed_units(
            ng.fixed_add(maximum, one, overflow=ng.OVERFLOW_SATURATE)),
            ng.VALUE_MAX)
        self.assertEqual(ng.decode_fixed_units(
            ng.fixed_sub(minimum, one, overflow=ng.OVERFLOW_SATURATE)),
            ng.VALUE_MIN)
        self.assertEqual(ng.decode_fixed_units(
            ng.fixed_negate(minimum, overflow=ng.OVERFLOW_SATURATE)),
            ng.VALUE_MAX)
        self.assertEqual(ng.decode_fixed_units(ng.fixed_add_many(
            [ng.encode_fixed_units(7), ng.encode_fixed_units(-9),
             ng.encode_fixed_units(5)])), 3)
        self.assertEqual(ng.decode_fixed_units(ng.fixed_add_many(
            [maximum, one, one], overflow=ng.OVERFLOW_SATURATE)),
            ng.VALUE_MAX)
        with self.assertRaises(ValueError):
            ng.fixed_add(one, one, overflow="wrap")

    def test_fixed_arithmetic_random_pairs(self):
        rng = random.Random(2029)
        for _ in range(1000):
            first = rng.randint(-(1 << 20), 1 << 20)
            second = rng.randint(-(1 << 20), 1 << 20)
            bits_first = ng.encode_fixed_units(first)
            bits_second = ng.encode_fixed_units(second)
            self.assertEqual(ng.decode_fixed_units(
                ng.fixed_add(bits_first, bits_second)), first + second)
            self.assertEqual(ng.decode_fixed_units(
                ng.fixed_sub(bits_first, bits_second)), first - second)
            self.assertEqual(ng.decode_fixed_units(
                ng.fixed_negate(bits_first)), -first)

    def test_fixed_arithmetic_rejects_nonfixed_inputs(self):
        good = ng.encode_fixed_units(1)
        with self.assertRaises(ng.NonNanError):
            ng.fixed_add(good, 0)
        with self.assertRaises(ng.NaNFieldError):
            ng.fixed_add(good, ng.encode_reserved(0x9, 1))
        with self.assertRaises(ng.NonNanError):
            ng.fixed_negate(0x7FF0000000000000)

    def test_module_contains_no_float_operations(self):
        source = Path(ng.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        float_literals = [node for node in ast.walk(tree)
                          if isinstance(node, ast.Constant)
                          and type(node.value) is float]
        float_names = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Name) and node.id == "float"]
        self.assertEqual(float_literals, [])
        self.assertEqual(float_names, [])


class IdentityTests(unittest.TestCase):
    def test_layout_constants(self):
        self.assertEqual(ng.N_HEADS, 256)
        self.assertEqual(ng.HEAD_BYTES, 128)
        self.assertEqual(ng.LANE_BYTES, 128)
        self.assertEqual(ng.PAIR_BYTES, 256)
        self.assertEqual(ng.PAIR_COUNT, 32640)
        self.assertEqual(ng.ROOT_BYTES, 8388608)
        report = ng.layout_report()
        self.assertEqual(report["heads"], 256)
        self.assertEqual(report["pairs"], 32640)
        self.assertEqual(report["directed_mappings"], 65280)
        self.assertEqual(report["root_bytes"], 8388608)
        self.assertEqual(ng.head_offset(255), 32640)
        self.assertEqual(ng.pair_cell_offset(254, 255),
                         32768 + 32639 * 256)

    def test_exhaustive_directed_mappings(self):
        seen = bytearray(ng.PAIR_COUNT)
        directed = 0
        for first, second in ng.iter_directed():
            directed += 1
            pair = ng.pair_id(first, second)
            self.assertEqual(pair, ng.pair_id(second, first))
            self.assertGreaterEqual(pair, 0)
            self.assertLess(pair, ng.PAIR_COUNT)
            canonical = ng.pair_heads(pair)
            self.assertEqual(canonical, (min(first, second),
                                         max(first, second)))
            self.assertEqual(ng.direction_lane(first, second),
                             0 if first < second else 1)
            forward = ng.mirror_address(first, second)
            reverse = ng.mirror_address(second, first)
            self.assertEqual(forward.pair, pair)
            self.assertEqual(reverse.pair, pair)
            self.assertEqual(forward.cell_offset, reverse.cell_offset)
            self.assertEqual(forward.cell_offset,
                             ng.pair_cell_offset(first, second))
            self.assertEqual(forward.cell_offset, reverse.cell_offset)
            self.assertEqual(forward.lane, 0 if first < second else 1)
            self.assertNotEqual(forward.lane, reverse.lane)
            self.assertEqual(forward.lane_offset,
                             forward.cell_offset
                             + forward.lane * ng.LANE_BYTES)
            self.assertEqual(reverse.lane_offset,
                             reverse.cell_offset
                             + reverse.lane * ng.LANE_BYTES)
            self.assertEqual(abs(forward.lane_offset - reverse.lane_offset),
                             ng.LANE_BYTES)
            self.assertTrue(0 <= forward.lane_offset < ng.ROOT_BYTES)
            self.assertTrue(0 <= reverse.lane_offset < ng.ROOT_BYTES)
            self.assertEqual(forward.heads, (first, second))
            self.assertEqual(reverse.heads, (second, first))
            seen[pair] = 1
        self.assertEqual(directed, 65280)
        self.assertEqual(sum(seen), ng.PAIR_COUNT)

    def test_pair_enumeration_is_complete_and_ordered(self):
        pairs = list(ng.iter_pairs())
        self.assertEqual(len(pairs), ng.PAIR_COUNT)
        self.assertEqual(len(set(pairs)), ng.PAIR_COUNT)
        for pair, (first, second) in enumerate(pairs):
            self.assertLess(first, second)
            self.assertEqual(ng.pair_id(first, second), pair)
            self.assertEqual(ng.pair_heads(pair), (first, second))

    def test_invalid_identity_inputs(self):
        bad_heads = [-1, 256, 10_000]
        for head in bad_heads:
            with self.assertRaises(ng.HeadIndexError):
                ng.pair_id(head, 0)
            with self.assertRaises(ng.HeadIndexError):
                ng.head_offset(head)
        with self.assertRaises(ng.HeadIndexError):
            ng.pair_id(7, 7)
        with self.assertRaises(ng.HeadIndexError):
            ng.direction_lane(7, 7)
        with self.assertRaises(ng.HeadIndexError):
            ng.pair_heads(-1)
        with self.assertRaises(ng.HeadIndexError):
            ng.pair_heads(ng.PAIR_COUNT)
        with self.assertRaises(TypeError):
            ng.pair_id(True, 0)
        with self.assertRaises(TypeError):
            ng.pair_id(0, False)

    def test_identity_runtime_does_not_touch_float(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("identity path used Python float")

        with mock.patch("builtins.float", side_effect=forbidden):
            for first, second in ((0, 1), (1, 0), (17, 203),
                                  (255, 0), (128, 127)):
                pair = ng.pair_id(first, second)
                self.assertEqual(ng.pair_heads(pair),
                                 (min(first, second), max(first, second)))
                address = ng.mirror_address(first, second)
                self.assertIsInstance(address.lane, int)
                self.assertIsInstance(address.lane_offset, int)
                self.assertIsInstance(ng.head_offset(first), int)
                self.assertIsInstance(ng.pair_cell_offset(first, second), int)
            self.assertEqual(ng.layout_report()["root_bytes"], 8388608)


class LaneTests(unittest.TestCase):
    def test_all_16_bit_field_values(self):
        for value in range(1 << 16):
            words = tuple(((value << 48) ^
                           (index * 0x9E3779B97F4A7C15)) & UINT64_MASK
                          for index in range(ng.LANE_WORD_COUNT))
            lane = ng.Lane(
                tag=value,
                flags=(1 << 16) - 1 - value,
                generation=(value << 48) | (value << 16) | value,
                reserved=(value << 16) | value,
                words=words,
            )
            raw = lane.pack()
            self.assertEqual(len(raw), 128)
            restored = ng.Lane.unpack(raw)
            self.assertEqual(restored, lane)
            self.assertEqual(restored.pack(), raw)
            self.assertEqual(restored.tag, value)
            self.assertEqual(restored.flags, (1 << 16) - 1 - value)
            self.assertEqual(restored.words, words)

    def test_little_endian_field_offsets_and_bit_preservation(self):
        words = tuple((index + 1) * 0x0101010101010101
                      for index in range(ng.LANE_WORD_COUNT))
        lane = ng.Lane(tag=0x4E41, flags=0xABCD, generation=0x1122334455667788,
                       reserved=0xDEADBEEF, words=words)
        raw = ng.pack_lane(lane)
        self.assertEqual(raw[ng.LANE_TAG_OFFSET:ng.LANE_FLAGS_OFFSET],
                         (0x4E41).to_bytes(2, "little"))
        self.assertEqual(raw[ng.LANE_FLAGS_OFFSET:ng.LANE_GENERATION_OFFSET],
                         (0xABCD).to_bytes(2, "little"))
        self.assertEqual(raw[ng.LANE_GENERATION_OFFSET:
                             ng.LANE_RESERVED_OFFSET],
                         (0x1122334455667788).to_bytes(8, "little"))
        self.assertEqual(raw[ng.LANE_RESERVED_OFFSET:ng.LANE_WORDS_OFFSET],
                         (0xDEADBEEF).to_bytes(4, "little"))
        for index, word in enumerate(words):
            start = ng.LANE_WORDS_OFFSET + index * 8
            self.assertEqual(raw[start:start + 8], word.to_bytes(8, "little"))
        self.assertEqual(lane.word(0), words[0])
        changed = lane.with_word(3, 7)
        self.assertEqual(changed.word(3), 7)
        self.assertEqual(changed.words[0], words[0])
        self.assertEqual(lane.word(3), words[3])
        self.assertEqual(ng.unpack_lane(raw), lane)

    def test_arbitrary_128_byte_lanes_are_bit_preserving(self):
        rng = random.Random(2029)
        for _ in range(512):
            raw = rng.randbytes(128)
            lane = ng.unpack_lane(raw)
            self.assertEqual(lane.pack(), raw)
            self.assertEqual(lane.tag,
                             int.from_bytes(raw[0:2], "little"))
            self.assertEqual(lane.flags,
                             int.from_bytes(raw[2:4], "little"))
            self.assertEqual(lane.generation,
                             int.from_bytes(raw[4:12], "little"))
            self.assertEqual(lane.reserved,
                             int.from_bytes(raw[12:16], "little"))
            for index in range(ng.LANE_WORD_COUNT):
                start = 16 + 8 * index
                self.assertEqual(lane.words[index],
                                 int.from_bytes(raw[start:start + 8], "little"))

    def test_lane_nan_payload_words(self):
        values = [ng.VALUE_MIN, -1, 0, 1, ng.VALUE_MAX]
        values += [index * 12345 for index in range(9)]
        words = tuple(ng.encode_fixed_units(value) for value in values)
        lane = ng.Lane(tag=ng.LANE_TAG_NAN_WORDS, generation=7,
                       words=words, flags=0, reserved=0)
        restored = ng.Lane.unpack(lane.pack())
        self.assertEqual(restored.tag, ng.LANE_TAG_NAN_WORDS)
        for index, value in enumerate(values):
            self.assertEqual(ng.decode_fixed_units(restored.word(index)),
                             value)
            self.assertTrue(ng.is_quiet_nan(restored.word(index)))

    def test_lane_validation(self):
        words = (0,) * ng.LANE_WORD_COUNT
        with self.assertRaises(ng.LaneError):
            ng.Lane(0, 0, words[:-1])
        with self.assertRaises(ng.LaneError):
            ng.Lane(0, 0, words + (0,))
        with self.assertRaises(ng.NaNFieldError):
            ng.Lane(1 << 16, 0, words)
        with self.assertRaises(ng.NaNFieldError):
            ng.Lane(0, 1 << 64, words)
        with self.assertRaises(ng.NaNFieldError):
            ng.Lane(0, 0, words, flags=1 << 16)
        with self.assertRaises(ng.NaNFieldError):
            ng.Lane(0, 0, words, reserved=1 << 32)
        with self.assertRaises(ng.NaNFieldError):
            ng.Lane(0, 0, (1 << 64,) + words[1:])
        with self.assertRaises(TypeError):
            ng.unpack_lane("x" * 128)
        with self.assertRaises(ng.LaneError):
            ng.unpack_lane(b"x" * 127)
        with self.assertRaises(ng.LaneError):
            ng.unpack_lane(b"x" * 129)
        with self.assertRaises(ng.LaneError):
            ng.Lane(0, 0, words).word(-1)
        with self.assertRaises(ng.LaneError):
            ng.Lane(0, 0, words).with_word(ng.LANE_WORD_COUNT, 0)


class GaugeTests(unittest.TestCase):
    def test_exact_gauge_cancellation_all_triangles(self):
        heads = 256
        store = ng.GaugeStore(heads)
        rng = random.Random(29)
        gauge = [rng.randint(-(1 << 40), 1 << 40) for _ in range(heads)]
        store.set_gauge(gauge)
        z_values = {}
        for first in range(1, heads):
            for second in range(first + 1, heads):
                value = rng.randint(-(1 << 20), 1 << 20)
                z_values[(first, second)] = value
                store.set_cycle(first, second, value)

        for (first, second), value in z_values.items():
            self.assertEqual(store.cycle_readout(first, second), value)
            self.assertEqual(store.circulation(first, second), value)
            self.assertEqual(store.canonical_flux(first, second), value)
            self.assertEqual(store.canonical_flux(second, first), -value)
            self.assertEqual(store.edge(first, second),
                             value + gauge[first] - gauge[second])
            self.assertEqual(store.edge(second, first),
                             -store.edge(first, second))
            self.assertEqual(store.cycle_readout(second, first), -value)

        shift = [rng.randint(-(1 << 50), 1 << 50) for _ in range(heads)]
        store.shift_gauge(shift)
        moved = store.gauge
        for (first, second), value in z_values.items():
            self.assertEqual(store.cycle_readout(first, second), value)
            self.assertEqual(store.canonical_flux(first, second), value)
            self.assertEqual(store.edge(first, second),
                             value + moved[first] - moved[second])

    def test_gauge_shift_changes_edges_but_not_cycles(self):
        store = ng.GaugeStore(16)
        store.set_gauge([index * 1000 for index in range(16)])
        store.set_cycle(3, 9, 12345)
        before = [store.edge(0, 3), store.edge(3, 9), store.edge(9, 0),
                  store.circulation(3, 9)]
        store.shift_gauge([-500, 700] + [13] * 14)
        after = [store.edge(0, 3), store.edge(3, 9), store.edge(9, 0),
                 store.circulation(3, 9)]
        self.assertNotEqual(before[:3], after[:3])
        self.assertEqual(before[3], 12345)
        self.assertEqual(after[3], 12345)
        self.assertEqual(store.canonical_flux(3, 9), 12345)
        self.assertEqual(store.canonical_flux(9, 3), -12345)

    def test_arbitrary_huge_gauge_cancels_exactly(self):
        store = ng.GaugeStore(32)
        huge = (1 << 100) - 12345
        gauge = [((index + 1) * huge) % (1 << 101) - (1 << 100)
                 for index in range(32)]
        store.set_gauge(gauge)
        store.set_cycle(5, 29, -(1 << 60) + 7)
        self.assertEqual(store.circulation(5, 29), -(1 << 60) + 7)
        self.assertEqual(store.canonical_flux(5, 29), -(1 << 60) + 7)
        self.assertEqual(store.edge(5, 29),
                         -(1 << 60) + 7 + gauge[5] - gauge[29])

    def test_payload_bounds_and_integer_cycle_under_gauge_noise(self):
        heads = 256
        store = ng.GaugeStore(heads)
        rng = random.Random(41)
        gauge = [rng.randint(-10000, 10000) for _ in range(heads)]
        store.set_gauge(gauge)
        z_values = {}
        for first in range(1, heads):
            for second in range(first + 1, heads):
                value = rng.randint(-10000, 10000)
                z_values[(first, second)] = value
                store.set_cycle(first, second, value)

        self.assertEqual(store.validate_payload_bounds(), 65280)

        for _ in range(200):
            first, second = rng.sample(range(1, heads), 2)
            if first > second:
                first, second = second, first
            value = z_values[(first, second)]
            bits_0i = ng.encode_fixed_units(store.edge(0, first))
            bits_ij = ng.encode_fixed_units(store.edge(first, second))
            bits_j0 = ng.encode_fixed_units(store.edge(second, 0))
            self.assertEqual(ng.decode_fixed_units(bits_0i)
                             + ng.decode_fixed_units(bits_ij)
                             + ng.decode_fixed_units(bits_j0), value)
            self.assertTrue(ng.is_quiet_nan(bits_0i))
            self.assertTrue(ng.is_quiet_nan(bits_ij))
            self.assertTrue(ng.is_quiet_nan(bits_j0))

        # A gauge can be too large for one fixed-width edge even when the
        # gauge-invariant cycle still cancels exactly.
        shift = [0] * heads
        shift[0] = ng.VALUE_MAX
        shift[1] = -ng.VALUE_MAX
        store.shift_gauge(shift)
        with self.assertRaises(ng.FixedPointOverflow):
            store.validate_payload_bounds()
        first, second = rng.sample(range(2, heads), 2)
        if first > second:
            first, second = second, first
        self.assertEqual(store.cycle_readout(first, second),
                         z_values[(first, second)])

    def test_gauge_store_validation_and_reset(self):
        store = ng.GaugeStore(8)
        with self.assertRaises(ng.HeadIndexError):
            ng.GaugeStore(2)
        with self.assertRaises(ng.HeadIndexError):
            store.set_gauge([0] * 7)
        with self.assertRaises(ng.HeadIndexError):
            store.shift_gauge([0] * 9)
        with self.assertRaises(ng.HeadIndexError):
            store.set_cycle(8, 1, 1)
        with self.assertRaises(ng.HeadIndexError):
            store.set_cycle(1, 1, 1)
        with self.assertRaises(TypeError):
            store.set_cycle(1, 2, True)
        with self.assertRaises(ng.HeadIndexError):
            store.circulation(0, 1, 1)
        self.assertEqual(store.gauge, (0,) * 8)
        store.set_gauge(range(8))
        store.shift_gauge([1] * 8)
        self.assertEqual(store.gauge, tuple(index + 1 for index in range(8)))
        store.reset_gauge()
        self.assertEqual(store.gauge, (0,) * 8)
        store.store_skew(2, 5, 9)
        self.assertEqual(store.cycle_value(2, 5), 9)
        self.assertEqual(store.cycle_value(5, 2), -9)
        store.store_skew(5, 2, 3)
        self.assertEqual(store.cycle_value(2, 5), -3)


if __name__ == "__main__":
    unittest.main()
