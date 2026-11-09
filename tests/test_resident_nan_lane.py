"""Tests for the raw NaN-payload lane ABI and compact native pair layout.

The suite is intentionally word-level: no test constructs a Python float to
interpret a lane.  It exercises exhaustive byte mutations, random 32-bit words,
quiet/signalling payload classes, identity immutability, mirror orientation,
exact modulo-``2**32`` cycle/gauge arithmetic, and the 256-byte pair layout.
"""
from __future__ import annotations

import ast
import inspect
import random
import struct
import unittest

from nanlmind_hermes.resident import nan_lane as lane_module
from nanlmind_hermes.resident.nan_lane import (
    CHANNEL_COUNT,
    FIRST_CHANNEL_WORD,
    FIRST_RESERVED_WORD,
    GENERATION_WORD,
    NATIVE_LANE_BYTES,
    NATIVE_PAIR_BYTES,
    NAN_EXPONENT_MASK,
    NAN_MANTISSA_MASK,
    NAN_QUIET_BIT,
    NAN_SIGN_BIT,
    ORIENT_FORWARD,
    ORIENT_REVERSE,
    PAIR_BYTES,
    PAIR_LANES,
    PAIR_STRUCT,
    PAIR_WORDS,
    QUIET_NAN_WORD,
    RESERVED_COUNT,
    SIGNALING_NAN_WORD,
    TAG_QUIET,
    TAG_RAW,
    TAG_SIGNALING,
    TAG_VERSION_WORD,
    LANE_BYTES,
    LANE_STRUCT,
    LANE_WORDS,
    LANE_VERSION,
    WORD_BYTES,
    WORD_MASK,
    WORD_STRUCT,
    IdentityRewriteError,
    Lane,
    LaneBoundsError,
    LaneError,
    LaneTagError,
    Pair,
    bump_generation,
    cycle_word,
    deposit_oriented,
    get_channel,
    get_generation,
    get_reserved,
    is_nan,
    is_quiet_nan,
    is_signaling_nan,
    mirror_lane,
    nan_class,
    pack_lane,
    pack_pair,
    read_word,
    set_channel,
    set_generation,
    set_reserved,
    split_tag_version,
    tag_version_word,
    unpack_lane,
    unpack_pair,
    write_word,
)


def raw_lane(channels=None, generation=0, reserved=(0, 0)) -> Lane:
    return Lane.new(
        TAG_RAW, channels=channels, generation=generation, reserved=reserved)


def quiet_word(rng: random.Random) -> int:
    sign = rng.getrandbits(1) << 31
    mantissa = NAN_QUIET_BIT | rng.getrandbits(22)
    return sign | NAN_EXPONENT_MASK | mantissa


def signaling_word(rng: random.Random) -> int:
    sign = rng.getrandbits(1) << 31
    mantissa = rng.getrandbits(22) & ~NAN_QUIET_BIT
    if mantissa == 0:
        mantissa = 1
    return sign | NAN_EXPONENT_MASK | mantissa


class LayoutTests(unittest.TestCase):
    def test_fixed_sizes_and_word_regions(self):
        self.assertEqual(LANE_WORDS, 32)
        self.assertEqual(WORD_BYTES, 4)
        self.assertEqual(LANE_BYTES, 128)
        self.assertEqual(NATIVE_LANE_BYTES, 128)
        self.assertEqual(PAIR_LANES, 2)
        self.assertEqual(PAIR_WORDS, 64)
        self.assertEqual(PAIR_BYTES, 256)
        self.assertEqual(NATIVE_PAIR_BYTES, 256)
        self.assertEqual(WORD_STRUCT.format, "<I")
        self.assertEqual(LANE_STRUCT.format, "<32I")
        self.assertEqual(PAIR_STRUCT.format, "<64I")
        self.assertEqual(WORD_STRUCT.size, 4)
        self.assertEqual(LANE_STRUCT.size, LANE_BYTES)
        self.assertEqual(PAIR_STRUCT.size, PAIR_BYTES)

        covered = (
            [TAG_VERSION_WORD]
            + list(range(FIRST_CHANNEL_WORD, GENERATION_WORD))
            + [GENERATION_WORD]
            + list(range(FIRST_RESERVED_WORD,
                         FIRST_RESERVED_WORD + RESERVED_COUNT))
        )
        self.assertEqual(sorted(covered), list(range(LANE_WORDS)))
        self.assertEqual(CHANNEL_COUNT, 28)
        self.assertEqual(len(covered), LANE_WORDS)

    def test_little_endian_word_order(self):
        channels = tuple((0x01020304 + index) & WORD_MASK
                         for index in range(CHANNEL_COUNT))
        lane = raw_lane(channels=channels, generation=0x11223344,
                        reserved=(0x55667788, 0x99AABBCC))
        packed = pack_lane(lane)
        self.assertEqual(len(packed), LANE_BYTES)
        self.assertEqual(packed[0:4], struct.pack("<I", tag_version_word(TAG_RAW)))
        for index, word in enumerate(lane.words):
            self.assertEqual(
                packed[index * WORD_BYTES:(index + 1) * WORD_BYTES],
                struct.pack("<I", word))
            self.assertEqual(struct.unpack_from("<I", packed, index * 4)[0], word)

    def test_pair_forward_then_reverse_layout(self):
        forward = raw_lane(channels=tuple(range(CHANNEL_COUNT)))
        reverse = raw_lane(channels=tuple(range(100, 100 + CHANNEL_COUNT)))
        pair = Pair(forward, reverse)
        packed = pack_pair(pair)
        self.assertEqual(len(packed), PAIR_BYTES)
        self.assertEqual(packed[:LANE_BYTES], forward.to_bytes())
        self.assertEqual(packed[LANE_BYTES:], reverse.to_bytes())
        self.assertEqual(unpack_pair(packed), pair)
        self.assertEqual(pair.words, forward.words + reverse.words)


class RoundTripTests(unittest.TestCase):
    PATTERNS = (
        0x00000000,
        0x00000001,
        0x3F800000,          # 1.0, but deliberately treated as opaque bits
        0x7F7FFFFF,          # maximum finite float bits
        0x7F800000,          # infinity bits
        0x7F800001,          # signalling NaN
        0x7FC00000,          # quiet NaN
        0xFFC00001,          # negative quiet NaN
        0x7FBFFFFF,          # signalling NaN, large payload
        0x80000000,          # negative zero bits
        0xFFFFFFFF,
        0xDEADBEEF,
        0x01234567,
        0x89ABCDEF,
    )

    def test_representative_words_round_trip(self):
        for pattern in self.PATTERNS:
            lane = raw_lane(
                channels=(pattern,) * CHANNEL_COUNT,
                generation=pattern,
                reserved=(pattern, pattern))
            packed = pack_lane(lane)
            self.assertEqual(len(packed), LANE_BYTES)
            self.assertEqual(unpack_lane(packed), lane)
            self.assertEqual(pack_lane(unpack_lane(packed)), packed)
            self.assertEqual(lane.words, (tag_version_word(TAG_RAW),)
                             + (pattern,) * CHANNEL_COUNT
                             + (pattern, pattern, pattern))

    def test_every_byte_value_at_every_byte_offset(self):
        base = pack_lane(raw_lane(
            channels=tuple((index * 0x01010101) & WORD_MASK
                           for index in range(CHANNEL_COUNT)),
            generation=0x10203040,
            reserved=(0x50607080, 0x90A0B0C0)))
        for byte_index in range(LANE_BYTES):
            for value in range(256):
                mutated = bytearray(base)
                mutated[byte_index] = value
                if byte_index >= WORD_BYTES:
                    lane = unpack_lane(mutated)
                    self.assertEqual(pack_lane(lane), bytes(mutated))
                else:
                    try:
                        lane = unpack_lane(mutated)
                    except LaneError:
                        continue
                    self.assertEqual(pack_lane(lane), bytes(mutated))

    def test_every_word_index_accepts_representative_patterns(self):
        base_words = list(raw_lane().words)
        for index in range(LANE_WORDS):
            for pattern in self.PATTERNS:
                words = list(base_words)
                if index == TAG_VERSION_WORD:
                    # Only the raw/version-1 identity is a valid identity.
                    words[index] = tag_version_word(TAG_RAW)
                else:
                    words[index] = pattern
                lane = Lane(words)
                packed = pack_lane(lane)
                self.assertEqual(unpack_lane(packed).words, lane.words)

    def test_random_32_bit_words_round_trip(self):
        rng = random.Random(2029)
        for _ in range(2000):
            words = [tag_version_word(TAG_RAW)]
            words.extend(rng.getrandbits(32) for _ in range(LANE_WORDS - 1))
            lane = Lane(words)
            packed = lane.to_bytes()
            self.assertEqual(len(packed), LANE_BYTES)
            self.assertEqual(unpack_lane(packed).words, lane.words)
            self.assertEqual(pack_lane(unpack_lane(packed)), packed)

    def test_random_quiet_and_signaling_payloads_round_trip(self):
        rng = random.Random(29029)
        for _ in range(300):
            quiet_words = tuple(quiet_word(rng) for _ in range(CHANNEL_COUNT))
            quiet = Lane.new(TAG_QUIET, channels=quiet_words,
                             generation=rng.getrandbits(32))
            self.assertEqual(unpack_lane(pack_lane(quiet)), quiet)
            self.assertTrue(all(is_quiet_nan(word) for word in quiet.channels))

            sig_words = tuple(signaling_word(rng) for _ in range(CHANNEL_COUNT))
            sig = Lane.new(TAG_SIGNALING, channels=sig_words,
                           generation=rng.getrandbits(32))
            self.assertEqual(unpack_lane(pack_lane(sig)), sig)
            self.assertTrue(all(is_signaling_nan(word) for word in sig.channels))

    def test_pack_lane_accepts_words_and_packed_bytes(self):
        lane = raw_lane(channels=tuple(range(CHANNEL_COUNT)))
        self.assertEqual(pack_lane(lane), pack_lane(lane.words))
        self.assertEqual(pack_lane(lane.to_bytes()), lane.to_bytes())
        self.assertEqual(lane.pack(), lane.to_bytes())

    def test_unpack_rejects_wrong_sizes_and_types(self):
        packed = pack_lane(raw_lane())
        for bad in (packed[:-1], packed + b"\x00", b"\x00" * 129):
            with self.assertRaises(LaneBoundsError):
                unpack_lane(bad)
        with self.assertRaises(TypeError):
            unpack_lane("not bytes")
        with self.assertRaises(TypeError):
            unpack_lane(None)
        with self.assertRaises(TypeError):
            Lane(b"\x00" * LANE_BYTES)


class WordAndChannelAccessTests(unittest.TestCase):
    def test_read_write_every_word_index(self):
        rng = random.Random(1729)
        lane = raw_lane()
        for index in range(LANE_WORDS):
            if index == TAG_VERSION_WORD:
                continue
            word = rng.getrandbits(32)
            updated = write_word(lane, index, word)
            self.assertEqual(read_word(updated, index), word)
            self.assertEqual(read_word(lane, index), lane.words[index])
            self.assertEqual(updated.words[index], word)
            self.assertNotEqual(updated, lane)
            # Every other word is untouched.
            for other in range(LANE_WORDS):
                if other != index:
                    self.assertEqual(updated.words[other], lane.words[other])

    def test_read_write_every_channel(self):
        rng = random.Random(1730)
        lane = raw_lane()
        for channel in range(CHANNEL_COUNT):
            word = rng.getrandbits(32)
            updated = set_channel(lane, channel, word)
            self.assertEqual(get_channel(updated, channel), word)
            self.assertEqual(
                updated.words[FIRST_CHANNEL_WORD + channel], word)
            self.assertEqual(get_channel(lane, channel), 0)
            self.assertEqual(updated, write_word(
                lane, FIRST_CHANNEL_WORD + channel, word))

    def test_generation_is_exact_modulo_2_to_32(self):
        lane = raw_lane()
        self.assertEqual(get_generation(lane), 0)
        self.assertEqual(bump_generation(lane).generation, 1)
        self.assertEqual(set_generation(lane, WORD_MASK).generation, WORD_MASK)
        self.assertEqual(bump_generation(set_generation(lane, WORD_MASK)).generation, 0)
        self.assertEqual(bump_generation(lane, -1).generation, WORD_MASK)
        self.assertEqual(bump_generation(lane, 1 << 40).generation,
                         (1 << 40) & WORD_MASK)

    def test_reserved_words(self):
        lane = raw_lane()
        updated = set_reserved(lane, 0, 0xDEADBEEF)
        self.assertEqual(get_reserved(updated, 0), 0xDEADBEEF)
        self.assertEqual(get_reserved(updated, 1), 0)
        updated = set_reserved(updated, 1, 0xCAFEBABE)
        self.assertEqual(get_reserved(updated, 1), 0xCAFEBABE)
        self.assertEqual(lane.reserved, (0, 0))

    def test_word_and_channel_bounds_and_type_denial(self):
        lane = raw_lane()
        for bad_index in (-1, LANE_WORDS, LANE_WORDS + 1, -100):
            with self.assertRaises(LaneBoundsError):
                read_word(lane, bad_index)
            with self.assertRaises(LaneBoundsError):
                write_word(lane, bad_index, 0)
        for bad_channel in (-1, CHANNEL_COUNT, CHANNEL_COUNT + 1):
            with self.assertRaises(LaneBoundsError):
                get_channel(lane, bad_channel)
            with self.assertRaises(LaneBoundsError):
                set_channel(lane, bad_channel, 0)
        for bad_reserved in (-1, RESERVED_COUNT, RESERVED_COUNT + 1):
            with self.assertRaises(LaneBoundsError):
                get_reserved(lane, bad_reserved)
            with self.assertRaises(LaneBoundsError):
                set_reserved(lane, bad_reserved, 0)

        for bad in (True, False):
            with self.assertRaises(LaneError):
                read_word(lane, bad)
            with self.assertRaises(LaneError):
                write_word(lane, bad, 0)
            with self.assertRaises(LaneError):
                get_channel(lane, bad)
            with self.assertRaises(LaneError):
                set_channel(lane, bad, 0)
            with self.assertRaises(LaneError):
                bump_generation(lane, bad)
            with self.assertRaises(LaneError):
                set_generation(lane, bad)

        for bad_word in (-1, WORD_MASK + 1, 1 << 40, 1.0, "1", object()):
            with self.assertRaises(LaneError):
                write_word(lane, 1, bad_word)
            with self.assertRaises(LaneError):
                set_channel(lane, 0, bad_word)

        for bad_generation in (-1, WORD_MASK + 1, 1 << 40, True):
            with self.assertRaises(LaneError):
                set_generation(lane, bad_generation)
        with self.assertRaises(LaneError):
            raw_lane(generation=-1)
        with self.assertRaises(LaneError):
            raw_lane(generation=WORD_MASK + 1)
        with self.assertRaises(LaneBoundsError):
            raw_lane(channels=(0,) * (CHANNEL_COUNT - 1))
        with self.assertRaises(LaneBoundsError):
            raw_lane(reserved=(0,))

    def test_lane_constructor_rejects_wrong_arity_and_bad_identity(self):
        with self.assertRaises(LaneBoundsError):
            Lane([tag_version_word(TAG_RAW)] * (LANE_WORDS - 1))
        with self.assertRaises(LaneBoundsError):
            Lane([tag_version_word(TAG_RAW)] * (LANE_WORDS + 1))
        with self.assertRaises(LaneTagError):
            Lane([0] * LANE_WORDS)
        with self.assertRaises(LaneTagError):
            Lane([(0x00FF << 16) | LANE_VERSION] + [0] * (LANE_WORDS - 1))
        with self.assertRaises(LaneTagError):
            Lane([(TAG_RAW << 16) | 2] + [0] * (LANE_WORDS - 1))


class TagAndPayloadTests(unittest.TestCase):
    def test_tag_version_word_round_trip(self):
        word = tag_version_word(TAG_QUIET)
        self.assertEqual(word, (TAG_QUIET << 16) | LANE_VERSION)
        self.assertEqual(split_tag_version(word), (TAG_QUIET, LANE_VERSION))
        for tag in (TAG_RAW, TAG_QUIET, TAG_SIGNALING):
            lane = Lane.new(tag)
            self.assertEqual(lane.tag, tag)
            self.assertEqual(lane.version, LANE_VERSION)
            self.assertEqual(lane.identity, tag_version_word(tag))
        with self.assertRaises(LaneTagError):
            tag_version_word(99)
        with self.assertRaises(LaneTagError):
            tag_version_word(TAG_RAW, version=2)
        with self.assertRaises(LaneTagError):
            Lane.new(99)

    def test_nan_predicates(self):
        self.assertFalse(is_nan(0x7F800000))       # infinity
        self.assertFalse(is_nan(0x00000000))
        self.assertTrue(is_nan(QUIET_NAN_WORD))
        self.assertTrue(is_quiet_nan(QUIET_NAN_WORD))
        self.assertFalse(is_signaling_nan(QUIET_NAN_WORD))
        self.assertTrue(is_nan(SIGNALING_NAN_WORD))
        self.assertTrue(is_signaling_nan(SIGNALING_NAN_WORD))
        self.assertFalse(is_quiet_nan(SIGNALING_NAN_WORD))
        self.assertTrue(is_quiet_nan(0xFFC00001))
        self.assertTrue(is_signaling_nan(0x7FBFFFFF))
        self.assertEqual(nan_class(QUIET_NAN_WORD), "quiet")
        self.assertEqual(nan_class(SIGNALING_NAN_WORD), "signaling")
        self.assertIsNone(nan_class(0x7F800000))
        self.assertIsNone(nan_class(0x3F800000))

    def test_raw_tag_accepts_every_word_pattern(self):
        patterns = (
            0x00000000, 0x7F800000, 0x7F800001, 0x7FC00000,
            0xFFC00001, 0x7FFFFFFF, 0xFFFFFFFF, 0x80000000,
        )
        for pattern in patterns:
            lane = raw_lane(channels=(pattern,) * CHANNEL_COUNT,
                            generation=pattern,
                            reserved=(pattern, pattern))
            self.assertEqual(lane.channels, (pattern,) * CHANNEL_COUNT)
            self.assertEqual(unpack_lane(pack_lane(lane)).channels,
                             (pattern,) * CHANNEL_COUNT)

    def test_quiet_tag_accepts_only_quiet_nan_channels(self):
        lane = Lane.new(TAG_QUIET)
        self.assertTrue(all(is_quiet_nan(word) for word in lane.channels))
        self.assertTrue(all(not is_signaling_nan(word) for word in lane.channels))
        with self.assertRaises(LaneTagError):
            Lane.new(TAG_QUIET, channels=(0,) * CHANNEL_COUNT)
        with self.assertRaises(LaneTagError):
            Lane.new(TAG_QUIET, channels=(SIGNALING_NAN_WORD,) * CHANNEL_COUNT)
        with self.assertRaises(LaneTagError):
            Lane.new(TAG_QUIET, channels=(0x7F800000,) * CHANNEL_COUNT)
        with self.assertRaises(LaneTagError):
            set_channel(lane, 0, SIGNALING_NAN_WORD)
        with self.assertRaises(LaneTagError):
            write_word(lane, FIRST_CHANNEL_WORD, 0)
        self.assertTrue(all(is_quiet_nan(word) for word in lane.channels))

    def test_signaling_tag_accepts_only_signaling_nan_channels(self):
        lane = Lane.new(TAG_SIGNALING)
        self.assertTrue(all(is_signaling_nan(word) for word in lane.channels))
        self.assertTrue(all(not is_quiet_nan(word) for word in lane.channels))
        with self.assertRaises(LaneTagError):
            Lane.new(TAG_SIGNALING, channels=(0,) * CHANNEL_COUNT)
        with self.assertRaises(LaneTagError):
            Lane.new(TAG_SIGNALING, channels=(QUIET_NAN_WORD,) * CHANNEL_COUNT)
        with self.assertRaises(LaneTagError):
            set_channel(lane, 0, QUIET_NAN_WORD)
        with self.assertRaises(LaneTagError):
            write_word(lane, FIRST_CHANNEL_WORD, 0x7F800000)

    def test_identity_is_immutable_after_creation(self):
        lane = raw_lane()
        identity = lane.identity
        self.assertEqual(write_word(lane, TAG_VERSION_WORD, identity), lane)
        for replacement in (
                tag_version_word(TAG_QUIET),
                tag_version_word(TAG_SIGNALING),
                (TAG_RAW << 16) | 0,
                (TAG_RAW << 16) | 0xFFFF,
                0xFFFFFFFF,
                0):
            with self.assertRaises(IdentityRewriteError):
                write_word(lane, TAG_VERSION_WORD, replacement)
            self.assertEqual(lane.identity, identity)

    def test_tag_channel_validation_is_reapplied_on_every_write(self):
        lane = Lane.new(TAG_QUIET)
        # A valid quiet word is accepted and a bad word cannot sneak in.
        self.assertTrue(is_quiet_nan(set_channel(
            lane, 5, 0xFFC00001).get_channel(5)))
        with self.assertRaises(LaneTagError):
            set_channel(lane, 5, 0x7F800001)
        with self.assertRaises(LaneTagError):
            write_word(lane, FIRST_CHANNEL_WORD + 5, 0x00000000)


class MirrorTests(unittest.TestCase):
    def test_mirror_reverses_only_channels(self):
        rng = random.Random(20290)
        lane = raw_lane(
            channels=tuple(rng.getrandbits(32) for _ in range(CHANNEL_COUNT)),
            generation=0x12345678,
            reserved=(0x9ABCDEF0, 0x0FEDCBA9))
        mirrored = mirror_lane(lane)
        self.assertEqual(mirrored.channels, tuple(reversed(lane.channels)))
        self.assertEqual(mirrored.identity, lane.identity)
        self.assertEqual(mirrored.tag, lane.tag)
        self.assertEqual(mirrored.version, lane.version)
        self.assertEqual(mirrored.generation, lane.generation)
        self.assertEqual(mirrored.reserved, lane.reserved)
        self.assertEqual(mirror_lane(mirrored), lane)
        self.assertEqual(unpack_lane(pack_lane(mirrored)), mirrored)

    def test_mirror_preserves_nan_class_and_payload_bits(self):
        rng = random.Random(20291)
        quiet = Lane.new(TAG_QUIET, channels=tuple(
            quiet_word(rng) for _ in range(CHANNEL_COUNT)))
        mirrored = quiet.mirror()
        self.assertTrue(all(is_quiet_nan(word) for word in mirrored.channels))
        self.assertEqual(tuple(sorted(mirrored.channels)),
                         tuple(sorted(quiet.channels)))
        self.assertEqual(mirror_lane(mirrored), quiet)

        sig = Lane.new(TAG_SIGNALING, channels=tuple(
            signaling_word(rng) for _ in range(CHANNEL_COUNT)))
        self.assertTrue(all(is_signaling_nan(word)
                            for word in mirror_lane(sig).channels))

    def test_pair_opposite_swaps_orientation_and_mirror_reverses_both(self):
        forward = raw_lane(channels=tuple(range(CHANNEL_COUNT)))
        reverse = raw_lane(channels=tuple(range(100, 100 + CHANNEL_COUNT)))
        pair = Pair(forward, reverse)
        opposite = pair.opposite()
        self.assertIs(opposite.forward, reverse)
        self.assertIs(opposite.reverse, forward)
        self.assertEqual(opposite.opposite(), pair)
        mirrored = pair.mirror()
        self.assertEqual(mirrored.forward, forward.mirror())
        self.assertEqual(mirrored.reverse, reverse.mirror())
        self.assertEqual(mirrored.mirror(), pair)


class PairTests(unittest.TestCase):
    def test_pair_new_sizes_and_accessors(self):
        pair = Pair.new()
        self.assertEqual(len(pair.forward.words), LANE_WORDS)
        self.assertEqual(len(pair.reverse.words), LANE_WORDS)
        self.assertEqual(len(pair.words), PAIR_WORDS)
        self.assertEqual(len(pair.to_bytes()), PAIR_BYTES)
        self.assertIs(pair.s_plus, pair.forward)
        self.assertIs(pair.s_minus, pair.reverse)
        self.assertIs(pair.forward_lane, pair.forward)
        self.assertIs(pair.reverse_lane, pair.reverse)
        self.assertIs(pair.lane(ORIENT_FORWARD), pair.forward)
        self.assertIs(pair.lane(ORIENT_REVERSE), pair.reverse)
        self.assertIs(pair.lane("forward"), pair.forward)
        self.assertIs(pair.lane("s-"), pair.reverse)

    def test_pair_round_trip_random_words(self):
        rng = random.Random(30290)
        for _ in range(100):
            words = [tag_version_word(TAG_RAW)]
            words.extend(rng.getrandbits(32) for _ in range(LANE_WORDS - 1))
            words.append(tag_version_word(TAG_RAW))
            words.extend(rng.getrandbits(32) for _ in range(LANE_WORDS - 1))
            pair = Pair(
                Lane(words[:LANE_WORDS]),
                Lane(words[LANE_WORDS:]))
            packed = pair.to_bytes()
            self.assertEqual(len(packed), PAIR_BYTES)
            self.assertEqual(pack_pair(pair), packed)
            self.assertEqual(unpack_pair(packed), pair)
            self.assertEqual(Pair.from_bytes(packed), pair)
            self.assertEqual(pack_pair(unpack_pair(packed)), packed)

    def test_pair_with_lane_and_orientation_validation(self):
        pair = Pair.new()
        replacement = raw_lane(channels=tuple(range(CHANNEL_COUNT)))
        self.assertIs(pair.with_lane(ORIENT_FORWARD, replacement).forward,
                      replacement)
        self.assertIs(pair.with_lane(ORIENT_REVERSE, replacement).reverse,
                      replacement)
        for bad in (0, 2, -2, 99, True, False):
            with self.assertRaises(LaneError):
                pair.lane(bad)
            with self.assertRaises(LaneError):
                pair.with_lane(bad, replacement)
            with self.assertRaises(LaneError):
                deposit_oriented(pair, 0, 0, bad)

    def test_pair_size_and_type_denial(self):
        packed = pack_pair(Pair.new())
        for bad in (packed[:-1], packed + b"\x00", b"\x00" * 257):
            with self.assertRaises(LaneBoundsError):
                unpack_pair(bad)
        with self.assertRaises(TypeError):
            unpack_pair("not bytes")
        with self.assertRaises(TypeError):
            Pair(Lane.new(), "not a lane")
        with self.assertRaises(TypeError):
            Pair(Lane.new(), 3)


class DepositAndCycleTests(unittest.TestCase):
    def test_deposit_oriented_on_lane_is_exact_modulo(self):
        lane = raw_lane()
        self.assertEqual(get_channel(deposit_oriented(lane, 3, 5), 3), 5)
        self.assertEqual(
            get_channel(deposit_oriented(lane, 3, 5, ORIENT_REVERSE), 3),
            (-5) & WORD_MASK)
        self.assertEqual(
            get_channel(deposit_oriented(lane, 3, WORD_MASK), 3), WORD_MASK)
        # Wrap-around and sign orientation.
        lane = deposit_oriented(lane, 3, WORD_MASK)
        lane = deposit_oriented(lane, 3, 2)
        self.assertEqual(get_channel(lane, 3), 1)
        lane = deposit_oriented(lane, 3, 1, ORIENT_REVERSE)
        self.assertEqual(get_channel(lane, 3), 0)
        self.assertEqual(
            get_channel(deposit_oriented(lane, 3, 1, "reverse"), 3),
            WORD_MASK)
        self.assertEqual(
            get_channel(deposit_oriented(lane, 3, 1, "forward"), 3), 1)

    def test_deposit_oriented_pair_keeps_exact_antisymmetry(self):
        pair = Pair.new()
        pair = deposit_oriented(pair, 2, 7)
        self.assertEqual(get_channel(pair.forward, 2), 7)
        self.assertEqual(get_channel(pair.reverse, 2), (-7) & WORD_MASK)
        pair = deposit_oriented(pair, 2, 3, ORIENT_REVERSE)
        self.assertEqual(get_channel(pair.forward, 2), 4)
        self.assertEqual(get_channel(pair.reverse, 2), (-4) & WORD_MASK)
        self.assertEqual(
            (get_channel(pair.forward, 2) + get_channel(pair.reverse, 2))
            & WORD_MASK, 0)
        # An independent lane operation leaves the pair representation alone.
        self.assertEqual(get_channel(pair.forward, 3), 0)

    def test_cycle_word_raw_arithmetic_is_modulo_2_to_32(self):
        cases = (
            (0, 0, 0, 0),
            (WORD_MASK, 1, 0, 0),
            (0, 0, 1, WORD_MASK),
            (0x80000000, 0x80000000, 0, 0),
            (WORD_MASK, WORD_MASK, WORD_MASK, WORD_MASK),
            (0x7FC00000, 0x7F800001, 0xFFFFFFFF, 0xFF400002),
            (1, 2, 3, 0),
        )
        for a, b, c, expected in cases:
            self.assertEqual(cycle_word(a, b, c), expected)
            self.assertEqual(cycle_word([a, b, c]), expected)
            self.assertEqual(cycle_word((a, b, c), channel=9), expected)
        self.assertEqual(cycle_word(WORD_MASK, 1, 0), 0)
        self.assertEqual(cycle_word(a=WORD_MASK, b=1, c=0), 0)
        self.assertIs(type(cycle_word(1, 2, 3)), int)

    def test_cycle_word_accepts_lanes_and_pairs(self):
        a = raw_lane(channels=tuple(range(CHANNEL_COUNT)))
        b = raw_lane(channels=tuple(range(10, 10 + CHANNEL_COUNT)))
        c = raw_lane(channels=tuple(range(20, 20 + CHANNEL_COUNT)))
        for channel in range(CHANNEL_COUNT):
            expected = (channel + 10 + channel - (20 + channel)) & WORD_MASK
            self.assertEqual(cycle_word(a, b, c, channel), expected)
            pair = Pair(a, c)
            self.assertEqual(cycle_word(pair, b, c, channel), expected)
        with self.assertRaises(LaneBoundsError):
            cycle_word(a, b, c, CHANNEL_COUNT)
        with self.assertRaises(LaneError):
            cycle_word(a, b, c, True)

    def test_cycle_gauge_cancels_exactly_for_arbitrary_32_bit_gauges(self):
        rng = random.Random(29029)
        for _ in range(200):
            gauge = [rng.getrandbits(32) for _ in range(3)]
            semantic = [rng.getrandbits(32) for _ in range(CHANNEL_COUNT)]
            e01 = raw_lane(channels=tuple(
                (gauge[0] - gauge[1]) & WORD_MASK
                for _ in range(CHANNEL_COUNT)))
            e12 = raw_lane(channels=tuple(
                (semantic[channel] + gauge[1] - gauge[2]) & WORD_MASK
                for channel in range(CHANNEL_COUNT)))
            e02 = raw_lane(channels=tuple(
                (gauge[0] - gauge[2]) & WORD_MASK
                for _ in range(CHANNEL_COUNT)))
            for channel in range(CHANNEL_COUNT):
                self.assertEqual(
                    cycle_word(e01, e12, e02, channel), semantic[channel])
            # A fresh vertex-gauge shift applied after construction leaves the
            # cycle word bit-exactly unchanged.
            shift = [rng.getrandbits(32) for _ in range(3)]
            for channel in range(CHANNEL_COUNT):
                e01 = deposit_oriented(
                    e01, channel, (shift[0] - shift[1]) & WORD_MASK)
                e12 = deposit_oriented(
                    e12, channel, (shift[1] - shift[2]) & WORD_MASK)
                e02 = deposit_oriented(
                    e02, channel, (shift[0] - shift[2]) & WORD_MASK)
                self.assertEqual(
                    cycle_word(e01, e12, e02, channel), semantic[channel])

    def test_cycle_gauge_with_pair_deposits(self):
        rng = random.Random(29030)
        gauge = [rng.getrandbits(32) for _ in range(3)]
        semantic = [rng.getrandbits(32) for _ in range(CHANNEL_COUNT)]
        p01 = Pair.new()
        p12 = Pair.new()
        p02 = Pair.new()
        for channel in range(CHANNEL_COUNT):
            p01 = deposit_oriented(
                p01, channel, (gauge[0] - gauge[1]) & WORD_MASK)
            p12 = deposit_oriented(
                p12, channel,
                (semantic[channel] + gauge[1] - gauge[2]) & WORD_MASK)
            p02 = deposit_oriented(
                p02, channel, (gauge[0] - gauge[2]) & WORD_MASK)
        for channel in range(CHANNEL_COUNT):
            self.assertEqual(
                cycle_word(p01, p12, p02, channel), semantic[channel])
            self.assertEqual(
                (get_channel(p01.forward, channel)
                 + get_channel(p01.reverse, channel)) & WORD_MASK, 0)
            self.assertEqual(
                (get_channel(p12.forward, channel)
                 + get_channel(p12.reverse, channel)) & WORD_MASK, 0)
            self.assertEqual(
                (get_channel(p02.forward, channel)
                 + get_channel(p02.reverse, channel)) & WORD_MASK, 0)

    def test_deposit_and_cycle_deny_bad_words_and_channels(self):
        lane = raw_lane()
        for bad_word in (-1, WORD_MASK + 1, 1 << 40, True, False):
            with self.assertRaises(LaneError):
                deposit_oriented(lane, 0, bad_word)
        for bad_channel in (-1, CHANNEL_COUNT, True):
            with self.assertRaises(LaneError):
                deposit_oriented(lane, bad_channel, 0)
        for bad in (None, "x", object()):
            with self.assertRaises((LaneError, TypeError)):
                cycle_word(lane, lane, bad)


class NoFloatPathTests(unittest.TestCase):
    def test_module_source_has_no_float_literals_calls_or_division(self):
        source = inspect.getsource(lane_module)
        tree = ast.parse(source)
        float_literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, float)
        ]
        self.assertEqual(float_literals, [])
        float_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
        ]
        self.assertEqual(float_calls, [])
        divisions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        ]
        self.assertEqual(divisions, [])
        # Only integer word formats are present; no f/d/e/G float codes.
        formats = {WORD_STRUCT.format, LANE_STRUCT.format, PAIR_STRUCT.format}
        self.assertEqual(formats, {"<I", "<32I", "<64I"})
        self.assertFalse(any(code in "".join(formats) for code in "fdeG"))

    def test_all_word_results_are_plain_integers(self):
        rng = random.Random(40290)
        lane = raw_lane(channels=tuple(
            rng.getrandbits(32) for _ in range(CHANNEL_COUNT)))
        values = [
            read_word(lane, 0),
            get_channel(lane, 0),
            get_generation(lane),
            get_reserved(lane, 0),
            cycle_word(lane, lane, lane, 0),
            mirror_lane(lane).get_channel(0),
            deposit_oriented(lane, 0, 1).get_channel(0),
        ]
        for value in values:
            self.assertIs(type(value), int)


class PublicApiTests(unittest.TestCase):
    def test_public_names_exist(self):
        for name in lane_module.__all__:
            self.assertTrue(hasattr(lane_module, name), name)

    def test_pack_unpack_composition_is_identity_for_every_raw_byte_pattern(self):
        # One pack/unpack composition for each possible byte value in each of
        # the 128 lane positions, using raw lanes so no byte is constrained.
        base = bytearray(pack_lane(raw_lane()))
        for index in range(LANE_BYTES):
            for value in range(256):
                changed = bytearray(base)
                changed[index] = value
                if index < WORD_BYTES:
                    word0 = int.from_bytes(changed[:4], "little")
                    tag, version = split_tag_version(word0)
                    if tag != TAG_RAW or version != LANE_VERSION:
                        continue
                lane = unpack_lane(changed)
                self.assertEqual(pack_lane(lane), bytes(changed))


if __name__ == "__main__":
    unittest.main()
