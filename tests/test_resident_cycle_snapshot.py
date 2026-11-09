"""Tests for the Round 030 ``NANCYC30`` snapshot inspector.

The suite uses the three supplied roots when they are present and otherwise
falls back to synthetic snapshots for the same layout.  Raw lanes are always
checked as 32-bit words; float decoding is only used where the native flux
calculation itself is defined that way.
"""
from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from nanlmind_hermes.resident import cycle_snapshot as cs
from nanlmind_hermes.resident import nan_lane


def _blank_snapshot(marker=cs.MARKER):
    raw = bytearray(marker + bytes(cs.ROOT_BYTES))
    assert len(raw) == cs.SNAPSHOT_BYTES
    return raw


def _cell_offset(i, j):
    identity = cs.pair_identity(i, j)
    return cs.MARKER_BYTES + cs.PAIR_REGION_OFFSET + (
        identity.index * cs.PAIR_BYTES)


def _lane_offset(i, j):
    offset = _cell_offset(i, j)
    return offset if i < j else offset + cs.LANE_BYTES


def _set_edge_bits(raw, i, j, channel, word):
    struct.pack_into("<I", raw, _lane_offset(i, j) + channel * 4, word)


def _set_edge_float(raw, i, j, channel, value):
    struct.pack_into("<f", raw, _lane_offset(i, j) + channel * 4, value)


def _skew(raw, i, j, channel, value):
    _set_edge_float(raw, i, j, channel, value)
    _set_edge_float(raw, j, i, channel, -value)


def _set_head(raw, index, data):
    if len(data) != cs.HEAD_BYTES:
        raise ValueError("head data must be 128 bytes")
    offset = cs.MARKER_BYTES + index * cs.HEAD_BYTES
    raw[offset:offset + cs.HEAD_BYTES] = data


def _flux(snapshot, i, j, channel):
    def edge(first, second):
        return cs.snapshot_float_words(snapshot, first, second, channel)

    total = cs._round_float32(edge(0, i) + edge(i, j))
    return cs._round_float32(total - edge(0, j))


class LayoutTests(unittest.TestCase):
    def test_exact_native_sizes(self):
        self.assertEqual(cs.MARKER, b"NANCYC30")
        self.assertEqual(cs.MARKER_BYTES, 8)
        self.assertEqual(cs.HEAD_COUNT, 256)
        self.assertEqual(cs.HEAD_BYTES, 128)
        self.assertEqual(cs.PAIR_COUNT, 32640)
        self.assertEqual(cs.DIRECTION_WORDS, 32)
        self.assertEqual(cs.LANE_BYTES, 128)
        self.assertEqual(cs.PAIR_BYTES, 256)
        self.assertEqual(cs.HEAD_REGION_BYTES, 32768)
        self.assertEqual(cs.ROOT_BYTES, 8388608)
        self.assertEqual(cs.SNAPSHOT_BYTES, 8388616)
        self.assertEqual(cs.PAIR_REGION_OFFSET, cs.HEAD_REGION_BYTES)
        self.assertEqual(
            cs.HEAD_REGION_BYTES + cs.PAIR_COUNT * cs.PAIR_BYTES,
            cs.ROOT_BYTES)

    def test_pair_index_matches_native_formula_exhaustively(self):
        seen = set()
        for first in range(cs.HEAD_COUNT):
            for second in range(first + 1, cs.HEAD_COUNT):
                index = cs.pair_index(first, second)
                self.assertEqual(cs.pair_index(second, first), index)
                self.assertEqual(cs.pair_heads(index), (first, second))
                seen.add(index)
        self.assertEqual(seen, set(range(cs.PAIR_COUNT)))
        self.assertEqual(cs.pair_index(0, 1), 0)
        self.assertEqual(cs.pair_index(254, 255), 32639)

    def test_pair_identity_exposes_mirror_on_same_cell(self):
        identity = cs.pair_identity(3, 240)
        self.assertEqual(identity.canonical, (3, 240))
        self.assertEqual(identity.mirror, (240, 3))
        self.assertEqual(identity.mirror_index, identity.index)
        self.assertEqual(cs.pair_identity(identity.index), identity)

    def test_bad_heads_indices_and_channels(self):
        for bad in (-1, cs.HEAD_COUNT, True, 1.5, "1"):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.pair_index(bad, 2)
            with self.assertRaises(cs.SnapshotRangeError):
                cs.Snapshot(_blank_snapshot()).head_bytes(bad)
        for bad in (-1, cs.PAIR_COUNT, True, 1.25, "0"):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.pair_heads(bad)
        for bad in (-1, cs.DIRECTION_WORDS, True, 2.0, "0"):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.snapshot_raw_lanes(_blank_snapshot(), 0, 1, bad)
        with self.assertRaises(cs.SnapshotRangeError):
            cs.snapshot_raw_lanes(_blank_snapshot(), 4, 4, 0)


class ParseTests(unittest.TestCase):
    def setUp(self):
        raw = _blank_snapshot()
        _set_head(raw, 0, bytes(range(128)))
        _set_head(raw, 255, bytes(range(128, 256)))
        _set_edge_bits(raw, 0, 1, 0, 0x7FC00001)
        _set_edge_bits(raw, 1, 0, 0, 0xFFC00001)
        for channel in range(1, cs.DIRECTION_WORDS):
            _set_edge_float(raw, 0, 1, channel, channel / 8.0)
            _set_edge_float(raw, 1, 0, channel, -channel / 8.0)
        self.raw = bytes(raw)
        self.snapshot = cs.parse_snapshot(self.raw, source="synthetic")

    def test_parse_reports_exact_sizes_and_digest(self):
        snapshot = self.snapshot
        self.assertEqual(snapshot.marker, cs.MARKER)
        self.assertEqual(snapshot.file_bytes, cs.SNAPSHOT_BYTES)
        self.assertEqual(snapshot.root_bytes, cs.ROOT_BYTES)
        self.assertEqual(snapshot.head_count, 256)
        self.assertEqual(snapshot.pair_count, 32640)
        self.assertEqual(snapshot.sha256,
                         hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(len(snapshot), cs.SNAPSHOT_BYTES)
        self.assertEqual(snapshot.source, "synthetic")

    def test_head_bytes_are_exact_and_bounded(self):
        self.assertEqual(self.snapshot.head_bytes(0), bytes(range(128)))
        self.assertEqual(self.snapshot.head_bytes(255),
                         bytes(range(128, 256)))
        self.assertEqual(tuple(self.snapshot.iter_heads(limit=2)),
                         (bytes(range(128)), bytes(cs.HEAD_BYTES)))
        self.assertEqual(sum(1 for _ in self.snapshot.iter_heads()), 256)

    def test_pair_cell_reads_both_directions_and_aliases(self):
        cell = self.snapshot.pair_cell(0)
        self.assertEqual(cell.index, 0)
        self.assertEqual(cell.canonical, (0, 1))
        self.assertEqual(cell.mirror, (1, 0))
        self.assertEqual(cell.mirror_index, 0)
        self.assertEqual(len(cell.forward_words), 32)
        self.assertEqual(len(cell.reverse_words), 32)
        self.assertEqual(cell.forward_words[0], 0x7FC00001)
        self.assertEqual(cell.reverse_words[0], 0xFFC00001)
        self.assertEqual(cell.forward, cell.forward_words)
        self.assertEqual(cell.reverse, cell.reverse_words)
        self.assertEqual(self.snapshot.pair_cell(1, 0), cell)
        self.assertEqual(cell.oriented_words(1, 0), cell.reverse_words)
        self.assertEqual(cell.oriented_words(0, 1), cell.forward_words)
        with self.assertRaises(cs.SnapshotRangeError):
            cell.oriented_words(0, 2)

        first_cells = tuple(self.snapshot.iter_pairs(limit=2))
        self.assertEqual(len(first_cells), 2)
        self.assertEqual(first_cells[0], cell)
        self.assertEqual(sum(1 for _ in self.snapshot.pair_cells()), 32640)

    def test_float_and_raw_views_are_consistent(self):
        for channel in range(1, cs.DIRECTION_WORDS):
            value = self.snapshot.float_words(0, 1, channel)
            self.assertEqual(value, channel / 8.0)
            bits = cs.snapshot_raw_lanes(self.snapshot, 0, 1, channel)
            self.assertEqual(
                bits, struct.unpack("<I", struct.pack("<f", value))[0])
        self.assertEqual(len(self.snapshot.float_words(0, 1)), 32)
        self.assertEqual(len(self.snapshot.raw_words(0, 1)), 32)

    def test_expected_digest_is_checked(self):
        digest = hashlib.sha256(self.raw).hexdigest()
        self.assertEqual(
            cs.parse_snapshot(self.raw, expected_sha256=digest).sha256,
            digest)
        with self.assertRaises(cs.SnapshotIntegrityError):
            cs.parse_snapshot(self.raw, expected_sha256="0" * 64)

    def test_malformed_and_truncated_bytes_are_rejected(self):
        cases = (
            (self.raw[:-1], cs.SnapshotSizeError),
            (self.raw + b"\x00", cs.SnapshotSizeError),
            (b"", cs.SnapshotSizeError),
            (b"BADMAGIC" + self.raw[8:], cs.SnapshotMarkerError),
            (b"NANCYC29" + self.raw[8:], cs.SnapshotMarkerError),
        )
        for data, error in cases:
            with self.assertRaises(error):
                cs.parse_snapshot(data)
            with self.assertRaises(error):
                cs.Snapshot(data)
        with self.assertRaises(TypeError):
            cs.parse_snapshot("not bytes")
        with self.assertRaises(TypeError):
            cs.Snapshot(None)

    def test_mutable_and_view_inputs_are_copied(self):
        mutable = bytearray(self.raw)
        view = memoryview(self.raw)
        a = cs.Snapshot(mutable)
        b = cs.Snapshot(view)
        self.assertEqual(a.raw, self.raw)
        self.assertEqual(b.raw, self.raw)
        mutable[0] ^= 0xFF
        self.assertEqual(a.raw, self.raw)


class SyntheticFluxTests(unittest.TestCase):
    def setUp(self):
        raw = _blank_snapshot()
        _skew(raw, 0, 1, 3, 1.25)
        _skew(raw, 1, 2, 3, 2.5)
        _skew(raw, 0, 2, 3, 0.75)
        self.raw = bytes(raw)
        self.snapshot = cs.parse_snapshot(self.raw)

    def test_cycle_reads_three_lanes_and_computes_native_flux(self):
        self.assertEqual(cs.snapshot_cycle(self.snapshot, 1, 2, 3), 3.0)
        self.assertEqual(cs.snapshot_cycle(self.snapshot, 2, 1, 3), -3.0)
        self.assertEqual(cs.snapshot_flux(self.snapshot, 0, 3), 3.0)
        self.assertEqual(self.snapshot.cycle(1, 2, 3), 3.0)
        self.assertEqual(_flux(self.snapshot, 1, 2, 3), 3.0)

    def test_cycle_uses_float32_rounding(self):
        raw = _blank_snapshot()
        _skew(raw, 0, 1, 0, 16777216.0)
        _skew(raw, 1, 2, 0, 1.0)
        _skew(raw, 0, 2, 0, -1.0)
        snapshot = cs.parse_snapshot(raw)
        # 2**24 + 1 rounds back to 2**24 in binary32.
        self.assertEqual(cs.snapshot_cycle(snapshot, 1, 2, 0), 16777216.0)

    def test_root_can_be_selected_and_must_be_distinct(self):
        raw = _blank_snapshot()
        _skew(raw, 5, 6, 1, 0.5)
        _skew(raw, 6, 7, 1, 1.5)
        _skew(raw, 5, 7, 1, -0.25)
        snapshot = cs.parse_snapshot(raw)
        self.assertEqual(
            cs.snapshot_cycle(snapshot, 6, 7, 1, root=5), 2.25)
        for bad in (6, 7):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.snapshot_cycle(snapshot, 6, 7, 1, root=bad)
        for bad in range(cs.FLUX_DIMENSIONS, cs.FLUX_DIMENSIONS + 2):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.snapshot_flux(snapshot, bad, 0)

    def test_zero_initialised_cell_passes_cycle_and_audit(self):
        snapshot = cs.parse_snapshot(_blank_snapshot())
        self.assertEqual(cs.snapshot_cycle(snapshot, 1, 2, 0), 0.0)
        report = cs.audit(snapshot, limit=4)
        self.assertTrue(report)
        self.assertTrue(report["ok"])
        self.assertTrue(report["size_ok"])
        self.assertTrue(report["marker_ok"])
        self.assertTrue(report["mirror_antisymmetry"])
        self.assertTrue(report["finite_bits"])


class RawBitBridgeTests(unittest.TestCase):
    PATTERNS = (
        0x00000000,
        0x80000000,
        0x3F800000,
        0x7F7FFFFF,
        0x7F800000,
        0x7F800001,
        0x7FC00000,
        0x7FC00001,
        0xFFC00002,
        0xDEADBEEF,
        0x01234567,
        0xFFFFFFFF,
    )

    def setUp(self):
        raw = _blank_snapshot()
        words = tuple(
            self.PATTERNS[index % len(self.PATTERNS)]
            for index in range(cs.DIRECTION_WORDS))
        for channel, word in enumerate(words):
            _set_edge_bits(raw, 0, 1, channel, word)
            mirror = word ^ cs.FLOAT_SIGN_BIT
            _set_edge_bits(raw, 1, 0, channel, mirror)
        self.words = words
        self.snapshot = cs.parse_snapshot(raw)

    def test_all_32_raw_words_round_trip_exactly(self):
        self.assertEqual(
            cs.snapshot_raw_lanes(self.snapshot, 0, 1), self.words)
        self.assertEqual(
            cs.snapshot_raw_lane(self.snapshot, 1, 0),
            tuple(word ^ cs.FLOAT_SIGN_BIT for word in self.words))
        cell = self.snapshot.pair_cell(0, 1)
        self.assertEqual(cell.forward_words, self.words)
        self.assertEqual(
            cell.reverse_words,
            tuple(word ^ cs.FLOAT_SIGN_BIT for word in self.words))

    def test_selected_float_bits_move_into_raw_nan_lane(self):
        for channel in range(cs.DIRECTION_WORDS):
            expected = cs.snapshot_raw_lanes(
                self.snapshot, 0, 1, channel)
            lane = cs.to_nan_lane(self.snapshot, 0, 1, channel)
            self.assertEqual(lane.tag, nan_lane.TAG_RAW)
            self.assertEqual(lane.get_channel(0), expected)
            self.assertEqual(cs.from_nan_lane(lane)[0], expected)
            self.assertTrue(
                all(word == 0 for word in lane.channels[1:]))
            self.assertEqual(
                nan_lane.unpack_lane(cs.to_nan_lane_bytes(
                    self.snapshot, 0, 1, channel)),
                lane)

    def test_raw_word_sequence_preserves_payloads_without_canonicalising(self):
        lane = cs.to_nan_lane(self.PATTERNS)
        self.assertEqual(lane.channels[:len(self.PATTERNS)], self.PATTERNS)
        self.assertTrue(
            all(word == 0 for word in lane.channels[len(self.PATTERNS):]))
        self.assertEqual(
            nan_lane.unpack_lane(lane.to_bytes()).channels,
            lane.channels)

    def test_lane_capacity_is_not_silently_truncated(self):
        with self.assertRaises(cs.SnapshotLaneCapacityError):
            cs.to_nan_lane(self.snapshot, 0, 1)
        with self.assertRaises(cs.SnapshotLaneCapacityError):
            cs.to_nan_lane(tuple(range(nan_lane.CHANNEL_COUNT + 1)))
        with self.assertRaises(cs.SnapshotLaneCapacityError):
            cs.to_nan_lane(self.words)
        with self.assertRaises(cs.SnapshotLaneCapacityError):
            cs.to_nan_lane([])
        with self.assertRaises(cs.SnapshotRangeError):
            cs.to_nan_lane(1 << 32)
        with self.assertRaises(TypeError):
            cs.to_nan_lane("words")
        with self.assertRaises(TypeError):
            cs.from_nan_lane(b"not a lane")

    def test_no_float_identity_or_merge_claim(self):
        self.assertIs(cs.CLAIMS_FLOAT_IDENTITY, False)
        self.assertIs(cs.PERFORMS_CANONICAL_MERGE, False)
        report = cs.audit(self.snapshot, limit=1)
        self.assertFalse(report["float_identity_claim"])
        self.assertFalse(report["canonical_merge"])
        source = Path(cs.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import numpy", source)


class AuditTests(unittest.TestCase):
    def test_mirror_mismatch_is_reported(self):
        raw = _blank_snapshot()
        _set_edge_float(raw, 0, 1, 4, 1.5)
        _set_edge_float(raw, 1, 0, 4, 1.5)
        report = cs.audit(raw, limit=1)
        self.assertFalse(report)
        self.assertFalse(report["ok"])
        self.assertFalse(report["mirror_antisymmetry"])
        self.assertEqual(report["mirror_mismatches"], 1)
        self.assertEqual(report["first_bad_pair"], 0)
        self.assertEqual(report["first_bad_channel"], 4)
        self.assertTrue(report["finite_bits"])

    def test_nonfinite_float_bits_are_reported(self):
        raw = _blank_snapshot()
        word = 0x7FC00001
        _set_edge_bits(raw, 0, 1, 0, word)
        _set_edge_bits(raw, 1, 0, 0, word ^ cs.FLOAT_SIGN_BIT)
        report = cs.audit(raw, limit=1)
        self.assertEqual(report["nonfinite_words"], 2)
        self.assertFalse(report["finite_bits"])
        self.assertTrue(report["mirror_antisymmetry"])
        self.assertFalse(report["ok"])

    def test_audit_limit_is_bounded_and_reported(self):
        snapshot = cs.parse_snapshot(_blank_snapshot())
        report = cs.audit(snapshot, limit=7)
        self.assertEqual(report["pairs_checked"], 7)
        self.assertEqual(report["pair_count"], 32640)
        self.assertFalse(report["all_pairs_checked"])
        self.assertTrue(report["size_ok"])
        self.assertEqual(report["file_bytes"], cs.SNAPSHOT_BYTES)
        self.assertEqual(report["root_bytes"], cs.ROOT_BYTES)
        for bad in (-1, cs.PAIR_COUNT + 1, True, "2"):
            with self.assertRaises(cs.SnapshotRangeError):
                cs.audit(snapshot, limit=bad)


class PathSecurityTests(unittest.TestCase):
    def test_traversal_absolute_and_arbitrary_names_are_denied(self):
        denied = (
            "../root_17.bin",
            "../../etc/passwd",
            "/etc/passwd",
            "/tmp/root_17.bin",
            "root_17.bin/../../root_17.bin",
            "subdir/root_17.bin",
            "evil.bin",
            "root_99.bin",
            "root_17.bin\x00",
            b"root_17.bin",
        )
        for source in denied:
            with self.assertRaises(cs.SnapshotPathError):
                cs.load_snapshot(source)
        for seed in (True, 0, 99):
            with self.assertRaises(cs.SnapshotPathError):
                cs.load_snapshot(seed)

    @unittest.skipUnless(cs.SHA256SUMS_PATH.exists(),
                         "supplied Round 030 roots are not present")
    def test_symlink_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "root_17.bin"
            os.symlink(cs.ROOT_DIR / "root_17.bin", link)
            with mock.patch.object(cs, "ROOT_DIR", Path(directory)), \
                    mock.patch.object(cs, "SHA256SUMS_PATH",
                                      cs.SHA256SUMS_PATH):
                with self.assertRaises(cs.SnapshotPathError):
                    cs.load_snapshot("root_17.bin")

    @unittest.skipUnless(cs.SHA256SUMS_PATH.exists(),
                         "supplied Round 030 roots are not present")
    def test_missing_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(cs, "ROOT_DIR", Path(directory)), \
                    mock.patch.object(cs, "SHA256SUMS_PATH",
                                      cs.SHA256SUMS_PATH):
                with self.assertRaises(cs.SnapshotPathError):
                    cs.load_snapshot("root_17.bin")


class SuppliedRootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not cs.SHA256SUMS_PATH.exists():
            raise unittest.SkipTest("supplied Round 030 roots are not present")
        cls.digests = dict(cs.supplied_root_digests())
        cls.snapshots = {
            name: cs.load_snapshot(name) for name in cls.digests
        }

    def test_every_supplied_root_matches_sha256sums_and_size(self):
        self.assertTrue(self.snapshots)
        for name, snapshot in self.snapshots.items():
            self.assertEqual(snapshot.file_bytes, cs.SNAPSHOT_BYTES)
            self.assertEqual(snapshot.root_bytes, cs.ROOT_BYTES)
            self.assertEqual(snapshot.marker, cs.MARKER)
            self.assertEqual(snapshot.sha256, self.digests[name])
            self.assertEqual(snapshot.source, name)

    def test_real_heads_pairs_and_last_pair(self):
        for snapshot in self.snapshots.values():
            self.assertEqual(len(tuple(snapshot.iter_heads(limit=3))), 3)
            self.assertEqual(len(snapshot.head_bytes(255)), 128)
            last = snapshot.pair_cell(cs.PAIR_COUNT - 1)
            self.assertEqual(last.canonical, (254, 255))
            self.assertEqual(last.index, 32639)
            self.assertEqual(len(last.forward_words), 32)
            self.assertEqual(len(last.reverse_words), 32)

    def test_real_mirror_antisymmetry_on_bounded_samples(self):
        for snapshot in self.snapshots.values():
            for index in (0, 1, 733, 16320, 32639):
                cell = snapshot.pair_cell(index)
                for channel in range(cs.DIRECTION_WORDS):
                    forward = cs._float_from_bits(
                        cell.forward_words[channel])
                    reverse = cs._float_from_bits(
                        cell.reverse_words[channel])
                    self.assertEqual(forward, -reverse)

    def test_real_flux_matches_direct_float_readout(self):
        for snapshot in self.snapshots.values():
            for dimension in (0, 1, 7, 23):
                first = 1 + 2 * dimension
                for channel in (0, 5, 15):
                    expected = _flux(snapshot, first, first + 1, channel)
                    self.assertEqual(
                        cs.snapshot_flux(snapshot, dimension, channel),
                        expected)
                    self.assertEqual(
                        cs.snapshot_cycle(snapshot, first, first + 1,
                                          channel),
                        expected)

    def test_real_raw_lane_matches_direct_struct_read(self):
        for snapshot in self.snapshots.values():
            for i, j in ((0, 1), (1, 0), (17, 200), (255, 3)):
                identity = cs.pair_identity(i, j)
                offset = (cs.PAIR_REGION_OFFSET
                          + identity.index * cs.PAIR_BYTES)
                direct = struct.unpack_from(
                    "<64I", snapshot.root, offset)
                expected = direct[:32] if i < j else direct[32:]
                self.assertEqual(
                    cs.snapshot_raw_lanes(snapshot, i, j), expected)

    def test_real_audit_on_bounded_sample(self):
        for snapshot in self.snapshots.values():
            report = cs.audit(snapshot, limit=64)
            self.assertTrue(report, report)
            self.assertTrue(report["size_ok"])
            self.assertTrue(report["marker_ok"])
            self.assertTrue(report["finite_bits"])
            self.assertTrue(report["mirror_antisymmetry"])
            self.assertFalse(report["float_identity_claim"])

    def test_real_seed_filename_and_exact_path_forms(self):
        names = cs.supplied_root_names()
        self.assertIn("root_17.bin", names)
        snapshot = cs.load_snapshot(17)
        self.assertEqual(snapshot.source, "root_17.bin")
        path = cs.ROOT_DIR / "root_17.bin"
        self.assertEqual(cs.load_snapshot(path).sha256, snapshot.sha256)
        self.assertEqual(cs.load_snapshot(str(path)).sha256, snapshot.sha256)

    def test_sha_mismatch_is_rejected(self):
        wrong = MappingProxyType({"root_17.bin": "0" * 64})
        with mock.patch.object(cs, "_sha256sums", return_value=wrong):
            with self.assertRaises(cs.SnapshotIntegrityError):
                cs.load_snapshot("root_17.bin")

    def test_verify_supplied_roots_report(self):
        report = cs.verify_supplied_roots(["root_17.bin"])
        self.assertEqual(set(report), {"root_17.bin"})
        self.assertTrue(report["root_17.bin"]["matches"])
        self.assertEqual(
            report["root_17.bin"]["sha256"], self.digests["root_17.bin"])


if __name__ == "__main__":
    unittest.main()
