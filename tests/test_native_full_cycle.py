"""Native C++ full-cycle root tests.

These tests compile ``experiments/round031/native_full_cycle.cpp`` with a C++20
compiler and call the exported C ABI through :mod:`ctypes`.  They verify the
complete 32,385-cycle root-star basis, the physical 8 MiB layout, exhaustive
cycle-id mapping, mirror identity, root-edge zeros, gauge invariance, exact
snapshots and small synthetic prediction/training runs.

The tests skip cleanly when no C++ compiler is available.  They do not modify
the supplied Round 031 experiment and make no language-model performance claim.
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "round031" / "native_full_cycle.cpp"

HEADS = 256
CLASSES = 16
FEATURES = 16
FULL_CYCLES = 32_385
PAIR_CELLS = 32_640
ROOT_BYTES = 8_388_608
HEAD_BYTES = HEADS * 128
SNAPSHOT_MAGIC = b"NANFULL1"
SNAPSHOT_BYTES = ROOT_BYTES + len(SNAPSHOT_MAGIC)


def find_compiler() -> str | None:
    for name in ("c++", "g++", "clang++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def cycle_id(i: int, j: int) -> int:
    """Python reference for the C++ lexicographic root-star rank."""
    if i > j:
        i, j = j, i
    if not (1 <= i < j <= HEADS - 1):
        raise ValueError((i, j))
    a = i - 1
    b = j - 1
    return a * (509 - a) // 2 + (b - a - 1)


def cell_index(i: int, j: int) -> int:
    if i > j:
        i, j = j, i
    return i * (2 * HEADS - 1 - i) // 2 + (j - i - 1)


def cycle_pairs():
    for i in range(1, HEADS - 1):
        for j in range(i + 1, HEADS):
            yield i, j


class NativeFullCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = find_compiler()
        if compiler is None:
            raise unittest.SkipTest(
                "no C++ compiler (c++, g++ or clang++) is available")
        if not SOURCE.is_file():
            raise RuntimeError("missing native source: %s" % SOURCE)

        cls.tmpdir = tempfile.TemporaryDirectory(prefix="nan_full_cycle_")
        cls.library_path = Path(cls.tmpdir.name) / "libnative_full_cycle.so"
        command = [
            compiler,
            "-std=c++20",
            "-O3",
            "-shared",
            "-fPIC",
            str(SOURCE),
            "-o",
            str(cls.library_path),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180)
        if completed.returncode != 0:
            raise RuntimeError(
                "native full-cycle build failed:\n%s\n%s"
                % (completed.stdout, completed.stderr))
        cls.lib = ctypes.CDLL(str(cls.library_path))
        cls._bind()

    @classmethod
    def _bind(cls):
        lib = cls.lib
        c_void_p = ctypes.c_void_p
        c_int = ctypes.c_int
        c_float = ctypes.c_float
        c_double = ctypes.c_double
        c_uint = ctypes.c_uint
        float_p = ctypes.POINTER(c_float)
        int_p = ctypes.POINTER(c_int)

        lib.nan_full_new.argtypes = [c_int]
        lib.nan_full_new.restype = c_void_p
        lib.nan_full_free.argtypes = [c_void_p]
        lib.nan_full_free.restype = None
        lib.nan_full_count.argtypes = [c_void_p]
        lib.nan_full_count.restype = c_int
        lib.nan_full_write.argtypes = [c_void_p, c_int, c_int, c_float]
        lib.nan_full_write.restype = c_int
        lib.nan_full_read.argtypes = [c_void_p, c_int, c_int]
        lib.nan_full_read.restype = c_float
        lib.nan_full_read_all.argtypes = [c_void_p, float_p, c_int]
        lib.nan_full_read_all.restype = c_int
        lib.nan_full_predict.argtypes = [c_void_p, float_p, float_p]
        lib.nan_full_predict.restype = c_int
        lib.nan_full_train_epoch.argtypes = [
            c_void_p, float_p, int_p, c_int, c_float, c_uint]
        lib.nan_full_train_epoch.restype = c_double
        lib.nan_full_gauge.argtypes = [c_void_p, c_uint, c_float]
        lib.nan_full_gauge.restype = c_int
        lib.nan_full_audit.argtypes = [c_void_p]
        lib.nan_full_audit.restype = c_int
        lib.nan_full_snapshot.argtypes = [c_void_p, ctypes.c_char_p]
        lib.nan_full_snapshot.restype = c_int
        lib.nan_full_restore.argtypes = [c_void_p, ctypes.c_char_p]
        lib.nan_full_restore.restype = c_int
        lib.nan_full_cycle_id.argtypes = [c_int, c_int]
        lib.nan_full_cycle_id.restype = c_int
        lib.nan_full_cycle_pair.argtypes = [c_int, int_p, int_p]
        lib.nan_full_cycle_pair.restype = c_int
        lib.nan_full_bytes.argtypes = [c_void_p]
        lib.nan_full_bytes.restype = c_int
        lib.nan_full_channels.argtypes = [c_void_p]
        lib.nan_full_channels.restype = c_int
        lib.nan_full_feature_dim.argtypes = [c_void_p]
        lib.nan_full_feature_dim.restype = c_int

    @classmethod
    def tearDownClass(cls):
        cls.lib = None
        cls.tmpdir.cleanup()

    # -- helpers ---------------------------------------------------------
    def new(self, mode: int = 0) -> int:
        handle = self.lib.nan_full_new(mode)
        self.assertIsNotNone(handle)
        self.addCleanup(self.lib.nan_full_free, handle)
        return handle

    def snapshot(self, handle: int, name: str) -> tuple[Path, bytes]:
        path = Path(self.tmpdir.name) / (
            "%s_%s.bin" % (self._testMethodName, name))
        self.assertEqual(
            self.lib.nan_full_snapshot(handle, os.fsencode(path)), 1)
        return path, path.read_bytes()

    def read_all(self, handle: int) -> bytes:
        values = (ctypes.c_float * (FULL_CYCLES * CLASSES))()
        self.assertEqual(
            self.lib.nan_full_read_all(handle, values, len(values)), 1)
        return memoryview(values).tobytes()

    # -- tests -----------------------------------------------------------
    def test_full_count_is_32385_and_256_mode_is_refused(self):
        handle = self.new(0)
        self.assertEqual(self.lib.nan_full_count(handle), FULL_CYCLES)
        self.assertEqual(self.lib.nan_full_count(handle), 32_385)
        self.assertNotEqual(self.lib.nan_full_count(handle), 256)
        self.assertEqual(self.lib.nan_full_bytes(handle), ROOT_BYTES)
        self.assertEqual(self.lib.nan_full_channels(handle), CLASSES)
        self.assertEqual(self.lib.nan_full_feature_dim(handle), FEATURES)

        full_handle = self.new(FULL_CYCLES)
        self.assertEqual(self.lib.nan_full_count(full_handle), FULL_CYCLES)

        # A 256-only / truncated mode must never allocate a root.
        self.assertIsNone(self.lib.nan_full_new(256))
        self.assertIsNone(self.lib.nan_full_new(1))
        self.assertIsNone(self.lib.nan_full_new(-1))

    def test_cycle_id_mapping_is_exhaustive_and_invertible(self):
        handle = self.new(0)
        seen = bytearray(FULL_CYCLES)
        for i, j in cycle_pairs():
            expected = cycle_id(i, j)
            self.assertEqual(
                self.lib.nan_full_cycle_id(i, j), expected,
                "cycle id mismatch for pair (%d, %d)" % (i, j))
            self.assertEqual(seen[expected], 0, "duplicate cycle id %d" % expected)
            seen[expected] = 1

            left = ctypes.c_int(-1)
            right = ctypes.c_int(-1)
            self.assertEqual(
                self.lib.nan_full_cycle_pair(
                    expected, ctypes.byref(left), ctypes.byref(right)), 1)
            self.assertEqual((left.value, right.value), (i, j))

        self.assertEqual(sum(seen), FULL_CYCLES)
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

        # Root edges, repeated heads and out-of-range ids are not cycles.
        self.assertEqual(self.lib.nan_full_cycle_id(0, 1), -1)
        self.assertEqual(self.lib.nan_full_cycle_id(1, 1), -1)
        self.assertEqual(self.lib.nan_full_cycle_id(0, 255), -1)
        self.assertEqual(self.lib.nan_full_cycle_id(1, 256), -1)
        left = ctypes.c_int(0)
        right = ctypes.c_int(0)
        self.assertEqual(
            self.lib.nan_full_cycle_pair(
                FULL_CYCLES, ctypes.byref(left), ctypes.byref(right)), 0)
        self.assertEqual(
            self.lib.nan_full_cycle_pair(
                -1, ctypes.byref(left), ctypes.byref(right)), 0)

    def test_physical_layout_mapping_mirror_and_root_zeros(self):
        handle = self.new(0)
        for cid in range(FULL_CYCLES):
            self.assertEqual(
                self.lib.nan_full_write(handle, cid, 0, float(cid + 1)), 1)

        _, data = self.snapshot(handle, "layout")
        self.assertEqual(len(data), SNAPSHOT_BYTES)
        self.assertEqual(data[:8], SNAPSHOT_MAGIC)
        self.assertEqual(data[8:8 + HEAD_BYTES], b"\x00" * HEAD_BYTES)

        # Read the raw physical words rather than the C accessors.  This
        # independently checks every pair -> cycle-id -> cell mapping.
        for i, j in cycle_pairs():
            cid = cycle_id(i, j)
            offset = 8 + HEAD_BYTES + cell_index(i, j) * 256
            expected = float(cid + 1)
            coefficient = struct.unpack_from("<f", data, offset)[0]
            mirror = struct.unpack_from("<f", data, offset + 128)[0]
            lane = struct.unpack_from("<f", data, offset + 64)[0]
            lane_mirror = struct.unpack_from("<f", data, offset + 192)[0]
            self.assertEqual(coefficient, expected)
            self.assertEqual(mirror, -expected)
            self.assertEqual(lane, expected)  # zero gauge initially
            self.assertEqual(lane_mirror, -expected)

        # The 255 root pair cells are physically zero in the star basis.
        for j in range(1, HEADS):
            offset = 8 + HEAD_BYTES + cell_index(0, j) * 256
            words = struct.unpack_from("<64f", data, offset)
            self.assertTrue(
                all(word == 0.0 for word in words),
                "non-zero root edge in cell (0, %d)" % j)

        self.assertEqual(self.lib.nan_full_audit(handle), 1)

    def test_all_32385_coefficients_and_16_channels_are_addressable(self):
        handle = self.new(0)
        expected = [0.0] * (FULL_CYCLES * CLASSES)
        for cid in range(FULL_CYCLES):
            channel = cid % CLASSES
            value = (cid % 251) * 0.125 - 4.0
            self.assertEqual(
                self.lib.nan_full_write(handle, cid, channel, value), 1)
            expected[cid * CLASSES + channel] = value

        actual = self.read_all(handle)
        self.assertEqual(
            list(struct.unpack("<%df" % len(expected), actual)), expected)

        for cid in (0, 1, 253, 254, 10_000, 32_384):
            channel = cid % CLASSES
            self.assertEqual(
                self.lib.nan_full_read(handle, cid, channel),
                expected[cid * CLASSES + channel])
            other = (channel + 1) % CLASSES
            self.assertEqual(self.lib.nan_full_read(handle, cid, other), 0.0)

        # Invalid ids, channels and lengths are refused, not truncated.
        self.assertEqual(self.lib.nan_full_write(handle, -1, 0, 1.0), 0)
        self.assertEqual(
            self.lib.nan_full_write(handle, FULL_CYCLES, 0, 1.0), 0)
        self.assertEqual(self.lib.nan_full_write(handle, 0, -1, 1.0), 0)
        self.assertEqual(self.lib.nan_full_write(handle, 0, CLASSES, 1.0), 0)
        self.assertTrue(math.isnan(self.lib.nan_full_read(handle, -1, 0)))
        self.assertTrue(
            math.isnan(self.lib.nan_full_read(handle, FULL_CYCLES, 0)))
        self.assertTrue(math.isnan(self.lib.nan_full_read(handle, 0, -1)))
        small = (ctypes.c_float * 16)()
        self.assertEqual(self.lib.nan_full_read_all(handle, small, 16), 0)
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

    def test_gauge_preserves_all_root_star_coefficients(self):
        handle = self.new(0)
        for cid in (0, 1, 2, 253, 254, 1_000, 20_000, FULL_CYCLES - 1):
            for channel in (0, 1, 7, 15):
                self.assertEqual(
                    self.lib.nan_full_write(
                        handle, cid, channel,
                        math.sin((cid + 1) * (channel + 1)) * 0.5), 1)

        before = self.read_all(handle)
        _, snapshot_before = self.snapshot(handle, "before_gauge")
        self.assertEqual(self.lib.nan_full_gauge(handle, 12_345, 0.75), 1)
        after = self.read_all(handle)

        # Every coefficient word, including zeros and signs, is unchanged.
        self.assertEqual(after, before)
        for cid in (0, 253, 20_000, FULL_CYCLES - 1):
            for channel in (0, 15):
                self.assertEqual(
                    self.lib.nan_full_read(handle, cid, channel),
                    struct.unpack_from(
                        "<f", before, (cid * CLASSES + channel) * 4)[0])

        # The gauge is real: gauge coordinates/raw lanes moved even though the
        # cycle coefficients did not.
        _, snapshot_after = self.snapshot(handle, "after_gauge")
        self.assertNotEqual(snapshot_before, snapshot_after)
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

        # A zero-sigma gauge is also accepted and changes nothing.
        _, zero_before = self.snapshot(handle, "zero_before")
        self.assertEqual(self.lib.nan_full_gauge(handle, 99, 0.0), 1)
        _, zero_after = self.snapshot(handle, "zero_after")
        self.assertEqual(zero_before, zero_after)
        self.assertEqual(self.read_all(handle), before)

    def test_snapshot_and_restore_are_bit_exact(self):
        handle = self.new(0)
        for cid in range(0, FULL_CYCLES, 97):
            self.assertEqual(
                self.lib.nan_full_write(handle, cid, cid % CLASSES,
                                        (cid % 37) * 0.25 - 3.0), 1)
        self.assertEqual(self.lib.nan_full_gauge(handle, 7, 0.5), 1)

        baseline = self.read_all(handle)
        path, snapshot_bytes = self.snapshot(handle, "exact")
        self.assertEqual(len(snapshot_bytes), SNAPSHOT_BYTES)
        self.assertEqual(snapshot_bytes[:8], SNAPSHOT_MAGIC)

        self.assertEqual(self.lib.nan_full_write(handle, 0, 0, 99.0), 1)
        self.assertEqual(self.lib.nan_full_gauge(handle, 8, 1.25), 1)
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

        self.assertEqual(
            self.lib.nan_full_restore(handle, os.fsencode(path)), 1)
        self.assertEqual(self.read_all(handle), baseline)
        _, restored_bytes = self.snapshot(handle, "restored")
        self.assertEqual(restored_bytes, snapshot_bytes)

        # Corrupt and truncated files are refused without partial overwrite.
        bad_magic = Path(self.tmpdir.name) / "bad_magic.bin"
        bad_magic.write_bytes(b"BADMAGIC" + b"\x00" * ROOT_BYTES)
        self.assertEqual(
            self.lib.nan_full_restore(handle, os.fsencode(bad_magic)), 0)
        self.assertEqual(self.read_all(handle), baseline)

        truncated = Path(self.tmpdir.name) / "truncated.bin"
        truncated.write_bytes(SNAPSHOT_MAGIC + b"\x00" * 17)
        self.assertEqual(
            self.lib.nan_full_restore(handle, os.fsencode(truncated)), 0)
        self.assertEqual(self.read_all(handle), baseline)
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

    def test_predict_and_train_epoch_use_the_full_basis(self):
        handle = self.new(0)
        rng = random.Random(7)
        sample_count = 12
        samples = (ctypes.c_float * (sample_count * FEATURES))()
        labels = (ctypes.c_int * sample_count)()
        for t in range(sample_count):
            x = [rng.uniform(-1.0, 1.0) for _ in range(FEATURES)]
            for d, value in enumerate(x):
                samples[t * FEATURES + d] = value
            labels[t] = max(
                range(4), key=lambda c: sum(x[c * 4 + d] for d in range(4)))

        output = (ctypes.c_float * CLASSES)()
        self.assertEqual(self.lib.nan_full_predict(handle, samples, output), 1)
        self.assertTrue(all(math.isfinite(v) for v in output))

        first = self.lib.nan_full_train_epoch(
            handle, samples, labels, sample_count, 0.2, 1)
        self.assertTrue(math.isfinite(first))
        losses = [first]
        for epoch in range(1, 8):
            loss = self.lib.nan_full_train_epoch(
                handle, samples, labels, sample_count, 0.2, epoch + 1)
            self.assertTrue(math.isfinite(loss))
            losses.append(loss)
        self.assertLess(losses[-1], first)
        self.assertLess(losses[-1], math.log(CLASSES))

        # Training is not a 16x16 window: it writes coefficient words across
        # the whole 32,385-cycle table.
        _, data = self.snapshot(handle, "trained")
        body = memoryview(data)[8 + HEAD_BYTES:].cast("f")
        channel_zero = body[0::64]
        nonzero = sum(1 for value in channel_zero[HEADS - 1:] if value != 0.0)
        self.assertGreater(nonzero, 30_000)

        self.assertEqual(self.lib.nan_full_predict(handle, samples, output), 1)
        self.assertTrue(all(math.isfinite(v) for v in output))
        self.assertEqual(self.lib.nan_full_audit(handle), 1)

        # Bad labels are refused by returning NaN, not by truncating memory.
        bad_labels = (ctypes.c_int * 1)(CLASSES)
        one_sample = (ctypes.c_float * FEATURES)()
        self.assertTrue(math.isnan(self.lib.nan_full_train_epoch(
            handle, one_sample, bad_labels, 1, 0.1, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
