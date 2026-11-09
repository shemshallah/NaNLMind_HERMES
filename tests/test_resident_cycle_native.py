"""Tests for the optional Round 030 native-cycle ctypes bridge.

The suite separates pure Python checks (unavailability receipts, digest
denial) from real native checks.  Native tests compile the supplied C++ source
once and are skipped when no C++ compiler is installed; a skipped native test
never fabricates a prediction, weight or audit result.
"""
from __future__ import annotations

import hashlib
import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes.resident import cycle_native as cn

SOURCE = cn.default_source_path()
MANIFEST = SOURCE.parent / "SHA256SUMS"
COMPILER = cn.find_compiler()

SAMPLE = [0.01 * index for index in range(cn.D)]
TRAIN_SAMPLES = [
    [((row * 7 + column * 3) % 17) / 17.0 for column in range(cn.D)]
    for row in range(4)
]
TRAIN_LABELS = [0, 1, 2, 3]


class UnavailabilityTests(unittest.TestCase):
    """Unavailable is an explicit receipt, never a simulated native result."""

    def test_missing_compiler_is_plain_unavailable(self):
        receipt = cn.probe_native_cycle(compiler_candidates=())

        self.assertEqual(receipt["schema"], cn.SCHEMA)
        self.assertEqual(receipt["status"], "unavailable")
        self.assertFalse(receipt["available"])
        self.assertIn("no C++ compiler", receipt["reason"])
        self.assertEqual(receipt["root_bytes"], 8_388_608)
        self.assertFalse(receipt["claims"]["canonical"])
        self.assertFalse(receipt["claims"]["raw_nan_payload_arithmetic"])

        with self.assertRaises(cn.NativeCycleUnavailable) as caught:
            cn.NativeCycleRoot(
                compiler_candidates=(),
                cache_dir=Path(tempfile.gettempdir()) / "unused-cycle-cache",
            )
        self.assertEqual(caught.exception.receipt["status"], "unavailable")
        self.assertFalse(caught.exception.receipt["available"])
        self.assertIn("no C++ compiler", caught.exception.reason)

    def test_missing_library_is_plain_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_native_cycle.so"
            receipt = cn.probe_native_cycle(missing)
            self.assertEqual(receipt["status"], "unavailable")
            self.assertFalse(receipt["available"])
            self.assertIn("not found", receipt["reason"])

            with self.assertRaises(cn.NativeCycleUnavailable):
                cn.NativeCycleRoot(missing)

    def test_missing_source_is_plain_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "native_cycle.cpp"
            receipt = cn.probe_native_cycle(missing)
            self.assertEqual(receipt["status"], "unavailable")
            self.assertFalse(receipt["available"])
            self.assertIn("source not found", receipt["reason"])

    def test_probe_reports_a_compiler_without_compiling(self):
        receipt = cn.probe_native_cycle()
        if COMPILER is None:
            self.assertEqual(receipt["status"], "unavailable")
        else:
            self.assertEqual(receipt["status"], "available")
            self.assertTrue(receipt["available"])
            self.assertTrue(Path(receipt["compiler"]).is_file())
            self.assertEqual(
                receipt["source_sha256"],
                hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            )


class VerificationTests(unittest.TestCase):
    """Source/library digests are checked before native code is executed."""

    def test_source_digest_mismatch_is_denied_before_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "native_cycle.cpp"
            source.write_bytes(SOURCE.read_bytes())
            wrong = hashlib.sha256(b"not the supplied source").hexdigest()
            manifest = base / "SHA256SUMS"
            manifest.write_text(
                f"{wrong}  native_cycle.cpp\n", encoding="utf-8")

            with mock.patch.object(
                cn.subprocess,
                "run",
                side_effect=AssertionError("compiler must not run"),
            ):
                with self.assertRaises(cn.NativeCycleVerificationError) as caught:
                    cn.compile_native_cycle(
                        source,
                        manifest=manifest,
                        cache_dir=base / "cache",
                    )
            self.assertIn("SHA-256 mismatch", str(caught.exception))

    def test_manifest_traversal_and_missing_source_entry_are_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "native_cycle.cpp"
            source.write_bytes(SOURCE.read_bytes())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = base / "SHA256SUMS"

            manifest.write_text(f"{digest}  ../outside.cpp\n", encoding="utf-8")
            with self.assertRaises(cn.NativeCycleVerificationError) as caught:
                cn.verify_native_cycle_source(source, manifest)
            self.assertIn("traversal", str(caught.exception).lower())

            manifest.write_text(
                f"{digest}  some_other_file.cpp\n", encoding="utf-8")
            with self.assertRaises(cn.NativeCycleVerificationError) as caught:
                cn.verify_native_cycle_source(source, manifest)
            self.assertIn("not covered", str(caught.exception))

    def test_library_digest_mismatch_is_denied_before_dlopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "fake_native_cycle.so"
            library.write_bytes(b"not actually a shared library")
            with mock.patch.object(
                cn.ctypes,
                "CDLL",
                side_effect=AssertionError("library must not be loaded"),
            ):
                with self.assertRaises(cn.NativeCycleVerificationError) as caught:
                    cn.NativeCycleRoot(library, expected_sha256="0" * 64)
            self.assertIn("SHA-256 mismatch", str(caught.exception))


@unittest.skipUnless(COMPILER is not None, "no C++ compiler available")
class NativeCycleTests(unittest.TestCase):
    """Real native checks; skipped without a working C++ compiler."""

    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory(prefix="cycle-native-test-")
        cls.cache_dir = Path(cls._temp.name) / "cache"
        cls.build = cn.compile_native_cycle(
            cache_dir=cls.cache_dir,
            timeout=180.0,
        )

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def setUp(self):
        self.root = cn.NativeCycleRoot(self.build.library)
        self.addCleanup(self.root.close)

    # -- build/cache -------------------------------------------------------

    def test_verified_build_and_cache_reuse(self):
        self.assertEqual(
            self.build.source_sha256,
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.build.source.resolve(), SOURCE.resolve())
        self.assertEqual(
            self.build.library_sha256,
            hashlib.sha256(self.build.library.read_bytes()).hexdigest(),
        )
        self.assertFalse(self.build.cached)
        self.assertEqual(self.build.verification["status"], "verified")

        again = cn.compile_native_cycle(cache_dir=self.cache_dir, timeout=180.0)
        self.assertTrue(again.cached)
        self.assertEqual(again.library, self.build.library)
        self.assertEqual(again.library_sha256, self.build.library_sha256)

    def test_cached_library_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / self.build.library.name
            sidecar = Path(tmp) / (self.build.library.name + ".sha256")
            shutil.copy2(self.build.library, library)
            shutil.copy2(
                self.build.library.with_name(
                    self.build.library.name + ".sha256"),
                sidecar,
            )
            with library.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(cn.NativeCycleVerificationError) as caught:
                cn.NativeCycleRoot(library)
            self.assertIn("SHA-256 mismatch", str(caught.exception))

    # -- predict / audit / gauge ------------------------------------------

    def test_predict_audit_and_gauge_invariants(self):
        weights = self.root.weights()
        self.assertEqual(len(weights), cn.K)
        self.assertTrue(all(len(row) == cn.D for row in weights))
        self.assertEqual(max(abs(value) for row in weights for value in row), 0.0)

        prediction = self.root.predict(SAMPLE)
        raw = self.root.raw_predict(SAMPLE)
        self.assertEqual(len(prediction), cn.K)
        self.assertEqual(len(raw), cn.K)
        self.assertTrue(all(math.isfinite(value) for value in prediction))
        self.assertTrue(all(value == 0.0 for value in prediction))
        self.assertTrue(all(value == 0.0 for value in raw))
        self.assertTrue(self.root.audit())

        batch = self.root.predict_batch([SAMPLE, SAMPLE])
        self.assertEqual(batch, [prediction, prediction])
        auto_batch = self.root.predict([SAMPLE, SAMPLE])
        self.assertEqual(auto_batch, [prediction, prediction])

        self.root.gauge(17, 1.5)
        shifted = self.root.weights()
        drift = max(
            abs(shifted[class_index][dimension] -
                weights[class_index][dimension])
            for class_index in range(cn.K) for dimension in range(cn.D)
        )
        self.assertLess(drift, 1e-5)
        shifted_prediction = self.root.predict(SAMPLE)
        self.assertLess(
            max(abs(a - b) for a, b in zip(shifted_prediction, prediction)),
            1e-4,
        )
        self.assertTrue(self.root.audit())

    def test_float32_boundary_and_nonfinite_rejection(self):
        # Tiny double values collapse to +0.0 only because the documented
        # boundary is binary32; outputs are widened exactly back to float.
        tiny = self.root.predict([1e-50] * cn.D)
        self.assertEqual(tiny, [0.0] * cn.K)
        for value in tiny:
            self.assertEqual(
                struct.unpack("<f", struct.pack("<f", value))[0], value)

        for bad in (math.nan, math.inf, -math.inf, 1e39):
            with self.subTest(bad=bad):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.predict([bad] * cn.D)

        self.root.train_epoch(TRAIN_SAMPLES, TRAIN_LABELS, lr=0.05, seed=3)
        trained = self.root.predict(SAMPLE)
        for value in trained:
            self.assertTrue(math.isfinite(value))
            self.assertEqual(
                struct.unpack("<f", struct.pack("<f", value))[0], value)

    # -- training ----------------------------------------------------------

    def test_train_epoch_moves_weights_and_train_is_bounded(self):
        before = self.root.weights()
        loss = self.root.train_epoch(
            TRAIN_SAMPLES, TRAIN_LABELS, lr=0.05, seed=5)
        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))
        after = self.root.weights()
        delta = max(
            abs(after[class_index][dimension] -
                before[class_index][dimension])
            for class_index in range(cn.K) for dimension in range(cn.D)
        )
        self.assertGreater(delta, 0.0)
        self.assertTrue(self.root.audit())

        losses = self.root.train(
            TRAIN_SAMPLES, TRAIN_LABELS, epochs=3, lr=0.02, seed=6)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertTrue(self.root.audit())

    # -- snapshots ---------------------------------------------------------

    def test_snapshot_is_exact_and_restore_is_bitwise(self):
        self.root.train_epoch(TRAIN_SAMPLES, TRAIN_LABELS, lr=0.05, seed=9)
        weights = self.root.weights()
        prediction = self.root.predict(SAMPLE)
        snapshot = Path(self._temp.name) / "root_exact.bin"

        self.assertTrue(self.root.snapshot(snapshot))
        self.assertEqual(snapshot.stat().st_size, cn.SNAPSHOT_BYTES)
        self.assertEqual(snapshot.stat().st_size, 8_388_616)
        with snapshot.open("rb") as handle:
            self.assertEqual(handle.read(8), b"NANCYC30")

        self.root.gauge(18, 2.0)
        self.assertNotEqual(self.root.weights(), weights)
        self.assertTrue(self.root.restore(snapshot))
        self.assertEqual(self.root.weights(), weights)
        self.assertEqual(self.root.predict(SAMPLE), prediction)
        self.assertTrue(self.root.audit())

    def test_snapshot_restore_reject_bad_size_marker_and_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            truncated = base / "truncated.bin"
            truncated.write_bytes(cn.SNAPSHOT_MAGIC + b"\x00" * 100)
            with self.assertRaises(cn.NativeCycleSnapshotError):
                self.root.restore(truncated)

            bad_marker = base / "bad_marker.bin"
            bad_marker.write_bytes(b"XXXXXXXX" + b"\x00" * cn.ROOT_BYTES)
            with self.assertRaises(cn.NativeCycleSnapshotError) as caught:
                self.root.restore(bad_marker)
            self.assertIn("marker", str(caught.exception))

            trailing = base / "trailing.bin"
            self.assertTrue(self.root.snapshot(trailing))
            with trailing.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(cn.NativeCycleSnapshotError) as caught:
                self.root.restore(trailing)
            self.assertIn("exactly", str(caught.exception))

            with self.assertRaises(cn.NativeCycleSnapshotError):
                self.root.restore(base / "does_not_exist.bin")

    # -- bounds / lifecycle ------------------------------------------------

    def test_dimension_class_range_and_path_bounds(self):
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.predict([0.0] * (cn.D - 1))
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.predict([0.0] * (cn.D + 1))
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.predict([])
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.predict([[0.0] * cn.D] * (cn.MAX_SAMPLES + 1))

        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.train_epoch(TRAIN_SAMPLES, TRAIN_LABELS[:-1], lr=0.05)
        for label in (-1, cn.K, True, 1.5):
            with self.subTest(label=label):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.train_epoch([SAMPLE], [label], lr=0.05)
        for lr in (0.0, -0.1, 2.0, math.nan, math.inf):
            with self.subTest(lr=lr):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.train_epoch(TRAIN_SAMPLES, TRAIN_LABELS, lr=lr)
        for seed in (-1, 2 ** 32, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.train_epoch(
                        TRAIN_SAMPLES, TRAIN_LABELS, lr=0.05, seed=seed)
        for epochs in (0, -1, cn.MAX_EPOCHS + 1, True, 1.5):
            with self.subTest(epochs=epochs):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.train(
                        TRAIN_SAMPLES, TRAIN_LABELS, epochs=epochs)

        for sigma in (-1.0, cn.MAX_GAUGE_SIGMA + 1.0, math.nan, math.inf):
            with self.subTest(sigma=sigma):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.gauge(1, sigma)
        for seed in (-1, 2 ** 31, True):
            with self.subTest(seed=seed):
                with self.assertRaises(cn.NativeCycleBoundsError):
                    self.root.gauge(seed, 1.0)

        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.snapshot(Path(self._temp.name) / ("x" * cn.MAX_PATH_BYTES))
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.snapshot("\x00bad")
        with self.assertRaises(cn.NativeCycleBoundsError):
            self.root.snapshot(Path(self._temp.name) / "missing_dir" / "x.bin")

    def test_closed_root_refuses_use_and_receipts_expose_no_pointer(self):
        receipt = self.root.receipt()
        self.assertEqual(receipt["status"], "available")
        self.assertEqual(receipt["root_bytes"], 8_388_608)
        self.assertNotIn("pointer", receipt)
        self.assertNotIn("handle", receipt)
        self.assertFalse(receipt["claims"]["canonical_merge"])
        self.assertFalse(receipt["claims"]["raw_nan_payload_arithmetic"])
        self.assertFalse(receipt["claims"]["numpy_required"])
        self.assertFalse(receipt["claims"]["torch_required"])
        self.assertFalse(hasattr(self.root, "ptr"))

        self.root.close()
        self.assertTrue(self.root.closed)
        self.assertFalse(self.root.available)
        self.root.close()
        with self.assertRaises(cn.NativeCycleClosedError):
            self.root.predict(SAMPLE)
        with self.assertRaises(cn.NativeCycleClosedError):
            self.root.weights()
        with self.assertRaises(cn.NativeCycleClosedError):
            self.root.audit()


if __name__ == "__main__":
    unittest.main()
