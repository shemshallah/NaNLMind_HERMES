"""Tests for the Round 029 native/Python raw-geometry comparison script."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "round029_native_compare.py"

_spec = importlib.util.spec_from_file_location(
    "round029_native_compare_under_test", SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/round029_native_compare.py")
rc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rc
_spec.loader.exec_module(rc)


def _fake_python_measurement() -> dict:
    return {
        "implementation": "python-bytearray-float32",
        "root_bytes": rc.ROOT_BYTES,
        "mirror_pairs": rc.PAIRS,
        "max_gauge_error": 9.5e-7,
        "max_reverse_error": 0.0,
        "finite_difference_error": 7.3e-5,
        "loss_before_step": 0.5,
        "loss_after_step": 0.25,
        "reads": rc.READS,
        "read_milliseconds": 510.0,
        "mlock_status": None,
        "passed": True,
        "layout": {
            "heads": rc.N,
            "head_bytes": rc.HEAD_BYTES,
            "pair_bytes": rc.PAIR_BYTES,
            "pairs": rc.PAIRS,
            "root_bytes": rc.ROOT_BYTES,
            "alignment_ok": True,
        },
    }


class Round029NativeCompareTests(unittest.TestCase):
    def test_python_raw_payload_invariants_are_bounded(self):
        measurement = rc.python_raw_measurement()

        self.assertEqual(rc.N * rc.HEAD_BYTES + rc.PAIRS * rc.PAIR_BYTES,
                         rc.ROOT_BYTES)
        self.assertEqual(rc.ROOT_BYTES, 8_388_608)
        self.assertEqual(rc.PAIRS, 32_640)
        self.assertEqual(measurement["root_bytes"], 8_388_608)
        self.assertEqual(measurement["mirror_pairs"], 32_640)
        self.assertGreaterEqual(measurement["max_gauge_error"], 0.0)
        self.assertLess(measurement["max_gauge_error"], rc.GAUGE_BOUND)
        self.assertEqual(measurement["max_reverse_error"], 0.0)
        self.assertGreaterEqual(measurement["finite_difference_error"], 0.0)
        self.assertLess(
            measurement["finite_difference_error"],
            rc.FINITE_DIFFERENCE_BOUND,
        )
        self.assertLess(
            measurement["loss_after_step"],
            measurement["loss_before_step"],
        )
        self.assertEqual(measurement["reads"], rc.READS)
        self.assertGreaterEqual(measurement["read_milliseconds"], 0.0)
        self.assertTrue(measurement["passed"])
        self.assertTrue(measurement["layout"]["alignment_ok"])
        self.assertGreater(rc.DEFAULT_TIMEOUT_SECONDS, 0.0)

    def test_manifest_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "round029"
            base.mkdir()
            source = base / "native_mirror.cpp"
            source.write_text("int main(){return 0;}\n", encoding="utf-8")
            outside = Path(tmp) / "outside.cpp"
            outside.write_text("int main(){return 1;}\n", encoding="utf-8")
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            sums = base / "SHA256SUMS"
            sums.write_text(f"{digest}  ../{outside.name}\n", encoding="utf-8")

            with self.assertRaises(rc.VerificationError) as caught:
                rc.verify_supplied_files(source, sums)
            self.assertIn("traversal", str(caught.exception).lower())

            sums.write_text(f"{digest}  {outside}\n", encoding="utf-8")
            with self.assertRaises(rc.VerificationError) as caught:
                rc.verify_supplied_files(source, sums)
            self.assertIn("traversal", str(caught.exception).lower())

    def test_source_digest_mismatch_is_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "round029"
            base.mkdir()
            source = base / "native_mirror.cpp"
            source.write_text("int main(){return 0;}\n", encoding="utf-8")
            wrong_digest = hashlib.sha256(b"not the supplied source").hexdigest()
            sums = base / "SHA256SUMS"
            sums.write_text(
                f"{wrong_digest}  native_mirror.cpp\n", encoding="utf-8")
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"

            with mock.patch.object(
                rc, "compile_native",
                side_effect=AssertionError("must not compile after denial"),
            ):
                code = rc.main([
                    "--source", str(source),
                    "--sha256sums", str(sums),
                    "--out-json", str(out_json),
                    "--out-md", str(out_md),
                ])

            self.assertEqual(code, rc.EXIT_DENIED)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "denied")
            self.assertEqual(report["verification"]["status"], "denied")
            self.assertIn(
                "mismatch", report["verification"]["reason"].lower())
            self.assertEqual(report["native"]["status"], "not_run")
            self.assertIsNone(report["native"]["measurement"])
            self.assertFalse(report["claims"]["native_execution"])
            self.assertTrue(out_md.is_file())

    def test_missing_compiler_yields_structured_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"
            with mock.patch.object(rc.shutil, "which", return_value=None):
                code = rc.main([
                    "--source", str(rc.DEFAULT_SOURCE),
                    "--out-json", str(out_json),
                    "--out-md", str(out_md),
                ])

            self.assertEqual(code, rc.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "python-only")
            self.assertEqual(report["mode"], "python-only")
            self.assertEqual(report["native"]["status"], "skipped")
            self.assertIn("compiler", report["native"]["reason"].lower())
            self.assertIsNone(report["native"]["measurement"])
            self.assertFalse(report["comparison"]["native_available"])
            self.assertIsNone(report["comparison"]["matched"])
            self.assertTrue(report["python"]["measurement"]["passed"])
            self.assertFalse(report["claims"]["native_execution"])
            self.assertTrue(out_md.is_file())

    def test_json_and_markdown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"
            measurement = _fake_python_measurement()
            with mock.patch.object(
                rc, "python_raw_measurement", return_value=measurement):
                code = rc.main([
                    "--python-only",
                    "--source", str(rc.DEFAULT_SOURCE),
                    "--out-json", str(out_json),
                    "--out-md", str(out_md),
                ])

            self.assertEqual(code, rc.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], rc.SCHEMA)
            for key in (
                "status",
                "mode",
                "source",
                "verification",
                "environment",
                "native",
                "python",
                "reference",
                "comparison",
                "claims",
                "notes",
            ):
                self.assertIn(key, report)
            self.assertEqual(report["status"], "python-only")
            self.assertEqual(report["mode"], "python-only")
            self.assertEqual(report["verification"]["status"], "verified")
            self.assertEqual(
                report["environment"]["target"],
                {"platform": "android", "architecture": "aarch64"},
            )
            self.assertEqual(report["python"]["measurement"], measurement)
            self.assertFalse(report["comparison"]["native_available"])
            self.assertIsNone(report["comparison"]["matched"])
            self.assertEqual(report["comparison"]["bounds"]["root_bytes"], 8_388_608)
            self.assertEqual(report["comparison"]["bounds"]["mirror_pairs"], 32_640)

            markdown = out_md.read_text(encoding="utf-8")
            for heading in (
                "# Round 029 native / Python raw geometry comparison",
                "## Environment",
                "## Digest verification",
                "## Measurements",
                "## Boundaries",
            ):
                self.assertIn(heading, markdown)
            self.assertIn("8388608", markdown)
            self.assertIn("Python raw", markdown)

    def test_python_only_makes_no_canonical_or_native_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"
            with mock.patch.object(
                rc, "python_raw_measurement",
                return_value=_fake_python_measurement(),
            ):
                code = rc.main([
                    "--python-only",
                    "--source", str(rc.DEFAULT_SOURCE),
                    "--out-json", str(out_json),
                    "--out-md", str(out_md),
                ])

            self.assertEqual(code, rc.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            claims = report["claims"]
            for key in (
                "canonical",
                "canonical_implementation",
                "native_execution",
                "native_benchmark",
                "android_benchmark",
                "portable_throughput",
                "semantic_superiority",
                "quantum_claim",
            ):
                self.assertFalse(claims[key], key)
            self.assertNotIn('"canonical": true',
                             out_json.read_text(encoding="utf-8").lower())

            markdown = out_md.read_text(encoding="utf-8")
            plain = markdown.replace("**", "")
            self.assertIn("not an Android benchmark", plain)
            self.assertIn("No canonical implementation", plain)
            self.assertIn("does **not** execute native code", markdown)

    @unittest.skipUnless(
        rc.find_compiler() is not None,
        "no C++ compiler available for the native comparison",
    )
    def test_native_compare_when_compiler_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "report.json"
            out_md = Path(tmp) / "report.md"
            code = rc.main([
                "--source", str(rc.DEFAULT_SOURCE),
                "--out-json", str(out_json),
                "--out-md", str(out_md),
                "--timeout", "120",
            ])

            self.assertEqual(code, rc.EXIT_OK)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["mode"], "native-compare")
            native = report["native"]
            self.assertEqual(native["status"], "ok")
            self.assertEqual(native["measurement"]["root_bytes"], 8_388_608)
            self.assertEqual(native["measurement"]["mirror_pairs"], 32_640)
            self.assertEqual(native["measurement"]["reads"], 500_000)
            comparison = report["comparison"]
            self.assertTrue(comparison["matched"])
            self.assertTrue(comparison["fields"]["root_bytes"]["match"])
            self.assertTrue(comparison["fields"]["mirror_pairs"]["match"])
            self.assertTrue(comparison["fields"]["max_reverse_error"]["match"])
            self.assertIn(
                "not Android",
                comparison["fields"]["read_milliseconds"]["native_label"],
            )
            self.assertTrue(report["claims"]["native_execution"])
            self.assertFalse(report["claims"]["canonical"])
            self.assertFalse(report["claims"]["android_benchmark"])
            self.assertTrue(out_md.is_file())


if __name__ == "__main__":
    unittest.main()
