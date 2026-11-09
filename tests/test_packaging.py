"""Packaging-production checks for the HERMES plugin bundle.

These tests are intentionally cheap: static MANIFEST/package-data coverage and
in-memory archive inspection only.  A real build is opt-in via
``NANLM_PACKAGING_BUILD=1`` so the normal suite never shells out to a builder.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "verify_package.py"

_spec = importlib.util.spec_from_file_location(
    "verify_package_under_test", _SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load scripts/verify_package.py")
vp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vp
_spec.loader.exec_module(vp)


def _requirements():
    return vp.build_requirements(ROOT)


def _wheel_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _sdist_bytes(entries, top="nanlmind_hermes-1.5.0"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(top + "/" + name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _full_wheel_entries(requirements):
    return [(name, b"payload\n") for name in requirements["wheel_required"]]


def _full_sdist_entries(requirements):
    return [(name, b"payload\n") for name in requirements["sdist_required"]]


class ManifestAndMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requirements = _requirements()

    def test_manifest_covers_every_required_sdist_file(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        missing = [
            path for path in self.requirements["sdist_required"]
            if not vp.manifest_covers(manifest, path)
        ]
        self.assertFalse(missing, "MANIFEST.in does not cover: " + ", ".join(missing))

    def test_manifest_recursive_include_carries_nan_sources(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertRegex(
            manifest,
            r"(?m)^recursive-include\s+nanlmind_hermes\b.*(?:\s|^)\*\.nas(?:\s|$)")
        self.assertRegex(
            manifest,
            r"(?m)^recursive-include\s+nanlmind_hermes\b.*(?:\s|^)\*\.nan(?:\s|$)")

    def test_package_data_covers_every_required_wheel_file(self):
        patterns = vp.read_package_data(ROOT)
        self.assertTrue(patterns, "pyproject package-data is empty")
        missing = vp.package_data_missing(self.requirements, patterns)
        self.assertFalse(
            missing,
            "pyproject package-data does not cover: " + ", ".join(missing))
        self.assertIn("nan/*.nan", patterns)
        self.assertIn("nan/*.nas", patterns)
        self.assertIn("session_docs/*.md", patterns)
        self.assertIn("mth/spec/*.json", patterns)
        self.assertIn("mth/nan16/*.nas", patterns)
        self.assertIn("mth/tests/*.nas", patterns)
        self.assertIn("nan0s/core/*.c", patterns)
        self.assertIn("nan0s/core/*.h", patterns)
        self.assertIn("nan0s/core/*.inc", patterns)
        self.assertIn("nan0s/tools/*.c", patterns)
        self.assertIn("nan0s/tools/*.awk", patterns)
        self.assertIn("nan0s/src/tools/*.nas", patterns)
        self.assertIn("nan0s/src/tests/*.nas", patterns)

    def test_mth_inventory_keeps_the_91_kernel_source_fixture_contract(self):
        manifest = json.loads(
            (ROOT / "nanlmind_hermes" / "mth" / "spec" /
             "wave-capabilities.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["kernels"]), vp.EXPECTED_MTH_KERNELS)
        self.assertEqual(len(self.requirements["pairs"]), vp.EXPECTED_MTH_KERNELS)
        for pair in self.requirements["pairs"]:
            with self.subTest(source=pair["source"], fixture=pair["fixture"]):
                self.assertTrue((ROOT / pair["source"]).is_file())
                self.assertTrue((ROOT / pair["fixture"]).is_file())
                self.assertIn(pair["source"], self.requirements["sdist_required"])
                self.assertIn(pair["fixture"], self.requirements["sdist_required"])

    def test_expected_payload_counts_are_concrete(self):
        counts = self.requirements["counts"]
        self.assertEqual(counts["session_docs"], 7)
        self.assertEqual(counts["mth_manifest_kernels"], 91)
        self.assertEqual(counts["mth_fixtures"], 91)
        self.assertGreaterEqual(counts["mth_nan16"], 91)
        self.assertEqual(counts["nan0s_core"], 9)
        self.assertEqual(counts["nan_kernels"], 1)


class ArchiveInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requirements = _requirements()

    def test_synthetic_wheel_with_full_payload_passes(self):
        data = _wheel_bytes(_full_wheel_entries(self.requirements))
        result = vp.inspect_archive_bytes(data, "wheel", self.requirements)
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["counts"]["session_docs"], 7)
        self.assertEqual(result["counts"]["mth_nan16"],
                         self.requirements["counts"]["mth_nan16"])
        self.assertEqual(result["counts"]["mth_fixtures"], 91)
        self.assertEqual(result["counts"]["nan_kernels"], 1)
        self.assertEqual(result["counts"]["nan0s_core"], 9)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["forbidden"], [])
        self.assertFalse(result["checks"]["plugin_manifest"]["applicable"])

    def test_synthetic_sdist_with_top_level_plugin_manifest_passes(self):
        data = _sdist_bytes(_full_sdist_entries(self.requirements))
        result = vp.inspect_archive_bytes(data, "sdist", self.requirements)
        self.assertTrue(result["pass"], result)
        self.assertTrue(result["checks"]["plugin_manifest"]["applicable"])
        self.assertTrue(result["checks"]["plugin_manifest"]["pass"])
        self.assertEqual(result["missing"], [])

    def test_missing_session_doc_is_reported(self):
        missing_name = f"{vp.PACKAGE}/session_docs/WORDNET_HW4.md"
        entries = [
            (name, payload) for name, payload in
            _full_wheel_entries(self.requirements) if name != missing_name
        ]
        result = vp.inspect_archive_bytes(
            _wheel_bytes(entries), "wheel", self.requirements)
        self.assertFalse(result["pass"])
        self.assertIn(missing_name, result["missing"])
        self.assertEqual(result["counts"]["session_docs"], 6)
        self.assertFalse(result["checks"]["session_docs"]["pass"])

    def test_missing_mth_fixture_breaks_the_source_pair(self):
        missing_fixture = (
            f"{vp.PACKAGE}/mth/tests/mth_k256_laplacian_test.nas")
        entries = [
            (name, payload) for name, payload in
            _full_wheel_entries(self.requirements) if name != missing_fixture
        ]
        result = vp.inspect_archive_bytes(
            _wheel_bytes(entries), "wheel", self.requirements)
        self.assertFalse(result["pass"])
        self.assertIn(missing_fixture, result["missing"])
        self.assertFalse(result["checks"]["mth_source_fixture_pairs"]["pass"])
        self.assertEqual(
            len(result["checks"]["mth_source_fixture_pairs"]["missing"]), 1)

    def test_state_secret_and_model_files_are_rejected(self):
        entries = _full_wheel_entries(self.requirements) + [
            (f"{vp.PACKAGE}/auth.sqlite", b""),
            (f"{vp.PACKAGE}/weights/model.safetensors", b""),
            (f"{vp.PACKAGE}/__pycache__/leaked.pyc", b""),
        ]
        result = vp.inspect_archive_bytes(
            _wheel_bytes(entries), "wheel", self.requirements)
        self.assertFalse(result["pass"])
        rejected = {item["name"]: item["reason"] for item in result["forbidden"]}
        self.assertIn(f"{vp.PACKAGE}/auth.sqlite", rejected)
        self.assertIn(f"{vp.PACKAGE}/weights/model.safetensors", rejected)
        self.assertIn(f"{vp.PACKAGE}/__pycache__/leaked.pyc", rejected)
        self.assertFalse(result["checks"]["no_forbidden_files"]["pass"])

    def test_private_key_content_is_rejected(self):
        entries = _full_wheel_entries(self.requirements) + [
            # Split the marker so this test source is not itself a match.
            (f"{vp.PACKAGE}/notes.txt",
             b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----\nnot-real\n"),
        ]
        result = vp.inspect_archive_bytes(
            _wheel_bytes(entries), "wheel", self.requirements)
        self.assertFalse(result["pass"])
        reasons = {item["reason"] for item in result["forbidden"]}
        self.assertIn("secret-content:private-key", reasons)

    def test_oversized_member_is_rejected(self):
        entries = _full_wheel_entries(self.requirements) + [
            (f"{vp.PACKAGE}/fixtures/huge_fixture.dat", b"x" * 64),
        ]
        with mock.patch.object(vp, "MAX_MEMBER_BYTES", 32):
            result = vp.inspect_archive_bytes(
                _wheel_bytes(entries), "wheel", self.requirements)
        self.assertFalse(result["pass"])
        self.assertEqual([item["name"] for item in result["oversized"]],
                         [f"{vp.PACKAGE}/fixtures/huge_fixture.dat"])
        self.assertFalse(result["checks"]["size_bounds"]["pass"])

    def test_corrupt_archive_fails_closed(self):
        result = vp.inspect_archive_bytes(
            b"not a real archive", "wheel", self.requirements)
        self.assertFalse(result["pass"])
        self.assertIn("error", result)


@unittest.skipUnless(
    os.environ.get("NANLM_PACKAGING_BUILD") == "1",
    "set NANLM_PACKAGING_BUILD=1 to run the real wheel/sdist build")
class RealBuildTests(unittest.TestCase):
    def test_real_build_archives_pass_inspection(self):
        if not vp.real_build_available():
            self.skipTest("neither python -m build nor pip wheel is available")
        timeout = float(os.environ.get("NANLM_PACKAGING_TIMEOUT", "300"))
        receipt = vp.verify_package(ROOT, timeout=timeout)
        self.addCleanup(
            shutil.rmtree, receipt.get("work_dir", ""), True)
        if not receipt["pass"]:
            self.fail(json.dumps(receipt, indent=2, sort_keys=True)[-12000:])


if __name__ == "__main__":
    unittest.main()
