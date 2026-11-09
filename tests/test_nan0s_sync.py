"""The vendored NaN0S snapshot must be exact and pinned, not a fork."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "nanlmind_hermes" / "nan0s"
MANIFEST = BUNDLED / "SOURCE.json"
CANONICAL = Path.home() / "nan0s"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Nan0sDependencyTests(unittest.TestCase):
    def test_manifest_hashes_match_bundled_snapshot(self):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], "nanlm.nan0s-dependency/1")
        self.assertTrue(data["canonical_commit"])
        self.assertGreaterEqual(len(data["files"]), 12)
        for relative, row in data["files"].items():
            path = BUNDLED / relative
            with self.subTest(relative=relative):
                self.assertTrue(path.is_file())
                self.assertEqual(_sha(path), row["sha256"])
                self.assertEqual(path.stat().st_size, row["bytes"])

    def test_canonical_checkout_is_byte_identical_when_present(self):
        if not (CANONICAL / "aarch64" / "android" / "core" / "nan16.c").is_file():
            self.skipTest("canonical NaN0S checkout not present")
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for relative, row in data["files"].items():
            if relative == "tools/nmk.awk":
                canonical = CANONICAL / "tools" / "nmk.awk"
            elif relative.startswith("src/"):
                canonical = CANONICAL / relative
            else:
                canonical = CANONICAL / "aarch64" / "android" / relative
            with self.subTest(relative=relative):
                self.assertTrue(canonical.is_file())
                self.assertEqual(_sha(canonical), row["sha256"])

    def test_runtime_prefers_external_nan0s(self):
        # The dependency search order must try the configured/sibling checkout
        # before the bundled snapshot.
        source = (ROOT / "nanlmind_hermes" / "nan0s_runtime.py").read_text(
            encoding="utf-8")
        external = source.index("Path.home() / \"nan0s\"")
        bundled = source.index("builtin = _layout(_BUNDLED)")
        self.assertLess(external, bundled)
        self.assertNotIn("nan0s_prefer_bundled", source)


if __name__ == "__main__":
    unittest.main()
