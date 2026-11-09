#!/usr/bin/env python3
"""Synchronize the vendored NaN0S snapshot from the canonical repository.

HERMES does not own a NaN16 implementation.  ``nanlmind_hermes/nan0s`` is an
exact, pinned snapshot of ``shemshallah/NaN0S`` (aarch64/android portable core
plus the N16I/interpreter sources); this tool copies and hashes it.

Usage:
    python3 scripts/sync_nan0s.py --source ~/nan0s --write
    python3 scripts/sync_nan0s.py --source ~/nan0s --check

The script never edits the canonical checkout.  A local edit under
``nanlmind_hermes/nan0s`` is detected by ``--check`` and by
``tests/test_nan0s_sync.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "nanlmind_hermes" / "nan0s"
MANIFEST = BUNDLED / "SOURCE.json"

# relative path in the bundled snapshot -> path under the canonical checkout
FILES = {
    "core/OPS.md": "aarch64/android/core/OPS.md",
    "core/nan16.c": "aarch64/android/core/nan16.c",
    "core/nan16.h": "aarch64/android/core/nan16.h",
    "core/nan16_int.h": "aarch64/android/core/nan16_int.h",
    "core/nan_dev.c": "aarch64/android/core/nan_dev.c",
    "core/nan_load.c": "aarch64/android/core/nan_load.c",
    "core/nan_render.c": "aarch64/android/core/nan_render.c",
    "core/nan_state.c": "aarch64/android/core/nan_state.c",
    "core/nan_net_driver.c": "aarch64/android/core/nan_net_driver.c",
    "core/nan_net_driver.h": "aarch64/android/core/nan_net_driver.h",
    "tools/nan_cli.c": "aarch64/android/tools/nan_cli.c",
    "tools/nmk.awk": "tools/nmk.awk",
    "src/tools/n16i.nas": "src/tools/n16i.nas",
    "src/tests/n16i_test.nas": "src/tests/n16i_test.nas",
    "src/tests/micro_block.nas": "src/tests/micro_block.nas",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_commit(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path.home() / "nan0s"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not (source / "aarch64" / "android" / "core" / "nan16.c").is_file():
        print("not a NaN0S checkout: %s" % source, file=sys.stderr)
        return 2
    if not args.write and not args.check:
        parser.error("choose --write or --check")
    expected = {}
    for vendored, canonical in FILES.items():
        src = source / canonical
        if not src.is_file():
            print("missing canonical file: %s" % src, file=sys.stderr)
            return 2
        expected[vendored] = {
            "sha256": digest(src),
            "bytes": src.stat().st_size,
        }
    if args.check:
        bad = 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for vendored, row in expected.items():
            path = BUNDLED / vendored
            if not path.is_file() or digest(path) != row["sha256"]:
                print("DRIFT %s" % vendored, file=sys.stderr)
                bad += 1
        if manifest.get("canonical_commit") != canonical_commit(source):
            print("DRIFT source commit: manifest %s canonical %s" % (
                manifest.get("canonical_commit"), canonical_commit(source)),
                file=sys.stderr)
            bad += 1
        print("nan0s sync: %s (%d files)" % ("PASS" if not bad else "FAIL",
                                             len(expected)))
        return 0 if not bad else 1
    for vendored, canonical in FILES.items():
        dest = BUNDLED / vendored
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / canonical, dest)
    manifest = {
        "schema": "nanlm.nan0s-dependency/1",
        "canonical_repository": "https://github.com/shemshallah/NaN0S.git",
        "canonical_commit": canonical_commit(source),
        "vendored_path": "nanlmind_hermes/nan0s",
        "note": ("Exact pinned snapshot. Do not edit locally; run "
                 "scripts/sync_nan0s.py from canonical NaN0S."),
        "files": expected,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("nan0s sync: wrote %d files from %s" % (len(expected), source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
