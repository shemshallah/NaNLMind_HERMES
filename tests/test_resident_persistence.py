"""Crash/recovery tests for resident universe checkpoint persistence.

Run with::

    python3 -m unittest tests.test_resident_persistence -v
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanlmind_hermes.resident.persistence import (
    LOCK_NAME,
    MAX_FILE_BYTES,
    SLOT_NAMES,
    PersistenceError,
    checkpoint,
    manifest_digest,
    recover,
    verify_file,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_sparse(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)
    return path


def patch_file(path: Path, offset: int, data: bytes) -> None:
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(data)


class CheckpointRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "checkpoints"

    def tearDown(self):
        self._temp.cleanup()

    def test_checkpoint_reopen_and_dual_slots(self):
        first_data = b"resident-universe-generation-one" * 8
        second_data = b"resident-universe-generation-two" * 8
        write_bytes(self.root / "universe-1.nlm", first_data)
        write_bytes(self.root / "universe-2.nlm", second_data)

        first = checkpoint(
            self.root, 1,
            {"note": "first", "files": [{"path": "universe-1.nlm"}]})
        self.assertEqual(first.generation, 1)
        self.assertEqual(first.slot, 0)
        self.assertEqual(first.path, self.root / SLOT_NAMES[0])
        self.assertEqual(first.manifest["note"], "first")
        entry = first.manifest["files"][0]
        self.assertEqual(entry["path"], "universe-1.nlm")
        self.assertEqual(entry["size"], len(first_data))
        self.assertEqual(entry["digest"], sha256(first_data))

        reopened = recover(self.root)
        self.assertEqual(reopened.generation, 1)
        self.assertEqual(reopened.slot, 0)
        self.assertEqual(reopened.manifest, first.manifest)
        self.assertEqual(reopened.digest, first.digest)

        second = checkpoint(
            self.root, 2,
            {"note": "second", "files": [{"path": "universe-2.nlm"}]})
        self.assertEqual(second.slot, 1)
        recovered = recover(self.root)
        self.assertEqual(recovered.generation, 2)
        self.assertEqual(recovered.manifest["note"], "second")
        self.assertEqual(
            recovered.manifest["files"][0]["digest"], sha256(second_data))

        # Exactly the two dual-slot manifests remain; the older generation is
        # retained for fallback until the next slot rotation.
        slots = sorted(path.name for path in self.root.iterdir()
                       if path.name.startswith("manifest."))
        self.assertEqual(slots, list(SLOT_NAMES))
        with (self.root / SLOT_NAMES[0]).open("r", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream)["generation"], 1)
        with (self.root / SLOT_NAMES[1]).open("r", encoding="utf-8") as stream:
            self.assertEqual(json.load(stream)["generation"], 2)

    def test_checkpoint_files_and_directory_permissions(self):
        write_bytes(self.root / "universe.nlm", b"payload")
        checkpoint(self.root, 1, {"files": [{"path": "universe.nlm"}]})
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        for name in SLOT_NAMES[:1]:
            slot = self.root / name
            self.assertEqual(stat.S_IMODE(slot.stat().st_mode), 0o600)
        lock = self.root / LOCK_NAME
        self.assertEqual(stat.S_IMODE(lock.stat().st_mode), 0o600)

    def test_single_file_manifest_shorthand(self):
        data = b"shorthand"
        write_bytes(self.root / "one.nlm", data)
        committed = checkpoint(self.root, 4, {"path": "one.nlm", "tag": "sh"})
        self.assertEqual(committed.manifest["tag"], "sh")
        self.assertEqual(committed.manifest["files"][0]["digest"], sha256(data))
        self.assertEqual(recover(self.root).manifest["files"][0]["path"], "one.nlm")


class CorruptLatestFallbackTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "checkpoints"

    def tearDown(self):
        self._temp.cleanup()

    def _two_generations(self):
        first_data = b"generation-one-payload"
        second_data = b"generation-two-payload"
        first_file = write_bytes(self.root / "gen-1.nlm", first_data)
        write_bytes(self.root / "gen-2.nlm", second_data)
        first = checkpoint(
            self.root, 1, {"files": [{"path": "gen-1.nlm"}]})
        second = checkpoint(
            self.root, 2, {"files": [{"path": "gen-2.nlm"}]})
        self.assertNotEqual(first.slot, second.slot)
        return first, second, first_file

    def test_truncated_latest_slot_falls_back(self):
        _first, second, _first_file = self._two_generations()
        latest = self.root / SLOT_NAMES[second.slot]
        raw = latest.read_bytes()
        latest.write_bytes(raw[:max(1, len(raw) // 2)])

        recovered = recover(self.root)
        self.assertEqual(recovered.generation, 1)
        self.assertEqual(
            recovered.manifest["files"][0]["path"], "gen-1.nlm")
        self.assertTrue(any("checksum" in item or "JSON" in item
                            for item in recovered.rejected))

    def test_bit_flipped_latest_slot_falls_back(self):
        _first, second, _first_file = self._two_generations()
        latest = self.root / SLOT_NAMES[second.slot]
        raw = bytearray(latest.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        latest.write_bytes(raw)

        recovered = recover(self.root)
        self.assertEqual(recovered.generation, 1)

    def test_digest_mismatched_latest_payload_falls_back(self):
        _first, _second, _first_file = self._two_generations()
        patch_file(self.root / "gen-2.nlm", 0, b"X")

        recovered = recover(self.root)
        self.assertEqual(recovered.generation, 1)
        self.assertTrue(any("payload verification" in item
                            for item in recovered.rejected))

    def test_truncated_latest_payload_falls_back(self):
        _first, _second, _first_file = self._two_generations()
        (self.root / "gen-2.nlm").write_bytes(b"short")

        self.assertEqual(recover(self.root).generation, 1)


class FailClosedTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "checkpoints"

    def tearDown(self):
        self._temp.cleanup()

    def _two_generations(self):
        write_bytes(self.root / "gen-1.nlm", b"one")
        write_bytes(self.root / "gen-2.nlm", b"two")
        checkpoint(self.root, 1, {"files": [{"path": "gen-1.nlm"}]})
        return checkpoint(self.root, 2, {"files": [{"path": "gen-2.nlm"}]})

    def test_missing_checkpoint_directory_fails_closed(self):
        with self.assertRaisesRegex(PersistenceError, "no valid resident checkpoint"):
            recover(self.root)

    def test_both_slots_corrupt_fails_closed(self):
        self._two_generations()
        for name in SLOT_NAMES:
            (self.root / name).write_bytes(b"torn")
        with self.assertRaisesRegex(PersistenceError, "no valid resident checkpoint"):
            recover(self.root)

    def test_checkpoint_refuses_to_reinitialize_corrupt_slots(self):
        self._two_generations()
        for name in SLOT_NAMES:
            (self.root / name).write_bytes(b"torn")
        with self.assertRaisesRegex(PersistenceError, "refusing to reinitialize"):
            checkpoint(
                self.root, 3, {"files": [{"path": "gen-1.nlm"}]})
        for name in SLOT_NAMES:
            self.assertEqual((self.root / name).read_bytes(), b"torn")

    def test_both_payloads_corrupt_fails_closed(self):
        self._two_generations()
        patch_file(self.root / "gen-1.nlm", 0, b"X")
        patch_file(self.root / "gen-2.nlm", 0, b"X")
        with self.assertRaisesRegex(PersistenceError, "no valid resident checkpoint"):
            recover(self.root)

    def test_symlinked_slot_is_rejected(self):
        self._two_generations()
        for name in SLOT_NAMES:
            slot = self.root / name
            if slot.exists():
                outside = self.root / ("outside-" + name)
                slot.replace(outside)
                slot.symlink_to(outside)
        with self.assertRaisesRegex(PersistenceError, "no valid resident checkpoint"):
            recover(self.root)


class DigestAndSparseTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "checkpoints"

    def tearDown(self):
        self._temp.cleanup()

    def test_manifest_digest_full_and_declared_regions(self):
        data = bytes(range(256)) * 4
        target = write_bytes(self.root / "data.nlm", data)

        self.assertEqual(manifest_digest(target), sha256(data))
        self.assertTrue(verify_file(target, {
            "size": len(data), "digest": sha256(data)}))
        self.assertFalse(verify_file(target, {
            "size": len(data), "digest": "f" * 64}))

        regions = [
            {"offset": 10, "length": 20},
            {"offset": 100, "length": 5},
        ]
        expected = sha256(data[10:30] + data[100:105])
        self.assertEqual(manifest_digest(target, {"regions": regions}), expected)
        # Declared order is normalised to ascending offsets.
        self.assertEqual(
            manifest_digest(target, {"regions": list(reversed(regions))}),
            expected)

        entry = {
            "size": len(data),
            "digest": expected,
            "regions": [
                {"offset": 10, "length": 20,
                 "digest": sha256(data[10:30])},
                {"offset": 100, "length": 5,
                 "digest": sha256(data[100:105])},
            ],
        }
        self.assertTrue(verify_file(target, entry))

        wrong_aggregate = dict(entry, digest="0" * 64)
        self.assertFalse(verify_file(target, wrong_aggregate))
        wrong_region = {
            "regions": [{"offset": 10, "length": 20, "digest": "0" * 64}],
        }
        self.assertFalse(verify_file(target, wrong_region))
        wrong_size = dict(entry, size=len(data) + 1)
        self.assertFalse(verify_file(target, wrong_size))

        patch_file(target, 10, b"Z")
        self.assertFalse(verify_file(target, entry))

        # A manifest mapping itself can be digested canonically (key order is
        # irrelevant), which is useful when callers cache manifest digests.
        self.assertEqual(
            manifest_digest({"b": 2, "a": 1}),
            sha256(b'{"a":1,"b":2}'))

    def test_checkpoint_fills_region_digests_and_recovers(self):
        data = bytes(range(64)) * 8
        target = write_bytes(self.root / "data.nlm", data)
        committed = checkpoint(self.root, 1, {
            "files": [{
                "path": "data.nlm",
                "regions": [
                    {"offset": 0, "length": 16},
                    {"offset": 400, "length": 32},
                ],
            }],
        })
        stored = committed.manifest["files"][0]
        self.assertEqual(
            stored["digest"], sha256(data[0:16] + data[400:432]))
        self.assertEqual(stored["regions"][0]["digest"], sha256(data[0:16]))
        self.assertEqual(stored["regions"][1]["digest"], sha256(data[400:432]))
        self.assertEqual(recover(self.root).generation, 1)
        self.assertTrue(verify_file(target, stored))

    def test_digest_and_region_mismatch_refused_before_publish(self):
        data = b"checkpoint payload"
        write_bytes(self.root / "data.nlm", data)
        with self.assertRaisesRegex(PersistenceError, "digest mismatch"):
            checkpoint(self.root, 1, {
                "files": [{"path": "data.nlm", "digest": "0" * 64}]})
        self.assertFalse((self.root / SLOT_NAMES[0]).exists())
        self.assertFalse((self.root / SLOT_NAMES[1]).exists())

        with self.assertRaisesRegex(PersistenceError, "region 0 digest mismatch"):
            checkpoint(self.root, 1, {
                "files": [{
                    "path": "data.nlm",
                    "regions": [{"offset": 0, "length": 4, "digest": "1" * 64}],
                }],
            })
        self.assertFalse((self.root / SLOT_NAMES[0]).exists())

        with self.assertRaisesRegex(PersistenceError, "size mismatch"):
            checkpoint(self.root, 1, {
                "files": [{"path": "data.nlm", "size": len(data) + 1}]})
        self.assertFalse((self.root / SLOT_NAMES[0]).exists())

    def test_sparse_file_regions_and_mmap(self):
        size = 4 * 1024 * 1024
        target = make_sparse(self.root / "sparse.nlm", size)
        header = b"NLMRES01" + b"\x00" * 8
        tail = b"resident-tail!!"
        patch_file(target, 0, header)
        patch_file(target, size - len(tail), tail)

        entry = {
            "path": "sparse.nlm",
            "regions": [
                {"offset": 0, "length": len(header)},
                {"offset": size - len(tail), "length": len(tail)},
            ],
        }
        committed = checkpoint(self.root, 1, {"files": [entry]})
        stored = committed.manifest["files"][0]
        self.assertEqual(stored["size"], size)
        self.assertEqual(
            stored["digest"], sha256(header + tail))
        self.assertEqual(recover(self.root).generation, 1)

        with target.open("rb") as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                self.assertEqual(
                    manifest_digest(mapped),
                    sha256(target.read_bytes()))
                self.assertEqual(
                    manifest_digest(mapped, stored),
                    stored["digest"])
                self.assertTrue(verify_file(mapped, stored))

    def test_sparse_file_over_one_gib_rejected(self):
        huge = make_sparse(self.root / "huge.nlm", MAX_FILE_BYTES + 1)
        with self.assertRaisesRegex(PersistenceError, "1 GiB"):
            checkpoint(self.root, 1, {"files": [{"path": "huge.nlm"}]})
        with self.assertRaisesRegex(PersistenceError, "1 GiB"):
            manifest_digest(huge)


class AtomicWriteAndGenerationTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name) / "checkpoints"

    def tearDown(self):
        self._temp.cleanup()

    def _payload(self, name: str, data: bytes) -> None:
        write_bytes(self.root / name, data)

    def _manifest(self, name: str) -> dict:
        return {"files": [{"path": name}]}

    def test_stale_temporary_files_are_cleaned(self):
        self._payload("one.nlm", b"one")
        stale = self.root / ".manifest.deadbeef.tmp"
        write_bytes(stale, b"partial write")
        checkpoint(self.root, 1, self._manifest("one.nlm"))
        self.assertFalse(stale.exists())
        self.assertEqual(list(self.root.glob(".manifest.*.tmp")), [])

    def test_oversized_manifest_is_refused_before_publish(self):
        self._payload("one.nlm", b"one")
        manifest = self._manifest("one.nlm")
        manifest["note"] = "x" * (1024 * 1024 + 1)
        with self.assertRaisesRegex(PersistenceError, "slot bound"):
            checkpoint(self.root, 1, manifest)
        self.assertFalse((self.root / SLOT_NAMES[0]).exists())
        self.assertFalse((self.root / SLOT_NAMES[1]).exists())

    def test_failed_atomic_replace_leaves_no_temp_or_slot(self):
        self._payload("one.nlm", b"one")
        with mock.patch(
                "nanlmind_hermes.resident.persistence.os.replace",
                side_effect=OSError("injected replace failure")):
            with self.assertRaises(OSError):
                checkpoint(self.root, 1, self._manifest("one.nlm"))
        self.assertEqual(list(self.root.glob(".manifest.*.tmp")), [])
        self.assertFalse((self.root / SLOT_NAMES[0]).exists())
        self.assertFalse((self.root / SLOT_NAMES[1]).exists())

        # A retry after the failed rename is accepted.
        retried = checkpoint(self.root, 1, self._manifest("one.nlm"))
        self.assertEqual(retried.generation, 1)
        self.assertEqual(recover(self.root).generation, 1)

    def test_generation_is_strictly_monotonic_and_only_two_slots_survive(self):
        self._payload("one.nlm", b"one")
        self._payload("two.nlm", b"two")
        self._payload("three.nlm", b"three")

        checkpoint(self.root, 1, self._manifest("one.nlm"))
        checkpoint(self.root, 2, self._manifest("two.nlm"))
        with self.assertRaisesRegex(PersistenceError, "stale checkpoint generation"):
            checkpoint(self.root, 2, self._manifest("two.nlm"))
        with self.assertRaisesRegex(PersistenceError, "stale checkpoint generation"):
            checkpoint(self.root, 1, self._manifest("one.nlm"))

        third = checkpoint(self.root, 3, self._manifest("three.nlm"))
        self.assertEqual(third.slot, 0)
        self.assertEqual(recover(self.root).generation, 3)

        slots = sorted(path.name for path in self.root.iterdir()
                       if path.name.startswith("manifest."))
        self.assertEqual(slots, list(SLOT_NAMES))

        # Generation 1 was evicted by the third checkpoint; the previous valid
        # generation (2) is the fallback after tearing generation 3.
        latest = self.root / SLOT_NAMES[third.slot]
        latest.write_bytes(latest.read_bytes()[:20])
        recovered = recover(self.root)
        self.assertEqual(recovered.generation, 2)
        self.assertEqual(
            recovered.manifest["files"][0]["path"], "two.nlm")

    def test_invalid_generation_values_are_refused(self):
        self._payload("one.nlm", b"one")
        for generation in (0, -1, True, "1"):
            with self.subTest(generation=generation):
                with self.assertRaises(PersistenceError):
                    checkpoint(self.root, generation, self._manifest("one.nlm"))


if __name__ == "__main__":
    unittest.main()
