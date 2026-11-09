"""Race-injected N16SNAP1 snapshot transport verification.

These tests mutate the directory name *after* a trusted descriptor is open.
A path-based read/write would be redirected to a foreign valid checkpoint.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nanlmind_hermes.chart_storage as storage
from nanlmind_hermes.universe_chart import Chart8MiB, OFF_EXTENSION


class SnapshotDescriptorRaceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        chart = Chart8MiB.genesis()
        self.parent = chart.root_hash
        self.original = bytes(chart.bytes)
        alternate = bytearray(self.original)
        alternate[OFF_EXTENSION + 2048] ^= 0x5A
        self.alternate = bytes(alternate)
        updated = bytearray(self.original)
        updated[OFF_EXTENSION + 2049] ^= 0x32
        self.updated = bytes(updated)
        self.real = self.base / "real"
        self.decoy = self.base / "decoy"
        storage.save_chart_snapshot(self.real, self.parent, self.original, 0)
        storage.save_chart_snapshot(self.decoy, self.parent, self.alternate, 0)

    def swap_child(self, level):
        if level == "language":
            node = self.real / "language"
            other = self.decoy / "language"
        else:
            node = self.real / "language" / "chart-checkpoints"
            other = self.decoy / "language" / "chart-checkpoints"
        held = node.with_name(node.name + "-held")
        node.rename(held)
        node.symlink_to(other, target_is_directory=True)
        def restore():
            node.unlink()
            held.rename(node)
        return restore

    def test_directory_retarget_during_verified_read_uses_original_fd(self):
        for level in ("language", "chart-checkpoints"):
            with self.subTest(level=level):
                prior_read = storage._read_at
                restore = None
                def redirected_read(fd, name):
                    nonlocal restore
                    if restore is None:
                        restore = self.swap_child(level)
                    return prior_read(fd, name)
                try:
                    with patch.object(storage, "_read_at", side_effect=redirected_read):
                        generation, page = storage.load_chart_snapshot(
                            self.real, self.parent
                        )
                    self.assertEqual((generation, page), (1, self.original))
                    self.assertEqual(
                        storage.load_chart_snapshot(self.decoy, self.parent),
                        (1, self.alternate),
                    )
                finally:
                    if restore:
                        restore()

    def test_directory_retarget_during_locked_cas_does_not_write_decoy(self):
        for level in ("language", "chart-checkpoints"):
            with self.subTest(level=level):
                prior_candidates = storage._candidates
                restore = None
                def redirected_candidates(fd, parent):
                    nonlocal restore
                    if restore is None:
                        restore = self.swap_child(level)
                    return prior_candidates(fd, parent)
                try:
                    with patch.object(storage, "_candidates",
                                      side_effect=redirected_candidates):
                        generation = storage.save_chart_snapshot(
                            self.real, self.parent, self.updated, 1,
                        )
                    self.assertEqual(generation, 2)
                    self.assertEqual(
                        storage.load_chart_snapshot(self.decoy, self.parent),
                        (1, self.alternate),
                    )
                finally:
                    if restore:
                        restore()
                self.assertEqual(
                    storage.load_chart_snapshot(self.real, self.parent),
                    (2, self.updated),
                )
                # Each subtest starts its own pinned base generation.
                if level == "language":
                    self.real = self.base / "real-checkpoints"
                    storage.save_chart_snapshot(self.real, self.parent,
                                                self.original, 0)

    def test_slot_replacement_with_symlink_just_before_open_fails_closed(self):
        slot = self.real / "language" / "chart-checkpoints" / storage.SLOT_NAMES[0]
        other = self.decoy / "language" / "chart-checkpoints" / storage.SLOT_NAMES[0]
        held = slot.with_name("held.n16s")
        original_open = os.open
        switched = [False]
        def swap_during_open(path, flags, *args, **kwargs):
            if path == storage.SLOT_NAMES[0] and not switched[0]:
                switched[0] = True
                slot.rename(held)
                slot.symlink_to(other)
            return original_open(path, flags, *args, **kwargs)
        try:
            with patch.object(storage.os, "open", side_effect=swap_during_open):
                with self.assertRaises(storage.ChartStorageError):
                    storage.load_chart_snapshot(self.real, self.parent)
            self.assertTrue(switched[0])
        finally:
            if switched[0]:
                slot.unlink()
                held.rename(slot)
        self.assertEqual(
            storage.load_chart_snapshot(self.real, self.parent),
            (1, self.original),
        )
        self.assertEqual(
            storage.load_chart_snapshot(self.decoy, self.parent),
            (1, self.alternate),
        )


if __name__ == "__main__":
    unittest.main()
