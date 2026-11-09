"""SHA-valid immutable-layout corruption must not become native fence authority."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes.chart_storage import load_chart_snapshot, save_chart_snapshot
from nanlmind_hermes.head_bank_transport import BANK_OFFSET
from nanlmind_hermes.native_head_fence import NativeHeadFenceError, SnapshotHeadBankFence
from nanlmind_hermes.universe_chart import (
    OFF_CANON, OFF_INDEX, OFF_PAIR_TABLE, Chart8MiB,
)
from nanlmind_hermes.universe_root import attach_universe_root


class NativeFenceReconstructionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state_dir = Path(temporary.name)
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.parent = self.chart.root_hash
        self.universe = 192

    def reader(self, folder=None):
        return SnapshotHeadBankFence(
            state_dir=folder or self.state_dir,
            parent_root_sha256=self.parent,
            universe_id=self.universe,
        )

    def test_forged_valid_sha_and_root_do_not_authorize_immutable_layout(self):
        # N16ROOT1 is still mounted and outer SHA recomputes correctly.
        for name, offset in (
            ("genome", 12), ("index", OFF_INDEX),
            ("canon", OFF_CANON), ("pair", OFF_PAIR_TABLE),
        ):
            with self.subTest(name=name):
                folder = self.state_dir / name
                page = bytearray(self.chart.bytes)
                page[offset] ^= 1
                save_chart_snapshot(folder, self.parent, page, 0)
                self.assertEqual(load_chart_snapshot(folder, self.parent),
                                 (1, bytes(page)))
                with self.assertRaisesRegex(
                        NativeHeadFenceError, "immutable reconstruction"):
                    self.reader(folder)(self.universe)
                # Refusal is read-only; nothing is silently reset.
                self.assertEqual(load_chart_snapshot(folder, self.parent),
                                 (1, bytes(page)))

    def test_newest_sha_valid_forgery_does_not_downgrade_to_old_generation(self):
        save_chart_snapshot(self.state_dir, self.parent, self.chart.bytes, 0)
        forged = bytearray(self.chart.bytes)
        forged[OFF_INDEX + 10] ^= 1
        save_chart_snapshot(self.state_dir, self.parent, forged, 1)
        with self.assertRaisesRegex(
                NativeHeadFenceError, "immutable reconstruction"):
            self.reader()(self.universe)
        self.assertEqual(load_chart_snapshot(self.state_dir, self.parent),
                         (2, bytes(forged)))

    def test_valid_native_bank_change_still_certifies(self):
        save_chart_snapshot(self.state_dir, self.parent, self.chart.bytes, 0)
        first = self.reader()(self.universe)
        changed = bytearray(self.chart.bytes)
        changed[BANK_OFFSET] ^= 0x5A
        save_chart_snapshot(self.state_dir, self.parent, changed, 1)
        latest = self.reader()(self.universe)
        self.assertEqual(latest["chart_generation"], 2)
        self.assertNotEqual(latest["bank_sha256"], first["bank_sha256"])
        self.assertEqual(latest["chart_sha256"], hashlib.sha256(changed).hexdigest())
        self.assertFalse(latest["canonical_training_committed"])

    def test_foreign_or_unreadable_canonical_source_denied(self):
        save_chart_snapshot(self.state_dir, self.parent, self.chart.bytes, 0)
        import nanlmind_hermes.native_chart_reconstruction as reconstruction
        foreign = Chart8MiB.genesis()
        foreign.root_hash = "f" * 64
        with patch.object(reconstruction, "load_chart", return_value=foreign):
            with self.assertRaisesRegex(NativeHeadFenceError, "immutable"):
                self.reader()(self.universe)
        with patch.object(reconstruction, "load_chart",
                          side_effect=ValueError("unreadable")):
            with self.assertRaisesRegex(NativeHeadFenceError, "immutable"):
                self.reader()(self.universe)


if __name__ == "__main__":
    unittest.main()
