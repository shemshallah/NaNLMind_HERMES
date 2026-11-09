"""N16SNAP1 fence: digest, generation, root, and foreign-universe isolation."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.chart_storage import save_chart_snapshot
from nanlmind_hermes.head_bank_transport import (
    BANK_BYTES, BANK_OFFSET, describe_guest_bank, stage_guest_bank_bytes,
)
from nanlmind_hermes.native_head_fence import (
    NativeHeadFenceError, SnapshotHeadBankFence,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class NativeHeadCheckpointFenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.parent = self.chart.root_hash
        self.universe = 12345
        self.reader = SnapshotHeadBankFence(
            state_dir=self.path, parent_root_sha256=self.parent,
            universe_id=self.universe,
        )

    def checkpoint(self, generation):
        return save_chart_snapshot(self.path, self.parent, self.chart.bytes, generation)

    def test_requires_existing_durable_snapshot_and_correct_universe(self):
        with self.assertRaisesRegex(NativeHeadFenceError, "no durable checkpoint"):
            self.reader(self.universe)
        with self.assertRaisesRegex(NativeHeadFenceError, "foreign"):
            self.reader(self.universe + 1)
        with self.assertRaises(NativeHeadFenceError):
            SnapshotHeadBankFence(
                state_dir=self.path, parent_root_sha256="G" * 64,
                universe_id=self.universe,
            )
        with self.assertRaises(NativeHeadFenceError):
            SnapshotHeadBankFence(
                state_dir=self.path, parent_root_sha256=self.parent,
                universe_id=True,
            )

    def test_exact_bank_digest_and_generation_follow_checkpoint_not_memory(self):
        first = self.checkpoint(0)
        self.assertEqual(first, 1)
        fence = self.reader(self.universe)
        self.assertEqual(fence["chart_generation"], 1)
        self.assertEqual(
            fence["bank_sha256"], sha(bytes(BANK_BYTES))
        )
        self.assertTrue(fence["snapshot_verified"])
        self.assertFalse(fence["native_guest_attached"])
        self.assertFalse(fence["canonical_training_committed"])
        raw = bytes([0xD3]) * BANK_BYTES
        initial = describe_guest_bank(self.chart)
        stage_guest_bank_bytes(
            self.chart, raw,
            payload_sha256=sha(raw),
            expected_region_sha256=initial["region_sha256"],
            expected_parent_root=self.parent,
            expected_snapshot_generation=0,
        )
        # In-memory modification must NOT produce a trusted durable fence.
        self.assertEqual(self.reader(self.universe)["bank_sha256"], sha(bytes(BANK_BYTES)))
        self.assertEqual(self.checkpoint(1), 2)
        updated = self.reader(self.universe)
        self.assertEqual(updated["chart_generation"], 2)
        self.assertEqual(updated["bank_sha256"], sha(raw))

    def test_corrupt_newest_snapshot_recovers_last_valid_generation(self):
        self.checkpoint(0)
        raw = bytes([0x22]) * BANK_BYTES
        initial = describe_guest_bank(self.chart)
        stage_guest_bank_bytes(
            self.chart, raw, payload_sha256=sha(raw),
            expected_region_sha256=initial["region_sha256"],
            expected_parent_root=self.parent,
            expected_snapshot_generation=0,
        )
        self.checkpoint(1)
        latest = self.path / "language" / "chart-checkpoints" / "chart.1.n16s"
        with latest.open("r+b") as file:
            file.seek(200)
            file.write(bytes([0xF0]))
        recovered = self.reader(self.universe)
        self.assertEqual(recovered["chart_generation"], 1)
        self.assertEqual(recovered["bank_sha256"], sha(bytes(BANK_BYTES)))

    def test_ambiguous_valid_same_generation_chart_denies_native_dispatch(self):
        from nanlmind_hermes.chart_storage import HEADER, MAGIC, SLOT_NAMES
        self.checkpoint(0)
        page = bytearray(self.chart.bytes)
        page[BANK_OFFSET] ^= 0x01
        folder = self.path / "language" / "chart-checkpoints"
        (folder / SLOT_NAMES[1]).write_bytes(
            HEADER.pack(MAGIC, 1, bytes.fromhex(self.parent),
                        hashlib.sha256(page).digest()) + page
        )
        with self.assertRaisesRegex(NativeHeadFenceError, "checkpoint unavailable"):
            self.reader(self.universe)
        self.assertEqual((folder / SLOT_NAMES[0]).stat().st_size,
                         HEADER.size + len(page))

    def test_fence_keeps_original_state_home_after_symlink_retarget(self):
        """A redirected alias must never switch the native checkpoint source."""
        original = self.path / "native-state-original"
        decoy = self.path / "native-state-decoy"
        original.mkdir()
        decoy.mkdir()
        alias = self.path / "native-state-alias"
        alias.symlink_to(original, target_is_directory=True)
        fence = SnapshotHeadBankFence(
            state_dir=alias, parent_root_sha256=self.parent,
            universe_id=self.universe,
        )
        self.assertEqual(fence._state_dir, original.resolve())
        self.assertEqual(
            save_chart_snapshot(original, self.parent, self.chart.bytes, 0), 1
        )
        # Another complete, hash-valid chart elsewhere is not this universe.
        alternate = bytearray(self.chart.bytes)
        alternate[BANK_OFFSET] ^= 0x01
        self.assertEqual(
            save_chart_snapshot(decoy, self.parent, alternate, 0), 1
        )
        alias.unlink()
        alias.symlink_to(decoy, target_is_directory=True)
        observed = fence(self.universe)
        self.assertEqual(observed["chart_generation"], 1)
        self.assertEqual(observed["bank_sha256"], sha(bytes(BANK_BYTES)))
        self.assertEqual(observed["chart_sha256"], sha(self.chart.bytes))
        self.assertNotEqual(observed["chart_sha256"], sha(alternate))

    def test_invalid_root_is_denied_even_with_valid_snapshot_sha(self):
        # N16SNAP1 validates page SHA, but a rootless valid chart MUST NOT be
        # accepted as authenticated NaN16 universe storage.
        bare = Chart8MiB.genesis()
        self.assertEqual(self.parent, bare.root_hash)
        save_chart_snapshot(self.path, self.parent, bare.bytes, 0)
        with self.assertRaisesRegex(NativeHeadFenceError, "not mounted"):
            self.reader(self.universe)


if __name__ == "__main__":
    unittest.main()
