"""A pinned state home must not follow a retargeted checkpoint subdirectory."""
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.chart_storage import (
    ChartStorageError, load_chart_snapshot, save_chart_snapshot,
)
from nanlmind_hermes.head_bank_transport import BANK_OFFSET
from nanlmind_hermes.native_head_fence import (
    NativeHeadFenceError, SnapshotHeadBankFence,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root


class NestedCheckpointAliasTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        chart = Chart8MiB.genesis()
        attach_universe_root(chart)
        self.parent = chart.root_hash
        self.unchanged = bytes(chart.bytes)
        alternate = bytearray(self.unchanged)
        alternate[BANK_OFFSET] ^= 1
        self.alternate = bytes(alternate)
        self.universe_id = 51

    def test_nested_language_or_checkpoint_symlink_cannot_switch_fence(self):
        for component in ("language", "chart-checkpoints"):
            with self.subTest(component=component):
                real = self.base / ("real-" + component)
                decoy = self.base / ("decoy-" + component)
                self.assertEqual(
                    save_chart_snapshot(real, self.parent, self.unchanged, 0), 1
                )
                self.assertEqual(
                    save_chart_snapshot(decoy, self.parent, self.alternate, 0), 1
                )
                fence = SnapshotHeadBankFence(
                    state_dir=real, parent_root_sha256=self.parent,
                    universe_id=self.universe_id,
                )
                prior = fence(self.universe_id)
                self.assertNotEqual(
                    prior["bank_sha256"],
                    SnapshotHeadBankFence(
                        state_dir=decoy, parent_root_sha256=self.parent,
                        universe_id=self.universe_id,
                    )(self.universe_id)["bank_sha256"],
                )
                original = real / "language"
                foreign = decoy / "language"
                if component == "chart-checkpoints":
                    original /= component
                    foreign /= component
                parked = original.with_name(original.name + "-parked")
                original.rename(parked)
                original.symlink_to(foreign, target_is_directory=True)
                try:
                    with self.assertRaisesRegex(ChartStorageError, "symlinks"):
                        load_chart_snapshot(real, self.parent)
                    with self.assertRaisesRegex(
                        NativeHeadFenceError, "checkpoint unavailable"
                    ):
                        fence(self.universe_id)
                    with self.assertRaisesRegex(ChartStorageError, "symlinks"):
                        save_chart_snapshot(
                            real, self.parent, self.unchanged, 1,
                        )
                    self.assertEqual(
                        load_chart_snapshot(decoy, self.parent),
                        (1, self.alternate),
                    )
                finally:
                    original.unlink()
                    parked.rename(original)
                self.assertEqual(fence(self.universe_id), prior)

    def test_non_directory_checkpoint_components_refused_not_bootstrapped(self):
        for component in ("language", "chart-checkpoints"):
            with self.subTest(component=component):
                home = self.base / ("file-" + component)
                if component == "language":
                    home.mkdir()
                    bad = home / "language"
                else:
                    (home / "language").mkdir(parents=True)
                    bad = home / "language" / "chart-checkpoints"
                bad.write_bytes(b"foreign")
                with self.assertRaisesRegex(ChartStorageError, "not a directory"):
                    load_chart_snapshot(home, self.parent)
                with self.assertRaisesRegex(ChartStorageError, "not a directory"):
                    save_chart_snapshot(home, self.parent, self.unchanged, 0)
                self.assertEqual(bad.read_bytes(), b"foreign")


if __name__ == "__main__":
    unittest.main()
