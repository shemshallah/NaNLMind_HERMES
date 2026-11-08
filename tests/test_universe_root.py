"""Root block is one binary mount, survives crash restore, fails closed."""
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.universe_chart import (
    Chart8MiB, LocalRunner, _RUNNERS, restore_local_chart,
)
from nanlmind_hermes.universe_root import (
    ROOT_BYTES, ROOT_END, ROOT_OFFSET, UniverseRootError,
    attach_universe_root, read_universe_root,
)
from nanlmind_hermes.head_bank_transport import BANK_OFFSET
from nanlmind_hermes.chart_storage import save_chart_snapshot


class Ctx:
    def __init__(self, path):
        self.path = str(path)

    def get_config(self, name, default=None):
        return self.path if name == "state_dir" else default


class BinaryUniverseRootTests(unittest.TestCase):
    def test_root_instantiated_inside_original_chart_without_duplicate_db(self):
        chart = Chart8MiB.genesis()
        self.assertEqual(read_universe_root(chart)["mounted"], False)
        self.assertEqual(ROOT_BYTES, 4096)
        self.assertLessEqual(ROOT_END, BANK_OFFSET)
        first = attach_universe_root(chart)
        self.assertTrue(first["mounted"])
        self.assertEqual(first["mirror_points"], 32640)
        self.assertEqual(first["head_addresses"], 256)
        self.assertEqual(first["l0_l9_layers"], list(range(10)))
        self.assertFalse(first["router_topology_attached"])
        self.assertFalse(first["native_guest_attached"])
        self.assertEqual(first, attach_universe_root(chart))
        self.assertEqual(len(chart.bytes), 8 * 1024 * 1024)

    def test_bad_nonzero_header_and_mixed_parent_denied_without_reformat(self):
        chart = Chart8MiB.genesis()
        chart.bytes[ROOT_OFFSET] = 1
        before = bytes(chart.bytes[ROOT_OFFSET:ROOT_END])
        with self.assertRaises(UniverseRootError):
            attach_universe_root(chart)
        self.assertEqual(before, bytes(chart.bytes[ROOT_OFFSET:ROOT_END]))
        other = Chart8MiB.genesis()
        attach_universe_root(other)
        other.root_hash = "f" * 64
        with self.assertRaises(UniverseRootError):
            read_universe_root(other)

    def test_runner_boot_checkpoint_and_restore_preserve_root(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = Ctx(folder)
            key = str(Path(folder))
            try:
                runner = LocalRunner.open(ctx)
                initial = read_universe_root(runner.chart)
                self.assertTrue(initial["mounted"])
                self.assertTrue(runner.chart._dirty)
                self.assertEqual(runner.status()["universe_root"]["mount_block_sha256"],
                                 initial["mount_block_sha256"])
                self.assertEqual(runner.checkpoint()["generation"], 1)
                _RUNNERS.pop(key, None)
                fresh = LocalRunner.open(ctx)
                self.assertFalse(fresh.chart._dirty)
                self.assertEqual(read_universe_root(fresh.chart)["mount_block_sha256"],
                                 initial["mount_block_sha256"])
                self.assertEqual(fresh.chart.snapshot_generation, 1)
            finally:
                _RUNNERS.pop(key, None)

    def test_root_survives_raw_snapshot_and_corrupt_header_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            chart = Chart8MiB.genesis()
            attach_universe_root(chart)
            save_chart_snapshot(Path(folder), chart.root_hash, chart.bytes, 0)
            restored = Chart8MiB.genesis()
            self.assertEqual(restore_local_chart(restored, folder), 1)
            self.assertTrue(read_universe_root(restored)["mounted"])
            restored.bytes[ROOT_OFFSET + 4] ^= 0xFF
            with self.assertRaises(UniverseRootError):
                attach_universe_root(restored)


if __name__ == "__main__":
    unittest.main()
