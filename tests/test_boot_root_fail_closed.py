"""HERMES plugin startup must never hide broken native universe authority."""
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes import register
from nanlmind_hermes.chart_storage import (
    ChartStorageError, save_chart_snapshot,
)
from nanlmind_hermes.universe_chart import Chart8MiB, LocalRunner
from nanlmind_hermes.universe_root import (
    ROOT_OFFSET, UniverseRootError, attach_universe_root, read_universe_root,
)


class TestContext:
    def __init__(self, directory):
        self.directory = directory
        self.tools_registered = []

    def get_config(self, key, default=None):
        return str(self.directory) if key == "state_dir" else default

    def register_tool(self, *args, **kwargs):
        self.tools_registered.append((args, kwargs))


class PluginBootRootTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        self.ctx = TestContext(self.state)

    def test_new_universe_boots_one_mounted_genesis_root(self):
        runner = LocalRunner.open(self.ctx)
        root = read_universe_root(runner.chart)
        self.assertTrue(root["mounted"])
        self.assertEqual(root["head_addresses"], 256)
        self.assertEqual(root["mirror_points"], 32640)
        self.assertEqual(root["l0_l9_layers"], list(range(10)))

    def test_valid_snapshot_sha_cannot_mask_corrupt_native_root(self):
        chart = Chart8MiB.genesis()
        attach_universe_root(chart)
        parent = chart.root_hash
        chart.bytes[ROOT_OFFSET + 8] ^= 0x01
        self.assertEqual(
            save_chart_snapshot(self.state, parent, chart.bytes, 0), 1,
        )
        with self.assertRaises(UniverseRootError):
            register(self.ctx)
        self.assertEqual(self.ctx.tools_registered, [])

    def test_invalid_snapshot_slot_must_abort_plugin_registration(self):
        folder = self.state / "language" / "chart-checkpoints"
        folder.mkdir(parents=True)
        (folder / "chart.0.n16s").write_bytes(b"corrupted-snapshot")
        with self.assertRaises(ChartStorageError):
            register(self.ctx)
        self.assertEqual(self.ctx.tools_registered, [])


if __name__ == "__main__":
    unittest.main()
