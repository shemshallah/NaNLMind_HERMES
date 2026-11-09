"""GitHub-runnable crash/reopen checks for the unmerged 8 MiB NaN16 chart."""
import os
import struct
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.chart_storage import (
    ChartStorageError,
    HEADER,
    PAGE_BYTES,
    load_chart_snapshot,
    save_chart_snapshot,
)
from nanlmind_hermes.universe_chart import (
    OFF_EXTENSION,
    Chart8MiB,
    ChartError,
    LocalRunner,
    _RUNNERS,
    restore_local_chart,
)


class Context:
    def __init__(self, folder):
        self.folder = Path(folder)

    def get_config(self, key, default=None):
        return str(self.folder) if key == "state_dir" else default


class ChartSnapshotTests(unittest.TestCase):
    def test_pair_of_atomic_slots_and_hash_verified_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            chart = Chart8MiB.genesis()
            self.assertEqual(load_chart_snapshot(folder, chart.root_hash), (0, None))
            chart.bytes[OFF_EXTENSION + 2048:OFF_EXTENSION + 2052] = b"NaN!"
            gen = save_chart_snapshot(folder, chart.root_hash, chart.bytes, 0)
            self.assertEqual(gen, 1)
            self.assertEqual(load_chart_snapshot(folder, chart.root_hash)[1], bytes(chart.bytes))
            chart.bytes[OFF_EXTENSION + 2048:OFF_EXTENSION + 2052] = b"Na16"
            gen = save_chart_snapshot(folder, chart.root_hash, chart.bytes, gen)
            self.assertEqual(gen, 2)
            newest = folder / "language" / "chart-checkpoints" / "chart.1.n16s"
            raw = bytearray(newest.read_bytes())
            raw[HEADER.size + OFF_EXTENSION + 2048] ^= 0xFF
            newest.write_bytes(raw)  # torn/latest corrupt; earlier slot remains
            recovered_gen, recovered = load_chart_snapshot(folder, chart.root_hash)
            self.assertEqual(recovered_gen, 1)
            self.assertEqual(recovered[OFF_EXTENSION + 2048:OFF_EXTENSION + 2052], b"NaN!")
            with self.assertRaises(ChartStorageError):
                save_chart_snapshot(folder, chart.root_hash, chart.bytes, 2)
            self.assertEqual(save_chart_snapshot(folder, chart.root_hash, chart.bytes, 1), 2)

    def test_two_valid_same_generation_slots_must_not_fork_single_chart(self):
        # Both slots have intact headers and valid SHA-256 values, but are
        # separate histories claiming the SAME durable generation.
        import hashlib
        from nanlmind_hermes.chart_storage import MAGIC, SLOT_NAMES
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            chart = Chart8MiB.genesis()
            original = bytes(chart.bytes)
            self.assertEqual(save_chart_snapshot(folder, chart.root_hash, original, 0), 1)
            fork = bytearray(original)
            fork[OFF_EXTENSION + 2048] ^= 0x01
            root = folder / "language" / "chart-checkpoints"
            slot1 = root / SLOT_NAMES[1]
            slot1.write_bytes(
                HEADER.pack(MAGIC, 1, bytes.fromhex(chart.root_hash),
                            hashlib.sha256(fork).digest()) + fork
            )
            with self.assertRaisesRegex(ChartStorageError, "ambiguous same-generation"):
                load_chart_snapshot(folder, chart.root_hash)
            with self.assertRaisesRegex(ChartStorageError, "ambiguous same-generation"):
                save_chart_snapshot(folder, chart.root_hash, original, 1,
                                    expected_page_sha256=hashlib.sha256(original).hexdigest())
            # An identical duplicate of a valid generation is not a fork.
            slot1.write_bytes((root / SLOT_NAMES[0]).read_bytes())
            self.assertEqual(load_chart_snapshot(folder, chart.root_hash), (1, original))
            self.assertEqual(save_chart_snapshot(folder, chart.root_hash, original, 1), 2)

    def test_refuse_invalid_header_and_stale_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            chart = Chart8MiB.genesis()
            self.assertEqual(save_chart_snapshot(folder, chart.root_hash, chart.bytes, 0), 1)
            with self.assertRaisesRegex(ChartStorageError, "stale"):
                save_chart_snapshot(folder, chart.root_hash, chart.bytes, 0)
            with self.assertRaises(ChartStorageError):
                save_chart_snapshot(folder, "invalid", chart.bytes, 1)
            with self.assertRaises(ChartStorageError):
                save_chart_snapshot(folder, chart.root_hash, b"not a chart", 1)
            new_root = "f" * 64
            self.assertEqual(load_chart_snapshot(folder, new_root), (0, None))
            slot0 = folder / "language" / "chart-checkpoints" / "chart.0.n16s"
            slot0.write_bytes(b"short")
            with self.assertRaises(ChartStorageError):
                load_chart_snapshot(folder, chart.root_hash)

    def test_reopen_preserves_guest_rails_but_never_writes_canonical(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            key = str(folder)
            ctx = Context(folder)
            try:
                runner = LocalRunner.open(ctx)
                self.assertEqual(runner.chart.stats()["mode"], "genesis")
                runner.commit(7)
                runner.commit(39)
                before_frame = runner.register.frame
                before_rails = (runner.register.forward_payload,
                                runner.register.back_payload)
                checkpoint = runner.checkpoint()
                self.assertEqual(checkpoint["generation"], 1)
                self.assertFalse(checkpoint["canonical_training_committed"])
                _RUNNERS.pop(key, None)  # process-like restart, load from disk
                fresh = LocalRunner.open(ctx)
                self.assertEqual(fresh.chart.snapshot_generation, 1)
                self.assertEqual(fresh.register.frame, before_frame)
                self.assertEqual((fresh.register.forward_payload,
                                  fresh.register.back_payload), before_rails)
                self.assertEqual(fresh.chart.bytes[OFF_EXTENSION:OFF_EXTENSION + 8],
                                 b"NLMREG01")
                fresh.commit(111)
                self.assertEqual(fresh.checkpoint()["generation"], 2)
                _RUNNERS.pop(key, None)
                again = LocalRunner.open(ctx)
                self.assertEqual(again.register.frame, before_frame + 1)
                self.assertEqual(again.chart.snapshot_generation, 2)
                self.assertFalse((folder / "universe" / "native" / "global_universe.nmg").exists())
            finally:
                _RUNNERS.pop(key, None)

    def test_refuse_snapshot_that_mutates_immutable_address_table(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            chart = Chart8MiB.genesis()
            chart.bytes[0x1000:0x1002] = b"\xFF\xFF"  # index table
            save_chart_snapshot(folder, chart.root_hash, chart.bytes, 0)
            with self.assertRaisesRegex(ChartError, "immutable"):
                restore_local_chart(Chart8MiB.genesis(), folder)

    def test_snapshot_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            chart = Chart8MiB.genesis()
            save_chart_snapshot(folder, chart.root_hash, chart.bytes, 0)
            root = folder / "language" / "chart-checkpoints"
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            slot = root / "chart.0.n16s"
            self.assertEqual(slot.stat().st_mode & 0o777, 0o600)


    def test_plugin_storage_inspect_and_explicit_checkpoint(self):
        from nanlmind_hermes.tools import chart_storage_tool
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            key = str(folder)
            ctx = Context(folder)
            try:
                initial = chart_storage_tool(ctx, "inspect")
                self.assertEqual(initial["disk_generation"], 0)
                self.assertFalse(initial["canonical_training_committed"])
                self.assertFalse(initial["native_head_bank_attached"])
                with self.assertRaisesRegex(ValueError, "approved"):
                    chart_storage_tool(ctx, "checkpoint", False)
                committed = chart_storage_tool(ctx, "checkpoint", True)
                self.assertEqual(committed["generation"], 1)
                self.assertTrue(committed["storage_only"])
                self.assertFalse(committed["nannet_merge_issued"])
                recovered = chart_storage_tool(ctx, "inspect")
                self.assertEqual(recovered["disk_generation"], 1)
                self.assertEqual(recovered["recovery"],
                                 "dual-slot-sha256-generation-fenced")
                with self.assertRaises(ValueError):
                    chart_storage_tool(ctx, "restore")
            finally:
                _RUNNERS.pop(key, None)


if __name__ == "__main__":
    unittest.main()
