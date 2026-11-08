"""Opaque native NaN16 bank chart transport: layout, leases and crash restore."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.head_bank_transport import (
    BANK_BYTES, BANK_END, BANK_GUEST_WORD_OFFSET, BANK_OFFSET,
    HeadBankTransportError, describe_guest_bank, export_guest_bank_bytes,
    stage_guest_bank_bytes,
)
from nanlmind_hermes.universe_chart import (
    Chart8MiB, HEAD_REGISTER_BYTES, HEAD_REGISTER_OFFSET, OFF_HEAD_SCRATCH,
    restore_local_chart,
)
from nanlmind_hermes.chart_storage import save_chart_snapshot


class HeadBankTransportTests(unittest.TestCase):
    def test_layout_opaque_roundtrip_and_original_head_scratch_untouched(self):
        chart = Chart8MiB.genesis()
        before_scratch = bytes(chart.bytes[OFF_HEAD_SCRATCH:OFF_HEAD_SCRATCH + 256 * 1024])
        self.assertGreaterEqual(BANK_OFFSET, HEAD_REGISTER_OFFSET + HEAD_REGISTER_BYTES)
        self.assertLessEqual(BANK_END, len(chart.bytes))
        self.assertEqual(BANK_GUEST_WORD_OFFSET * 8, BANK_OFFSET)
        self.assertEqual(BANK_BYTES, 67838 * 8)
        initial = describe_guest_bank(chart)
        self.assertFalse(initial["has_bytes"])
        self.assertFalse(initial["native_guest_attached"])
        self.assertEqual(initial["heads"], 255)
        raw = bytes((i * 17 + 23) % 256 for i in range(BANK_BYTES))
        staged = stage_guest_bank_bytes(
            chart, raw, payload_sha256=hashlib.sha256(raw).hexdigest(),
            expected_region_sha256=initial["region_sha256"],
            expected_parent_root=chart.root_hash, expected_snapshot_generation=0,
        )
        self.assertTrue(staged["has_bytes"])
        self.assertFalse(staged["format_checked_by_native_guest"])
        self.assertEqual(export_guest_bank_bytes(chart), raw)
        self.assertEqual(before_scratch,
                         bytes(chart.bytes[OFF_HEAD_SCRATCH:OFF_HEAD_SCRATCH + 256 * 1024]))
        self.assertTrue(chart._dirty)

    def test_stale_root_generation_digest_and_invalid_length_atomicity(self):
        chart = Chart8MiB.genesis()
        initial = describe_guest_bank(chart)
        raw = bytes([0xA6]) * BANK_BYTES
        base = dict(payload_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_region_sha256=initial["region_sha256"],
                    expected_parent_root=chart.root_hash,
                    expected_snapshot_generation=0)
        for variant, change in (
                ("wrong size", {"payload": raw[:-1]}),
                ("wrong digest", {"payload_sha256": "f" * 64}),
                ("wrong root", {"expected_parent_root": "0" * 64}),
                ("stale generation", {"expected_snapshot_generation": 1}),
                ("stale bank digest", {"expected_region_sha256": "e" * 64})):
            with self.subTest(variant=variant):
                kwargs = dict(base)
                payload = change.pop("payload", raw) if "payload" in change else raw
                kwargs.update(change)
                with self.assertRaises(HeadBankTransportError):
                    stage_guest_bank_bytes(chart, payload, **kwargs)
                self.assertEqual(describe_guest_bank(chart)["region_sha256"],
                                 initial["region_sha256"])
        self.assertFalse(chart._dirty is False and any(export_guest_bank_bytes(chart)))

    def test_full_chart_sha256_checkpoint_recovers_opaque_native_region(self):
        with tempfile.TemporaryDirectory() as temp:
            chart = Chart8MiB.genesis()
            initial = describe_guest_bank(chart)
            raw = bytes([0x11]) * BANK_BYTES
            stage_guest_bank_bytes(
                chart, raw, payload_sha256=hashlib.sha256(raw).hexdigest(),
                expected_region_sha256=initial["region_sha256"],
                expected_parent_root=chart.root_hash, expected_snapshot_generation=0,
            )
            generation = save_chart_snapshot(Path(temp), chart.root_hash, chart.bytes, 0)
            self.assertEqual(generation, 1)
            restored = Chart8MiB.genesis()
            self.assertEqual(restore_local_chart(restored, temp), 1)
            self.assertEqual(export_guest_bank_bytes(restored), raw)
            self.assertFalse(describe_guest_bank(restored)["native_guest_attached"])


if __name__ == "__main__":
    unittest.main()
