"""Tests for the bounded resident superblock/layout."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident import (
    ALIGN,
    MAGIC,
    MAX_BYTES,
    MIN_BYTES,
    RESIDENT_VERSION,
    SECTION_ARENA,
    SECTION_CHART,
    SECTION_HEADS,
    align,
)
from nanlmind_hermes.resident.layout import (
    DATA_OFFSET,
    HEADER_SLOT_SIZE,
    LayoutCorruptError,
    LayoutFullError,
    ResidentLayout,
)

MIB = 1024 * 1024
SIZE_32MIB = 32 * MIB


def _raw_write(path: Path, offset: int, data: bytes) -> None:
    with open(path, "r+b") as handle:
        handle.seek(offset)
        handle.write(data)


class ResidentLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_is_sparse_and_allocate_is_sequential_and_aligned(self):
        path = self.tmpdir / "universe.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        try:
            stat = path.stat()
            self.assertEqual(stat.st_size, SIZE_32MIB)
            self.assertLess(stat.st_blocks * 512, stat.st_size)
            self.assertLessEqual(stat.st_size, MAX_BYTES)

            header = layout.header()
            self.assertEqual(header.magic, MAGIC)
            self.assertEqual(header.version, RESIDENT_VERSION)
            self.assertEqual(header.total_size, SIZE_32MIB)
            self.assertEqual(header.data_offset, DATA_OFFSET)
            self.assertEqual(header.generation, 0)
            self.assertGreaterEqual(header.data_offset, 2 * HEADER_SLOT_SIZE)
            self.assertEqual(len(header.manifest_digest), 32)

            initial_digest = layout.manifest_digest
            first = layout.allocate(SECTION_CHART, 100)
            second = layout.allocate(SECTION_CHART, 200)
            third = layout.allocate(SECTION_HEADS, 64)

            self.assertEqual(first, DATA_OFFSET)
            self.assertEqual(first % ALIGN, 0)
            self.assertEqual(second, first + align(100))
            self.assertEqual(second % ALIGN, 0)
            self.assertEqual(third, second + align(200))
            self.assertEqual(third % ALIGN, 0)
            self.assertNotEqual(layout.manifest_digest, initial_digest)

            chart = layout.section(SECTION_CHART)
            self.assertEqual(chart.offset, first)
            self.assertEqual(chart.used, align(100) + align(200))
            self.assertEqual(chart.size, chart.used)
            self.assertEqual(chart.generation, 2)
            self.assertEqual(chart.extents, ((first, align(100) + align(200)),))

            heads = layout.section(SECTION_HEADS)
            self.assertEqual(heads.offset, third)
            self.assertEqual(heads.used, align(64))
            self.assertEqual(heads.generation, layout.generation)

            self.assertEqual(layout.generation, 3)
            self.assertEqual(header.generation + 3, layout.generation)
            self.assertEqual(
                layout.used_bytes, align(100) + align(200) + align(64))
            self.assertEqual(
                layout.free_bytes,
                SIZE_32MIB - layout.header().data_offset - layout.used_bytes)
            self.assertGreaterEqual(layout.used_bytes, 364)
        finally:
            layout.close()

    def test_flush_reopen_preserves_offsets_generation_and_digest(self):
        path = self.tmpdir / "universe.nlr"
        with ResidentLayout.create(path, size=SIZE_32MIB) as layout:
            chart_offset = layout.allocate(SECTION_CHART, 128)
            arena_offset = layout.allocate(SECTION_ARENA, 512)
            layout.append(SECTION_HEADS, b"head-data")
            layout.flush()

            saved_header = layout.header()
            saved_chart = layout.section(SECTION_CHART)
            saved_arena = layout.section(SECTION_ARENA)
            saved_manifest = layout.manifest_digest

        self.assertEqual(len(saved_manifest), 32)

        reopened = ResidentLayout.open(path, size=SIZE_32MIB)
        try:
            header = reopened.header()
            self.assertEqual(header.magic, MAGIC)
            self.assertEqual(header.version, RESIDENT_VERSION)
            self.assertEqual(header.generation, saved_header.generation)
            self.assertEqual(header.total_size, SIZE_32MIB)
            self.assertEqual(header.manifest_digest, saved_manifest)
            self.assertEqual(reopened.manifest_digest, saved_manifest)

            chart = reopened.section(SECTION_CHART)
            self.assertEqual(chart.offset, chart_offset)
            self.assertEqual(chart.used, saved_chart.used)
            self.assertEqual(chart.generation, saved_chart.generation)
            self.assertEqual(chart.digest, saved_chart.digest)

            arena = reopened.section(SECTION_ARENA)
            self.assertEqual(arena.offset, arena_offset)
            self.assertEqual(arena.used, saved_arena.used)
            self.assertEqual(arena.generation, saved_arena.generation)

            heads = reopened.section(SECTION_HEADS)
            self.assertEqual(reopened.read_at(heads.offset, 9), b"head-data")

            # The append cursor survived, so a new allocation starts exactly
            # after all previously used bytes, regardless of which sections
            # produced them.
            used_before = reopened.used_bytes
            following = reopened.allocate(SECTION_CHART, 32)
            self.assertEqual(following, DATA_OFFSET + used_before)
            self.assertGreater(reopened.generation, saved_header.generation)
            self.assertEqual(
                reopened.section(SECTION_CHART).used,
                saved_chart.used + align(32))
        finally:
            reopened.close()

    def test_reject_requested_size_above_one_gib(self):
        path = self.tmpdir / "too-big.nlr"
        with self.assertRaises(ValueError):
            ResidentLayout.create(path, size=MAX_BYTES + 1)
        self.assertFalse(path.exists())

        # An over-ceiling file is rejected before it is mapped.
        oversized = self.tmpdir / "oversized.nlr"
        oversized.write_bytes(b"")
        os.truncate(oversized, MAX_BYTES + 1)
        with self.assertRaises(ValueError):
            ResidentLayout.open(oversized)

        # The minimum is enforced as well.
        with self.assertRaises(ValueError):
            ResidentLayout.create(self.tmpdir / "tiny.nlr", size=MIN_BYTES - 1)

    def test_reject_corrupt_magic_in_both_header_copies(self):
        path = self.tmpdir / "corrupt-magic.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        layout.allocate(SECTION_CHART, 64)
        layout.flush()
        layout.close()

        bad = b"BROKEN!!"
        _raw_write(path, 0, bad)
        _raw_write(path, HEADER_SLOT_SIZE, bad)
        with self.assertRaisesRegex(LayoutCorruptError, "magic"):
            ResidentLayout.open(path)

    def test_torn_header_recovers_valid_copy(self):
        for damaged_slot in (0, 1):
            with self.subTest(damaged_slot=damaged_slot):
                path = self.tmpdir / ("torn-%d.nlr" % damaged_slot)
                layout = ResidentLayout.create(path, size=SIZE_32MIB)
                offset = layout.allocate(SECTION_CHART, 256)
                layout.flush()
                expected_generation = layout.generation
                expected_used = layout.section(SECTION_CHART).used
                layout.close()

                # Damage the checksummed body of one copy.  The other copy is
                # complete and must be selected.
                _raw_write(
                    path,
                    damaged_slot * HEADER_SLOT_SIZE + 512,
                    b"\xff" * 64,
                )
                with ResidentLayout.open(path) as recovered:
                    self.assertEqual(recovered.generation, expected_generation)
                    self.assertEqual(
                        recovered.section(SECTION_CHART).offset, offset)
                    self.assertEqual(
                        recovered.section(SECTION_CHART).used, expected_used)

                # Opening healed the damaged slot, so a second reopen works.
                with ResidentLayout.open(path) as healed:
                    self.assertEqual(healed.generation, expected_generation)

    def test_fail_closed_when_both_header_copies_are_torn(self):
        path = self.tmpdir / "both-torn.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        layout.allocate(SECTION_ARENA, 1)
        layout.flush()
        layout.close()

        for slot in (0, 1):
            _raw_write(path, slot * HEADER_SLOT_SIZE + 1024, b"\xff" * 64)
        with self.assertRaises(LayoutCorruptError):
            ResidentLayout.open(path)

    def test_stale_but_valid_header_copy_does_not_override_newer(self):
        path = self.tmpdir / "stale-copy.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        layout.allocate(SECTION_CHART, 64)
        layout.flush()
        with open(path, "rb") as handle:
            handle.seek(HEADER_SLOT_SIZE)
            old_copy = handle.read(HEADER_SLOT_SIZE)

        layout.allocate(SECTION_CHART, 64)
        layout.flush()
        generation = layout.generation
        used = layout.section(SECTION_CHART).used
        layout.close()

        # Restore the older but checksum-valid second copy.  The newer first
        # copy must still win.
        _raw_write(path, HEADER_SLOT_SIZE, old_copy)
        with ResidentLayout.open(path) as reopened:
            self.assertEqual(reopened.generation, generation)
            self.assertEqual(reopened.section(SECTION_CHART).used, used)

    def test_manifest_detects_data_corruption(self):
        path = self.tmpdir / "corrupt-data.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        offset = layout.allocate(SECTION_ARENA, 64)
        layout.flush()
        layout.close()

        _raw_write(path, offset + 7, b"\x01")
        with self.assertRaises(LayoutCorruptError):
            ResidentLayout.open(path)

    def test_allocation_is_bounded_and_fails_when_full(self):
        path = self.tmpdir / "bounded.nlr"
        layout = ResidentLayout.create(path, size=SIZE_32MIB)
        chunk = MIB
        allocations = 0
        try:
            while True:
                try:
                    offset = layout.allocate(SECTION_ARENA, chunk)
                except LayoutFullError:
                    break
                allocations += 1
                self.assertEqual(offset % ALIGN, 0)
                self.assertLessEqual(offset + align(chunk), layout.size)
                self.assertLessEqual(layout.used_bytes, layout.size)

            self.assertGreater(allocations, 0)
            self.assertLess(layout.free_bytes, align(chunk))
            self.assertLessEqual(layout.used_bytes, layout.size)
            self.assertEqual(
                layout.section(SECTION_ARENA).used, allocations * align(chunk))
            self.assertEqual(layout.generation, allocations)
            self.assertEqual(
                layout.header().generation, allocations)

            # A request larger than the whole universe fails without mutating
            # the layout.
            used_before = layout.used_bytes
            generation_before = layout.generation
            with self.assertRaises((LayoutFullError, MemoryError, ValueError)):
                layout.allocate(SECTION_ARENA, SIZE_32MIB)
            self.assertEqual(layout.used_bytes, used_before)
            self.assertEqual(layout.generation, generation_before)
        finally:
            layout.close()

    def test_zero_length_allocation_does_not_advance_generation(self):
        path = self.tmpdir / "zero.nlr"
        with ResidentLayout.create(path, size=SIZE_32MIB) as layout:
            generation = layout.generation
            offset = layout.allocate(SECTION_CHART, 0)
            self.assertEqual(offset, layout.header().data_offset)
            self.assertEqual(layout.generation, generation)
            self.assertEqual(layout.section(SECTION_CHART).used, 0)

    def test_context_manager_and_idempotent_close(self):
        path = self.tmpdir / "context.nlr"
        with ResidentLayout.create(path, size=SIZE_32MIB) as layout:
            layout.allocate(SECTION_CHART, 8)
        layout.close()
        self.assertFalse(layout.is_open)


if __name__ == "__main__":
    unittest.main()
