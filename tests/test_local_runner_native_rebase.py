"""One live HERMES chart, preserving dirty MP/temporal edits across NaN16 head commit.

The journal stand-in tests only the local in-memory rebase; actual signed
leases and real NaN16 head instructions run in the three-repository CI.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.chart_storage import load_chart_snapshot, save_chart_snapshot
from nanlmind_hermes.head_bank_transport import BANK_BYTES, BANK_OFFSET, BANK_END
from nanlmind_hermes.native_head_checkpoint import NativeHeadCheckpointCoordinator
from nanlmind_hermes.native_head_fence import SnapshotHeadBankFence
from nanlmind_hermes.universe_chart import (
    Chart8MiB, ChartError, LocalRunner, TemporalRegister, make_word,
)
from nanlmind_hermes.universe_root import attach_universe_root


def digest(value):
    return hashlib.sha256(value).hexdigest()


class TestContext:
    def __init__(self, state):
        self.state = state

    def get_config(self, key, default=None):
        return str(self.state) if key == "state_dir" else default


class VerifiedJournalDouble:
    """Persistent-result shape for storage tests; not a real lease signer."""

    def __init__(self, fence, universe, after):
        self.fence = fence
        self.universe = universe
        self.after = after
        certified = fence(universe)
        self.source_chart_sha256 = certified["chart_sha256"]
        self.source_root_block_sha256 = certified["root_block_sha256"]
        self.epoch = 19
        self.ring_epoch_reader = lambda: self.epoch
        self.state = "completed_staging_only"
        self.confirms = 0

    def inspect(self, *, request, lease_receipt, network_head, include_bank):
        if lease_receipt["universe_id"] != self.universe or network_head != 0:
            raise AssertionError("wrong journal scope")
        return {
            "state": self.state,
            "bank_after_sha256": digest(self.after),
            "source_snapshot_bound": True,
            "source_chart_sha256": self.source_chart_sha256,
            "source_root_block_sha256": self.source_root_block_sha256,
            "raw_bank": self.after if include_bank else None,
            "request_sha256": digest(b"test-journal-request"),
            "checkpoint_generation": (
                request.chart_generation + 1
                if self.state == "checkpointed_locally" else None
            ),
            "local_checkpoint_verified": self.state == "checkpointed_locally",
            "canonical_training_committed": False,
        }

    def confirm_local_checkpoint(self, *, request, lease_receipt, network_head):
        self.confirms += 1
        latest = self.fence(self.universe)
        assert latest["chart_generation"] == request.chart_generation + 1
        assert latest["bank_sha256"] == digest(self.after)
        self.state = "checkpointed_locally"
        return {
            "local_checkpoint_verified": True,
            "bank_after_sha256": digest(self.after),
            "checkpoint_generation": latest["chart_generation"],
            "request_sha256": digest(b"test-journal-request"),
        }


class SingleLiveRunnerRebaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name)
        self.ctx = TestContext(self.state)
        chart = Chart8MiB.genesis()
        attach_universe_root(chart)
        self.parent = chart.root_hash
        register = TemporalRegister.open(self.ctx, chart)
        self.runner = LocalRunner(chart, register)
        self.assertEqual(
            save_chart_snapshot(self.state, self.parent, chart.bytes, 0), 1
        )
        chart.snapshot_generation = 1
        chart._dirty = False
        self.universe = 4455
        self.fence = SnapshotHeadBankFence(
            state_dir=self.state, parent_root_sha256=self.parent,
            universe_id=self.universe,
        )
        self.raw = bytes([0xE4]) * BANK_BYTES
        self.journal = VerifiedJournalDouble(self.fence, self.universe, self.raw)
        self.coordinator = NativeHeadCheckpointCoordinator(
            fence=self.fence, journal=self.journal,
        )
        self.request = SimpleNamespace(
            parent_root_sha256=self.parent,
            chart_generation=1,
            ring_epoch=19,
            bank_before_sha256=digest(bytes(BANK_BYTES)),
        )
        self.receipt = {"universe_id": self.universe}

    def rebase(self, **kwargs):
        return self.runner.commit_native_head_observation(
            coordinator=kwargs.get("coordinator", self.coordinator),
            request=kwargs.get("request", self.request),
            lease_receipt=kwargs.get("lease_receipt", self.receipt),
            network_head=kwargs.get("network_head", 0),
        )

    def test_native_bank_rebases_only_bank_and_preserves_single_live_chart(self):
        original = self.runner.chart
        original_register = self.runner.register.chart
        original_heads = self.runner.heads.chart
        self.runner.chart.write_rail(17, 136, 0, make_word(0x1234))
        before_splus = self.runner.chart.read_rail(17, 136, 0)
        before_nonbank = (
            bytes(self.runner.chart.bytes[:BANK_OFFSET]),
            bytes(self.runner.chart.bytes[BANK_END:]),
        )
        result = self.rebase()
        self.assertTrue(result["head_bank_rebased"])
        self.assertTrue(result["single_chart_object_preserved"])
        self.assertTrue(result["pending_nonbank_edits_preserved"])
        self.assertFalse(result["canonical_training_committed"])
        self.assertEqual(self.runner.chart.snapshot_generation, 2)
        self.assertIs(self.runner.chart, original)
        self.assertIs(self.runner.register.chart, original_register)
        self.assertIs(self.runner.heads.chart, original_heads)
        self.assertEqual(self.runner.chart.read_rail(17, 136, 0), before_splus)
        self.assertEqual(
            bytes(self.runner.chart.bytes[:BANK_OFFSET]), before_nonbank[0]
        )
        self.assertEqual(
            bytes(self.runner.chart.bytes[BANK_END:]), before_nonbank[1]
        )
        self.assertEqual(
            bytes(self.runner.chart.bytes[BANK_OFFSET:BANK_END]), self.raw
        )
        self.assertTrue(self.runner.chart._dirty)
        # Local dirty rails can now be checkpointed on top of gen2 without
        # losing the newly learned native head update.
        later = self.runner.checkpoint()
        self.assertEqual(later["generation"], 3)
        _, stored = load_chart_snapshot(self.state, self.parent)
        self.assertEqual(
            stored[BANK_OFFSET:BANK_END], self.raw
        )
        self.assertEqual(
            stored[self.runner.chart.cell_offset(17, 136):][0:8],
            make_word(0x1234).to_bytes(8, "little"),
        )

    def test_followup_checkpoint_rejects_same_generation_page_replacement(self):
        """Dirty rails must not overwrite another valid same-generation page."""
        from nanlmind_hermes.chart_storage import (
            ChartStorageError, HEADER, MAGIC, SLOT_NAMES,
        )
        self.runner.chart.write_rail(17, 136, 0, make_word(0x2222))
        committed = self.rebase()
        self.assertEqual(committed["chart_generation"], 2)
        self.assertTrue(self.runner.chart._dirty)
        stored_generation, stored = load_chart_snapshot(self.state, self.parent)
        self.assertEqual(stored_generation, 2)
        # Model a valid but different N16SNAP1 page at the same generation.
        # Generation-only CAS would accept and overwrite this page.
        changed = bytearray(stored)
        changed[0x7000] ^= 0x01
        folder = self.state / "language" / "chart-checkpoints"
        newest = folder / SLOT_NAMES[1]
        newest.write_bytes(
            HEADER.pack(
                MAGIC, 2, bytes.fromhex(self.parent),
                hashlib.sha256(changed).digest(),
            ) + changed
        )
        self.assertEqual(load_chart_snapshot(self.state, self.parent), (2, bytes(changed)))
        with self.assertRaisesRegex(ChartStorageError, "page digest"):
            self.runner.checkpoint()
        self.assertEqual(self.runner.chart.snapshot_generation, 2)
        self.assertTrue(self.runner.chart._dirty)
        self.assertEqual(load_chart_snapshot(self.state, self.parent), (2, bytes(changed)))

    def test_restored_checkpoint_uses_verified_source_page_digest(self):
        first = self.rebase()
        self.assertTrue(first["head_bank_rebased"])
        fresh_chart = Chart8MiB.genesis()
        from nanlmind_hermes.universe_chart import restore_local_chart
        self.assertEqual(restore_local_chart(fresh_chart, self.state), 2)
        self.assertEqual(
            fresh_chart._checkpoint_page_sha256,
            self.fence(self.universe)["chart_sha256"],
        )
        fresh_runner = LocalRunner(
            fresh_chart, TemporalRegister.open(self.ctx, fresh_chart),
        )
        fresh_runner.chart.write_rail(17, 136, 0, make_word(0x4321))
        committed = fresh_runner.checkpoint()
        self.assertEqual(committed["generation"], 3)
        self.assertEqual(
            fresh_runner.chart._checkpoint_page_sha256,
            self.fence(self.universe)["chart_sha256"],
        )

    def test_symlink_and_real_state_dir_share_single_cached_chart(self):
        """Two names for one storage home must not create two minds."""
        from nanlmind_hermes.universe_chart import _RUNNERS
        real = self.state / "same-universe-real"
        real.mkdir()
        alias = self.state / "same-universe-link"
        alias.symlink_to(real, target_is_directory=True)
        key = str(real.resolve())
        _RUNNERS.pop(key, None)
        try:
            first = LocalRunner.open(TestContext(real))
            second = LocalRunner.open(TestContext(alias))
            self.assertIs(first, second)
            self.assertIs(first.chart, second.chart)
            self.assertIs(first.register.chart, first.chart)
            self.assertIs(first.heads.chart, first.chart)
            self.assertEqual(len(first.chart.bytes), 8 * 1024 * 1024)
            self.assertIs(_RUNNERS[key], first)
            # The writer's own checkpoint sequence remains shared.
            generation = first.checkpoint()["generation"]
            self.assertEqual(generation, 1)
            self.assertEqual(second.chart.snapshot_generation, generation)
        finally:
            _RUNNERS.pop(key, None)

    def test_boot_via_symlink_cannot_redirect_state_during_retarget(self):
        """An alias moved mid-open cannot split the cached chart from its IO."""
        from unittest.mock import patch
        from nanlmind_hermes.universe_chart import _RUNNERS
        real = self.state / "pinned-universe-real"
        decoy = self.state / "pinned-universe-other"
        real.mkdir()
        decoy.mkdir()
        alias = self.state / "pinned-universe-link"
        alias.symlink_to(real, target_is_directory=True)
        key = str(real.resolve())
        _RUNNERS.pop(key, None)
        try:
            real_genesis = Chart8MiB.genesis

            def move_alias_during_boot():
                # LocalRunner has already resolved the real cache key, but
                # the register has not yet been instantiated.
                alias.unlink()
                alias.symlink_to(decoy, target_is_directory=True)
                return real_genesis()

            with patch.object(Chart8MiB, "genesis", side_effect=move_alias_during_boot):
                runner = LocalRunner.open(TestContext(alias))
            self.assertEqual(alias.resolve(), decoy.resolve())
            self.assertEqual(runner.register.state_dir, real.resolve())
            self.assertEqual(runner.heads.state_dir, real.resolve())
            # Sidecars and the durable chart checkpoint must remain under the
            # original resolved directory, not follow the changed alias.
            runner.register.save()
            result = runner.checkpoint()
            self.assertEqual(result["generation"], 1)
            self.assertTrue(
                (real / "language" / "temporal_register.json").is_file()
            )
            self.assertTrue(
                (real / "language" / "chart-checkpoints" / "chart.0.n16s").is_file()
            )
            self.assertFalse((decoy / "language").exists())
            self.assertIs(LocalRunner.open(TestContext(real)), runner)
            self.assertEqual(runner.chart.snapshot_generation, 1)
        finally:
            _RUNNERS.pop(key, None)

    def test_concurrent_plugin_opens_share_one_chart_object(self):
        """Two simultaneous HERMES boots must not split one state directory."""
        import threading
        from unittest.mock import patch
        from nanlmind_hermes.universe_chart import _RUNNERS
        state = self.state / "parallel-plugin-open"
        ctx = TestContext(state)
        cache_key = str(state)
        _RUNNERS.pop(cache_key, None)
        genesis_entered = threading.Event()
        release_genesis = threading.Event()
        second_attempted = threading.Event()
        second_done = threading.Event()
        built = []
        errors = []
        genesis_calls = [0]
        original_genesis = Chart8MiB.genesis

        def slow_genesis():
            genesis_calls[0] += 1
            genesis_entered.set()
            if not release_genesis.wait(10):
                raise RuntimeError("cache startup test barrier timed out")
            return original_genesis()

        def open_runner(second=False):
            if second:
                second_attempted.set()
            try:
                built.append(LocalRunner.open(ctx))
            except BaseException as exc:
                errors.append(exc)
            finally:
                if second:
                    second_done.set()

        first = threading.Thread(target=open_runner)
        second = threading.Thread(target=lambda: open_runner(True))
        try:
            with patch.object(Chart8MiB, "genesis", side_effect=slow_genesis):
                first.start()
                self.assertTrue(genesis_entered.wait(10))
                second.start()
                self.assertTrue(second_attempted.wait(10))
                self.assertFalse(second_done.wait(0.25))
                release_genesis.set()
                first.join(15)
                second.join(15)
        finally:
            release_genesis.set()
            first.join(15) if first.ident is not None else None
            second.join(15) if second.ident is not None else None
            _RUNNERS.pop(cache_key, None)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(genesis_calls[0], 1)
        self.assertEqual(len(built), 2)
        self.assertIs(built[0], built[1])
        self.assertIs(built[0].chart, built[1].chart)
        self.assertIs(built[0].register.chart, built[0].chart)
        self.assertIs(built[0].heads.chart, built[0].chart)

    def test_parallel_native_rebase_and_ordinary_checkpoint_serialize(self):
        """The one cached chart must not publish a dirty checkpoint mid-rebase."""
        import threading
        entered_native = threading.Event()
        release_native = threading.Event()
        checkpoint_entered = threading.Event()
        checkpoint_done = threading.Event()
        errors = []
        results = {}
        self.runner.chart.write_rail(17, 136, 0, make_word(0x7291))
        original_commit = self.coordinator.commit

        def gated_native_commit(**kwargs):
            entered_native.set()
            if not release_native.wait(10):
                raise RuntimeError("native checkpoint test barrier timed out")
            return original_commit(**kwargs)

        self.coordinator.commit = gated_native_commit
        identity = id(self.runner.chart)

        def native_worker():
            try:
                results["native"] = self.rebase()
            except BaseException as exc:
                errors.append(exc)

        def local_worker():
            checkpoint_entered.set()
            try:
                results["local"] = self.runner.checkpoint()
            except BaseException as exc:
                errors.append(exc)
            finally:
                checkpoint_done.set()

        native = threading.Thread(target=native_worker)
        local = threading.Thread(target=local_worker)
        try:
            native.start()
            self.assertTrue(entered_native.wait(10))
            local.start()
            self.assertTrue(checkpoint_entered.wait(10))
            # A generation-only or disk-only lock cannot serialize the
            # still-in-progress HERMES cached-chart rebase.
            self.assertFalse(checkpoint_done.wait(0.25))
        finally:
            release_native.set()
            native.join(15)
            if local.ident is not None:
                local.join(15)
        self.assertFalse(native.is_alive())
        self.assertFalse(local.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(results["native"]["head_bank_rebased"])
        self.assertEqual(results["native"]["chart_generation"], 2)
        self.assertEqual(results["local"]["generation"], 3)
        self.assertEqual(id(self.runner.chart), identity)
        self.assertIs(self.runner.register.chart, self.runner.chart)
        self.assertIs(self.runner.heads.chart, self.runner.chart)
        generation, page = load_chart_snapshot(self.state, self.parent)
        self.assertEqual(generation, 3)
        self.assertEqual(page[BANK_OFFSET:BANK_END], self.raw)
        offset = self.runner.chart.cell_offset(17, 136)
        self.assertEqual(
            page[offset:offset + 8], make_word(0x7291).to_bytes(8, "little"),
        )
        self.assertFalse(self.runner.chart._dirty)

    def test_replay_does_not_replace_cached_state_or_increment_generation(self):
        first = self.rebase()
        self.assertTrue(first["head_bank_rebased"])
        again = self.rebase()
        self.assertFalse(again["head_bank_rebased"])
        self.assertEqual(again["chart_generation"], 2)
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)

    def test_stale_cached_bank_fails_before_checkpoint(self):
        self.runner.chart.bytes[BANK_OFFSET] = 0x7C
        with self.assertRaisesRegex(ChartError, "bank changed"):
            self.rebase()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)
        self.assertEqual(self.journal.confirms, 0)

    def test_wrong_parent_and_unrelated_state_directory_fail_closed(self):
        wrong = SimpleNamespace(
            parent_root_sha256=digest(b"unrelated parent"),
            chart_generation=1, ring_epoch=19,
            bank_before_sha256=digest(bytes(BANK_BYTES)),
        )
        with self.assertRaisesRegex(ChartError, "parent/generation"):
            self.rebase(request=wrong)
        foreign = SnapshotHeadBankFence(
            state_dir=self.state / "foreign", parent_root_sha256=self.parent,
            universe_id=self.universe,
        )
        other = NativeHeadCheckpointCoordinator(
            fence=foreign, journal=self.journal,
        )
        with self.assertRaisesRegex(ChartError, "another state directory"):
            self.rebase(coordinator=other)

    def test_ring_handoff_failure_preserves_uncommitted_local_chart(self):
        self.journal.epoch = 20
        with self.assertRaisesRegex(ValueError, "ring epoch"):
            self.rebase()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)
        self.assertEqual(self.runner.chart.snapshot_generation, 1)


if __name__ == "__main__":
    unittest.main()
