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
