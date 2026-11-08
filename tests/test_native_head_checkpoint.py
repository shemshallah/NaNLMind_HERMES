"""N16SNAP1 local bank checkpoint CAS and interruption reconciliation.

Journal double below tests only the local byte-persistence boundary; real
signed-lease and native instruction execution are tested in NaNROUTER's
three-repository native-head CI workflow.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.chart_storage import (
    ChartStorageError, load_chart_snapshot, save_chart_snapshot,
)
from nanlmind_hermes.head_bank_transport import (
    BANK_OFFSET, BANK_END, BANK_BYTES,
)
from nanlmind_hermes.native_head_checkpoint import (
    NativeHeadCheckpointCoordinator, NativeHeadCheckpointConflict,
    NativeHeadCheckpointError,
)
from nanlmind_hermes.native_head_fence import (
    SnapshotHeadBankFence, NativeHeadFenceError,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root


def sha(value):
    return hashlib.sha256(value).hexdigest()


class FakeCompletedJournal:
    """Controlled source of staged bytes; NOT an authorization substitute."""

    def __init__(self, *, fence, universe, raw):
        self.fence = fence
        self.universe = universe
        self.raw = raw
        self.after = sha(raw)
        certified = fence(universe)
        self.source_chart_sha256 = certified["chart_sha256"]
        self.source_root_block_sha256 = certified["root_block_sha256"]
        self.source_snapshot_bound = True
        self.state = "completed_staging_only"
        self.confirmations = 0
        self.fail_confirm_once = False
        self.pending = False
        self.epoch = 31
        self.ring_epoch_reader = lambda: self.epoch

    def inspect(self, *, request, lease_receipt, network_head, include_bank):
        if type(include_bank) is not bool or network_head != 0:
            raise AssertionError("must inspect the journal under head 0")
        if lease_receipt["universe_id"] != self.universe:
            raise NativeHeadCheckpointConflict("foreign journal universe")
        if self.pending:
            return {"state": "pending_or_ambiguous",
                    "canonical_training_committed": False}
        return {
            "state": self.state,
            "raw_bank": (self.raw if include_bank and
                         self.state == "completed_staging_only" else None),
            "bank_after_sha256": self.after,
            "canonical_training_committed": False,
            "source_snapshot_bound": self.source_snapshot_bound,
            "source_chart_sha256": self.source_chart_sha256,
            "source_root_block_sha256": self.source_root_block_sha256,
            "local_checkpoint_verified": self.state == "checkpointed_locally",
            "checkpoint_generation": (
                self.confirmed_generation
                if self.state == "checkpointed_locally" else None
            ),
            "request_sha256": sha(b"test-verified-request"),
        }

    def confirm_local_checkpoint(self, *, request, lease_receipt, network_head):
        self.confirmations += 1
        if self.fail_confirm_once:
            self.fail_confirm_once = False
            raise RuntimeError("simulated power failure before ACK")
        fence = self.fence(self.universe)
        if fence["chart_generation"] != request.chart_generation + 1:
            raise AssertionError("wrong generation ack")
        if fence["bank_sha256"] != self.after:
            raise AssertionError("wrong bank ack")
        self.state = "checkpointed_locally"
        self.confirmed_generation = fence["chart_generation"]
        return {
            "local_checkpoint_verified": True,
            "request_sha256": sha(b"test-verified-request"),
            "bank_after_sha256": self.after,
            "checkpoint_generation": fence["chart_generation"],
        }


class NativeHeadCheckpointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        chart = Chart8MiB.genesis()
        attach_universe_root(chart)
        self.parent = chart.root_hash
        self.original_page = bytes(chart.bytes)
        self.assertEqual(
            save_chart_snapshot(self.path, self.parent, chart.bytes, 0), 1
        )
        self.universe = 12345
        self.fence = SnapshotHeadBankFence(
            state_dir=self.path, parent_root_sha256=self.parent,
            universe_id=self.universe,
        )
        self.raw = bytes([0x8D]) * BANK_BYTES
        self.journal = FakeCompletedJournal(
            fence=self.fence, universe=self.universe, raw=self.raw,
        )
        self.coordinator = NativeHeadCheckpointCoordinator(
            fence=self.fence, journal=self.journal
        )
        self.request = SimpleNamespace(
            parent_root_sha256=self.parent,
            chart_generation=1,
            ring_epoch=31,
            bank_before_sha256=sha(bytes(BANK_BYTES)),
        )
        self.receipt = {"universe_id": self.universe}

    def commit(self, **kwargs):
        return self.coordinator.commit(
            request=kwargs.pop("request", self.request),
            lease_receipt=kwargs.pop("lease_receipt", self.receipt),
            network_head=kwargs.pop("network_head", 0),
        )

    def test_full_page_sha_cas_rejects_different_current_page(self):
        candidate = bytearray(self.original_page)
        candidate[BANK_OFFSET:BANK_END] = self.raw
        with self.assertRaisesRegex(ChartStorageError, "page digest"):
            save_chart_snapshot(
                self.path, self.parent, candidate, 1,
                expected_page_sha256="a" * 64,
            )
        generation, page = load_chart_snapshot(self.path, self.parent)
        self.assertEqual(generation, 1)
        self.assertEqual(page, self.original_page)
        with self.assertRaisesRegex(ChartStorageError, "invalid expected"):
            save_chart_snapshot(
                self.path, self.parent, candidate, 1,
                expected_page_sha256="Z" * 64,
            )
        with self.assertRaisesRegex(ChartStorageError, "requires an existing"):
            save_chart_snapshot(
                self.path, self.parent, candidate, 0,
                expected_page_sha256=sha(self.original_page),
            )

    def test_exact_old_page_only_bank_changes_and_replay_reconciles(self):
        committed = self.commit()
        self.assertEqual(committed["chart_generation"], 2)
        self.assertTrue(committed["local_checkpoint_verified"])
        self.assertFalse(committed["recovered_prior_local_checkpoint"])
        self.assertFalse(committed["canonical_training_committed"])
        self.assertFalse(committed["nannet_merge_issued"])
        self.assertFalse(committed["native_execution_newly_invoked"])
        generation, page = load_chart_snapshot(self.path, self.parent)
        self.assertEqual(generation, 2)
        self.assertEqual(page[BANK_OFFSET:BANK_END], self.raw)
        self.assertEqual(page[:BANK_OFFSET], self.original_page[:BANK_OFFSET])
        self.assertEqual(page[BANK_END:], self.original_page[BANK_END:])
        replayed = self.commit()
        self.assertTrue(replayed["recovered_prior_local_checkpoint"])
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)

    def test_power_loss_after_snapshot_before_ack_never_double_writes(self):
        self.journal.fail_confirm_once = True
        with self.assertRaisesRegex(RuntimeError, "power failure"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)
        self.assertEqual(self.journal.state, "completed_staging_only")
        result = self.commit()
        self.assertTrue(result["recovered_prior_local_checkpoint"])
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)
        self.assertEqual(self.journal.state, "checkpointed_locally")

    def test_historical_tombstone_valid_after_more_native_chart_generations(self):
        first = self.commit()
        self.assertEqual(first["chart_generation"], 2)
        # Next independent local checkpoint advances the chart and may
        # replace the bank; historical dispatch must remain acknowledged.
        _, page = load_chart_snapshot(self.path, self.parent)
        different = bytearray(page)
        different[BANK_OFFSET:BANK_END] = bytes([0x6F]) * BANK_BYTES
        save_chart_snapshot(self.path, self.parent, different, 2)
        older = self.commit()
        self.assertTrue(older["historical_local_checkpoint_verified"])
        self.assertEqual(older["chart_generation"], 2)
        self.assertEqual(older["current_chart_generation"], 3)
        self.assertFalse(older["current_chart_contains_result"])
        self.assertFalse(older["canonical_training_committed"])

    def test_same_bank_result_plus_unrelated_chart_change_is_not_acknowledged(self):
        # A competing writer can write the same resulting bank together with
        # unrelated chart changes. Matching the bank SHA and generation is
        # insufficient: the actual snapshot transition must be bank-only.
        patched = bytearray(self.original_page)
        patched[BANK_OFFSET:BANK_END] = self.raw
        patched[0x300000] ^= 1  # mutable descriptor region, not index or bank
        save_chart_snapshot(self.path, self.parent, patched, 1)
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "unrelated chart cells"):
            self.commit()
        self.assertEqual(self.journal.confirmations, 0)

    def test_stale_ring_epoch_denied_before_new_chart_write(self):
        self.journal.epoch = 32
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "ring epoch changed"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_authenticated_journal_changed_between_inspections_denied(self):
        original = self.journal.inspect
        calls = [0]

        def revoked_between(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise ValueError("router lease revoked between inspections")
            return original(*args, **kwargs)

        self.journal.inspect = revoked_between
        with self.assertRaisesRegex(ValueError, "revoked"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)
        self.assertEqual(calls[0], 2)

    def test_unbound_legacy_native_staging_refuses_new_chart_commit(self):
        self.journal.source_snapshot_bound = False
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "no certified source chart"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)
        self.assertEqual(self.journal.confirmations, 0)

    def test_stage_with_correct_bank_but_different_source_chart_refused(self):
        self.journal.source_chart_sha256 = sha(b"another exact chart page")
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "router source chart differs"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_stage_with_wrong_mounted_root_digest_refused(self):
        self.journal.source_root_block_sha256 = sha(b"wrong-root-descriptor")
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "native source root block changed"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_crash_recovery_demands_original_precommit_chart_identity(self):
        self.journal.fail_confirm_once = True
        with self.assertRaisesRegex(RuntimeError, "power failure"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)
        self.journal.source_chart_sha256 = sha(b"forged-crash-source")
        with self.assertRaisesRegex(
                NativeHeadCheckpointConflict, "unrelated chart cells or lost source"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 2)
        self.assertEqual(self.journal.state, "completed_staging_only")

    def test_pending_or_ambiguous_does_not_touch_chart(self):
        self.journal.pending = True
        with self.assertRaisesRegex(NativeHeadCheckpointConflict, "not completed"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_source_bank_digest_mismatch_denied(self):
        wrong = SimpleNamespace(
            parent_root_sha256=self.parent,
            chart_generation=1,
            bank_before_sha256=sha(b"wrong head bank"),
        )
        with self.assertRaisesRegex(NativeHeadCheckpointConflict, "source bank"):
            self.commit(request=wrong)
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_missing_or_corrupt_staged_bank_denied(self):
        self.journal.raw = b"short"
        with self.assertRaisesRegex(NativeHeadCheckpointError, "missing or corrupt"):
            self.commit()
        self.journal.raw = bytes([0xAB]) * BANK_BYTES
        with self.assertRaisesRegex(NativeHeadCheckpointError, "missing or corrupt"):
            self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_foreign_universe_and_root_denied_before_mutation(self):
        with self.assertRaisesRegex(NativeHeadCheckpointConflict, "foreign journal"):
            self.commit(lease_receipt={"universe_id": self.universe + 1})
        wrong = SimpleNamespace(
            parent_root_sha256=sha(b"foreign-root"),
            chart_generation=1,
            bank_before_sha256=sha(bytes(BANK_BYTES)),
        )
        with self.assertRaisesRegex(NativeHeadCheckpointConflict, "parent root"):
            self.commit(request=wrong)
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_stale_generation_denied(self):
        updated = bytearray(self.original_page)
        updated[BANK_OFFSET:BANK_END] = bytes([0x81]) * BANK_BYTES
        save_chart_snapshot(self.path, self.parent, updated, 1)
        with self.assertRaisesRegex(NativeHeadCheckpointConflict, "exact next durable chart"):
            self.commit()

    def test_chart_changed_between_fence_and_snapshot_read_is_denied(self):
        from unittest.mock import patch
        import nanlmind_hermes.native_head_checkpoint as module
        read_real_snapshot = module.load_chart_snapshot

        def racy_snapshot(*args, **kwargs):
            generation, page = read_real_snapshot(*args, **kwargs)
            altered = bytearray(page)
            altered[0x7000] ^= 1
            return generation, bytes(altered)

        with patch.object(module, "load_chart_snapshot",
                          side_effect=racy_snapshot):
            with self.assertRaisesRegex(
                    NativeHeadCheckpointConflict, "full-page digest changed"):
                self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)

    def test_new_same_generation_chart_replacement_denied(self):
        # Directly exercise the under-lock digest fence against an altered
        # page that still claims the same generation, without use of a
        # forged second authorizing journal.
        newer = bytearray(self.original_page)
        newer[BANK_OFFSET:BANK_END] = bytes([0x1F]) * BANK_BYTES
        from unittest.mock import patch
        import nanlmind_hermes.native_head_checkpoint as module
        original_save = module.save_chart_snapshot

        def conflicting_save(*args, **kwargs):
            kwargs["expected_page_sha256"] = sha(b"stale page")
            return original_save(*args, **kwargs)

        with patch.object(module, "save_chart_snapshot",
                          side_effect=conflicting_save):
            with self.assertRaisesRegex(
                    NativeHeadCheckpointConflict, "snapshot CAS"):
                self.commit()
        self.assertEqual(self.fence(self.universe)["chart_generation"], 1)


if __name__ == "__main__":
    unittest.main()
