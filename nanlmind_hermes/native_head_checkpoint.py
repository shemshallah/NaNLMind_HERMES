"""Local N16SNAP1 commit for a *journal-verified* native head-bank observation.

HERMES owns ONE 8 MiB chart. The NaNROUTER authenticated observation journal
must already contain a completed, verified native NaN16 result. No opaque
client-supplied payload may be committed directly through this adapter.

This code publishes an unmerged local chart checkpoint only. It does not
authenticate a NANNET HTTP caller, issue a global merge, execute NaN16, or
interpret native NaN class faces. It is intended only for trusted in-process
deployment wiring to the genuine router journal and the HERMES chart.
"""
from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

from .chart_storage import (
    ChartStorageError, load_chart_snapshot, load_chart_snapshot_pair,
    save_chart_snapshot,
)
from .head_bank_transport import BANK_BYTES, BANK_OFFSET, BANK_END
from .native_head_fence import SnapshotHeadBankFence, NativeHeadFenceError
from .universe_root import read_universe_root, UniverseRootError


class NativeHeadCheckpointError(ValueError):
    pass


class NativeHeadCheckpointConflict(NativeHeadCheckpointError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class NativeHeadCheckpointCoordinator:
    """Stage only authenticated journal output via N16SNAP1 full-page CAS.

    The writer is deployment-internal and MUST receive the actual
    HermesHeadObservationJournal; its .inspect() enforces authenticated lease
    scope and verifies native guest receipts. A duck-typed journal from an
    untrusted caller would not provide authentication, so this class must
    never be exposed as an untrusted plugin or HTTP payload constructor.
    """

    def __init__(self, *, fence: SnapshotHeadBankFence, journal):
        if type(fence) is not SnapshotHeadBankFence:
            raise NativeHeadCheckpointError("trusted HERMES snapshot fence required")
        if not callable(getattr(journal, "inspect", None)) or not callable(
            getattr(journal, "confirm_local_checkpoint", None)
        ):
            raise NativeHeadCheckpointError("durable router observation journal required")
        self.fence = fence
        self.journal = journal

    def _after(self, *, request, universe_id, bank_after_sha256):
        state = self.fence(universe_id)
        if (state.get("snapshot_verified") is not True
                or type(request.chart_generation) is not int
                or state.get("chart_generation") != request.chart_generation + 1
                or state.get("parent_root_sha256") != request.parent_root_sha256
                or not hmac.compare_digest(
                    str(state.get("bank_sha256")), bank_after_sha256
                )):
            raise NativeHeadCheckpointConflict("native result not in exact next durable chart")
        # A bank SHA alone could accidentally match a different head update
        # in the same chart generation. Validate BOTH adjacent checkpoint
        # pages and require every byte *outside* the reserved native bank to
        # match exactly, including N16ROOT1, mirrored topology, and rails.
        try:
            prior, current = load_chart_snapshot_pair(
                self.fence._state_dir, self.fence._parent,
                request.chart_generation,
            )
        except (ChartStorageError, OSError, ValueError) as exc:
            raise NativeHeadCheckpointConflict(
                "cannot prove adjacent native chart checkpoint transition"
            ) from exc
        if (state.get("chart_sha256") != _sha(current)
                or not hmac.compare_digest(
                    _sha(prior[BANK_OFFSET:BANK_END]),
                    request.bank_before_sha256,
                )
                or not hmac.compare_digest(
                    _sha(current[BANK_OFFSET:BANK_END]),
                    bank_after_sha256,
                )
                or prior[:BANK_OFFSET] != current[:BANK_OFFSET]
                or prior[BANK_END:] != current[BANK_END:]):
            raise NativeHeadCheckpointConflict(
                "native update modified unrelated chart cells or lost source page"
            )
        return state

    def _result(self, *, request, lease_receipt, network_head,
                universe_id, after_sha256, recovered):
        state = self._after(
            request=request, universe_id=universe_id,
            bank_after_sha256=after_sha256,
        )
        confirmed = self.journal.confirm_local_checkpoint(
            request=request, lease_receipt=lease_receipt,
            network_head=network_head,
        )
        if (confirmed.get("local_checkpoint_verified") is not True
                or confirmed.get("bank_after_sha256") != after_sha256
                or confirmed.get("checkpoint_generation") != state["chart_generation"]):
            raise NativeHeadCheckpointError("router did not confirm exact chart checkpoint")
        return {
            "schema": "nanlm.native-head-local-checkpoint/1",
            "request_sha256": confirmed["request_sha256"],
            "bank_after_sha256": after_sha256,
            "parent_root_sha256": state["parent_root_sha256"],
            "chart_generation": state["chart_generation"],
            "root_block_sha256": state["root_block_sha256"],
            "snapshot_verified": True,
            "local_checkpoint_verified": True,
            "recovered_prior_local_checkpoint": recovered,
            "authenticated_router_journal_checked": True,
            "native_execution_newly_invoked": False,
            "canonical_training_committed": False,
            "nannet_merge_issued": False,
        }

    def commit(self, *, request, lease_receipt, network_head):
        """Commit or reconcile one completed native observation without reexec.

        Three outcomes:
        - pending/ambiguous journal entry: refuse without touching chart
        - completed journal entry: CAS exact old snapshot -> native bytes
        - checkpoint already present: reconcile and confirm without writing
        """
        if type(lease_receipt) is not dict:
            raise NativeHeadCheckpointError("router lease receipt required")
        universe_id = lease_receipt.get("universe_id")
        if type(universe_id) is not int:
            raise NativeHeadCheckpointError("native universe ID required")
        if type(network_head) is not int:
            raise NativeHeadCheckpointError("native network head required")
        # .inspect() checks current router-held lease MAC/SQL/expiry and
        # verifies the durable native guest response and request binding.
        staged = self.journal.inspect(
            request=request, lease_receipt=lease_receipt,
            network_head=network_head, include_bank=True,
        )
        if not isinstance(staged, dict):
            raise NativeHeadCheckpointError("unrecognized authenticated staging response")
        if staged.get("state") not in (
            "completed_staging_only", "checkpointed_locally",
        ):
            raise NativeHeadCheckpointConflict("native dispatch is not completed")
        if staged.get("canonical_training_committed") is not False:
            raise NativeHeadCheckpointError("unexpected training promotion in staging")
        after_sha256 = staged.get("bank_after_sha256")
        if (type(after_sha256) is not str or len(after_sha256) != 64
                or any(c not in "0123456789abcdef" for c in after_sha256)):
            raise NativeHeadCheckpointError("invalid staged bank SHA-256")
        # Do not let an arbitrary universe identifier select a different
        # HERMES chart. The fence itself is bound to trusted deployment IDs.
        state = self.fence(universe_id)
        if state.get("parent_root_sha256") != request.parent_root_sha256:
            raise NativeHeadCheckpointConflict("canonical parent root changed")
        if staged["state"] == "checkpointed_locally":
            # A permanent router tombstone outlives the original two
            # checkpoint slots. Legitimate later training generations must
            # not invalidate historical at-most-once acknowledgements.
            if (staged.get("local_checkpoint_verified") is not True
                    or type(staged.get("checkpoint_generation")) is not int
                    or staged["checkpoint_generation"] != request.chart_generation + 1):
                raise NativeHeadCheckpointError("invalid historical checkpoint tombstone")
            return {
                "schema": "nanlm.native-head-local-checkpoint/1",
                "request_sha256": staged["request_sha256"],
                "bank_after_sha256": after_sha256,
                "parent_root_sha256": state["parent_root_sha256"],
                "chart_generation": staged["checkpoint_generation"],
                "current_chart_generation": state["chart_generation"],
                "current_chart_contains_result": (
                    state["chart_generation"] == staged["checkpoint_generation"]
                    and hmac.compare_digest(state["bank_sha256"], after_sha256)
                ),
                "historical_local_checkpoint_verified": True,
                "local_checkpoint_verified": True,
                "recovered_prior_local_checkpoint": True,
                "authenticated_router_journal_checked": True,
                "native_execution_newly_invoked": False,
                "canonical_training_committed": False,
                "nannet_merge_issued": False,
            }
        raw_bank = staged.get("raw_bank")
        if type(raw_bank) is not bytes or len(raw_bank) != BANK_BYTES or (
                not hmac.compare_digest(_sha(raw_bank), after_sha256)):
            raise NativeHeadCheckpointError("staged native bank bytes are missing or corrupt")

        generation = state["chart_generation"]
        if generation == request.chart_generation + 1:
            # Power loss after successful N16SNAP1 commit but before journal
            # confirmation: don't execute again or perform an extra write.
            return self._result(
                request=request, lease_receipt=lease_receipt,
                network_head=network_head, universe_id=universe_id,
                after_sha256=after_sha256, recovered=True,
            )
        if generation != request.chart_generation:
            raise NativeHeadCheckpointConflict("chart generation superseded native work")
        if not hmac.compare_digest(
            state["bank_sha256"], request.bank_before_sha256
        ):
            raise NativeHeadCheckpointConflict("stale source bank digest")
        try:
            before_generation, page = load_chart_snapshot(
                self.fence._state_dir, self.fence._parent
            )
        except (ValueError, OSError) as exc:
            raise NativeHeadCheckpointError("verified HERMES chart unavailable") from exc
        if before_generation != generation or page is None:
            raise NativeHeadCheckpointConflict("chart generation changed during read")
        # N16SNAP1 outer SHA is insufficient to authenticate N16ROOT1.
        # Reject rootless or tampered chart pages, even if SHA is valid.
        try:
            root = read_universe_root(
                SimpleNamespace(bytes=page, root_hash=request.parent_root_sha256)
            )
        except (UniverseRootError, ValueError) as exc:
            raise NativeHeadCheckpointError("binary universe root is invalid") from exc
        if (root.get("mounted") is not True or
                root["mount_block_sha256"] != state["root_block_sha256"]):
            raise NativeHeadCheckpointConflict("universe root descriptor changed")
        if not hmac.compare_digest(
                _sha(page[BANK_OFFSET:BANK_END]), request.bank_before_sha256):
            raise NativeHeadCheckpointConflict("bank changed during snapshot read")

        # Revalidate the router-owned ring epoch and authenticated staging
        # record immediately before the durable N16SNAP1 CAS. An old
        # observation cannot initiate a NEW chart write after ring handoff.
        # Historic already-checkpointed dispatches remain replayable.
        epoch_reader = getattr(self.journal, "ring_epoch_reader", None)
        if not callable(epoch_reader):
            raise NativeHeadCheckpointError("live router ring epoch authority required")
        active_epoch = epoch_reader()
        if type(active_epoch) is not int or active_epoch < 1:
            raise NativeHeadCheckpointConflict("router ring epoch unavailable")
        if (type(request.ring_epoch) is not int
                or request.ring_epoch != active_epoch):
            raise NativeHeadCheckpointConflict("router ring epoch changed before checkpoint")
        fresh = self.journal.inspect(
            request=request, lease_receipt=lease_receipt,
            network_head=network_head, include_bank=False,
        )
        if (not isinstance(fresh, dict)
                or fresh.get("state") != "completed_staging_only"
                or fresh.get("bank_after_sha256") != after_sha256):
            raise NativeHeadCheckpointConflict(
                "authenticated journal stage changed before checkpoint"
            )

        # Isolate native NaN16 bank to its reserved bytes: no other chart
        # cells, mirror routes or temporal rails can be overwritten.
        updated = bytearray(page)
        updated[BANK_OFFSET:BANK_END] = raw_bank
        try:
            committed_generation = save_chart_snapshot(
                self.fence._state_dir, request.parent_root_sha256, updated,
                expected_generation=request.chart_generation,
                expected_page_sha256=_sha(page),
            )
        except ChartStorageError as exc:
            raise NativeHeadCheckpointConflict(
                "snapshot CAS rejected stale native head observation"
            ) from exc
        if committed_generation != request.chart_generation + 1:
            raise NativeHeadCheckpointError("snapshot returned unexpected generation")
        return self._result(
            request=request, lease_receipt=lease_receipt,
            network_head=network_head, universe_id=universe_id,
            after_sha256=after_sha256, recovered=False,
        )
