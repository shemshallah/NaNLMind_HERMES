"""Read-only N16SNAP1-to-native-bank fence for authenticated router staging.

This adapter is configured on the HERMES side from trusted local deployment
settings. It never follows a request-provided chart path or universe ID,
formats NaN16 words, writes checkpoints, or declares canonical model training.
It binds one HERMES universe identifier to its actual SHA-verified, recovered
8 MiB chart checkpoint and its N16ROOT1 descriptor.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from .chart_storage import ChartStorageError, load_chart_snapshot, PAGE_BYTES
from .head_bank_transport import BANK_OFFSET, BANK_END, BANK_BYTES
from .universe_root import read_universe_root, UniverseRootError


class NativeHeadFenceError(ValueError):
    pass


class SnapshotHeadBankFence:
    """Fixed, read-only checkpoint authority; never use user paths or roots.

    Not a distributed lock: users of the returned generation MUST fence their
    writes with the existing N16SNAP1 expected-generation CAS. It checks only
    local durable checkpoint bytes, not lease, ring or NANNET authenticity.
    """

    def __init__(self, *, state_dir: str | Path, parent_root_sha256: str,
                 universe_id: int):
        if type(universe_id) is not int or not 1 <= universe_id <= 0x7FFFFFFF:
            raise NativeHeadFenceError("trusted native universe ID required")
        if type(parent_root_sha256) is not str or len(parent_root_sha256) != 64:
            raise NativeHeadFenceError("trusted parent root SHA-256 required")
        try:
            if bytes.fromhex(parent_root_sha256).hex() != parent_root_sha256:
                raise ValueError("noncanonical digest")
        except ValueError as exc:
            raise NativeHeadFenceError("trusted lowercase parent root required") from exc
        self._state_dir = Path(state_dir).expanduser()
        self._parent = parent_root_sha256
        self._universe = universe_id

    def __call__(self, universe_id: int) -> dict:
        if type(universe_id) is not int or universe_id != self._universe:
            raise NativeHeadFenceError("foreign native universe chart requested")
        try:
            generation, page = load_chart_snapshot(self._state_dir, self._parent)
        except (ChartStorageError, ValueError, OSError) as exc:
            raise NativeHeadFenceError("valid N16SNAP1 checkpoint unavailable") from exc
        if type(generation) is not int or generation < 1 or page is None:
            raise NativeHeadFenceError("native chart has no durable checkpoint")
        if len(page) != PAGE_BYTES or BANK_END > PAGE_BYTES or (
                BANK_END - BANK_OFFSET != BANK_BYTES):
            raise NativeHeadFenceError("native bank geometry is invalid")
        try:
            root = read_universe_root(
                SimpleNamespace(bytes=page, root_hash=self._parent)
            )
        except (UniverseRootError, ValueError) as exc:
            raise NativeHeadFenceError("mounted N16ROOT1 is absent or invalid") from exc
        if root.get("mounted") is not True:
            raise NativeHeadFenceError("native universe root not mounted")
        return {
            "schema": "nanlm.native-head-checkpoint-fence/1",
            "universe_id": self._universe,
            "parent_root_sha256": self._parent,
            "chart_generation": generation,
            "bank_sha256": hashlib.sha256(page[BANK_OFFSET:BANK_END]).hexdigest(),
            "chart_sha256": hashlib.sha256(page).hexdigest(),
            "root_block_sha256": root["mount_block_sha256"],
            "snapshot_verified": True,
            "native_guest_attached": False,
            "canonical_training_committed": False,
        }
