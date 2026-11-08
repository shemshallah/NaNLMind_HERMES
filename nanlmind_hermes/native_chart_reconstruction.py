"""Reconstruct the immutable layout before certifying a native chart fence.

A N16SNAP1 page SHA proves byte integrity, not that those bytes retain the
canonical address/cell mapping. The mounted N16ROOT1 descriptor also does not
cover the index, canonical-address or pair tables. The local live runner
already rejects a changed immutable layout on restore; the stand-alone
authenticated dispatch fence must enforce the same boundary.

This is read-only structural verification against the existing canonical
NLMGLOB3 source (or deterministic genesis if no source exists), not a second
active universe, NaN16 execution, or host-side learning implementation.
"""
from __future__ import annotations

import struct
from pathlib import Path

from .universe_chart import (
    ADDRESS_COUNT, CANONICAL_COUNT, MIRROR_PAIR_COUNT, OFF_CANON, OFF_INDEX,
    OFF_PAIR_TABLE, Chart8MiB, load_chart,
)

# Keep identical to restore_local_chart's immutable layout. Never compare
# mutable native bank, temporal rails, heads, or locally trained registers.
IMMUTABLE_SPANS = (
    (0, 40),
    (OFF_INDEX, ADDRESS_COUNT * 2),
    (OFF_CANON, CANONICAL_COUNT * 2),
    (OFF_PAIR_TABLE, MIRROR_PAIR_COUNT * 2),
)


class NativeChartReconstructionError(ValueError):
    """Checkpoint does not match its trusted canonical source layout."""


def verify_native_chart_reconstruction(
    *, page: bytes, state_dir: str | Path, parent_root_sha256: str
) -> None:
    """Fail closed before an outer-SHA-valid chart becomes dispatch authority.

    A trained deployment is reconstructed from its existing canonical source;
    if that source disappears, a trained-root checkpoint cannot be silently
    reinterpreted as genesis. Source must resolve to exactly the configured
    parent, and immutable bytes must match the same source used by LocalRunner
    restore. No chart state is written or registered in the live runner cache.
    """
    try:
        source = load_chart(state_dir)
        reference = source if source is not None else Chart8MiB.genesis()
    except (OSError, ValueError, IndexError, struct.error) as exc:
        raise NativeChartReconstructionError(
            "canonical native chart source cannot be reconstructed"
        ) from exc
    if reference.root_hash != parent_root_sha256:
        raise NativeChartReconstructionError(
            "canonical native chart source parent does not match fence"
        )
    for offset, length in IMMUTABLE_SPANS:
        if page[offset:offset + length] != reference.bytes[offset:offset + length]:
            raise NativeChartReconstructionError(
                "native chart immutable addressing differs from canonical source"
            )
