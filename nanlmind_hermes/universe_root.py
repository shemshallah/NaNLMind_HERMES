"""Self-describing binary root for ONE HERMES 8 MiB NaN16 chart.

This 4 KiB region is a *mount descriptor*, not a new universe, a ZIP, or a
SQLite substitute. HERMES genesis instantiates the existing chart in memory;
a root block names how the NaNROUTER MP namespace and L0-L9 relate to it.
The real leaves remain in the existing MC0DB/BS0DB and native class-bearing
NaN16 cells remain owned by the native guest. No host mutation of head banks.

A missing all-zero root initializes deterministically; any partially written,
mismatched, corrupt or foreign root fails closed, never reinitializes.
The existing N16SNAP1 dual-slot chart snapshot provides crash recovery.
"""
from __future__ import annotations

import hashlib
import hmac
import struct

from .head_bank_transport import BANK_OFFSET, BANK_WORDS, BANK_END
from .universe_chart import (
    Chart8MiB, HEAD_REGISTER_OFFSET, HEAD_REGISTER_BYTES, PAGE_BYTES,
)

ROOT_MAGIC = b"N16ROOT1"
ROOT_VERSION = 1
ROOT_OFFSET = HEAD_REGISTER_OFFSET + HEAD_REGISTER_BYTES
ROOT_BYTES = 4096
ROOT_END = ROOT_OFFSET + ROOT_BYTES
ROOT_NAMESPACE = "nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0"
ROOT_HEADER = struct.Struct("<8s10I32s32s")
ROOT_CHECKSUM_BYTES = 32
HEAD_FIXED_POINTS = 256
MP_POINTS = 32640
CANONICAL_CELLS = HEAD_FIXED_POINTS + MP_POINTS
LAYER_COUNT = 10
# The byte geometry is fixed. A real router mapping is NOT yet attached.
MAP_STORAGE = "MC0DB:read-through"
LEAF_STORAGE = "BS0DB:content-addressed"
NO_TOPOLOGY_DIGEST = bytes(32)


class UniverseRootError(ValueError):
    pass


def _span(chart: Chart8MiB):
    if chart is None or len(chart.bytes) != PAGE_BYTES:
        raise UniverseRootError("requires one 8 MiB native chart")
    if (ROOT_OFFSET % 8 or ROOT_OFFSET < HEAD_REGISTER_OFFSET + HEAD_REGISTER_BYTES
            or ROOT_END > BANK_OFFSET or BANK_END > PAGE_BYTES):
        raise UniverseRootError("binary root overlaps NaN16 chart registers or head bank")
    return memoryview(chart.bytes)[ROOT_OFFSET:ROOT_END]


def _pack(parent_sha: str):
    if type(parent_sha) is not str or len(parent_sha) != 64:
        raise UniverseRootError("invalid canonical parent root")
    try:
        parent = bytes.fromhex(parent_sha)
    except ValueError as exc:
        raise UniverseRootError("invalid canonical parent root") from exc
    if parent.hex() != parent_sha:
        raise UniverseRootError("invalid canonical parent root")
    header = ROOT_HEADER.pack(
        ROOT_MAGIC, ROOT_VERSION, PAGE_BYTES, HEAD_FIXED_POINTS, MP_POINTS,
        CANONICAL_CELLS, LAYER_COUNT, BANK_OFFSET, BANK_WORDS,
        ROOT_OFFSET, ROOT_BYTES, parent, NO_TOPOLOGY_DIGEST,
    )
    digest = hashlib.sha256(header).digest()
    return header + digest + bytes(ROOT_BYTES - len(header) - ROOT_CHECKSUM_BYTES)


def _validate(chart, raw: bytes):
    if len(raw) != ROOT_BYTES or raw[:8] != ROOT_MAGIC:
        raise UniverseRootError("invalid binary universe root magic")
    expected = _pack(chart.root_hash)
    if not hmac.compare_digest(raw, expected):
        raise UniverseRootError(
            "universe root is corrupt, foreign, or geometry/version mismatch"
        )


def attach_universe_root(chart: Chart8MiB) -> dict:
    """Idempotent boot: allocate a descriptor inside the existing 8 MiB chart."""
    region = _span(chart)
    raw = bytes(region)
    if not any(raw):
        region[:] = _pack(chart.root_hash)
        chart._dirty = True
    else:
        _validate(chart, raw)
    return read_universe_root(chart)


def read_universe_root(chart: Chart8MiB) -> dict:
    """Sanitized metadata only, not a raw descriptor or authenticated binding."""
    region = _span(chart)
    raw = bytes(region)
    if not any(raw):
        return {"schema": "nanlm.universe-root/1", "mounted": False}
    _validate(chart, raw)
    from .mirror_topology import MirrorTopology
    topology = MirrorTopology.status()
    return {
        "schema": "nanlm.universe-root/1",
        "mounted": True,
        "magic": ROOT_MAGIC.decode("ascii"),
        "byte_offset": ROOT_OFFSET,
        "byte_length": ROOT_BYTES,
        "head_addresses": HEAD_FIXED_POINTS,
        "mirror_points": MP_POINTS,
        "canonical_cells": CANONICAL_CELLS,
        "l0_l9_layers": list(range(LAYER_COUNT)),
        "nan_root": ROOT_NAMESPACE,
        "head_route": ROOT_NAMESPACE + "::HW0:<head>",
        "mirror_route": ROOT_NAMESPACE + "::HW0:MP<ordinal>",
        "binary_chart_bytes": PAGE_BYTES,
        "native_head_bank_offset": BANK_OFFSET,
        "native_head_bank_words": BANK_WORDS,
        "parent_root_sha256": chart.root_hash,
        "mount_block_sha256": hashlib.sha256(raw).hexdigest(),
        "local_topology_attached": True,
        "topology_sha256": topology["topology_sha256"],
        "topology_status": topology,
        "recursive_route_examples": [
            ROOT_NAMESPACE + "::HW0:7::L5",
            ROOT_NAMESPACE + "::HW0:MP123::L3",
            ROOT_NAMESPACE + "::HW0:7::HW0:MP123::L3",
        ],
        "router_topology_attached": False,
        "router_index": MAP_STORAGE,
        "leaf_store": LEAF_STORAGE,
        "native_guest_attached": False,
        "canonical_training_committed": False,
    }
