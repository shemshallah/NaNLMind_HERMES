"""Byte-exact HERMES transport for the native NaN16 head-bank guest region.

The NaN0S guest, not Python, validates slots, NaN classes, generations and
head-learning updates. This module only reserves a non-overlapping chart slice,
checks SHA-256 and lease digests, and moves opaque bytes into the unmerged chart.
It never claims the guest executed, that a bank is formatted, or that training
was accepted by the canonical NANNET/MTH authority.
"""
from __future__ import annotations

import hashlib
import hmac

from .universe_chart import (
    HEAD_REGISTER_BYTES, HEAD_REGISTER_OFFSET, OFF_EXTENSION, PAGE_BYTES,
)

BANK_OFFSET = 0x640000
BANK_WORDS = 67838  # NaN0S/src/nannet/n16_head_store.nas: NH_WORDS
BANK_BYTES = BANK_WORDS * 8
BANK_END = BANK_OFFSET + BANK_BYTES
BANK_GUEST_WORD_OFFSET = BANK_OFFSET // 8
BANK_HEADS = 255
BANK_FEATURES = 128


class HeadBankTransportError(ValueError):
    pass


def _region(chart):
    if chart is None or len(chart.bytes) != PAGE_BYTES:
        raise HeadBankTransportError("expected an 8 MiB chart")
    if not (OFF_EXTENSION < HEAD_REGISTER_OFFSET
            and HEAD_REGISTER_OFFSET + HEAD_REGISTER_BYTES <= BANK_OFFSET
            and BANK_OFFSET % 8 == 0 and BANK_END <= PAGE_BYTES):
        raise HeadBankTransportError("native bank overlaps a reserved chart region")
    return memoryview(chart.bytes)[BANK_OFFSET:BANK_END]


def _sha(value: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise HeadBankTransportError("expected lowercase SHA-256")
    try:
        digest = bytes.fromhex(value)
    except ValueError as exc:
        raise HeadBankTransportError("expected lowercase SHA-256") from exc
    if digest.hex() != value:
        raise HeadBankTransportError("expected lowercase SHA-256")
    return value


def describe_guest_bank(chart) -> dict:
    """Read-only geometry, occupancy and digest (not a native validity check)."""
    raw = _region(chart)
    return {
        "schema": "nanlm.native-head-bank-transport/1",
        "byte_offset": BANK_OFFSET,
        "guest_word_offset": BANK_GUEST_WORD_OFFSET,
        "byte_length": BANK_BYTES,
        "word_length": BANK_WORDS,
        "heads": BANK_HEADS,
        "features_per_head": BANK_FEATURES,
        "region_sha256": hashlib.sha256(raw).hexdigest(),
        "has_bytes": any(raw),
        "format_checked_by_native_guest": False,
        "native_guest_attached": False,
        "canonical_training_committed": False,
        "nannet_merge_issued": False,
    }


def export_guest_bank_bytes(chart) -> bytes:
    """Copy an opaque bank for a configured native VM; no interpretation."""
    return bytes(_region(chart))


def stage_guest_bank_bytes(chart, payload: bytes, *, payload_sha256: str,
                           expected_region_sha256: str,
                           expected_parent_root: str,
                           expected_snapshot_generation: int) -> dict:
    """Stage a native guest's opaque output behind explicit root/digest leases.

    A digest proves byte identity, *not* provenance or native execution. Caller
    must authenticate the guest/receipt separately; after export the native
    guest must invoke nh_open/nh_get. This is an in-process staging function,
    intentionally not exposed as an arbitrary plugin write tool.
    """
    region = _region(chart)
    if type(payload) is not bytes or len(payload) != BANK_BYTES:
        raise HeadBankTransportError("native bank payload length mismatch")
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(),
                               _sha(payload_sha256)):
        raise HeadBankTransportError("native bank payload digest mismatch")
    if type(expected_parent_root) is not str or not hmac.compare_digest(
            chart.root_hash, _sha(expected_parent_root)):
        raise HeadBankTransportError("canonical parent root mismatch")
    if type(expected_snapshot_generation) is not int or (
            expected_snapshot_generation != chart.snapshot_generation):
        raise HeadBankTransportError("stale chart snapshot generation")
    if not hmac.compare_digest(hashlib.sha256(region).hexdigest(),
                               _sha(expected_region_sha256)):
        raise HeadBankTransportError("stale bank region digest")
    # No header, opcode or checksum emulation. The native guest owns validity.
    region[:] = payload
    chart._dirty = True
    return describe_guest_bank(chart)
