"""Read-only, fail-closed NaN16 claim-probe input from a verified graph.

This stages a deterministic, bounded flat record batch for NaNROUTER's native
mth_graph_claim_probe.nas. It NEVER runs that kernel, creates an execution
receipt, or authorizes learning. The guest, not Python, decides which
relationship bitmask a subsequent query returns.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct

from .router_graph_history import RouterGraphHistoryVerifier
from .router_graph_page import RouterGraphPageError
from .universe_root import read_universe_root


class NativeGraphClaimInputError(RouterGraphPageError):
    """Unverified, stale, incomplete or oversized native input."""


_WIRE_DOMAIN = b"nanlm.nan16.graph-claim-probe/1\0"
_PROVENANCE_DOMAIN = b"nanlm.nan16.graph-claim-provenance/1\0"
_ENVELOPE_DOMAIN = b"nanlm.nan16.graph-claim-envelope/1\0"
_RELATION_CODE = {"supports": 1, "refutes": 2}
_MAX_RECORDS = 1024


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def pack_native_graph_claim_input(
    verifier: RouterGraphHistoryVerifier, state_pages: list,
    history_pages: list, *, max_records: int = _MAX_RECORDS,
) -> dict:
    """Recheck the exact immutable snapshot and stage non-authorizing words.

    The original verifier is an independently completed replay; fresh replay
    also detects post-verification mutation of either page batch. The output
    is 4 U32 words per active support/refute candidate in state-page order.
    Its digests detect accidental change, NOT sender authentication.
    """
    if (type(verifier) is not RouterGraphHistoryVerifier
            or not verifier.status()["history_verified"]
            or verifier.status()["failed"]
            or type(state_pages) is not list
            or not 1 <= len(state_pages) <= 128
            or type(history_pages) is not list
            or not 1 <= len(history_pages) <= 128
            or type(max_records) is not int
            or not 1 <= max_records <= _MAX_RECORDS):
        raise NativeGraphClaimInputError(
            "complete history and bounded native graph staging required"
        )

    chart = verifier.state.chart
    fresh = RouterGraphHistoryVerifier(chart)
    for page in state_pages:
        fresh.add_state_page(page)
    if not fresh.status()["state_verified"]:
        raise NativeGraphClaimInputError("native graph state is incomplete")
    for page in history_pages:
        fresh.add_history_page(page)
    if (not fresh.status()["history_verified"]
            or fresh.status()["failed"]
            or not hmac.compare_digest(
                fresh.state._commitment, verifier.state._commitment
            )):
        raise NativeGraphClaimInputError(
            "native graph pages differ from independently verified history"
        )

    root = read_universe_root(chart)
    if (root.get("mounted") is not True
            or chart.root_hash != fresh.state.parent
            or chart.snapshot_generation != fresh.state.generation):
        raise NativeGraphClaimInputError("native graph chart epoch changed")

    words = []
    provenance = []
    for page in state_pages:
        for edge in page["entries"]:
            if (edge["status"] != "candidate"
                    or edge["relation"] not in _RELATION_CODE):
                continue
            if len(provenance) >= max_records:
                raise NativeGraphClaimInputError(
                    "native claim input exceeds complete record bound"
                )
            words.extend((
                edge["src"], edge["dst"], edge["layer"],
                _RELATION_CODE[edge["relation"]],
            ))
            provenance.append({
                "record_index": len(provenance),
                "edge_id": edge["edge_id"],
                "generation": edge["generation"],
                "source_head": edge["source_head"],
                "evidence_id": edge["evidence_id"],
                "evidence_version": edge["evidence_version"],
                "evidence_digest": edge["evidence_digest"],
            })

    raw = struct.pack(">" + str(len(words)) + "I", *words)
    word_digest = hashlib.sha256(
        _WIRE_DOMAIN + len(provenance).to_bytes(4, "big") + raw
    ).hexdigest()
    provenance_digest = hashlib.sha256(
        _PROVENANCE_DOMAIN + _canonical(provenance)
    ).hexdigest()

    header = fresh.state._header
    binding = {
        "schema": "nanlm.router-native-claim-probe-input/1",
        "graph_commitment_sha256": fresh.state._commitment,
        "history_sha256": header["history_sha256"],
        "topology_sha256": header["topology_sha256"],
        "parent_root_sha256": header["parent_root_sha256"],
        "chart_generation": header["chart_generation"],
        "records_count": len(provenance),
        "word_encoding": "u32be",
        "word_count": len(words),
        "word_digest_sha256": word_digest,
        "provenance_sha256": provenance_digest,
    }
    return {
        **binding,
        "envelope_sha256": hashlib.sha256(
            _ENVELOPE_DOMAIN + _canonical(binding)
        ).hexdigest(),
        "words": words,
        "provenance": provenance,
        "history_verified": True,
        "chart_root_bound": True,
        "sender_authenticated": False,
        "evidence_integrity_verified": False,
        "native_mth_executed": False,
        "native_admission_verified": False,
        "learning_admitted": False,
        "mathematical_truth_promoted": False,
    }


__all__ = ("NativeGraphClaimInputError", "pack_native_graph_claim_input")
