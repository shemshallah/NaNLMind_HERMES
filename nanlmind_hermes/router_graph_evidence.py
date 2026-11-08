"""Verify graph revision evidence bytes against an operator-trusted leaf resolver.

This is a non-authorizing audit. It independently replays the full graph state
and history against one resident HERMES chart before resolving referenced
immutable MathLeaf versions. The resolver must be injected by trusted
deployment code; a request, graph page or sender attestation may not select
the evidence authority. This establishes digest integrity only, NEVER the
truth, proof status, MTH execution or admissibility of the evidence.
"""
from __future__ import annotations

import hashlib
import hmac
import json

from .router_graph_history import RouterGraphHistoryVerifier
from .router_graph_page import RouterGraphPageError
from .universe_root import read_universe_root


class GraphEvidenceIntegrityError(RouterGraphPageError):
    pass


def audit_graph_revision_evidence(verifier, state_pages, history_pages, *,
                                  resolve_trusted_leaf):
    """Revalidate *every* historical edge evidence reference or fail closed.

    The read resolver contract is resolve_trusted_leaf(leaf_id, version) -> a
    typed immutable leaf with leaf_id, version and digest() members.
    Resolving from the signed router packet would not be independent. This
    helper does not and cannot install a resolver as a network capability.
    """
    if (type(verifier) is not RouterGraphHistoryVerifier
            or not verifier.status()["history_verified"]
            or verifier.status()["failed"]
            or type(state_pages) is not list
            or not 1 <= len(state_pages) <= 128
            or type(history_pages) is not list
            or not 1 <= len(history_pages) <= 128
            or not callable(resolve_trusted_leaf)):
        raise GraphEvidenceIntegrityError(
            "complete graph replay, bounded pages and trusted resolver required")

    # Recheck the exact page contents; callers may have modified packet objects
    # after the first (successful) history replay.
    fresh = RouterGraphHistoryVerifier(verifier.state.chart)
    for page in state_pages:
        fresh.add_state_page(page)
    if not fresh.status()["state_verified"]:
        raise GraphEvidenceIntegrityError("incomplete evidence-state snapshot")

    lookup = {}
    revision_count = 0
    audit = hashlib.sha256(b"nanlm.hermes.graph-evidence-integrity/1\0")
    for page in history_pages:
        fresh.add_history_page(page)
        for item in page["history_entries"]:
            record = json.loads(item["payload"])
            key = (record["evidence_id"], record["evidence_version"])
            digest = record["evidence_digest"]
            # A single immutable leaf version cannot have competing digests.
            if key in lookup and lookup[key] != digest:
                raise GraphEvidenceIntegrityError(
                    "one immutable evidence version has conflicting digests")
            lookup[key] = digest
            reference = [
                item["edge_id"], item["generation"],
                record["evidence_id"], record["evidence_version"], digest,
            ]
            raw = json.dumps(reference, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
            audit.update(len(raw).to_bytes(8, "big") + raw)
            revision_count += 1

    if (not fresh.status()["history_verified"]
            or not hmac.compare_digest(
                fresh.state._commitment, verifier.state._commitment)):
        raise GraphEvidenceIntegrityError(
            "evidence pages differ from chart-bound replayed history")

    for (leaf_id, version), expected in sorted(lookup.items()):
        try:
            leaf = resolve_trusted_leaf(leaf_id, version)
            if (leaf is None or type(leaf.leaf_id) is not str
                    or leaf.leaf_id != leaf_id
                    or type(leaf.version) is not int or leaf.version != version):
                raise GraphEvidenceIntegrityError("wrong immutable evidence identity")
            actual = leaf.digest()
            if (type(actual) is not str or len(actual) != 64
                    or not hmac.compare_digest(actual, expected)):
                raise GraphEvidenceIntegrityError("immutable evidence digest mismatch")
        except GraphEvidenceIntegrityError:
            raise
        except Exception as exc:
            raise GraphEvidenceIntegrityError(
                "trusted evidence reference is absent or unreadable") from exc

    chart = verifier.state.chart
    root = read_universe_root(chart)
    if (not root.get("mounted")
            or chart.root_hash != verifier.state.parent
            or chart.snapshot_generation != verifier.state.generation):
        raise GraphEvidenceIntegrityError("resident graph chart changed during audit")
    return {
        "schema": "nanlm.router-graph-evidence-integrity/1",
        "graph_commitment_sha256": fresh.state._commitment,
        "history_sha256": fresh.state._header["history_sha256"],
        "evidence_reference_digest_sha256": audit.hexdigest(),
        "historical_revisions_checked": revision_count,
        "distinct_leaf_versions_revalidated": len(lookup),
        "graph_state_verified": True,
        "graph_history_verified": True,
        "evidence_digest_integrity_verified": True,
        "evidence_provenance_authenticated": False,
        "sender_authenticated": False,
        "evidence_mathematically_proven": False,
        "native_mth_executed": False,
        "learning_admitted": False,
        "mathematical_truth_promoted": False,
    }


__all__ = ("GraphEvidenceIntegrityError", "audit_graph_revision_evidence")
