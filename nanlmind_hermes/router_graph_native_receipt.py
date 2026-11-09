"""Verify a worker-signed native graph claim, without admitting learning.

The trusted caller injects independent key pins, MathLeaf resolver, lease
digest, guest image digest and nonce. This verifier replays graph pages,
authenticates the router's signature, audits every historical evidence
reference, and reconstructs the exact staged NaN16 U32 payload. A worker
signature attests its claimed guest result, NOT independently measured
execution, mathematical truth, replay consumption or model admission.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time

from .crypto_compat import Ed25519PublicKey, InvalidSignature
from .router_graph_attestation import verify_graph_sender_attestation
from .router_graph_evidence import audit_graph_revision_evidence
from .router_graph_native_input import pack_native_graph_claim_input
from .router_graph_page import RouterGraphPageError

_RECEIPT_DOMAIN = b"nanlm.hermes.graph-native-worker-receipt/1\0"
SCHEMA = "nanlm.hermes.graph-native-worker-receipt/1"
_IDENT = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX128 = re.compile(r"[0-9a-f]{128}\Z")
_FIELDS = (
    "schema", "worker_id", "key_id", "graph_envelope_sha256",
    "word_digest_sha256", "provenance_sha256", "graph_commitment_sha256",
    "history_sha256", "chart_generation", "router_attestation_sha256",
    "evidence_reference_digest_sha256", "guest_image_sha256",
    "rule_epoch", "lease_sha256", "nonce", "query_src", "query_dst",
    "query_layer", "guest_status", "result_mask", "issued_at",
)


class NativeGraphReceiptError(RouterGraphPageError):
    """Invalid, stale, unauthenticated or overclaimed native claim receipt."""


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return type(value) is str and _HEX64.fullmatch(value) is not None


def verify_native_graph_claim_receipt(
    verifier, state_pages, history_pages, sender_attestation, receipt, *,
    expected_router_id, expected_router_key_id, pinned_router_public_key_hex,
    resolve_trusted_leaf, expected_worker_id, expected_worker_key_id,
    pinned_worker_public_key_hex, expected_guest_image_sha256,
    expected_rule_epoch, expected_lease_sha256, expected_nonce,
    expected_query, now=None,
):
    """Check signed read-only guest claim with independent replay/audit.

    Verification does NOT consume nonce or lease, execute an instruction,
    authorize a model write, or certify a proof. The trusted worker's
    signature alone cannot independently prove guest execution.
    """
    if (type(expected_query) is not tuple or len(expected_query) != 3
            or any(type(x) is not int for x in expected_query)
            or not 0 <= expected_query[0] < 32896
            or not 0 <= expected_query[1] < 32896
            or not 0 <= expected_query[2] <= 9
            or type(expected_rule_epoch) is not int or expected_rule_epoch < 0
            or type(expected_worker_id) is not str
            or _IDENT.fullmatch(expected_worker_id) is None
            or type(expected_worker_key_id) is not str
            or _IDENT.fullmatch(expected_worker_key_id) is None
            or any(not _digest(x) for x in (
                expected_guest_image_sha256, expected_lease_sha256,
                pinned_worker_public_key_hex))
            or type(expected_nonce) is not str
            or _HEX32.fullmatch(expected_nonce) is None):
        raise NativeGraphReceiptError("trusted native dispatch bindings missing")
    if now is None:
        now = int(time.time())
    if type(now) is not int or now < 1:
        raise NativeGraphReceiptError("trusted clock missing")

    # These are ACTUAL independent operations, not booleans asserted by a
    # wire packet or the worker. None of them authorizes learning.
    sender = verify_graph_sender_attestation(
        verifier, sender_attestation, expected_router_id=expected_router_id,
        expected_key_id=expected_router_key_id,
        public_key_hex=pinned_router_public_key_hex, now=now)
    evidence = audit_graph_revision_evidence(
        verifier, state_pages, history_pages,
        resolve_trusted_leaf=resolve_trusted_leaf)
    staged = pack_native_graph_claim_input(
        verifier, state_pages, history_pages)

    if (type(receipt) is not dict
            or set(receipt) != set(_FIELDS) | {"signature_hex"}
            or receipt["schema"] != SCHEMA
            or receipt["worker_id"] != expected_worker_id
            or receipt["key_id"] != expected_worker_key_id
            or type(receipt["issued_at"]) is not int
            or not now - 300 <= receipt["issued_at"] <= now + 30
            or type(receipt["rule_epoch"]) is not int
            or receipt["rule_epoch"] != expected_rule_epoch
            or type(receipt["chart_generation"]) is not int
            or type(receipt["guest_status"]) is not int
            or receipt["guest_status"] != 1
            or type(receipt["result_mask"]) is not int
            or receipt["result_mask"] not in (0, 1, 2, 3)
            or type(receipt["signature_hex"]) is not str
            or _HEX128.fullmatch(receipt["signature_hex"]) is None
            or type(receipt["nonce"]) is not str
            or receipt["nonce"] != expected_nonce
            or tuple(receipt[k] for k in (
                "query_src", "query_dst", "query_layer")) != expected_query
            or any(type(receipt[k]) is not int for k in (
                "query_src", "query_dst", "query_layer"))):
        raise NativeGraphReceiptError("malformed, overclaimed or expired native receipt")

    bindings = {
        "graph_envelope_sha256": staged["envelope_sha256"],
        "word_digest_sha256": staged["word_digest_sha256"],
        "provenance_sha256": staged["provenance_sha256"],
        "graph_commitment_sha256": staged["graph_commitment_sha256"],
        "history_sha256": staged["history_sha256"],
        "chart_generation": staged["chart_generation"],
        "router_attestation_sha256": sender["attestation_sha256"],
        "evidence_reference_digest_sha256":
            evidence["evidence_reference_digest_sha256"],
        "guest_image_sha256": expected_guest_image_sha256,
        "lease_sha256": expected_lease_sha256,
    }
    for name, value in bindings.items():
        actual = receipt[name]
        if type(value) is int:
            if type(actual) is not int or actual != value:
                raise NativeGraphReceiptError("native receipt epoch mismatch: " + name)
        elif type(actual) is not str or not hmac.compare_digest(actual, value):
            raise NativeGraphReceiptError("native receipt digest mismatch: " + name)
    body = {k: receipt[k] for k in _FIELDS}
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(pinned_worker_public_key_hex)).verify(
                bytes.fromhex(receipt["signature_hex"]),
                _RECEIPT_DOMAIN + _canonical(body))
    except (ValueError, InvalidSignature) as exc:
        raise NativeGraphReceiptError("pinned native worker signature rejected") from exc

    return {
        "schema": "nanlm.hermes.graph-native-claim-verification/1",
        "graph_history_verified": True,
        "sender_authenticated": True,
        "evidence_digest_integrity_verified": True,
        "native_result_worker_attested": True,
        "worker_id": expected_worker_id,
        "graph_envelope_sha256": staged["envelope_sha256"],
        "word_digest_sha256": staged["word_digest_sha256"],
        "receipt_sha256": hashlib.sha256(_canonical(receipt)).hexdigest(),
        "query": list(expected_query),
        "candidate_mask": receipt["result_mask"],
        "nonce_replay_protected": False,
        "lease_dispatch_authorized": False,
        "native_execution_independently_verified": False,
        "native_admission_verified": False,
        "learning_admitted": False,
        "mathematical_truth_promoted": False,
    }


__all__ = ("NativeGraphReceiptError", "verify_native_graph_claim_receipt")
