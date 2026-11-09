"""Verify detached router Ed25519 graph sender identity against pinned key.

The caller MUST supply a public key independently pinned to router_id/key_id
by its trusted configuration. Full graph state AND history must first be
independently replayed from one actual HERMES chart. This is not authorization
for native MTH, learning, chart writes or canonical model promotion.

An issued timestamp and nonce are checked but NOT consumed in a durable
replay registry here. Authentication is to a verified *signed statement*, not
at-most-once dispatch permission.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time

from .crypto_compat import Ed25519PublicKey, InvalidSignature
from .router_graph_history import RouterGraphHistoryVerifier
from .router_graph_page import RouterGraphPageError, _sha
from .universe_root import read_universe_root

SCHEMA = "nanrouter.graph-sender-attestation/1"
_DOMAIN = b"nanrouter.graph-sender-ed25519/1\0"
_FIELDS = (
    "schema", "router_id", "key_id", "graph_commitment_sha256",
    "topology_sha256", "parent_root_sha256", "chart_generation",
    "history_sha256", "nonce", "issued_at",
)
_KEYID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_SIG = re.compile(r"[0-9a-f]{128}\Z")
_KEY = re.compile(r"[0-9a-f]{64}\Z")


class GraphSenderVerificationError(RouterGraphPageError):
    pass


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def verify_graph_sender_attestation(
        verifier: RouterGraphHistoryVerifier, attestation: dict, *,
        expected_router_id: str, expected_key_id: str, public_key_hex: str,
        now: int | None = None, max_age_seconds: int = 300,
        future_skew_seconds: int = 30) -> dict:
    """Return an authenticated *read-only* history report or raise.

    A valid signature alone never changes graph/page bytes, executes kernels,
    reserves jobs or admits learning. Caller-provided key material must NEVER
    be used as the independent trusted public-key pin.
    """
    if (not isinstance(verifier, RouterGraphHistoryVerifier)
            or not verifier.status()["history_verified"]):
        raise GraphSenderVerificationError("complete verified history required")
    if (type(expected_router_id) is not str
            or _KEYID.fullmatch(expected_router_id) is None
            or type(expected_key_id) is not str
            or _KEYID.fullmatch(expected_key_id) is None
            or type(public_key_hex) is not str or _KEY.fullmatch(public_key_hex) is None):
        raise GraphSenderVerificationError("trusted router key pin is missing or invalid")
    if (type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 3600
            or type(future_skew_seconds) is not int
            or not 0 <= future_skew_seconds <= 300):
        raise GraphSenderVerificationError("invalid attestation time policy")
    if now is None:
        now = int(time.time())
    if type(now) is not int or now < 1:
        raise GraphSenderVerificationError("invalid trusted clock")
    if (type(attestation) is not dict
            or set(attestation) != set(_FIELDS) | {
                "signature_hex", "signed_by_router", "native_mth_receipt",
                "mathematical_truth_promoted",
            }):
        raise GraphSenderVerificationError("malformed signed router envelope")
    if (attestation["schema"] != SCHEMA
            or attestation["signed_by_router"] is not True
            or attestation["native_mth_receipt"] is not False
            or attestation["mathematical_truth_promoted"] is not False
            or attestation["router_id"] != expected_router_id
            or attestation["key_id"] != expected_key_id
            or type(attestation["nonce"]) is not str
            or _NONCE.fullmatch(attestation["nonce"]) is None
            or type(attestation["signature_hex"]) is not str
            or _SIG.fullmatch(attestation["signature_hex"]) is None
            or type(attestation["issued_at"]) is not int
            or type(attestation["chart_generation"]) is not int
            or not all(_sha(attestation[k]) for k in (
                "graph_commitment_sha256", "topology_sha256",
                "parent_root_sha256", "history_sha256"
            ))):
        raise GraphSenderVerificationError("invalid or overclaimed sender attestation")
    issued = attestation["issued_at"]
    if issued < now - max_age_seconds or issued > now + future_skew_seconds:
        raise GraphSenderVerificationError("expired or future-dated graph attestation")
    chart = verifier.state.chart
    header = verifier.state._header
    root = read_universe_root(chart)
    if (not root.get("mounted") or chart.root_hash != verifier.state.parent
            or chart.snapshot_generation != verifier.state.generation):
        raise GraphSenderVerificationError("resident chart epoch changed")
    for name, expected in (
        ("graph_commitment_sha256", verifier.state._commitment),
        ("topology_sha256", header["topology_sha256"]),
        ("parent_root_sha256", header["parent_root_sha256"]),
        ("chart_generation", header["chart_generation"]),
        ("history_sha256", header["history_sha256"]),
    ):
        actual = attestation[name]
        if (isinstance(expected, str) and not hmac.compare_digest(actual, expected)
                or isinstance(expected, int) and actual != expected):
            raise GraphSenderVerificationError("signed graph binding mismatch: " + name)
    body = {key: attestation[key] for key in _FIELDS}
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(attestation["signature_hex"]), _DOMAIN + _canonical(body))
    except (ValueError, InvalidSignature) as exc:
        raise GraphSenderVerificationError("graph sender signature rejected") from exc
    return {
        **verifier.status(),
        "schema": "nanlm.router-graph-sender-verification/1",
        "sender_authenticated": True,
        "router_id": expected_router_id,
        "key_id": expected_key_id,
        "attestation_sha256": hashlib.sha256(_canonical(attestation)).hexdigest(),
        "nonce_replay_protected": False,
        "native_mth_executed": False,
        "learning_admitted": False,
        "mathematical_truth_promoted": False,
    }


__all__ = ("GraphSenderVerificationError", "verify_graph_sender_attestation")
