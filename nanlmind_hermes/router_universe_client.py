"""Paired HERMES installation -> NANNET portal -> NaNROUTER universe leases.

The only remote target here is the *already pinned* NANNET authority.
The portal verifies installation signatures and scopes; the router creates,
signs, and verifies the durable lease. No guest-provided address, owner tag,
node point, deadline or generation is accepted as authorization.

Local receipts are encrypted at rest and remain inert until the router's
live resolve endpoint approves a head. This adapter does not pretend to have
loaded a NaN16 guest registry, committed a route, or merged training state.
"""
from __future__ import annotations

import ipaddress
import hashlib
import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .auth import decrypt_payload, encrypt_payload, load_installation
from .auth_tools import _config, _master_key, _signed_headers, _universe_dir, require_scope


class UniverseLeaseError(ValueError):
    pass


_AAD = b"NANNET/HERMES/router-universes/v1"
_SCHEMA = "nanlmind-hermes/router-universes.v1"
_ALIAS = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _state_path(ctx):
    return _universe_dir(ctx) / "router_universe_leases.json"


def _read(ctx):
    path = _state_path(ctx)
    if not path.exists():
        return {"schema": _SCHEMA, "leases": {}, "pending": {}}
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        state = json.loads(decrypt_payload(_master_key(ctx), envelope, _AAD))
        if (not isinstance(state, dict) or state.get("schema") != _SCHEMA
                or not isinstance(state.get("leases"), dict)
                or not isinstance(state.get("pending"), dict)):
            raise UniverseLeaseError("invalid encrypted router-universe state")
        return state
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise UniverseLeaseError("cannot decrypt existing universe leases; restore local state/key") from exc


def _write(ctx, state):
    path = _state_path(ctx)
    envelope = encrypt_payload(
        _master_key(ctx),
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode(), _AAD)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(envelope, separators=(",", ":")) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _alias(value):
    if not isinstance(value, str) or not _ALIAS.fullmatch(value):
        raise UniverseLeaseError("universe alias must be 1..40 ASCII letters, digits, _ or -")
    return value


def _origin(ctx):
    origin = _config(ctx, "nannet_auth_url").rstrip("/")
    url = urlsplit(origin)
    try:
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = (url.hostname or "").lower() == "localhost"
    if (not url.hostname or url.username or url.password
            or url.path or url.query or url.fragment
            or not (url.scheme == "https" or (url.scheme == "http" and loopback))):
        raise UniverseLeaseError("pinned NANNET HTTPS authority origin is required")
    if not _config(ctx, "nannet_issuer_public_key"):
        raise UniverseLeaseError("NANNET issuer key must be pinned")
    return origin


def _post(ctx, action, payload):
    if action not in ("open", "resolve", "renew", "release", "dispatch", "ack", "inspect", "math"):
        raise UniverseLeaseError("unknown router universe operation")
    require_scope(ctx, "hermes:lease")  # includes online revocation and scope check
    root = _universe_dir(ctx)
    doc, private = load_installation(root, _master_key(ctx))
    token_env = json.loads((root / "nannet_token.json").read_text(encoding="utf-8"))
    token = decrypt_payload(
        _master_key(ctx), token_env,
        ("nannet-token-v1:" + doc["installation_id"]).encode()).decode()
    path = "/v1/hermes/leases/" + action
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > (150000 if action == "dispatch" else 16384):
        raise UniverseLeaseError("HERMES lease request exceeds portal limit")
    headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json",
               "Accept": "application/json", "Cache-Control": "no-store"}
    headers.update(_signed_headers(private, "POST", path, raw))
    request = urllib.request.Request(_origin(ctx) + path, data=raw, headers=headers, method="POST")
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=15) as response:
            if response.status != 200 or response.headers.get_content_type() != "application/json":
                raise UniverseLeaseError("unsupported NANNET router response")
            limit = 155000 if action in ("dispatch", "inspect") else 32768
            data = response.read(limit + 1)
            if len(data) > limit:
                raise UniverseLeaseError("NANNET router response exceeds size bound")
    except urllib.error.HTTPError as exc:
        raise UniverseLeaseError("NANNET portal rejected router lease operation (HTTP " + str(exc.code) + ")") from None
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise UniverseLeaseError("NANNET router lease authority is unavailable; no local state changed") from None
    try:
        result = json.loads(data.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError()
        return doc["installation_id"], result
    except (UnicodeError, ValueError) as exc:
        raise UniverseLeaseError("invalid response from router lease authority") from exc


def _valid_receipt(receipt, installation_id, previous=None):
    if not isinstance(receipt, dict):
        raise UniverseLeaseError("router did not return a lease receipt")
    if (receipt.get("schema") != "nanrouter.hermes-lease.v1"
            or receipt.get("installation_id") != installation_id
            or receipt.get("network_port") != "BS0:0"
            or receipt.get("hardware_root") != "HW0:0"
            or receipt.get("heads") != 255 or receipt.get("nannet_port_head") != 255):
        raise UniverseLeaseError("unexpected HERMES lease authority or address namespace")
    mac = receipt.get("receipt_mac")
    if not isinstance(mac, str) or not re.fullmatch(r"[a-f0-9]{64}", mac):
        raise UniverseLeaseError("router receipt is missing its opaque signature")
    for field, lower, upper in (
        ("slot", 0, 15), ("lease_id", 1, 0x7fffffff),
        ("universe_id", 1, 0x7fffffff), ("owner_tag", 1, 0x7fffffff),
        ("generation", 1, 0x7fffffff), ("revision", 1, 0x7fffffff),
        ("node_point", 1, 256), ("expires_at", 1, 0x7fffffff),
    ):
        value = receipt.get(field)
        if type(value) is not int or not lower <= value <= upper:
            raise UniverseLeaseError("router receipt has invalid " + field)
    if receipt["expires_at"] <= int(time.time()):
        raise UniverseLeaseError("router issued expired universe lease")
    if previous:
        for key in ("lease_id", "universe_id", "slot", "generation",
                    "node_point", "owner_tag", "installation_id"):
            if receipt[key] != previous[key]:
                raise UniverseLeaseError("renewal changed the active lease identity")
        if receipt["revision"] <= previous["revision"]:
            raise UniverseLeaseError("lease revision did not advance")
        if receipt["expires_at"] <= previous["expires_at"]:
            raise UniverseLeaseError("lease deadline did not advance")
    return receipt


def status(ctx):
    state = _read(ctx)
    leases = []
    for alias, lease in sorted(state["leases"].items()):
        leases.append({
            "alias": alias, "universe_id": lease.get("universe_id"),
            "slot": lease.get("slot"), "generation": lease.get("generation"),
            "revision": lease.get("revision"), "node_point": lease.get("node_point"),
            "expires_at": lease.get("expires_at"),
            "locally_expired": type(lease.get("expires_at")) is int and lease["expires_at"] <= int(time.time()),
        })
    return {
        "leases": leases, "pending_aliases": sorted(state["pending"]),
        "network_port": "BS0:0", "hardware_root": "HW0:0",
        "local_head_count": 255, "nannet_port_head": 255,
        "native_guest_activated": False,
        "remote_lease_is_live_only_after_resolve": True,
    }


def open_universe(ctx, alias="primary", ttl_seconds=900):
    alias = _alias(alias)
    if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= 3600:
        raise UniverseLeaseError("ttl_seconds must be 60..3600")
    state = _read(ctx)
    if alias in state["leases"]:
        raise UniverseLeaseError("universe alias already leased; release it first")
    if len(state["leases"]) >= 16:
        raise UniverseLeaseError("local NaN16 lease registry has only 16 slots")
    pending = state["pending"].get(alias)
    if pending is None:
        pending = {"request_id": "hermes-" + secrets.token_hex(16), "ttl_seconds": ttl_seconds}
        state["pending"][alias] = pending
        _write(ctx, state)  # retries retain the same request ID
    elif pending.get("ttl_seconds") != ttl_seconds:
        raise UniverseLeaseError("pending request has a different TTL; retry unchanged")
    installation_id, response = _post(ctx, "open", pending)
    receipt = _valid_receipt(response.get("receipt"), installation_id)
    if any(existing.get("slot") == receipt["slot"] for existing in state["leases"].values()):
        raise UniverseLeaseError("router reused an active local NaN16 lease slot")
    state["leases"][alias] = receipt
    state["pending"].pop(alias, None)
    _write(ctx, state)
    return {"opened": True, "alias": alias, "universe_id": receipt["universe_id"],
            "slot": receipt["slot"], "generation": receipt["generation"],
            "node_point": receipt["node_point"], "expires_at": receipt["expires_at"],
            "native_guest_activated": False}


def resolve(ctx, alias="primary", head_index=0):
    alias = _alias(alias)
    if type(head_index) is not int or not 0 <= head_index <= 254:
        raise UniverseLeaseError("head 255 is the NANNET port, not a learning head")
    receipt = _read(ctx)["leases"].get(alias)
    if receipt is None:
        raise UniverseLeaseError("universe alias has no local receipt")
    installation_id, response = _post(ctx, "resolve", {
        "receipt": receipt, "head_index": head_index})
    _valid_receipt(receipt, installation_id)
    route = response.get("route")
    if not isinstance(route, dict) or route.get("admitted") is not True:
        raise UniverseLeaseError("router did not admit the head route")
    for field in ("lease_id", "universe_id", "slot", "node_point",
                  "owner_tag", "generation", "expires_at"):
        if route.get(field) != receipt[field]:
            raise UniverseLeaseError("router resolution does not match saved lease binding")
    if route.get("head_index") != head_index or route.get("network_port") != "BS0:0":
        raise UniverseLeaseError("router resolved a different head or port")
    return {"admitted": True, "alias": alias, "head_index": head_index,
            "universe_id": receipt["universe_id"],
            "slot": receipt["slot"], "generation": receipt["generation"],
            "node_point": receipt["node_point"],
            "native_guest_activated": route.get("native_guest_route_admitted") is True,
            "execution_committed": route.get("native_execution_committed") is True}


def renew(ctx, alias="primary", ttl_seconds=900):
    alias = _alias(alias)
    state = _read(ctx)
    old = state["leases"].get(alias)
    if old is None:
        raise UniverseLeaseError("universe alias has no local receipt")
    installation_id, response = _post(ctx, "renew", {
        "receipt": old, "ttl_seconds": ttl_seconds})
    state["leases"][alias] = _valid_receipt(response.get("receipt"), installation_id, old)
    _write(ctx, state)
    return {"renewed": True, "alias": alias,
            "revision": state["leases"][alias]["revision"],
            "expires_at": state["leases"][alias]["expires_at"]}


def release(ctx, alias="primary"):
    alias = _alias(alias)
    state = _read(ctx)
    old = state["leases"].get(alias)
    if old is None:
        raise UniverseLeaseError("universe alias has no local receipt")
    _, response = _post(ctx, "release", {"receipt": old})
    if response.get("released") is not True or response.get("lease_id") != old["lease_id"]:
        raise UniverseLeaseError("router did not confirm revocation")
    state["leases"].pop(alias)
    _write(ctx, state)
    return {"released": True, "alias": alias, "universe_id": old["universe_id"]}


def dispatch(ctx, alias="primary", head_index=0, dispatch_id="", frame_hex=""):
    """Send exactly one opaque N16T frame to the verified native guest.

    A caller-chosen dispatch_id is mandatory for safe retries. The router
    journals ambiguous failures and refuses to execute them twice.
    """
    alias = _alias(alias)
    if type(head_index) is not int or not 0 <= head_index <= 254:
        raise UniverseLeaseError("head 255 is reserved for NANNET")
    if (not isinstance(dispatch_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{12,128}", dispatch_id)):
        raise UniverseLeaseError("stable dispatch_id must be 12..128 safe ASCII characters")
    if (not isinstance(frame_hex, str) or len(frame_hex) % 2 != 0
            or not 72 <= len(frame_hex) <= 131144
            or re.fullmatch("[0-9a-fA-F]+", frame_hex) is None):
        raise UniverseLeaseError("N16T frame_hex must be 36..65572 bytes")
    receipt = _read(ctx)["leases"].get(alias)
    if receipt is None:
        raise UniverseLeaseError("cannot dispatch without an active local universe receipt")
    installation, result = _post(ctx, "dispatch", {
        "receipt": receipt, "head_index": head_index,
        "dispatch_id": dispatch_id, "frame_hex": frame_hex})
    _valid_receipt(receipt, installation)
    if (result.get("dispatch_id") != dispatch_id
            or result.get("head_index") != head_index
            or result.get("lease_id") != receipt["lease_id"]
            or result.get("universe_id") != receipt["universe_id"]
            or result.get("native_guest_route_admitted") is not True
            or result.get("n16t_native_guest_exchange") is not True):
        raise UniverseLeaseError("router did not attest native guest dispatch")
    expected_request = hashlib.sha256(bytes.fromhex(frame_hex)).hexdigest()
    if result.get("frame_sha256") != expected_request:
        raise UniverseLeaseError("router request frame digest mismatch")
    answer_hex = result.get("response_frame_hex")
    if (not isinstance(answer_hex, str)
            or not 72 <= len(answer_hex) <= 131144 or len(answer_hex) % 2
            or re.fullmatch("[0-9a-fA-F]+", answer_hex) is None):
        raise UniverseLeaseError("router returned an invalid N16T response")
    response_sha = hashlib.sha256(bytes.fromhex(answer_hex)).hexdigest()
    if result.get("response_sha256") != response_sha:
        raise UniverseLeaseError("router response frame digest mismatch")
    return {
        "alias": alias, "dispatch_id": dispatch_id, "head_index": head_index,
        "universe_id": receipt["universe_id"], "response_frame_hex": answer_hex,
        "response_sha256": response_sha,
        "native_guest_route_admitted": True,
        "n16t_native_guest_exchange": True,
        "acknowledged": result.get("acknowledged") is True,
        "replayed_from_journal": result.get("replayed_from_journal") is True,
        "model_training_committed": False,
    }


def acknowledge(ctx, alias="primary", head_index=0,
                dispatch_id="", response_sha256=""):
    alias = _alias(alias)
    if type(head_index) is not int or not 0 <= head_index <= 254:
        raise UniverseLeaseError("head 255 is reserved for NANNET")
    if (not isinstance(dispatch_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{12,128}", dispatch_id)):
        raise UniverseLeaseError("invalid N16T dispatch_id")
    if (not isinstance(response_sha256, str)
            or not re.fullmatch("[a-f0-9]{64}", response_sha256)):
        raise UniverseLeaseError("invalid response SHA-256")
    receipt = _read(ctx)["leases"].get(alias)
    if receipt is None:
        raise UniverseLeaseError("cannot acknowledge without a local lease")
    installation, result = _post(ctx, "ack", {
        "receipt": receipt, "head_index": head_index,
        "dispatch_id": dispatch_id, "response_sha256": response_sha256})
    _valid_receipt(receipt, installation)
    if (result.get("acknowledged") is not True
            or result.get("dispatch_id") != dispatch_id
            or result.get("response_sha256") != response_sha256):
        raise UniverseLeaseError("router did not acknowledge matching N16T result")
    return {"acknowledged": True, "alias": alias,
            "dispatch_id": dispatch_id, "response_sha256": response_sha256,
            "model_training_committed": False}



def inspect_dispatch(ctx, alias="primary", head_index=0, dispatch_id=""):
    """Inspect a router-journaled dispatch without re-executing a N16T frame.

    A pending_or_ambiguous result is deliberately NOT retried. For completed
    frames the response digest is independently checked before exposing it.
    All state remains bound to the current authenticated lease/head.
    """
    alias = _alias(alias)
    if type(head_index) is not int or not 0 <= head_index <= 254:
        raise UniverseLeaseError("head 255 is reserved for NANNET")
    if (not isinstance(dispatch_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{12,128}", dispatch_id)):
        raise UniverseLeaseError("invalid N16T dispatch_id")
    receipt = _read(ctx)["leases"].get(alias)
    if receipt is None:
        raise UniverseLeaseError("cannot inspect dispatch without a local active lease")
    installation, result = _post(ctx, "inspect", {
        "receipt": receipt, "head_index": head_index,
        "dispatch_id": dispatch_id})
    _valid_receipt(receipt, installation)
    if (result.get("dispatch_id") != dispatch_id
            or result.get("head_index") != head_index
            or result.get("lease_id") != receipt["lease_id"]
            or result.get("universe_id") != receipt["universe_id"]
            or result.get("model_training_committed") is not False):
        raise UniverseLeaseError("router inspection has an invalid lease/head binding")
    state = result.get("state")
    if state not in ("not_recorded", "pending_or_ambiguous", "completed", "acknowledged"):
        raise UniverseLeaseError("router returned an unsupported dispatch recovery state")
    response = {
        "alias": alias, "dispatch_id": dispatch_id,
        "head_index": head_index, "universe_id": receipt["universe_id"],
        "state": state, "model_training_committed": False,
    }
    if state == "pending_or_ambiguous":
        digest = result.get("frame_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise UniverseLeaseError("router pending frame digest is invalid")
        response["frame_sha256"] = digest
    if state in ("completed", "acknowledged"):
        raw = result.get("response_frame_hex")
        if (not isinstance(raw, str) or not 72 <= len(raw) <= 131144
                or len(raw) % 2
                or re.fullmatch(r"[0-9a-fA-F]+", raw) is None):
            raise UniverseLeaseError("router inspection returned an invalid native frame")
        response_digest = hashlib.sha256(bytes.fromhex(raw)).hexdigest()
        if result.get("response_sha256") != response_digest:
            raise UniverseLeaseError("router inspection response digest mismatch")
        if result.get("n16t_native_guest_exchange") is not True:
            raise UniverseLeaseError("router inspection lacks native N16T guest evidence")
        response["response_frame_hex"] = raw
        response["response_sha256"] = response_digest
        response["acknowledged"] = (state == "acknowledged")
    return response



def math_u32(ctx, alias="primary", head_index=0,
             kernel_id="mth.u32.add_checked", a=0, b=0):
    """Run a pinned native NaN16 MTH checked arithmetic kernel via BS0:0.

    This is an actual native guest calculation, not an N16T handshake nor a
    Python arithmetic stand-in. It cannot mutate the canonical learner.
    """
    alias = _alias(alias)
    if type(head_index) is not int or not 0 <= head_index <= 254:
        raise UniverseLeaseError("head 255 is the NANNET port, not MTH computation")
    if kernel_id not in ("mth.u32.add_checked", "mth.u32.sub_checked"):
        raise UniverseLeaseError("MTH kernel is not in the pinned guest allowlist")
    for name, word in (("a", a), ("b", b)):
        if type(word) is not int or not 0 <= word <= 0xffffffff:
            raise UniverseLeaseError("native MTH " + name + " must be a uint32 payload")
    lease = _read(ctx)["leases"].get(alias)
    if lease is None:
        raise UniverseLeaseError("a signed router lease is required for MTH execution")
    installation, response = _post(ctx, "math", {
        "receipt": lease, "head_index": head_index,
        "kernel_id": kernel_id, "a": a, "b": b,
    })
    _valid_receipt(lease, installation)
    if (response.get("kernel_id") != kernel_id
            or response.get("native_nan16_execution") is not True
            or response.get("native_guest_route_admitted") is not True
            or response.get("native_guest_universe_id") != lease["universe_id"]
            or response.get("native_guest_node_point") != lease["node_point"]
            or response.get("node_point") != lease["node_point"]
            or response.get("universe_id") != lease["universe_id"]
            or response.get("head_index") != head_index
            or response.get("model_training_committed") is not False):
        raise UniverseLeaseError("native MTH result is not bound to the leased HERMES head")
    result = response.get("value_u32")
    if type(result) is not int or not 0 <= result <= 0xffffffff:
        raise UniverseLeaseError("native MTH result is not a 32-bit word")
    if type(response.get("exact")) is not bool:
        raise UniverseLeaseError("native MTH exactness flag is malformed")
    return {
        "alias": alias, "kernel_id": kernel_id, "head_index": head_index,
        "universe_id": lease["universe_id"], "node_point": lease["node_point"],
        "result_u32": result, "exact": response["exact"],
        "native_nan16_execution": True, "model_training_committed": False,
    }
