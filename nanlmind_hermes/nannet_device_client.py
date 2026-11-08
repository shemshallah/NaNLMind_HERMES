"""Android-compatible NANNET hardware-node enrollment for Hermes.

This module is strictly a host HTTPS/credential-storage boundary, not a
NaN16DE guest route executor and not a model/head allocator. NANNET owns the
HW lease; the NaNROUTER native guest gates own per-universe head eligibility.

HW0..HW3 are fixed, HW4 is WordNet, HW5..HW255 are dynamic BS0 hardware
identities. The BS0:0 runner-ring NANNET port is NOT HW0:0. A HW lease never
authorizes a HERMES compute universe or changes the BS0 learner state.
"""
from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .auth import encrypt_payload, decrypt_payload
from .auth_tools import _master_key, _universe_dir, _config

TERMS = "nannet-device-registration-v1"
_AAD = b"NANNET/HERMES/hardware-lease-v1"
_ADDRESS = re.compile(r"^nan://GM0:SC0:MC0:BS0:HW([0-9]{1,3})$")
_STATE_NAME = "nannet_hw_lease.json"


class HardwareEnrollmentError(ValueError):
    """Fail-closed NANNET hardware-lease boundary error."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _base_url(ctx):
    raw = _config(ctx, "nannet_auth_url").rstrip("/")
    parts = urlsplit(raw)
    try:
        local = ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        local = (parts.hostname or "").lower() == "localhost"
    if (not parts.hostname or parts.username or parts.password
            or parts.path or parts.query or parts.fragment
            or (parts.scheme != "https"
                and not (local and parts.scheme == "http"))):
        raise HardwareEnrollmentError(
            "configure a trusted NANNET HTTPS origin (loopback HTTP only for tests)")
    return raw


def _state_path(ctx):
    return _universe_dir(ctx) / _STATE_NAME


def _read(ctx):
    path = _state_path(ctx)
    if not path.exists():
        return {}
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        raw = decrypt_payload(_master_key(ctx), envelope, _AAD)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict) or data.get("schema") != "nannet-hw-lease/1":
            raise HardwareEnrollmentError("invalid encrypted hardware-lease state")
        return data
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HardwareEnrollmentError(
            "cannot decrypt existing HW identity; restore its key/state, do not replace it") from exc


def _write(ctx, state):
    path = _state_path(ctx)
    raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    encrypted = encrypt_payload(_master_key(ctx), raw, _AAD)
    temp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(encrypted, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _http_json(ctx, path, body=None, token=""):
    """Bounded HTTPS-only adapter with redirects forbidden (no token forwarding)."""
    if path not in (
        "/v1/devices/register", "/v1/devices/rotate",
        "/v1/devices/heartbeat", "/v1/devices/release",
        "/v1/devices/report"):
        raise HardwareEnrollmentError("unapproved NANNET device route")
    if token and (not isinstance(token, str) or
                  len(token) > 4096 or
                  any(x in token for x in ("\r", "\n", "\x00"))):
        raise HardwareEnrollmentError("invalid bearer format")
    data = json.dumps(body or {}, separators=(",", ":")).encode("utf-8")
    if len(data) > 8192:
        raise HardwareEnrollmentError("hardware device request too large")
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "Cache-Control": "no-store"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(_base_url(ctx) + path, data=data,
                                     headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=15) as response:
            raw = response.read(32769)
            if len(raw) > 32768:
                raise HardwareEnrollmentError("NANNET device response too large")
    except urllib.error.HTTPError as exc:
        raise HardwareEnrollmentError(
            "NANNET device request rejected: HTTP " + str(exc.code)) from None
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise HardwareEnrollmentError(
            "NANNET HTTPS device authority unavailable; no lease changed locally") from None
    try:
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError()
    except (ValueError, UnicodeError) as exc:
        raise HardwareEnrollmentError("invalid NANNET device response") from exc
    return decoded


def _report():
    return {
        "platform": "Hermes",
        "os_version": (platform.release() or "unknown")[:96],
        "app_version": "nanlmind-hermes/1.3.2",
        "abi": (platform.machine() or "unknown")[:96],
        "cpu_cores": max(1, min(os.cpu_count() or 1, 65535)),
        "capabilities": ["nan16de", "n16t", "nannet-hw-lease"],
    }


def _validate_binding(receipt, previous=None):
    if not isinstance(receipt, dict):
        raise HardwareEnrollmentError("device receipt must be an object")
    address = receipt.get("address", "")
    match = _ADDRESS.fullmatch(address) if isinstance(address, str) else None
    if not match or not 5 <= int(match.group(1)) <= 255:
        raise HardwareEnrollmentError("NANNET attempted to lease reserved or invalid HW identity")
    if previous and previous.get("address") and previous["address"] != address:
        raise HardwareEnrollmentError("hardware lease identity changed without release")
    point = int(match.group(1)) + 1
    if "node_point" in receipt and (isinstance(receipt["node_point"], bool)
                                     or receipt["node_point"] != point):
        raise HardwareEnrollmentError("NANNET address and node_point disagree")
    lease_id = receipt.get("lease_id", "")
    if not isinstance(lease_id, str) or not 1 <= len(lease_id) <= 128:
        raise HardwareEnrollmentError("missing or invalid hardware lease ID")
    if previous and previous.get("lease_id") and previous["lease_id"] != lease_id:
        raise HardwareEnrollmentError("active hardware lease ID changed")
    deadline = receipt.get("lease_expires_at")
    if (isinstance(deadline, bool) or not isinstance(deadline, int)
            or deadline <= 0):
        raise HardwareEnrollmentError("invalid lease deadline")
    if "token" in receipt and (not isinstance(receipt["token"], str)
                                or not receipt["token"].startswith(("nnt_", "hwh_"))):
        raise HardwareEnrollmentError("NANNET returned an invalid device bearer")
    return point


def status(ctx):
    state = _read(ctx)
    return {
        "configured": bool(_config(ctx, "nannet_auth_url")),
        "device_identity_present": bool(state.get("device_secret")),
        "bearer_present": bool(state.get("token")),
        "address": state.get("address"),
        "node_point": state.get("node_point"),
        "lease_id": state.get("lease_id"),
        "lease_expires_at": state.get("lease_expires_at"),
        "auth_method": "nannet-android-compatible-device-registration",
        "fixed_hermes_endpoint": "HW2",
        "dynamic_pool": "BS0/HW5-HW255",
        "network_port": "BS0:0",
        "hardware_root": "HW0:0",
        "native_universe_lease_active": False,
    }


def register(ctx, consent=False):
    if consent is not True:
        raise HardwareEnrollmentError("explicit device enrollment consent required")
    state = _read(ctx)
    if not state:
        state = {"schema": "nannet-hw-lease/1",
                 "device_secret": secrets.token_urlsafe(32)}
        # Preserve the installation secret even if the network request fails.
        _write(ctx, state)
    if state.get("token"):
        receipt = _http_json(ctx, "/v1/devices/rotate", {}, state["token"])
    else:
        receipt = _http_json(ctx, "/v1/devices/register",
                             {"device_secret": state["device_secret"],
                              "consent": True, "terms_version": TERMS,
                              "system_report": _report()})
    point = _validate_binding(receipt, state)
    token = receipt.get("token")
    if not token:
        raise HardwareEnrollmentError("NANNET did not issue a new hardware bearer")
    state.update({
        "device_id": receipt.get("device_id", state.get("device_id")),
        "address": receipt["address"], "node_point": point,
        "lease_id": receipt["lease_id"], "lease_expires_at": receipt["lease_expires_at"],
        "token": token,
    })
    _write(ctx, state)
    return {**status(ctx), "registered": True,
            "bearer_saved_encrypted": True}


def heartbeat(ctx):
    state = _read(ctx)
    if not state.get("token"):
        raise HardwareEnrollmentError("register this NANNET hardware node before heartbeat")
    receipt = _http_json(ctx, "/v1/devices/heartbeat", {}, state["token"])
    _validate_binding(receipt, state)
    state["lease_expires_at"] = receipt["lease_expires_at"]
    if receipt.get("token"):
        state["token"] = receipt["token"]
    _write(ctx, state)
    return {**status(ctx), "renewed": True}


def release(ctx):
    state = _read(ctx)
    if not state.get("token"):
        raise HardwareEnrollmentError("no active hardware token to release")
    receipt = _http_json(ctx, "/v1/devices/release", {}, state["token"])
    if receipt.get("address") != state.get("address") or not (
            receipt.get("released") is True or receipt.get("token_revoked") is True):
        raise HardwareEnrollmentError("NANNET did not confirm revocation of the bound HW identity")
    state["last_released_address"] = state.get("address")
    for key in ("token", "lease_id", "address", "node_point", "lease_expires_at"):
        state.pop(key, None)
    _write(ctx, state)
    return {**status(ctx), "released": True, "remote_confirmed": True}


def report(ctx, report_type, summary, details=""):
    state = _read(ctx)
    if not state.get("token"):
        raise HardwareEnrollmentError("device registration is required before a report")
    if report_type not in ("device_error", "system_error", "network_error",
                           "security_concern", "other"):
        raise HardwareEnrollmentError("unrecognized report type")
    if not isinstance(summary, str) or not 1 <= len(summary) <= 240:
        raise HardwareEnrollmentError("report summary must contain 1..240 characters")
    if not isinstance(details, str) or len(details) > 4000:
        raise HardwareEnrollmentError("report details must be at most 4000 characters")
    receipt = _http_json(ctx, "/v1/devices/report",
                         {"report_type": report_type, "summary": summary,
                          "details": details, "system_report": {}}, state["token"])
    if receipt.get("accepted") is not True and receipt.get("state") != "received":
        raise HardwareEnrollmentError("NANNET did not acknowledge report")
    return {"reported": True, "report_id": receipt.get("report_id"),
            "address": state.get("address")}
