"""Hermes hardware identity/key lifecycle.

Priority:
1. HW_HERMES_API from the host/system key store;
2. one persisted Hermes hardware key in the plugin state directory;
3. auto-register one Hermes hardware identity with NANNET and persist its key.

Android test/device tokens use a separate namespace and are never reused here.
"""
from __future__ import annotations

import json
import os
import platform
import secrets
import urllib.request
from pathlib import Path

TERMS_VERSION = "nannet-device-registration-v1"


class HardwareKey:
    def __init__(self, key: str, address: str, source: str):
        self.key = key
        self.address = address
        self.source = source


def _post(url: str, path: str, payload: dict, token: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _path(state_dir: str | Path) -> Path:
    return Path(state_dir).expanduser() / "hw_hermes_api.json"


def ensure_hardware_key(portal_url: str, state_dir: str | Path) -> HardwareKey:
    env_key = os.environ.get("HW_HERMES_API", "").strip()
    if env_key:
        return HardwareKey(env_key, os.environ.get("HW_HERMES_ADDRESS", ""), "environment")
    path = _path(state_dir)
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            if isinstance(doc.get("key"), str) and doc["key"]:
                return HardwareKey(doc["key"], doc.get("address", ""), "stored")
        except Exception:
            pass
    if not portal_url:
        raise RuntimeError("NANNET_AUTH_URL is required to register Hermes hardware")
    secret = secrets.token_urlsafe(48)
    result = _post(portal_url, "/v1/hermes/register", {
        "consent": True,
        "terms_version": TERMS_VERSION,
        "device_secret": secret,
        "hardware_kind": "hermes",
        "system_report": {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "component": "nanlmind-hermes",
        },
    })
    key = result.get("token", "")
    if not isinstance(key, str) or not key.startswith("hwh_"):
        raise RuntimeError("NANNET did not issue a Hermes hardware key")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "key": key,
        "address": result.get("address", ""),
        "device_id": result.get("device_id", ""),
        "secret": secret,
        "created_at": int(__import__("time").time()),
    }, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return HardwareKey(key, result.get("address", ""), "registered")


def authorize(portal_url: str, scope: str, key: HardwareKey) -> bool:
    result = _post(portal_url, "/v1/authorize", {"scope": scope}, key.key)
    if not result.get("authorized") or result.get("scope") != scope:
        raise RuntimeError("NANNET rejected Hermes hardware scope")
    return True
