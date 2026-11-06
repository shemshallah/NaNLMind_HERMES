"""Hermes-facing local installation identity handlers."""
from __future__ import annotations
import base64
import os
from .auth import create_installation, load_installation


def _universe_dir(ctx):
    try:
        configured = ctx.get_config("state_dir", default="~/.local/share/nanlmind-hermes")
    except Exception:
        configured = "~/.local/share/nanlmind-hermes"
    from pathlib import Path
    path = Path(str(configured)).expanduser() / "universe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def auth_init(ctx):
    encoded = os.environ.get("NANNET_MASTER_KEY", "")
    if not encoded:
        raise ValueError("NANNET_MASTER_KEY is not set; provision an independent random 256-bit key in the host OS secret store")
    try:
        master = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except Exception as exc:
        raise ValueError("NANNET_MASTER_KEY must be base64url encoded") from exc
    identity = create_installation(_universe_dir(ctx), master)
    return {"installation_id": identity["installation_id"], "public_key": identity["public_key"],
            "identity_path": identity["path"], "private_key_encrypted_at_rest": True,
            "master_key_returned": False, "pairing_authority_configured": False}


def auth_status(ctx):
    path = _universe_dir(ctx) / "installation.json"
    encoded = os.environ.get("NANNET_MASTER_KEY", "")
    result = {"identity_present": path.exists(), "master_key_present": bool(encoded),
              "pairing_authority_configured": False}
    if not path.exists():
        return result
    try:
        master = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        doc, _ = load_installation(path.parent, master)
        result.update({"installation_id": doc["installation_id"], "public_key": doc["public_key"],
                       "identity_unlocked": True})
    except Exception:
        result["identity_unlocked"] = False
    return result
