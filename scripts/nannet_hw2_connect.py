#!/usr/bin/env python3
"""Operator setup/probe for the permanent HERMES HW2 NANNET endpoint.

This is a one-shot setup and connectivity probe, not a daemon. It creates or
unlocks the local installation identity, pins the authority issuer, pairs once
with a short-lived pairing code, verifies the guest scopes, optionally asks the
existing device client to enroll the fixed HW2 hardware node, then makes one
installation-signed ``POST /v1/authorize`` request and prints a sanitized JSON
receipt.

The receipt never contains the master key, installation private key, pairing
code, bearer token, request signature, or a receipt MAC. Secrets are accepted
only through files/options, never as positional arguments.

HW2 is the permanent HERMES hardware endpoint at node point 3. It is not local
head 3 and it is not the shared ``BS0:0`` runner-ring port.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes import auth, auth_tools, nannet_device_client  # noqa: E402
from nanlmind_hermes.crypto_compat import Ed25519PublicKey  # noqa: E402

SCHEMA = "nannet-hw2-connect-receipt/1"
REQUIRED_SCOPES = ("training:execute", "universe:read")
DEFAULT_STATE_DIR = "~/.local/share/nanlmind-hermes"

HW2_PERMANENT_ENDPOINT = "HW2"
HW2_ADDRESS = "nan://GM0:SC0:MC0:BS0:0:HW2:0"
HW2_LEGACY_ADDRESS = "nan://HW2:BS0:MC0:SC0:GM0"
HW2_DEVICE_ADDRESS = "nan://GM0:SC0:MC0:BS0:HW2"
HW2_NODE_POINT = 3
NANNET_PORT = "BS0:0"

HW2_OPERATOR_ACTION = (
    "Permanent HW2 is a reserved HERMES endpoint, not a dynamically allocatable "
    "HW5-HW255 device lease. The NANNET authority operator must bind this "
    "installation to the fixed HW2/node-point-3 endpoint through the reserved "
    "endpoint provisioning path; POST /v1/devices/register currently allocates "
    "only HW5-HW255. Re-run --register-hw2 after that binding exists."
)

_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


class ConnectError(RuntimeError):
    """A fail-closed operator setup or probe error."""

    def __init__(self, message: str, code: str = "connect_failed"):
        super().__init__(message)
        self.code = code


class _Parser(argparse.ArgumentParser):
    """Argument parser that never echoes unrecognized values to stderr."""

    def error(self, message):  # noqa: D401 - argparse override
        raise ConnectError(
            "invalid command line options; secrets are accepted only through "
            "the documented file/option arguments, never as positional values",
            "usage",
        )


class OperatorContext:
    """Minimal plugin context used by the existing auth/device helpers."""

    def __init__(self, state_dir: Path, authority: str, issuer_key: str):
        self._config = {
            "state_dir": str(state_dir),
            "nannet_auth_url": authority,
            "nannet_issuer_public_key": issuer_key,
        }

    def get_config(self, name, default=None):
        return self._config.get(name, default)


def _decode_key(value: str, label: str) -> bytes:
    text = str(value or "").strip()
    if not text or not _KEY_RE.fullmatch(text):
        raise ConnectError(label + " must be unpadded base64url", "invalid_key")
    padded = text + "=" * (-len(text) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        raise ConnectError(label + " is not valid base64url", "invalid_key") from None
    if len(raw) != 32:
        raise ConnectError(label + " must decode to exactly 32 bytes", "invalid_key")
    return raw


def _read_master_key(path_value: str) -> bytes:
    if not path_value:
        raise ConnectError("--master-key-file is required", "missing_master_key")
    path = Path(path_value).expanduser()
    try:
        info = path.stat()
    except OSError:
        raise ConnectError(
            "master key file is missing or unreadable", "missing_master_key"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise ConnectError("master key path must be a regular file", "invalid_master_key")
    if info.st_mode & 0o077:
        raise ConnectError(
            "master key file must be owner-only; run chmod 600 on it first",
            "unsafe_master_key",
        )
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        raise ConnectError(
            "master key file must be readable ASCII base64url", "invalid_master_key"
        ) from None
    return _decode_key(text, "master key")


def _read_issuer_key(args) -> tuple[bytes, str, str]:
    if bool(args.issuer_key_file) == bool(args.issuer_key):
        raise ConnectError(
            "provide exactly one of --issuer-key-file or --issuer-key", "missing_issuer_key"
        )
    if args.issuer_key_file:
        path = Path(args.issuer_key_file).expanduser()
        try:
            value = path.read_text(encoding="ascii")
        except (OSError, UnicodeError):
            raise ConnectError(
                "issuer key file is missing or unreadable", "missing_issuer_key"
            ) from None
    else:
        value = args.issuer_key
    raw = _decode_key(value, "issuer public key")
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    fingerprint = hashlib.sha256(raw).hexdigest()[:16]
    return raw, encoded, fingerprint


def _validate_authority(value: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        raise ConnectError("--authority is required", "missing_authority")
    parts = urlsplit(raw)
    try:
        port = parts.port
    except ValueError:
        raise ConnectError("authority URL contains an invalid port", "invalid_authority") from None
    try:
        loopback = ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        loopback = (parts.hostname or "").lower() == "localhost"
    if parts.scheme not in ("https", "http"):
        raise ConnectError("authority must use HTTPS (HTTP only on loopback)", "invalid_authority")
    if parts.scheme == "http" and not loopback:
        raise ConnectError(
            "plain HTTP is allowed only for a loopback authority", "unsafe_authority"
        )
    if (not parts.hostname or parts.username or parts.password
            or parts.path or parts.query or parts.fragment
            or port is not None and not 1 <= port <= 65535):
        raise ConnectError(
            "authority must be a bare origin with no credentials, path, query, or fragment",
            "invalid_authority",
        )
    return raw


def _redact(value: str, secrets: list[str]) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _safe_exception(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:300] if text else type(exc).__name__


@contextmanager
def _patched_secret(master: bytes):
    encoded = base64.urlsafe_b64encode(master).rstrip(b"=").decode("ascii")
    with patch.object(auth_tools, "get_secret", return_value=encoded):
        yield


def _ensure_identity(ctx, master: bytes):
    root = auth_tools._universe_dir(ctx)
    path = root / "installation.json"
    if path.exists():
        doc, private = auth.load_installation(root, master)
        return doc, private, False
    auth.create_installation(root, master)
    doc, private = auth.load_installation(root, master)
    return doc, private, True


def _load_claims(ctx, master: bytes, doc: dict, issuer_raw: bytes) -> dict:
    root = auth_tools._universe_dir(ctx)
    token_path = root / "nannet_token.json"
    if not token_path.exists():
        raise auth.AuthError("no paired token is stored; pass --pairing-code")
    try:
        envelope = json.loads(token_path.read_text(encoding="utf-8"))
        token = auth.decrypt_payload(
            master, envelope, ("nannet-token-v1:" + doc["installation_id"]).encode()
        ).decode("utf-8")
        claims = auth.verify_token(
            token, Ed25519PublicKey.from_public_bytes(issuer_raw)
        )
    except Exception as exc:
        if isinstance(exc, auth.AuthError):
            raise
        raise auth.AuthError("stored token cannot be decrypted or verified") from exc
    if claims.get("sub") != doc["installation_id"]:
        raise auth.AuthError("stored token belongs to a different installation")
    return claims


def _is_hw2_binding(address, node_point) -> bool:
    if isinstance(node_point, bool):
        return False
    if node_point is not None:
        try:
            if int(node_point) != HW2_NODE_POINT:
                return False
        except (TypeError, ValueError):
            return False
    return address in (HW2_ADDRESS, HW2_LEGACY_ADDRESS, HW2_DEVICE_ADDRESS)


def _probe_hw2(ctx, requested: bool) -> dict:
    result = {
        "requested": bool(requested),
        "permanent_endpoint": HW2_PERMANENT_ENDPOINT,
        "address": HW2_ADDRESS,
        "legacy_compatibility_address": HW2_LEGACY_ADDRESS,
        "device_client_address": HW2_DEVICE_ADDRESS,
        "node_point": HW2_NODE_POINT,
        "local_head_index_same_number": HW2_NODE_POINT,
        "network_port": NANNET_PORT,
        "distinct_from_local_head": True,
        "distinct_from_runner_port": True,
    }
    try:
        current = nannet_device_client.status(ctx)
    except Exception:
        current = {}
    result["existing_address"] = current.get("address")
    result["device_token_present"] = bool(current.get("bearer_present"))
    if not requested:
        result["status"] = "not_requested"
        return result
    try:
        receipt = nannet_device_client.register(ctx, consent=True)
    except Exception as exc:
        reason = _safe_exception(exc)
        result["status"] = (
            "blocked_reserved_slot" if "reserved or invalid" in reason else "blocked"
        )
        result["reason"] = reason
        result["operator_action"] = HW2_OPERATOR_ACTION
        return result
    address = receipt.get("address")
    point = receipt.get("node_point")
    if _is_hw2_binding(address, point):
        result.update({
            "status": "registered",
            "authority_bound_address": address,
            "lease_expires_at": receipt.get("lease_expires_at"),
        })
        return result
    released = False
    try:
        nannet_device_client.release(ctx)
        released = True
    except Exception:
        pass
    result.update({
        "status": "not_hw2",
        "authority_returned_address": address,
        "authority_returned_node_point": point,
        "dynamic_lease_released": released,
        "operator_action": HW2_OPERATOR_ACTION,
    })
    return result


def _write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    )
    temp = Path(handle.name)
    try:
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _emit(payload: dict, stream=None) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
          file=stream or sys.stdout)


def run(args):
    """Run the setup/probe flow and return (receipt, exit_code)."""
    authority = _validate_authority(args.authority)
    master = _read_master_key(args.master_key_file)
    issuer_raw, issuer_encoded, issuer_fingerprint = _read_issuer_key(args)
    ctx = OperatorContext(Path(args.state_dir).expanduser(), authority, issuer_encoded)
    secrets = [base64.urlsafe_b64encode(master).rstrip(b"=").decode("ascii"),
               str(args.pairing_code or "")]

    def fail(message, code):
        raise ConnectError(_redact(message, secrets), code)

    try:
        doc, _private, created = _ensure_identity(ctx, master)
    except Exception as exc:
        fail("cannot create or unlock the installation identity: " + _safe_exception(exc),
             "identity")

    paired_now = False
    with _patched_secret(master):
        if args.pairing_code:
            try:
                auth_tools.auth_pair(ctx, args.pairing_code)
            except Exception as exc:
                fail("pairing failed closed: " + _safe_exception(exc), "pairing")
            paired_now = True
        try:
            claims = _load_claims(ctx, master, doc, issuer_raw)
        except Exception as exc:
            fail("stored token is missing or invalid: " + _safe_exception(exc), "token")
        scopes = sorted(set(claims.get("scope", [])))
        missing = [scope for scope in REQUIRED_SCOPES if scope not in scopes]
        if missing:
            fail(
                "paired token lacks required scope(s): " + ", ".join(missing)
                + "; re-pair with a code granting " + ", ".join(REQUIRED_SCOPES),
                "scope",
            )
        hw2 = _probe_hw2(ctx, bool(args.register_hw2))
        try:
            auth_tools.require_scope(ctx, "training:execute")
        except Exception as exc:
            fail("signed /v1/authorize failed closed: " + _safe_exception(exc),
                 "authorize")

    now = int(time.time())
    complete = (not args.register_hw2) or hw2.get("status") == "registered"
    receipt = {
        "schema": SCHEMA,
        "ok": True,
        "complete": complete,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "authority": authority,
        "transport": "loopback-http" if authority.startswith("http://") else "https",
        "installation_id": doc["installation_id"],
        "identity": {
            "state": "created" if created else "loaded",
            "encrypted_at_rest": True,
        },
        "issuer": {
            "fingerprint_sha256_prefix": issuer_fingerprint,
            "pinned": True,
            "verified_during_pair": paired_now,
        },
        "scopes": scopes,
        "required_scopes": sorted(REQUIRED_SCOPES),
        "token": {
            "valid": True,
            "role": claims.get("role"),
            "expires_at": claims.get("exp"),
            "remaining_seconds": max(0, int(claims.get("exp", 0)) - now),
        },
        "authorization": {
            "endpoint": "/v1/authorize",
            "method": "POST",
            "signed": True,
            "signed_requests": 1,
            "scope": "training:execute",
            "authorized": True,
        },
        "hw2": hw2,
        "canonical_merge": {"enabled": False, "attempted": False},
        "boundary": (
            "One live pairing/authorization probe against the configured authority. "
            "This receipt is not a router universe lease, HW compute grant, proof of "
            "production deployment, or canonical merge."
        ),
    }
    return receipt, (0 if complete else 2)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--authority", default="", metavar="URL",
                        help="NANNET authority origin; HTTPS in production, HTTP only on loopback")
    issuer = parser.add_mutually_exclusive_group()
    issuer.add_argument("--issuer-key-file", default="", metavar="PATH",
                        help="file containing the pinned issuer Ed25519 public key (base64url)")
    issuer.add_argument("--issuer-key", default="", metavar="KEY",
                        help="pinned issuer Ed25519 public key (base64url, public value only)")
    parser.add_argument("--master-key-file", default="", metavar="PATH",
                        help="owner-only file containing the 32-byte base64url installation master key")
    parser.add_argument("--pairing-code", default="", metavar="CODE",
                        help="one-time guest pairing code; omit when a valid encrypted token already exists")
    parser.add_argument("--register-hw2", action="store_true",
                        help="ask the existing device client to enroll the fixed HW2/node-point-3 endpoint")
    parser.add_argument("--report", nargs="?", const="-", default="", metavar="PATH",
                        help="write the same sanitized JSON receipt to this owner-only file "
                             "(omit PATH or use '-' for stdout only)")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, metavar="PATH",
                        help="local plugin state directory (default: %(default)s)")
    return parser


def main(argv=None) -> int:
    args = None
    try:
        parser = build_parser()
        args, unknown = parser.parse_known_args(argv)
        if unknown:
            raise ConnectError(
                "positional or unrecognized arguments are not accepted; provide "
                "secrets only through --master-key-file and --pairing-code",
                "usage",
            )
        receipt, exit_code = run(args)
    except ConnectError as exc:
        _emit({"schema": SCHEMA, "ok": False, "complete": False,
               "error": {"code": exc.code, "message": str(exc)[:400]}}, sys.stderr)
        return 1
    except Exception as exc:  # no traceback or raw exception text: may contain secrets
        _emit({"schema": SCHEMA, "ok": False, "complete": False,
               "error": {"code": "internal_error", "message": type(exc).__name__}},
              sys.stderr)
        return 1

    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False)
    if args.report and args.report != "-":
        try:
            _write_report(Path(args.report).expanduser(), text + "\n")
        except OSError:
            _emit({"schema": SCHEMA, "ok": False, "complete": False,
                   "error": {"code": "report_write_failed",
                             "message": "could not write the sanitized report file"}},
                  sys.stderr)
            return 1
    print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
