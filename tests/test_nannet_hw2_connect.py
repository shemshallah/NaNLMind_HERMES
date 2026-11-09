"""Exercise the HW2 operator probe against a real loopback authority."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

from nanlmind_hermes import auth
from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.http_api import make_server
from nanlmind_hermes.pairing import PairingAuthority

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nannet_hw2_connect.py"
HW2_ADDRESS = "nan://GM0:SC0:MC0:BS0:0:HW2:0"
FORBIDDEN_KEYS = {
    "bearer", "private_key", "master_key", "pairing_code",
    "signature", "receipt_mac", "mac",
}


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _all_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


class Hw2ConnectTests(unittest.TestCase):
    def setUp(self):
        guard = warnings.catch_warnings()
        guard.__enter__()
        warnings.simplefilter("ignore", ResourceWarning)
        self.addCleanup(guard.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.authority_master = secrets.token_bytes(32)
        self.authority = PairingAuthority(
            self.root / "authority.sqlite", self.authority_master
        )
        self.server = make_server(self.authority, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

        self.master = secrets.token_bytes(32)
        self.master_b64 = _b64(self.master)
        self.master_path = self.root / "client-master.key"
        self.master_path.write_text(self.master_b64 + "\n", encoding="ascii")
        self.master_path.chmod(0o600)

        self.issuer_path = self.root / "issuer.pub"
        self.issuer_path.write_text(
            self.authority.issuer_public_key() + "\n", encoding="ascii"
        )
        self.state_dir = self.root / "state"

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _command(self, *extra, authority=None, issuer_path=None, master_path=None):
        return [
            sys.executable,
            str(SCRIPT),
            "--authority", authority or self.url,
            "--issuer-key-file", str(issuer_path or self.issuer_path),
            "--master-key-file", str(master_path or self.master_path),
            "--state-dir", str(self.state_dir),
            *extra,
        ]

    def _run(self, *extra, **kwargs):
        return subprocess.run(
            self._command(*extra, **kwargs),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _pairing_code(self):
        return self.authority.create_pairing_code(
            ["universe:read", "training:execute"], ttl_seconds=600
        )["pairing_code"]

    def _combined_output(self, completed, report_path=None):
        text = completed.stdout + completed.stderr
        if report_path and Path(report_path).exists():
            text += Path(report_path).read_text(encoding="utf-8")
        return text

    def _assert_no_secrets(self, text, pairing_code="", token="", private_raw=b""):
        self.assertNotIn(self.master_b64, text)
        if pairing_code:
            self.assertNotIn(pairing_code, text)
        if token:
            self.assertNotIn(token, text)
        if private_raw:
            self.assertNotIn(private_raw.hex(), text)
            self.assertNotIn(_b64(private_raw), text)

    def test_missing_or_invalid_master_key_fails_closed(self):
        missing = self.root / "does-not-exist.key"
        completed = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--authority", "http://127.0.0.1:1",
                "--issuer-key-file", str(self.issuer_path),
                "--master-key-file", str(missing),
                "--state-dir", str(self.state_dir),
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 1)
        error = json.loads(completed.stderr)
        self.assertFalse(error["ok"])
        self.assertEqual(error["error"]["code"], "missing_master_key")
        self.assertNotIn(self.master_b64, completed.stdout + completed.stderr)

        bad_value = "not-a-32-byte-key-and-must-not-be-echoed"
        bad_path = self.root / "bad-master.key"
        bad_path.write_text(bad_value + "\n", encoding="ascii")
        bad_path.chmod(0o600)
        completed = self._run(master_path=bad_path, authority="http://127.0.0.1:1")
        self.assertEqual(completed.returncode, 1)
        error = json.loads(completed.stderr)
        self.assertEqual(error["error"]["code"], "invalid_key")
        self.assertNotIn(bad_value, completed.stdout + completed.stderr)

    def test_issuer_mismatch_fails_closed_before_token_storage(self):
        code = self._pairing_code()
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        wrong_issuer = self.root / "wrong-issuer.pub"
        wrong_issuer.write_text(_b64(other) + "\n", encoding="ascii")

        completed = self._run("--pairing-code", code, issuer_path=wrong_issuer)
        self.assertEqual(completed.returncode, 1)
        error = json.loads(completed.stderr)
        self.assertFalse(error["ok"])
        self.assertEqual(error["error"]["code"], "pairing")
        self.assertIn("issuer", error["error"]["message"])
        self._assert_no_secrets(completed.stdout + completed.stderr, code)
        self.assertFalse(
            (self.state_dir / "universe" / "nannet_token.json").exists()
        )

    def test_successful_pair_signed_authorize_and_sanitized_report(self):
        code = self._pairing_code()
        report_path = self.root / "receipt.json"
        completed = self._run(
            "--pairing-code", code, "--report", str(report_path)
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["schema"], "nannet-hw2-connect-receipt/1")
        self.assertTrue(receipt["ok"])
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["scopes"], ["training:execute", "universe:read"])
        self.assertTrue(receipt["token"]["valid"])
        self.assertGreater(receipt["token"]["remaining_seconds"], 0)
        self.assertTrue(receipt["authorization"]["authorized"])
        self.assertEqual(receipt["authorization"]["signed_requests"], 1)
        self.assertEqual(receipt["hw2"]["address"], HW2_ADDRESS)
        self.assertEqual(
            receipt["hw2"]["device_client_address"],
            "nan://GM0:SC0:MC0:BS0:HW2",
        )
        self.assertEqual(receipt["hw2"]["node_point"], 3)
        self.assertEqual(receipt["hw2"]["network_port"], "BS0:0")
        self.assertTrue(receipt["hw2"]["distinct_from_local_head"])
        self.assertTrue(receipt["hw2"]["distinct_from_runner_port"])
        self.assertFalse(receipt["canonical_merge"]["enabled"])
        self.assertFalse(receipt["canonical_merge"]["attempted"])

        issuer_raw = auth._unb64(self.authority.issuer_public_key())
        self.assertEqual(
            receipt["issuer"]["fingerprint_sha256_prefix"],
            hashlib.sha256(issuer_raw).hexdigest()[:16],
        )
        self.assertTrue(receipt["issuer"]["pinned"])
        self.assertTrue(receipt["issuer"]["verified_during_pair"])

        stored = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(stored, receipt)
        mode = report_path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

        universe = self.state_dir / "universe"
        doc, private = auth.load_installation(universe, self.master)
        self.assertEqual(receipt["installation_id"], doc["installation_id"])
        envelope = json.loads((universe / "nannet_token.json").read_text())
        token = auth.decrypt_payload(
            self.master, envelope,
            ("nannet-token-v1:" + doc["installation_id"]).encode(),
        ).decode()
        private_raw = private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        text = self._combined_output(completed, report_path)
        self._assert_no_secrets(text, code, token, private_raw)
        self.assertNotIn("Bearer ", text)
        lowered = {str(key).lower() for key in _all_keys(receipt)}
        self.assertFalse(lowered & FORBIDDEN_KEYS)

        with sqlite3.connect(self.authority.database) as db:
            count = db.execute("SELECT COUNT(*) FROM request_nonces").fetchone()[0]
        self.assertEqual(count, 1)

    def test_no_secret_appears_in_failed_pairing_output(self):
        invalid_code = "nnh_invalid_code_that_must_not_be_echoed"
        completed = self._run("--pairing-code", invalid_code)
        self.assertEqual(completed.returncode, 1)
        text = completed.stdout + completed.stderr
        error = json.loads(completed.stderr)
        self.assertEqual(error["error"]["code"], "pairing")
        self._assert_no_secrets(text, invalid_code)
        self.assertNotIn("nnt_", text)

    def test_register_hw2_never_fabricates_reserved_binding(self):
        code = self._pairing_code()
        completed = self._run("--pairing-code", code, "--register-hw2")
        self.assertEqual(completed.returncode, 2, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertTrue(receipt["ok"])
        self.assertFalse(receipt["complete"])
        hw2 = receipt["hw2"]
        self.assertTrue(hw2["requested"])
        self.assertIn(hw2["status"], ("not_hw2", "blocked_reserved_slot"))
        self.assertNotEqual(hw2["status"], "registered")
        self.assertEqual(hw2["address"], HW2_ADDRESS)
        self.assertEqual(hw2["node_point"], 3)
        self.assertIn("operator_action", hw2)
        if hw2["status"] == "not_hw2":
            self.assertNotIn("HW2", hw2["authority_returned_address"])
            self.assertTrue(hw2["dynamic_lease_released"])
        self._assert_no_secrets(completed.stdout + completed.stderr, code)
        with sqlite3.connect(self.authority.database) as db:
            count = db.execute("SELECT COUNT(*) FROM request_nonces").fetchone()[0]
        self.assertEqual(count, 1)

    def test_non_loopback_plain_http_is_rejected_without_network(self):
        completed = self._run(authority="http://nannet.example")
        self.assertEqual(completed.returncode, 1)
        error = json.loads(completed.stderr)
        self.assertEqual(error["error"]["code"], "unsafe_authority")
        self.assertNotIn(self.master_b64, completed.stdout + completed.stderr)

    def test_positional_secret_is_rejected_without_echo(self):
        secret = "positional-secret-that-must-not-appear"
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), secret],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 1)
        text = completed.stdout + completed.stderr
        self.assertNotIn(secret, text)
        error = json.loads(completed.stderr)
        self.assertEqual(error["error"]["code"], "usage")


if __name__ == "__main__":
    unittest.main()
