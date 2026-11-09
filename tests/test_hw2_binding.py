"""Fixed HW2 NANNET binding through a real loopback authority.

These tests exercise the authority-owned permanent HERMES endpoint. They never
contact a production network, and they assert that no bearer or private key is
leaked through client receipts, status output, or encrypted state files.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import secrets
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes import auth, auth_tools
from nanlmind_hermes import nannet_device_client as client
from nanlmind_hermes.crypto_compat import serialization
from nanlmind_hermes.devices import (
    DeviceRegistry, FIXED_HW2_ADDRESS, FIXED_HW2_BIND_PATH,
    FIXED_HW2_NODE_POINT, FIXED_HW2_REVOKE_PATH, FIXED_HW2_STATUS_PATH,
    TERMS_VERSION,
)
from nanlmind_hermes.http_api import make_server
from nanlmind_hermes.pairing import PairingAuthority

ADMIN_SCOPES = ["installation:admin", "training:execute", "universe:read"]


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class FakeContext:
    def __init__(self, state_dir, url, issuer):
        self._config = {
            "state_dir": str(state_dir),
            "nannet_auth_url": url,
            "nannet_issuer_public_key": issuer,
        }

    def get_config(self, name, default=None):
        return self._config.get(name, default)


class FixedHw2BindingTests(unittest.TestCase):
    def setUp(self):
        guard = warnings.catch_warnings()
        guard.__enter__()
        warnings.simplefilter("ignore", ResourceWarning)
        self.addCleanup(guard.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.authority = PairingAuthority(
            self.root / "authority.sqlite", secrets.token_bytes(32))
        self.server = make_server(self.authority, "127.0.0.1", 0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def _installation(self, name, role="admin", scopes=None):
        state_dir = self.root / name
        universe = state_dir / "universe"
        master = secrets.token_bytes(32)
        ctx = FakeContext(state_dir, self.url, self.authority.issuer_public_key())
        with patch.object(auth_tools, "get_secret", return_value=_b64(master)):
            auth.create_installation(universe, master)
            code = self.authority.create_pairing_code(
                scopes or ADMIN_SCOPES, ttl_seconds=600, role=role
            )["pairing_code"]
            auth_tools.auth_pair(ctx, code)
        doc, private = auth.load_installation(universe, master)
        return ctx, master, doc, private

    def _installation_token(self, ctx, master, doc):
        root = Path(ctx.get_config("state_dir")) / "universe"
        envelope = json.loads((root / "nannet_token.json").read_text("utf-8"))
        return auth.decrypt_payload(
            master, envelope,
            ("nannet-token-v1:" + doc["installation_id"]).encode()).decode("utf-8")

    def _fixed_state(self, ctx, master):
        root = Path(ctx.get_config("state_dir")) / "universe"
        path = root / "nannet_hw2_binding.json"
        envelope = json.loads(path.read_text("utf-8"))
        return path, json.loads(auth.decrypt_payload(
            master, envelope, client._FIXED_AAD).decode("utf-8"))

    def _post(self, path, body, token="", extra_headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(
            self.url + path, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {}
            return exc.code, payload

    def _signed_post(self, path, token, private, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        envelope = auth.sign_request(private, "POST", path, data)
        return self._post(path, data, token, {
            "X-NANNET-Timestamp": str(envelope["timestamp"]),
            "X-NANNET-Nonce": envelope["nonce"],
            "X-NANNET-Body-SHA256": envelope["body_sha256"],
            "X-NANNET-Signature": envelope["signature"],
        })

    def _fixed_row_count(self):
        with sqlite3.connect(self.authority.database) as db:
            return db.execute(
                "SELECT COUNT(*) FROM fixed_hw_bindings").fetchone()[0]

    def test_admin_fixed_scope_binds_canonical_hw2_and_guest_token(self):
        ctx, master, doc, private = self._installation("admin-a")
        out, err = io.StringIO(), io.StringIO()
        with patch.object(client, "_master_key", return_value=master):
            with self.assertRaisesRegex(client.HardwareEnrollmentError, "consent"):
                client.register_fixed_hw2(ctx)
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                receipt = client.register_fixed_hw2(ctx, consent=True)
                local = client.fixed_hw2_status(ctx)
                remote = client.fixed_hw2_remote_status(ctx)
                report = client.report_fixed_hw2(
                    ctx, "network_error", "loopback fixed HW2 probe")

        self.assertEqual(receipt["installation_id"], doc["installation_id"])
        self.assertEqual(receipt["address"], FIXED_HW2_ADDRESS)
        self.assertEqual(receipt["node_point"], FIXED_HW2_NODE_POINT)
        self.assertEqual(receipt["lease_id"], local["lease_id"])
        self.assertEqual(receipt["scopes"], sorted(set(receipt["scopes"])))
        self.assertIn("training:execute", receipt["scopes"])
        self.assertTrue(receipt["permanent"])
        self.assertTrue(receipt["bearer_saved_encrypted"])
        self.assertNotIn("token", receipt)
        self.assertTrue(local["fixed_hw2_bound"])
        self.assertTrue(remote["remote_verified"])
        self.assertTrue(report["reported"])

        path, state = self._fixed_state(ctx, master)
        token = state["token"]
        self.assertTrue(token.startswith("nnt_"))
        claims = self.authority.verify_access_token(
            token[4:], required_scope="training:execute")
        self.assertEqual(claims["role"], "guest")
        self.assertEqual(claims["sub"], doc["installation_id"])

        with sqlite3.connect(self.authority.database) as db:
            row = db.execute(
                "SELECT device_id,address,node_point,hw_slot,revoked_at "
                "FROM fixed_hw_bindings WHERE lease_id=?",
                (receipt["lease_id"],)).fetchone()
        self.assertEqual(row, (doc["installation_id"], FIXED_HW2_ADDRESS, 3, 2, None))

        surfaces = json.dumps([receipt, local, remote, report], sort_keys=True)
        encrypted = path.read_text("utf-8")
        private_raw = private.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption())
        for secret in (token, _b64(master), private_raw.hex(),
                       _b64(private_raw), "Bearer ", "private_key"):
            self.assertNotIn(secret, surfaces)
            self.assertNotIn(secret, encrypted)
            self.assertNotIn(secret, out.getvalue() + err.getvalue())

    def test_guest_pairing_is_denied_binding(self):
        ctx, master, doc, private = self._installation(
            "guest-b", role="guest", scopes=["training:execute", "universe:read"])
        token = self._installation_token(ctx, master, doc)
        status, payload = self._signed_post(
            FIXED_HW2_BIND_PATH, token, private, b"{}")
        self.assertIn(status, (401, 403))
        self.assertNotIn("token", payload)

        with patch.object(client, "_master_key", return_value=master):
            with self.assertRaisesRegex(
                    client.HardwareEnrollmentError, "admin|installation:admin"):
                client.register_fixed_hw2(ctx, consent=True)
        state_path = (Path(ctx.get_config("state_dir")) / "universe" /
                      "nannet_hw2_binding.json")
        self.assertFalse(state_path.exists())
        self.assertEqual(self._fixed_row_count(), 0)

    def test_binding_is_idempotent_for_same_installation(self):
        ctx, master, doc, private = self._installation("admin-idem")
        with patch.object(client, "_master_key", return_value=master):
            first = client.register_fixed_hw2(ctx, consent=True)
            _path, first_state = self._fixed_state(ctx, master)
            second = client.register_fixed_hw2(ctx, consent=True)
            _path, second_state = self._fixed_state(ctx, master)

        self.assertEqual(second["lease_id"], first["lease_id"])
        self.assertEqual(second["address"], FIXED_HW2_ADDRESS)
        self.assertEqual(second["node_point"], FIXED_HW2_NODE_POINT)
        self.assertEqual(second["installation_id"], doc["installation_id"])
        self.assertNotEqual(first_state["token"], second_state["token"])
        self.assertEqual(self._fixed_row_count(), 1)
        with sqlite3.connect(self.authority.database) as db:
            active = db.execute(
                "SELECT COUNT(*) FROM fixed_hw_bindings "
                "WHERE hw_slot=2 AND revoked_at IS NULL").fetchone()[0]
        self.assertEqual(active, 1)

        old_status, _ = self._post(
            FIXED_HW2_STATUS_PATH, {}, first_state["token"])
        new_status, body = self._post(
            FIXED_HW2_STATUS_PATH, {}, second_state["token"])
        self.assertEqual(old_status, 401)
        self.assertEqual(new_status, 200)
        self.assertEqual(body["address"], FIXED_HW2_ADDRESS)
        self.assertNotIn("token", body)

    def test_another_installation_cannot_claim_active_hw2_until_revoked(self):
        ctx_a, master_a, doc_a, _private_a = self._installation("admin-owner")
        ctx_b, master_b, doc_b, private_b = self._installation("admin-other")
        with patch.object(client, "_master_key", return_value=master_a):
            owner = client.register_fixed_hw2(ctx_a, consent=True)

        token_b = self._installation_token(ctx_b, master_b, doc_b)
        status, payload = self._signed_post(
            FIXED_HW2_BIND_PATH, token_b, private_b, b"{}")
        self.assertEqual(status, 400)
        self.assertIn("already bound", payload["error"])
        with patch.object(client, "_master_key", return_value=master_b):
            with self.assertRaisesRegex(
                    client.HardwareEnrollmentError, "already bound"):
                client.register_fixed_hw2(ctx_b, consent=True)
        b_state = (Path(ctx_b.get_config("state_dir")) / "universe" /
                   "nannet_hw2_binding.json")
        self.assertFalse(b_state.exists())

        token_a = self._installation_token(ctx_a, master_a, doc_a)
        revoke_body = json.dumps({"installation_id": doc_a["installation_id"]}).encode()
        status, revoked = self._signed_post(
            FIXED_HW2_REVOKE_PATH, token_a,
            auth.load_installation(
                Path(ctx_a.get_config("state_dir")) / "universe", master_a)[1],
            revoke_body)
        self.assertEqual(status, 200, revoked)
        self.assertTrue(revoked["revoked"])
        self.assertEqual(revoked["installation_id"], doc_a["installation_id"])

        with patch.object(client, "_master_key", return_value=master_b):
            replacement = client.register_fixed_hw2(ctx_b, consent=True)
        self.assertEqual(replacement["installation_id"], doc_b["installation_id"])
        self.assertEqual(replacement["address"], FIXED_HW2_ADDRESS)
        self.assertEqual(replacement["node_point"], FIXED_HW2_NODE_POINT)
        self.assertNotEqual(replacement["lease_id"], owner["lease_id"])
        with sqlite3.connect(self.authority.database) as db:
            active = db.execute(
                "SELECT device_id FROM fixed_hw_bindings "
                "WHERE hw_slot=2 AND revoked_at IS NULL").fetchall()
            total = db.execute(
                "SELECT COUNT(*) FROM fixed_hw_bindings").fetchone()[0]
        self.assertEqual(active, [(doc_b["installation_id"],)])
        self.assertEqual(total, 2)

    def test_normal_device_registration_never_allocates_reserved_hw0_hw4(self):
        registry = DeviceRegistry(self.authority.database, authority=self.authority)
        now = int(time.time())
        direct = registry.register({
            "device_secret": secrets.token_urlsafe(32),
            "consent": True,
            "terms_version": TERMS_VERSION,
            "system_report": {"platform": "Hermes"},
            "address": FIXED_HW2_ADDRESS,
            "node_point": FIXED_HW2_NODE_POINT,
        }, now=now)
        self.assertGreaterEqual(int(direct["address"].rsplit("HW", 1)[1]), 5)
        self.assertNotIn(":HW2:", direct["address"])

        status, via_http = self._post("/v1/devices/register", {
            "device_secret": secrets.token_urlsafe(32),
            "consent": True,
            "terms_version": TERMS_VERSION,
            "system_report": {"platform": "Hermes"},
            "address": FIXED_HW2_ADDRESS,
            "node_point": FIXED_HW2_NODE_POINT,
        })
        self.assertEqual(status, 201)
        self.assertGreaterEqual(int(via_http["address"].rsplit("HW", 1)[1]), 5)
        self.assertNotIn(":HW2:", via_http["address"])

    def test_authority_rejects_caller_supplied_address_or_node_point(self):
        ctx, master, doc, private = self._installation("admin-owned-values")
        token = self._installation_token(ctx, master, doc)
        for body in (
            {"address": "nan://GM0:SC0:MC0:BS0:0:HW3:0"},
            {"address": FIXED_HW2_ADDRESS, "node_point": 4},
            {"node_point": True},
            {"address": "nan://GM0:SC0:MC0:BS0:0:HW2:0", "node_point": 3,
             "lease_id": "caller-selected"},
        ):
            with self.subTest(body=body):
                status, payload = self._signed_post(
                    FIXED_HW2_BIND_PATH, token, private,
                    json.dumps(body).encode())
                if body.get("lease_id"):
                    self.assertEqual(status, 200, payload)
                    self.assertNotEqual(payload["lease_id"], "caller-selected")
                    self.assertEqual(payload["address"], FIXED_HW2_ADDRESS)
                    self.assertEqual(payload["node_point"], FIXED_HW2_NODE_POINT)
                else:
                    self.assertEqual(status, 400, payload)
                    self.assertNotIn("token", payload)
        self.assertEqual(self._fixed_row_count(), 1)

    def test_binding_fails_closed_on_issuer_mismatch_and_missing_pairing(self):
        ctx, master, doc, private = self._installation("admin-pin")
        wrong = FakeContext(
            ctx.get_config("state_dir"), self.url,
            _b64(auth.Ed25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw)))
        with patch.object(client, "_master_key", return_value=master):
            with self.assertRaisesRegex(client.HardwareEnrollmentError, "pinned issuer"):
                client.register_fixed_hw2(wrong, consent=True)
        self.assertEqual(self._fixed_row_count(), 0)

        empty = self.root / "unpaired"
        (empty / "universe").mkdir(parents=True)
        unpaired = FakeContext(empty, self.url, self.authority.issuer_public_key())
        with patch.object(client, "_master_key", return_value=master):
            with self.assertRaisesRegex(
                    client.HardwareEnrollmentError, "pair|identity"):
                client.register_fixed_hw2(unpaired, consent=True)
        self.assertEqual(self._fixed_row_count(), 0)


if __name__ == "__main__":
    unittest.main()
