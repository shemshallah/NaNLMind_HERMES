"""HERMES router-universe lifecycle, encrypted persistence and head protection."""
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes import auth_tools, router_universe_client as client


INSTALL = "12345678-1111-4111-8111-123456789abc"


class Context:
    def __init__(self, state_dir):
        self.state_dir = str(state_dir)

    def get_config(self, name, default=None):
        return {
            "state_dir": self.state_dir,
            "nannet_auth_url": "https://nannet.example.test",
            "nannet_issuer_public_key": "issuer-key-from-trusted-channel",
        }.get(name, default)


def receipt(slot, generation=1, revision=1, expiry=None):
    return {
        "schema": "nanrouter.hermes-lease.v1",
        "issuer": "nanrouter/BS0:0",
        "network_port": "BS0:0",
        "hardware_root": "HW0:0",
        "installation_id": INSTALL,
        "slot": slot,
        "lease_id": 100 + slot,
        "universe_id": 200 + slot,
        "generation": generation,
        "revision": revision,
        "owner_tag": 300,
        "node_point": 3,
        "expires_at": int(time.time()) + 1800 if expiry is None else expiry,
        "heads": 255,
        "nannet_port_head": 255,
        "receipt_mac": "a" * 64,
    }


class RouterUniverseClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ctx = Context(self.temp.name)
        encoded = base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("=")
        self.secret = patch.object(auth_tools, "get_secret", return_value=encoded)
        self.secret.start()
        self.addCleanup(self.secret.stop)

    def test_multiple_universes_share_node_but_not_state_and_encrypt_receipts(self):
        result = [receipt(0), receipt(1)]
        with patch.object(client, "_post",
                          side_effect=[(INSTALL, {"receipt": result[0]}),
                                       (INSTALL, {"receipt": result[1]})]) as remote:
            a = client.open_universe(self.ctx, "alpha")
            b = client.open_universe(self.ctx, "beta")
        self.assertEqual(remote.call_count, 2)
        self.assertNotEqual(a["universe_id"], b["universe_id"])
        self.assertEqual(a["node_point"], b["node_point"])
        self.assertFalse(a["native_guest_activated"])
        state_path = Path(self.temp.name) / "universe" / "router_universe_leases.json"
        encrypted = state_path.read_text()
        self.assertNotIn('"receipt_mac"', encrypted)
        self.assertNotIn(INSTALL, encrypted)
        self.assertEqual(len(client.status(self.ctx)["leases"]), 2)
        with patch.object(client, "_post", return_value=(INSTALL, {"route": {
            "admitted": True, "lease_id": 100, "universe_id": 200,
            "slot": 0, "node_point": 3, "owner_tag": 300,
            "generation": 1, "expires_at": result[0]["expires_at"],
            "head_index": 254, "network_port": "BS0:0",
        }})):
            resolved = client.resolve(self.ctx, "alpha", 254)
        self.assertTrue(resolved["admitted"])
        self.assertFalse(resolved["execution_committed"])

    def test_reject_reserved_port_bad_alias_and_identity_mismatched_receipts(self):
        with self.assertRaises(client.UniverseLeaseError):
            client.open_universe(self.ctx, "../traverse")
        with self.assertRaises(client.UniverseLeaseError):
            client.resolve(self.ctx, "primary", 255)
        with self.assertRaises(client.UniverseLeaseError):
            client.resolve(self.ctx, "primary", True)
        wrong = {**receipt(0), "installation_id": "someone-else"}
        with self.assertRaises(client.UniverseLeaseError):
            client._valid_receipt(wrong, INSTALL)
        with self.assertRaises(client.UniverseLeaseError):
            client._valid_receipt({**receipt(0), "node_point": 0}, INSTALL)

    def test_renewal_old_receipt_rollbacks_rejected_and_release(self):
        original = receipt(0)
        updated = {**original, "revision": 2, "expires_at": original["expires_at"] + 1000}
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": original})):
            client.open_universe(self.ctx, "main")
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": updated})):
            data = client.renew(self.ctx, "main")
        self.assertEqual(data["revision"], 2)
        self.assertEqual(client.status(self.ctx)["leases"][0]["revision"], 2)
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": original})):
            with self.assertRaises(client.UniverseLeaseError):
                client.renew(self.ctx, "main")
        with patch.object(client, "_post", return_value=(
                INSTALL, {"released": True, "lease_id": original["lease_id"]})):
            out = client.release(self.ctx, "main")
        self.assertTrue(out["released"])
        self.assertEqual(client.status(self.ctx)["leases"], [])

    def test_pending_request_id_persists_across_transport_error(self):
        with patch.object(client, "_post", side_effect=client.UniverseLeaseError("offline")):
            with self.assertRaises(client.UniverseLeaseError):
                client.open_universe(self.ctx, "pending")
        state = client._read(self.ctx)
        saved_request_id = state["pending"]["pending"]["request_id"]
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": receipt(0)})) as remote:
            client.open_universe(self.ctx, "pending")
        self.assertEqual(remote.call_args.args[1], "open")
        self.assertEqual(remote.call_args.args[2]["request_id"], saved_request_id)
        self.assertEqual(client.status(self.ctx)["pending_aliases"], [])


if __name__ == "__main__":
    unittest.main()
