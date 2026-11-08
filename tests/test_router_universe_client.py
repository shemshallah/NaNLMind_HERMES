"""HERMES router-universe lifecycle, encrypted persistence and head protection."""
import base64
import json
import hashlib
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

    def test_native_n16t_dispatch_ack_and_mismatched_response_digest(self):
        lease = receipt(0)
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": lease})):
            client.open_universe(self.ctx, "test")
        frame = b"N16T" + bytes(32)
        digest = hashlib.sha256(frame).hexdigest()
        result = {
            "dispatch_id": "stable-router-dispatch-001",
            "head_index": 254, "lease_id": lease["lease_id"],
            "universe_id": lease["universe_id"],
            "native_guest_route_admitted": True,
            "n16t_native_guest_exchange": True,
            "frame_sha256": digest,
            "response_frame_hex": frame.hex(),
            "response_sha256": digest,
            "acknowledged": False,
            "replayed_from_journal": False,
            "model_training_committed": False,
        }
        with patch.object(client, "_post", return_value=(INSTALL, result)):
            sent = client.dispatch(
                self.ctx, "test", 254, "stable-router-dispatch-001", frame.hex())
        self.assertTrue(sent["n16t_native_guest_exchange"])
        self.assertFalse(sent["model_training_committed"])
        with patch.object(client, "_post",
                          return_value=(INSTALL, {**result, "response_sha256": "0" * 64})):
            with self.assertRaises(client.UniverseLeaseError):
                client.dispatch(
                    self.ctx, "test", 254, "stable-router-dispatch-001", frame.hex())
        with self.assertRaises(client.UniverseLeaseError):
            client.dispatch(self.ctx, "test", 255, "stable-router-dispatch-001", frame.hex())
        with patch.object(client, "_post",
                          return_value=(INSTALL, {
                              "acknowledged": True,
                              "dispatch_id": "stable-router-dispatch-001",
                              "response_sha256": digest,
                          })):
            self.assertTrue(client.acknowledge(
                self.ctx, "test", 254, "stable-router-dispatch-001",
                digest)["acknowledged"])
        with self.assertRaises(client.UniverseLeaseError):
            client.acknowledge(
                self.ctx, "test", 254, "stable-router-dispatch-001", "not-a-digest")

    def test_dispatch_inspection_rejects_cross_head_and_tampered_recovery_frame(self):
        lease = receipt(0)
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": lease})):
            client.open_universe(self.ctx, "recover")
        dispatch_id = "stable-router-recovery-0001"
        base = {
            "dispatch_id": dispatch_id, "head_index": 19,
            "lease_id": lease["lease_id"], "universe_id": lease["universe_id"],
            "model_training_committed": False,
        }
        with patch.object(client, "_post",
                          return_value=(INSTALL, {**base, "state": "not_recorded"})):
            self.assertEqual(client.inspect_dispatch(
                self.ctx, "recover", 19, dispatch_id)["state"], "not_recorded")
        with patch.object(client, "_post",
                          return_value=(INSTALL, {
                              **base, "state": "pending_or_ambiguous",
                              "frame_sha256": "a" * 64})):
            pending = client.inspect_dispatch(self.ctx, "recover", 19, dispatch_id)
            self.assertEqual(pending["state"], "pending_or_ambiguous")
            self.assertNotIn("response_frame_hex", pending)
        frame = b"N16T" + bytes(32)
        digest = hashlib.sha256(frame).hexdigest()
        completed = {
            **base, "state": "completed", "response_frame_hex": frame.hex(),
            "response_sha256": digest, "n16t_native_guest_exchange": True,
        }
        with patch.object(client, "_post", return_value=(INSTALL, completed)):
            result = client.inspect_dispatch(self.ctx, "recover", 19, dispatch_id)
            self.assertEqual(result["response_sha256"], digest)
            self.assertFalse(result["acknowledged"])
        for modified in (
            {**completed, "response_sha256": "0" * 64},
            {**completed, "head_index": 20},
            {**completed, "model_training_committed": True},
            {**completed, "n16t_native_guest_exchange": False},
        ):
            with patch.object(client, "_post", return_value=(INSTALL, modified)):
                with self.assertRaises(client.UniverseLeaseError):
                    client.inspect_dispatch(self.ctx, "recover", 19, dispatch_id)
        with self.assertRaises(client.UniverseLeaseError):
            client.inspect_dispatch(self.ctx, "recover", 255, dispatch_id)


    def test_native_mth_requires_signed_lease_head_and_attested_output(self):
        lease = receipt(0)
        with patch.object(client, "_post", return_value=(INSTALL, {"receipt": lease})):
            client.open_universe(self.ctx, "math")
        output = {
            "kernel_id": "mth.u32.add_checked", "head_index": 7,
            "node_point": lease["node_point"],
            "universe_id": lease["universe_id"],
            "native_guest_universe_id": lease["universe_id"],
            "native_guest_node_point": lease["node_point"],
            "native_guest_route_admitted": True, "native_nan16_execution": True,
            "value_u32": 16, "exact": True,
            "model_training_committed": False,
        }
        with patch.object(client, "_post", return_value=(INSTALL, output)):
            result = client.math_u32(self.ctx, "math", 7, "mth.u32.add_checked", 7, 9)
        self.assertEqual(result["result_u32"], 16)
        self.assertTrue(result["native_nan16_execution"])
        self.assertFalse(result["model_training_committed"])
        for corrupted in (
            {**output, "native_nan16_execution": False},
            {**output, "native_guest_node_point": 255},
            {**output, "native_guest_universe_id": 999},
            {**output, "model_training_committed": True},
            {**output, "exact": "true"},
        ):
            with patch.object(client, "_post", return_value=(INSTALL, corrupted)):
                with self.assertRaises(client.UniverseLeaseError):
                    client.math_u32(self.ctx, "math", 7, "mth.u32.add_checked", 7, 9)
        for bad_kernel in ("mth.vector.unknown", "", None):
            with self.assertRaises(client.UniverseLeaseError):
                client.math_u32(self.ctx, "math", 7, bad_kernel, 7, 9)
        with self.assertRaises(client.UniverseLeaseError):
            client.math_u32(self.ctx, "math", 255, "mth.u32.add_checked", 7, 9)
        with self.assertRaises(client.UniverseLeaseError):
            client.math_u32(self.ctx, "math", 7, "mth.u32.add_checked", -1, 9)



if __name__ == "__main__":
    unittest.main()
