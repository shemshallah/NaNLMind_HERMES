"""Android-style NANNET enrollment must remain a thin encrypted host boundary."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanlmind_hermes import nannet_device_client as client


class FakeContext:
    def __init__(self, root, url="https://nannet.example"):
        self.root = root
        self.url = url

    def get_config(self, key, default=None):
        if key == "state_dir":
            return str(self.root)
        if key == "nannet_auth_url":
            return self.url
        return default


class HardwareEnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = FakeContext(self.tmp.name)
        self.key_patch = patch.object(client, "_master_key", return_value=b"k" * 32)
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, token="nnt_first"):
        return {
            "device_id": "install-A",
            "address": "nan://GM0:SC0:MC0:BS0:HW5",
            "node_point": 6,
            "lease_id": "lease-1",
            "lease_expires_at": 2000000000,
            "token": token,
        }

    def test_register_heartbeat_report_release_and_encrypted_state(self):
        calls = []

        def remote(ctx, path, body=None, token=""):
            calls.append((path, body, token))
            if path == "/v1/devices/register":
                self.assertTrue(body["consent"])
                self.assertEqual(body["terms_version"], client.TERMS)
                self.assertEqual(body["system_report"]["platform"], "Hermes")
                self.assertEqual(len(body["device_secret"]), 43)
                return self.fixture()
            if path == "/v1/devices/rotate":
                self.assertEqual(token, "nnt_first")
                return self.fixture("nnt_rotated")
            if path == "/v1/devices/heartbeat":
                self.assertEqual(token, "nnt_rotated")
                return {**self.fixture(), "token": "nnt_renewed",
                        "lease_expires_at": 2000000100}
            if path == "/v1/devices/report":
                self.assertEqual(token, "nnt_renewed")
                self.assertEqual(body["report_type"], "network_error")
                return {"accepted": True, "report_id": "report-1"}
            if path == "/v1/devices/release":
                self.assertEqual(token, "nnt_renewed")
                return {"released": True, "address": self.fixture()["address"]}
            raise AssertionError(path)

        with patch.object(client, "_http_json", side_effect=remote):
            with self.assertRaisesRegex(client.HardwareEnrollmentError, "consent"):
                client.register(self.ctx)
            first = client.register(self.ctx, consent=True)
            self.assertEqual(first["address"], self.fixture()["address"])
            self.assertTrue(first["bearer_saved_encrypted"])
            self.assertNotIn("token", first)
            encrypted = (Path(self.tmp.name) / "universe" /
                         "nannet_hw_lease.json").read_text()
            self.assertNotIn("nnt_first", encrypted)
            self.assertNotIn(calls[0][1]["device_secret"], encrypted)
            second = client.register(self.ctx, consent=True)
            self.assertEqual(second["address"], first["address"])
            self.assertEqual(second["lease_id"], first["lease_id"])
            heartbeat = client.heartbeat(self.ctx)
            self.assertTrue(heartbeat["renewed"])
            report = client.report(self.ctx, "network_error", "Example outage")
            self.assertEqual(report["report_id"], "report-1")
            released = client.release(self.ctx)
            self.assertTrue(released["remote_confirmed"])
            self.assertFalse(released["bearer_present"])
            self.assertTrue(released["device_identity_present"])
            self.assertEqual(len(calls), 5)

    def test_wordnet_and_other_reserved_hw_slots_are_rejected(self):
        for hw in range(5):
            with self.subTest(hw=hw), patch.object(
                client, "_http_json",
                return_value={**self.fixture(),
                              "address": f"nan://GM0:SC0:MC0:BS0:HW{hw}",
                              "node_point": hw+1}
            ):
                with self.assertRaisesRegex(client.HardwareEnrollmentError,
                                            "reserved or invalid"):
                    client.register(self.ctx, consent=True)
                self.assertFalse(client.status(self.ctx)["bearer_present"])

    def test_wrong_node_point_rejected_without_saving_bearer(self):
        bad = {**self.fixture(), "node_point": 5}
        with patch.object(client, "_http_json", return_value=bad):
            with self.assertRaisesRegex(client.HardwareEnrollmentError,
                                        "node_point disagree"):
                client.register(self.ctx, consent=True)
        self.assertFalse(client.status(self.ctx)["bearer_present"])

    def test_bearer_is_never_exposed_on_status(self):
        with patch.object(client, "_http_json", return_value=self.fixture()):
            client.register(self.ctx, consent=True)
        value = client.status(self.ctx)
        self.assertNotIn("token", value)
        self.assertNotIn("device_secret", value)
        self.assertEqual(value["network_port"], "BS0:0")
        self.assertEqual(value["hardware_root"], "HW0:0")
        self.assertFalse(value["native_universe_lease_active"])

    def test_non_https_and_credential_in_url_denied(self):
        for url in ("http://nannet.example", "https://name:password@nannet.example",
                    "https://nannet.example/proxy", ""):
            with self.subTest(url=url):
                with self.assertRaises(client.HardwareEnrollmentError):
                    client._base_url(FakeContext(self.tmp.name, url))

    def test_hw255_maps_to_last_point_256(self):
        receipt = {**self.fixture(),
                   "address": "nan://GM0:SC0:MC0:BS0:HW255",
                   "node_point": 256}
        self.assertEqual(client._validate_binding(receipt), 256)

    def test_report_requires_valid_size_and_enrollment(self):
        with self.assertRaisesRegex(client.HardwareEnrollmentError, "registration"):
            client.report(self.ctx, "other", "missing")
        with patch.object(client, "_http_json", return_value=self.fixture()):
            client.register(self.ctx, consent=True)
        with self.assertRaisesRegex(client.HardwareEnrollmentError, "240"):
            client.report(self.ctx, "other", "s"*241)

    def test_corrupted_encrypted_identity_fails_closed(self):
        root=Path(self.tmp.name) / "universe"
        root.mkdir(parents=True,exist_ok=True)
        (root/"nannet_hw_lease.json").write_text('{"garbled":true}')
        with self.assertRaisesRegex(client.HardwareEnrollmentError,
                                    "cannot decrypt"):
            client.register(self.ctx, consent=True)


if __name__ == "__main__":
    unittest.main()
