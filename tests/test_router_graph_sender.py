"""Cryptographic, pinned-key sender identity is distinct from MTH admission."""
import json
import unittest

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import (
    GraphSenderVerificationError, verify_graph_sender_attestation, _DOMAIN,
)
from test_router_graph_history import HistoryReplayTests


def canonical(obj):
    return json.dumps(obj, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


class GraphSenderTests(unittest.TestCase):
    def setUp(self):
        fixture = HistoryReplayTests()
        fixture.setUp()
        self.chart = fixture.chart
        self.verifier = fixture.verify(fixture.history)
        self.key = Ed25519PrivateKey.generate()
        self.other_key = Ed25519PrivateKey.generate()
        self.now = 1750000000

    def public(self, private):
        return private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw).hex()

    def envelope(self, private=None, **updates):
        header = self.verifier.state._header
        body = dict(
            schema="nanrouter.graph-sender-attestation/1",
            router_id="nanrouter-live",
            key_id="sender-v1",
            graph_commitment_sha256=self.verifier.state._commitment,
            topology_sha256=header["topology_sha256"],
            parent_root_sha256=header["parent_root_sha256"],
            chart_generation=header["chart_generation"],
            history_sha256=header["history_sha256"],
            nonce="a" * 32,
            issued_at=self.now,
        )
        body.update(updates)
        key = private if private is not None else self.key
        return {
            **body, "signature_hex": key.sign(_DOMAIN + canonical(body)).hex(),
            "signed_by_router": True,
            "native_mth_receipt": False,
            "mathematical_truth_promoted": False,
        }

    def verify(self, attestation, **kw):
        args = dict(
            expected_router_id="nanrouter-live",
            expected_key_id="sender-v1",
            public_key_hex=self.public(self.key),
            now=self.now,
        )
        args.update(kw)
        return verify_graph_sender_attestation(
            self.verifier, attestation, **args)

    def test_pinned_real_ed25519_signature_authenticates_only_sender(self):
        result = self.verify(self.envelope())
        self.assertTrue(result["history_verified"])
        self.assertTrue(result["sender_authenticated"])
        self.assertFalse(result["nonce_replay_protected"])
        for flag in ("native_mth_executed", "learning_admitted",
                     "mathematical_truth_promoted"):
            self.assertFalse(result[flag])

    def test_wrong_key_invalid_signature_and_untrusted_key_field(self):
        cases = [
            (self.envelope(private=self.other_key), {}),
            (self.envelope(), {"public_key_hex": self.public(self.other_key)}),
            (self.envelope(), {"expected_router_id": "another-router"}),
            (self.envelope(), {"expected_key_id": "retired-key"}),
        ]
        for att, options in cases:
            with self.subTest(options=options), self.assertRaises(
                    GraphSenderVerificationError):
                self.verify(att, **options)
        tampered = self.envelope()
        tampered["signature_hex"] = "0" * 128
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(tampered)
        injected = self.envelope()
        injected["public_key_hex"] = self.public(self.other_key)
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(injected)

    def test_wrong_graph_epoch_and_digest_denied_even_if_signed(self):
        for field, value in (
            ("history_sha256", "0" * 64),
            ("chart_generation", 100),
            ("graph_commitment_sha256", "f" * 64),
            ("topology_sha256", "0" * 64),
            ("parent_root_sha256", "0" * 64),
        ):
            with self.subTest(field=field), self.assertRaises(
                    GraphSenderVerificationError):
                self.verify(self.envelope(**{field: value}))
        valid = self.envelope()
        self.chart.snapshot_generation += 1
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(valid)

    def test_expired_future_bad_nonce_and_overclaims_denied(self):
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(self.envelope(issued_at=self.now - 301))
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(self.envelope(issued_at=self.now + 31))
        for broken in ({"nonce": "bad"}, {"issued_at": True},
                       {"chart_generation": True}):
            with self.subTest(broken=broken), self.assertRaises(
                    GraphSenderVerificationError):
                self.verify(self.envelope(**broken))
        for key in ("native_mth_receipt", "mathematical_truth_promoted"):
            att = self.envelope()
            att[key] = True
            with self.assertRaises(GraphSenderVerificationError):
                self.verify(att)
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(self.envelope(), public_key_hex="")
        with self.assertRaises(GraphSenderVerificationError):
            self.verify(self.envelope(), now=True)


if __name__ == "__main__":
    unittest.main()
