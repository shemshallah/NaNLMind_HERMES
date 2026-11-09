"""End-to-end signed graph tool: actual paired read token and pinned key.

The only mocked transport is the online authorization authority's HTTP
response; token issuance, signature and encrypted installation state are real.
"""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from nanlmind_hermes import register
from nanlmind_hermes.auth import (
    _unb64, create_installation, encrypt_payload,
    generate_master_key, issue_token,
)
from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import _DOMAIN, GraphSenderVerificationError
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import LocalRunner
from nanlmind_hermes.universe_root import attach_universe_root, read_universe_root
from test_router_graph_history import edge, fixture


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def encoded(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class Context:
    def __init__(self, root, config):
        self.root = root
        self.config = {"state_dir": str(root), **config}
        self.tools = {}

    def get_config(self, name, default=None):
        return self.config.get(name, default)

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw["handler"]


class SignedToolIntegrationTests(unittest.TestCase):
    def test_tool_enforces_actual_read_credential_and_independent_sender_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authdir = root / "universe"
            authdir.mkdir()
            master = _unb64(generate_master_key())
            identity = create_installation(authdir, master)
            issuer = Ed25519PrivateKey.generate()
            token = issue_token(
                issuer, identity["installation_id"],
                ["universe:read"], ttl_seconds=300)
            envelope = encrypt_payload(
                master, token.encode(),
                ("nannet-token-v1:" + identity["installation_id"]).encode())
            (authdir / "nannet_token.json").write_text(json.dumps(envelope))
            sender = Ed25519PrivateKey.generate()
            sender_pub = sender.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw).hex()
            ctx = Context(root, {
                "nannet_auth_url": "https://test-authority.invalid",
                "nannet_issuer_public_key": encoded(
                    issuer.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw)),
                "router_graph_sender_id": "router-0",
                "router_graph_sender_key_id": "key-1",
                "router_graph_sender_public_key_hex": sender_pub,
            })
            register(ctx)
            chart = LocalRunner.open(ctx).chart
            if not read_universe_root(chart).get("mounted"):
                attach_universe_root(chart)
            revisions = [edge("a", 1), edge("a", 2, "retracted", 2),
                         edge("b", 1, "candidate", 3)]
            pages = [fixture(chart, revisions, state_offset=i)
                     for i in range(2)]
            history_pages = [fixture(chart, revisions, history_offset=i)
                             for i in range(3)]
            header = {k: pages[0][k] for k in (
                "graph_commitment_sha256", "topology_sha256",
                "parent_root_sha256", "chart_generation", "history_sha256")}
            body = dict(
                schema="nanrouter.graph-sender-attestation/1",
                router_id="router-0", key_id="key-1",
                **header, nonce="b" * 32, issued_at=1750000000)
            # Make issued_at contemporary; avoid introducing a static expired
            # test credential while retaining deterministic signature validation.
            import time
            body["issued_at"] = int(time.time())
            signed = {**body,
                      "signature_hex": sender.sign(_DOMAIN + canonical(body)).hex(),
                      "signed_by_router": True,
                      "native_mth_receipt": False,
                      "mathematical_truth_promoted": False}
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "authorized": True,
                "installation_id": identity["installation_id"],
            }).encode()
            handler = ctx.tools["nanlm_router_graph_verify"]
            with patch("nanlmind_hermes.auth_tools.get_secret",
                       return_value=encoded(master)), patch(
                    "nanlmind_hermes.auth_tools.urllib.request.urlopen",
                    return_value=response):
                status = json.loads(handler(dict(
                    pages=pages, history_pages=history_pages,
                    attestation=signed)))
                self.assertTrue(status["state_verified"])
                self.assertTrue(status["history_verified"])
                self.assertTrue(status["sender_authenticated"])
                self.assertFalse(status["nonce_replay_protected"])
                self.assertFalse(status["learning_admitted"])
                self.assertFalse(status["native_mth_executed"])
                self.assertFalse(status["mathematical_truth_promoted"])
                legacy = json.loads(handler({"pages": pages}))
                self.assertTrue(legacy["state_verified"])
                self.assertFalse(legacy["sender_authenticated"])
                with self.assertRaises(RouterGraphPageError):
                    handler(dict(pages=pages, attestation=signed))
                ctx.config["router_graph_sender_public_key_hex"] = ""
                with self.assertRaises(GraphSenderVerificationError):
                    handler(dict(pages=pages, history_pages=history_pages,
                                 attestation=signed))


if __name__ == "__main__":
    unittest.main()
