"""Authenticated graph navigation tool: real read scope and pinned sender key."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nanlmind_hermes import register
from nanlmind_hermes.auth import (
    _unb64, create_installation, encrypt_payload,
    generate_master_key, issue_token,
)
from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import (
    _DOMAIN, GraphSenderVerificationError,
)
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import LocalRunner
from nanlmind_hermes.universe_root import attach_universe_root, read_universe_root
from test_router_graph_history import edge, fixture
from test_router_graph_tool_signed import Context, canonical, encoded


class GraphWalkToolTests(unittest.TestCase):
    def test_signed_candidate_navigation_never_admits_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "universe"
            home.mkdir()
            master = _unb64(generate_master_key())
            identity = create_installation(home, master)
            issuer = Ed25519PrivateKey.generate()
            token = issue_token(
                issuer, identity["installation_id"],
                ["universe:read"], ttl_seconds=300)
            envelope = encrypt_payload(
                master, token.encode(),
                ("nannet-token-v1:" + identity["installation_id"]).encode())
            (home / "nannet_token.json").write_text(json.dumps(envelope))
            sender = Ed25519PrivateKey.generate()
            pub = sender.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw).hex()
            ctx = Context(root, {
                "nannet_auth_url": "https://test-authority.invalid",
                "nannet_issuer_public_key": encoded(
                    issuer.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw)),
                "router_graph_sender_id": "router-walk",
                "router_graph_sender_key_id": "walk-key-1",
                "router_graph_sender_public_key_hex": pub,
            })
            register(ctx)
            chart = LocalRunner.open(ctx).chart
            if not read_universe_root(chart).get("mounted"):
                attach_universe_root(chart)
            revisions = [
                edge("a", 1, "candidate", 1),
                dict(edge("b", 1, "candidate", 2), relation="refutes"),
                edge("c", 1, "candidate", 3),
                edge("c", 2, "retracted", 4),
            ]
            pages = [fixture(chart, revisions, state_offset=i)
                     for i in range(3)]
            history = [fixture(chart, revisions, history_offset=i)
                       for i in range(4)]
            header = {k: pages[0][k] for k in (
                "graph_commitment_sha256", "topology_sha256",
                "parent_root_sha256", "chart_generation", "history_sha256")}
            body = dict(
                schema="nanrouter.graph-sender-attestation/1",
                router_id="router-walk", key_id="walk-key-1",
                **header, nonce="c" * 32, issued_at=int(time.time()))
            signed = {
                **body, "signature_hex": sender.sign(
                    _DOMAIN + canonical(body)).hex(),
                "signed_by_router": True,
                "native_mth_receipt": False,
                "mathematical_truth_promoted": False,
            }
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "authorized": True,
                "installation_id": identity["installation_id"],
            }).encode()
            handler = ctx.tools["nanlm_router_graph_walk"]
            with patch("nanlmind_hermes.auth_tools.get_secret",
                       return_value=encoded(master)), patch(
                    "nanlmind_hermes.auth_tools.urllib.request.urlopen",
                    return_value=response):
                before = bytes(chart.bytes)
                verified = json.loads(handler({
                    "pages": pages, "history_pages": history,
                    "attestation": signed, "start": 7, "target": 265,
                    "relation": "supports", "max_hops": 1,
                }))
                self.assertTrue(verified["history_verified"])
                self.assertTrue(verified["sender_authenticated"])
                self.assertFalse(verified["nonce_replay_protected"])
                self.assertFalse(verified["evidence_integrity_verified"])
                self.assertFalse(verified["learning_admitted"])
                self.assertFalse(verified["native_mth_executed"])
                self.assertFalse(verified["mathematical_truth_promoted"])
                self.assertEqual(verified["paths"][0]["edges"][0]["relation"],
                                 "supports")
                # Contradictions must remain visible despite the relation filter.
                self.assertEqual(verified["contested_pairs"], [
                    {"src": 7, "dst": 265, "layer": 4,
                     "relations": ["refutes", "supports"]}])
                self.assertEqual(bytes(chart.bytes), before)
                self.assertFalse(json.loads(handler({
                    "pages": pages, "history_pages": history,
                    "start": 7, "target": 265,
                }))["sender_authenticated"])
                with self.assertRaises(RouterGraphPageError):
                    handler({"pages": pages, "history_pages": history[:-1],
                             "start": 7})
                ctx.config["router_graph_sender_public_key_hex"] = ""
                with self.assertRaises(GraphSenderVerificationError):
                    handler({"pages": pages, "history_pages": history,
                             "attestation": signed, "start": 7})


if __name__ == "__main__":
    unittest.main()
