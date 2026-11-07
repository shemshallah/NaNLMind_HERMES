import base64, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.auth import create_installation, encrypt_payload, issue_token, generate_master_key, _unb64
from nanlmind_hermes.auth_tools import auth_pair, auth_status, require_scope


class FakeContext:
    def __init__(self, root, config=None):
        self.root = root
        self.config = {"state_dir": str(root), **(config or {})}
    def get_config(self, name, default=None):
        return self.config.get(name, default)


class AuthToolTests(unittest.TestCase):
    def test_scope_required_after_pairing_and_online_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "universe"
            root.mkdir()
            master = _unb64(generate_master_key())
            identity = create_installation(root, master)
            issuer = Ed25519PrivateKey.generate()
            public = issuer.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            token = issue_token(issuer, identity["installation_id"], ["training:execute"], ttl_seconds=300)
            envelope = encrypt_payload(master, token.encode(), ("nannet-token-v1:" + identity["installation_id"]).encode())
            (root / "nannet_token.json").write_text(json.dumps(envelope))
            encoded = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
            issuer_encoded = base64.urlsafe_b64encode(public).rstrip(b"=").decode()
            ctx = FakeContext(Path(tmp), {
                "nannet_auth_url": "http://127.0.0.1:9999",
                "nannet_issuer_public_key": issuer_encoded,
            })
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "authorized": True, "installation_id": identity["installation_id"],
            }).encode()
            with patch("nanlmind_hermes.auth_tools.get_secret", return_value=encoded), \
                 patch("nanlmind_hermes.auth_tools.urllib.request.urlopen", return_value=response):
                self.assertTrue(require_scope(ctx, "training:execute"))
                with self.assertRaises(ValueError):
                    require_scope(ctx, "snapshot:write")

    def test_pairing_pins_stable_issuer_and_stores_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "universe"
            root.mkdir()
            master = _unb64(generate_master_key())
            identity = create_installation(root, master)
            issuer = Ed25519PrivateKey.generate()
            public = issuer.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            token = issue_token(issuer, identity["installation_id"], ["universe:read", "training:execute"], ttl_seconds=300)
            encoded_master = base64.urlsafe_b64encode(master).rstrip(b"=").decode()
            encoded_issuer = base64.urlsafe_b64encode(public).rstrip(b"=").decode()
            ctx = FakeContext(Path(tmp), {
                "nannet_auth_url": "https://nannet.example",
                "nannet_issuer_public_key": encoded_issuer,
            })
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "token": token,
                "issuer_public_key": encoded_issuer,
                "installation_id": identity["installation_id"],
                "scopes": ["training:execute", "universe:read"],
                "role": "guest",
                "expires_at": 9999999999,
            }).encode()
            with patch("nanlmind_hermes.auth_tools.get_secret", return_value=encoded_master), \
                 patch("nanlmind_hermes.auth_tools.urllib.request.urlopen", return_value=response):
                paired = auth_pair(ctx, "nnh_test_pairing_code")
                self.assertTrue(paired["paired"])
                self.assertTrue(paired["issuer_key_pinned"])
                self.assertTrue((root / "nannet_token.json").exists())
                status = auth_status(ctx)
                self.assertTrue(status["token_valid"])
                self.assertEqual(status["token_scopes"], ["training:execute", "universe:read"])

    def test_pairing_rejects_issuer_key_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "universe"
            root.mkdir()
            master = _unb64(generate_master_key())
            identity = create_installation(root, master)
            issuer = Ed25519PrivateKey.generate()
            public = issuer.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            token = issue_token(issuer, identity["installation_id"], ["universe:read"], ttl_seconds=300)
            ctx = FakeContext(Path(tmp), {
                "nannet_auth_url": "https://nannet.example",
                "nannet_issuer_public_key": base64.urlsafe_b64encode(b"\x00" * 32).rstrip(b"=").decode(),
            })
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "token": token,
                "issuer_public_key": base64.urlsafe_b64encode(public).rstrip(b"=").decode(),
                "installation_id": identity["installation_id"],
                "scopes": ["universe:read"],
                "role": "guest",
            }).encode()
            with patch("nanlmind_hermes.auth_tools.get_secret", return_value=base64.urlsafe_b64encode(master).rstrip(b"=").decode()), \
                 patch("nanlmind_hermes.auth_tools.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "does not match pinned"):
                    auth_pair(ctx, "nnh_test_pairing_code")

    def test_unpaired_installation_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp), {
                "nannet_auth_url": "https://nannet.example",
                "nannet_issuer_public_key": "pinned",
            })
            with self.assertRaisesRegex(ValueError, "must pair"):
                require_scope(ctx, "training:execute")

    def test_missing_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "authority URL"):
                require_scope(FakeContext(Path(tmp)), "training:execute")


if __name__ == "__main__":
    unittest.main()
