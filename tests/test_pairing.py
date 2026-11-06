import tempfile, unittest
from pathlib import Path
from nanlmind_hermes.auth import AuthError, generate_master_key, _unb64, identity_public_key
from nanlmind_hermes.pairing import PairingAuthority

class PairingTests(unittest.TestCase):
    def test_pairing_code_is_single_use_and_issues_scoped_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            master=_unb64(generate_master_key())
            authority=PairingAuthority(Path(tmp)/"authority.sqlite",master)
            code=authority.create_pairing_code(["training:execute","universe:read"],ttl_seconds=300,now=1000)
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
            import uuid, base64
            raw=Ed25519PrivateKey.generate().public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
            pub=base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
            result=authority.pair(code["pairing_code"],str(uuid.uuid4()),pub,now=1010)
            self.assertEqual(result["scopes"],["training:execute","universe:read"])
            revoked=authority.revoke_token(result["token"],now=1020)
            self.assertTrue(revoked["revoked"])
            with self.assertRaises(AuthError): authority.verify_access_token(result["token"],now=1020)
            with self.assertRaises(AuthError):
                authority.pair(code["pairing_code"],str(uuid.uuid4()),pub,now=1020)

    def test_admin_guest_key_separation_and_identity_revocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            master = _unb64(generate_master_key())
            authority = PairingAuthority(Path(tmp) / "authority.sqlite", master)
            self.assertEqual(identity_public_key(master, "guest", "same"), identity_public_key(master, "guest", "same"))
            self.assertNotEqual(identity_public_key(master, "guest", "same"), identity_public_key(master, "admin", "same"))
            with self.assertRaises(AuthError):
                authority.create_pairing_code(["installation:admin"], role="guest")
            code = authority.create_pairing_code(["installation:admin", "training:execute"], role="admin")
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
            import uuid, base64
            raw = Ed25519PrivateKey.generate().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            public = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
            identity = str(uuid.uuid4())
            result = authority.pair(code["pairing_code"], identity, public)
            self.assertEqual(result["role"], "admin")
            self.assertEqual(authority.verify_access_token(result["token"], "installation:admin")["role"], "admin")
            authority.revoke_identity("admin", identity)
            with self.assertRaises(AuthError):
                authority.verify_access_token(result["token"], "installation:admin")

    def test_expired_pairing_code_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            authority=PairingAuthority(Path(tmp)/"authority.sqlite",_unb64(generate_master_key()))
            code=authority.create_pairing_code(["universe:read"],ttl_seconds=60,now=100)
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
            import uuid,base64
            raw=Ed25519PrivateKey.generate().public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
            pub=base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
            with self.assertRaises(AuthError): authority.pair(code["pairing_code"],str(uuid.uuid4()),pub,now=160)

if __name__=="__main__": unittest.main()
