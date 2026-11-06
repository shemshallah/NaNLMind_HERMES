import base64, tempfile, unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from nanlmind_hermes.auth import AuthError, TokenRegistry, create_installation, decrypt_payload, encrypt_payload, generate_master_key, issue_token, load_installation, sign_request, verify_request, verify_token

def key_bytes():
    return base64.urlsafe_b64decode(generate_master_key() + "==")

class AuthTests(unittest.TestCase):
    def test_installation_private_key_is_encrypted_and_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            master=key_bytes(); created=create_installation(tmp,master); doc,private=load_installation(tmp,master)
            self.assertEqual(created["installation_id"],doc["installation_id"]); self.assertNotIn("private_key",doc)
            with self.assertRaises(AuthError): load_installation(tmp,key_bytes())
    def test_scoped_signed_tokens_expire_and_revoke(self):
        issuer=Ed25519PrivateKey.generate(); token=issue_token(issuer,"install-1",["training:execute"],60,now=1000)
        claims=verify_token(token,issuer.public_key(),"training:execute",now=1010); self.assertEqual(claims["sub"],"install-1")
        with self.assertRaises(AuthError): verify_token(token,issuer.public_key(),"snapshot:write",now=1010)
        with self.assertRaises(AuthError): verify_token(token,issuer.public_key(),now=1060)
        with self.assertRaises(AuthError): verify_token(token,issuer.public_key(),now=1010,revoked_ids={claims["jti"]})
    def test_signed_request_and_replay_registry(self):
        private=Ed25519PrivateKey.generate(); body=b'{"run":true}'
        env=sign_request(private,"POST","/v1/train",body,timestamp=500,nonce="abcdefghijklmnop")
        nonce=verify_request(private.public_key(),"POST","/v1/train",body,env,now=500)
        with self.assertRaises(AuthError): verify_request(private.public_key(),"POST","/v1/train",b"tampered",env,now=500)
        with tempfile.TemporaryDirectory() as tmp:
            registry=TokenRegistry(Path(tmp)/"auth.sqlite"); registry.consume_nonce("install-1",nonce,now=500)
            with self.assertRaises(AuthError): registry.consume_nonce("install-1",nonce,now=501)
    def test_authenticated_payload_encryption(self):
        master=key_bytes(); env=encrypt_payload(master,b"universe snapshot",b"snapshot:v1")
        self.assertEqual(decrypt_payload(master,env,b"snapshot:v1"),b"universe snapshot")
        with self.assertRaises(AuthError): decrypt_payload(master,env,b"wrong-context")

if __name__ == "__main__": unittest.main()
