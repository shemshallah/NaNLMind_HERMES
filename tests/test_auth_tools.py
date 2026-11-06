import os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from nanlmind_hermes.auth import create_installation, encrypt_payload, issue_token, generate_master_key, _unb64
from nanlmind_hermes.auth_tools import require_scope

class FakeContext:
    def __init__(self, root): self.root=root
    def get_config(self, name, default=None): return str(self.root)

class AuthToolTests(unittest.TestCase):
    def test_scope_required_after_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"universe"; root.mkdir()
            master=_unb64(generate_master_key())
            identity=create_installation(root,master)
            issuer=Ed25519PrivateKey.generate()
            public=issuer.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
            token=issue_token(issuer,identity["installation_id"],["training:execute"],ttl_seconds=300)
            envelope=encrypt_payload(master,token.encode(),("nannet-token-v1:"+identity["installation_id"]).encode())
            (root/"nannet_token.json").write_text(__import__("json").dumps(envelope))
            encoded=__import__("base64").urlsafe_b64encode(master).rstrip(b"=").decode()
            issuer_encoded=__import__("base64").urlsafe_b64encode(public).rstrip(b"=").decode()
            with patch.dict(os.environ,{"NANNET_MASTER_KEY":encoded,"NANNET_ISSUER_PUBLIC_KEY":issuer_encoded},clear=True):
                ctx=FakeContext(Path(tmp))
                self.assertTrue(require_scope(ctx,"training:execute"))
                with self.assertRaises(ValueError): require_scope(ctx,"snapshot:write")

    def test_unpaired_local_installation_remains_local_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{},clear=True):
                self.assertFalse(require_scope(FakeContext(Path(tmp)),"training:execute"))

if __name__=="__main__": unittest.main()
