import base64, json, tempfile, threading, unittest, urllib.request
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from nanlmind_hermes.auth import _unb64, generate_master_key
from nanlmind_hermes.pairing import PairingAuthority
from nanlmind_hermes.http_api import make_server

class PairingHttpTests(unittest.TestCase):
    def test_loopback_pair_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            authority=PairingAuthority(Path(tmp)/"auth.sqlite",_unb64(generate_master_key()))
            code=authority.create_pairing_code(["training:execute"])
            server=make_server(authority,"127.0.0.1",0)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            try:
                private=Ed25519PrivateKey.generate()
                raw=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
                public=base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
                import uuid
                payload={"pairing_code":code["pairing_code"],"installation_id":str(uuid.uuid4()),"public_key":public}
                req=urllib.request.Request("http://127.0.0.1:%d/v1/pair"%server.server_address[1],data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
                with urllib.request.urlopen(req,timeout=5) as response: result=json.loads(response.read())
                self.assertEqual(result["scopes"],["training:execute"])
                self.assertEqual(result["issuer_public_key"],authority.issuer_public_key())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_non_loopback_requires_tls(self):
        with tempfile.TemporaryDirectory() as tmp:
            authority=PairingAuthority(Path(tmp)/"auth.sqlite",_unb64(generate_master_key()))
            with self.assertRaises(ValueError): make_server(authority,"0.0.0.0",0)

if __name__=="__main__": unittest.main()
