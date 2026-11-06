import os, sys, tempfile, threading, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_core = os.environ.get("NANLM_CORE_ROOT", "")
if _core:
    sys.path.insert(0, _core)
try:
    from nanlmind_nannet.pairing import PairingAuthority
    from nanlmind_nannet.auth import generate_master_key, _unb64
    from runner.training_merge import make_server
    from nanlmind_hermes.hw_hermes import ensure_hardware_key, authorize
    HAVE_CORE = True
except Exception:
    HAVE_CORE = False


@unittest.skipUnless(HAVE_CORE, "NaNLMind core not available")
class HardwareHermesTests(unittest.TestCase):
    def test_register_once_authorize_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            auth = PairingAuthority(Path(tmp) / "auth.sqlite", _unb64(generate_master_key()))
            server = make_server("127.0.0.1", 0, Path(tmp), "merge-token", auth)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                first = ensure_hardware_key(url, Path(tmp) / "state")
                self.assertEqual(first.source, "registered")
                self.assertTrue(first.key.startswith("hwh_"))
                self.assertTrue(authorize(url, "universe:read", first))
                second = ensure_hardware_key(url, Path(tmp) / "state")
                self.assertEqual(second.source, "stored")
                self.assertEqual(second.key, first.key)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
