import subprocess
import sys
import unittest
from pathlib import Path


class OptionalCryptoImportTests(unittest.TestCase):
    def test_plugin_registers_without_crypto_package(self):
        code = r"""
import importlib.abc
import sys

class BlockCrypto(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cryptography" or fullname.startswith("cryptography."):
            raise ModuleNotFoundError("blocked optional cryptography", name="cryptography")
        return None

sys.meta_path.insert(0, BlockCrypto())
import nanlmind_hermes

class Context:
    def __init__(self):
        self.tools = []
    def get_config(self, key, default=None):
        return default
    def register_tool(self, **kwargs):
        self.tools.append(kwargs["name"])

ctx = Context()
nanlmind_hermes.register(ctx)
assert "nanlm_chat" in ctx.tools
assert "nanlm_math_train" in ctx.tools
assert "nanlm_auth_init" in ctx.tools
print("registered", len(ctx.tools), "tools without cryptography")
"""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=root,
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("without cryptography", result.stdout)


if __name__ == "__main__":
    unittest.main()
