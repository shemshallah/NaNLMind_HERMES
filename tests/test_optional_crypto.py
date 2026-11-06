import os
import subprocess
import sys
import unittest
from pathlib import Path


class OptionalCryptoImportTests(unittest.TestCase):
    def test_plugin_registers_without_crypto_package(self):
        code = r"""
import importlib.abc
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

class BlockCrypto(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cryptography" or fullname.startswith("cryptography."):
            raise ModuleNotFoundError("blocked optional cryptography", name="cryptography")
        return None

sys.meta_path.insert(0, BlockCrypto())
import nanlmind_hermes

class Provider:
    def complete(self, **kwargs):
        if kwargs.get("purpose") == "nanlmind-hermes-chat":
            return SimpleNamespace(text="Hermes is ready.", provider="fake", model="test")
        return SimpleNamespace(
            text=json.dumps({"discoveries": [{
                "kind": "hypothesis", "subject": "chat-is-learning",
                "resolution": "sentence", "polarity": "neutral"
            }]}),
            provider="fake", model="reflect-test"
        )

class Context:
    def __init__(self, root):
        self.tools = {}
        self.llm = Provider()
        self.root = root
    def get_config(self, key, default=None):
        return str(self.root) if key == "state_dir" else default
    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs["handler"]

with tempfile.TemporaryDirectory() as tmp:
    ctx = Context(Path(tmp))
    nanlmind_hermes.register(ctx)
    assert "nanlm_chat" in ctx.tools
    assert "nanlm_chat_history" in ctx.tools
    assert "nanlm_math_train" in ctx.tools
    assert "nanlm_generate_curriculum" in ctx.tools
    assert "nanlm_curriculum_experiment" in ctx.tools
    assert "nanlm_auth_init" in ctx.tools
    result = json.loads(ctx.tools["nanlm_chat"]({"message": "Try an open-ended experiment."}))
    assert result["response"] == "Hermes is ready."
    assert result["reflection"]["memory_merge"]["accepted"] == 1
    history = json.loads(ctx.tools["nanlm_chat_history"]({}))
    assert len(history["messages"]) == 2
    print("registered", len(ctx.tools), "tools and completed chat without cryptography")
"""
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=root,
            text=True, capture_output=True, timeout=20,
            env={**os.environ, "NANLM_AUTH_TEST_BYPASS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("without cryptography", result.stdout)


if __name__ == "__main__":
    unittest.main()
