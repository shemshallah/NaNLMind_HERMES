import re
import unittest
from pathlib import Path

from nanlmind_hermes import register


class FakeContext:
    def __init__(self):
        self.tools = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def get_config(self, key, default=None):
        return default


class ManifestTests(unittest.TestCase):
    def test_declared_tools_match_registration(self):
        text = Path("plugin.yaml").read_text(encoding="utf-8")
        block = text.split("provides_tools:", 1)[1].split("config_schema:", 1)[0]
        declared = set(re.findall(r"^  - (\S+)$", block, flags=re.MULTILINE))
        ctx = FakeContext()
        register(ctx)
        self.assertEqual(declared, set(ctx.tools))
        self.assertEqual(len(declared), 24)

    def test_every_tool_has_schema_and_handler(self):
        ctx = FakeContext()
        register(ctx)
        for name, tool in ctx.tools.items():
            self.assertTrue(callable(tool["handler"]), name)
            self.assertEqual(tool["schema"]["name"], name)
            self.assertEqual(tool["schema"]["parameters"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
