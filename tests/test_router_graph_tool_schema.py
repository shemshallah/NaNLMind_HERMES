"""History verifier remains a protected, bounded, read-only plugin operation."""
import unittest
from nanlmind_hermes import register


class Context:
    def __init__(self):
        self.tools = {}

    def get_config(self, key, default=None):
        return default

    def register_tool(self, **kw):
        self.tools[kw["name"]] = kw


class RouterGraphToolContract(unittest.TestCase):
    def test_history_pages_are_optional_and_bounded(self):
        ctx = Context()
        register(ctx)
        tool = ctx.tools["nanlm_router_graph_verify"]
        parameters = tool["schema"]["parameters"]
        self.assertEqual(parameters["required"], ["pages"])
        for key in ("pages", "history_pages"):
            prop = parameters["properties"][key]
            self.assertEqual(prop["type"], "array")
            self.assertEqual(prop["minItems"], 1)
            self.assertEqual(prop["maxItems"], 128)
        self.assertTrue(callable(tool["handler"]))


if __name__ == "__main__":
    unittest.main()
