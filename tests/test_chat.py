import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.tools import chat, chat_history, status


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        purpose = kwargs.get("purpose")
        if purpose == "nanlmind-hermes-chat":
            return SimpleNamespace(text="A river is a flowing body of water.", provider="fake", model="chat-test")
        return SimpleNamespace(
            text=json.dumps({"discoveries": [{
                "kind": "relation", "subject": "river", "predicate": "is_a",
                "object": "flowing-water-body", "resolution": "lexical",
                "polarity": "neutral", "confidence": 0.9,
                "evidence": {"turns": 1}, "provenance": {"test": True}
            }]}),
            provider="fake", model="reflection-test"
        )


class FakeContext:
    def __init__(self, root):
        self.config = {"state_dir": str(root)}
        self.llm = FakeLLM()

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class HermesChatTests(unittest.TestCase):
    def test_open_ended_chat_persists_transcript_and_reflects_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "Explain what a river is.", session_id="demo")
            self.assertEqual(result["response"], "A river is a flowing body of water.")
            self.assertEqual(result["provider"], "fake")
            self.assertTrue(result["reflection"]["parse_ok"])
            self.assertEqual(result["reflection"]["proposed_discoveries"], 1)
            self.assertEqual(result["reflection"]["memory_merge"]["accepted"], 1)
            history = chat_history(ctx, "demo")
            self.assertEqual([x["role"] for x in history["messages"]], ["user", "assistant"])
            self.assertEqual(status(ctx)["mathematical_memory"]["events"], 1)
            self.assertEqual(len(ctx.llm.calls), 2)

    def test_chat_survives_optional_reflection_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            def complete(**kwargs):
                if kwargs.get("purpose") == "nanlmind-hermes-chat":
                    return SimpleNamespace(text="Hello.", provider="fake", model="chat-test")
                raise RuntimeError("reflection unavailable")
            ctx.llm.complete = complete
            result = chat(ctx, "Hello", session_id="fallback")
            self.assertEqual(result["response"], "Hello.")
            self.assertIn("RuntimeError", result["reflection"]["error"])
            self.assertEqual(len(chat_history(ctx, "fallback")["messages"]), 2)

    def test_large_plain_text_is_not_forced_through_n16txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "A long ordinary text passage. " * 1000, session_id="plain")
            self.assertTrue(result["response"])
            self.assertEqual(len(chat_history(ctx, "plain")["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
