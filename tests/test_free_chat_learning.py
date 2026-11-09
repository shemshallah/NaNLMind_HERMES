import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.chat import chat
from nanlmind_hermes.layered_learning import (
    coordinator_for_context,
    reset_coordinator_cache,
)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("purpose") == "nanlmind-hermes-chat":
            return SimpleNamespace(text="A river is water.", provider="fake",
                                   model="free-chat-learning")
        return SimpleNamespace(
            text=json.dumps({"discoveries": [{
                "kind": "relation", "subject": "river", "predicate": "is",
                "object": "water", "resolution": "lexical",
                "polarity": "supports", "confidence": 0.9,
                "evidence": {"turn": 1}, "provenance": {"test": True},
            }]}),
            provider="fake", model="reflection",
        )


class ProviderContext:
    def __init__(self, root, *, chat_learning=True):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "provider",
            "chat_learning": chat_learning,
        }
        self.llm = FakeLLM()

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class NativeContext:
    def __init__(self, root):
        self.config = {
            "state_dir": str(root),
            "chat_mode": "native",
            "chat_learning": True,
        }

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class FreeChatLearningTests(unittest.TestCase):
    def setUp(self):
        reset_coordinator_cache()

    def tearDown(self):
        reset_coordinator_cache()

    def test_provider_chat_feeds_all_ten_layers_and_shared_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            ctx = ProviderContext(root)
            first = chat(ctx, "A river is water.", session_id="free-learning")
            layered = first["layered_learning"]
            self.assertEqual(layered["layers"], 10)
            self.assertEqual(layered["layer_names"],
                             ["bit", "nibble", "byte", "phonetic", "morpheme",
                              "lexical", "word_completion", "sentence",
                              "paragraph", "narrative"])
            self.assertGreaterEqual(layered["scratch_records"], 10)
            self.assertGreaterEqual(layered["relations"], 1)
            self.assertGreater(layered["head_balance"]["active_heads"], 0)
            self.assertFalse(layered["canonical_commit"])
            self.assertEqual(layered["scope"], "derived-plugin-scratch")

            second = chat(ctx, "Water can flow.", session_id="free-learning")
            self.assertEqual(second["layered_learning"]["turn"], 2)
            self.assertGreaterEqual(second["layered_learning"]["scratch_records"], 20)

            state_path = root / "language" / "layered_learning.json"
            self.assertTrue(state_path.is_file())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["turns"], 2)
            self.assertGreaterEqual(len(state["rows"]), 1)

            # The explicit learning status and free chat observe the same
            # coordinator/scratch view.
            status = coordinator_for_context(ctx).status()
            self.assertEqual(status["turns"], 2)
            self.assertGreaterEqual(status["scratch"]["records"], 20)
            self.assertTrue(status["scratch_verify"]["ok"])

    def test_native_chat_also_feeds_layered_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            ctx = NativeContext(root)
            result = chat(ctx, "A river is water.", session_id="native-free")
            self.assertEqual(result["provider"], "nanlm-native")
            self.assertEqual(result["layered_learning"]["layers"], 10)
            self.assertGreaterEqual(result["layered_learning"]["scratch_records"], 10)
            self.assertTrue((root / "language" / "layered_learning.json").is_file())

    def test_chat_learning_false_disables_layered_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            ctx = ProviderContext(root, chat_learning=False)
            result = chat(ctx, "A river is water.", session_id="disabled")
            self.assertEqual(result["layered_learning"], {"status": "disabled"})
            self.assertFalse((root / "language" / "layered_learning.json").exists())

    def test_layered_failure_does_not_destroy_the_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            ctx = ProviderContext(root)

            def fail(*args, **kwargs):
                raise RuntimeError("scratch unavailable")

            import nanlmind_hermes.layered_learning as layered
            original = layered.coordinator_for_context
            layered.coordinator_for_context = fail
            try:
                result = chat(ctx, "A river is water.", session_id="resilient")
            finally:
                layered.coordinator_for_context = original
            self.assertEqual(result["response"], "A river is water.")
            self.assertEqual(result["layered_learning"]["status"],
                             "layered_learning_failed")
            self.assertIn("scratch unavailable", result["layered_learning"]["error"])


if __name__ == "__main__":
    unittest.main()
