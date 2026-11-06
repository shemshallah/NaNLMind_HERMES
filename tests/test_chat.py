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
            self.assertEqual(result["surface_measurements"]["records"], 6)
            self.assertEqual(result["surface_measurements"]["memory_merge"]["accepted"], 6)
            self.assertEqual(status(ctx)["mathematical_memory"]["events"], 7)
            self.assertEqual(len(ctx.llm.calls), 2)

    def test_reflection_uses_bounded_prior_conversation_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            chat(ctx, "Earlier: a penguin cannot fly.", session_id="continuity")
            chat(ctx, "Now compare that with a fictional flying penguin.", session_id="continuity")
            reflection_calls = [
                call for call in ctx.llm.calls
                if call.get("purpose") == "nanlmind-hermes-multiresolution-reflection"
            ]
            self.assertEqual(len(reflection_calls), 2)
            prompt = json.loads(reflection_calls[1]["messages"][1]["content"])
            self.assertEqual(len(prompt["prior_context"]), 2)
            self.assertIn("penguin cannot fly", prompt["prior_context"][0]["content"])
            self.assertIn("fictional flying penguin", prompt["user_text"])
            self.assertLessEqual(len(prompt["prior_context"][0]["content"]), 4000)
            self.assertTrue(any(claim.get("subject") == "river"
                                for claim in prompt["prior_universe_memory"]["claims"]))

    def test_chat_measures_exact_bit_nibble_byte_and_text_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "A A river.\n\nThe river flows.", session_id="measure")
            self.assertEqual(result["surface_measurements"]["resolutions"],
                             ["bit", "nibble", "byte", "lexical", "sentence", "paragraph"])
            from nanlmind_hermes.tools import _memory_store
            store = _memory_store(ctx)
            for subject, expected_layer in (
                ("surface:bit-profile:", 0),
                ("surface:nibble-profile:", 0),
                ("surface:byte-profile:", 0),
                ("surface:orthographic-profile:", 3),
                ("surface:sentence-profile:", 5),
                ("surface:paragraph-profile:", 6),
            ):
                rows = store.query(subject=subject + result["source_sha256"][:20])
                self.assertEqual(len(rows), 1, subject)
                self.assertEqual(rows[0]["layer"], expected_layer, subject)
            bit = store.query(subject="surface:bit-profile:" + result["source_sha256"][:20])[0]
            self.assertEqual(bit["expression"]["op"], "utf8_bit_profile")
            self.assertEqual(bit["expression"]["bit_length"],
                             bit["expression"]["zero_bits"] + bit["expression"]["one_bits"])

    def test_sentence_and_paragraph_segmentation_uses_real_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "First sentence. Second sentence!\\n\\nThird sentence?", session_id="boundaries")
            from nanlmind_hermes.tools import _memory_store
            store = _memory_store(ctx)
            digest = result["source_sha256"][:20]
            sentence = store.query(subject="surface:sentence-profile:" + digest)[0]
            paragraph = store.query(subject="surface:paragraph-profile:" + digest)[0]
            self.assertEqual(sentence["expression"]["sentence_count"], 4)
            self.assertEqual(paragraph["expression"]["paragraph_count"], 2)

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
            result = chat(ctx, "x" * (1024 * 1024), session_id="plain")
            self.assertTrue(result["response"])
            self.assertEqual(result["surface_measurements"]["records"], 6)
            self.assertEqual(len(chat_history(ctx, "plain")["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
