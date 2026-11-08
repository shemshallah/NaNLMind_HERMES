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
        self.config = {"state_dir": str(root), "chat_mode": "provider"}
        self.llm = FakeLLM()

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class NativeChatContext:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_learning": False, "chat_mode": "native"}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class HermesChatTests(unittest.TestCase):
    def test_native_chat_without_a_language_provider_parses_updates_and_synthesizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            first = chat(ctx, "A river is water.", session_id="native-demo")
            self.assertEqual(first["provider"], "nanlm-native")
            self.assertEqual(first["native_synthesis"]["synthesis"],
                             "temporal_mirror_lm/1")
            self.assertEqual(first["native_synthesis"]["response_templates"], 0)
            self.assertGreaterEqual(first["reflection"]["proposed_discoveries"], 1)
            self.assertTrue(first["scratch"]["scratch_id"].startswith("chat-"))
            self.assertEqual(len(chat_history(ctx, "native-demo")["messages"]), 2)
            self.assertGreaterEqual(first["native_synthesis"]["heads"]["leaf_count"], 1)
            self.assertGreaterEqual(first["native_synthesis"]["heads"]["head_count"], 1)
            chat(ctx, "Water can flow.", session_id="native-demo")
            second = chat(ctx, "What is a river?", session_id="native-demo")
            self.assertEqual(second["provider"], "nanlm-native")
            self.assertTrue(second["response"])
            frame = second["native_synthesis"]["temporal_frame"]
            self.assertEqual(frame["scored_per_step"],
                             second["native_synthesis"]["vocabulary"]["vocabulary"])
            self.assertTrue(frame["mirror_state"])
            self.assertTrue(second["response"].endswith((".", "!", "?")))
            self.assertGreaterEqual(len(second["native_synthesis"]["temporal_frame"]["sentences"]), 1)

    def test_native_chat_replies_are_exposure_only_and_questions_are_not_learned(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            chat(ctx, "A river is water.", session_id="exposure")
            chat(ctx, "Water can flow.", session_id="exposure")
            taught = chat(ctx, "What is a river?", session_id="exposure")
            self.assertEqual(taught["native_synthesis"]["response_templates"], 0)
            self.assertEqual(taught["native_synthesis"]["retrieval"], 0)
            self.assertTrue(taught["response"])
            for scripted in ("I recall", "Here is", "I learned", "I have no",
                             "Teach me"):
                self.assertNotIn(scripted, taught["response"])
            chat(ctx, "What can a river do?", session_id="exposure")
            from nanlmind_hermes.tools import _memory_store
            relations = [row for row in _memory_store(ctx).query(limit=500)
                         if row.get("kind") == "relation"]
            self.assertFalse(any("what can a river do" in str(row).casefold()
                                 for row in relations))
            miss = chat(ctx, "What is dark matter?", session_id="exposure")
            self.assertTrue(miss["response"])
            self.assertGreaterEqual(
                miss["native_synthesis"]["temporal_frame"]["output_drive"], 0.35)
            self.assertTrue(miss["response"].endswith((".", "!", "?")))

    def test_chat_answers_yes_no_from_the_learned_relation_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            chat(ctx, "Copper is a chemical element.", session_id="store")
            reply = chat(ctx, "Is copper a chemical element?", session_id="store")
            self.assertTrue(reply["native_synthesis"]["structured"])
            self.assertIn("chemical element", reply["response"].lower())

    def test_correction_rejects_the_answer_and_deliberates_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            for sentence in ("A bird can fly.", "A bird is an animal.",
                             "Birds have wings."):
                chat(ctx, sentence, session_id="correct")
            first = chat(ctx, "What can a bird do?", session_id="correct")
            self.assertTrue(first["response"])
            feedback = chat(ctx, "No, that's incorrect.", session_id="correct")
            synthesis = feedback["native_synthesis"]
            self.assertTrue(synthesis["correction"])
            self.assertTrue(feedback["response"])
            self.assertNotEqual(feedback["response"], first["response"])
            self.assertTrue(synthesis["temporal_frame"]["considered"])

    def test_forced_futures_do_not_collapse_to_one_attractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            for sentence in ("A river is water.", "A bird is an animal.",
                             "Fish live in water."):
                chat(ctx, sentence, session_id="attractor")
            first = chat(ctx, "What is a volcano?", session_id="attractor")
            second = chat(ctx, "What is a quark?", session_id="attractor")
            third = chat(ctx, "What is a comet?", session_id="attractor")
            replies = [first["response"], second["response"], third["response"]]
            self.assertTrue(all(replies))
            self.assertTrue(all(reply.endswith((".", "!", "?")) for reply in replies))
            self.assertGreater(len(set(replies)), 1)
            self.assertGreaterEqual(
                first["native_synthesis"]["temporal_frame"]["output_drive"], 0.35)

    def test_native_chat_predicts_forward_from_the_learned_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = NativeChatContext(Path(tmp))
            for sentence in ("The sun is hot.", "The sun causes light.",
                             "Rain is water.", "Cold water can freeze."):
                chat(ctx, sentence, session_id="focus")
            reply = chat(ctx, "Tell me about the sun.", session_id="focus")
            self.assertTrue(reply["response"])
            self.assertNotEqual(reply["response"], "Tell me about the sun.")
            self.assertIn("sun", reply["response"].lower())
            self.assertEqual(reply["native_synthesis"]["response_templates"], 0)
            frame = reply["native_synthesis"]["temporal_frame"]
            self.assertIn(frame["anchor"], {"sun"})
            self.assertGreaterEqual(frame["steps"], 1)

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

    def test_chat_autonomously_proposes_clocked_mode_without_transmitting_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "Compare these source documents.",
                          session_id="clocked-proposal",
                          sources=[{"id": "spec-a", "text": "The system uses NaN16."},
                                   {"id": "spec-b", "text": "The system uses Boolean heads."}])
            self.assertEqual(result["clocked_mode"]["mode_plan"]["mode"], "CLOCKED_COMPARE")
            self.assertEqual(result["clocked_mode"]["task_plan"]["schema"],
                             "nanlm.clocked-task-plan/1")
            self.assertEqual(len(result["clocked_mode"]["approval_digest"]), 64)
            self.assertTrue(result["clocked_mode"]["mode_plan"]["requires_approval"])
            self.assertTrue(result["clocked_scratch_id"].startswith("chat-"))
            chat_call = next(call for call in ctx.llm.calls
                             if call.get("purpose") == "nanlmind-hermes-chat")
            self.assertIn("approval_digest", chat_call["messages"][1]["content"])
            selector_call = next(call for call in ctx.llm.calls
                                 if call.get("purpose") == "nanlm-hermes-clocked-mode-selection")
            selector_payload = selector_call["messages"][1]["content"]
            self.assertNotIn("The system uses NaN16.", selector_payload)
            self.assertNotIn("The system uses Boolean heads.", selector_payload)
            self.assertEqual(len(chat_history(ctx, "clocked-proposal")["messages"]), 2)

    def test_chat_system_prompt_includes_architecture_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            chat(ctx, "What are you?", session_id="architecture")
            system = ctx.llm.calls[0]["messages"][0]["content"]
            self.assertIn("C(256,2)=32,640", system)
            self.assertIn("not a transformer", system)
            self.assertIn("NANNET", system)

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
                ("surface:nibble-profile:", 1),
                ("surface:byte-profile:", 2),
                ("surface:orthographic-profile:", 5),
                ("surface:sentence-profile:", 7),
                ("surface:paragraph-profile:", 8),
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
            result = chat(ctx, "First sentence. Second sentence!\n\nThird sentence?", session_id="boundaries")
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


    def test_free_form_chat_reflection_updates_native_heads_and_hypergraph(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_native.py"
            fake.write_text(
                "import pathlib, sys\n"
                "mode = sys.argv[1]\n"
                "if mode == 'logic-train':\n"
                " state=pathlib.Path(sys.argv[4]); state.parent.mkdir(parents=True,exist_ok=True); state.write_bytes(b'gate')\n"
                " print('stage=boolean_foundation passed=yes correct=1024/1024 accuracy=1 detail=distinct_rules=16 failed_heads=0')\n"
                " print('stage=deep_logic_circuits passed=yes correct=32/32 accuracy=1 detail=circuits')\n"
                " print('stage=words_after_logic passed=yes correct=10/10 accuracy=1 detail=words')\n"
                " print('state_bytes=4 state_digest=' + 'a'*64)\n"
                " print('result=PASS')\n"
                "else:\n"
                " records, state = sys.argv[2], pathlib.Path(sys.argv[3]); state.parent.mkdir(parents=True,exist_ok=True); state.write_bytes(b'native')\n"
                " print('relation_train status=updated epochs=1 train_events=1 duplicate_events=0 changed_heads=3 mirror_pair_updates=3 hyperedges_added=1 state_bytes=6 state_digest_before=' + 'b'*64 + ' state_digest_after=' + 'c'*64)\n",
                encoding="utf-8",
            )
            class LearningLLM:
                def complete(self, **kwargs):
                    if kwargs.get("purpose") == "nanlmind-hermes-chat":
                        return SimpleNamespace(text="Penguins can swim.", provider="fake", model="chat")
                    return SimpleNamespace(
                        text=json.dumps({"discoveries": [{
                            "kind": "relation", "subject": "penguin", "predicate": "can",
                            "object": "swim", "resolution": "lexical",
                            "polarity": "supports", "confidence": 0.95,
                            "evidence": {"turn": 1}, "provenance": {"alignment_id": "penguin-swim"}
                        }]}),
                        provider="fake", model="reflection",
                    )
            ctx = FakeContext(root)
            ctx.llm = LearningLLM()
            ctx.config["runner_command"] = [sys.executable, str(fake)]
            result = chat(ctx, "Penguins can swim.", session_id="native-learning")
            self.assertEqual(result["native_learning"]["status"], "updated")
            self.assertEqual(result["native_learning"]["changed_heads"], 3)
            self.assertEqual(result["native_learning"]["mirror_pair_updates"], 3)
            self.assertEqual(result["native_learning"]["hyperedges_added"], 1)
            self.assertTrue(Path(result["native_learning"]["native_state_path"]).is_file())

    def test_large_plain_text_is_not_forced_through_n16txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext(Path(tmp))
            result = chat(ctx, "x" * (1024 * 1024), session_id="plain")
            self.assertTrue(result["response"])
            self.assertEqual(result["surface_measurements"]["records"], 6)
            self.assertEqual(len(chat_history(ctx, "plain")["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
