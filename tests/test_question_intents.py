import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.chat import _native_parse
from nanlmind_hermes.mth_layers import structured_candidates
from nanlmind_hermes.tools import chat

TEACHING = (
    "Dogs bark.",
    "Cats meow.",
    "Birds fly.",
    "Red and blue create purple.",
    "Yellow and blue create green.",
    "A square has four sides.",
    "Water freezes into ice.",
    "Cold causes freezing.",
    "Ice is frozen water.",
)


def relation_rows():
    rows = []
    for index, sentence in enumerate(TEACHING):
        for record in _native_parse(sentence, "intent-test", "0" * 64):
            item = record.get("item") or {}
            if item.get("kind") == "relation":
                rows.append({**item, "confidence": 0.9})
    return rows


class QuestionIntentTests(unittest.TestCase):
    def test_action_questions_use_the_exact_subject(self):
        rows = relation_rows()
        dogs = structured_candidates("What do dogs do?", rows)
        cats = structured_candidates("What do cats do?", rows)
        birds = structured_candidates("What do birds do?", rows)
        self.assertEqual(dogs[0]["text"], "Dogs can bark.")
        self.assertEqual(cats[0]["text"], "Cats can meow.")
        self.assertEqual(birds[0]["text"], "Birds can fly.")

    def test_red_blue_is_not_overwritten_by_yellow_blue(self):
        rows = relation_rows()
        for question in ("What do red and blue make?",
                         "What does red and blue make?"):
            with self.subTest(question=question):
                candidates = structured_candidates(question, rows)
                self.assertEqual(candidates[0]["text"], "Red and blue make purple.")
                self.assertNotIn("green", candidates[0]["text"].casefold())

    def test_reverse_object_lookup_returns_square(self):
        rows = relation_rows()
        candidates = structured_candidates("What shape has four sides?", rows)
        self.assertEqual(candidates[0]["text"], "A square has four sides.")

    def test_causal_question_returns_ice_evidence(self):
        rows = relation_rows()
        candidates = structured_candidates("What happens when water freezes?", rows)
        self.assertTrue(candidates)
        joined = " ".join(item["text"].casefold() for item in candidates)
        self.assertIn("ice", joined)

    def test_causal_question_prefers_outcome_over_event_restatement(self):
        rows = relation_rows() + [
            {"kind": "relation", "subject": "Cold", "predicate": "causes",
             "object": "water to freeze", "confidence": 0.99},
        ]
        candidates = structured_candidates("What happens when water freezes?", rows)
        self.assertTrue(candidates)
        first = candidates[0]["text"].casefold()
        self.assertTrue("ice" in first or "frozen" in first, first)
        self.assertNotIn("cold causes", first)

    def test_unrelated_question_has_no_structured_answer(self):
        rows = relation_rows()
        self.assertEqual(structured_candidates(
            "What is the capital of France?", rows), [])


class _NativeChatContext:
    def __init__(self, root):
        self.config = {"state_dir": str(root), "chat_learning": False,
                       "chat_mode": "native"}

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class NativeChatQuestionIntentTests(unittest.TestCase):
    """End-to-end native chat path, not only direct relation candidates."""

    def test_native_chat_answers_the_plateau_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _NativeChatContext(Path(tmp))
            for statement in (
                "Dogs bark.",
                "Cats meow.",
                "Birds fly.",
                "Red and blue create purple.",
                "Yellow and blue create green.",
                "A square has four sides.",
                "Frozen water is ice.",
            ):
                chat(ctx, statement, session_id="question-intents")

            dogs = chat(ctx, "What do dogs do?",
                        session_id="question-intents")
            self.assertTrue(dogs["native_synthesis"]["structured"])
            self.assertIn("bark", dogs["response"].casefold())

            cats = chat(ctx, "What do cats do?",
                        session_id="question-intents")
            self.assertTrue(cats["native_synthesis"]["structured"])
            self.assertIn("meow", cats["response"].casefold())

            red_blue = chat(ctx, "What do red and blue make?",
                            session_id="question-intents")
            self.assertTrue(red_blue["native_synthesis"]["structured"])
            self.assertIn("purple", red_blue["response"].casefold())
            self.assertNotIn("green", red_blue["response"].casefold())

            freezing = chat(ctx, "What happens when water freezes?",
                            session_id="question-intents")
            self.assertTrue(freezing["native_synthesis"]["structured"])
            self.assertTrue("ice" in freezing["response"].casefold()
                            or "frozen" in freezing["response"].casefold())

            shaped = chat(ctx, "What shape has four sides?",
                          session_id="question-intents")
            self.assertTrue(shaped["native_synthesis"]["structured"])
            self.assertIn("square", shaped["response"].casefold())

    def test_native_chat_does_not_answer_an_unlearned_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _NativeChatContext(Path(tmp))
            chat(ctx, "Dogs bark.", session_id="question-intents-miss")
            miss = chat(ctx, "What do dragons do?",
                        session_id="question-intents-miss")
            self.assertFalse(miss["native_synthesis"]["structured"])


if __name__ == "__main__":
    unittest.main()
