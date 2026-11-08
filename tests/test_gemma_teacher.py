import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.gemma_teacher import (
    GemmaTeacher,
    GemmaTeacherError,
    HermesLLMTransport,
    LLMTeacher,
    OllamaTransport,
    OpenAICompatibleTransport,
    build_prompt,
    parse_response,
    teacher_for_context,
)
from nanlmind_hermes.layered_learning import LayeredLearningCoordinator, LearningPolicy

RECORDS = [
    {"subject": "copper", "predicate": "is", "object": "metal",
     "definition": "a reddish metal", "layer": 5, "confidence": 0.9,
     "evidence": "teacher"},
    {"subject": "metal", "predicate": "conducts", "object": "electricity",
     "definition": "a conductor", "layer": 7, "confidence": 0.8,
     "evidence": "teacher"},
]


class _FakeLLM:
    def complete(self, messages, **kwargs):
        class Result:
            text = json.dumps(RECORDS)
        return Result()


class _FakeCtx:
    def __init__(self, config):
        self.config = config
        self.llm = _FakeLLM()

    def get_config(self, name, default=None):
        return self.config.get(name, default)


class GemmaTeacherTests(unittest.TestCase):
    def test_parse_and_validate(self):
        parsed = parse_response("prefix " + json.dumps(RECORDS) + " suffix")
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["subject"], "copper")
        self.assertEqual(parsed[1]["layer"], 7)
        with self.assertRaises(GemmaTeacherError):
            parse_response("not json")
        with self.assertRaises(GemmaTeacherError):
            parse_response(json.dumps([{"subject": "", "object": "x"}]))
        with self.assertRaises(GemmaTeacherError):
            parse_response(json.dumps([{"subject": "a", "object": "b", "layer": 10}]))

    def test_prompt_is_bounded(self):
        prompt = build_prompt("metals", 4)
        self.assertIn("metals", prompt)
        self.assertIn("L0-L9", prompt) if "L0-L9" in prompt else None
        with self.assertRaises(GemmaTeacherError):
            build_prompt("", 4)
        with self.assertRaises(GemmaTeacherError):
            build_prompt("metals", 0)

    def test_teacher_for_context_is_provider_universal(self):
        hermes = teacher_for_context(_FakeCtx({}))
        self.assertIsInstance(hermes.transport, HermesLLMTransport)
        ollama = teacher_for_context(_FakeCtx({
            "teacher_provider": "ollama",
            "teacher_endpoint": "http://127.0.0.1:11434/api/generate",
            "teacher_model": "gemma4:31b-cloud",
        }))
        self.assertIsInstance(ollama.transport, OllamaTransport)
        openai = teacher_for_context(_FakeCtx({
            "teacher_provider": "openai",
            "teacher_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "teacher_model": "local-model",
        }))
        self.assertIsInstance(openai.transport, OpenAICompatibleTransport)
        self.assertIsInstance(LLMTeacher(HermesLLMTransport(_FakeCtx({}))), LLMTeacher)

    def test_teach_feeds_layered_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = LayeredLearningCoordinator(
                Path(directory),
                policy=LearningPolicy(min_support=1, min_confidence=0.1,
                                      require_explicit_approval=False),
            )
            teacher = GemmaTeacher(lambda _prompt: json.dumps(RECORDS))
            result = teacher.teach(coordinator, "metals", 2)
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["observation"]["turn"], 1)
            self.assertGreaterEqual(result["observation"]["scratch"]["records"], 10)
            self.assertEqual(len(coordinator.rows), 2)
            self.assertFalse(result["canonical_training_committed"])


if __name__ == "__main__":
    unittest.main()
