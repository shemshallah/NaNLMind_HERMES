import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import corpus_generation_eval as evalm


def _prompt(identifier, prompt, category="custom", groups=None):
    return {
        "id": identifier,
        "category": category,
        "prompt": prompt,
        "keyword_groups": groups or [],
    }


class PromptLoadingTests(unittest.TestCase):
    def test_builtin_prompts_cover_requested_task_families(self):
        prompts, meta = evalm.load_prompts()
        categories = {prompt["category"] for prompt in prompts}
        for expected in ("explanation", "comparison", "causal",
                         "counterfactual", "story", "definition",
                         "arithmetic"):
            self.assertIn(expected, categories)
        self.assertEqual(meta["source"], "builtin")
        self.assertEqual(meta["count"], len(prompts))
        self.assertEqual(len({prompt["id"] for prompt in prompts}),
                         len(prompts))
        self.assertTrue(all(prompt["keyword_groups"] for prompt in prompts))

    def test_jsonl_prompts_support_keywords_groups_and_plain_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes.jsonl"
            path.write_text("\n".join([
                json.dumps({
                    "id": "one",
                    "category": "explanation",
                    "prompt": "Explain a river.",
                    "keywords": ["river", "water"],
                }),
                json.dumps({
                    "id": "two",
                    "category": "definition",
                    "question": "What is ice?",
                    "keyword_groups": [["ice"], ["frozen", "cold"]],
                }),
                json.dumps("Tell a short story."),
                "",
            ]), encoding="utf-8")
            prompts, meta = evalm.load_prompts(path)
        self.assertEqual(meta["source"], str(path))
        self.assertEqual([prompt["id"] for prompt in prompts],
                         ["one", "two", "prompt-003"])
        self.assertEqual(prompts[0]["keyword_groups"],
                         [["river"], ["water"]])
        self.assertEqual(prompts[1]["keyword_groups"],
                         [["ice"], ["frozen", "cold"]])
        self.assertEqual(prompts[2]["category"], "custom")
        self.assertEqual(prompts[2]["keyword_groups"], [])

    def test_invalid_jsonl_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evalm.load_prompts(path)

            path.write_text("\n".join([
                json.dumps({"id": "same", "prompt": "one"}),
                json.dumps({"id": "same", "prompt": "two"}),
            ]), encoding="utf-8")
            with self.assertRaises(ValueError):
                evalm.load_prompts(path)


class CorpusTests(unittest.TestCase):
    def test_corpus_directory_loads_text_jsonl_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            (root / "nested").mkdir(parents=True)
            (root / "a.txt").write_text(
                "The sky is blue.\nWater flows downhill.\n", encoding="utf-8")
            (root / "nested" / "b.jsonl").write_text(
                json.dumps({"text": "A river is water."}) + "\n" +
                json.dumps({"statement": "Ice is frozen water."}) + "\n",
                encoding="utf-8")
            (root / "c.json").write_text(
                json.dumps({"statements": ["The sun is hot."]}),
                encoding="utf-8")
            reference, meta = evalm.load_corpus(root)
        self.assertIsNotNone(reference)
        self.assertEqual(meta["provided"], True)
        self.assertEqual(meta["files"], 3)
        self.assertEqual(meta["sentences"], 5)
        self.assertIn("The sky is blue.",
                      reference.texts)
        self.assertIn("Ice is frozen water.",
                      reference.texts)
        self.assertGreater(meta["tokens"], 0)
        self.assertEqual(len(meta["sha256"]), 64)

    def test_copied_sentence_detection_is_word_based_and_bounded(self):
        reference = evalm.CorpusReference([
            "The sky is blue.",
            "Water flows downhill.",
            "Water flows.",  # shorter than the copy threshold
        ])
        copied = reference.copied_sentences(
            "THE SKY, is blue. Water flows downhill.")
        self.assertIn("The sky is blue.", copied)
        self.assertIn("Water flows downhill.", copied)
        self.assertNotIn("Water flows.", copied)
        self.assertTrue(reference.contains_prompt("The sky is blue."))
        self.assertFalse(reference.contains_prompt("Why is the sky blue?"))


class MetricTests(unittest.TestCase):
    def test_surface_metrics_measure_empty_length_repetition_and_keywords(self):
        response = "The river can carry water and water can flow."
        metrics = evalm.measure_response(
            response,
            "Explain why a river can carry water.",
            keyword_groups=[["river"], ["water"], ["flow", "carry"]],
        )
        self.assertFalse(metrics["empty"])
        self.assertEqual(metrics["tokens"], 9)
        self.assertGreater(metrics["characters"], metrics["tokens"])
        self.assertEqual(metrics["unique_tokens"], 7)
        self.assertEqual(metrics["max_token_repetition"], 2)
        self.assertGreater(metrics["repeated_token_ratio"], 0.0)
        self.assertEqual(metrics["keyword_coverage"], 1.0)
        self.assertEqual(metrics["keywords_matched"], 3)
        self.assertEqual(metrics["novel_reference"], "prompt_only")
        self.assertIsNone(metrics["copied_training_sentence"])
        self.assertIsNone(metrics["prompt_in_corpus"])
        self.assertIsNone(metrics["next_token_nll"])

        empty = evalm.measure_response(
            "   ", "Explain why.", keyword_groups=[["river"]])
        self.assertTrue(empty["empty"])
        self.assertEqual(empty["tokens"], 0)
        self.assertEqual(empty["keyword_coverage"], 0.0)
        self.assertEqual(empty["max_token_repetition"], 0)
        self.assertEqual(empty["novel_3gram_ratio"], 0.0)

    def test_corpus_relative_copy_and_novel_three_grams(self):
        reference = evalm.CorpusReference([
            "The sky is blue.",
            "Water flows downhill.",
        ])
        copied = evalm.measure_response(
            "The sky is blue.",
            "What color is the sky?",
            keyword_groups=[["sky"], ["blue"]],
            corpus=reference,
        )
        self.assertTrue(copied["copied_training_sentence"])
        self.assertIn("The sky is blue.",
                      copied["copied_training_sentences"])
        self.assertEqual(copied["novel_reference"], "corpus+prompt")
        self.assertEqual(copied["novel_3grams"], 0)
        self.assertEqual(copied["novel_3gram_ratio"], 0.0)
        self.assertFalse(copied["prompt_in_corpus"])

        novel = evalm.measure_response(
            "The sky is green and the river is warm.",
            "What color is the sky?",
            keyword_groups=[["sky"]],
            corpus=reference,
        )
        self.assertGreater(novel["novel_3grams"], 0)
        self.assertGreater(novel["novel_3gram_ratio"], 0.0)
        self.assertLessEqual(novel["novel_3gram_ratio"], 1.0)

    def test_keyword_coverage_uses_alternatives_and_does_not_stem(self):
        metrics = evalm.measure_response(
            "A cat is an animal. The cat can run.",
            "Compare a cat and a bird.",
            keyword_groups=[["cat", "cats"], ["bird", "birds"], ["run"]],
        )
        self.assertEqual(metrics["keywords_matched"], 2)
        self.assertEqual(metrics["keywords_total"], 3)
        self.assertEqual(metrics["missing_keywords"], [["bird", "birds"]])
        self.assertAlmostEqual(metrics["keyword_coverage"], 0.6667)


class NLLTests(unittest.TestCase):
    class ConstantModel:
        def __init__(self, probability=0.5):
            self.probability = probability
            self.calls = []

        def _prob(self, context2, context1, target):
            self.calls.append((context2, context1, target))
            return self.probability

    def test_next_token_nll_and_perplexity_are_deterministic(self):
        model = self.ConstantModel(0.5)
        result = evalm.score_next_token_nll(model, "prompt", "cat sat")
        self.assertEqual(result["tokens_scored"], 2)
        self.assertAlmostEqual(result["mean_negative_log_likelihood"],
                               math.log(2), places=6)
        self.assertAlmostEqual(result["perplexity"], 2.0, places=4)
        self.assertEqual(len(model.calls), 2)

        empty = evalm.score_next_token_nll(model, "prompt", "")
        self.assertEqual(empty["tokens_scored"], 0)
        self.assertIsNone(empty["perplexity"])


class JudgeTests(unittest.TestCase):
    def test_fenced_json_rubric_is_normalised(self):
        content = "```json\n" + json.dumps({
            "answers_prompt": 1,
            "coherent": True,
            "novel_combination": 0,
            "unsupported_claim": "false",
            "notes": " diagnostic ",
        }) + "\n```"
        result = evalm.parse_judge_content(content)
        self.assertIs(result["answers_prompt"], True)
        self.assertIs(result["coherent"], True)
        self.assertIs(result["novel_combination"], False)
        self.assertIs(result["unsupported_claim"], False)
        self.assertEqual(result["notes"], " diagnostic ")

    def test_invalid_rubrics_become_errors(self):
        missing = evalm.parse_judge_content(json.dumps({
            "answers_prompt": 1,
            "coherent": 1,
        }))
        self.assertIn("error", missing)
        invalid = evalm.parse_judge_content(json.dumps({
            "answers_prompt": 2,
            "coherent": 1,
            "novel_combination": 0,
            "unsupported_claim": 0,
            "notes": "x",
        }))
        self.assertIn("error", invalid)
        self.assertIn("error", evalm.parse_judge_content("not json"))


class StudentAndReportTests(unittest.TestCase):
    class FakeStudent:
        interface = "nanlm_chat"

        def __init__(self):
            self.calls = []

        def reply(self, prompt):
            self.calls.append(prompt)
            if prompt.startswith("What is a river"):
                return {
                    "response": "",
                    "provider": "fake-native",
                    "model": "temporal-mirror/test",
                    "probe": True,
                    "state_mutated": False,
                }
            return {
                "response": "The sky is blue and water flows downhill.",
                "provider": "fake-native",
                "model": "temporal-mirror/test",
                "probe": True,
                "state_mutated": False,
                "native_synthesis": {
                    "synthesis": "temporal_mirror_lm/1",
                    "response_templates": 0,
                    "retrieval": 0,
                    "probe": True,
                    "state_mutated": False,
                    "temporal_frame": {
                        "anchor": "sky",
                        "steps": 8,
                        "score": -1.25,
                        "coverage": 0.5,
                        "output_drive": 1.0,
                        "mirror_state": True,
                    },
                },
            }

    def _state(self, root):
        (root / "language").mkdir(parents=True)
        (root / "language" / "temporal_mirror.json").write_text(
            json.dumps({
                "schema": "nanlm.temporal-mirror-lm/1",
                "vocab": {"the": 1, "sky": 2},
                "seen": ["The sky is blue."],
                "bi": {"the": {"sky": 1}},
                "tri": {"the\x00sky": {"is": 1}},
            }), encoding="utf-8")
        (root / "language" / "layered_learning.json").write_text(
            json.dumps({"turns": 3, "rows": [{"turn": 1}]}),
            encoding="utf-8")

    def _report(self, root):
        student = self.FakeStudent()
        corpus = evalm.CorpusReference([
            "The sky is blue.",
            "Water flows downhill.",
            "A river is water.",
        ])
        prompts = [
            _prompt("one", "Explain why a river can carry water.",
                    "explanation", [["river"], ["water"]]),
            _prompt("two", "What is a river?", "definition",
                    [["river"], ["water"]]),
        ]
        judged = []

        def fake_judge(prompt, response, references):
            judged.append((prompt, response, references))
            return {
                "answers_prompt": bool(response),
                "coherent": bool(response),
                "novel_combination": False,
                "unsupported_claim": False,
                "notes": "fake diagnostic",
            }

        report = evalm.run_evaluation(
            root, prompts, student=student, corpus=corpus,
            judge=fake_judge, nll_model=NLLTests.ConstantModel(),
            nll_requested=True,
            prompt_meta={"source": "test", "count": 2, "sha256": "x"},
            corpus_meta={"provided": True, "sentences": 3},
            judge_meta={"enabled": True, "model": "fake-judge"},
        )
        return report, student, judged

    def test_run_evaluation_records_replies_metrics_judge_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            root.mkdir()
            self._state(root)
            report, student, judged = self._report(root)

            self.assertEqual(report["schema"], evalm.SCHEMA)
            self.assertEqual(len(report["rows"]), 2)
            self.assertEqual(student.calls, [
                "Explain why a river can carry water.",
                "What is a river?",
            ])
            self.assertEqual(len(judged), 2)
            first, second = report["rows"]
            self.assertTrue(first["metrics"]["copied_training_sentence"])
            self.assertGreater(first["metrics"]["novel_3grams"], 0)
            self.assertEqual(first["student"]["provider"], "fake-native")
            self.assertEqual(first["student"]["probe"], True)
            self.assertEqual(first["student"]["state_mutated"], False)
            self.assertEqual(first["student"]["temporal_frame"]["anchor"],
                             "sky")
            self.assertEqual(second["response"], "")
            self.assertTrue(second["metrics"]["empty"])
            self.assertEqual(first["judge"]["answers_prompt"], True)
            self.assertEqual(second["judge"]["answers_prompt"], False)

            summary = report["summary"]
            self.assertEqual(summary["prompts"], 2)
            self.assertEqual(summary["empty_responses"], 1)
            self.assertEqual(summary["empty_rate"], 0.5)
            self.assertEqual(summary["copied_training_sentences"], 1)
            self.assertEqual(summary["judge"]["judged"], 2)
            self.assertEqual(summary["judge"]["answers_prompt"], 1)
            self.assertTrue(summary["nll"]["available"])
            self.assertGreater(summary["nll"]["tokens_scored"], 0)
            self.assertIn("explanation", summary["by_category"])
            self.assertIn("definition", summary["by_category"])
            self.assertEqual(summary["probe_read_only_rows"], 2)
            self.assertTrue(report["state_probe"]["unchanged"])
            self.assertEqual(report["state"]["temporal_mirror"]["vocabulary"],
                             2)
            self.assertEqual(report["state"]["layered_learning"]["turns"], 3)
            self.assertFalse(report["boundaries"]["llm_wrote_student_reply"])
            self.assertFalse(report["boundaries"]["judge_is_ground_truth"])
            self.assertFalse(report["boundaries"]["held_out_proven"])

    def test_report_writes_json_and_markdown_with_reply_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            root.mkdir()
            self._state(root)
            report, _, _ = self._report(root)
            output = Path(tmp) / "nested" / "report.json"
            json_path, markdown_path = evalm.write_report(report, output)
            self.assertEqual(json_path, output)
            self.assertEqual(markdown_path, output.with_suffix(".md"))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], evalm.SCHEMA)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Explain why a river can carry water.", markdown)
            self.assertIn("The sky is blue and water flows downhill.", markdown)
            self.assertIn("fake diagnostic", markdown)
            self.assertIn("Boundaries and interpretation", markdown)
            self.assertIn("not ground truth", markdown)

    def test_markdown_output_path_uses_json_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            root.mkdir()
            self._state(root)
            report, _, _ = self._report(root)
            requested = Path(tmp) / "report.md"
            json_path, markdown_path = evalm.write_report(report, requested)
            self.assertEqual(json_path, requested.with_suffix(".json"))
            self.assertEqual(markdown_path, requested)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())


class NativeStudentProbeTests(unittest.TestCase):
    def test_native_student_calls_registered_chat_with_probe_true(self):
        calls = []
        original = evalm.register

        def fake_register(ctx):
            def handler(params):
                calls.append(params)
                return json.dumps({
                    "response": "A native reply.",
                    "provider": "nanlm-native",
                    "model": "temporal-mirror/1",
                    "probe": True,
                    "state_mutated": False,
                })
            ctx.tools["nanlm_chat"] = handler

        evalm.register = fake_register
        try:
            with tempfile.TemporaryDirectory() as tmp:
                student = evalm.NativeChatStudent(Path(tmp))
                result = student.reply("What is a river?")
        finally:
            evalm.register = original
        self.assertEqual(result["response"], "A native reply.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["message"], "What is a river?")
        self.assertIs(calls[0]["probe"], True)
        self.assertEqual(calls[0]["session_id"], evalm.DEFAULT_SESSION_ID)


if __name__ == "__main__":
    unittest.main()
