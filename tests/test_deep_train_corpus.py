import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.temporal_lm import TemporalMirrorLM
from scripts import deep_train_corpus as deep

SCHEMA = deep.TRAINABLE_SCHEMA


def _record(record_id, text, split="train", stage="synthetic",
            schema=SCHEMA, source="test"):
    return {
        "schema": schema,
        "record_id": record_id,
        "text": text,
        "stage": stage,
        "source": source,
        "split": split,
        "provenance": {},
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class FakeProvider:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error:
            raise RuntimeError(self.error)
        if not self.responses:
            raise RuntimeError("no fake response queued")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


class LoadTests(unittest.TestCase):
    def test_load_validates_dedupes_and_reports_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.jsonl"
            _write_jsonl(first, [
                _record("a1", "Alpha beta gamma."),
                _record("a2", "Delta epsilon zeta.", stage="two"),
                _record("a1", "Duplicate alpha sentence."),
                _record("bad-schema", "Other schema text.", schema="other"),
                _record("bad-text", "   "),
                _record("bad-split", "A valid text with words.", split="mystery"),
                {"schema": SCHEMA, "text": "No identifier.", "split": "train"},
            ])
            with first.open("a", encoding="utf-8") as stream:
                stream.write("this is not json\n")
            _write_jsonl(root / "b.jsonl", [
                _record("b1", "Validation sentence here.", split="validation"),
                _record("a2", "Cross-file duplicate."),
            ])
            loaded = deep.load_trainable(root)

        counts = loaded["totals"]
        self.assertEqual(loaded["files"], ["a.jsonl", "b.jsonl"])
        self.assertEqual(counts["records"], 9)
        self.assertEqual(counts["accepted"], 3)
        self.assertEqual(counts["train"], 2)
        self.assertEqual(counts["validation"], 1)
        self.assertEqual(counts["duplicates"], 2)
        self.assertEqual(counts["rejected"], 4)
        self.assertEqual(counts["malformed"], 1)
        self.assertEqual(counts["invalid_schema"], 1)
        self.assertEqual(counts["invalid_record_id"], 1)
        self.assertEqual(counts["invalid_text"], 1)
        self.assertEqual(counts["invalid_split"], 1)
        self.assertEqual(
            [item["record_id"] for item in loaded["train"]], ["a1", "a2"])
        self.assertEqual(
            [item["record_id"] for item in loaded["validation"]], ["b1"])
        by_path = {item["path"]: item for item in loaded["per_file"]}
        self.assertEqual(by_path["a.jsonl"]["records"], 7)
        self.assertEqual(by_path["a.jsonl"]["accepted"], 2)
        self.assertEqual(by_path["a.jsonl"]["duplicates"], 1)
        self.assertEqual(by_path["a.jsonl"]["rejected"], 4)
        self.assertEqual(by_path["a.jsonl"]["malformed"], 1)
        self.assertEqual(by_path["b.jsonl"]["accepted"], 1)
        self.assertEqual(by_path["b.jsonl"]["duplicates"], 1)
        self.assertFalse(loaded["validation_fallback"]["used"])
        self.assertEqual(loaded["declared_validation"], 1)
        self.assertEqual(len(loaded["digests"]["corpus"]), 64)

    def test_validation_fallback_by_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "all.jsonl", [
                _record("r%03d" % index,
                        "Record number %d contains several ordinary words."
                        % index, stage="fallback")
                for index in range(200)
            ])
            first = deep.load_trainable(root)
            second = deep.load_trainable(root)

        self.assertTrue(first["validation_fallback"]["used"])
        self.assertEqual(first["validation_fallback"]["moved"], 10)
        self.assertEqual(len(first["validation"]), 10)
        self.assertEqual(len(first["train"]), 190)
        self.assertEqual(
            {item["record_id"] for item in first["validation"]},
            {item["record_id"] for item in second["validation"]},
        )
        self.assertTrue(all(
            item["split"] == "validation" for item in first["validation"]))
        self.assertEqual(first["stages"]["fallback"],
                         {"train": 190, "validation": 10})


class MetricTests(unittest.TestCase):
    def test_evaluate_heldout_reports_fixed_sample_metrics(self):
        model = TemporalMirrorLM()
        for text in ("the cat sat on the mat.", "a dog ran in the park."):
            self.assertTrue(model.observe(text))
        records = [
            {"record_id": "v2", "text": "the cat sat on the mat."},
            {"record_id": "v1", "text": "a dog ran in the park."},
        ]
        metrics = deep.evaluate_heldout(model, records)
        self.assertEqual(metrics["records"], 2)
        self.assertGreater(metrics["tokens_scored"], 0)
        self.assertGreater(metrics["mean_negative_log_likelihood"], 0.0)
        self.assertIsNotNone(metrics["perplexity"])
        self.assertGreater(metrics["vocabulary_coverage"], 0.0)
        self.assertEqual(metrics["record_ids_digest"],
                         deep._digest_ids(records))

    def test_select_eval_sample_is_fixed_and_bounded(self):
        records = [
            {"record_id": "r%02d" % index, "text": "sentence %d here." % index}
            for index in range(20)
        ]
        first = deep.select_eval_sample(records, 5)
        second = deep.select_eval_sample(records, 5)
        self.assertEqual(len(first), 5)
        self.assertEqual([item["record_id"] for item in first],
                         [item["record_id"] for item in second])
        self.assertEqual(len(deep.select_eval_sample(records, 0)), 20)

    def test_train_epochs_clears_only_replay_guard_and_saves(self):
        model = TemporalMirrorLM()
        model.seen = ["stale-digest"]
        train = [
            {"record_id": "t1", "text": "alpha beta gamma delta."},
            {"record_id": "t2", "text": "epsilon zeta eta theta."},
        ]
        heldout = [{"record_id": "v1", "text": "alpha beta epsilon zeta."}]
        probes = [{
            "source_record_id": "t1",
            "prompt": "alpha beta",
            "expected": "gamma",
            "origin": "test",
        }]
        saves = []

        def save(epoch):
            saves.append(epoch)
            return {
                "epoch": epoch,
                "state_dir": "/tmp/synthetic-state",
                "model_path": "/tmp/synthetic-state/model.json",
                "model_sha256": "%064x" % epoch,
            }

        result = deep.train_epochs(
            model, train, heldout, 2, save, probes=probes)

        self.assertEqual(saves, [1, 2])
        self.assertEqual(len(result["epochs_run"]), 2)
        for row in result["epochs_run"]:
            self.assertEqual(row["observed"], 2)
            self.assertEqual(row["rejected"], 0)
            self.assertEqual(row["checkpoint"]["epoch"], row["epoch"])
            self.assertEqual(row["model"]["vocabulary"], 9)
            self.assertGreater(row["model"]["bigram_contexts"], 0)
            self.assertGreater(row["model"]["trigram_contexts"], 0)
            self.assertIsNotNone(row["mean_nll_delta_from_before"])
            self.assertIsNotNone(row["probes_delta_from_before"])
        self.assertNotIn("stale-digest", model.seen)
        self.assertGreater(result["before"]["heldout"]["tokens_scored"], 0)
        self.assertGreater(result["before"]["probes"]["tokens_scored"], 0)
        self.assertEqual(result["after"]["heldout"],
                         result["epochs_run"][-1]["heldout"])
        self.assertIsNotNone(
            result["after"]["delta_from_before"]["mean_nll_delta"])


class GemmaPlanTests(unittest.TestCase):
    def _train(self):
        return [
            {"record_id": "t1", "text": "alpha beta gamma delta.",
             "stage": "one", "source": "test"},
            {"record_id": "t2", "text": "epsilon zeta eta theta.",
             "stage": "two", "source": "test"},
            {"record_id": "t3", "text": "iota kappa lambda mu.",
             "stage": "one", "source": "test"},
        ]

    def test_plan_is_parsed_bounded_and_applied(self):
        provider = FakeProvider([json.dumps({
            "rationale": "focus one",
            "epochs": 99,
            "max_records_per_epoch": 999,
            "focus_stages": ["one", "unknown"],
            "notes": "bounded",
        })])
        result = deep.request_plan(
            provider, self._train(), default_epochs=5, max_epochs=20,
            max_samples=3)
        self.assertEqual(result["status"], "applied")
        self.assertIsNone(result["error"])
        self.assertEqual(result["plan"]["epochs"], 20)
        self.assertEqual(result["plan"]["max_records_per_epoch"], 3)
        self.assertEqual(result["plan"]["focus_stages"], ["one"])
        self.assertTrue(result["plan"]["clamped"])
        self.assertEqual(result["samples"], 3)
        self.assertEqual(len(provider.calls), 1)

    def test_plan_invalid_json_and_provider_error_fall_back(self):
        invalid = deep.request_plan(
            FakeProvider(["not json"]), self._train(),
            default_epochs=4, max_samples=2)
        self.assertEqual(invalid["status"], "fallback")
        self.assertEqual(invalid["plan"]["epochs"], 4)
        self.assertIn("error", invalid)
        self.assertTrue(invalid["error"])

        failed = deep.request_plan(
            FakeProvider(error="endpoint down"), self._train(),
            default_epochs=4, max_samples=2)
        self.assertEqual(failed["status"], "fallback")
        self.assertEqual(failed["plan"]["epochs"], 4)
        self.assertIn("endpoint down", failed["error"])

    def test_fenced_json_is_accepted_but_surrounding_prose_is_not(self):
        payload = json.dumps({
            "rationale": "fenced",
            "epochs": 3,
            "max_records_per_epoch": 0,
            "focus_stages": [],
            "notes": "",
        })
        fenced = deep.request_plan(
            FakeProvider(["```json\n" + payload + "\n```"]),
            self._train(), default_epochs=5, max_samples=2)
        self.assertEqual(fenced["status"], "applied")
        self.assertEqual(fenced["plan"]["epochs"], 3)

        prose = deep.request_plan(
            FakeProvider(["here is the plan: " + payload]),
            self._train(), default_epochs=5, max_samples=2)
        self.assertEqual(prose["status"], "fallback")

    def test_ordered_records_applies_focus_and_cap(self):
        records = self._train()
        plan = {"focus_stages": ["two"], "max_records_per_epoch": 2}
        ordered = deep.ordered_train_records(records, plan)
        self.assertEqual([item["record_id"] for item in ordered], ["t2", "t1"])


class GemmaProbeTests(unittest.TestCase):
    def _train(self):
        return [
            {"record_id": "t1", "text": "alpha beta gamma delta epsilon.",
             "stage": "one", "source": "test"},
            {"record_id": "t2", "text": "one two three four five six.",
             "stage": "two", "source": "test"},
            {"record_id": "t3", "text": "red green blue yellow orange.",
             "stage": "one", "source": "test"},
        ]

    def test_probes_are_strictly_validated_and_no_heldout_text_is_sent(self):
        provider = FakeProvider([json.dumps({"probes": [
            {"source_record_id": "t1", "prompt": "alpha beta",
             "expected": "gamma"},
            {"source_record_id": "t1", "prompt": "alpha be",
             "expected": "ta"},
            {"source_record_id": "t1", "prompt": "alpha beta",
             "expected": "gam"},
            {"source_record_id": "missing", "prompt": "alpha",
             "expected": "beta"},
        ]})])
        result = deep.request_probes(
            provider, self._train(), limit=4, max_samples=3)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(result["probes"]), 1)
        self.assertEqual(result["probes"][0]["expected"], "gamma")
        self.assertEqual(result["rejected"], 3)
        self.assertIn("prompt_not_boundary", result["rejected_reasons"])
        self.assertIn("expected_not_boundary", result["rejected_reasons"])
        self.assertIn("unknown_source", result["rejected_reasons"])
        prompt = json.dumps(provider.calls[0]["messages"])
        self.assertNotIn("HELDOUTMARKER", prompt)

    def test_invalid_probes_fall_back_to_train_derived_probes(self):
        trainer = self._train()
        result = deep.request_probes(
            FakeProvider([json.dumps({"probes": []})]),
            trainer, limit=4, max_samples=3)
        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["error"])
        self.assertGreater(len(result["probes"]), 0)
        by_id = {item["record_id"]: item["text"] for item in trainer}
        for probe in result["probes"]:
            self.assertEqual(probe["origin"], "deterministic")
            self.assertIn(probe["prompt"], by_id[probe["source_record_id"]])
            self.assertIn(probe["expected"],
                          by_id[probe["source_record_id"]])

    def test_evaluate_probes_scores_expected_continuations(self):
        model = TemporalMirrorLM()
        for text in ("alpha beta gamma delta.", "one two three four."):
            model.observe(text)
        metrics = deep.evaluate_probes(model, [{
            "source_record_id": "t1",
            "prompt": "alpha beta",
            "expected": "gamma",
        }])
        self.assertEqual(metrics["probes"], 1)
        self.assertEqual(metrics["tokens_scored"], 1)
        self.assertGreater(metrics["mean_negative_log_likelihood"], 0.0)
        self.assertEqual(metrics["rows"][0]["source_record_id"], "t1")


class ReportTests(unittest.TestCase):
    def test_run_writes_json_and_markdown_with_safe_teacher_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainable = root / "trainable"
            _write_jsonl(trainable / "train.jsonl", [
                _record("t1", "alpha beta gamma delta epsilon."),
                _record("t2", "one two three four five."),
                _record("t3", "red green blue yellow orange."),
                _record("t4", "six seven eight nine ten."),
            ])
            _write_jsonl(trainable / "validation.jsonl", [
                _record("v1", "HELDOUTMARKER zeta eta theta.",
                        split="validation"),
                _record("v2", "HELDOUTMARKER iota kappa lambda.",
                        split="validation"),
            ])
            plan = json.dumps({
                "rationale": "small plan",
                "epochs": 2,
                "max_records_per_epoch": 0,
                "focus_stages": [],
                "notes": "",
            })
            probes = json.dumps({"probes": [{
                "source_record_id": "t1",
                "prompt": "alpha beta",
                "expected": "gamma",
            }]})
            provider = FakeProvider([plan, probes])
            state = root / "state"
            output = root / "results.json"

            def checkpoint(epoch):
                return {
                    "epoch": epoch,
                    "state_dir": str(state),
                    "model_path": str(state / "language" / "temporal_mirror.json"),
                    "model_present": True,
                    "model_bytes": 10,
                    "model_sha256": "%064x" % (epoch + 100),
                    "temporal_schema": deep.TEMPORAL_SCHEMA,
                    "register_path": str(state / "language" / "temporal_register.json"),
                    "register_sha256": "%064x" % (epoch + 200),
                    "chart_checkpoint_dir": str(
                        state / "language" / "chart-checkpoints"),
                    "chart_checkpoint_generation": epoch,
                }

            record = deep.run_deep_training(
                trainable, state, output,
                epochs=2,
                max_eval_records=1,
                gemma_provider=provider,
                gemma_plan=True,
                gemma_probes=True,
                gemma_url="http://fake.invalid/v1/chat/completions",
                gemma_model="fake-gemma",
                gemma_max_samples=4,
                checkpoint_fn=checkpoint,
            )

            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".md").exists())
            saved = json.loads(output.read_text(encoding="utf-8"))
            markdown = output.with_suffix(".md").read_text(encoding="utf-8")

        self.assertEqual(saved["schema"], deep.REPORT_SCHEMA)
        self.assertEqual(saved["epochs"]["requested"], 2)
        self.assertEqual(saved["epochs"]["source"], "cli")
        self.assertEqual(len(saved["epochs_run"]), 2)
        self.assertEqual(saved["split"]["eval_sample_records"], 1)
        self.assertEqual(saved["counts"]["train"], 4)
        self.assertEqual(saved["counts"]["validation"], 2)
        self.assertEqual(saved["counts"]["training_observed"], 8)
        self.assertEqual(saved["gemma"]["plan"]["status"], "applied")
        self.assertEqual(saved["gemma"]["probes"]["status"], "applied")
        self.assertEqual(saved["boundaries"]["canonical_merge"], False)
        self.assertEqual(saved["boundaries"]["proof_claim"], False)
        self.assertFalse(saved["gemma"]["heldout_text_sent"])
        self.assertEqual(saved["state"]["final_model_sha256"],
                         "%064x" % 102)
        all_prompts = json.dumps([call["messages"] for call in provider.calls])
        self.assertNotIn("HELDOUTMARKER", all_prompts)
        self.assertIn("Held-out next-token metrics", markdown)
        self.assertIn("Boundaries", markdown)
        self.assertIn("no canonical merge", markdown)
        self.assertIn("Gemma teacher/evaluator", markdown)


if __name__ == "__main__":
    unittest.main()
