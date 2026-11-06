import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.tools import CURRICULUM_STAGES, generate_curriculum, generate_relation_curriculum
from nanlmind_hermes.policy import load_n16txt


class CurriculumLLM:
    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        purpose = kwargs["purpose"]
        stage = purpose.removeprefix("nanlmind-curriculum-")
        return SimpleNamespace(
            text="\n".join(f"{stage} example {i}: {stage} context and evidence." for i in range(4)),
            provider="fake-provider", model="fake-curriculum",
        )


class CurriculumTests(unittest.TestCase):
    def test_balanced_curriculum_covers_logic_through_narrative(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def __init__(self):
                    self.llm = CurriculumLLM()
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default

            ctx = Context()
            import hashlib
            from nanlmind_hermes.tools import _memory_store
            _memory_store(ctx).ingest(
                {"kind": "hypothesis", "subject": "seeded-universe-claim",
                 "provenance": {"resolution": "bit"}},
                source_digest=hashlib.sha256(b"seed-memory").hexdigest(),
            )
            _memory_store(ctx).reorganize()
            result = generate_curriculum(ctx, "English and mathematical reasoning", 4)
            self.assertIn("seeded-universe-claim", ctx.llm.calls[0]["messages"][1]["content"])
            info = load_n16txt(result["corpus_path"])
            self.assertEqual(len(CURRICULUM_STAGES), 12)
            self.assertEqual(len(ctx.llm.calls), 12)
            self.assertEqual(info.unique_records, 4 * len(CURRICULUM_STAGES))
            self.assertEqual(result["records"], 4 * len(CURRICULUM_STAGES))
            self.assertEqual([x["stage"] for x in result["stages"]],
                             [x[0] for x in CURRICULUM_STAGES])
            self.assertEqual(result["stages"][2]["stage"], "story_saturation")
            self.assertEqual(result["stages"][-1]["stage"], "narrative")
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(manifest["record_count"], 4 * len(CURRICULUM_STAGES))
            self.assertEqual(manifest["stages"][0]["provider"], "fake-provider")

    def test_curriculum_experiment_connects_structured_generation_to_native_relation_trainer(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default
            ctx = Context()
            observed = {}
            corpus = {
                "schema": "nanlm-structured-relation-curriculum/1",
                "topic_key": "topic-key", "resolution_order": ["lexical"],
                "frozen_challenge_sets": 1, "stages": [{"resolution": "lexical"}],
                "records": [
                    {"split": "train", "source_digest": "a" * 64,
                     "item": {"kind": "relation", "subject": "penguin", "predicate": "is_a",
                              "object": "bird", "polarity": "supports", "confidence": 1.0,
                              "provenance": {"resolution": "lexical"}}},
                    {"split": "test", "source_digest": "b" * 64,
                     "item": {"kind": "relation", "subject": "ostrich", "predicate": "is_a",
                              "object": "bird", "polarity": "supports", "confidence": 1.0,
                              "provenance": {"resolution": "lexical"}}},
                ],
            }
            def fake_train(context, records, mode, epochs):
                observed["records"] = records
                observed["mode"] = mode
                observed["epochs"] = epochs
                return {"status": "updated", "resolution_metrics": {"5": {"relation_polarity_accuracy": 0.75}}}
            with patch("nanlmind_hermes.tools.generate_relation_curriculum", return_value=corpus), \
                 patch("nanlmind_hermes.tools._native_relation_train", side_effect=fake_train):
                from nanlmind_hermes.tools import run_curriculum_experiment
                result = run_curriculum_experiment(ctx, "logic and language", 2, 4, 128, ["lexical_words"])
            self.assertEqual(len(observed["records"]), 2)
            self.assertEqual(observed["mode"], "structured_teacher_training")
            self.assertEqual(observed["epochs"], 4)
            self.assertEqual(result["training"]["resolution_metrics"]["5"]["relation_polarity_accuracy"], 0.75)
            self.assertEqual(result["legacy_max_samples_per_head_ignored"], 128)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            saved = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(saved["schema"], "nanlm-curriculum-experiment/2")

    def test_relation_curriculum_freezes_a_separate_heldout_set_per_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            class RelationTeacher:
                def __init__(self):
                    self.round = 0
                def complete(self, **kwargs):
                    self.round += 1
                    records = []
                    for i in range(6):
                        records.append({
                            "subject": f"entity-{self.round}-{i}",
                            "predicate": "has_relation",
                            "object": f"object-{i}",
                            "polarity": "supports" if i % 2 == 0 else "refutes",
                            "confidence": 0.9,
                            "evidence": {"source": "fixture", "index": i},
                        })
                    return SimpleNamespace(text=json.dumps({"records": records}),
                                            provider="fixture", model="relation-teacher")
            class Context:
                def __init__(self):
                    self.llm = RelationTeacher()
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default
            ctx = Context()
            first = generate_relation_curriculum(ctx, "penguin relations", 6, ["lexical_words"])
            second = generate_relation_curriculum(ctx, "penguin relations", 6, ["lexical_words"])
            first_stage, second_stage = first["stages"][0], second["stages"][0]
            self.assertEqual(first_stage["challenge_sha256"], second_stage["challenge_sha256"])
            self.assertEqual(first_stage["frozen_test"], second_stage["frozen_test"])
            self.assertEqual(len([x for x in first["records"] if x["split"] == "test"]), 2)
            self.assertEqual(len([x for x in second["records"] if x["split"] == "test"]), 2)
            self.assertEqual(first["resolution_order"], ["lexical"])
            self.assertTrue(all(x["item"]["provenance"]["independently_fact_verified"] is False
                                for x in second["records"]))

    def test_shared_memory_context_bounds_large_expressions(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default
            ctx = Context()
            from nanlmind_hermes.tools import _memory_store, _memory_context
            store = _memory_store(ctx)
            store.ingest(
                {"kind": "hypothesis", "subject": "large-expression",
                 "expression": {"blob": "z" * 5000},
                 "provenance": {"resolution": "bit"}},
                source_digest=hashlib.sha256(b"large").hexdigest(),
            )
            store.reorganize()
            context = _memory_context(ctx, layer=0)
            expression = context["claims"][0]["expression"]
            self.assertTrue(expression["truncated"])
            self.assertEqual(len(expression["preview"]), 1200)

    def test_curriculum_rejects_unsafe_batch_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default
            for count in (0, 65, True):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    generate_curriculum(Context(), "logic", count)


if __name__ == "__main__":
    unittest.main()
