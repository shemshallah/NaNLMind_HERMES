import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nanlmind_hermes.tools import CURRICULUM_STAGES, generate_curriculum
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
            result = generate_curriculum(ctx, "English and mathematical reasoning", 4)
            info = load_n16txt(result["corpus_path"])
            self.assertEqual(len(CURRICULUM_STAGES), 10)
            self.assertEqual(len(ctx.llm.calls), 10)
            self.assertEqual(info.unique_records, 40)
            self.assertEqual(result["records"], 40)
            self.assertEqual([x["stage"] for x in result["stages"]],
                             [x[0] for x in CURRICULUM_STAGES])
            self.assertEqual(result["stages"][-1]["stage"], "narrative")
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(manifest["record_count"], 40)
            self.assertEqual(manifest["stages"][0]["provider"], "fake-provider")

    def test_curriculum_experiment_connects_generation_to_shared_trainer(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            class Context:
                def __init__(self):
                    self.llm = CurriculumLLM()
                def get_config(self, key, default=None):
                    return str(tmp) if key == "state_dir" else default
            ctx = Context()
            observed = {}
            def fake_train(context, corpus_path, epochs, max_samples_per_head):
                observed["corpus"] = load_n16txt(corpus_path)
                observed["epochs"] = epochs
                observed["samples"] = max_samples_per_head
                return {"status": "ok", "head_count": 256,
                        "merge": {"accepted": 256}, "metrics": {"heldout_accuracy": 0.61}}
            with patch("nanlmind_hermes.tools.math_train", side_effect=fake_train):
                from nanlmind_hermes.tools import run_curriculum_experiment
                result = run_curriculum_experiment(ctx, "logic and language", 2, 4, 128)
            self.assertEqual(observed["corpus"].unique_records, 20)
            self.assertEqual(observed["epochs"], 4)
            self.assertEqual(observed["samples"], 128)
            self.assertEqual(result["training"]["head_count"], 256)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            saved = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(saved["schema"], "nanlm-curriculum-experiment/1")

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
