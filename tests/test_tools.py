import json
import sys
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.tools import math_train, status


class FakeContext:
    def __init__(self, config):
        self.config = config

    def get_config(self, key, default=None):
        return self.config.get(key, default)


class HermesMathTrainTests(unittest.TestCase):

    def test_relation_corpus_preserves_resolution_polarity_and_train_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = FakeContext({"state_dir": str(Path(tmp) / "state")})
            from nanlmind_hermes.tools import _relation_corpus
            records = [
                {"split": "test", "source_digest": "a" * 64,
                 "item": {"subject": "penguin", "predicate": "can", "object": "fly",
                          "polarity": "refutes", "confidence": 0.8,
                          "provenance": {"resolution": "lexical"}}},
                {"split": "train", "source_digest": "b" * 64,
                 "item": {"subject": "penguin", "predicate": "is_a", "object": "bird",
                          "polarity": "supports", "confidence": 1.0,
                          "provenance": {"resolution": "lexical"}}},
                {"split": "train", "source_digest": "c" * 64,
                 "item": {"subject": "river", "predicate": "flows_through", "object": "valley",
                          "polarity": "supports", "confidence": 0.9,
                          "provenance": {"resolution": "narrative"}}},
            ]
            result = _relation_corpus(ctx, records)
            lines = Path(result["path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "N16REL/1")
            self.assertTrue(lines[1].startswith("train\\t"))
            self.assertTrue(lines[2].startswith("train\\t"))
            self.assertTrue(lines[3].startswith("test\\t"))
            self.assertEqual(result["counts"]["train"]["lexical"], 1)
            self.assertEqual(result["counts"]["test"]["lexical"], 1)
            self.assertEqual(result["counts"]["train"]["narrative"], 1)

    def test_native_discoveries_and_literal_relations_share_durable_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_native.py"
            fake.write_text(
                "import hashlib, json, pathlib, sys\n"
                "mode = sys.argv[1]\n"
                "if mode == 'logic-train':\n"
                " state = pathlib.Path(sys.argv[4]); state.parent.mkdir(parents=True, exist_ok=True); state.write_bytes(b'boolean-foundation')\n"
                " print('logic_train boolean_reps=32 circuit_reps=64 state_path=' + str(state))\n"
                " print('stage=boolean_foundation passed=yes correct=1024/1024 accuracy=1 detail=256 heads x 4 truth-table rows | distinct_rules=16 | failed_heads=0')\n"
                " print('stage=deep_logic_circuits passed=yes correct=32/32 accuracy=1 detail=composed gates')\n"
                " print('stage=words_after_logic passed=yes correct=10/10 accuracy=1 detail=word probes')\n"
                " print('state_bytes=19 state_digest=' + '0' * 64)\n"
                " print('result=PASS')\n"
                " raise SystemExit(0)\n"
                "_, corpus, epochs, samples, state, trace = sys.argv[1:]\n"
                "data=pathlib.Path(corpus).read_bytes(); digest=hashlib.sha256(data).hexdigest()\n"
                "sp=pathlib.Path(state); sp.parent.mkdir(parents=True,exist_ok=True)\n"
                "sp.write_bytes((sp.read_bytes() if sp.exists() else b'')+b'x')\n"
                "tp=pathlib.Path(trace); tp.parent.mkdir(parents=True,exist_ok=True)\n"
                "with tp.open('w') as out:\n"
                " for h in range(256):\n"
                "  e={'schema':'nanlm-hermes-discovery/1','kind':'hypothesis','subject':'native-'+str(h),"
                "'head':h,'layer':0,'source_digest':digest,'hyperedge_id':h+1,'hyperedge_label':h,"
                "'expression':{'op':'truth_table','head':h},'confidence':0.6,'polarity':'neutral',"
                "'evidence':{'test_samples':4},'provenance':{}}\n"
                "  out.write(json.dumps(e)+'\\n')\n"
                "print('trained_heads=256 heldout_accuracy=0.6 majority_baseline_accuracy=0.5')\n",
                encoding="utf-8",
            )
            lines = ["penguin can fly.", "penguin cannot fly."]
            lines.extend(f"neutral corpus filler record {i} " + ("context " * 8) for i in range(30))
            corpus = root / "corpus.n16txt"
            corpus.write_text("N16TXT/1\n" + "\n".join(lines) + "\n", encoding="utf-8")
            ctx = FakeContext({
                "state_dir": str(root / "state"),
                "runner_command": [sys.executable, str(fake)],
            })
            result = math_train(ctx, str(corpus), epochs=2, max_samples_per_head=32)
            self.assertEqual(result["discoveries"], 256)
            self.assertEqual(result["merge"]["accepted"], 256)
            self.assertEqual(result["surface_extraction"]["discoveries"], 2)
            self.assertEqual(result["surface_extraction"]["merge"]["accepted"], 2)
            self.assertEqual(result["reorganization"]["contested_subjects"], 1)
            self.assertEqual(result["boolean_foundation"]["status"], "passed")
            self.assertEqual(result["boolean_foundation"]["stages"][0]["correct"], 1024)
            self.assertEqual(result["boolean_foundation"]["stages"][0]["total"], 1024)
            self.assertTrue(Path(result["boolean_foundation"]["state_path"]).is_file())
            memory = status(ctx)["mathematical_memory"]
            self.assertEqual(memory["events"], 258)
            self.assertTrue(status(ctx)["native_checkpoint_present"])


if __name__ == "__main__":
    unittest.main()
