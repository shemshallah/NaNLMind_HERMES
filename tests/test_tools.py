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
    def test_native_discoveries_and_literal_relations_share_durable_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake_native.py"
            fake.write_text(
                "import hashlib, json, pathlib, sys\n"
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
            memory = status(ctx)["mathematical_memory"]
            self.assertEqual(memory["events"], 258)
            self.assertTrue(status(ctx)["native_checkpoint_present"])


if __name__ == "__main__":
    unittest.main()
