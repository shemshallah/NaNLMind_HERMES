import json
import sys
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.runner import evaluate_native, run_bounded, run_math_train


class NativeHarnessTests(unittest.TestCase):
    def _fake_cli(self, root):
        script = root / "fake_nanlm.py"
        script.write_text(
            "import pathlib, sys\n"
            "mode = sys.argv[1]\n"
            "if mode == 'bootstrap':\n"
            " print('bootstrap corpus=' + sys.argv[2])\n"
            " print('epoch=1 chunk_end=10 bytes_seen=10 slabs=1 bpc=2.5 top1=0.42 alpha_top1=0.5 word_start_top1=0.6')\n"
            "elif mode == 'byte':\n"
            " print('topology acc=0.4 bpc=3.1')\n"
            " print('reference acc=0.4 bpc=3.1')\n"
            " print('order0 acc=0.2 bpc=4.2')\n"
            "elif mode == 'kn':\n"
            " print('top1=0.55 bpc=2.7')\n"
            "elif mode == 'ctx2':\n"
            " print('top1=0.5 bpc=2.9')\n"
            "else: raise SystemExit(3)\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def test_native_bootstrap_creates_reproducibility_artifact_not_fake_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "sample.n16txt"
            corpus.write_text("N16TXT/1\nhello\nworld\n", encoding="utf-8")
            result = run_bounded(self._fake_cli(root), corpus, root / "state", 1, 5)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["native_subcommand"], "bootstrap")
            self.assertAlmostEqual(result["training_metrics"]["bpc"], 2.5)
            self.assertFalse(result["checkpoint_present"])
            self.assertFalse(result["resume_supported"])
            manifest = json.loads((Path(result["artifact_dir"]) / "manifest.json").read_text())
            self.assertEqual(manifest["corpus_sha256"], result["corpus_sha256"])
            self.assertTrue(Path(result["training_log"]).exists())

    def test_native_evaluation_runs_topology_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "sample.txt"
            corpus.write_text("hello world " * 20, encoding="utf-8")
            result = evaluate_native(self._fake_cli(root), corpus, root / "state")
            evaluations = result["evaluations"]
            self.assertEqual(set(evaluations), {"topology_byte", "kneser_ney_order2", "context2"})
            self.assertAlmostEqual(evaluations["topology_byte"]["metrics"]["bpc"], 3.1)
            self.assertAlmostEqual(evaluations["kneser_ney_order2"]["metrics"]["top1"], 0.55)

    def test_empty_runner_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "sample.txt"
            corpus.write_text("x", encoding="utf-8")
            from nanlmind_hermes.runner import RunnerError
            with self.assertRaises(RunnerError):
                run_bounded([], corpus, root / "state", 1)


    def _fake_math_cli(self, root):
        script = root / "fake_math_cli.py"
        script.write_text(
            "import hashlib, json, pathlib, sys\n"
            "mode, corpus, epochs, samples, state, trace = sys.argv[1:]\n"
            "data = pathlib.Path(corpus).read_bytes()\n"
            "digest = hashlib.sha256(data).hexdigest()\n"
            "sp = pathlib.Path(state); sp.parent.mkdir(parents=True, exist_ok=True)\n"
            "prior = sp.read_bytes() if sp.exists() else b''\n"
            "sp.write_bytes(prior + b'x')\n"
            "tp = pathlib.Path(trace); tp.parent.mkdir(parents=True, exist_ok=True)\n"
            "with tp.open('w', encoding='utf-8') as out:\n"
            " for h in range(256):\n"
            "  event = {'schema':'nanlm-hermes-discovery/1','kind':'hypothesis',"
            "'subject':'head-'+str(h),'head':h,'layer':0,'source_digest':digest,'hyperedge_id':h+1,'hyperedge_label':h,"
            "'expression':{'op':'truth_table','head':h},'confidence':0.5,"
            "'polarity':'neutral','evidence':{'test_samples':10},'provenance':{}}\n"
            "  out.write(json.dumps(event)+'\\n')\n"
            "print('trained_heads=256 resumed=' + str(bool(prior)) + ' heldout_accuracy=0.61 majority_baseline_accuracy=0.50')\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def test_math_train_exports_all_heads_and_resumes_global_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.bin"
            corpus.write_bytes(b"mathematical universe " * 100)
            command = self._fake_math_cli(root)
            first = run_math_train(command, corpus, root / "universe", epochs=2, max_samples_per_head=32)
            self.assertEqual(first["head_count"], 256)
            self.assertEqual(first["discoveries"], 256)
            self.assertAlmostEqual(first["metrics"]["heldout_accuracy"], 0.61)
            self.assertAlmostEqual(first["metrics"]["majority_baseline_accuracy"], 0.50)
            state = Path(first["native_state_path"])
            self.assertEqual(state.read_bytes(), b"x")
            second = run_math_train(command, corpus, root / "universe", epochs=1, max_samples_per_head=16)
            self.assertEqual(second["head_count"], 256)
            self.assertEqual(state.read_bytes(), b"xx")
            self.assertEqual(second["corpus_sha256"], first["corpus_sha256"])

    def test_math_train_rejects_incomplete_head_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.bin"
            corpus.write_bytes(b"mathematical universe " * 100)
            script = root / "partial.py"
            script.write_text(
                "import hashlib, json, pathlib, sys\n"
                "_, corpus, epochs, samples, state, trace = sys.argv[1:]\n"
                "data=pathlib.Path(corpus).read_bytes(); digest=hashlib.sha256(data).hexdigest()\n"
                "pathlib.Path(state).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(state).write_bytes(b'x')\n"
                "pathlib.Path(trace).write_text(json.dumps({'schema':'nanlm-hermes-discovery/1',"
                "'kind':'hypothesis','subject':'only-one','head':0,'layer':0,"
                "'source_digest':digest,'confidence':0.5,'polarity':'neutral'})+'\\n')\n",
                encoding="utf-8",
            )
            from nanlmind_hermes.runner import RunnerError
            with self.assertRaises(RunnerError):
                run_math_train([sys.executable, str(script)], corpus, root / "universe")


if __name__ == "__main__":
    unittest.main()
