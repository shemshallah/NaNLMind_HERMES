import json
import sys
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.runner import run_bounded


class RunnerTests(unittest.TestCase):
    def test_runner_receives_train_mode_and_checkpoint_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "sample.n16txt"
            corpus.write_text("N16TXT/1\nhello\n", encoding="utf-8")
            script = root / "fake_runner.py"
            script.write_text(
                "import pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "assert args[0] == 'train'\n"
                "checkpoint = pathlib.Path(args[args.index('--checkpoint') + 1])\n"
                "checkpoint.write_text('checkpoint')\n",
                encoding="utf-8",
            )
            result = run_bounded(
                [sys.executable, str(script)], corpus, root / "state", 1, 5
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["checkpoint_present"])
            saved = json.loads((root / "state" / "last_run.json").read_text())
            self.assertEqual(saved["duration_budget_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
