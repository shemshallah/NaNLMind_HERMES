"""Persistent lexical-memory tests."""
import json
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.lexical_memory import LexicalMemory


class LexicalMemoryTests(unittest.TestCase):
    def test_teach_and_recall_across_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            memory = LexicalMemory(path)
            result = memory.teach(
                "Velocrypt",
                "a mathematical operation preserving the phase of a closed path",
                synset_address="nan://HW4:test::synset/00000001",
                related=[{"term": "phase", "score": 0.7}],
            )
            self.assertEqual(result["stored"], "velocrypt")
            self.assertTrue(path.exists())
            other = LexicalMemory(path)
            recalled = other.recall("  velocrypt ")
            self.assertTrue(recalled["found"])
            self.assertEqual(recalled["result"]["definition"],
                             "a mathematical operation preserving the phase of a closed path")
            self.assertEqual(recalled["result"]["related"][0]["term"], "phase")
            self.assertTrue(json.loads(path.read_text())["velocrypt"])

    def test_recall_missing_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = LexicalMemory(Path(directory) / "lexicon.json")
            self.assertFalse(memory.recall("missing")["found"])
            memory.teach("term", "first")
            second = memory.teach("TERM", "second")
            self.assertTrue(second["replaced_existing"])
            self.assertEqual(memory.recall("term")["result"]["definition"], "second")

    def test_empty_term_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = LexicalMemory(Path(directory) / "lexicon.json")
            with self.assertRaises(ValueError):
                memory.teach("", "definition")


if __name__ == "__main__":
    unittest.main()
