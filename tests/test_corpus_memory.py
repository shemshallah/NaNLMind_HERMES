"""Corpus ingestion and retrieval tests."""
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.corpus_memory import CorpusStore


class CorpusMemoryTests(unittest.TestCase):
    def test_ingest_deduplicate_split_and_retrieve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text(
                "A bird has feathers and wings.\n\nA bird has feathers and wings.\n\n"
                "A fish lives in water.",
                encoding="utf-8",
            )
            (root / "b.txt").write_text(
                "A plant grows in soil.",
                encoding="utf-8",
            )
            store = CorpusStore()
            stats = store.ingest([root])
            self.assertEqual(stats["documents"], 3)
            self.assertEqual(sum(stats["splits"].values()), 3)
            results = store.retrieve("bird feathers", limit=2)
            self.assertEqual(results[0]["text"], "A bird has feathers and wings.")
            self.assertIn("source", results[0])
            self.assertTrue(results[0]["sha256"])

    def test_manifest_roundtrip_and_split_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("A device measures pressure.", encoding="utf-8")
            store = CorpusStore()
            store.ingest([root])
            path = root / "manifest.json"
            store.save(path)
            loaded = CorpusStore()
            loaded.load(path)
            self.assertEqual(loaded.stats(), store.stats())
            self.assertEqual(len(loaded.retrieve("pressure", split="train")), 1)
            self.assertEqual(len(loaded.retrieve("pressure", split="test")), 0)

    def test_deterministic_hashes_and_chunk_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one two three four five six", encoding="utf-8")
            first = CorpusStore()
            second = CorpusStore()
            first.ingest([root], max_tokens=2)
            second.ingest([root], max_tokens=2)
            self.assertEqual(len(first.documents), 3)
            self.assertEqual([doc["sha256"] for doc in first.documents],
                             [doc["sha256"] for doc in second.documents])


if __name__ == "__main__":
    unittest.main()
