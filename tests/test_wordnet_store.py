import tempfile
import unittest
import zipfile
from pathlib import Path

from nanlmind_hermes.wordnet_store import WordNetStore, WordNetStoreError


def build_archive(path):
    noun = "\n".join([
        "00001740 03 n 01 entity 0 0 0 0 0 0 | something that exists",
        "00001741 03 n 01 physical_entity 0 0 1 @ 00001740 n 0000 | an entity with physical form",
        "00001742 03 n 02 copper 0 metal 0 2 @ 00001741 n 0000 #m 00001741 n 0000 | a reddish metal",
    ]) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("wn3.1.dict/dict/data.noun", noun)
        archive.writestr("wn3.1.dict/dict/data.verb", "")
        archive.writestr("wn3.1.dict/dict/data.adj", "")
        archive.writestr("wn3.1.dict/dict/data.adv", "")


class WordNetStoreTests(unittest.TestCase):
    def test_streams_sparse_graph_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wn.zip"
            build_archive(path)
            store = WordNetStore.from_archive(path)
        self.assertEqual(len(store.synsets), 3)
        self.assertIn("copper", store.index)
        self.assertIn(1742, store.index["copper"])
        self.assertEqual(store.stats()["positions"], {"n": 3})
        definition = store.universe.define(1742)
        self.assertEqual(definition["lemmas"][0], "copper")
        self.assertTrue(definition["hypernyms"])
        self.assertEqual(definition["hypernyms"][0]["target_offset"], 1741)
        self.assertEqual(store.universe.query("copper")["synsets"][0]["offset"], 1742)

    def test_max_synsets_and_missing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wn.zip"
            build_archive(path)
            store = WordNetStore.from_archive(path, max_synsets=2)
        self.assertEqual(len(store.synsets), 2)
        with self.assertRaises(WordNetStoreError):
            WordNetStore.from_archive("/nonexistent/wn.zip")
        with self.assertRaises(WordNetStoreError):
            WordNetStore.from_synsets([{"lemmas": []}])


if __name__ == "__main__":
    unittest.main()
