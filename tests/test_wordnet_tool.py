import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from nanlmind_hermes import register
from nanlmind_hermes.layered_learning import reset_coordinator_cache


def build_archive(path):
    noun = "\n".join([
        "00001739 03 n 01 abstraction 0 0 0 0 0 0 | a general concept",
        "00001740 03 n 01 entity 0 0 1 @ 00001739 n 0000 | something that exists",
        "00001741 03 n 01 physical_entity 0 0 1 @ 00001740 n 0000 | an entity with physical form",
        "00001742 03 n 02 copper 0 metal 0 2 @ 00001741 n 0000 #m 00001741 n 0000 | a reddish metal",
    ]) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("wn3.1.dict/dict/data.noun", noun)
        archive.writestr("wn3.1.dict/dict/data.verb", "")
        archive.writestr("wn3.1.dict/dict/data.adj", "")
        archive.writestr("wn3.1.dict/dict/data.adv", "")


class FakeContext:
    def __init__(self, root, archive):
        self.config = {
            "state_dir": str(root),
            "wordnet_archive": str(archive),
            "chat_mode": "provider",
            "chat_learning": False,
        }
        self.tools = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs


class WordNetToolTests(unittest.TestCase):
    def setUp(self):
        reset_coordinator_cache()

    def tearDown(self):
        reset_coordinator_cache()

    def test_wordnet_ingest_reads_canonical_hw4_route_into_all_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            archive = Path(tmp) / "wn.zip"
            build_archive(archive)
            ctx = FakeContext(root, archive)
            register(ctx)
            result = json.loads(ctx.tools["nanlm_wordnet_ingest"]["handler"](
                {"word": "copper", "depth": 2}))
            self.assertEqual(result["wordnet"]["hw4_route"],
                             "nan://GM0:SC0:MC0:BS0:0:HW4:0")
            self.assertEqual(result["wordnet"]["hw4_legacy_route"],
                             "nan://HW4:BS0:MC0:SC0:GM0")
            self.assertEqual(result["wordnet"]["nannet_port"], "BS0:0")
            self.assertEqual(len(result["layers"]), 10)
            self.assertGreaterEqual(result["scratch"]["records"], 10)
            self.assertTrue(any(row["predicate"] == "defined_as"
                                for row in result["wordnet"]["relations"]))
            status = json.loads(ctx.tools["nanlm_learning_status"]["handler"]({}))
            self.assertEqual(status["dictionary_hw4"]["route"],
                             "nan://GM0:SC0:MC0:BS0:0:HW4:0")
            self.assertEqual(status["dictionary_hw4"]["legacy_route"],
                             "nan://HW4:BS0:MC0:SC0:GM0")

    def test_wordnet_depth_bounds_hypernym_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            archive = Path(tmp) / "wn.zip"
            build_archive(archive)
            ctx = FakeContext(root, archive)
            register(ctx)
            handler = ctx.tools["nanlm_wordnet_ingest"]["handler"]
            shallow = json.loads(handler({"word": "copper", "depth": 1}))
            deep = json.loads(handler({"word": "copper", "depth": 3}))
            self.assertEqual(shallow["wordnet"]["depth_used"], 1)
            self.assertEqual(deep["wordnet"]["depth_used"], 3)
            self.assertEqual([item["offset"] for item in shallow["wordnet"]["ancestry"]],
                             [1741])
            self.assertEqual([item["offset"] for item in deep["wordnet"]["ancestry"]],
                             [1741, 1740, 1739])
            shallow_relations = shallow["wordnet"]["relations"]
            deep_relations = deep["wordnet"]["relations"]
            self.assertTrue(any(row["predicate"] == "defined_as"
                                for row in shallow_relations))
            self.assertTrue(any(row["predicate"] == "kind_of"
                                for row in shallow_relations))
            self.assertGreater(len(deep_relations), len(shallow_relations))
            self.assertGreater(
                sum(row["predicate"] == "kind_of" for row in deep_relations),
                sum(row["predicate"] == "kind_of" for row in shallow_relations))

    def test_missing_wordnet_archive_fails_closed_instead_of_inventing_a_word(self):
        from nanlmind_hermes import wordnet_store
        original = wordnet_store.DEFAULT_CANDIDATES
        wordnet_store.DEFAULT_CANDIDATES = ()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "state"
                missing = Path(tmp) / "missing.zip"
                ctx = FakeContext(root, missing)
                register(ctx)
                with self.assertRaises(Exception) as raised:
                    ctx.tools["nanlm_wordnet_ingest"]["handler"](
                        {"word": "copper", "depth": 2})
                self.assertIn("HW4 dictionary universe is not loaded",
                              str(raised.exception))
        finally:
            wordnet_store.DEFAULT_CANDIDATES = original


if __name__ == "__main__":
    unittest.main()
