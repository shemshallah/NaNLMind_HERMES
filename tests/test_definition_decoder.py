"""Structured definition decoder tests."""
import unittest

from nanlmind_hermes.definition_decoder import DefinitionAnswerer


class DefinitionDecoderTests(unittest.TestCase):
    def setUp(self):
        self.synsets = {
            1: {"offset": 1, "pos": "n", "lemmas": ["entity"],
                "definition": "that which exists",
                "pointers": []},
            2: {"offset": 2, "pos": "n", "lemmas": ["animal"],
                "definition": "a living organism",
                "pointers": [{"relation": "hypernym", "target_offset": 1}]},
            3: {"offset": 3, "pos": "n", "lemmas": ["bird"],
                "definition": "a warm-blooded egg-laying vertebrate",
                "pointers": [{"relation": "hypernym", "target_offset": 2}]},
            4: {"offset": 4, "pos": "n", "lemmas": ["penguin"],
                "definition": "a flightless bird of cold southern regions",
                "pointers": [{"relation": "hypernym", "target_offset": 3}]},
        }
        self.index = {"entity": [1], "animal": [2], "bird": [3],
                      "penguin": [4]}
        self.answerer = DefinitionAnswerer(self.synsets, self.index)

    def test_definition_search_finds_penguin(self):
        results = self.answerer.search("flightless bird southern regions")
        self.assertEqual(results[0]["offset"], 4)

    def test_what_and_is_a_answers_are_structured(self):
        record = self.answerer.answer_what("penguin")
        self.assertEqual(record["definition"],
                         "a flightless bird of cold southern regions")
        answer = self.answerer.answer_is_a("penguin", "animal")
        self.assertTrue(answer["answer"])
        self.assertEqual(answer["path"], [4, 3, 2])

    def test_false_ancestry(self):
        answer = self.answerer.answer_is_a("bird", "penguin")
        self.assertFalse(answer["answer"])
        self.assertEqual(answer["path"], [])

    def test_typed_relation_readout(self):
        self.assertEqual(
            self.answerer.answer_relation("penguin", "hypernym")[0]["target_offset"], 3)


if __name__ == "__main__":
    unittest.main()
