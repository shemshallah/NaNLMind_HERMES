"""Dictionary universe route and inheritance tests."""
import unittest

from nanlmind_hermes.dictionary_universe import (
    HW4_LINK,
    HW4_ROUTE,
    DictionaryUniverse,
)


class DictionaryUniverseTests(unittest.TestCase):
    def setUp(self):
        synsets = {
            1: {"offset": 1, "lemmas": ["entity"], "hypernyms": []},
            2: {"offset": 2, "lemmas": ["animal"], "hypernyms": [1]},
            3: {"offset": 3, "lemmas": ["bird"], "hypernyms": [2]},
            4: {"offset": 4, "lemmas": ["penguin"], "hypernyms": [3]},
        }
        index = {"penguin": [4], "bird": [3], "animal": [2]}
        self.universe = DictionaryUniverse(synsets, index, {2: "living"})

    def test_inheritance_through_hypernyms(self):
        self.assertEqual(self.universe.inherit(4), ["living"])
        result = self.universe.query("penguin")
        self.assertEqual(result["synsets"][0]["inherited"], ["living"])

    def test_hw4_addresses_and_bidirectional_link(self):
        address = self.universe.synset_address(4)
        self.assertTrue(address.startswith(HW4_ROUTE + "::synset/"))
        word = self.universe.word_address("penguin", 4)
        self.assertIn("::word/penguin/", word)
        relation = self.universe.relation_address(4, 2, "hypernym")
        self.assertTrue(relation.startswith(HW4_ROUTE + "::relation/"))
        interface = self.universe.interface()
        self.assertEqual(interface["link"], HW4_LINK)
        self.assertEqual(interface["upstream"], "HW2:0")
        self.assertTrue(interface["bidirectional"])


if __name__ == "__main__":
    unittest.main()
