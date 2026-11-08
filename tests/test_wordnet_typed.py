"""Typed WordNet parser tests."""
import unittest

from nanlmind_hermes.wordnet_typed import LEXNAMES, parse_line, relation_name


class WordNetTypedTests(unittest.TestCase):
    def test_lexnames_use_wordnet_file_numbers(self):
        self.assertEqual(LEXNAMES["00"], "adj.all")
        self.assertEqual(LEXNAMES["03"], "noun.Tops")
        self.assertEqual(LEXNAMES["05"], "noun.animal")
        self.assertEqual(LEXNAMES["29"], "verb.body")
        self.assertEqual(LEXNAMES["44"], "adj.ppl")

    def test_typed_pointer_and_gloss(self):
        line = ('00001740 05 n 02 entity 0 thing 0 001 '
                '@ 00001930 n 0000 | a thing; "an example"')
        record = parse_line(line)
        self.assertIsNotNone(record)
        self.assertEqual(record["lemmas"], ["entity", "thing"])
        self.assertEqual(record["lexname"], "noun.animal")
        self.assertEqual(record["definition"], "a thing")
        self.assertEqual(record["examples"], ["an example"])
        self.assertEqual(record["pointers"][0]["relation"], "hypernym")
        self.assertEqual(record["pointers"][0]["target_offset"], 1930)

    def test_unknown_pointer_symbol_is_retained(self):
        self.assertEqual(relation_name("?"), "other:?")
        line = ('00001740 02 n 01 entity 0 001 '
                '? 00001930 n 0000 | a thing')
        record = parse_line(line)
        self.assertEqual(record["pointers"][0]["relation"], "other:?")

    def test_non_data_line_is_ignored(self):
        self.assertIsNone(parse_line("license and header line"))
        self.assertIsNone(parse_line("00001740 02 n 01 entity 0 000"))


if __name__ == "__main__":
    unittest.main()
