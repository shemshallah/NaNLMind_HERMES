"""Conservative definition-acquisition tests."""
import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.semantic_acquisition import (
    SemanticAcquisition,
    possible_heads,
)


class SemanticAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = {
            1: {"offset": 1, "lemmas": ["entity"], "definition": "that which exists",
                "pointers": []},
            2: {"offset": 2, "lemmas": ["plant"], "definition": "a living organism",
                "pointers": [{"relation": "hypernym", "target_offset": 1}]},
            3: {"offset": 3, "lemmas": ["vascular plant"],
                "definition": "a plant with vascular tissue",
                "pointers": [{"relation": "hypernym", "target_offset": 2}]},
            4: {"offset": 4, "lemmas": ["saxifrage"],
                "definition": "a plant of the saxifrage family",
                "pointers": [{"relation": "hypernym", "target_offset": 3}]},
            5: {"offset": 5, "lemmas": ["bank"], "definition": "a financial institution",
                "pointers": []},
            6: {"offset": 6, "lemmas": ["bank"], "definition": "sloping land",
                "pointers": []},
        }
        self.index = {lemma: [] for record in self.nodes.values()
                      for lemma in record["lemmas"]}
        for offset, record in self.nodes.items():
            for lemma in record["lemmas"]:
                self.index[lemma].append(offset)

    def test_conservative_parser(self):
        self.assertEqual(possible_heads(
            "a kind of saxifrage that has fluorescent petals"), ["saxifrage"])
        self.assertEqual(possible_heads(
            "a type of vascular plant with purple flowers"), ["vascular plant"])
        self.assertEqual(possible_heads("an animal that runs"), [])

    def test_teach_and_two_hop_inference(self):
        engine = SemanticAcquisition(self.nodes, self.index)
        result = engine.teach(
            "axolume", "a kind of saxifrage that has fluorescent petals")
        self.assertTrue(result["resolved"])
        answer = engine.ask("axolume", "vascular plant")
        self.assertEqual(answer["answer"], "yes")
        self.assertEqual(answer["distance"], 2)

    def test_unknown_category_abstains(self):
        engine = SemanticAcquisition(self.nodes, self.index)
        engine.teach("axolume", "a kind of saxifrage that has fluorescent petals")
        answer = engine.ask("axolume", "animal")
        self.assertEqual(answer["answer"], "unknown")

    def test_ambiguous_head_is_not_resolved(self):
        engine = SemanticAcquisition(self.nodes, self.index)
        result = engine.teach("bankling", "a kind of bank that is unusual")
        self.assertFalse(result["resolved"])
        self.assertTrue(result["requires_disambiguation"])
        self.assertEqual(engine.ask("bankling", "entity")["answer"], "unknown")

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concepts.json"
            first = SemanticAcquisition(self.nodes, self.index, path)
            first.teach("axolume",
                        "a kind of saxifrage that has fluorescent petals")
            second = SemanticAcquisition(self.nodes, self.index, path)
            self.assertEqual(second.ask("axolume", "plant")["answer"], "yes")
            self.assertIn("axolume", second.explain("axolume"))


if __name__ == "__main__":
    unittest.main()
