import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.dictionary_universe import DictionaryUniverse
from nanlmind_hermes.layered_learning import (
    HEAD_COUNT,
    LayeredLearningCoordinator,
    LayeredLearningError,
    LearningPolicy,
)


def dictionary():
    synsets = {
        100: {
            "offset": 100, "pos": "n", "lemmas": ["copper"], "gloss": "a metal",
            "definition": "a reddish metal that conducts electricity",
            "pointers": [{"relation": "hypernym", "target_offset": 200,
                          "target_pos": "n"}],
            "hypernyms": [200],
        },
        200: {
            "offset": 200, "pos": "n", "lemmas": ["metal"], "gloss": "an element",
            "definition": "a solid material that conducts heat and electricity",
            "pointers": [], "hypernyms": [],
        },
    }
    return DictionaryUniverse(synsets, {"copper": [100], "metal": [200]})


class LayeredLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "state"

    def make(self, policy=None, with_dictionary=False):
        return LayeredLearningCoordinator(
            self.root,
            dictionary=dictionary() if with_dictionary else None,
            policy=policy or LearningPolicy(
                min_support=1, min_confidence=0.1,
                require_explicit_approval=False, auto_sync=False,
            ),
        )

    def test_observe_projects_all_ten_layers_and_255_heads(self):
        coordinator = self.make()
        result = coordinator.observe(
            "Copper is a metal because it conducts electricity.",
            relations=[{"subject": "copper", "predicate": "is", "object": "metal"}],
        )
        self.assertEqual(len(result["layers"]), 10)
        self.assertEqual([item["layer"] for item in result["layers"]],
                         ["bit", "nibble", "byte", "phonetic", "morpheme",
                          "lexical", "word_completion", "sentence", "paragraph",
                          "narrative"])
        self.assertEqual(result["scratch"]["logical_bytes"], 1024 ** 3)
        self.assertGreaterEqual(result["scratch"]["records"], 10)
        self.assertGreater(result["head_balance"]["active_heads"], 0)
        self.assertLessEqual(result["head_balance"]["active_heads"], HEAD_COUNT)
        self.assertGreater(result["concept_saturation"]["concepts"], 0)

    def test_sync_requires_approval_when_policy_says_so(self):
        coordinator = LayeredLearningCoordinator(
            self.root,
            policy=LearningPolicy(min_support=1, min_confidence=0.1,
                                  require_explicit_approval=True),
        )
        coordinator.observe("Copper is a metal.",
                            relations=[{"subject": "copper", "predicate": "is",
                                        "object": "metal"}])
        blocked = coordinator.synchronize()
        self.assertEqual(blocked["status"], "approval_required")
        promoted = coordinator.synchronize(approved=True)
        self.assertEqual(promoted["status"], "promoted")
        self.assertGreaterEqual(promoted["head_register"]["leaf_count"], 1)

    def test_repeated_relation_promotes_into_head_register(self):
        coordinator = self.make()
        for _ in range(3):
            coordinator.observe("Copper is a metal.",
                                relations=[{"subject": "copper", "predicate": "is",
                                            "object": "metal"}])
        report = coordinator.synchronize(approved=True)
        self.assertEqual(report["status"], "promoted")
        self.assertEqual(report["head_register"]["leaf_count"], 1)
        self.assertEqual(report["head_register"]["layers"], ["lexical"])
        status = coordinator.status()
        self.assertEqual(status["promoted"], 1)
        self.assertTrue(status["scratch_verify"]["ok"])

    def test_wordnet_ingestion_uses_hw4_route(self):
        coordinator = self.make(with_dictionary=True)
        result = coordinator.ingest_word("copper", depth=2)
        self.assertEqual(result["wordnet"]["hw4_route"],
                         "nan://HW4:BS0:MC0:SC0:GM0")
        self.assertGreaterEqual(len(result["wordnet"]["relations"]), 2)
        predicates = {item["predicate"] for item in result["wordnet"]["relations"]}
        self.assertIn("defined_as", predicates)
        self.assertIn("kind_of", predicates)
        records = list(coordinator.space.records(layer="lexical"))
        self.assertTrue(any(record["payload"].get("relation")
                            for record in records))

    def test_state_survives_reopen(self):
        first = self.make()
        first.observe("Copper is a metal.",
                      relations=[{"subject": "copper", "predicate": "is",
                                  "object": "metal"}])
        first.close()
        second = self.make()
        self.assertEqual(second.turns, 1)
        self.assertEqual(len(second.rows), 1)
        self.assertGreater(second.status()["head_ensemble"]["active_heads"], 0)

    def test_rejects_bad_text_layer_and_policy(self):
        coordinator = self.make()
        with self.assertRaises(LayeredLearningError):
            coordinator.observe("")
        with self.assertRaises(LayeredLearningError):
            LearningPolicy(min_support=0)
        with self.assertRaises(LayeredLearningError):
            LearningPolicy(max_scratch_bytes=1)


if __name__ == "__main__":
    unittest.main()
