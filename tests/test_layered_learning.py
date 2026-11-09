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
            "pointers": [{"relation": "hypernym", "target_offset": 300,
                          "target_pos": "n"}],
            "hypernyms": [300],
        },
        300: {
            "offset": 300, "pos": "n", "lemmas": ["material"],
            "definition": "the substance from which things are made",
            "pointers": [{"relation": "hypernym", "target_offset": 400,
                          "target_pos": "n"}],
            "hypernyms": [400],
        },
        400: {
            "offset": 400, "pos": "n", "lemmas": ["entity"],
            "definition": "something that exists",
            "pointers": [], "hypernyms": [],
        },
    }
    properties = {200: "metal", 300: "material", 400: "entity"}
    return DictionaryUniverse(
        synsets,
        {"copper": [100], "metal": [200], "material": [300], "entity": [400]},
        properties,
    )


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

    def test_relation_polarity_and_confidence_are_preserved(self):
        coordinator = self.make()
        result = coordinator.observe(
            "Copper is not a metal.",
            relations=[{
                "subject": "Copper", "predicate": "is", "object": "Metal",
                "polarity": "refutes", "confidence": 0.8,
            }],
        )
        row = result["relations"][0]
        self.assertEqual(row["subject"], "copper")
        self.assertEqual(row["object"], "metal")
        self.assertEqual(row["polarity"], "refutes")
        self.assertAlmostEqual(row["confidence"], 0.8)
        candidates = coordinator._candidate_rows()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["polarity"], "refutes")
        self.assertAlmostEqual(candidates[0]["confidence"], 0.8)

        defaulted = coordinator.observe(
            "Copper is a metal.",
            relations=[{"subject": "copper", "predicate": "is", "object": "metal"}],
        )["relations"][0]
        self.assertEqual(defaulted["polarity"], "neutral")
        self.assertAlmostEqual(defaulted["confidence"], 0.5)

        with self.assertRaises(LayeredLearningError):
            coordinator.observe("Copper maybe is a metal.", relations=[{
                "subject": "copper", "predicate": "is", "object": "metal",
                "polarity": "maybe",
            }])
        with self.assertRaises(LayeredLearningError):
            coordinator.observe("Copper is a metal.", relations=[{
                "subject": "copper", "predicate": "is", "object": "metal",
                "confidence": 1.5,
            }])

        coordinator.close()
        reopened = self.make()
        persisted = next(
            row for row in reopened.rows
            if row["subject"] == "copper" and row["polarity"] == "refutes")
        self.assertAlmostEqual(persisted["confidence"], 0.8)
        self.assertEqual(len(reopened._candidate_rows()), 2)

    def test_supported_and_refuted_triples_do_not_merge(self):
        coordinator = self.make(policy=LearningPolicy(
            min_support=2, min_confidence=0.1,
            require_explicit_approval=False, auto_sync=False,
        ))
        supported = {"subject": "copper", "predicate": "is", "object": "metal",
                     "polarity": "supports", "confidence": 0.9}
        refuted = {"subject": "copper", "predicate": "is", "object": "metal",
                   "polarity": "refutes", "confidence": 0.9}
        for _ in range(2):
            coordinator.observe("Copper is / is not a metal.",
                                relations=[supported, refuted])
        candidates = coordinator._candidate_rows()
        self.assertEqual(len(candidates), 2)
        self.assertEqual({row["polarity"] for row in candidates},
                         {"supports", "refutes"})
        self.assertTrue(all(row["support"] == 2 for row in candidates))
        report = coordinator.synchronize(approved=True)
        self.assertEqual(report["candidates"], 2)
        self.assertEqual(report["promoted"], 2)
        self.assertEqual(report["head_register"]["leaf_count"], 2)

    def test_candidate_grouping_includes_confidence(self):
        coordinator = self.make()
        for confidence in (0.9, 0.6):
            coordinator.observe("Copper is a metal.", relations=[{
                "subject": "copper", "predicate": "is", "object": "metal",
                "polarity": "supports", "confidence": confidence,
            }])
        candidates = coordinator._candidate_rows()
        self.assertEqual(len(candidates), 2)
        self.assertEqual({row["confidence"] for row in candidates}, {0.9, 0.6})
        self.assertTrue(all(row["polarity"] == "supports"
                            for row in candidates))

    def test_all_relations_are_embedded_in_every_layer_leaf(self):
        relations = [
            {"subject": "copper", "predicate": "is", "object": "metal",
             "polarity": "supports", "confidence": 0.9},
            {"subject": "copper", "predicate": "conducts", "object": "electricity",
             "polarity": "supports", "confidence": 0.7},
            {"subject": "copper", "predicate": "is", "object": "brittle",
             "polarity": "refutes", "confidence": 0.6},
        ]
        coordinator = self.make()
        result = coordinator.observe("Copper is a metal.", relations=relations)
        expected = [
            {**item,
             "subject": item["subject"].casefold(),
             "predicate": item["predicate"].casefold(),
             "object": item["object"].casefold()}
            for item in relations
        ]
        for layer in result["layers"]:
            records = list(coordinator.space.records(layer=layer["layer"]))
            payload = records[-1]["payload"]
            self.assertEqual(payload["relation"], expected[0])
            self.assertEqual(payload["relations"], expected)
            if layer["layer"] in {"sentence", "narrative"}:
                self.assertEqual(layer["features"]["relation"], expected[0])

    def test_wordnet_depth_changes_bounded_ancestry_and_relations(self):
        coordinator = self.make(with_dictionary=True)
        shallow = coordinator.ingest_word("copper", depth=1)
        deep = coordinator.ingest_word("copper", depth=3)
        self.assertEqual(shallow["wordnet"]["depth"], 1)
        self.assertEqual(shallow["wordnet"]["depth_used"], 1)
        self.assertEqual(deep["wordnet"]["depth"], 3)
        self.assertEqual(deep["wordnet"]["depth_used"], 3)
        self.assertEqual([item["offset"] for item in shallow["wordnet"]["ancestry"]],
                         [200])
        self.assertEqual([item["offset"] for item in deep["wordnet"]["ancestry"]],
                         [200, 300, 400])
        self.assertEqual(shallow["wordnet"]["synsets"][0]["inherited"], ["metal"])
        self.assertEqual(deep["wordnet"]["synsets"][0]["inherited"],
                         ["entity", "material", "metal"])
        shallow_predicates = [item["predicate"]
                              for item in shallow["wordnet"]["relations"]]
        deep_predicates = [item["predicate"]
                           for item in deep["wordnet"]["relations"]]
        self.assertIn("defined_as", shallow_predicates)
        self.assertIn("kind_of", shallow_predicates)
        self.assertIn("defined_as", deep_predicates)
        self.assertIn("kind_of", deep_predicates)
        self.assertGreater(len(deep["wordnet"]["relations"]),
                           len(shallow["wordnet"]["relations"]))
        self.assertGreater(deep_predicates.count("kind_of"),
                           shallow_predicates.count("kind_of"))

    def test_wordnet_ingestion_uses_canonical_hw4_nannet_route(self):
        coordinator = self.make(with_dictionary=True)
        result = coordinator.ingest_word("copper", depth=2)
        self.assertEqual(result["wordnet"]["hw4_route"],
                         "nan://GM0:SC0:MC0:BS0:0:HW4:0")
        self.assertEqual(result["wordnet"]["hw4_legacy_route"],
                         "nan://HW4:BS0:MC0:SC0:GM0")
        self.assertEqual(result["wordnet"]["hw4_link"], "HW4:0")
        self.assertEqual(result["wordnet"]["nannet_port"], "BS0:0")
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
