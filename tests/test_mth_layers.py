"""MTH root 4x4 table and L0-L9 layer curriculum tests."""
import unittest

import tempfile
from pathlib import Path

from nanlmind_hermes.mth_layers import (
    LAYER_NAMES,
    MTHHeadMapper,
    MTHLayerStack,
    MTHScorer,
    apply_bits,
    bool_class,
    root_table,
)
from nanlmind_hermes.universe_chart import (
    HEAD_REGISTER_MAGIC,
    HEAD_REGISTER_OFFSET,
    LEAF_MAGIC,
    OFF_HEAD_SCRATCH,
    Chart8MiB,
    HeadScratchRegister,
)
from nanlmind_hermes.temporal_lm import TemporalMirrorLM

ROWS = [
    {"kind": "relation", "subject": "a volcano", "predicate": "is",
     "object": "a mountain", "confidence": 0.75, "polarity": "supports"},
    {"kind": "relation", "subject": "a river", "predicate": "is",
     "object": "a flowing body of water", "confidence": 0.8, "polarity": "supports"},
    {"kind": "relation", "subject": "water", "predicate": "can",
     "object": "freeze", "confidence": 0.9, "polarity": "supports"},
    {"kind": "relation", "subject": "rain", "predicate": "causes",
     "object": "rivers to rise", "confidence": 0.7, "polarity": "supports"},
    {"kind": "relation", "subject": "the sky", "predicate": "is",
     "object": "a component of the earth", "confidence": 0.8, "polarity": "supports"},
    {"kind": "relation", "subject": "hot", "predicate": "is",
     "object": "the opposite of cold", "confidence": 0.9, "polarity": "supports"},
]
SENTENCES = [
    "A volcano is a mountain.",
    "A river is a flowing body of water.",
    "Water can freeze.",
    "Rain causes rivers to rise.",
    "The sky is a component of the earth.",
    "Hot is the opposite of cold.",
]


class RootTableTests(unittest.TestCase):
    def test_root_4x4_covers_all_sixteen_class_pairs(self):
        table = root_table()
        self.assertEqual(len(table), 4)
        self.assertEqual({len(row) for row in table}, {4})
        nibbles = {cell["class_nibble"] for row in table for cell in row}
        self.assertEqual(nibbles, set(range(16)))

    def test_root_boolean_functions(self):
        self.assertEqual(bool_class(8, 1, 1), 1)   # AND
        self.assertEqual(bool_class(8, 1, 0), 0)
        self.assertEqual(bool_class(14, 0, 0), 0)  # OR
        self.assertEqual(bool_class(14, 0, 1), 1)
        self.assertEqual(bool_class(6, 1, 0), 1)   # XOR
        self.assertEqual(bool_class(6, 1, 1), 0)
        self.assertEqual(apply_bits(8, 0b1010, 0b1100, 4), 0b1000)
        self.assertEqual(apply_bits(14, 0b1010, 0b1100, 8), 0b1110)


class LayerTests(unittest.TestCase):
    def test_curriculum_learns_bits_to_speech(self):
        stack = MTHLayerStack()
        summary = stack.learn(SENTENCES, ROWS)
        self.assertEqual(set(summary["layers"]), set(LAYER_NAMES))
        self.assertEqual(summary["layers"]["bit"]["accuracy"], 1.0)
        self.assertEqual(summary["layers"]["nibble"]["accuracy"], 1.0)
        self.assertEqual(summary["layers"]["byte"]["accuracy"], 1.0)
        self.assertGreater(summary["layers"]["sentence"]["examples"], 0)
        self.assertGreater(summary["edges"], 0)
        self.assertFalse(summary["native_verified"])
        self.assertGreater(summary["vocabulary"], 0)

    def test_scorer_prefers_the_typed_truth_over_a_stitched_sentence(self):
        stack = MTHLayerStack()
        stack.learn(SENTENCES, ROWS)
        scorer = MTHScorer(stack)
        good = scorer.score("What is a volcano?", "A volcano is a mountain.")
        stitched = scorer.score(
            "What is a volcano?",
            "A volcano is a flowing body of water can freeze.")
        self.assertGreater(good["truth"], stitched["truth"])
        self.assertGreater(good["score"], stitched["score"])

    def test_every_layer_is_in_the_forward_pass_and_scoring(self):
        stack = MTHLayerStack()
        stack.learn(SENTENCES, ROWS)
        forward = stack.forward("Rain causes rivers to rise.",
                                context="Rain is water.")
        self.assertEqual(set(forward), set(LAYER_NAMES))
        self.assertTrue(all(forward[name] for name in LAYER_NAMES))
        scorer = MTHScorer(stack)
        scored = scorer.score("Why does rain cause rivers to rise?",
                              "Rain causes rivers to rise.",
                              context="Rain is water.")
        self.assertEqual(set(scored["layers"]), set(LAYER_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scored["layers"].values()))
        self.assertGreater(scored["layers"]["narrative"], 0.5)
        stitched = scorer.score("Why does rain cause rivers to rise?",
                                "Rain causes rivers to rise. A bird is an animal.")
        self.assertLess(stitched["coherence"], scored["coherence"] + 1.0)

    def test_heads_use_root_transforms_and_fill_the_scratch_register(self):
        stack = MTHLayerStack()
        stack.learn(SENTENCES, ROWS)
        mapper = MTHHeadMapper(stack)
        leaves = [mapper.map_relation(row) for row in ROWS]
        self.assertTrue(all(0 <= leaf["head"] <= 255 for leaf in leaves))
        self.assertTrue(all(0 <= leaf["mask"] <= 15 for leaf in leaves))
        self.assertGreater(len({leaf["head"] for leaf in leaves}), 4)
        self.assertEqual(leaves, [mapper.map_relation(row) for row in ROWS])
        self.assertEqual(len({leaf["digest"] for leaf in leaves}), len(leaves))
        roles = {leaf["role"] for leaf in leaves}
        self.assertIn("component_of", roles)
        self.assertIn("opposition", roles)
        sky = next(leaf for leaf in leaves if leaf["role"] == "component_of")
        self.assertEqual(sky["rail"] in {"s+", "s-"}, True)
        self.assertIn("UNIT:WHOLE:COMPONENT:QUERY", sky["tags"])
        with tempfile.TemporaryDirectory() as tmp:
            chart = Chart8MiB.genesis()
            register = HeadScratchRegister(chart, Path(tmp))
            status = register.organize(ROWS, mapper)
            self.assertEqual(status["leaf_count"], len(ROWS))
            self.assertGreater(status["head_count"], 0)
            self.assertEqual(chart.bytes[HEAD_REGISTER_OFFSET:HEAD_REGISTER_OFFSET + 8],
                             HEAD_REGISTER_MAGIC)
            head = status["leaves"][0]["head"]
            scratch = OFF_HEAD_SCRATCH + head * 1024
            self.assertEqual(chart.bytes[scratch:scratch + 8], LEAF_MAGIC)

    def test_structured_answers_come_from_learned_relations(self):
        from nanlmind_hermes.mth_layers import structured_candidates
        rows = ROWS + [
            {"kind": "relation", "subject": "copper", "predicate": "is",
             "object": "a chemical element", "confidence": 0.9, "polarity": "supports"},
        ]
        answer = structured_candidates("Is copper a chemical element?", rows)
        self.assertTrue(answer)
        self.assertIn("chemical element", answer[0]["text"])
        self.assertEqual(answer[0]["role"], "member_of:chemical")
        sky = structured_candidates("Is the sky a component of the earth?", rows)
        self.assertTrue(sky)
        self.assertEqual(sky[0]["role"], "component_of")
        hot = structured_candidates("Is hot the opposite of cold?", rows)
        self.assertTrue(hot)
        self.assertEqual(hot[0]["role"], "opposition")
        self.assertFalse(structured_candidates("Is copper a geometric element?", rows))

    def test_temporal_deliberation_exposes_mth_scores(self):
        model = TemporalMirrorLM()
        stack = MTHLayerStack()
        stack.learn(SENTENCES, ROWS)
        model.mth = MTHScorer(stack)
        for sentence in SENTENCES:
            model.observe(sentence)
        prediction = model.generate("What is a volcano?",
                                    allowed_seed={"volcano", "river", "water"})
        self.assertTrue(prediction["text"])
        self.assertTrue(prediction["considered"])
        self.assertTrue(any(row.get("mth") for row in prediction["considered"]))


if __name__ == "__main__":
    unittest.main()
