"""MTH root 4x4 table and L0-L9 layer curriculum tests."""
import unittest

from nanlmind_hermes.mth_layers import (
    LAYER_NAMES,
    MTHLayerStack,
    MTHScorer,
    apply_bits,
    bool_class,
    root_table,
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
]
SENTENCES = [
    "A volcano is a mountain.",
    "A river is a flowing body of water.",
    "Water can freeze.",
    "Rain causes rivers to rise.",
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
