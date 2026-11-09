"""Tests for the resident pair/temporal corpus learner."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.resident.learning import ResidentLearner, extract_relations
from nanlmind_hermes.resident.universe import ResidentUniverse

SIZE = 16 * 1024 * 1024


class ResidentLearningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_resident_pair_channel_learns_and_survives_reopen(self):
        path = self.root / "learning.nlmres"
        with ResidentUniverse.create(path, size=SIZE) as universe:
            learner = ResidentLearner(universe)
            before = learner.evaluate([
                {"text": "A penguin is a bird.", "source": "heldout"},
                {"text": "A river carries water.", "source": "heldout"},
            ])
            train = [
                {"text": "A penguin is a bird.", "head": 3},
                {"text": "A penguin swims in water.", "head": 3},
                {"text": "A river carries water.", "head": 7},
                {"text": "A bird can fly.", "head": 9},
            ]
            receipt = learner.observe_batch(train, batch_size=2)
            self.assertGreaterEqual(receipt["learned"], 3)
            after = learner.evaluate([
                {"text": "A penguin is a bird.", "source": "heldout"},
                {"text": "A river carries water.", "source": "heldout"},
            ])
            self.assertLessEqual(
                after["channels"]["resident"]["mean_negative_log_likelihood"],
                before["channels"]["resident"]["mean_negative_log_likelihood"])
            self.assertGreater(universe.leaves.count, 2)
            probe = learner.next_words("penguin", top_k=3)
            self.assertEqual(len(probe), 3)
            temporal = self.root / "temporal.json"
            learner.save_temporal(temporal)
            self.assertTrue(temporal.is_file())

        with ResidentUniverse.open(path) as reopened:
            learner = ResidentLearner(reopened)
            self.assertTrue(learner.load_temporal(self.root / "temporal.json"))
            before = learner.temporal.observe("A new sentence is new.")
            self.assertTrue(before)
            self.assertEqual(learner.next_words("penguin", top_k=1)[0]["word"],
                                learner.next_words("penguin", top_k=1)[0]["word"])

    def test_definition_parser_is_explicit_and_bounded(self) -> None:
        relation = extract_relations(
            "annoyance means the act of troubling or annoying someone.",
            stage="05_word_concept")
        self.assertEqual(len(relation), 1)
        self.assertEqual(relation[0]["predicate"], "means")
        self.assertEqual(relation[0]["subject"], "annoyance")
        self.assertEqual(extract_relations("What is a penguin?"), [])
        self.assertEqual(extract_relations("Alice was beginning to get tired."),
                         [])

    def test_questions_are_not_learned_as_exposure(self):
        with ResidentUniverse.create(self.root / "controls.nlmres",
                                     size=SIZE) as universe:
            learner = ResidentLearner(universe)
            before = universe.leaves.count
            result = learner.observe_batch([
                {"text": "What is a penguin?"},
            ])
            self.assertEqual(result["learned"], 0)
            self.assertEqual(universe.leaves.count, before)

    def test_definition_relations_are_ingested_when_enabled(self):
        with ResidentUniverse.create(self.root / "definitions.nlmres",
                                     size=SIZE) as universe:
            learner = ResidentLearner(universe, extract_relations=True)
            learner.observe_batch([
                {"text": "annoyance means the act of troubling someone.",
                 "stage": "05_word_concept"},
            ])
            self.assertEqual(len(universe.relations), 1)
            self.assertGreaterEqual(universe.relations.status()["hyperedges"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
