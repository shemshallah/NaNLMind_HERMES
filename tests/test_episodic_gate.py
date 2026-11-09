"""Tests for the pure-Python open-set episodic copy/abstain gate."""
from __future__ import annotations

import random
import unittest
from pathlib import Path

from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.episodic_gate import (
    CANDIDATES,
    EpisodicCycleMemory,
    LogisticGate,
    corrupt_name,
    edit_distance,
    evaluate,
    load_pool,
    make_episode,
    match_episode,
    ngram_similarity,
    stress,
    train_gate,
)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "experiments/round034/nan_round034/wordnet_episodic_pool.json"


class EpisodicGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = load_pool(POOL)

    def test_pool_and_episode_shape(self):
        self.assertEqual(set(self.pool), {"train", "val", "test"})
        self.assertGreaterEqual(len(self.pool["test"]), 100)
        episode = make_episode(self.pool["test"], random.Random(1))
        self.assertEqual(len(episode.names), CANDIDATES)
        self.assertEqual(len(episode.records), CANDIDATES)
        self.assertTrue(all(len(record) == 24 for record in episode.records))
        self.assertTrue(episode.present)
        absent = make_episode(self.pool["test"], random.Random(1), present=False)
        self.assertFalse(absent.present)
        self.assertIsNone(absent.target)
        self.assertNotEqual(absent.query, "")

    def test_corruption_and_distance_are_deterministic(self):
        self.assertEqual(corrupt_name("penguin", random.Random(3), 0), "penguin")
        self.assertEqual(edit_distance("penguin", "penguin"), 0)
        self.assertEqual(edit_distance("penguin", "penguins"), 1)
        self.assertGreater(ngram_similarity("penguin", "penguin"), 0.99)
        self.assertEqual(corrupt_name("penguin", random.Random(3), 2),
                         corrupt_name("penguin", random.Random(3), 2))

    def test_router_learns_and_abstains(self):
        gate = train_gate(self.pool, seed=17)
        self.assertIsInstance(gate, LogisticGate)
        result = evaluate(self.pool["test"], gate, count=200, corruption=2)
        self.assertGreater(result["accept_accuracy"], 0.60)
        self.assertGreater(result["positive_accept_rate"], 0.60)
        self.assertFalse(result["canonical_merge"])
        self.assertFalse(result["language_generation_claim"])

    def test_cycle_copy_is_gauge_invariant_but_raw_edge_copy_is_not(self):
        cells = ResidentCells.format_region(bytearray(REGION_BYTES))
        memory = FullCycleMemory(cells)
        episode = make_episode(self.pool["test"], random.Random(9),
                               present=True, corruption=2)
        episodic = EpisodicCycleMemory(memory)
        for slot, record in enumerate(episode.records):
            episodic.store(slot, record)
        target = episode.target
        self.assertIsNotNone(target)
        receipt = episodic.copy(target, episode.records[target], seed=340)
        self.assertTrue(receipt.cycle_exact)
        self.assertFalse(receipt.raw_exact)
        self.assertEqual(receipt.record, episode.records[target])
        self.assertEqual(episodic.audit()["dimension"], 32385)

    def test_stress_reports_corruption_levels_without_generation_claim(self):
        gate = train_gate(self.pool, seed=29)
        report = stress(self.pool["test"], gate, count=60)
        self.assertEqual(set(report), {"0", "2", "4", "6"})
        for item in report.values():
            self.assertFalse(item["canonical_merge"])
            self.assertGreaterEqual(item["queries"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
