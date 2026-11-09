"""Tests for two/three-source resident memory composition."""
from __future__ import annotations

import unittest

from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.composition_gate import (
    ANCHOR,
    compose_path,
    relation_conflict,
    CompositionMemory,
    Fact,
    TwoFactGate,
    build_synthetic_chains,
    compose_three,
    compose_two,
    evaluate_composition,
    evaluate_gate,
)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory


class CompositionGateTests(unittest.TestCase):
    def test_two_hop_composes_or_abstains(self):
        a = Fact("mopper", "worker")
        b = Fact("worker", "person")
        positive = compose_two(a, b)
        self.assertTrue(positive.accepted)
        self.assertEqual(positive.answer, "mopper is a person.")
        negative = compose_two(a, Fact("stone", "solid"))
        self.assertFalse(negative.accepted)
        self.assertEqual(negative.answer, ANCHOR)

    def test_three_hop_composition_and_broken_chain(self):
        facts = (Fact("aura", "symptom"), Fact("symptom", "evidence"),
                 Fact("evidence", "information"))
        result = compose_three(*facts)
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "aura is a kind of information.")
        broken = compose_three(facts[0], facts[1], Fact("rocket", "propulsion"))
        self.assertFalse(broken.accepted)
        self.assertEqual(broken.answer, ANCHOR)

    def test_learned_gate_and_synthetic_accuracy(self):
        positives, negatives = build_synthetic_chains(80, seed=35)
        gate = TwoFactGate().fit(positives[:60], negatives[:60])
        report = evaluate_gate(gate, positives[60:], negatives[60:])
        self.assertGreater(report["accuracy"], 0.9)
        composition = evaluate_composition(positives[60:], negatives[60:])
        self.assertGreater(composition["accuracy"], 0.9)
        self.assertFalse(composition["canonical_merge"])

    def test_contradictory_relation_abstains_and_long_chain_composes(self):
        left = Fact("mopper", "worker", "is a")
        right = Fact("worker", "person", "is a")
        conflict = Fact("worker", "person", "opposes")
        self.assertTrue(relation_conflict(right.relation, conflict.relation))
        self.assertFalse(compose_two(left, conflict).accepted)
        path = (Fact("a", "b"), Fact("b", "c"), Fact("c", "d"),
                Fact("d", "e"), Fact("e", "f"))
        result = compose_path(path)
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "a is a kind of f.")
        self.assertFalse(compose_path((path[0], path[1], path[2],
                                       Fact("wrong", "f"))).accepted)

    def test_cycle_composition_survives_gauge_while_raw_edge_does_not(self):
        cells = ResidentCells.format_region(bytearray(REGION_BYTES))
        memory = FullCycleMemory(cells)
        store = CompositionMemory(memory)
        facts = (Fact("aura", "symptom"), Fact("symptom", "evidence"),
                 Fact("evidence", "information"))
        for slot, fact in enumerate(facts):
            store.store(slot, fact)
        store.gauge(seed=351)
        read = [store.read_cycle(slot) for slot in range(3)]
        self.assertEqual(tuple(read), facts)
        raw = [store.read_raw(slot) for slot in range(3)]
        self.assertNotEqual(tuple(raw), facts)
        result = compose_three(*read)
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "aura is a kind of information.")
        self.assertEqual(store.audit()["dimension"], 32385)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
