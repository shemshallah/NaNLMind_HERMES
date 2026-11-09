"""Tests for the read-only quantum-memory bridge."""
from __future__ import annotations

import copy
import json
import unittest

from nanlmind_hermes.concept_matrix import ConceptMatrix
from nanlmind_hermes.quantum_bridge import (
    CLAIMS,
    LOCALITY,
    SCHEMA,
    SECTION_ORDER,
    QuantumMemoryBridge,
)

PAIRS = (
    ("alpha", "beta"),
    ("alpha", "gamma"),
    ("alpha", "delta"),
    ("beta", "gamma"),
    ("beta", "delta"),
    ("gamma", "delta"),
)
WEIGHTS = {
    "lexical": (0.9, 0.7, 0.5, 0.8, 0.6, 0.4),
    "sentence": (0.2, 0.9, 0.4, 0.6, 0.3, 0.8),
}


def _row(subject: str, predicate: str, obj: str, group: str,
         confidence: float, nested: bool = False) -> dict:
    item = {
        "kind": "relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "expression": None,
        "confidence": confidence,
        "provenance": {"resolution": group},
    }
    if nested:
        return {"item": item, "source_digest": "a" * 64}
    return item


def complete_rows() -> list[dict]:
    rows = []
    for group in ("lexical", "sentence"):
        for (subject, obj), weight in zip(PAIRS, WEIGHTS[group]):
            rows.append(_row(
                subject, "links", obj, group, weight,
                nested=(group == "lexical")))
    return rows


class QuantumBridgeTests(unittest.TestCase):
    def test_complete_rows_expose_supported_sections(self):
        data = QuantumMemoryBridge(memory_rows=complete_rows()).to_dict()
        self.assertEqual(data["schema"], SCHEMA)
        self.assertEqual(data["locality"], LOCALITY)
        self.assertFalse(data["canonical_state_modified"])
        self.assertEqual(data["claims"], CLAIMS)
        self.assertEqual(
            data["provenance"]["counts"]["rows_received"], 12)
        self.assertEqual(data["provenance"]["counts"]["rows_used"], 12)
        self.assertEqual(data["provenance"]["counts"]["vertices"], 4)
        self.assertEqual(data["provenance"]["counts"]["edges"], 6)

        gauge = data["sections"]["gauge_cycle"]
        self.assertTrue(gauge["available"], gauge["reason"])
        self.assertEqual(gauge["graph"]["vertex_count"], 4)
        self.assertEqual(gauge["graph"]["observed_edges"], 6)
        self.assertEqual(gauge["graph"]["cycle_rank"], 3)
        self.assertEqual(gauge["graph"]["fundamental_cycle_count"], 3)
        self.assertEqual(gauge["graph"]["triangle_count"], 4)
        invariants = gauge["invariants"]
        self.assertTrue(invariants["available"], invariants["reason"])
        self.assertEqual(invariants["fundamental_cycle_count"], 3)
        self.assertLess(invariants["reconstruction_error"], 1e-6)
        self.assertLess(invariants["max_cycle_divergence"], 1e-8)

        retrieval = gauge["gauge_fixed_retrieval"]
        self.assertTrue(retrieval["available"], retrieval["reason"])
        self.assertEqual(retrieval["pattern_count"], 2)
        self.assertEqual(retrieval["shared_observed_pairs"], 6)
        self.assertTrue(retrieval["best_group"].startswith(
            "provenance:resolution:"))
        self.assertEqual(
            len(retrieval["group_residuals"]), 2)

        codec = gauge["codec"]
        self.assertTrue(codec["available"], codec["reason"])
        self.assertGreaterEqual(codec["basis_dimensions"], 1)
        self.assertLess(codec["reconstruction_error"], 1e-5)

        state = gauge["cycle_state"]
        self.assertTrue(state["available"], state["reason"])
        self.assertGreaterEqual(state["stored_edges"], 1)
        self.assertLess(state["gauge_shift_residual"], 1e-8)
        self.assertLess(state["max_divergence"], 1e-9)

        phase = data["sections"]["phase_coherence"]
        self.assertTrue(phase["available"], phase["reason"])
        self.assertEqual(phase["triangles_used"], 4)
        self.assertGreaterEqual(phase["mean_triple_coherence"], 0.0)
        self.assertLessEqual(phase["mean_triple_coherence"], 1.0)
        self.assertGreaterEqual(phase["min_triple_coherence"], 0.0)
        self.assertLessEqual(phase["max_triple_coherence"], 1.0)
        self.assertGreaterEqual(phase["phase_locking_order"], 0.0)
        self.assertLessEqual(phase["phase_locking_order"], 1.0)
        phase_memory = phase["phase_memory"]
        self.assertTrue(phase_memory["available"], phase_memory["reason"])
        self.assertEqual(phase_memory["pattern_count"], 2)
        self.assertLess(phase_memory["global_phase_score_delta"], 1e-9)
        self.assertTrue(phase_memory["real_channel_agreement"])
        typed_loops = phase["typed_loops"]
        self.assertTrue(typed_loops["available"], typed_loops["reason"])
        self.assertEqual(typed_loops["loops_used"], 4)
        self.assertGreaterEqual(typed_loops["min_trace_norm"], 0.0)
        self.assertLessEqual(typed_loops["max_trace_norm"], 1.0)

        pair = data["sections"]["pair_spectral"]
        self.assertTrue(pair["available"], pair["reason"])
        pair_space = pair["pair_space"]
        self.assertTrue(pair_space["available"], pair_space["reason"])
        self.assertEqual(pair_space["pair_count"], 6)
        self.assertEqual(
            sum(row["multiplicity"]
                for row in pair_space["johnson_spectrum"]), 6)
        self.assertGreaterEqual(
            pair_space["return_probability_at_revival"], 0.999)
        self.assertGreaterEqual(pair_space["ideal_grover_queries"], 0)
        spectral = pair["spectral"]
        self.assertTrue(spectral["available"], spectral["reason"])
        self.assertEqual(spectral["matrix_size"], 4)
        self.assertEqual(len(spectral["fiedler_values"]), 2)

        robust = data["sections"]["robust_recovery"]
        self.assertTrue(robust["available"], robust["reason"])
        self.assertIn(robust["gauge_flat"], (True, False))
        self.assertGreaterEqual(robust["flatness_ratio"], 0.0)
        self.assertLessEqual(robust["flatness_ratio"], 1.0)
        self.assertEqual(robust["pairs_used"], 6)

        self.assertFalse(data["sections"]["concept_graph"]["available"])
        self.assertTrue(data["sections"]["concept_graph"]["reason"])
        self.assertFalse(data["sections"]["semantic_state"]["available"])
        self.assertTrue(data["sections"]["semantic_state"]["reason"])

        for name in SECTION_ORDER:
            section = data["sections"][name]
            self.assertEqual(section["locality"], LOCALITY)
            self.assertTrue(section["derived"])
            self.assertEqual(len(section["input_sha256"]), 64)

    def test_empty_input_fails_closed(self):
        for bridge in (QuantumMemoryBridge(),
                       QuantumMemoryBridge(memory_rows=[])):
            data = bridge.to_dict()
            for name in SECTION_ORDER:
                section = data["sections"][name]
                self.assertFalse(section["available"], name)
                self.assertTrue(section["reason"], name)
            summary = bridge.summary()
            self.assertEqual(
                [row["section"] for row in summary["sections"]],
                list(SECTION_ORDER))
            self.assertTrue(all(
                not row["available"] for row in summary["sections"]))
            self.assertTrue(all(
                row["reason"] for row in summary["sections"]))
            self.assertEqual(summary["locality"], LOCALITY)
            self.assertEqual(len(summary["input_sha256"]), 64)
            self.assertTrue(all(
                value is None for value in summary["headline"].values()))

        empty_matrix = QuantumMemoryBridge(
            concept_graph=ConceptMatrix()).to_dict()
        self.assertFalse(
            empty_matrix["sections"]["concept_graph"]["available"])
        self.assertTrue(
            empty_matrix["sections"]["concept_graph"]["reason"])

    def test_unsupported_rows_fail_closed(self):
        rows = [
            {"kind": "relation"},
            "not-a-row",
            {"subject": "a", "predicate": "p", "object": "a"},
            {"subject": "a", "predicate": "p", "object": "b"},
        ]
        bridge = QuantumMemoryBridge(memory_rows=rows)
        data = bridge.to_dict()
        counts = data["provenance"]["counts"]
        self.assertEqual(counts["rows_received"], 4)
        self.assertEqual(counts["rows_used"], 1)
        self.assertEqual(counts["rows_skipped"], 3)
        reasons = {
            row["reason"] for row in data["provenance"]["skipped_rows"]
        }
        self.assertEqual(reasons, {
            "missing-subject-predicate-object",
            "row-is-not-a-mapping",
            "self-loop",
        })
        for name in (
            "gauge_cycle", "phase_coherence", "pair_spectral",
            "robust_recovery", "concept_graph", "semantic_state",
        ):
            self.assertFalse(data["sections"][name]["available"], name)
            self.assertTrue(data["sections"][name]["reason"], name)
        self.assertNotEqual(
            data["provenance"]["input_sha256"],
            QuantumMemoryBridge().to_dict()["provenance"]["input_sha256"])

    def test_output_is_deterministic(self):
        rows = complete_rows()
        first = QuantumMemoryBridge(memory_rows=rows)
        second = QuantumMemoryBridge(memory_rows=list(reversed(rows)))
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True),
            json.dumps(second.to_dict(), sort_keys=True))
        self.assertEqual(first.summary(), second.summary())
        self.assertEqual(first.to_dict(), first.to_dict())

        mutated = first.to_dict()
        mutated["sections"]["gauge_cycle"]["graph"]["vertices"][0] = "mutated"
        mutated["provenance"]["rows"][0]["subject"] = "mutated"
        fresh = first.to_dict()
        self.assertNotEqual(
            fresh["sections"]["gauge_cycle"]["graph"]["vertices"][0],
            "mutated")
        self.assertNotEqual(
            fresh["provenance"]["rows"][0]["subject"], "mutated")

        summary = first.summary()
        summary["headline"]["fiedler_values"].append(999.0)
        self.assertNotIn(
            999.0, first.summary()["headline"]["fiedler_values"])

    def test_inputs_are_not_mutated(self):
        rows = complete_rows()
        row_snapshot = copy.deepcopy(rows)
        QuantumMemoryBridge(memory_rows=rows)
        self.assertEqual(rows, row_snapshot)

        graph = {
            "nodes": ["a", "b", "c"],
            "edges": [
                {"subject": "a", "predicate": "links", "object": "b",
                 "weight": 1.0},
                {"subject": "b", "predicate": "links", "object": "c",
                 "weight": 2.0},
                {"subject": "a", "predicate": "links", "object": "c",
                 "weight": 3.0},
            ],
        }
        graph_snapshot = copy.deepcopy(graph)
        QuantumMemoryBridge(concept_graph=graph)
        self.assertEqual(graph, graph_snapshot)

        matrix = ConceptMatrix()
        matrix.observe("alpha links beta", {
            "subject": "alpha", "predicate": "links", "object": "beta"})
        matrix.observe("beta links gamma", {
            "subject": "beta", "predicate": "links", "object": "gamma"})
        pairs_snapshot = copy.deepcopy(matrix.pairs)
        concepts_snapshot = copy.deepcopy(matrix.concepts)
        statements = matrix.statements
        QuantumMemoryBridge(concept_graph=matrix)
        self.assertEqual(matrix.pairs, pairs_snapshot)
        self.assertEqual(matrix.concepts, concepts_snapshot)
        self.assertEqual(matrix.statements, statements)

    def test_concept_graph_only(self):
        graph = {
            "nodes": ["a", "b", "c"],
            "edges": [
                {"subject": "a", "predicate": "links", "object": "b",
                 "weight": 1.0},
                {"subject": "b", "predicate": "links", "object": "c",
                 "weight": 2.0},
                {"subject": "a", "predicate": "links", "object": "c",
                 "weight": 3.0},
            ],
        }
        bridge = QuantumMemoryBridge(concept_graph=graph)
        data = bridge.to_dict()
        gauge = data["sections"]["gauge_cycle"]
        self.assertTrue(gauge["available"], gauge["reason"])
        self.assertEqual(gauge["graph"]["vertex_count"], 3)
        self.assertEqual(gauge["graph"]["observed_edges"], 3)
        self.assertEqual(gauge["graph"]["cycle_rank"], 1)
        self.assertFalse(gauge["gauge_fixed_retrieval"]["available"])
        self.assertTrue(gauge["gauge_fixed_retrieval"]["reason"])

        phase = data["sections"]["phase_coherence"]
        self.assertTrue(phase["available"], phase["reason"])
        self.assertEqual(phase["triangles_used"], 1)
        self.assertFalse(phase["phase_memory"]["available"])
        self.assertTrue(phase["phase_memory"]["reason"])
        self.assertTrue(phase["typed_loops"]["available"])

        robust = data["sections"]["robust_recovery"]
        self.assertTrue(robust["available"], robust["reason"])
        self.assertEqual(robust["pairs_used"], 3)

        concept = data["sections"]["concept_graph"]
        self.assertTrue(concept["available"], concept["reason"])
        self.assertEqual(concept["source"], "mapping")
        self.assertEqual(concept["nodes"], 3)
        self.assertEqual(concept["edges_used"], 3)
        self.assertLessEqual(len(concept["top_concepts"]), 8)
        self.assertLessEqual(len(concept["top_pairs"]), 8)
        self.assertEqual(
            bridge.summary()["headline"]["cycle_dimension"], 1)

    def test_concept_matrix_input(self):
        matrix = ConceptMatrix()
        matrix.observe("alpha links beta", {
            "subject": "alpha", "predicate": "links", "object": "beta"})
        matrix.observe("beta links gamma", {
            "subject": "beta", "predicate": "links", "object": "gamma"})
        matrix.observe("alpha links gamma", {
            "subject": "alpha", "predicate": "links", "object": "gamma"})
        bridge = QuantumMemoryBridge(concept_graph=matrix)
        concept = bridge.to_dict()["sections"]["concept_graph"]
        self.assertTrue(concept["available"], concept["reason"])
        self.assertEqual(concept["source"], "ConceptMatrix")
        self.assertEqual(concept["statements"], matrix.statements)
        self.assertEqual(concept["nodes"], len(matrix.concepts))
        self.assertEqual(concept["edges_used"], len(matrix.pairs))
        self.assertTrue(concept["roles"])

    def test_semantic_state_optional(self):
        nodes = {
            0: {"hypernyms": [1]},
            1: {"hypernyms": [2]},
            2: {},
        }
        bridge = QuantumMemoryBridge(
            memory_rows=complete_rows(),
            semantic_nodes=nodes,
            semantic_start=0,
            semantic_depth=2,
            semantic_dimensions=8,
        )
        semantic = bridge.to_dict()["sections"]["semantic_state"]
        self.assertTrue(semantic["available"], semantic["reason"])
        self.assertEqual(semantic["nodes"], 3)
        self.assertEqual(semantic["ancestor_edges"], 2)
        self.assertEqual(semantic["dimensions"], 8)
        self.assertIsNotNone(semantic["ordinary_norm"])
        self.assertIsNotNone(semantic["cycle_norm"])
        self.assertEqual(len(semantic["ordinary_sha256"]), 64)
        self.assertEqual(len(semantic["cycle_sha256"]), 64)
        self.assertFalse(
            QuantumMemoryBridge(
                memory_rows=complete_rows()).to_dict()
            ["sections"]["semantic_state"]["available"])

    def test_bounded_output(self):
        rows = []
        vertices = [f"v{index:02d}" for index in range(10)]
        for first in range(len(vertices)):
            for second in range(first + 1, len(vertices)):
                rows.append(_row(
                    vertices[first], "links", vertices[second],
                    "bulk", 0.5))
        graph_edges = [
            {"subject": vertices[0], "predicate": "links",
             "object": vertices[1], "weight": 1.0}
            for _ in range(10)
        ]
        bridge = QuantumMemoryBridge(
            memory_rows=rows,
            concept_graph={"nodes": vertices, "edges": graph_edges},
            max_rows=8,
            max_graph_edges=4,
            max_vertices=5,
            max_triangles=3,
            max_channels=3,
            max_patterns=2,
        )
        data = bridge.to_dict()
        counts = data["provenance"]["counts"]
        self.assertTrue(counts["rows_truncated"])
        self.assertEqual(counts["rows_received"], 45)
        self.assertLessEqual(counts["rows_used"], 8)
        self.assertLessEqual(len(data["provenance"]["rows"]), 8)
        gauge = data["sections"]["gauge_cycle"]
        self.assertLessEqual(len(gauge["graph"]["vertices"]), 5)
        self.assertLessEqual(
            gauge["gauge_fixed_retrieval"].get("pattern_count", 0), 2)
        phase = data["sections"]["phase_coherence"]
        self.assertLessEqual(phase["triangles_used"], 3)
        self.assertLessEqual(
            len(phase["phase_memory"].get("channels", [])), 3)
        concept = data["sections"]["concept_graph"]
        self.assertLessEqual(concept["edges_used"], 4)
        self.assertLessEqual(len(concept["top_concepts"]), 8)
        self.assertLessEqual(len(concept["top_pairs"]), 8)
        json.dumps(data, sort_keys=True)
        summary = bridge.summary()
        self.assertEqual(len(summary["sections"]), len(SECTION_ORDER))


if __name__ == "__main__":
    unittest.main()
