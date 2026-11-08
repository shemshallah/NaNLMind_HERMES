import tempfile
import unittest
from pathlib import Path

from nanlmind_hermes.layered_learning import LayeredLearningCoordinator, LearningPolicy
from nanlmind_hermes.mirror_topology import (
    EXPECTED_TOPOLOGY_DIGEST,
    MirrorLeafBindings,
    MirrorTopology,
    MirrorTopologyError,
    pair_heads,
    pair_number,
)
from nanlmind_hermes.mth_layers import ROOT_NAMES
from nanlmind_hermes.self_modification import (
    BooleanGate,
    MthToolFacade,
    SelfModificationEngine,
    SelfModificationError,
    SelfModificationPolicy,
)

EVIDENCE = "a" * 64


class MirrorTopologyTests(unittest.TestCase):
    def test_counts_digest_and_full_address_coverage(self):
        status = MirrorTopology.status()
        self.assertEqual(status["heads"], 256)
        self.assertEqual(status["mirror_points"], 32_640)
        self.assertEqual(status["canonical_cells"], 32_896)
        self.assertEqual(status["raw_addresses"], 65_536)
        self.assertEqual(status["topology_sha256"], EXPECTED_TOPOLOGY_DIGEST)
        seen = set()
        for raw in range(65_536):
            table = MirrorTopology.from_raw(raw)
            seen.add(table["canonical_cell"])
            self.assertEqual(MirrorTopology.from_raw(raw)["canonical_cell"],
                             table["canonical_cell"])
        self.assertEqual(len(seen), 32_896)

    def test_all_mirror_microtables_round_trip(self):
        for mp in range(32_640):
            table = MirrorTopology.mp(mp)
            a, b = pair_heads(mp)
            self.assertEqual((table.a, table.b), (a, b))
            self.assertEqual(table.canonical_address,
                             MirrorTopology.head(0).forward_address if mp < 0 else
                             MirrorTopology.mp(mp).forward_address)
            self.assertEqual(pair_number(a, b), mp)

    def test_head_and_mp_routes(self):
        head = MirrorTopology.head(7).as_dict()
        self.assertEqual(head["route"], "HW0:7")
        self.assertEqual(head["canonical_cell"], 7)
        mp = MirrorTopology.mp(0)
        self.assertEqual((mp.a, mp.b), pair_heads(0))
        route = MirrorTopology.route(head=7, layer=5)
        parsed = MirrorTopology.parse_route(route)
        self.assertEqual(parsed["head"], 7)
        self.assertEqual(parsed["layer"], 5)
        mp_route = MirrorTopology.route(mp=0, layer=3)
        self.assertEqual(MirrorTopology.parse_route(mp_route)["mp"], 0)
        constrained = MirrorTopology.route(head=7, target_mp=pair_number(7, 9), layer=3)
        parsed = MirrorTopology.parse_route(constrained)
        self.assertIsNotNone(parsed["target_mp"])

    def test_nannet0_cannot_be_a_leaf_target(self):
        with self.assertRaises(MirrorTopologyError):
            MirrorTopology.head(256)
        with self.assertRaises(MirrorTopologyError):
            MirrorTopology.mp(32_640)


class LeafBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bindings = MirrorLeafBindings(Path(self.tmp.name) / "memory.sqlite3")

    def test_cas_and_history(self):
        mp = pair_number(1, 2)
        first = self.bindings.bind(
            mp=mp, leaf_id="leaf-a", leaf_version=1, leaf_digest="a" * 64,
            source_head=1, expected_generation=0,
        )
        self.assertEqual(first["generation"], 1)
        second = self.bindings.bind(
            mp=mp, leaf_id="leaf-b", leaf_version=2, leaf_digest="b" * 64,
            source_head=2, expected_generation=1,
        )
        self.assertEqual(second["generation"], 2)
        with self.assertRaises(MirrorTopologyError):
            self.bindings.bind(
                mp=mp, leaf_id="leaf-c", leaf_version=3, leaf_digest="c" * 64,
                source_head=1, expected_generation=1,
            )
        self.assertEqual(self.bindings.resolve(mp)["leaf_id"], "leaf-b")
        self.assertEqual(len(self.bindings.history(mp)), 2)
        self.assertEqual(self.bindings.stats()["bindings"], 1)

    def test_source_head_must_be_an_endpoint(self):
        mp = pair_number(1, 2)
        with self.assertRaises(MirrorTopologyError):
            self.bindings.bind(
                mp=mp, leaf_id="leaf", leaf_version=1, leaf_digest="a" * 64,
                source_head=3, expected_generation=0,
            )


class BooleanAndSelfModificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.coordinator = LayeredLearningCoordinator(
            self.root,
            policy=LearningPolicy(
                min_support=1, min_confidence=0.1,
                require_explicit_approval=False, auto_sync=False,
            ),
        )
        self.bindings = MirrorLeafBindings(self.root / "mathematical_memory.sqlite3")
        self.engine = SelfModificationEngine(
            self.coordinator, self.bindings,
            policy=SelfModificationPolicy(
                min_boolean_confidence=0.5,
                require_evidence=True,
                require_approval_for_promotion=True,
            ),
        )

    def test_boolean_gate_reports_root_function(self):
        gate = BooleanGate()
        result = gate.evaluate(
            {"subject": "copper", "predicate": "is", "object": "metal"}, head=1
        )
        self.assertIn(result["root_function"], list(ROOT_NAMES.values()))
        self.assertEqual(result["head"], 1)
        self.assertIn(result["truth_bit"], (0, 1))
        self.assertGreaterEqual(result["confidence"], 0.0)
        with self.assertRaises(SelfModificationError):
            gate.evaluate({"subject": "a", "predicate": "is", "object": "b"}, head=0)

    def test_relation_mutation_writes_scratch_and_binding(self):
        report = self.engine.propose_relation(
            {"subject": "copper", "predicate": "is", "object": "metal"},
            layer="lexical", source_head=1, evidence_sha256=EVIDENCE,
            strength_q16=32_768, polarity=1, expected_generation=0, tick=4,
        )
        self.assertEqual(report["status"], "mutated")
        self.assertEqual(report["source_head"], 1)
        self.assertEqual(report["scratch"]["layer"], 5)
        self.assertTrue(report["scratch"]["address"].endswith(")"))
        self.assertFalse(report["canonical_training_committed"])
        if report["binding"] is not None:
            resolved = self.bindings.resolve(report["binding"]["mp"])
            self.assertEqual(resolved["leaf_digest"], report["leaf_digest"])
        self.assertEqual(report["boolean"]["head"], 1)

    def test_stale_relation_binding_fails_closed(self):
        relation = {"subject": "copper", "predicate": "is", "object": "metal"}
        first = self.engine.propose_relation(
            relation, layer=5, source_head=1, evidence_sha256=EVIDENCE,
            expected_generation=0,
        )
        if first["binding"] is None:
            self.skipTest("boolean mapping chose a palindromic head")
        with self.assertRaises(SelfModificationError):
            self.engine.propose_relation(
                relation, layer=5, source_head=1, evidence_sha256=EVIDENCE,
                expected_generation=0,
            )

    def test_modify_leaf_is_versioned(self):
        mp = pair_number(1, 2)
        first = self.engine.modify_leaf(
            mp=mp, leaf_id="leaf-a", payload={"value": 1}, source_head=1,
            evidence_sha256=EVIDENCE, expected_generation=0, tick=1,
        )
        self.assertEqual(first["status"], "leaf-modified")
        second = self.engine.modify_leaf(
            mp=mp, leaf_id="leaf-b", payload={"value": 2}, source_head=2,
            evidence_sha256=EVIDENCE, expected_generation=1, tick=2,
        )
        self.assertEqual(second["binding"]["generation"], 2)
        with self.assertRaises(SelfModificationError):
            self.engine.modify_leaf(
                mp=mp, leaf_id="leaf-c", payload={"value": 3}, source_head=1,
                evidence_sha256=EVIDENCE, expected_generation=1,
            )

    def test_all_mth_tool_names_are_available(self):
        tools = [item["name"] for item in self.engine.mth.definitions()]
        self.assertEqual(tools, ["mth.catalog", "mth.inspect", "mth.categories",
                                 "mth.compose", "mth.prepare", "mth.route"])
        self.assertEqual(self.engine.use_mth_tool("mth.catalog")["total"], 10)
        self.assertEqual(self.engine.use_mth_tool("mth.inspect", {"layer": 5})["layer"], 5)
        self.assertTrue(self.engine.use_mth_tool("mth.categories")["layers"])
        composed = self.engine.use_mth_tool("mth.compose", {"plan": [0, 1, 2]})
        self.assertEqual(composed["plan"]["steps"], 3)
        prepared = self.engine.use_mth_tool("mth.prepare", {"layer": 7})
        self.assertEqual(prepared["plan"]["layers"], [7])
        routed = self.engine.use_mth_tool(
            "mth.route", {"address": MirrorTopology.route(head=7, layer=5)}
        )
        self.assertEqual(routed["route"]["head"], 7)

    def test_status_reports_counts_and_policy(self):
        status = self.engine.status()
        self.assertEqual(status["topology"]["heads"], 256)
        self.assertEqual(status["topology"]["mirror_points"], 32_640)
        self.assertEqual(status["topology"]["raw_addresses"], 65_536)
        self.assertEqual(status["bindings"]["bindings"], 0)
        self.assertFalse(status["nannet0_writable"])


if __name__ == "__main__":
    unittest.main()
