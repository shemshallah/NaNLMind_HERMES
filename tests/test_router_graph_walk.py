"""Verified candidate navigation: bounded, non-promotional, chart-bound."""
import copy
import unittest

from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.router_graph_walk import (
    VerifiedGraphWalkError, verified_candidate_walk,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from test_router_graph_history import edge, fixture


def routed(node):
    return f"HW0:{node}" if node < 256 else f"HW0:MP{node - 256}"


def state_page(chart, records, offset):
    page = fixture(chart, records, state_offset=offset, limit=1)
    for record in page["entries"]:
        record["src_route"] = routed(record["src"])
        record["dst_route"] = routed(record["dst"])
    return page


class VerifiedGraphWalkTests(unittest.TestCase):
    def setUp(self):
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        a = dict(edge("a"), src=7, dst=9, source_head=7,
                 relation="supports", layer=4)
        b = dict(edge("b"), src=9, dst=10, source_head=9,
                 relation="isa", layer=5)
        c = dict(edge("c"), src=7, dst=9, source_head=7,
                 relation="refutes", layer=4)
        d = dict(edge("d", 1, "candidate", 1),
                 src=7, dst=11, source_head=7, layer=4)
        d_retract = dict(d, generation=2, status="retracted", tick=2)
        self.revisions = [a, b, c, d, d_retract]
        self.pages = [state_page(self.chart, self.revisions, i)
                      for i in range(4)]
        self.verifier = RouterGraphHistoryVerifier(self.chart)
        for page in self.pages:
            self.verifier.add_state_page(page)
        for i in range(5):
            self.verifier.add_history_page(fixture(
                self.chart, self.revisions, history_offset=i, limit=1))
        self.assertTrue(self.verifier.status()["history_verified"])

    def query(self, **kwargs):
        return verified_candidate_walk(
            self.verifier, self.pages, start=7, **kwargs)

    def test_multihop_preserves_provenance_and_contested_claims(self):
        result = self.query(target=10, max_hops=2)
        self.assertEqual(result["reached_nodes"], 3)
        self.assertEqual(result["matched_candidate_edges"], 3)
        self.assertEqual(len(result["paths"]), 1)
        path = result["paths"][0]["edges"]
        self.assertEqual([(x["src"], x["dst"]) for x in path],
                         [(7, 9), (9, 10)])
        self.assertTrue(all(x["evidence_digest"] == "f" * 64 for x in path))
        self.assertEqual(result["contested_pairs"], [
            {"src": 7, "dst": 9, "layer": 4,
             "relations": ["refutes", "supports"]}
        ])
        self.assertNotIn(11, [p["target"] for p in result["paths"]])
        for forbidden in (
            "sender_authenticated", "evidence_integrity_verified",
            "native_mth_executed", "learning_admitted",
            "mathematical_truth_promoted",
        ):
            self.assertFalse(result[forbidden], forbidden)

    def test_filters_zero_hop_and_direction_are_not_inference(self):
        self.assertEqual(self.query(target=7, max_hops=0)["paths"][0]["edges"], [])
        self.assertEqual(self.query(target=10, max_hops=1)["paths"], [])
        self.assertEqual(self.query(target=10, layer=4)["paths"], [])
        self.assertEqual(self.query(target=10, relation="isa")["paths"], [])
        expected_contested = [{
            "src": 7, "dst": 9, "layer": 4,
            "relations": ["refutes", "supports"],
        }]
        for only_relation in ("supports", "refutes"):
            narrowed = self.query(target=9, relation=only_relation)
            self.assertEqual(narrowed["matched_candidate_edges"], 1)
            self.assertEqual(narrowed["contested_pairs"], expected_contested)
            self.assertEqual(narrowed["paths"][0]["edges"][0]["relation"],
                             only_relation)
        self.assertEqual(self.query(target=11)["paths"], [])
        self.assertEqual(self.query(target=10, max_hops=2)["paths"],
                         self.query(target=10, max_hops=2)["paths"])

    def test_tampered_or_truncated_state_refused_even_after_replay(self):
        changed = copy.deepcopy(self.pages)
        changed[0]["entries"][0]["relation"] = "causes"
        with self.assertRaises(RouterGraphPageError):
            verified_candidate_walk(self.verifier, changed, start=7)
        with self.assertRaises(RouterGraphPageError):
            verified_candidate_walk(self.verifier, self.pages[:3], start=7)
        changed = copy.deepcopy(self.pages)
        changed[1], changed[2] = changed[2], changed[1]
        with self.assertRaises(RouterGraphPageError):
            verified_candidate_walk(self.verifier, changed, start=7)

    def test_fail_closed_on_bounds_and_chart_epoch_change(self):
        for kwargs in (
            {"max_edges": 2}, {"max_nodes": 2},
            {"max_hops": 9}, {"max_hops": True},
            {"max_edges": True}, {"layer": True},
            {"relation": "unproved"}, {"target": 32896},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(
                    VerifiedGraphWalkError):
                self.query(**kwargs)
        self.chart.snapshot_generation += 1
        with self.assertRaises(RouterGraphPageError):
            self.query(target=10)

    def test_history_is_mandatory_even_if_state_is_valid(self):
        partial = RouterGraphHistoryVerifier(self.chart)
        for page in self.pages:
            partial.add_state_page(page)
        with self.assertRaises(VerifiedGraphWalkError):
            verified_candidate_walk(partial, self.pages, start=7)


if __name__ == "__main__":
    unittest.main()
