"""Provenance-preserving, non-authorizing HERMES->NaN16 claim word staging."""
import copy
import struct
import unittest

from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_native_input import (
    NativeGraphClaimInputError, pack_native_graph_claim_input,
)
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from test_router_graph_history import edge, fixture
from test_router_graph_walk import state_page


class NativeGraphClaimInputTests(unittest.TestCase):
    def setUp(self):
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        a = dict(edge("a"), relation="supports")
        b = dict(edge("b"), relation="refutes")
        c = dict(edge("c"), relation="isa")
        d = dict(edge("d", 1, "candidate", 1))
        d = dict(d, dst=266)
        tombstone = dict(d, generation=2, status="retracted", tick=2)
        self.revisions = [a, b, c, d, tombstone]
        self.state_pages = [
            state_page(self.chart, self.revisions, i) for i in range(4)
        ]
        self.history_pages = [
            fixture(self.chart, self.revisions, history_offset=i)
            for i in range(len(self.revisions))
        ]
        self.verifier = RouterGraphHistoryVerifier(self.chart)
        for p in self.state_pages:
            self.verifier.add_state_page(p)
        for p in self.history_pages:
            self.verifier.add_history_page(p)
        self.assertTrue(self.verifier.status()["history_verified"])

    def stage(self, **kw):
        return pack_native_graph_claim_input(
            self.verifier, self.state_pages, self.history_pages, **kw
        )

    def test_contested_claims_preserved_and_payload_is_u32be(self):
        before = bytes(self.chart.bytes)
        result = self.stage()
        self.assertEqual(result["records_count"], 2)
        self.assertEqual(result["word_count"], 8)
        self.assertEqual(result["words"], [7, 265, 4, 1, 7, 265, 4, 2])
        self.assertEqual(
            struct.pack(">8I", *result["words"]).hex(),
            "0000000700000109000000040000000100000007000001090000000400000002",
        )
        self.assertEqual(
            [item["edge_id"] for item in result["provenance"]], ["a", "b"]
        )
        self.assertEqual(
            [item["evidence_digest"] for item in result["provenance"]],
            ["f" * 64, "f" * 64],
        )
        self.assertTrue(result["history_verified"])
        self.assertTrue(result["chart_root_bound"])
        for forbidden in (
            "sender_authenticated", "evidence_integrity_verified",
            "native_mth_executed", "native_admission_verified",
            "learning_admitted", "mathematical_truth_promoted",
        ):
            self.assertFalse(result[forbidden], forbidden)
        self.assertEqual(bytes(self.chart.bytes), before)
        self.assertEqual(self.stage(), result)

    def test_incomplete_or_mutated_pages_never_emit_guest_words(self):
        for state, history in (
            (self.state_pages[:-1], self.history_pages),
            (self.state_pages, self.history_pages[:-1]),
        ):
            with self.subTest(mode="truncation"), self.assertRaises(
                RouterGraphPageError
            ):
                pack_native_graph_claim_input(self.verifier, state, history)
        altered = copy.deepcopy(self.state_pages)
        altered[0]["entries"][0]["relation"] = "refutes"
        with self.assertRaises(RouterGraphPageError):
            pack_native_graph_claim_input(
                self.verifier, altered, self.history_pages
            )
        altered = copy.deepcopy(self.history_pages)
        altered[0]["history_entries"][0]["payload"] = altered[0][
            "history_entries"
        ][0]["payload"].replace("supports", "refutes")
        with self.assertRaises(RouterGraphPageError):
            pack_native_graph_claim_input(
                self.verifier, self.state_pages, altered
            )

    def test_capacity_and_epoch_boundaries_are_fail_closed(self):
        with self.assertRaises(NativeGraphClaimInputError):
            self.stage(max_records=1)
        for bad in (0, -1, 1025, True):
            with self.subTest(bad=bad), self.assertRaises(
                NativeGraphClaimInputError
            ):
                self.stage(max_records=bad)
        self.chart.snapshot_generation += 1
        with self.assertRaises(RouterGraphPageError):
            self.stage()

    def test_empty_native_claims_are_not_a_refutation(self):
        chart = Chart8MiB.genesis()
        attach_universe_root(chart)
        state = [fixture(chart, [])]
        history = [fixture(chart, [])]
        verifier = RouterGraphHistoryVerifier(chart)
        verifier.add_state_page(state[0])
        verifier.add_history_page(history[0])
        payload = pack_native_graph_claim_input(verifier, state, history)
        self.assertEqual(payload["records_count"], 0)
        self.assertEqual(payload["words"], [])
        self.assertTrue(payload["history_verified"])
        self.assertFalse(payload["learning_admitted"])


if __name__ == "__main__":
    unittest.main()
