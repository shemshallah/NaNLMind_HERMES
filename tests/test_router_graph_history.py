"""Adversarial, independent replay tests: no second chart and no learning write."""
import hashlib
import json
import unittest

from nanlmind_hermes.mirror_topology import TOPOLOGY_DIGEST
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def edge(eid="a", generation=1, status="candidate", tick=1):
    return dict(edge_id=eid, src=7, dst=265, layer=4,
                relation="supports", evidence_id="nml:evidence:1",
                evidence_version=1, evidence_digest="f" * 64,
                source_head=7, generation=generation, tick=tick,
                status=status)


def fixture(chart, revisions, state_offset=0, history_offset=0, limit=1):
    revisions = sorted(revisions, key=lambda r: (r["edge_id"], r["generation"]))
    latest = {}
    for record in revisions:
        latest[record["edge_id"]] = record
    current = [latest[k] for k in sorted(latest)]
    state_hash = hashlib.sha256(b"nanrouter.nan-layer-state/1\0")
    history_hash = hashlib.sha256(b"nanrouter.nan-layer-history/1\0")
    layers = [0] * 10
    active = retracted = 0
    for record in current:
        raw = canon(record)
        state_hash.update(len(raw).to_bytes(8, "big") + raw)
        layers[record["layer"]] += 1
        active += record["status"] == "candidate"
        retracted += record["status"] == "retracted"
    history = []
    for record in revisions:
        entry = dict(edge_id=record["edge_id"], generation=record["generation"],
                     payload=canon(record).decode())
        history.append(entry)
        raw = canon([entry["edge_id"], entry["generation"], entry["payload"]])
        history_hash.update(len(raw).to_bytes(8, "big") + raw)
    header = dict(
        schema="nanrouter.nan-layer-master-commitment/1",
        topology_sha256=TOPOLOGY_DIGEST,
        parent_root_sha256=chart.root_hash,
        chart_generation=chart.snapshot_generation,
        total_edges=len(current), candidate_edges=active,
        retracted_edges=retracted, layer_counts=layers,
        state_sha256=state_hash.hexdigest(),
        history_sha256=history_hash.hexdigest(),
    )
    part = current[state_offset:state_offset + limit]
    hist = history[history_offset:history_offset + limit]
    def routed(r):
        return {
            **r,
            "src_route": f"HW0:{r['src']}",
            "dst_route": f"HW0:MP{r['dst'] - 256}",
        }
    return {
        **header,
        "graph_commitment_sha256": hashlib.sha256(
            b"nanrouter.nan-layer-commitment/1\0" + canon(header)).hexdigest(),
        "offset": state_offset, "entries": [routed(r) for r in part],
        "next_offset": state_offset + len(part)
            if state_offset + len(part) < len(current) else None,
        "history_offset": history_offset, "history_entries": hist,
        "history_total": len(history),
        "history_next_offset": history_offset + len(hist)
            if history_offset + len(hist) < len(history) else None,
        "snapshot_consistent": True, "signed_by_router": False,
        "native_mth_receipt": False, "mathematical_truth_promoted": False,
    }


class HistoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.history = [
            edge("a", 1, "candidate", 1),
            edge("a", 2, "retracted", 2),
            edge("a", 3, "candidate", 3),
            edge("b", 1, "candidate", 4),
        ]

    def verify(self, records):
        verifier = RouterGraphHistoryVerifier(self.chart)
        count = len({r["edge_id"] for r in records})
        for i in range(count):
            verifier.add_state_page(fixture(
                self.chart, records, state_offset=i))
        for i in range(len(records)):
            verifier.add_history_page(fixture(
                self.chart, records, history_offset=i))
        return verifier

    def test_full_replay_preserves_reactivation_and_never_claims_admission(self):
        result = self.verify(self.history).status()
        self.assertTrue(result["state_verified"])
        self.assertTrue(result["history_verified"])
        self.assertEqual(result["verified_history_revisions"], 4)
        for forbidden in ("sender_authenticated", "native_mth_executed",
                          "learning_admitted", "mathematical_truth_promoted"):
            self.assertFalse(result[forbidden], forbidden)

    def test_empty_history_only_matches_empty_state(self):
        verifier = RouterGraphHistoryVerifier(self.chart)
        verifier.add_state_page(fixture(self.chart, []))
        self.assertTrue(verifier.add_history_page(
            fixture(self.chart, []))["history_verified"])

    def test_invalid_revision_transitions_rejected_even_with_rehashed_commitment(self):
        mutants = []
        gap = [dict(x) for x in self.history]
        gap[1]["generation"] = 4
        mutants.append(gap)
        backwards = [dict(x) for x in self.history]
        backwards[1]["tick"] = 0
        mutants.append(backwards)
        retarget = [dict(x) for x in self.history]
        retarget[1]["src"] = 8
        mutants.append(retarget)
        tombstone_evidence = [dict(x) for x in self.history]
        tombstone_evidence[1]["evidence_digest"] = "0" * 64
        mutants.append(tombstone_evidence)
        invalid_genesis = [dict(x) for x in self.history]
        invalid_genesis[0]["status"] = "retracted"
        mutants.append(invalid_genesis)
        for mutated in mutants:
            with self.subTest(mutated=mutated), self.assertRaises(
                    RouterGraphPageError):
                self.verify(mutated)

    def test_history_page_replay_tampering_and_stale_chart_fail_closed(self):
        verifier = RouterGraphHistoryVerifier(self.chart)
        for i in range(2):
            verifier.add_state_page(fixture(self.chart, self.history,
                                            state_offset=i))
        first = fixture(self.chart, self.history, history_offset=0)
        verifier.add_history_page(first)
        with self.assertRaises(RouterGraphPageError):
            verifier.add_history_page(first)
        self.assertTrue(verifier.status()["failed"])
        altered = RouterGraphHistoryVerifier(self.chart)
        for i in range(2):
            altered.add_state_page(fixture(self.chart, self.history,
                                           state_offset=i))
        forged = fixture(self.chart, self.history, history_offset=0)
        forged["history_entries"][0]["payload"] = forged[
            "history_entries"][0]["payload"].replace("supports", "refutes")
        with self.assertRaises(RouterGraphPageError):
            altered.add_history_page(forged)
        stale = RouterGraphHistoryVerifier(self.chart)
        for i in range(2):
            stale.add_state_page(fixture(self.chart, self.history,
                                         state_offset=i))
        good = fixture(self.chart, self.history)
        self.chart.snapshot_generation += 1
        with self.assertRaises(RouterGraphPageError):
            stale.add_history_page(good)


if __name__ == "__main__":
    unittest.main()
