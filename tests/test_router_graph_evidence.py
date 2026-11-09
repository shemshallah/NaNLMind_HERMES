"""Fail-closed evidence integrity audit over complete real chart-bound revision replay."""
import copy
import unittest

from nanlmind_hermes.router_graph_evidence import (
    GraphEvidenceIntegrityError, audit_graph_revision_evidence,
)
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from test_router_graph_history import edge, fixture


class Leaf:
    def __init__(self, leaf_id="nml:evidence:1", version=1,
                 content_digest="f" * 64):
        self.leaf_id = leaf_id
        self.version = version
        self.content_digest = content_digest

    def digest(self):
        return self.content_digest


class GraphEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.chart = Chart8MiB.genesis()
        attach_universe_root(self.chart)
        self.revisions = [
            edge("a", 1, "candidate", 1),
            edge("a", 2, "retracted", 2),
            edge("b", 1, "candidate", 3),
        ]
        self.pages = [fixture(self.chart, self.revisions, state_offset=i)
                      for i in range(2)]
        self.history_pages = [
            fixture(self.chart, self.revisions, history_offset=i)
            for i in range(3)
        ]
        self.verifier = RouterGraphHistoryVerifier(self.chart)
        for page in self.pages:
            self.verifier.add_state_page(page)
        for page in self.history_pages:
            self.verifier.add_history_page(page)
        self.assertTrue(self.verifier.status()["history_verified"])

    def audit(self, *, resolver=None, pages=None, history=None, verifier=None):
        return audit_graph_revision_evidence(
            self.verifier if verifier is None else verifier,
            self.pages if pages is None else pages,
            self.history_pages if history is None else history,
            resolve_trusted_leaf=resolver or (lambda _id, _v: Leaf()),
        )

    def test_every_revision_checked_distinct_reference_deduped(self):
        report = self.audit()
        self.assertTrue(report["evidence_digest_integrity_verified"])
        self.assertEqual(report["historical_revisions_checked"], 3)
        self.assertEqual(report["distinct_leaf_versions_revalidated"], 1)
        self.assertEqual(len(report["evidence_reference_digest_sha256"]), 64)
        for flag in (
            "evidence_provenance_authenticated", "sender_authenticated",
            "evidence_mathematically_proven", "native_mth_executed",
            "learning_admitted", "mathematical_truth_promoted",
        ):
            self.assertFalse(report[flag], flag)

    def test_missing_mismatched_or_wrong_version_leaf_refused(self):
        for provider in (
            lambda _id, _v: None,
            lambda _id, _v: Leaf(content_digest="0" * 64),
            lambda _id, _v: Leaf(leaf_id="nml:other"),
            lambda _id, _v: Leaf(version=2),
        ):
            with self.subTest(provider=provider), self.assertRaises(
                    GraphEvidenceIntegrityError):
                self.audit(resolver=provider)

    def test_history_tampering_after_original_replay_rejected(self):
        tampered = copy.deepcopy(self.history_pages)
        tampered[0]["history_entries"][0]["payload"] = (
            tampered[0]["history_entries"][0]["payload"].replace(
                "supports", "refutes"))
        with self.assertRaises(RouterGraphPageError):
            self.audit(history=tampered)
        with self.assertRaises(GraphEvidenceIntegrityError):
            self.audit(history=self.history_pages[:1])
        with self.assertRaises(RouterGraphPageError):
            self.audit(pages=self.pages[:1])

    def test_trusted_resolver_cannot_be_supplied_by_missing_history(self):
        partial = RouterGraphHistoryVerifier(self.chart)
        for page in self.pages:
            partial.add_state_page(page)
        with self.assertRaises(GraphEvidenceIntegrityError):
            self.audit(verifier=partial)
        with self.assertRaises(GraphEvidenceIntegrityError):
            audit_graph_revision_evidence(
                self.verifier, self.pages, self.history_pages,
                resolve_trusted_leaf=None)

    def test_stale_chart_epoch_denied(self):
        self.chart.snapshot_generation += 1
        with self.assertRaises(RouterGraphPageError):
            self.audit()


if __name__ == "__main__":
    unittest.main()
