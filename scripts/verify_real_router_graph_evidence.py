"""Actual NaNROUTER MathStore -> HERMES independent history/evidence audit.

The trusted resolver is bound here by the test harness, not selected from
the peer's graph projection or Ed25519 envelope.
"""
from __future__ import annotations

import tempfile

from nanlmind_hermes.router_graph_evidence import (
    GraphEvidenceIntegrityError, audit_graph_revision_evidence,
)
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from runner_riddle.math_objects import MathLeaf
from runner_riddle.math_store import MathStore
from runner_riddle.nan_layer_graph import NanLayerGraph
from runner_riddle.wave_store import WaveStore


def main():
    chart = Chart8MiB.genesis()
    attach_universe_root(chart)
    chart_bytes = bytes(chart.bytes)
    chart_gen = chart.snapshot_generation
    with tempfile.TemporaryDirectory(prefix="nan-graph-evidence-") as folder:
        bs, mc = folder + "/bs.sqlite", folder + "/mc.sqlite"
        WaveStore(bs, mc)
        store = MathStore(bs, mc)
        graph = NanLayerGraph(store)
        leaf_a = store.put_leaf(MathLeaf(
            leaf_id="nml:graph:evidence-a", kind="scalar",
            type={"kind": "scalar"}, content={"value": 7},
            provenance={"producer": "trusted-store-crossrepo-ci"}))
        leaf_b = store.put_leaf(MathLeaf(
            leaf_id="nml:graph:evidence-b", kind="scalar",
            type={"kind": "scalar"}, content={"value": 12},
            provenance={"producer": "trusted-store-crossrepo-ci"}))

        def propose(eid, dst, evidence, tick, expected=0):
            return graph.propose(
                edge_id=eid, src_kind="head", src_id=7,
                dst_kind="mp", dst_id=dst, layer=4,
                relation="supports", source_head=7,
                expected_generation=expected, tick=tick,
                evidence_id=evidence.leaf_id, evidence_version=evidence.version,
                evidence_digest=evidence.digest())

        propose("a", 9, leaf_a, 1)
        graph.retract(edge_id="a", source_head=7, expected_generation=1, tick=2)
        propose("b", 11, leaf_b, 3)
        propose("b", 11, leaf_b, 4, expected=1)

        def project(offset, history_offset=None):
            return graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                offset=offset, limit=1,
                history_offset=history_offset, history_limit=1)

        state_pages = [project(i) for i in range(2)]
        history_pages = [project(0, i) for i in range(4)]
        replay = RouterGraphHistoryVerifier(chart)
        for page in state_pages:
            replay.add_state_page(page)
        for page in history_pages:
            replay.add_history_page(page)
        assert replay.status()["history_verified"]

        report = audit_graph_revision_evidence(
            replay, state_pages, history_pages,
            resolve_trusted_leaf=store.get_leaf)
        assert report["historical_revisions_checked"] == 4
        assert report["distinct_leaf_versions_revalidated"] == 2
        assert report["evidence_digest_integrity_verified"]
        assert not report["evidence_provenance_authenticated"]
        assert not report["evidence_mathematically_proven"]
        assert not report["native_mth_executed"]
        assert not report["learning_admitted"]
        assert not report["mathematical_truth_promoted"]

        def wrong_version(_leaf_id, _version):
            return store.get_leaf(leaf_a.leaf_id, leaf_a.version)
        try:
            audit_graph_revision_evidence(
                replay, state_pages, history_pages,
                resolve_trusted_leaf=wrong_version)
        except GraphEvidenceIntegrityError:
            pass
        else:
            raise AssertionError("wrong trusted evidence version was accepted")

        assert chart.snapshot_generation == chart_gen
        assert bytes(chart.bytes) == chart_bytes

    print("PASS: 4 actual graph revisions / 2 immutable MathLeaf versions "
          "revalidated by independent HERMES replay against trusted existing "
          "MC0DB+BS0DB, no chart/head writes, no learning authorization")


if __name__ == "__main__":
    main()
