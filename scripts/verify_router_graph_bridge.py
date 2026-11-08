"""Run the actual NaNROUTER -> HERMES graph commitment contract.

GitHub Actions checks out NaNROUTER into ./router. There is one HERMES chart;
the router's existing MC0DB/BS0DB files only act as the canonical graph source.
"""
from __future__ import annotations

import tempfile

from nanlmind_hermes.mirror_topology import TOPOLOGY_DIGEST as HERMES_TOPOLOGY
from nanlmind_hermes.router_graph_page import (
    RouterGraphPageError, RouterGraphPageVerifier,
)
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from runner_riddle.math_objects import MathLeaf
from runner_riddle.math_store import MathStore
from runner_riddle.mirror_point_register import TOPOLOGY_DIGEST as ROUTER_TOPOLOGY
from runner_riddle.nan_layer_graph import NanLayerGraph
from runner_riddle.wave_store import WaveStore


def assert_denied(action):
    try:
        action()
    except RouterGraphPageError:
        return
    raise AssertionError("unsafe graph projection was accepted")


def main():
    assert HERMES_TOPOLOGY == ROUTER_TOPOLOGY, "full MP topology digests differ"
    chart = Chart8MiB.genesis()
    root = attach_universe_root(chart)
    assert root["topology_sha256"] == ROUTER_TOPOLOGY
    with tempfile.TemporaryDirectory(prefix="nan-router-hermes-") as folder:
        bs, mc = folder + "/bs.sqlite", folder + "/mc.sqlite"
        WaveStore(bs, mc)
        store = MathStore(bs, mc)
        graph = NanLayerGraph(store)
        leaf = store.put_leaf(MathLeaf(
            leaf_id="nml:bridge:evidence", kind="scalar",
            type={"kind": "scalar"}, content={"value": 7},
            provenance={"producer": "cross-repo-ci"}))
        evidence = {
            "evidence_id": leaf.leaf_id,
            "evidence_version": 1,
            "evidence_digest": leaf.digest(),
        }
        graph.propose(edge_id="bridge-1", src_kind="head", src_id=7,
                      dst_kind="mp", dst_id=9, layer=4,
                      relation="supports", source_head=7, tick=2, **evidence)
        graph.propose(edge_id="bridge-2", src_kind="mp", src_id=9,
                      dst_kind="head", dst_id=10, layer=9,
                      relation="refutes", source_head=10, tick=2, **evidence)
        graph.retract(edge_id="bridge-1", source_head=7,
                      expected_generation=1, tick=3)

        def projection(offset):
            return graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                offset=offset, limit=1)

        first, second = projection(0), projection(1)
        assert first["graph_commitment_sha256"] == second["graph_commitment_sha256"]
        verifier = RouterGraphPageVerifier(chart)
        partial = verifier.add_page(first)
        assert not partial["state_verified"]
        complete = verifier.add_page(second)
        assert complete["state_verified"]
        assert complete["verified_entries"] == 2
        assert not complete["sender_authenticated"]
        assert not complete["native_mth_executed"]

        replay = RouterGraphPageVerifier(chart)
        replay.add_page(first)
        assert_denied(lambda: replay.add_page(first))
        altered = projection(1)
        altered["entries"][0]["relation"] = "supports"
        bad = RouterGraphPageVerifier(chart)
        bad.add_page(first)
        assert_denied(lambda: bad.add_page(altered))
        mixed = RouterGraphPageVerifier(chart)
        mixed.add_page(first)
        chart.snapshot_generation += 1
        assert_denied(lambda: mixed.add_page(second))
    print("PASS: real NaNROUTER graph -> HERMES chart-root-bound "
          "2-page verification, tombstone, tamper, replay, stale epoch")


if __name__ == "__main__":
    main()
