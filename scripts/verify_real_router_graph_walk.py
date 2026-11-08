"""Live NaNROUTER MC0DB -> HERMES verified, bounded multihop walk.

The graph is a read-only overlay: current chart state is never mutated and the
router's existing MC0DB/BS0DB are the sole graph/evidence source.
"""
from __future__ import annotations

import tempfile
import time

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import verify_graph_sender_attestation
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_walk import verified_candidate_walk, VerifiedGraphWalkError
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from runner_riddle.graph_sender_attestation import sign_graph_attestation
from runner_riddle.math_objects import MathLeaf
from runner_riddle.math_store import MathStore
from runner_riddle.nan_layer_graph import NanLayerGraph
from runner_riddle.wave_store import WaveStore


def main():
    chart = Chart8MiB.genesis()
    attach_universe_root(chart)
    generation = chart.snapshot_generation
    initial = bytes(chart.bytes)
    with tempfile.TemporaryDirectory(prefix="nan-graph-walk-") as directory:
        bs, mc = directory + "/bs.sqlite", directory + "/mc.sqlite"
        WaveStore(bs, mc)
        store = MathStore(bs, mc)
        graph = NanLayerGraph(store)
        leaf = store.put_leaf(MathLeaf(
            leaf_id="nml:walk:live", kind="scalar",
            type={"kind": "scalar"}, content={"value": 11},
            provenance={"producer": "cross-repo-ci"}))
        evidence = dict(evidence_id=leaf.leaf_id, evidence_version=1,
                        evidence_digest=leaf.digest())
        def add(eid, sk, si, dk, di, layer, relation, head, tick):
            graph.propose(
                edge_id=eid, src_kind=sk, src_id=si,
                dst_kind=dk, dst_id=di, layer=layer,
                relation=relation, source_head=head, tick=tick,
                **evidence)
        add("a-removed", "head", 7, "mp", 9, 4, "supports", 7, 1)
        add("b-support", "head", 7, "mp", 9, 4, "supports", 7, 2)
        add("c-refute", "head", 7, "mp", 9, 4, "refutes", 7, 2)
        add("d-bridge", "mp", 9, "head", 10, 6, "depends_on", 10, 3)
        add("e-chain", "head", 10, "mp", 11, 9, "isa", 10, 4)
        graph.retract(edge_id="a-removed", source_head=7,
                      expected_generation=1, tick=5)

        def page(offset, history_offset=None):
            return graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                offset=offset, limit=1,
                history_offset=history_offset, history_limit=1)

        pages = [page(i) for i in range(5)]
        replay = RouterGraphHistoryVerifier(chart)
        for item in pages:
            replay.add_state_page(item)
        for index in range(6):
            replay.add_history_page(page(0, index))
        assert replay.status()["history_verified"]
        assert replay.status()["verified_history_revisions"] == 6

        signed_key = Ed25519PrivateKey.generate()
        public = signed_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw).hex()
        now = int(time.time())
        envelope = sign_graph_attestation(
            pages[0], router_id="router-walk-ci", key_id="walk-v1",
            private_key=signed_key, issued_at=now, nonce="b" * 32)
        auth = verify_graph_sender_attestation(
            replay, envelope, expected_router_id="router-walk-ci",
            expected_key_id="walk-v1", public_key_hex=public, now=now)
        assert auth["sender_authenticated"] and auth["history_verified"]

        walked = verified_candidate_walk(
            replay, pages, start=7, target=256 + 11, max_hops=3)
        path = walked["paths"][0]["edges"]
        assert [(x["src"], x["dst"]) for x in path] == [
            (7, 265), (265, 10), (10, 267)]
        assert walked["matched_candidate_edges"] == 4
        assert walked["contested_pairs"] == [
            {"src": 7, "dst": 265, "layer": 4,
             "relations": ["refutes", "supports"]}]
        assert all(x["evidence_digest"] == leaf.digest() for x in path)
        assert not walked["sender_authenticated"]  # helper never carries an auth capability
        assert not walked["evidence_integrity_verified"]
        assert not walked["native_mth_executed"]
        assert not walked["learning_admitted"]
        assert not walked["mathematical_truth_promoted"]

        try:
            verified_candidate_walk(
                replay, pages, start=7, target=267, max_edges=3)
        except VerifiedGraphWalkError:
            pass
        else:
            raise AssertionError("incomplete bounded candidate query succeeded")

        assert chart.snapshot_generation == generation
        assert bytes(chart.bytes) == initial
    print("PASS: real MC0DB revision tombstone -> full HERMES history replay -> "
          "Ed25519 pinned signature -> 3-hop candidate path & competing claims, "
          "all chart bytes unchanged, no native MTH or learning admission")


if __name__ == "__main__":
    main()
