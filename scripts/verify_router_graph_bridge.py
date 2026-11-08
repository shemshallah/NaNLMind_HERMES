"""Run the actual NaNROUTER -> HERMES graph commitment contract.

GitHub Actions checks out NaNROUTER into ./router. There is one HERMES chart;
the router's existing MC0DB/BS0DB files only act as the canonical graph source.
"""
from __future__ import annotations

import tempfile
import time

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import verify_graph_sender_attestation
from runner_riddle.graph_sender_attestation import sign_graph_attestation

from nanlmind_hermes.mirror_topology import TOPOLOGY_DIGEST as HERMES_TOPOLOGY
from nanlmind_hermes.router_graph_page import (
    RouterGraphPageError, RouterGraphPageVerifier,
)
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
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

        # Independently replay all three actual MC0DB revisions, including
        # the retraction, against the already verified present-state graph.
        replay_history = RouterGraphHistoryVerifier(chart)
        replay_history.add_state_page(first)
        replay_history.add_state_page(second)
        for index in range(3):
            history_page = graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                limit=1, history_offset=index, history_limit=1,
            )
            status = replay_history.add_history_page(history_page)
        assert status["history_verified"]
        assert status["verified_history_revisions"] == 3
        assert not status["sender_authenticated"]
        assert not status["native_mth_executed"]
        assert not status["learning_admitted"]

        # Real router signer -> actual HERMES recipient. The expected public
        # key and identity are injected independently, never from the packet.
        signing_key = Ed25519PrivateKey.generate()
        trusted_pub_hex = signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw).hex()
        now = int(time.time())
        signed = sign_graph_attestation(
            first, router_id="router-ci-0", key_id="ci-key-1",
            private_key=signing_key, issued_at=now, nonce="a" * 32)
        signed_status = verify_graph_sender_attestation(
            replay_history, signed, expected_router_id="router-ci-0",
            expected_key_id="ci-key-1", public_key_hex=trusted_pub_hex, now=now)
        assert signed_status["sender_authenticated"]
        assert not signed_status["nonce_replay_protected"]
        assert not signed_status["native_mth_executed"]
        assert not signed_status["learning_admitted"]
        forged = dict(signed, history_sha256="0" * 64)
        assert_denied(lambda: verify_graph_sender_attestation(
            replay_history, forged, expected_router_id="router-ci-0",
            expected_key_id="ci-key-1", public_key_hex=trusted_pub_hex, now=now))
        assert_denied(lambda: verify_graph_sender_attestation(
            replay_history, signed, expected_router_id="router-ci-1",
            expected_key_id="ci-key-1", public_key_hex=trusted_pub_hex, now=now))

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
          "2-page state, 3-revision history replay, tombstone, "
          "tamper, Ed25519 sender authentication, stale epoch; no MTH promotion")


if __name__ == "__main__":
    main()
