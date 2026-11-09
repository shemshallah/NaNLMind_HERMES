"""Live NaNROUTER MC0DB -> HERMES full replay -> native guest input.

Runs without creating a second chart or executing a training update.
"""
import copy
import tempfile
import time

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import verify_graph_sender_attestation
from nanlmind_hermes.router_graph_evidence import audit_graph_revision_evidence

from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_native_input import pack_native_graph_claim_input
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from runner_riddle.graph_sender_attestation import sign_graph_attestation
from runner_riddle.math_objects import MathLeaf
from runner_riddle.math_store import MathStore
from runner_riddle.nan_layer_graph import NanLayerGraph
from runner_riddle.wave_store import WaveStore


def main(*, on_verified_graph=None):
    """Build one real graph; optional trusted CI hook executes before DB closes."""
    if on_verified_graph is not None and not callable(on_verified_graph):
        raise TypeError("trusted graph handoff hook must be callable")
    chart = Chart8MiB.genesis()
    attach_universe_root(chart)
    before = bytes(chart.bytes)
    with tempfile.TemporaryDirectory(prefix="nan-native-claim-input-") as path:
        bs, mc = path + "/bs.sqlite", path + "/mc.sqlite"
        WaveStore(bs, mc)
        store = MathStore(bs, mc)
        graph = NanLayerGraph(store)
        leaf = store.put_leaf(MathLeaf(
            leaf_id="nml:claim:real", kind="scalar",
            type={"kind": "scalar"}, content={"value": 17},
            provenance={"producer": "real-router-native-input-ci"}))
        evidence = {
            "evidence_id": leaf.leaf_id, "evidence_version": 1,
            "evidence_digest": leaf.digest(),
        }

        def add(eid, relation, dst=9, tick=1):
            graph.propose(
                edge_id=eid, src_kind="head", src_id=7,
                dst_kind="mp", dst_id=dst, layer=4,
                relation=relation, source_head=7, tick=tick,
                **evidence)

        add("a-support", "supports")
        add("b-refute", "refutes")
        add("c-non-probe", "isa")
        add("d-tombstone", "supports", dst=11)
        graph.retract(
            edge_id="d-tombstone", source_head=7,
            expected_generation=1, tick=2)

        def page(offset=0, history_offset=None):
            return graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                offset=offset, limit=1,
                history_offset=history_offset, history_limit=1)

        state_pages = [page(i) for i in range(4)]
        history_pages = [page(0, i) for i in range(5)]
        verifier = RouterGraphHistoryVerifier(chart)
        for item in state_pages:
            verifier.add_state_page(item)
        for item in history_pages:
            verifier.add_history_page(item)
        assert verifier.status()["history_verified"]

        # Distinct trust checks: local operator-pinned router signature, then
        # immutable MathStore evidence for EVERY revision including tombstones.
        signer = Ed25519PrivateKey.generate()
        public_hex = signer.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        now = int(time.time())
        attestation = sign_graph_attestation(
            state_pages[0], router_id="native-graph-router-ci",
            key_id="native-graph-key-ci", private_key=signer,
            issued_at=now, nonce="d" * 32,
        )
        authenticated = verify_graph_sender_attestation(
            verifier, attestation,
            expected_router_id="native-graph-router-ci",
            expected_key_id="native-graph-key-ci",
            public_key_hex=public_hex, now=now,
        )
        assert authenticated["sender_authenticated"]
        evidence_audit = audit_graph_revision_evidence(
            verifier, state_pages, history_pages,
            resolve_trusted_leaf=store.get_leaf,
        )
        assert evidence_audit["evidence_digest_integrity_verified"]
        assert evidence_audit["historical_revisions_checked"] == 5
        assert not evidence_audit["learning_admitted"]

        packet = pack_native_graph_claim_input(
            verifier, state_pages, history_pages)
        assert packet["words"] == [7, 265, 4, 1, 7, 265, 4, 2]
        assert packet["records_count"] == 2
        assert [p["evidence_digest"] for p in packet["provenance"]] == [
            leaf.digest(), leaf.digest()]
        assert packet["history_sha256"] == state_pages[0]["history_sha256"]
        assert not packet["sender_authenticated"]
        assert not packet["native_mth_executed"]
        assert not packet["learning_admitted"]

        tampered = copy.deepcopy(history_pages)
        tampered[0]["history_entries"][0]["payload"] = tampered[0][
            "history_entries"][0]["payload"].replace("supports", "refutes")
        try:
            pack_native_graph_claim_input(
                verifier, state_pages, tampered)
        except RouterGraphPageError:
            pass
        else:
            raise AssertionError("tampered history staged native words")

        assert before == bytes(chart.bytes)
        # The callback is a trusted local CI harness, not network input. It
        # receives the SAME open MathStore, verifier, signed graph and packet.
        # Normal callers retain the original packet-only behavior.
        if on_verified_graph is not None:
            on_verified_graph(
                chart=chart, verifier=verifier, state_pages=state_pages,
                history_pages=history_pages, sender_attestation=attestation,
                router_public_key_hex=public_hex, math_store=store,
                packet=packet, now=now)
        assert before == bytes(chart.bytes)

    print("PASS: real MC0DB candidate graph + retraction -> HERMES history "
          "replay -> pinned sender + immutable MathLeaf audit -> bounded NaN16 U32 input, opposing claims intact, "
          "untampered chart, NO native execution or learning admission")
    return packet


if __name__ == "__main__":
    main()
