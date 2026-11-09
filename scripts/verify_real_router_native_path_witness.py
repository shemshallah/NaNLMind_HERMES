"""CI: real MC0DB graph -> complete HERMES replay -> native NaN16 path witness.

The host stages independently verified data and a proposed path. The NaN16
guest computes relation masks; this fixture cannot authorize learning.
"""
import copy
import hashlib
import struct
import tempfile
import time
from pathlib import Path

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import verify_graph_sender_attestation
from nanlmind_hermes.router_graph_evidence import audit_graph_revision_evidence
from nanlmind_hermes.router_graph_history import RouterGraphHistoryVerifier
from nanlmind_hermes.router_graph_native_input import pack_native_graph_claim_input
from nanlmind_hermes.router_graph_page import RouterGraphPageError
from nanlmind_hermes.router_graph_walk import verified_candidate_walk
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
    before = bytes(chart.bytes)
    with tempfile.TemporaryDirectory(prefix="real-native-path-") as tmp:
        bs, mc = tmp + "/bs.sqlite", tmp + "/mc.sqlite"
        WaveStore(bs, mc)
        store = MathStore(bs, mc)
        graph = NanLayerGraph(store)
        leaf = store.put_leaf(MathLeaf(
            leaf_id="nml:path:evidence", kind="scalar",
            type={"kind": "scalar"}, content={"value": 17},
            provenance={"producer": "real-multihop-ci"},
        ))
        evidence = dict(
            evidence_id=leaf.leaf_id, evidence_version=1,
            evidence_digest=leaf.digest(),
        )

        def add(name, src_kind, src_id, dst_kind, dst_id, layer, relation, owner):
            graph.propose(
                edge_id=name, src_kind=src_kind, src_id=src_id,
                dst_kind=dst_kind, dst_id=dst_id, layer=layer,
                relation=relation, source_head=owner, tick=1, **evidence,
            )

        add("a-support", "head", 7, "mp", 9, 4, "supports", 7)
        add("b-refute", "head", 7, "mp", 9, 4, "refutes", 7)
        add("c-next", "mp", 9, "head", 9, 5, "supports", 9)
        add("d-last", "head", 9, "mp", 11, 8, "refutes", 9)
        add("e-tombstone", "head", 7, "mp", 10, 4, "supports", 7)
        graph.retract(edge_id="e-tombstone", source_head=7,
                      expected_generation=1, tick=2)
        add("f-isa", "head", 7, "mp", 13, 4, "isa", 7)

        def page(offset=0, history_offset=None):
            return graph.master_graph_page(
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                offset=offset, limit=1,
                history_offset=history_offset, history_limit=1,
            )

        state = [page(i) for i in range(6)]
        history = [page(0, i) for i in range(7)]
        verifier = RouterGraphHistoryVerifier(chart)
        for item in state:
            verifier.add_state_page(item)
        for item in history:
            verifier.add_history_page(item)
        assert verifier.status()["history_verified"]

        signer = Ed25519PrivateKey.generate()
        key_hex = signer.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()
        now = int(time.time())
        attestation = sign_graph_attestation(
            state[0], router_id="native-path-ci-router",
            key_id="native-path-ci-key", private_key=signer,
            issued_at=now, nonce="e" * 32,
        )
        signature = verify_graph_sender_attestation(
            verifier, attestation,
            expected_router_id="native-path-ci-router",
            expected_key_id="native-path-ci-key",
            public_key_hex=key_hex, now=now,
        )
        assert signature["sender_authenticated"]
        audit = audit_graph_revision_evidence(
            verifier, state, history, resolve_trusted_leaf=store.get_leaf,
        )
        assert audit["evidence_digest_integrity_verified"]
        assert audit["historical_revisions_checked"] == 7
        assert not audit["learning_admitted"]

        packet = pack_native_graph_claim_input(verifier, state, history)
        assert packet["records_count"] == 4
        records = [packet["words"][i:i+4] for i in range(0, 16, 4)]
        assert sorted(records) == sorted([
            [7, 265, 4, 1], [7, 265, 4, 2],
            [265, 9, 5, 1], [9, 267, 8, 2],
        ])
        wire = struct.pack(">16I", *packet["words"])
        digest = hashlib.sha256(
            b"nanlm.nan16.graph-claim-probe/1\0"
            + (4).to_bytes(4, "big") + wire
        ).hexdigest()
        assert digest == packet["word_digest_sha256"]
        walk = verified_candidate_walk(
            verifier, state, start=7, target=267,
            max_hops=3, max_edges=1024,
        )
        assert walk["history_verified"]
        assert walk["graph_commitment_sha256"] == packet[
            "graph_commitment_sha256"]
        assert walk["history_sha256"] == packet["history_sha256"]
        assert len(walk["paths"]) == 1
        hops = [[e["src"], e["dst"], e["layer"]]
                for e in walk["paths"][0]["edges"]]
        assert hops == [[7, 265, 4], [265, 9, 5], [9, 267, 8]]
        assert any(c["src"] == 7 and c["dst"] == 265
                   for c in walk["contested_pairs"])
        for label in ("sender_authenticated", "native_mth_executed",
                      "native_admission_verified", "learning_admitted",
                      "mathematical_truth_promoted"):
            assert packet[label] is False

        tampered = copy.deepcopy(history)
        item = tampered[0]["history_entries"][0]
        item["payload"] = item["payload"].replace("supports", "refutes")
        for bad in (tampered, history[:-1]):
            try:
                pack_native_graph_claim_input(verifier, state, bad)
            except RouterGraphPageError:
                pass
            else:
                raise AssertionError("unverified history reached the guest")
        assert before == bytes(chart.bytes)

    target = Path("router/mth/tests/hermes_real_graph_path_witness_test.nas")
    kernel = target.parent.parent / "nan16/mth_graph_path_witness.nas"
    if not target.parent.is_dir() or not kernel.is_file():
        raise RuntimeError("pinned merged NaNROUTER #110 required")
    rows = "\n".join(".word " + ",".join(map(str, row)) for row in records)
    route = "\n".join(".word " + ",".join(map(str, row)) for row in hops)
    asm = r'''; Generated from real signed/audited MC0DB history. No host result.
.org 0
JMP @real_path_main
.include "../nan16/mth_graph_path_witness.nas"
real_path_main:
  MOVI r13,1
  MOVI r1,@real_records
  MOVI r2,4
  MOVI r3,@real_hops
  MOVI r4,3
  MOVI r5,@real_output
  CALL @mth_graph_path_witness
  CMP r1,r13
  JMP[NE] @real_path_fail
  MOVI r6,@real_output
  LD r7,r6,0
  MOVI r8,3
  CMP r7,r8
  JMP[NE] @real_path_fail
  LD r7,r6,1
  CMP r7,r13
  JMP[NE] @real_path_fail
  LD r7,r6,2
  MOVI r8,2
  CMP r7,r8
  JMP[NE] @real_path_fail
  LD r7,r6,3
  MOVI r8,99
  CMP r7,r8
  JMP[NE] @real_path_fail

  ; Missing third candidate must fail without touching output.
  MOVI r6,@real_output
  MOVI r7,77
  ST r7,r6,0
  ST r7,r6,1
  ST r7,r6,2
  MOVI r6,@real_hops
  MOVI r7,268
  ST r7,r6,7
  MOVI r1,@real_records
  MOVI r2,4
  MOVI r3,@real_hops
  MOVI r4,3
  MOVI r5,@real_output
  CALL @mth_graph_path_witness
  CMP r1,r0
  JMP[NE] @real_path_fail
  MOVI r6,@real_output
  MOVI r8,77
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @real_path_fail
  LD r7,r6,1
  CMP r7,r8
  JMP[NE] @real_path_fail
  LD r7,r6,2
  CMP r7,r8
  JMP[NE] @real_path_fail

  MOVI r1,@real_path_pass
  CALL @real_path_puts
  HALT
real_path_fail:
  MOVI r1,@real_path_bad
  CALL @real_path_puts
  HALT
real_path_puts:
  MOVI r4,255
real_path_puts_word:
  LD r2,r1,0
  MOVI r3,4
real_path_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @real_path_puts_byte
  ADD r1,r1,r13
  JMP @real_path_puts_word
real_path_pass:
.str "HERMES REAL MC0DB PATH WITNESS PASS\n"
real_path_bad:
.str "HERMES REAL MC0DB PATH WITNESS FAIL\n"
.org 4096
real_records:
''' + rows + r'''
.org 5000
real_hops:
''' + route + r'''
.org 6000
real_output:
.word 99,99,99,99,99,99,99,99
'''
    target.write_text(asm, encoding="utf-8")
    print("PASS: real signed MathLeaf-audited MC0DB replay staged",
          len(records), "candidate records and", len(hops),
          "native witness hops; no learning admission")


if __name__ == "__main__":
    main()
