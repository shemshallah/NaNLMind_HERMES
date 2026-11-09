"""Real MC0DB -> complete HERMES replay -> native guest -> signed result.

Everything before NaN16 is metadata authentication and input staging.
The query bitmask is READ from the actual native VM serial output, then
signed by a fresh CI-only test worker, and verified by HERMES against the SAME
chart, history, MathStore and immutable word payload. Never authorizes learning.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from nanlmind_hermes.crypto_compat import Ed25519PrivateKey, serialization
from nanlmind_hermes.router_graph_attestation import verify_graph_sender_attestation
from nanlmind_hermes.router_graph_evidence import audit_graph_revision_evidence
from nanlmind_hermes.router_graph_native_receipt import (
    _RECEIPT_DOMAIN, _canonical, SCHEMA, NativeGraphReceiptError,
    verify_native_graph_claim_receipt,
)
from verify_real_router_native_claim_input import main as build_real_graph


def _guest_source(packet: dict, query: tuple[int, int, int]) -> str:
    if packet["records_count"] != 2 or packet["words"] != [
        7, 265, 4, 1, 7, 265, 4, 2,
    ]:
        raise AssertionError("unexpected real verified MC0DB candidate words")
    records = "\n".join(
        ".word " + ",".join(str(v) for v in packet["words"][i:i + 4])
        for i in range(0, len(packet["words"]), 4)
    )
    a, b, layer = query
    return rf"""; CI-only live native guest: prints computed output, not an oracle.
.org 0
JMP @receipt_ci_main
.include "../nan16/mth_graph_claim_probe.nas"

receipt_ci_main:
  MOVI r13,1
  MOVI r1,@receipt_ci_records
  MOVI r2,2
  MOVI r3,{a}
  MOVI r4,{b}
  MOVI r5,{layer}
  MOVI r6,@receipt_ci_output
  CALL @mth_graph_claim_probe
  CMP r1,r13
  JMP[NE] @receipt_ci_error
  MOVI r1,@receipt_ci_label
  CALL @receipt_ci_puts
  MOVI r7,@receipt_ci_output
  LD r8,r7,0
  MOVI r9,48
  ADD r8,r8,r9
  PUTC r8
  MOVI r8,10
  PUTC r8
  HALT
receipt_ci_error:
  MOVI r1,@receipt_ci_bad
  CALL @receipt_ci_puts
  HALT
receipt_ci_puts:
  MOVI r4,255
receipt_ci_puts_word:
  LD r2,r1,0
  MOVI r3,4
receipt_ci_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @receipt_ci_puts_byte
  ADD r1,r1,r13
  JMP @receipt_ci_puts_word
receipt_ci_label:
.str "NATIVE_MASK="
receipt_ci_bad:
.str "NATIVE_ERROR\n"

.org 4096
receipt_ci_records:
{records}
.org 5000
receipt_ci_output:
.word 0
"""


def _run_real_guest(packet, query) -> tuple[int, str]:
    root = Path("router/mth/tests")
    nmk = Path("nan0s/tools/nmk.awk")
    vm = Path("nan0s/aarch64/android/nan_cli")
    if not root.is_dir() or not nmk.is_file() or not vm.is_file():
        raise RuntimeError("pinned NaNROUTER and NaN0S native CI checkouts required")
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="hermes_receipt_",
            suffix=".nas", dir=root, delete=False) as handle:
        path = Path(handle.name)
        handle.write(_guest_source(packet, query))
    try:
        asm = subprocess.run(
            ["awk", "-f", str(nmk), "-v", "CMD=emit", str(path)],
            capture_output=True, text=True, check=True,
        )
        image = asm.stdout.encode("utf-8")
        if not image:
            raise AssertionError("empty assembled native image")
        with tempfile.TemporaryDirectory(prefix="hermes-guest-receipt-") as td:
            binary = Path(td) / "claim.nan"
            binary.write_bytes(image)
            result = subprocess.run(
                [str(vm), "--img", str(binary), "--steps", "2000000",
                 "--serial"], capture_output=True, text=True, check=True,
            )
        if "NATIVE_ERROR" in result.stdout:
            raise AssertionError("native guest rejected verified snapshot")
        matches = re.findall(r"^NATIVE_MASK=([0-3])$", result.stdout, re.M)
        if len(matches) != 1:
            raise AssertionError("no unique valid native mask in guest serial output")
        return int(matches[0]), hashlib.sha256(image).hexdigest()
    finally:
        path.unlink(missing_ok=True)


def _check_same_chart(**ctx):
    chart = ctx["chart"]
    before = bytes(chart.bytes)
    verifier = ctx["verifier"]
    pages, history = ctx["state_pages"], ctx["history_pages"]
    sender_attestation = ctx["sender_attestation"]
    now = ctx["now"]
    packet = ctx["packet"]
    store = ctx["math_store"]
    sender = verify_graph_sender_attestation(
        verifier, sender_attestation,
        expected_router_id="native-graph-router-ci",
        expected_key_id="native-graph-key-ci",
        public_key_hex=ctx["router_public_key_hex"], now=now,
    )
    evidence = audit_graph_revision_evidence(
        verifier, pages, history, resolve_trusted_leaf=store.get_leaf)
    key = Ed25519PrivateKey.generate()  # CI-only test worker, NOT a deployment pin.
    worker_public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw).hex()
    lease_digest = hashlib.sha256(
        b"ci-only-untrusted-nonadmission-lease-sentinel/1").hexdigest()
    for query, oracle in (((7, 265, 4), 3), ((7, 266, 4), 0)):
        mask, image_sha = _run_real_guest(packet, query)
        if mask != oracle:
            raise AssertionError("actual native result differs from graph fixture")
        receipt_body = {
            "schema": SCHEMA, "worker_id": "ci-native-worker",
            "key_id": "ci-ephemeral-worker-key",
            "graph_envelope_sha256": packet["envelope_sha256"],
            "word_digest_sha256": packet["word_digest_sha256"],
            "provenance_sha256": packet["provenance_sha256"],
            "graph_commitment_sha256": packet["graph_commitment_sha256"],
            "history_sha256": packet["history_sha256"],
            "chart_generation": packet["chart_generation"],
            "router_attestation_sha256": sender["attestation_sha256"],
            "evidence_reference_digest_sha256":
                evidence["evidence_reference_digest_sha256"],
            "guest_image_sha256": image_sha,
            "rule_epoch": 17, "lease_sha256": lease_digest,
            "nonce": "e" * 32 if mask == 3 else "f" * 32,
            "query_src": query[0], "query_dst": query[1],
            "query_layer": query[2], "guest_status": 1,
            "result_mask": mask, "issued_at": now,
        }
        receipt = {
            **receipt_body,
            "signature_hex": key.sign(
                _RECEIPT_DOMAIN + _canonical(receipt_body)).hex(),
        }
        kwargs = {
            "expected_router_id": "native-graph-router-ci",
            "expected_router_key_id": "native-graph-key-ci",
            "pinned_router_public_key_hex": ctx["router_public_key_hex"],
            "resolve_trusted_leaf": store.get_leaf,
            "expected_worker_id": "ci-native-worker",
            "expected_worker_key_id": "ci-ephemeral-worker-key",
            "pinned_worker_public_key_hex": worker_public,
            "expected_guest_image_sha256": image_sha,
            "expected_rule_epoch": 17,
            "expected_lease_sha256": lease_digest,
            "expected_nonce": receipt_body["nonce"],
            "expected_query": query, "now": now,
        }
        report = verify_native_graph_claim_receipt(
            verifier, pages, history, sender_attestation, receipt, **kwargs)
        if (report["candidate_mask"] != mask
                or not report["native_result_worker_attested"]
                or any(report[k] for k in (
                    "native_execution_independently_verified",
                    "nonce_replay_protected", "lease_dispatch_authorized",
                    "native_admission_verified", "learning_admitted",
                    "mathematical_truth_promoted"))):
            raise AssertionError("native result not bound or falsely promoted")
        forged = dict(receipt)
        forged["result_mask"] = (mask + 1) % 4
        try:
            verify_native_graph_claim_receipt(
                verifier, pages, history, sender_attestation, forged, **kwargs)
        except NativeGraphReceiptError:
            pass
        else:
            raise AssertionError("forged native serial result was accepted")
    if bytes(chart.bytes) != before:
        raise AssertionError("native receipt CI mutated resident HERMES chart")
    print("PASS: MC0DB -> pinned sender -> historical MathLeaf -> actual "
          "NaN16 guest serial 3/0 -> signed worker receipts -> HERMES exact "
          "verification; no nonce consumption, training or truth promotion")


if __name__ == "__main__":
    build_real_graph(on_verified_graph=_check_same_chart)
