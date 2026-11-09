"""Execute real MC0DB-derived HERMES candidate inputs inside native NaN16.

The Python code supplies verified data and a test oracle only. Exact claim
lookup and contradictory bitmask computation happen in the guest .nas kernel.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from verify_real_router_native_claim_input import main as build_real_packet


def main():
    packet = build_real_packet()
    if (packet["records_count"] != 2 or packet["word_count"] != 8
            or len(packet["words"]) != 4 * packet["records_count"]
            or any(packet[key] for key in (
                "sender_authenticated", "native_mth_executed",
                "learning_admitted", "mathematical_truth_promoted"))):
        raise AssertionError("unexpected untrusted NaN16 test packet")
    raw = struct.pack(">8I", *packet["words"])
    digest = hashlib.sha256(
        b"nanlm.nan16.graph-claim-probe/1\0"
        + packet["records_count"].to_bytes(4, "big") + raw
    ).hexdigest()
    if digest != packet["word_digest_sha256"]:
        raise AssertionError("staged native input digest mismatch")

    target = Path("router/mth/tests/hermes_real_claim_probe_test.nas")
    if not target.parent.is_dir():
        raise RuntimeError("real NaNROUTER native source checkout is required")
    records = "\n".join(
        ".word " + ",".join(str(x) for x in packet["words"][i:i+4])
        for i in range(0, len(packet["words"]), 4)
    )
    source = r'''; Generated ONLY for CI from real HERMES-replayed MC0DB records.
.org 0
JMP @hermes_real_claim_main
.include "../nan16/mth_graph_claim_probe.nas"

hermes_real_claim_main:
  MOVI r13,1
  ; Two opposing candidates for head 7 -> MP9, layer L4.
  MOVI r1,@hermes_real_records
  MOVI r2,2
  MOVI r3,7
  MOVI r4,265
  MOVI r5,4
  MOVI r6,@hermes_real_out
  CALL @mth_graph_claim_probe
  CMP r1,r13
  JMP[NE] @hermes_real_claim_fail
  MOVI r7,@hermes_real_out
  LD r8,r7,0
  MOVI r9,3
  CMP r8,r9
  JMP[NE] @hermes_real_claim_fail

  ; No candidate for MP10 at L4: absent, not refuted.
  MOVI r1,@hermes_real_records
  MOVI r2,2
  MOVI r3,7
  MOVI r4,266
  MOVI r5,4
  MOVI r6,@hermes_real_out
  CALL @mth_graph_claim_probe
  CMP r1,r13
  JMP[NE] @hermes_real_claim_fail
  MOVI r7,@hermes_real_out
  LD r8,r7,0
  CMP r8,r0
  JMP[NE] @hermes_real_claim_fail

  MOVI r1,@hermes_real_claim_pass
  CALL @hermes_real_puts
  HALT
hermes_real_claim_fail:
  MOVI r1,@hermes_real_claim_bad
  CALL @hermes_real_puts
  HALT
hermes_real_puts:
  MOVI r4,255
hermes_real_puts_word:
  LD r2,r1,0
  MOVI r3,4
hermes_real_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @hermes_real_puts_byte
  ADD r1,r1,r13
  JMP @hermes_real_puts_word
hermes_real_claim_pass:
.str "HERMES MC0DB TO NATIVE CLAIM PASS\n"
hermes_real_claim_bad:
.str "HERMES MC0DB TO NATIVE CLAIM FAIL\n"

.org 4096
hermes_real_records:
'''+records+r'''
.org 5000
hermes_real_out:
.word 0
'''
    target.write_text(source, encoding="utf-8")
    print(
        "Prepared real history-replayed input with", packet["records_count"],
        "native claim records; word_digest_sha256 =", digest,
    )


if __name__ == "__main__":
    main()
