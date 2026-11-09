.org 0
JMP @pf_test_main
.include "../nan16/mth_preflight_gate.nas"

pf_test_main:
  MOVI r13,1
  MOVI r4,@pf_candidate
  MOVI r5,@pf_rule
  MOVI r6,@pf_result
  ADD r1,r4,r0
  ADD r2,r5,r0
  ADD r3,r6,r0
  CALL @mth_preflight_gate
  CMP r1,r0
  JMP[NE] @pf_test_fail

  ; A changed candidate epoch must fail closed as STALE_EPOCH (2).
  MOVI r4,@pf_candidate
  MOVI r5,@pf_rule
  MOVI r6,@pf_result
  MOVI r7,6
  ST r7,r4,9
  ADD r1,r4,r0
  ADD r2,r5,r0
  ADD r3,r6,r0
  CALL @mth_preflight_gate
  MOVI r7,2
  CMP r1,r7
  JMP[NE] @pf_test_fail

  MOVI r1,@pf_pass
  CALL @pf_puts
  HALT
pf_test_fail:
  MOVI r1,@pf_fail
  CALL @pf_puts
  HALT

pf_puts:
  MOVI r4,255
pf_puts_word:
  LD r2,r1,0
  MOVI r3,4
pf_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @pf_puts_byte
  ADD r1,r1,r13
  JMP @pf_puts_word

pf_pass:
.str "MTH PREFLIGHT GATE PASS\n"
pf_fail:
.str "MTH PREFLIGHT GATE FAIL\n"

.org 4096
; Candidate: ABI, epoch, lhs/rhs/out types, lhs/rhs domains,
; lhs/rhs dimensions, candidate epoch, required/available caps,
; proof level, enabled, then four rule digest words.
pf_candidate:
.word 1,7,11,12,13,3,3,8,8,7,3,7,2,1,100,101,102,103,104,105,106,107
; Rule: ABI, epoch, signature, domain, shape policy, minimum proof,
; enabled, then four digest words.
pf_rule:
.word 1,7,11,12,13,3,1,1,1,100,101,102,103,104,105,106,107
.org 5000
pf_result:
.word 0
