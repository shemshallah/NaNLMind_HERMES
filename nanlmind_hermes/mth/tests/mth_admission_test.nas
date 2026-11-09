.org 0
JMP @mth_admit_test_main
.include "../nan16/mth_preflight_gate.nas"
.include "../nan16/mth_preflight.nas"
.include "../nan16/mth_admission.nas"

mth_admit_test_main:
  MOVI r13,1
  MOVI r12,@mth_admit_scratch

  ; Both rule/candidate and typed-object checks pass.
  MOVI r1,@mth_admit_candidate
  MOVI r2,@mth_admit_rule
  MOVI r3,@mth_admit_left
  MOVI r4,@mth_admit_right
  MOVI r5,@mth_admit_contract
  MOVI r6,1
  MOVI r7,@mth_admit_output
  CALL @mth_admit
  CMP r1,r0
  JMP[NE] @mth_admit_fail_1

  ; A stale candidate epoch is denied by the first gate (16 + 2).
  MOVI r8,@mth_admit_candidate
  MOVI r9,6
  ST r9,r8,9
  MOVI r1,@mth_admit_candidate
  MOVI r2,@mth_admit_rule
  MOVI r3,@mth_admit_left
  MOVI r4,@mth_admit_right
  MOVI r5,@mth_admit_contract
  MOVI r6,1
  MOVI r7,@mth_admit_output
  CALL @mth_admit
  MOVI r9,18
  CMP r1,r9
  JMP[NE] @mth_admit_fail_2

  ; Restore epoch, then create a typed shape mismatch. The second gate
  ; rejects it (32 + 6) and leaves output metadata untouched.
  MOVI r8,@mth_admit_candidate
  MOVI r9,7
  ST r9,r8,9
  MOVI r8,@mth_admit_right
  MOVI r9,2
  ST r9,r8,2
  MOVI r8,@mth_admit_output
  MOVI r9,99
  ST r9,r8,0
  MOVI r1,@mth_admit_candidate
  MOVI r2,@mth_admit_rule
  MOVI r3,@mth_admit_left
  MOVI r4,@mth_admit_right
  MOVI r5,@mth_admit_contract
  MOVI r6,1
  MOVI r7,@mth_admit_output
  CALL @mth_admit
  MOVI r9,38
  CMP r1,r9
  JMP[NE] @mth_admit_fail_3
  MOVI r8,@mth_admit_output
  LD r10,r8,0
  MOVI r9,99
  CMP r10,r9
  JMP[NE] @mth_admit_fail_4

  MOVI r1,@mth_admit_pass
  CALL @mth_admit_puts
  HALT

mth_admit_fail_1:
  MOVI r1,1
  MOVI r2,@mth_admit_scratch
  MOVI r3,47
  ADD r2,r2,r3
  ST r1,r2,0
  JMP @mth_admit_fail
mth_admit_fail_2:
  MOVI r1,2
  MOVI r2,@mth_admit_scratch
  MOVI r3,47
  ADD r2,r2,r3
  ST r1,r2,0
  JMP @mth_admit_fail
mth_admit_fail_3:
  MOVI r1,3
  MOVI r2,@mth_admit_scratch
  MOVI r3,47
  ADD r2,r2,r3
  ST r1,r2,0
  JMP @mth_admit_fail
mth_admit_fail_4:
  MOVI r1,4
  MOVI r2,@mth_admit_scratch
  MOVI r3,47
  ADD r2,r2,r3
  ST r1,r2,0
  JMP @mth_admit_fail
mth_admit_fail:
  MOVI r1,@mth_admit_bad
  CALL @mth_admit_puts
  HALT

mth_admit_puts:
  MOVI r4,255
mth_admit_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_admit_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_admit_puts_byte
  ADD r1,r1,r13
  JMP @mth_admit_puts_word

.org 40000
mth_admit_candidate:
.word 1,7,4,4,3,1,1,3,3,7,1,1,2,1,100,101,102,103,104,105,106,107
mth_admit_rule:
.word 1,7,4,4,3,1,1,1,1,100,101,102,103,104,105,106,107
mth_admit_left:
.word 4,1,3,1,1,@mth_admit_left_payload,3,1
mth_admit_right:
.word 4,1,3,1,1,@mth_admit_right_payload,3,1
mth_admit_contract:
.word 4,4,3,1,1,0,1,1,1,0
mth_admit_output:
.word 0,0,0,0,0,@mth_admit_output_payload,1,0
mth_admit_pass:
.str "MTH COMPOSED ADMISSION PASS\n"
mth_admit_bad:
.str "MTH COMPOSED ADMISSION FAIL\n"
.org 40200
mth_admit_left_payload:
.word 1,2,3
mth_admit_right_payload:
.word 4,5,6
mth_admit_output_payload:
.word 0
mth_admit_scratch:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
