.org 0
JMP @mth_jahs_gate_test_main
.include "../nan16/mth_jahs_gate.nas"

mth_jahs_gate_test_main:
  MOVI r13,1

  ; Canonical seal 0x4A414853 ("JAHS") must pass.
  MOVI r1,0x4853
  MOVHI r1,0x4A41
  CALL @mth_jahs_gate
  CMP r1,r13
  JMP[NE] @mth_jahs_gate_fail

  ; Zero must be rejected.
  MOVI r1,0
  CALL @mth_jahs_gate
  CMP r1,r0
  JMP[NE] @mth_jahs_gate_fail

  ; One below the magic (0x4A414852) must be rejected.
  MOVI r1,0x4852
  MOVHI r1,0x4A41
  CALL @mth_jahs_gate
  CMP r1,r0
  JMP[NE] @mth_jahs_gate_fail

  ; One above the magic (0x4A414854) must be rejected.
  MOVI r1,0x4854
  MOVHI r1,0x4A41
  CALL @mth_jahs_gate
  CMP r1,r0
  JMP[NE] @mth_jahs_gate_fail

  MOVI r1,@mth_jahs_gate_pass
  CALL @mth_jahs_gate_puts
  HALT

mth_jahs_gate_fail:
  MOVI r1,@mth_jahs_gate_bad
  CALL @mth_jahs_gate_puts
  HALT

mth_jahs_gate_puts:
  MOVI r4,255
mth_jahs_gate_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_jahs_gate_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_jahs_gate_puts_byte
  ADD r1,r1,r13
  JMP @mth_jahs_gate_puts_word

mth_jahs_gate_pass:
.str "MTH JAHS GATE PASS\n"
mth_jahs_gate_bad:
.str "MTH JAHS GATE FAIL\n"
