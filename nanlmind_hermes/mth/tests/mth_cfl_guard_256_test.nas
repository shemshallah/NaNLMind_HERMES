.org 0
JMP @mth_cfl_guard_256_test_main
.include "../nan16/mth_cfl_guard_256.nas"

mth_cfl_guard_256_test_main:
  MOVI r13,1

  ; Zero A (65,536 words). MOVI + MOVHI builds 65,536.
  LA r4,@mth_cfl_guard_256_a
  MOVI r6,0
  MOVHI r6,0x0001
mth_cfl_guard_256_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cfl_guard_256_zero_a

  ; A[i,i] = 2.0, stride 257.
  LA r5,@mth_cfl_guard_256_a
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_cfl_guard_256_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cfl_guard_256_diag

  ; Case 1: diag(2), dt=0.5 -> lambda=2, dt*lambda=1 <= 2 accept.
  LA r9,@mth_cfl_guard_256_dt
  MOVI r10,0
  MOVHI r10,0x3F00
  ST r10,r9,0
  LA r1,@mth_cfl_guard_256_a
  LA r2,@mth_cfl_guard_256_dt
  LA r3,@mth_cfl_guard_256_lambda
  CALL @mth_cfl_guard_256
  CMP r1,r13
  JMP[NE] @mth_cfl_guard_256_fail
  LA r4,@mth_cfl_guard_256_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_cfl_guard_256_fail

  ; Case 2: diag(2), dt=1.5 -> lambda=2, dt*lambda=3 > 2 reject.
  ; Poison lambda first so the reject path must rewrite it.
  LA r9,@mth_cfl_guard_256_dt
  MOVI r10,0
  MOVHI r10,0x3FC0
  ST r10,r9,0
  LA r9,@mth_cfl_guard_256_lambda
  ST r0,r9,0
  LA r1,@mth_cfl_guard_256_a
  LA r2,@mth_cfl_guard_256_dt
  LA r3,@mth_cfl_guard_256_lambda
  CALL @mth_cfl_guard_256
  CMP r1,r0
  JMP[NE] @mth_cfl_guard_256_fail
  LA r4,@mth_cfl_guard_256_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_cfl_guard_256_fail

  ; Case 3: exact boundary. diag(2), dt=1.0 -> dt*lambda=2.0 accept.
  LA r9,@mth_cfl_guard_256_dt
  MOVI r10,0
  MOVHI r10,0x3F80
  ST r10,r9,0
  LA r1,@mth_cfl_guard_256_a
  LA r2,@mth_cfl_guard_256_dt
  LA r3,@mth_cfl_guard_256_lambda
  CALL @mth_cfl_guard_256
  CMP r1,r13
  JMP[NE] @mth_cfl_guard_256_fail
  LA r4,@mth_cfl_guard_256_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_cfl_guard_256_fail

  ; Case 3b: last row/column coverage. diag(2) except A[255,255]=6.0
  ; -> lambda=6, dt*lambda=3 > 2 reject. This fails if the row or column
  ; bound stops at 255 elements instead of 256.
  LA r4,@mth_cfl_guard_256_a
  MOVI r6,0xFFFF
  ADD r8,r4,r6
  MOVI r7,0
  MOVHI r7,0x40C0
  ST r7,r8,0
  LA r9,@mth_cfl_guard_256_dt
  MOVI r10,0
  MOVHI r10,0x3F00
  ST r10,r9,0
  LA r9,@mth_cfl_guard_256_lambda
  ST r0,r9,0
  LA r1,@mth_cfl_guard_256_a
  LA r2,@mth_cfl_guard_256_dt
  LA r3,@mth_cfl_guard_256_lambda
  CALL @mth_cfl_guard_256
  CMP r1,r0
  JMP[NE] @mth_cfl_guard_256_fail
  LA r4,@mth_cfl_guard_256_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x40C0
  CMP r5,r6
  JMP[NE] @mth_cfl_guard_256_fail

  ; Case 4: zero A -> lambda=0, dt*lambda=0 <= 2 accept. Clear only the
  ; diagonal; every off-diagonal word is still zero from the first pass.
  LA r5,@mth_cfl_guard_256_a
  MOVI r6,256
  MOVI r8,0
mth_cfl_guard_256_clear_diag:
  ADD r9,r5,r8
  ST r0,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cfl_guard_256_clear_diag

  ; Poison lambda; require lambda=+0.0 and accept with dt=0.5.
  LA r9,@mth_cfl_guard_256_lambda
  MOVI r10,0
  MOVHI r10,0x4000
  ST r10,r9,0
  LA r9,@mth_cfl_guard_256_dt
  MOVI r10,0
  MOVHI r10,0x3F00
  ST r10,r9,0
  LA r1,@mth_cfl_guard_256_a
  LA r2,@mth_cfl_guard_256_dt
  LA r3,@mth_cfl_guard_256_lambda
  CALL @mth_cfl_guard_256
  CMP r1,r13
  JMP[NE] @mth_cfl_guard_256_fail
  LA r4,@mth_cfl_guard_256_lambda
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_cfl_guard_256_fail

  LA r1,@mth_cfl_guard_256_pass
  CALL @mth_cfl_guard_256_puts
  HALT

mth_cfl_guard_256_fail:
  LA r1,@mth_cfl_guard_256_bad
  CALL @mth_cfl_guard_256_puts
  HALT

mth_cfl_guard_256_puts:
  MOVI r4,255
mth_cfl_guard_256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_cfl_guard_256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_cfl_guard_256_puts_byte
  ADD r1,r1,r13
  JMP @mth_cfl_guard_256_puts_word

.org 512
mth_cfl_guard_256_a:
.word 0
.org 66048
mth_cfl_guard_256_dt:
.word 0
.org 66049
mth_cfl_guard_256_lambda:
.word 0
.org 66050
mth_cfl_guard_256_pass:
.str "MTH CFL GUARD 256 PASS\n"
mth_cfl_guard_256_bad:
.str "MTH CFL GUARD 256 FAIL\n"
