; MTH 128x128 float32 power-iteration fixture — NaN16 only.
;
; A = 2I is built programmatically, x = 1 for all 128 entries, and the
; kernel runs 3 iterations. The fixture checks:
;   1. status r1=1;
;   2. lambda is bit-exactly 2.0;
;   3. v[0] == v[127] (uniform eigenvector symmetry);
;   4. sum v[i]^2 is within two ulps of 1.0 (unit norm in float32);
;   5. an all-zero x returns status 0 and leaves lambda untouched;
;   6. iterations=0 returns status 0.

.org 0
JMP @mth_power_iter128_test_main
.include "../nan16/mth_power_iter128.nas"

mth_power_iter128_test_main:
  MOVI r13,1

  ; Zero A (16,384 words), then set the diagonal to 2.0.
  MOVI r4,@mth_power_iter128_a
  MOVI r6,16384
mth_power_iter128_test_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_power_iter128_test_zero_a

  MOVI r5,@mth_power_iter128_a
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x4000
mth_power_iter128_test_diag:
  ST r7,r5,0
  MOVI r8,129
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_power_iter128_test_diag

  ; x = 1.0 for all 128 entries.
  MOVI r2,@mth_power_iter128_x
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_power_iter128_test_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_power_iter128_test_fill_x

  ; Run 3 bounded power iterations.
  MOVI r1,@mth_power_iter128_a
  MOVI r2,@mth_power_iter128_x
  MOVI r3,3
  MOVI r4,@mth_power_iter128_lambda
  MOVI r5,@mth_power_iter128_v
  MOVI r6,@mth_power_iter128_scratch
  CALL @mth_power_iter128
  CMP r1,r13
  JMP[NE] @mth_power_iter128_test_fail

  ; lambda must be exactly 2.0 (0x40000000).
  MOVI r4,@mth_power_iter128_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_power_iter128_test_fail

  ; v[0] == v[127].
  MOVI r4,@mth_power_iter128_v
  LD r5,r4,0
  MOVI r6,127
  ADD r7,r4,r6
  LD r7,r7,0
  CMP r5,r7
  JMP[NE] @mth_power_iter128_test_fail

  ; sum v[i]^2 must be within two ulps of 1.0.
  MOVI r4,@mth_power_iter128_v
  MOVI r6,0
  MOVI r7,0
mth_power_iter128_test_sumsq:
  ADD r8,r4,r6
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,128
  CMP r6,r8
  JMP[LT] @mth_power_iter128_test_sumsq
  MOVI r8,0xFFFE
  MOVHI r8,0x3F7F           ; 1.0 - 2 ulp
  CMP r7,r8
  JMP[LT] @mth_power_iter128_test_fail
  MOVI r9,2
  MOVHI r9,0x3F80           ; 1.0 + 2 ulp
  CMP r7,r9
  JMP[GT] @mth_power_iter128_test_fail

  ; Zero vector must be rejected with status 0 and no lambda write.
  MOVI r2,@mth_power_iter128_x
  MOVI r6,128
mth_power_iter128_test_zero_x:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_power_iter128_test_zero_x

  MOVI r4,@mth_power_iter128_lambda
  MOVI r7,0
  MOVHI r7,0xDEAD
  ST r7,r4,0

  MOVI r1,@mth_power_iter128_a
  MOVI r2,@mth_power_iter128_x
  MOVI r3,3
  MOVI r4,@mth_power_iter128_lambda
  MOVI r5,@mth_power_iter128_v
  MOVI r6,@mth_power_iter128_scratch
  CALL @mth_power_iter128
  CMP r1,r0
  JMP[NE] @mth_power_iter128_test_fail

  MOVI r4,@mth_power_iter128_lambda
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_power_iter128_test_fail

  ; Restore x=1, then verify iterations=0 is rejected.
  MOVI r2,@mth_power_iter128_x
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_power_iter128_test_refill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_power_iter128_test_refill_x

  MOVI r1,@mth_power_iter128_a
  MOVI r2,@mth_power_iter128_x
  MOVI r3,0
  MOVI r4,@mth_power_iter128_lambda
  MOVI r5,@mth_power_iter128_v
  MOVI r6,@mth_power_iter128_scratch
  CALL @mth_power_iter128
  CMP r1,r0
  JMP[NE] @mth_power_iter128_test_fail

  ; iterations=65 is also outside the bounded 1..64 contract.
  MOVI r1,@mth_power_iter128_a
  MOVI r2,@mth_power_iter128_x
  MOVI r3,65
  MOVI r4,@mth_power_iter128_lambda
  MOVI r5,@mth_power_iter128_v
  MOVI r6,@mth_power_iter128_scratch
  CALL @mth_power_iter128
  CMP r1,r0
  JMP[NE] @mth_power_iter128_test_fail

  MOVI r1,@mth_power_iter128_test_pass
  CALL @mth_power_iter128_test_puts
  HALT

mth_power_iter128_test_fail:
  MOVI r1,@mth_power_iter128_test_bad
  CALL @mth_power_iter128_test_puts
  HALT

mth_power_iter128_test_puts:
  MOVI r4,255
mth_power_iter128_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_power_iter128_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_power_iter128_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_power_iter128_test_puts_word

.org 1024
mth_power_iter128_a:
.word 0
.org 17664
mth_power_iter128_x:
.word 0
.org 17920
mth_power_iter128_scratch:
.word 0
.org 18176
mth_power_iter128_v:
.word 0
.org 18432
mth_power_iter128_lambda:
.word 0
.org 18496
mth_power_iter128_test_pass:
.str "MTH POWER_ITER128 PASS\n"
mth_power_iter128_test_bad:
.str "MTH POWER_ITER128 FAIL\n"
