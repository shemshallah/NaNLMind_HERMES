; MTH 128x128 float32 top singular-pair fixture — NaN16 only.
;
; A = 2I is built programmatically, x = 1 for all 128 entries, and the
; kernel runs 3 iterations. The fixture checks:
;   1. status r1=1;
;   2. sigma is bit-exactly 2.0;
;   3. u[0] == u[127] and v[0] == v[127];
;   4. sum u[i]^2 and sum v[i]^2 are within two float32 ulps of 1.0;
;   5. an all-zero x returns status 0 and leaves sigma untouched;
;   6. iterations=0 and iterations=33 return status 0.

.org 0
JMP @mth_svd_top128_test_main
.include "../nan16/mth_svd_top128.nas"

mth_svd_top128_test_main:
  MOVI r13,1

  ; Zero A (16,384 words), then set the diagonal to 2.0.
  MOVI r4,@mth_svd_top128_test_a
  MOVI r6,16384
mth_svd_top128_test_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top128_test_zero_a

  MOVI r5,@mth_svd_top128_test_a
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x4000
mth_svd_top128_test_diag:
  ST r7,r5,0
  MOVI r8,129
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top128_test_diag

  ; x = 1.0 for all 128 entries.
  MOVI r2,@mth_svd_top128_test_x
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_svd_top128_test_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top128_test_fill_x

  ; Run 3 bounded two-sided power iterations.
  MOVI r1,@mth_svd_top128_test_a
  MOVI r2,@mth_svd_top128_test_x
  MOVI r3,3
  MOVI r4,@mth_svd_top128_test_u
  MOVI r5,@mth_svd_top128_test_v
  MOVI r6,@mth_svd_top128_test_sigma
  MOVI r7,@mth_svd_top128_test_s1
  MOVI r8,@mth_svd_top128_test_s2
  CALL @mth_svd_top128
  CMP r1,r13
  JMP[NE] @mth_svd_top128_test_fail

  ; sigma must be exactly 2.0 (0x40000000).
  MOVI r4,@mth_svd_top128_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top128_test_fail

  ; u[0] == u[127].
  MOVI r4,@mth_svd_top128_test_u
  LD r5,r4,0
  MOVI r6,127
  ADD r7,r4,r6
  LD r7,r7,0
  CMP r5,r7
  JMP[NE] @mth_svd_top128_test_fail

  ; v[0] == v[127].
  MOVI r4,@mth_svd_top128_test_v
  LD r5,r4,0
  MOVI r6,127
  ADD r7,r4,r6
  LD r7,r7,0
  CMP r5,r7
  JMP[NE] @mth_svd_top128_test_fail

  ; sum u[i]^2 must be within two ulps of 1.0.
  MOVI r4,@mth_svd_top128_test_u
  MOVI r6,0
  MOVI r7,0
mth_svd_top128_test_sumsq_u:
  ADD r8,r4,r6
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,128
  CMP r6,r8
  JMP[LT] @mth_svd_top128_test_sumsq_u
  MOVI r8,0xFFFE
  MOVHI r8,0x3F7F
  CMP r7,r8
  JMP[LT] @mth_svd_top128_test_fail
  MOVI r9,2
  MOVHI r9,0x3F80
  CMP r7,r9
  JMP[GT] @mth_svd_top128_test_fail

  ; sum v[i]^2 must be within two ulps of 1.0.
  MOVI r4,@mth_svd_top128_test_v
  MOVI r6,0
  MOVI r7,0
mth_svd_top128_test_sumsq_v:
  ADD r8,r4,r6
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,128
  CMP r6,r8
  JMP[LT] @mth_svd_top128_test_sumsq_v
  MOVI r8,0xFFFE
  MOVHI r8,0x3F7F
  CMP r7,r8
  JMP[LT] @mth_svd_top128_test_fail
  MOVI r9,2
  MOVHI r9,0x3F80
  CMP r7,r9
  JMP[GT] @mth_svd_top128_test_fail

  ; Zero vector must be rejected with status 0 and no sigma write.
  MOVI r2,@mth_svd_top128_test_x
  MOVI r6,128
mth_svd_top128_test_zero_x:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top128_test_zero_x

  MOVI r4,@mth_svd_top128_test_sigma
  MOVI r7,0
  MOVHI r7,0xDEAD
  ST r7,r4,0

  MOVI r1,@mth_svd_top128_test_a
  MOVI r2,@mth_svd_top128_test_x
  MOVI r3,3
  MOVI r4,@mth_svd_top128_test_u
  MOVI r5,@mth_svd_top128_test_v
  MOVI r6,@mth_svd_top128_test_sigma
  MOVI r7,@mth_svd_top128_test_s1
  MOVI r8,@mth_svd_top128_test_s2
  CALL @mth_svd_top128
  CMP r1,r0
  JMP[NE] @mth_svd_top128_test_fail

  MOVI r4,@mth_svd_top128_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top128_test_fail

  ; Restore x=1, then verify iterations=0 is rejected.
  MOVI r2,@mth_svd_top128_test_x
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_svd_top128_test_refill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top128_test_refill_x

  MOVI r1,@mth_svd_top128_test_a
  MOVI r2,@mth_svd_top128_test_x
  MOVI r3,0
  MOVI r4,@mth_svd_top128_test_u
  MOVI r5,@mth_svd_top128_test_v
  MOVI r6,@mth_svd_top128_test_sigma
  MOVI r7,@mth_svd_top128_test_s1
  MOVI r8,@mth_svd_top128_test_s2
  CALL @mth_svd_top128
  CMP r1,r0
  JMP[NE] @mth_svd_top128_test_fail

  ; iterations=33 is also outside the bounded 1..32 contract.
  MOVI r1,@mth_svd_top128_test_a
  MOVI r2,@mth_svd_top128_test_x
  MOVI r3,33
  MOVI r4,@mth_svd_top128_test_u
  MOVI r5,@mth_svd_top128_test_v
  MOVI r6,@mth_svd_top128_test_sigma
  MOVI r7,@mth_svd_top128_test_s1
  MOVI r8,@mth_svd_top128_test_s2
  CALL @mth_svd_top128
  CMP r1,r0
  JMP[NE] @mth_svd_top128_test_fail

  MOVI r1,@mth_svd_top128_test_pass
  CALL @mth_svd_top128_test_puts
  HALT

mth_svd_top128_test_fail:
  MOVI r1,@mth_svd_top128_test_bad
  CALL @mth_svd_top128_test_puts
  HALT

mth_svd_top128_test_puts:
  MOVI r4,255
mth_svd_top128_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_svd_top128_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_svd_top128_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_svd_top128_test_puts_word

.org 1024
mth_svd_top128_test_a:
.word 0
.org 17408
mth_svd_top128_test_x:
.word 0
.org 17536
mth_svd_top128_test_u:
.word 0
.org 17664
mth_svd_top128_test_v:
.word 0
.org 17792
mth_svd_top128_test_sigma:
.word 0
.org 17796
mth_svd_top128_test_s1:
.word 0
.org 17924
mth_svd_top128_test_s2:
.word 0
.org 18176
mth_svd_top128_test_pass:
.str "MTH SVD TOP128 PASS\n"
mth_svd_top128_test_bad:
.str "MTH SVD TOP128 FAIL\n"
