; MTH 256x256 float32 top singular-pair fixture — NaN16 only.
;
; This fixture's emitted image is 67,597 NaN16 words, above the
; historical 65,536-word / 512 KiB guest ceiling and inside the
; 131,072-word / 1 MiB target. It is meant for the 1 MiB GLT guest.
;
; Case 1: A = 2I, x = 1, iterations = 3. Checks:
;   1. status r1=1;
;   2. sigma is bit-exactly 2.0;
;   3. u[0] == u[255] and v[0] == v[255];
;   4. sum u[i]^2 and sum v[i]^2 are within two float32 ulps of 1.0.
;
; Case 2: A = 2*e0*e1^T (only A[0,1]=2), x = 1, iterations = 3.
; This asymmetric rank-one matrix has the exact singular pair
; sigma=2, u=e0, v=e1, and it forces the A^T*u path to be distinct
; from A*v. Checks sigma=2, u[0]=1, u[1]=0, v[0]=0, v[1]=1.
;
; Invalid cases: an all-zero x returns status 0 and leaves sigma
; untouched; A=0 is degenerate and also returns status 0 without writing
; sigma; iterations=0 and iterations=33 return status 0.

.org 0
JMP @mth_svd_top256_test_main
.include "../nan16/mth_svd_top256.nas"

mth_svd_top256_test_main:
  MOVI r13,1

  ; Zero A (65,536 words), then set the diagonal to 2.0.
  LA r4,@mth_svd_top256_test_a
  MOVI r6,0
  MOVHI r6,0x1
mth_svd_top256_test_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top256_test_zero_a

  LA r5,@mth_svd_top256_test_a
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_svd_top256_test_diag:
  ST r7,r5,0
  MOVI r8,257
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top256_test_diag

  ; x = 1.0 for all 256 entries.
  LA r2,@mth_svd_top256_test_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_svd_top256_test_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top256_test_fill_x

  ; Run 3 bounded two-sided power iterations on A=2I.
  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,3
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r13
  JMP[NE] @mth_svd_top256_test_fail

  ; sigma must be exactly 2.0 (0x40000000).
  LA r4,@mth_svd_top256_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail

  ; u[0] == u[255].
  LA r4,@mth_svd_top256_test_u
  LD r5,r4,0
  MOVI r6,255
  ADD r7,r4,r6
  LD r7,r7,0
  CMP r5,r7
  JMP[NE] @mth_svd_top256_test_fail

  ; v[0] == v[255].
  LA r4,@mth_svd_top256_test_v
  LD r5,r4,0
  MOVI r6,255
  ADD r7,r4,r6
  LD r7,r7,0
  CMP r5,r7
  JMP[NE] @mth_svd_top256_test_fail

  ; sum u[i]^2 must be within two ulps of 1.0.
  LA r4,@mth_svd_top256_test_u
  MOVI r6,0
  MOVI r7,0
mth_svd_top256_test_sumsq_u:
  ADD r8,r4,r6
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_svd_top256_test_sumsq_u
  MOVI r8,0xFFFE
  MOVHI r8,0x3F7F
  CMP r7,r8
  JMP[LT] @mth_svd_top256_test_fail
  MOVI r9,2
  MOVHI r9,0x3F80
  CMP r7,r9
  JMP[GT] @mth_svd_top256_test_fail

  ; sum v[i]^2 must be within two ulps of 1.0.
  LA r4,@mth_svd_top256_test_v
  MOVI r6,0
  MOVI r7,0
mth_svd_top256_test_sumsq_v:
  ADD r8,r4,r6
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_svd_top256_test_sumsq_v
  MOVI r8,0xFFFE
  MOVHI r8,0x3F7F
  CMP r7,r8
  JMP[LT] @mth_svd_top256_test_fail
  MOVI r9,2
  MOVHI r9,0x3F80
  CMP r7,r9
  JMP[GT] @mth_svd_top256_test_fail

  ; Case 2: rebuild A as the asymmetric rank-one 2*e0*e1^T.
  LA r4,@mth_svd_top256_test_a
  MOVI r6,0
  MOVHI r6,0x1
mth_svd_top256_test_zero_a2:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top256_test_zero_a2

  ; A=0 is degenerate: status 0 and sigma untouched.
  LA r4,@mth_svd_top256_test_sigma
  MOVI r7,0
  MOVHI r7,0xDEAD
  ST r7,r4,0

  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,3
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r0
  JMP[NE] @mth_svd_top256_test_fail

  LA r4,@mth_svd_top256_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail

  LA r4,@mth_svd_top256_test_a
  ADD r4,r4,r13
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,0

  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,3
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r13
  JMP[NE] @mth_svd_top256_test_fail

  LA r4,@mth_svd_top256_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail

  ; u[0] must be 1.0 and u[1] must be 0.
  LA r4,@mth_svd_top256_test_u
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_svd_top256_test_fail

  ; v[0] must be 0 and v[1] must be 1.0.
  LA r4,@mth_svd_top256_test_v
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_svd_top256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail

  ; Zero vector must be rejected with status 0 and no sigma write.
  LA r2,@mth_svd_top256_test_x
  MOVI r6,256
mth_svd_top256_test_zero_x:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top256_test_zero_x

  LA r4,@mth_svd_top256_test_sigma
  MOVI r7,0
  MOVHI r7,0xDEAD
  ST r7,r4,0

  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,3
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r0
  JMP[NE] @mth_svd_top256_test_fail

  LA r4,@mth_svd_top256_test_sigma
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top256_test_fail

  ; iterations=0 is outside the bounded 1..32 contract.
  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,0
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r0
  JMP[NE] @mth_svd_top256_test_fail

  ; iterations=33 is also outside the bounded 1..32 contract.
  LA r1,@mth_svd_top256_test_a
  LA r2,@mth_svd_top256_test_x
  MOVI r3,33
  LA r4,@mth_svd_top256_test_u
  LA r5,@mth_svd_top256_test_v
  LA r6,@mth_svd_top256_test_sigma
  LA r7,@mth_svd_top256_test_scratch
  CALL @mth_svd_top256
  CMP r1,r0
  JMP[NE] @mth_svd_top256_test_fail

  LA r1,@mth_svd_top256_test_pass
  CALL @mth_svd_top256_test_puts
  HALT

mth_svd_top256_test_fail:
  LA r1,@mth_svd_top256_test_bad
  CALL @mth_svd_top256_test_puts
  HALT

mth_svd_top256_test_puts:
  MOVI r4,255
mth_svd_top256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_svd_top256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_svd_top256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_svd_top256_test_puts_word

; Working set: A 65,536 words at 1024, then x/u/v/sigma/scratch.
.org 1024
mth_svd_top256_test_a:
.word 0
.org 66560
mth_svd_top256_test_x:
.word 0
.org 66816
mth_svd_top256_test_u:
.word 0
.org 67072
mth_svd_top256_test_v:
.word 0
.org 67328
mth_svd_top256_test_sigma:
.word 0
.org 67329
mth_svd_top256_test_scratch:
.word 0
.org 67585
mth_svd_top256_test_pass:
.str "MTH SVD TOP256 PASS\n"
mth_svd_top256_test_bad:
.str "MTH SVD TOP256 FAIL\n"
