; MTH 256x256 top-two singular-pair fixture — NaN16 only.
;
; The emitted image is about 69,000 NaN16 words. It exceeds the historical
; 65,536-word / 512 KiB guest ceiling and is inside the 131,072-word / 1 MiB
; target. It is meant for the 1 MiB GLT guest.
;
; Case 1: A = diag(3,2,0,...), x = e0, iterations = 3. The exact first
; pair is (sigma=3,u=e0,v=e0) and the exact second pair is
; (sigma=2,u=e1,v=e1). Checks status, bit-exact sigmas, the first two
; entries of every direction, unit norms, exact orthogonality, and that A
; and x were not modified.
;
; Case 2: the same A with x = 1 and iterations = 8. This exercises the
; ordinary P_v1 x second seed. Checks status, sigma1 ~= 3, sigma2 ~= 2,
; unit norms, and near-orthogonality.
;
; Rejections:
;   - zero x returns status 0 and leaves both sigma outputs untouched;
;   - iterations = 0 returns status 0 and leaves both sigmas untouched;
;   - iterations = 33 returns status 0;
;   - A = 0 is degenerate: status 0, both sigmas untouched;
;   - A = 2*e0*e1^T is rank one: the first pair succeeds with sigma1=2,
;     but the deflated second pair has zero norm, so status is 0 and
;     sigma2 is untouched.

.org 0
JMP @mth_svd_top2_256_test_main
.include "../nan16/mth_svd_top2_256.nas"

mth_svd_top2_256_test_main:
  MOVI r13,1

  ; Zero A (65,536 words), then set A[0,0]=3.0 and A[1,1]=2.0.
  LA r4,@mth_svd_top2_256_test_a
  MOVI r6,0
  MOVHI r6,0x1
mth_svd_top2_256_test_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top2_256_test_zero_a

  LA r5,@mth_svd_top2_256_test_a
  MOVI r7,0
  MOVHI r7,0x4040
  ST r7,r5,0
  MOVI r8,257
  ADD r5,r5,r8
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r5,0

  ; x = e0.
  LA r1,@mth_svd_top2_256_test_x
  CALL @mth_svd_top2_256_test_zero_vec
  LA r1,@mth_svd_top2_256_test_x
  MOVI r2,0
  MOVHI r2,0x3F80
  ST r2,r1,0

  ; Case 1: 3 iterations from e0.
  MOVI r11,3
  CALL @mth_svd_top2_256_test_run
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail

  ; sigma1 == 3.0.
  LA r4,@mth_svd_top2_256_test_sigma1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; sigma2 == 2.0.
  LA r4,@mth_svd_top2_256_test_sigma2
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; u1 == e0.
  LA r4,@mth_svd_top2_256_test_u1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; v1 == e0.
  LA r4,@mth_svd_top2_256_test_v1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; u2 == e1.
  LA r4,@mth_svd_top2_256_test_u2
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; v2 == e1.
  LA r4,@mth_svd_top2_256_test_v2
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Unit norms.
  LA r1,@mth_svd_top2_256_test_u1
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v1
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_u2
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v2
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Exact orthogonality for the diagonal case.
  LA r1,@mth_svd_top2_256_test_u1
  LA r2,@mth_svd_top2_256_test_u2
  CALL @mth_svd_top2_256_test_dot
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v1
  LA r2,@mth_svd_top2_256_test_v2
  CALL @mth_svd_top2_256_test_dot
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; A was not modified.
  LA r4,@mth_svd_top2_256_test_a
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,256
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  MOVI r6,257
  ADD r4,r4,r6
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; x was not modified.
  LA r4,@mth_svd_top2_256_test_x
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Case 2: 8 iterations from the all-ones direction.
  LA r1,@mth_svd_top2_256_test_x
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_svd_top2_256_test_fill_vec
  MOVI r11,8
  CALL @mth_svd_top2_256_test_run
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail

  ; sigma1 ~= 3.0 and sigma2 ~= 2.0 (within 0.01).
  LA r1,@mth_svd_top2_256_test_sigma1
  MOVI r2,0
  MOVHI r2,0x4040
  CALL @mth_svd_top2_256_test_approx_sigma
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_sigma2
  MOVI r2,0
  MOVHI r2,0x4000
  CALL @mth_svd_top2_256_test_approx_sigma
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Unit norms and near-orthogonality.
  LA r1,@mth_svd_top2_256_test_u1
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v1
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_u2
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v2
  CALL @mth_svd_top2_256_test_unit_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_u1
  LA r2,@mth_svd_top2_256_test_u2
  CALL @mth_svd_top2_256_test_orth_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r1,@mth_svd_top2_256_test_v1
  LA r2,@mth_svd_top2_256_test_v2
  CALL @mth_svd_top2_256_test_orth_ok
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Zero x is rejected and both sigma outputs stay untouched.
  LA r1,@mth_svd_top2_256_test_x
  CALL @mth_svd_top2_256_test_zero_vec
  LA r4,@mth_svd_top2_256_test_sigma1
  MOVI r5,0
  MOVHI r5,0xDEAD
  ST r5,r4,0
  LA r4,@mth_svd_top2_256_test_sigma2
  ST r5,r4,0
  MOVI r11,3
  CALL @mth_svd_top2_256_test_run
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma2
  LD r5,r4,0
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; iterations=0 is outside the bounded 1..32 contract.
  LA r1,@mth_svd_top2_256_test_x
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_svd_top2_256_test_fill_vec
  LA r4,@mth_svd_top2_256_test_sigma1
  MOVI r5,0
  MOVHI r5,0xDEAD
  ST r5,r4,0
  LA r4,@mth_svd_top2_256_test_sigma2
  ST r5,r4,0
  MOVI r11,0
  CALL @mth_svd_top2_256_test_run
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma2
  LD r5,r4,0
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; iterations=33 is outside the bounded 1..32 contract.
  MOVI r11,33
  CALL @mth_svd_top2_256_test_run
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; A=0 is degenerate: status 0 and both sigmas untouched.
  LA r4,@mth_svd_top2_256_test_a
  MOVI r6,0
  MOVHI r6,0x1
mth_svd_top2_256_test_zero_a2:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_svd_top2_256_test_zero_a2
  LA r4,@mth_svd_top2_256_test_sigma1
  MOVI r5,0
  MOVHI r5,0xDEAD
  ST r5,r4,0
  LA r4,@mth_svd_top2_256_test_sigma2
  ST r5,r4,0
  MOVI r11,3
  CALL @mth_svd_top2_256_test_run
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma2
  LD r5,r4,0
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; Rank-one A=2*e0*e1^T: first pair exists, deflated second pair is zero.
  LA r4,@mth_svd_top2_256_test_a
  ADD r4,r4,r13
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,0
  LA r4,@mth_svd_top2_256_test_sigma1
  MOVI r5,0
  MOVHI r5,0xDEAD
  ST r5,r4,0
  LA r4,@mth_svd_top2_256_test_sigma2
  ST r5,r4,0
  MOVI r11,3
  CALL @mth_svd_top2_256_test_run
  CMP r1,r0
  JMP[NE] @mth_svd_top2_256_test_fail

  ; sigma1 was written as 2.0, sigma2 stayed untouched.
  LA r4,@mth_svd_top2_256_test_sigma1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_sigma2
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0xDEAD
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  ; First pair for the rank-one case is u1=e0, v1=e1.
  LA r4,@mth_svd_top2_256_test_u1
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail
  LA r4,@mth_svd_top2_256_test_v1
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_svd_top2_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_svd_top2_256_test_fail

  LA r1,@mth_svd_top2_256_test_pass
  CALL @mth_svd_top2_256_test_puts
  HALT

mth_svd_top2_256_test_fail:
  LA r1,@mth_svd_top2_256_test_bad
  CALL @mth_svd_top2_256_test_puts
  HALT

; Test-only kernel launcher. r11 = iterations; all spans are the fixture's
; fixed working set. Saves and restores r15 around the kernel call.
mth_svd_top2_256_test_run:
  MOVI r12,@mth_svd_top2_256_test_run_return
  ST r15,r12,0
  LA r1,@mth_svd_top2_256_test_a
  LA r2,@mth_svd_top2_256_test_x
  ADD r3,r11,r0
  LA r4,@mth_svd_top2_256_test_u1
  LA r5,@mth_svd_top2_256_test_v1
  LA r6,@mth_svd_top2_256_test_sigma1
  LA r7,@mth_svd_top2_256_test_u2
  LA r8,@mth_svd_top2_256_test_v2
  LA r9,@mth_svd_top2_256_test_sigma2
  LA r10,@mth_svd_top2_256_test_scratch
  CALL @mth_svd_top2_256
  MOVI r12,@mth_svd_top2_256_test_run_return
  LD r15,r12,0
  RET

; Zero 256 words at r1.
mth_svd_top2_256_test_zero_vec:
  MOVI r4,0
  MOVI r5,256
mth_svd_top2_256_test_zero_vec_loop:
  ADD r6,r1,r4
  ST r0,r6,0
  ADD r4,r4,r13
  CMP r4,r5
  JMP[LT] @mth_svd_top2_256_test_zero_vec_loop
  MOVI r1,1
  RET

; Fill 256 words at r1 with the float32 value in r2.
mth_svd_top2_256_test_fill_vec:
  MOVI r4,0
  MOVI r5,256
mth_svd_top2_256_test_fill_vec_loop:
  ADD r6,r1,r4
  ST r2,r6,0
  ADD r4,r4,r13
  CMP r4,r5
  JMP[LT] @mth_svd_top2_256_test_fill_vec_loop
  MOVI r1,1
  RET

; Return r1=1 iff |sum(v[i]^2)-1| <= 0.001.
mth_svd_top2_256_test_unit_ok:
  ADD r4,r1,r0
  MOVI r5,0
  MOVI r6,0
mth_svd_top2_256_test_unit_sum:
  ADD r7,r4,r5
  LD r7,r7,0
  FMUL r7,r7,r7
  FADD r6,r6,r7
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_svd_top2_256_test_unit_sum
  MOVI r7,0
  MOVHI r7,0x3F80
  FSUB r8,r6,r7
  MOVI r9,0x126F
  MOVHI r9,0x3A83
  FCMP r8,r9
  JMP[GT] @mth_svd_top2_256_test_unit_bad
  MOVI r9,0x126F
  MOVHI r9,0xBA83
  FCMP r8,r9
  JMP[LT] @mth_svd_top2_256_test_unit_bad
  MOVI r1,1
  RET
mth_svd_top2_256_test_unit_bad:
  MOVI r1,0
  RET

; Return r1=sum a[i]*b[i], increasing i, one accumulator.
mth_svd_top2_256_test_dot:
  ADD r4,r1,r0
  ADD r5,r2,r0
  MOVI r6,0
  MOVI r7,0
mth_svd_top2_256_test_dot_loop:
  ADD r8,r4,r6
  LD r8,r8,0
  ADD r10,r5,r6
  LD r10,r10,0
  FMUL r8,r8,r10
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_svd_top2_256_test_dot_loop
  ADD r1,r7,r0
  RET

; Return r1=1 iff |dot(a,b)| <= 0.001.
mth_svd_top2_256_test_orth_ok:
  ADD r4,r1,r0
  ADD r5,r2,r0
  MOVI r6,0
  MOVI r7,0
mth_svd_top2_256_test_orth_sum:
  ADD r8,r4,r6
  LD r8,r8,0
  ADD r10,r5,r6
  LD r10,r10,0
  FMUL r8,r8,r10
  FADD r7,r7,r8
  ADD r6,r6,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_svd_top2_256_test_orth_sum
  MOVI r9,0x126F
  MOVHI r9,0x3A83
  FCMP r7,r9
  JMP[GT] @mth_svd_top2_256_test_orth_bad
  MOVI r9,0x126F
  MOVHI r9,0xBA83
  FCMP r7,r9
  JMP[LT] @mth_svd_top2_256_test_orth_bad
  MOVI r1,1
  RET
mth_svd_top2_256_test_orth_bad:
  MOVI r1,0
  RET

; Return r1=1 iff |*r1 - r2| <= 0.01.
mth_svd_top2_256_test_approx_sigma:
  LD r4,r1,0
  FSUB r4,r4,r2
  MOVI r5,0xD70A
  MOVHI r5,0x3C23
  FCMP r4,r5
  JMP[GT] @mth_svd_top2_256_test_approx_bad
  MOVI r5,0xD70A
  MOVHI r5,0xBC23
  FCMP r4,r5
  JMP[LT] @mth_svd_top2_256_test_approx_bad
  MOVI r1,1
  RET
mth_svd_top2_256_test_approx_bad:
  MOVI r1,0
  RET

mth_svd_top2_256_test_puts:
  MOVI r4,255
mth_svd_top2_256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_svd_top2_256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_svd_top2_256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_svd_top2_256_test_puts_word

; Test helper static link word. It must stay in the low 16-bit code span
; because the helper reaches it with MOVI, not LA.
mth_svd_top2_256_test_run_return:
.word 0

; Working set: A 65,536 words at 1024, then x/u1/v1/sigma1/u2/v2/sigma2
; and the 1024-word scratch span.
.org 1024
mth_svd_top2_256_test_a:
.word 0
.org 66560
mth_svd_top2_256_test_x:
.word 0
.org 66816
mth_svd_top2_256_test_u1:
.word 0
.org 67072
mth_svd_top2_256_test_v1:
.word 0
.org 67328
mth_svd_top2_256_test_sigma1:
.word 0
.org 67329
mth_svd_top2_256_test_u2:
.word 0
.org 67585
mth_svd_top2_256_test_v2:
.word 0
.org 67841
mth_svd_top2_256_test_sigma2:
.word 0
.org 67842
mth_svd_top2_256_test_scratch:
.word 0
.org 68866
mth_svd_top2_256_test_pass:
.str "MTH SVD TOP2 256 PASS\n"
mth_svd_top2_256_test_bad:
.str "MTH SVD TOP2 256 FAIL\n"
