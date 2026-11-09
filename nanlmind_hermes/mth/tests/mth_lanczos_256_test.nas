; MTH 256x256 bounded Lanczos-style top-k fixture — NaN16 only.
;
; The emitted image is about 73,000 NaN16 words. It is inside the
; 131,072-word / 1 MiB guest target and above the historical 65,536-word /
; 512 KiB budget.
;
; Case 1: A = diag(3,2,1,0,...), q0 = 1, k = 2, iterations = 16.
;   The top two Ritz values are 3 and 2; the two vectors are unit-norm and
;   orthogonal, and their dominant components align with e0 and e1.
;
; Case 2: the same A, q0 = e0, k = 2, iterations = 4.
;   q0 lies on the first eigenvector, so the initial second column is
;   exactly exhausted and the deterministic coordinate-axis fallback must
;   supply e1. The exact result is values {3,2}, vectors {e0,e1}; values
;   and the four checked components are bit-exact.
;
; Case 3: A = [[2,1],[1,2]] in the leading 2x2 block, zero elsewhere,
;   q0 = 1, k = 2, iterations = 16. Eigenvalues 3 and 1 with vectors
;   (1,1)/sqrt(2) and (1,-1)/sqrt(2). This exercises the full matvec and
;   a non-diagonal Rayleigh quotient.
;
; Case 4: A = 0 is degenerate: status 0 and values_out untouched.
;
; Bounds: k = 0, k = 9, iterations = 0, iterations = 33, and zero q0 all
;   return status 0 with values_out untouched.
;
; Case 5: A = diag(3,2,1,0,...), k = 1, iterations = 16: value 3.
;
; Case 6: A = diag(8,6,4,2,0,...), k = 4, iterations = 16: values
;   {8,6,4,2}, four unit-norm mutually orthogonal vectors.
;
; Case 8: A = diag(8,7,6,5,4,3,2,1,0,...), q0 = e0, k = 8,
;   iterations = 1. q0 is exhausted after the first column, so the
;   coordinate-axis fallback supplies e1..e7; the exact output is values
;   8..1 and eight unit-norm mutually orthogonal axis vectors.
;
; Case 7: the same A as Case 1 with q0 = e1, k = 2, iterations = 4.
;   The extraction order is lambda 2 then lambda 3, so the final
;   descending selection sort must swap both values and vector columns.

.org 0
JMP @mth_lanczos_256_test_main
.include "../nan16/mth_lanczos_256.nas"

mth_lanczos_256_test_main:
  MOVI r13,1

  ; ---------- Case 1 ----------
  CALL @mth_lanczos_256_test_zero_a
  CALL @mth_lanczos_256_test_set_diag_321
  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_lanczos_256_test_fill_vec

  MOVI r3,2
  MOVI r4,16
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; values == 3, 2 within 1e-3.
  LA r1,@mth_lanczos_256_test_values
  MOVI r2,0
  MOVHI r2,0x4040
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  ADD r1,r1,r13
  MOVI r2,0
  MOVHI r2,0x4000
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; Two unit-norm, mutually orthogonal vectors.
  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,2
  CALL @mth_lanczos_256_test_basis_ok
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; V0[0]^2 > 0.98, V1[1]^2 > 0.98.
  LA r1,@mth_lanczos_256_test_vectors
  LD r2,r1,0
  FMUL r2,r2,r2
  MOVI r3,0xE148
  MOVHI r3,0x3F7A
  FCMP r2,r3
  JMP[LE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,256
  ADD r1,r1,r2
  LD r2,r1,1
  FMUL r2,r2,r2
  MOVI r3,0xE148
  MOVHI r3,0x3F7A
  FCMP r2,r3
  JMP[LE] @mth_lanczos_256_test_fail

  ; A and q0 were not modified.
  LA r4,@mth_lanczos_256_test_a
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,257
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,514
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,771
  CMP r5,r0
  JMP[NE] @mth_lanczos_256_test_fail
  LA r4,@mth_lanczos_256_test_q0
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,255
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 2: exact breakdown fallback ----------
  CALL @mth_lanczos_256_test_zero_a
  CALL @mth_lanczos_256_test_set_diag_321
  LA r1,@mth_lanczos_256_test_q0
  CALL @mth_lanczos_256_test_zero_vec
  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  ST r2,r1,0

  MOVI r3,2
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; values are bit-exactly 3.0 and 2.0.
  LA r4,@mth_lanczos_256_test_values
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  ; V0[0]^2 == 1, V0[1] == 0, V1[0] == 0, V1[1]^2 == 1.
  LA r4,@mth_lanczos_256_test_vectors
  LD r5,r4,0
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,1
  CMP r5,r0
  JMP[NE] @mth_lanczos_256_test_fail
  MOVI r6,256
  ADD r4,r4,r6
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,1
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 3: non-diagonal block [[2,1],[1,2]] ----------
  CALL @mth_lanczos_256_test_zero_a
  LA r4,@mth_lanczos_256_test_a
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,0
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r4,1
  ST r7,r4,256
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,257

  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_lanczos_256_test_fill_vec

  MOVI r3,2
  MOVI r4,16
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; values == 3, 1.
  LA r1,@mth_lanczos_256_test_values
  MOVI r2,0
  MOVHI r2,0x4040
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  ADD r1,r1,r13
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,2
  CALL @mth_lanczos_256_test_basis_ok
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; V0[0]*V0[1] > 0.4 and V1[0]*V1[1] < -0.4.
  LA r1,@mth_lanczos_256_test_vectors
  LD r2,r1,0
  LD r3,r1,1
  FMUL r2,r2,r3
  MOVI r4,0xCCCD
  MOVHI r4,0x3ECC
  FCMP r2,r4
  JMP[LE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,256
  ADD r1,r1,r2
  LD r2,r1,0
  LD r3,r1,1
  FMUL r2,r2,r3
  MOVI r4,0xCCCD
  MOVHI r4,0xBECC
  FCMP r2,r4
  JMP[GE] @mth_lanczos_256_test_fail

  ; ---------- Case 4: A = 0 rejection ----------
  CALL @mth_lanczos_256_test_zero_a
  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_lanczos_256_test_fill_vec
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,2
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Bounds ----------
  CALL @mth_lanczos_256_test_zero_a
  CALL @mth_lanczos_256_test_set_diag_321

  ; k = 0.
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,0
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; k = 9.
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,9
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; iterations = 0.
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,2
  MOVI r4,0
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; iterations = 33.
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,2
  MOVI r4,33
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; zero q0.
  LA r1,@mth_lanczos_256_test_q0
  CALL @mth_lanczos_256_test_zero_vec
  CALL @mth_lanczos_256_test_set_sentinel
  MOVI r3,2
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r0
  JMP[NE] @mth_lanczos_256_test_fail
  CALL @mth_lanczos_256_test_check_sentinel
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 5: k = 1 ----------
  CALL @mth_lanczos_256_test_zero_a
  CALL @mth_lanczos_256_test_set_diag_321
  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_lanczos_256_test_fill_vec

  MOVI r3,1
  MOVI r4,16
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  MOVI r2,0
  MOVHI r2,0x4040
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,1
  CALL @mth_lanczos_256_test_basis_ok
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 6: k = 4 ----------
  CALL @mth_lanczos_256_test_zero_a
  LA r4,@mth_lanczos_256_test_a
  MOVI r7,0
  MOVHI r7,0x4100
  ST r7,r4,0
  MOVI r7,0
  MOVHI r7,0x40C0
  ST r7,r4,257
  MOVI r7,0
  MOVHI r7,0x4080
  ST r7,r4,514
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,771

  LA r1,@mth_lanczos_256_test_q0
  MOVI r2,0
  MOVHI r2,0x3F80
  CALL @mth_lanczos_256_test_fill_vec

  MOVI r3,4
  MOVI r4,16
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  MOVI r2,0
  MOVHI r2,0x4100
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  ADD r1,r1,r13
  MOVI r2,0
  MOVHI r2,0x40C0
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  MOVI r2,2
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0x4080
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_values
  MOVI r2,3
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0x4000
  MOVI r3,0x126F
  MOVHI r3,0x3A83
  CALL @mth_lanczos_256_test_close_value
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,4
  CALL @mth_lanczos_256_test_basis_ok
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 8: k = 8 exact diagonal via fallback axes ----------
  CALL @mth_lanczos_256_test_zero_a
  LA r4,@mth_lanczos_256_test_a
  MOVI r7,0
  MOVHI r7,0x4100
  ST r7,r4,0
  MOVI r7,0
  MOVHI r7,0x40E0
  ST r7,r4,257
  MOVI r7,0
  MOVHI r7,0x40C0
  ST r7,r4,514
  MOVI r7,0
  MOVHI r7,0x40A0
  ST r7,r4,771
  MOVI r7,0
  MOVHI r7,0x4080
  ST r7,r4,1028
  MOVI r7,0
  MOVHI r7,0x4040
  ST r7,r4,1285
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,1542
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r4,1799

  LA r1,@mth_lanczos_256_test_q0
  CALL @mth_lanczos_256_test_zero_vec
  LA r4,@mth_lanczos_256_test_q0
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,0

  MOVI r3,8
  MOVI r4,1
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; values are bit-exactly 8,7,6,5,4,3,2,1.
  LA r4,@mth_lanczos_256_test_values
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4100
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x40E0
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,2
  MOVI r6,0
  MOVHI r6,0x40C0
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,3
  MOVI r6,0
  MOVHI r6,0x40A0
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,4
  MOVI r6,0
  MOVHI r6,0x4080
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,5
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,6
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,7
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_vectors
  MOVI r2,8
  CALL @mth_lanczos_256_test_basis_ok
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; V0[0]^2 == 1 and V7[7]^2 == 1.
  LA r4,@mth_lanczos_256_test_vectors
  LD r5,r4,0
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  MOVI r6,1792
  ADD r4,r4,r6
  LD r5,r4,7
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  ; ---------- Case 7: extraction order 2 then 3 must be sorted ----------
  ; q0 = e1 makes the first power vector e1 (lambda 2) and the fallback
  ; second vector e0 (lambda 3). The final selection sort must swap both
  ; the values and the two vector columns.
  CALL @mth_lanczos_256_test_zero_a
  CALL @mth_lanczos_256_test_set_diag_321
  LA r1,@mth_lanczos_256_test_q0
  CALL @mth_lanczos_256_test_zero_vec
  LA r4,@mth_lanczos_256_test_q0
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,1

  MOVI r3,2
  MOVI r4,4
  CALL @mth_lanczos_256_test_run
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_test_fail

  ; values are bit-exactly 3.0 then 2.0.
  LA r4,@mth_lanczos_256_test_values
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  ; Columns were swapped: V0[0]^2 == 1, V1[1]^2 == 1.
  LA r4,@mth_lanczos_256_test_vectors
  LD r5,r4,0
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail
  MOVI r6,256
  ADD r4,r4,r6
  LD r5,r4,1
  FMUL r5,r5,r5
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_lanczos_256_test_fail

  LA r1,@mth_lanczos_256_test_pass
  CALL @mth_lanczos_256_test_puts
  HALT

mth_lanczos_256_test_fail:
  LA r1,@mth_lanczos_256_test_bad
  CALL @mth_lanczos_256_test_puts
  HALT

; Test-only kernel launcher. The caller sets r3 = k and r4 = iterations.
; Saves and restores r15 around the kernel call.
mth_lanczos_256_test_run:
  MOVI r12,@mth_lanczos_256_test_run_return
  ST r15,r12,0
  LA r1,@mth_lanczos_256_test_a
  LA r2,@mth_lanczos_256_test_q0
  LA r5,@mth_lanczos_256_test_values
  LA r6,@mth_lanczos_256_test_vectors
  LA r7,@mth_lanczos_256_test_scratch
  MOVI r13,1
  CALL @mth_lanczos_256
  MOVI r12,@mth_lanczos_256_test_run_return
  LD r15,r12,0
  RET

; Zero all 65,536 words of A.
mth_lanczos_256_test_zero_a:
  LA r4,@mth_lanczos_256_test_a
  MOVI r6,0
  MOVHI r6,0x1
mth_lanczos_256_test_zero_a_loop:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_lanczos_256_test_zero_a_loop
  RET

; A = diag(3,2,1,0,...).
mth_lanczos_256_test_set_diag_321:
  LA r4,@mth_lanczos_256_test_a
  MOVI r7,0
  MOVHI r7,0x4040
  ST r7,r4,0
  MOVI r7,0
  MOVHI r7,0x4000
  ST r7,r4,257
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r4,514
  RET

; Zero the 256 words at r1.
mth_lanczos_256_test_zero_vec:
  MOVI r4,0
  MOVI r5,256
mth_lanczos_256_test_zero_vec_loop:
  ADD r6,r1,r4
  ST r0,r6,0
  ADD r4,r4,r13
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_test_zero_vec_loop
  RET

; Fill the 256 words at r1 with the float32 in r2.
mth_lanczos_256_test_fill_vec:
  MOVI r4,0
  MOVI r5,256
mth_lanczos_256_test_fill_vec_loop:
  ADD r6,r1,r4
  ST r2,r6,0
  ADD r4,r4,r13
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_test_fill_vec_loop
  RET

; Return r1=1 iff |*r1 - r2| <= r3.
mth_lanczos_256_test_close_value:
  LD r4,r1,0
  FSUB r4,r4,r2
  FCMP r4,r3
  JMP[GT] @mth_lanczos_256_test_close_bad
  FSUB r5,r0,r3
  FCMP r4,r5
  JMP[LT] @mth_lanczos_256_test_close_bad
  MOVI r1,1
  RET
mth_lanczos_256_test_close_bad:
  MOVI r1,0
  RET

; Return r1=1 iff the first r2 columns of vectors at r1 are each unit-norm
; and mutually orthogonal within 1e-3. No nested calls.
mth_lanczos_256_test_basis_ok:
  ADD r3,r1,r0
  ADD r4,r2,r0
  MOVI r5,0
mth_lanczos_256_test_basis_i:
  CMP r5,r4
  JMP[GE] @mth_lanczos_256_test_basis_good
  ADD r9,r5,r0
  SHL r9,r9,8
  ADD r9,r3,r9
  MOVI r7,0
  MOVI r8,0
mth_lanczos_256_test_basis_norm:
  ADD r11,r9,r7
  LD r11,r11,0
  FMUL r11,r11,r11
  FADD r8,r8,r11
  ADD r7,r7,r13
  MOVI r11,256
  CMP r7,r11
  JMP[LT] @mth_lanczos_256_test_basis_norm
  MOVI r11,0
  MOVHI r11,0x3F80
  FSUB r8,r8,r11
  MOVI r11,0x126F
  MOVHI r11,0x3A83
  FCMP r8,r11
  JMP[GT] @mth_lanczos_256_test_basis_bad
  MOVI r11,0x126F
  MOVHI r11,0xBA83
  FCMP r8,r11
  JMP[LT] @mth_lanczos_256_test_basis_bad
  ADD r6,r5,r13
mth_lanczos_256_test_basis_pair:
  CMP r6,r4
  JMP[GE] @mth_lanczos_256_test_basis_next
  ADD r10,r6,r0
  SHL r10,r10,8
  ADD r10,r3,r10
  MOVI r7,0
  MOVI r8,0
mth_lanczos_256_test_basis_dot:
  ADD r11,r9,r7
  LD r11,r11,0
  ADD r12,r10,r7
  LD r12,r12,0
  FMUL r11,r11,r12
  FADD r8,r8,r11
  ADD r7,r7,r13
  MOVI r11,256
  CMP r7,r11
  JMP[LT] @mth_lanczos_256_test_basis_dot
  MOVI r11,0x126F
  MOVHI r11,0x3A83
  FCMP r8,r11
  JMP[GT] @mth_lanczos_256_test_basis_bad
  MOVI r11,0x126F
  MOVHI r11,0xBA83
  FCMP r8,r11
  JMP[LT] @mth_lanczos_256_test_basis_bad
  ADD r6,r6,r13
  JMP @mth_lanczos_256_test_basis_pair
mth_lanczos_256_test_basis_next:
  ADD r5,r5,r13
  JMP @mth_lanczos_256_test_basis_i
mth_lanczos_256_test_basis_good:
  MOVI r1,1
  RET
mth_lanczos_256_test_basis_bad:
  MOVI r1,0
  RET

; Write the 0xDEAD5A5A sentinel to both values_out slots.
mth_lanczos_256_test_set_sentinel:
  LA r4,@mth_lanczos_256_test_values
  MOVI r5,0x5A5A
  MOVHI r5,0xDEAD
  ST r5,r4,0
  ST r5,r4,1
  RET

; Return r1=1 iff both values_out slots still hold 0xDEAD5A5A.
mth_lanczos_256_test_check_sentinel:
  LA r4,@mth_lanczos_256_test_values
  MOVI r5,0x5A5A
  MOVHI r5,0xDEAD
  LD r6,r4,0
  CMP r6,r5
  JMP[NE] @mth_lanczos_256_test_sentinel_bad
  LD r6,r4,1
  CMP r6,r5
  JMP[NE] @mth_lanczos_256_test_sentinel_bad
  MOVI r1,1
  RET
mth_lanczos_256_test_sentinel_bad:
  MOVI r1,0
  RET

mth_lanczos_256_test_puts:
  MOVI r4,255
mth_lanczos_256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_lanczos_256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_lanczos_256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_lanczos_256_test_puts_word

; Test helper static link word. It must stay in the low 16-bit code span
; because the helper reaches it with MOVI, not LA.
mth_lanczos_256_test_run_return:
.word 0

; Working set. Code, static kernel frame, and test helpers fit below 4096.
.org 4096
mth_lanczos_256_test_a:
.word 0
.org 69632
mth_lanczos_256_test_q0:
.word 0
.org 69888
mth_lanczos_256_test_values:
.word 0
.org 69896
mth_lanczos_256_test_vectors:
.word 0
.org 71944
mth_lanczos_256_test_scratch:
.word 0
.org 72968
mth_lanczos_256_test_pass:
.str "MTH LANCZOS 256 PASS\n"
mth_lanczos_256_test_bad:
.str "MTH LANCZOS 256 FAIL\n"
