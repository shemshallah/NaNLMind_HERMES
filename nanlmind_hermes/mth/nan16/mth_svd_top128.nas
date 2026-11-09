; MTH native 128x128 top singular-pair kernel — NaN16 float32 only.
;
; ABI:
;   r1 = A, 16,384 float32 payloads, row-major 128x128
;   r2 = x, 128 float32 input direction
;   r3 = iterations, integer 1..32
;   r4 = u output, 128 float32 (left singular direction)
;   r5 = v output, 128 float32 (right singular direction)
;   r6 = sigma output, one float32
;   r7 = scratch1, 128 float32 (A*v workspace)
;   r8 = scratch2, 128 float32 (A^T*u workspace)
;   r13 = 1
;   returns r1=1 success
;   returns r1=0 for a zero input direction, nonpositive or out-of-range
;          iterations, or a degenerate (zero-norm) iterate
;   clobbers r1-r12, r14
;
; Power iteration on A^T A, written as the two-sided iteration:
;
;   v = x / ||x||
;   repeat `iterations` times:
;     u = A v;      u = u / ||u||
;     v = A^T u;    v = v / ||v||
;   u = A v; sigma = ||u||; u = u / sigma
;
; A*v is delegated to the verified mth_matvec128_f32 kernel by include.
; A^T*u is the private helper below: for each output column j it
; accumulates over increasing row index i with one accumulator.
; All floating-point reductions use increasing index order, one
; accumulator, no FMA, and no host math.
;
; sigma is measured from the unnormalized final A*v, so for A=2I it is
; bit-exactly 2.0. u is written only after sigma has been validated.
;
; Static save slots: the kernel is non-reentrant and non-concurrent.
; mth_svd_top128_frame holds eight words:
;   [0] A, [1] x, [2] remaining iterations, [3] u output,
;   [4] v output, [5] sigma output, [6] scratch1, [7] scratch2.
; mth_svd_top128_return holds the single saved r15 link word. The link
; is saved before any nested call and restored on success and failure.

.include "mth_matvec128_f32.nas"

mth_svd_top128:
  ; Save the single-link return address before any nested call.
  MOVI r9,@mth_svd_top128_return
  ST r15,r9,0

  ; Bounded iteration count: 1..32.
  CMP r3,r0
  JMP[LE] @mth_svd_top128_invalid
  MOVI r9,32
  CMP r3,r9
  JMP[GT] @mth_svd_top128_invalid

  ; Save caller pointers and the iteration counter across nested calls.
  MOVI r10,@mth_svd_top128_frame
  ST r1,r10,0
  ST r2,r10,1
  ST r3,r10,2
  ST r4,r10,3
  ST r5,r10,4
  ST r6,r10,5
  ST r7,r10,6
  ST r8,r10,7

  ; v = x/||x||, sum of squares in increasing i.
  LD r2,r10,1
  LD r5,r10,4
  MOVI r8,0
  MOVI r9,0
mth_svd_top128_xnorm:
  ADD r11,r2,r8
  LD r11,r11,0
  FMUL r11,r11,r11
  FADD r9,r9,r11
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_xnorm
  FSQRT r9,r9
  FCMP r9,r0
  JMP[EQ] @mth_svd_top128_invalid
  MOVI r8,0
mth_svd_top128_xnorm_store:
  ADD r11,r2,r8
  LD r11,r11,0
  FDIV r11,r11,r9
  ADD r12,r5,r8
  ST r11,r12,0
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_xnorm_store

mth_svd_top128_iter:
  ; scratch1 = A*v using the verified dense matvec.
  MOVI r10,@mth_svd_top128_frame
  LD r1,r10,0
  LD r2,r10,4
  LD r3,r10,6
  CALL @mth_matvec128_f32
  CMP r1,r13
  JMP[NE] @mth_svd_top128_invalid

  ; u = scratch1/||scratch1||, sum of squares in increasing i.
  MOVI r10,@mth_svd_top128_frame
  LD r7,r10,6
  LD r4,r10,3
  MOVI r8,0
  MOVI r9,0
mth_svd_top128_unorm:
  ADD r11,r7,r8
  LD r11,r11,0
  FMUL r11,r11,r11
  FADD r9,r9,r11
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_unorm
  FSQRT r9,r9
  FCMP r9,r0
  JMP[EQ] @mth_svd_top128_invalid
  MOVI r8,0
mth_svd_top128_unorm_store:
  ADD r11,r7,r8
  LD r11,r11,0
  FDIV r11,r11,r9
  ADD r12,r4,r8
  ST r11,r12,0
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_unorm_store

  ; scratch2 = A^T*u with increasing row index per output column.
  MOVI r10,@mth_svd_top128_frame
  LD r1,r10,0
  LD r2,r10,3
  LD r3,r10,7
  CALL @mth_svd_top128_transpose
  CMP r1,r13
  JMP[NE] @mth_svd_top128_invalid

  ; v = scratch2/||scratch2||, sum of squares in increasing i.
  MOVI r10,@mth_svd_top128_frame
  LD r7,r10,7
  LD r5,r10,4
  MOVI r8,0
  MOVI r9,0
mth_svd_top128_vnorm:
  ADD r11,r7,r8
  LD r11,r11,0
  FMUL r11,r11,r11
  FADD r9,r9,r11
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_vnorm
  FSQRT r9,r9
  FCMP r9,r0
  JMP[EQ] @mth_svd_top128_invalid
  MOVI r8,0
mth_svd_top128_vnorm_store:
  ADD r11,r7,r8
  LD r11,r11,0
  FDIV r11,r11,r9
  ADD r12,r5,r8
  ST r11,r12,0
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_vnorm_store

  ; Bounded loop: stop when the saved counter reaches zero.
  LD r9,r10,2
  SUB r9,r9,r13
  ST r9,r10,2
  CMP r9,r0
  JMP[GT] @mth_svd_top128_iter

  ; Final scratch1 = A*v at the normalized v.
  MOVI r10,@mth_svd_top128_frame
  LD r1,r10,0
  LD r2,r10,4
  LD r3,r10,6
  CALL @mth_matvec128_f32
  CMP r1,r13
  JMP[NE] @mth_svd_top128_invalid

  ; sigma = ||scratch1||, increasing i.
  MOVI r10,@mth_svd_top128_frame
  LD r7,r10,6
  LD r6,r10,5
  MOVI r8,0
  MOVI r9,0
mth_svd_top128_sigma_norm:
  ADD r11,r7,r8
  LD r11,r11,0
  FMUL r11,r11,r11
  FADD r9,r9,r11
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_sigma_norm
  FSQRT r9,r9
  FCMP r9,r0
  JMP[EQ] @mth_svd_top128_invalid
  ST r9,r6,0

  ; u = scratch1/sigma, increasing i.
  LD r4,r10,3
  MOVI r8,0
mth_svd_top128_final_u:
  ADD r11,r7,r8
  LD r11,r11,0
  FDIV r11,r11,r9
  ADD r12,r4,r8
  ST r11,r12,0
  ADD r8,r8,r13
  MOVI r11,128
  CMP r8,r11
  JMP[LT] @mth_svd_top128_final_u

  MOVI r9,@mth_svd_top128_return
  LD r15,r9,0
  MOVI r1,1
  RET

mth_svd_top128_invalid:
  MOVI r9,@mth_svd_top128_return
  LD r15,r9,0
  MOVI r1,0
  RET

; Private A^T*u helper.
;   r1 = A, r2 = u, r3 = out, r13 = 1
;   returns r1=1; clobbers r4-r12, r14
; out[j] = sum_{i=0}^{127} A[i,j]*u[i], increasing i, one accumulator.
mth_svd_top128_transpose:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_svd_top128_transpose_col:
  MOVI r9,0
  ADD r10,r4,r7
  ADD r11,r5,r0
  MOVI r8,0
mth_svd_top128_transpose_row:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  MOVI r12,128
  ADD r10,r10,r12
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,128
  CMP r8,r12
  JMP[LT] @mth_svd_top128_transpose_row
  ADD r12,r6,r7
  ST r9,r12,0
  ADD r7,r7,r13
  MOVI r12,128
  CMP r7,r12
  JMP[LT] @mth_svd_top128_transpose_col
  MOVI r1,1
  RET

; Static save frame: A, x, remaining iterations, u, v, sigma,
; scratch1, scratch2. Non-reentrant/non-concurrent.
mth_svd_top128_frame:
.word 0,0,0,0,0,0,0,0
; Single saved r15 link word.
mth_svd_top128_return:
.word 0
