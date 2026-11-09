; MTH native 256x256 top-two singular-pair kernel — NaN16 float32 only.
;
; ABI:
;   r1  = A, 65,536 float32 payloads, row-major 256x256
;   r2  = x, 256 float32 input direction
;   r3  = iterations, integer 1..32
;   r4  = u1 output, 256 float32 (left singular direction, pair 1)
;   r5  = v1 output, 256 float32 (right singular direction, pair 1)
;   r6  = sigma1 output, one float32
;   r7  = u2 output, 256 float32 (left singular direction, pair 2)
;   r8  = v2 output, 256 float32 (right singular direction, pair 2)
;   r9  = sigma2 output, one float32
;   r10 = scratch, at least 1024 float32 (the kernel uses the first 256)
;   r13 = 1
;   returns r1=1 success
;   returns r1=0 for a zero input direction, nonpositive or out-of-range
;          iterations, a zero-norm iterate in either pair, or a zero
;          second singular value after deflation
;   clobbers r1-r12, r14
;
; First pair — two-sided power iteration on A^T A:
;
;   v1 = x / ||x||
;   repeat `iterations` times:
;     u1 = A v1;      u1 = u1 / ||u1||
;     v1 = A^T u1;    v1 = v1 / ||v1||
;   u1 = A v1; sigma1 = ||u1||; u1 = u1 / sigma1
;
; Second pair — the same two-sided iteration on the deflated operator
; P_u1 A P_v1, where P_z = I - z z^T, started from P_v1 x:
;
;   v2 = P_v1 x / ||P_v1 x||
;   repeat `iterations` times:
;     u2 = P_u1 (A v2);      u2 = u2 / ||u2||
;     v2 = P_v1 (A^T u2);    v2 = v2 / ||v2||
;   u2 = P_u1 (A v2); sigma2 = ||P_u1 (A v2)||; u2 = u2 / sigma2
;
; A is never written. The projection is applied after each matvec, before
; renormalization: A*v2 has its component along u1 removed and A^T*u2 has
; its component along v1 removed. A dot product is reduced in increasing
; index i with one accumulator; the subtraction and the norm accumulation
; are a second increasing-index pass; normalization is a third. If
; P_v1 x has zero norm (x lies exactly on v1), a deterministic fallback
; seed e_k is used, where k is the first index with the smallest v1[k]^2.
; That seed is projected against v1 before the iteration, so the second
; pair remains orthogonal to the first.
;
; sigma1 and sigma2 are measured from the unnormalized deflated matvec
; immediately before the final normalization. A sigma is written only
; after its norm has been validated. For A=diag(3,2,0,...) with x=e0 the
; outputs are bit-exactly sigma1=3.0, sigma2=2.0.
;
; All floating-point reductions use increasing index order, one
; accumulator, no FMA, and no host math.
;
; Static save slots: the kernel is non-reentrant and non-concurrent.
; mth_svd_top2_256_frame holds eleven words:
;   [0] A, [1] x, [2] remaining iterations, [3] u1 output,
;   [4] v1 output, [5] sigma1 output, [6] u2 output,
;   [7] v2 output, [8] sigma2 output, [9] scratch,
;   [10] initial iteration count (reloaded for the second pair).
; mth_svd_top2_256_return holds the single saved r15 link word. The link
; is saved before any nested call and restored on success and failure.
; The four private helpers are leaves, so the one saved link remains
; valid across every CALL and the helpers do not save r15 themselves.

mth_svd_top2_256:
  ; Save the single-link return address before any nested call.
  MOVI r11,@mth_svd_top2_256_return
  ST r15,r11,0

  ; Bounded iteration count: 1..32.
  CMP r3,r0
  JMP[LE] @mth_svd_top2_256_invalid
  MOVI r11,32
  CMP r3,r11
  JMP[GT] @mth_svd_top2_256_invalid

  ; Save caller pointers and the iteration counter across nested calls.
  MOVI r12,@mth_svd_top2_256_frame
  ST r1,r12,0
  ST r2,r12,1
  ST r3,r12,2
  ST r4,r12,3
  ST r5,r12,4
  ST r6,r12,5
  ST r7,r12,6
  ST r8,r12,7
  ST r9,r12,8
  ST r10,r12,9
  ST r3,r12,10

  ; v1 = x/||x||: copy x to v1, then normalize v1 in place.
  LD r2,r12,1
  LD r5,r12,4
  MOVI r4,0
mth_svd_top2_256_copy_x:
  ADD r6,r2,r4
  LD r6,r6,0
  ADD r7,r5,r4
  ST r6,r7,0
  ADD r4,r4,r13
  MOVI r6,256
  CMP r4,r6
  JMP[LT] @mth_svd_top2_256_copy_x
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,4
  CALL @mth_svd_top2_256_normalize
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

mth_svd_top2_256_first_iter:
  ; u1 = normalize(A*v1).
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,4
  LD r3,r12,3
  CALL @mth_svd_top2_256_matvec
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,3
  CALL @mth_svd_top2_256_normalize
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

  ; v1 = normalize(A^T*u1).
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,3
  LD r3,r12,4
  CALL @mth_svd_top2_256_transpose
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,4
  CALL @mth_svd_top2_256_normalize
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

  ; Bounded loop: stop when the saved counter reaches zero.
  MOVI r12,@mth_svd_top2_256_frame
  LD r11,r12,2
  SUB r11,r11,r13
  ST r11,r12,2
  CMP r11,r0
  JMP[GT] @mth_svd_top2_256_first_iter

  ; Final u1 = A*v1 and sigma1 = ||A*v1||.
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,4
  LD r3,r12,3
  CALL @mth_svd_top2_256_matvec
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,3
  CALL @mth_svd_top2_256_normalize
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r11,r12,5
  ST r2,r11,0

  ; Second seed: v2 = P_v1 x / ||P_v1 x||.
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,1
  LD r2,r12,4
  LD r3,r12,7
  CALL @mth_svd_top2_256_project_norm
  CMP r1,r13
  JMP[EQ] @mth_svd_top2_256_second_ready

  ; x was exactly on v1. Pick k minimizing v1[k]^2, put e_k in v2, and
  ; project that coordinate seed against v1.
  MOVI r12,@mth_svd_top2_256_frame
  LD r5,r12,4
  LD r6,r5,0
  FMUL r6,r6,r6
  MOVI r7,1
  MOVI r8,0
mth_svd_top2_256_min_loop:
  ADD r9,r5,r7
  LD r9,r9,0
  FMUL r9,r9,r9
  FCMP r9,r6
  JMP[GE] @mth_svd_top2_256_min_skip
  ADD r6,r9,r0
  ADD r8,r7,r0
mth_svd_top2_256_min_skip:
  ADD r7,r7,r13
  MOVI r9,256
  CMP r7,r9
  JMP[LT] @mth_svd_top2_256_min_loop

  MOVI r12,@mth_svd_top2_256_frame
  LD r4,r12,7
  MOVI r7,0
mth_svd_top2_256_zero_v2:
  ST r0,r4,0
  ADD r4,r4,r13
  ADD r7,r7,r13
  MOVI r9,256
  CMP r7,r9
  JMP[LT] @mth_svd_top2_256_zero_v2
  MOVI r12,@mth_svd_top2_256_frame
  LD r4,r12,7
  ADD r4,r4,r8
  MOVI r9,0
  MOVHI r9,0x3F80
  ST r9,r4,0
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,7
  LD r2,r12,4
  LD r3,r12,7
  CALL @mth_svd_top2_256_project_norm
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

mth_svd_top2_256_second_ready:
  ; Restore the iteration count for the deflated second pair.
  MOVI r12,@mth_svd_top2_256_frame
  LD r11,r12,10
  ST r11,r12,2

mth_svd_top2_256_second_iter:
  ; scratch = A*v2; u2 = normalize(P_u1 scratch).
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,7
  LD r3,r12,9
  CALL @mth_svd_top2_256_matvec
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,9
  LD r2,r12,3
  LD r3,r12,6
  CALL @mth_svd_top2_256_project_norm
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

  ; scratch = A^T*u2; v2 = normalize(P_v1 scratch).
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,6
  LD r3,r12,9
  CALL @mth_svd_top2_256_transpose
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,9
  LD r2,r12,4
  LD r3,r12,7
  CALL @mth_svd_top2_256_project_norm
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid

  ; Bounded loop: stop when the saved counter reaches zero.
  MOVI r12,@mth_svd_top2_256_frame
  LD r11,r12,2
  SUB r11,r11,r13
  ST r11,r12,2
  CMP r11,r0
  JMP[GT] @mth_svd_top2_256_second_iter

  ; Final u2 = P_u1 (A*v2) and sigma2 = ||P_u1 (A*v2)||.
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,0
  LD r2,r12,7
  LD r3,r12,9
  CALL @mth_svd_top2_256_matvec
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r1,r12,9
  LD r2,r12,3
  LD r3,r12,6
  CALL @mth_svd_top2_256_project_norm
  CMP r1,r13
  JMP[NE] @mth_svd_top2_256_invalid
  MOVI r12,@mth_svd_top2_256_frame
  LD r11,r12,8
  ST r2,r11,0

  MOVI r11,@mth_svd_top2_256_return
  LD r15,r11,0
  MOVI r1,1
  RET

mth_svd_top2_256_invalid:
  MOVI r11,@mth_svd_top2_256_return
  LD r15,r11,0
  MOVI r1,0
  RET

; Private A*v helper.
;   r1 = A, r2 = v, r3 = out, r13 = 1
;   returns r1=1; clobbers r4-r12, r14
; out[i] = sum_{j=0}^{255} A[i,j]*v[j], increasing j, one accumulator.
mth_svd_top2_256_matvec:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  ADD r10,r4,r0
  MOVI r7,0
mth_svd_top2_256_matvec_row:
  MOVI r9,0
  ADD r11,r5,r0
  MOVI r8,0
mth_svd_top2_256_matvec_col:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,256
  CMP r8,r12
  JMP[LT] @mth_svd_top2_256_matvec_col
  ADD r12,r6,r7
  ST r9,r12,0
  ADD r7,r7,r13
  MOVI r12,256
  CMP r7,r12
  JMP[LT] @mth_svd_top2_256_matvec_row
  MOVI r1,1
  RET

; Private A^T*u helper.
;   r1 = A, r2 = u, r3 = out, r13 = 1
;   returns r1=1; clobbers r4-r12, r14
; out[j] = sum_{i=0}^{255} A[i,j]*u[i], increasing i, one accumulator.
mth_svd_top2_256_transpose:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_svd_top2_256_transpose_col:
  MOVI r9,0
  ADD r10,r4,r7
  ADD r11,r5,r0
  MOVI r8,0
mth_svd_top2_256_transpose_row:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  MOVI r12,256
  ADD r10,r10,r12
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,256
  CMP r8,r12
  JMP[LT] @mth_svd_top2_256_transpose_row
  ADD r12,r6,r7
  ST r9,r12,0
  ADD r7,r7,r13
  MOVI r12,256
  CMP r7,r12
  JMP[LT] @mth_svd_top2_256_transpose_col
  MOVI r1,1
  RET

; Private in-place normalization.
;   r1 = vector, r13 = 1
;   returns r1=1 and r2=||vector|| on success
;   returns r1=0 without writing the vector when ||vector|| == 0
;   clobbers r3-r8
; norm = sqrt(sum v[i]^2), increasing i, one accumulator; then
; v[i] = v[i]/norm, increasing i.
mth_svd_top2_256_normalize:
  ADD r4,r1,r0
  MOVI r5,0
  MOVI r6,0
mth_svd_top2_256_norm_sum:
  ADD r7,r4,r5
  LD r7,r7,0
  FMUL r7,r7,r7
  FADD r6,r6,r7
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_svd_top2_256_norm_sum
  FSQRT r6,r6
  FCMP r6,r0
  JMP[EQ] @mth_svd_top2_256_norm_zero
  MOVI r5,0
mth_svd_top2_256_norm_store:
  ADD r7,r4,r5
  LD r8,r7,0
  FDIV r8,r8,r6
  ST r8,r7,0
  ADD r5,r5,r13
  MOVI r8,256
  CMP r5,r8
  JMP[LT] @mth_svd_top2_256_norm_store
  ADD r2,r6,r0
  MOVI r1,1
  RET
mth_svd_top2_256_norm_zero:
  MOVI r1,0
  RET

; Private projection and in-place normalization.
;   r1 = base, r2 = ref, r3 = out, r13 = 1
;   out[i] = base[i] - (sum_j ref[j]*base[j]) * ref[i], then out normalized
;   returns r1=1 and r2=||out_before_normalization|| on success
;   returns r1=0 without normalizing when that norm == 0
;   BASE and OUT may be the same span; REF must be distinct from OUT.
;   The dot is reduced in increasing j with one accumulator. The subtract
;   and norm accumulation are a second increasing-index pass; the divide
;   is a third. Clobbers r3-r12.
mth_svd_top2_256_project_norm:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
  MOVI r9,0
mth_svd_top2_256_proj_dot:
  ADD r10,r4,r7
  LD r10,r10,0
  ADD r11,r5,r7
  LD r11,r11,0
  FMUL r10,r10,r11
  FADD r9,r9,r10
  ADD r7,r7,r13
  MOVI r10,256
  CMP r7,r10
  JMP[LT] @mth_svd_top2_256_proj_dot
  MOVI r7,0
  MOVI r8,0
mth_svd_top2_256_proj_sub:
  ADD r10,r4,r7
  LD r10,r10,0
  ADD r11,r5,r7
  LD r11,r11,0
  FMUL r11,r11,r9
  FSUB r10,r10,r11
  ADD r12,r6,r7
  ST r10,r12,0
  FMUL r10,r10,r10
  FADD r8,r8,r10
  ADD r7,r7,r13
  MOVI r10,256
  CMP r7,r10
  JMP[LT] @mth_svd_top2_256_proj_sub
  FSQRT r8,r8
  FCMP r8,r0
  JMP[EQ] @mth_svd_top2_256_proj_zero
  MOVI r7,0
mth_svd_top2_256_proj_norm:
  ADD r10,r6,r7
  LD r11,r10,0
  FDIV r11,r11,r8
  ST r11,r10,0
  ADD r7,r7,r13
  MOVI r11,256
  CMP r7,r11
  JMP[LT] @mth_svd_top2_256_proj_norm
  ADD r2,r8,r0
  MOVI r1,1
  RET
mth_svd_top2_256_proj_zero:
  MOVI r1,0
  RET

; Static save frame: A, x, remaining iterations, u1, v1, sigma1,
; u2, v2, sigma2, scratch. Non-reentrant/non-concurrent.
mth_svd_top2_256_frame:
.word 0,0,0,0,0,0,0,0,0,0,0
; Single saved r15 link word.
mth_svd_top2_256_return:
.word 0
