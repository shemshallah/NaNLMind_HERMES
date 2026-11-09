; MTH bounded deterministic 256x256 symmetric top-k kernel — NaN16 float32 only.
;
; ABI:
;   r1  = A, 65,536 float32 payloads, row-major 256x256 symmetric
;   r2  = q0, 256 float32 start direction (not modified)
;   r3  = k, integer 1..8
;   r4  = iterations per vector, integer 1..32
;   r5  = values_out, k float32 (descending Ritz values)
;   r6  = vectors_out, k*256 float32 (Ritz vectors, contiguous, column j at
;         vectors_out + j*256)
;   r7  = scratch, at least 1024 float32 (the kernel uses the first 256)
;   r13 = 1
;   returns r1=1 success
;   returns r1=0 for k outside 1..8, iterations outside 1..32, a zero q0,
;          an exactly exhausted deflated direction, or a zero-norm iterate
;   clobbers r1-r12, r14
;
; Method. The ABI fixes the stored basis at k vectors and gives a bounded
; "iterations per vector" count, so this kernel implements the bounded
; power-iteration Ritz extraction path rather than a growing three-term
; Lanczos recurrence (which would need up to iterations+1 basis vectors).
; It is a Lanczos-style orthogonal (deflated) power iteration with full
; two-pass reorthogonalization against every previously locked basis
; column:
;
;   1. Build an orthonormal initial basis from q0. Column 0 is q0/||q0||.
;      Column 1 uses the deterministic axis scan directly because q0 is
;      exactly inside span(V_0). Columns j>1 start as q0 with columns
;      0..j-1 projected out. If that is exactly zero, a deterministic
;      fallback scans the coordinate axes e_p, p = 0,1,...,255, takes the
;      first residual with nonzero norm, orthonormalizes it against
;      columns 0..j-1, and uses it. The scan is deterministic and bounded.
;   2. For j = 0..k-1, run exactly `iterations` steps of
;         w = A * V_j;  project out V_0..V_{j-1} (two passes);
;         V_j = w / ||w||.
;      A zero norm after projection is a degenerate direction and returns 0.
;      Columns 0..j-1 are locked, so each finished column is orthonormal to
;      all earlier columns. Later seeds are reprocessed and reorthogonalized
;      when their turn arrives.
;   3. Compute the Rayleigh quotient lambda_j = V_j^T A V_j for every
;      column, then selection-sort the (lambda, column) pairs in descending
;      order. Those Rayleigh quotients are the Ritz values of the k
;      one-dimensional subspaces spanned by the orthonormal columns.
;
; All arithmetic is NaN16 float32. Every reduction is an increasing-index,
; one-accumulator sum. There is no FMA, no reassociation, no host math, and
; no randomness. The same inputs produce the same outputs bit for bit.
;
; A is never written. q0 is never written. values_out is written only after
; every column has been computed; on a rejected call it is untouched.
; vectors_out may contain partial intermediate columns after a rejection.
;
; Static save frame (non-reentrant, non-concurrent):
;   [0] A            [1] q0           [2] k            [3] iterations
;   [4] values_out   [5] vectors_out  [6] scratch      [7] column index
;   [8] remaining iterations          [9] unused       [10] unused
;   [11] w workspace [12] unused      [13] fallback axis scan index
; mth_lanczos_256_return holds the single saved r15 link word, saved before
; any nested call and restored on both the success and invalid paths. The
; private helpers are leaves, so that one saved link remains valid across
; every CALL and the helpers do not save r15 themselves.
;
; Working set (caller spans): A 65,536 + vectors_out up to 2,048 +
; scratch 1,024 + q0 256 + values 8 = 68,872 words = 550,976 bytes. A
; fixture that reserves the maximum k=8 spans is about 73,000 words,
; inside the 131,072-word / 1 MiB guest target and above the historical
; 65,536-word / 512 KiB budget.

mth_lanczos_256:
  ; Save the single-link return address before any nested call.
  MOVI r12,@mth_lanczos_256_return
  ST r15,r12,0

  ; Bounded k: 1..8.
  CMP r3,r0
  JMP[LE] @mth_lanczos_256_invalid
  MOVI r12,8
  CMP r3,r12
  JMP[GT] @mth_lanczos_256_invalid

  ; Bounded iterations-per-vector: 1..32.
  CMP r4,r0
  JMP[LE] @mth_lanczos_256_invalid
  MOVI r12,32
  CMP r4,r12
  JMP[GT] @mth_lanczos_256_invalid

  ; Save caller-owned pointers and counters across nested calls.
  MOVI r12,@mth_lanczos_256_frame
  ST r1,r12,0
  ST r2,r12,1
  ST r3,r12,2
  ST r4,r12,3
  ST r5,r12,4
  ST r6,r12,5
  ST r7,r12,6

  ; w is the first 256 words of scratch. The contract allows 1024 words;
  ; the remaining space is reserved for callers/future variants.
  ADD r8,r7,r0
  ST r8,r12,11

  ; V_0 = q0.
  MOVI r12,@mth_lanczos_256_frame
  LD r2,r12,1
  LD r3,r12,5
  MOVI r4,0
mth_lanczos_256_copy_q0:
  ADD r5,r2,r4
  LD r5,r5,0
  ADD r6,r3,r4
  ST r5,r6,0
  ADD r4,r4,r13
  MOVI r5,256
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_copy_q0

  ; Normalize V_0. A zero q0 is degenerate.
  MOVI r12,@mth_lanczos_256_frame
  LD r1,r12,5
  CALL @mth_lanczos_256_normalize
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_invalid

  ; Build initial orthonormal columns 1..k-1.
  MOVI r8,1
  MOVI r12,@mth_lanczos_256_frame
  ST r8,r12,7

mth_lanczos_256_build:
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r9,r12,2
  CMP r8,r9
  JMP[GE] @mth_lanczos_256_build_done

  ; q0 lies exactly in span(V_0) for j=1. Use the deterministic axis
  ; seed directly instead of normalizing cancellation noise.
  CMP r8,r13
  JMP[EQ] @mth_lanczos_256_axis_seed

  ; V_j = q0 (copy).
  LD r2,r12,1
  LD r3,r12,5
  MOVI r4,0
  ADD r5,r8,r0
  SHL r5,r5,8
  ADD r3,r3,r5
mth_lanczos_256_build_copy:
  ADD r5,r2,r4
  LD r5,r5,0
  ADD r6,r3,r4
  ST r5,r6,0
  ADD r4,r4,r13
  MOVI r5,256
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_build_copy

  ; Project V_j against columns 0..j-1 (two passes), then normalize.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r1,r1,r9
  LD r2,r12,5
  ADD r3,r8,r0
  CALL @mth_lanczos_256_orth

  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r1,r1,r9
  CALL @mth_lanczos_256_normalize
  CMP r1,r13
  JMP[EQ] @mth_lanczos_256_build_next

  ; Scan coordinate axes in increasing order and take the first nonzero
  ; residual. This path is reached when the q0 seed is exhausted and
  ; directly for the j=1 column.
mth_lanczos_256_axis_seed:
  MOVI r8,0
  MOVI r12,@mth_lanczos_256_frame
  ST r8,r12,13

mth_lanczos_256_fallback:
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,13
  MOVI r4,256
  CMP r8,r4
  JMP[GE] @mth_lanczos_256_invalid

  ; Zero V_j.
  LD r9,r12,7
  LD r2,r12,5
  ADD r3,r9,r0
  SHL r3,r3,8
  ADD r2,r2,r3
  MOVI r4,0
mth_lanczos_256_fallback_zero:
  ADD r5,r2,r4
  ST r0,r5,0
  ADD r4,r4,r13
  MOVI r5,256
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_fallback_zero

  ; V_j[p] = 1.0.
  ADD r5,r2,r8
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r5,0

  ; Orthonormalize the axis seed against columns 0..j-1.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r1,r1,r9
  LD r2,r12,5
  ADD r3,r8,r0
  CALL @mth_lanczos_256_orth

  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r1,r1,r9
  CALL @mth_lanczos_256_normalize
  CMP r1,r13
  JMP[EQ] @mth_lanczos_256_build_next

  ; This axis was already in the span. Try the next one.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,13
  ADD r8,r8,r13
  ST r8,r12,13
  JMP @mth_lanczos_256_fallback

mth_lanczos_256_build_next:
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  ADD r8,r8,r13
  ST r8,r12,7
  JMP @mth_lanczos_256_build

mth_lanczos_256_build_done:
  ; Now run the bounded deflated power iteration for each stored column.
  MOVI r8,0
  MOVI r12,@mth_lanczos_256_frame
  ST r8,r12,7

mth_lanczos_256_outer:
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r9,r12,2
  CMP r8,r9
  JMP[GE] @mth_lanczos_256_outer_done

  LD r10,r12,3
  ST r10,r12,8

mth_lanczos_256_inner:
  ; w = A * V_j.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,0
  LD r2,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r2,r2,r9
  LD r3,r12,11
  CALL @mth_lanczos_256_matvec

  ; Project w against locked columns 0..j-1 (two passes).
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,11
  LD r2,r12,5
  ADD r3,r8,r0
  CALL @mth_lanczos_256_orth

  ; V_j = w / ||w||. A zero norm is degenerate (for example A = 0).
  MOVI r12,@mth_lanczos_256_frame
  LD r1,r12,11
  CALL @mth_lanczos_256_normalize
  CMP r1,r13
  JMP[NE] @mth_lanczos_256_invalid

  ; Copy w into V_j.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r2,r12,11
  LD r3,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r3,r3,r9
  MOVI r4,0
mth_lanczos_256_inner_copy:
  ADD r5,r2,r4
  LD r5,r5,0
  ADD r6,r3,r4
  ST r5,r6,0
  ADD r4,r4,r13
  MOVI r5,256
  CMP r4,r5
  JMP[LT] @mth_lanczos_256_inner_copy

  ; Bounded loop: stop when the saved per-vector counter reaches zero.
  MOVI r12,@mth_lanczos_256_frame
  LD r10,r12,8
  SUB r10,r10,r13
  ST r10,r12,8
  CMP r10,r0
  JMP[GT] @mth_lanczos_256_inner

  ; Next column.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  ADD r8,r8,r13
  ST r8,r12,7
  JMP @mth_lanczos_256_outer

mth_lanczos_256_outer_done:
  ; Rayleigh quotients lambda_j = V_j^T A V_j.
  MOVI r8,0
  MOVI r12,@mth_lanczos_256_frame
  ST r8,r12,7

mth_lanczos_256_rayleigh:
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r9,r12,2
  CMP r8,r9
  JMP[GE] @mth_lanczos_256_sort

  ; w = A * V_j.
  LD r1,r12,0
  LD r2,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r2,r2,r9
  LD r3,r12,11
  CALL @mth_lanczos_256_matvec

  ; lambda = dot(V_j, w), increasing index, one accumulator.
  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r1,r12,5
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r1,r1,r9
  LD r2,r12,11
  CALL @mth_lanczos_256_dot

  MOVI r12,@mth_lanczos_256_frame
  LD r8,r12,7
  LD r9,r12,4
  ADD r9,r9,r8
  ST r1,r9,0

  LD r8,r12,7
  ADD r8,r8,r13
  ST r8,r12,7
  JMP @mth_lanczos_256_rayleigh

mth_lanczos_256_sort:
  ; Selection sort: descending values with the matching vectors_out columns.
  MOVI r12,@mth_lanczos_256_frame
  LD r11,r12,2
  LD r10,r12,5
  LD r9,r12,4
  MOVI r8,0

mth_lanczos_256_sort_outer:
  CMP r8,r11
  JMP[GE] @mth_lanczos_256_success

  ADD r7,r8,r0                 ; m = i
  ADD r3,r9,r7
  LD r5,r3,0                   ; value_m
  ADD r6,r8,r13                ; j = i+1

mth_lanczos_256_sort_scan:
  CMP r6,r11
  JMP[GE] @mth_lanczos_256_sort_scan_done
  ADD r3,r9,r6
  LD r4,r3,0                   ; value_j
  FCMP r4,r5
  JMP[LE] @mth_lanczos_256_sort_scan_next
  ADD r7,r6,r0                 ; m = j
  ADD r5,r4,r0                 ; value_m = value_j
mth_lanczos_256_sort_scan_next:
  ADD r6,r6,r13
  JMP @mth_lanczos_256_sort_scan

mth_lanczos_256_sort_scan_done:
  ; Swap values[i] and values[m].
  ADD r3,r9,r8
  ADD r2,r9,r7
  LD r1,r3,0
  LD r4,r2,0
  ST r4,r3,0
  ST r1,r2,0

  CMP r7,r8
  JMP[EQ] @mth_lanczos_256_sort_next_i

  ; Swap vectors_out columns i and m through w.
  MOVI r1,256
  MUL r2,r8,r1
  ADD r2,r10,r2                ; V_i
  MUL r3,r7,r1
  ADD r3,r10,r3                ; V_m
  MOVI r12,@mth_lanczos_256_frame
  LD r4,r12,11                 ; w

  MOVI r6,0
mth_lanczos_256_sort_copy_i:
  ADD r1,r2,r6
  LD r1,r1,0
  ADD r14,r4,r6
  ST r1,r14,0
  ADD r6,r6,r13
  MOVI r1,256
  CMP r6,r1
  JMP[LT] @mth_lanczos_256_sort_copy_i

  MOVI r6,0
mth_lanczos_256_sort_copy_m:
  ADD r1,r3,r6
  LD r1,r1,0
  ADD r14,r2,r6
  ST r1,r14,0
  ADD r6,r6,r13
  MOVI r1,256
  CMP r6,r1
  JMP[LT] @mth_lanczos_256_sort_copy_m

  MOVI r6,0
mth_lanczos_256_sort_copy_w:
  ADD r1,r4,r6
  LD r1,r1,0
  ADD r14,r3,r6
  ST r1,r14,0
  ADD r6,r6,r13
  MOVI r1,256
  CMP r6,r1
  JMP[LT] @mth_lanczos_256_sort_copy_w

mth_lanczos_256_sort_next_i:
  ADD r8,r8,r13
  JMP @mth_lanczos_256_sort_outer

mth_lanczos_256_success:
  MOVI r12,@mth_lanczos_256_return
  LD r15,r12,0
  MOVI r1,1
  RET

mth_lanczos_256_invalid:
  MOVI r12,@mth_lanczos_256_return
  LD r15,r12,0
  MOVI r1,0
  RET

; Private A*v helper.
;   r1 = A, r2 = v, r3 = out, r13 = 1
;   returns r1=1; clobbers r4-r12, r14
; out[i] = sum_{j=0}^{255} A[i,j]*v[j], increasing j, one accumulator.
; The A and out pointers advance one row and one word per output row.
mth_lanczos_256_matvec:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_lanczos_256_matvec_row:
  MOVI r9,0
  ADD r10,r4,r0
  ADD r11,r5,r0
  MOVI r8,0
mth_lanczos_256_matvec_col:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,256
  CMP r8,r12
  JMP[LT] @mth_lanczos_256_matvec_col
  ST r9,r6,0
  MOVI r12,256
  ADD r4,r4,r12
  ADD r6,r6,r13
  ADD r7,r7,r13
  MOVI r12,256
  CMP r7,r12
  JMP[LT] @mth_lanczos_256_matvec_row
  MOVI r1,1
  RET

; Private in-place normalization.
;   r1 = vector, r13 = 1
;   returns r1=1 and r2=||vector|| on success
;   returns r1=0 without writing the vector when ||vector|| == 0
;   clobbers r3-r8
mth_lanczos_256_normalize:
  ADD r4,r1,r0
  MOVI r5,0
  MOVI r6,0
mth_lanczos_256_norm_sum:
  ADD r7,r4,r5
  LD r7,r7,0
  FMUL r7,r7,r7
  FADD r6,r6,r7
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_lanczos_256_norm_sum
  FSQRT r6,r6
  FCMP r6,r0
  JMP[EQ] @mth_lanczos_256_norm_zero
  MOVI r5,0
mth_lanczos_256_norm_store:
  ADD r7,r4,r5
  LD r8,r7,0
  FDIV r8,r8,r6
  ST r8,r7,0
  ADD r5,r5,r13
  MOVI r8,256
  CMP r5,r8
  JMP[LT] @mth_lanczos_256_norm_store
  ADD r2,r6,r0
  MOVI r1,1
  RET
mth_lanczos_256_norm_zero:
  MOVI r1,0
  RET

; Private two-pass full reorthogonalization.
;   r1 = vector, r2 = basis, r3 = count, r13 = 1
;   vector[i] -= sum_{t=0}^{255} basis[col*256+t]*vector[t] for every
;   locked column 0..count-1, then the whole pass is repeated once.
;   Each dot is an increasing-index, one-accumulator reduction.
;   Clobbers r3-r12, r14. Returns r1=1.
mth_lanczos_256_orth:
  ADD r4,r1,r0                 ; vector
  ADD r5,r2,r0                 ; basis
  ADD r6,r3,r0                 ; count
  MOVI r7,0                    ; pass
mth_lanczos_256_orth_pass:
  MOVI r8,0                    ; column
mth_lanczos_256_orth_col:
  CMP r8,r6
  JMP[GE] @mth_lanczos_256_orth_pass_done

  ; colptr = basis + column*256.
  ADD r9,r8,r0
  SHL r9,r9,8
  ADD r9,r5,r9

  ; dot = sum_t colptr[t]*vector[t], increasing t, one accumulator.
  MOVI r10,0
  MOVI r11,0
mth_lanczos_256_orth_dot:
  ADD r12,r9,r10
  LD r12,r12,0
  ADD r14,r4,r10
  LD r14,r14,0
  FMUL r12,r12,r14
  FADD r11,r11,r12
  ADD r10,r10,r13
  MOVI r12,256
  CMP r10,r12
  JMP[LT] @mth_lanczos_256_orth_dot

  ; vector[t] -= dot*colptr[t], increasing t.
  MOVI r10,0
mth_lanczos_256_orth_sub:
  ADD r12,r9,r10
  LD r12,r12,0
  FMUL r12,r12,r11
  ADD r3,r4,r10
  LD r14,r3,0
  FSUB r14,r14,r12
  ST r14,r3,0
  ADD r10,r10,r13
  MOVI r12,256
  CMP r10,r12
  JMP[LT] @mth_lanczos_256_orth_sub

  ADD r8,r8,r13
  JMP @mth_lanczos_256_orth_col

mth_lanczos_256_orth_pass_done:
  ADD r7,r7,r13
  MOVI r12,2
  CMP r7,r12
  JMP[LT] @mth_lanczos_256_orth_pass
  MOVI r1,1
  RET

; Private 256-word dot product.
;   r1 = a, r2 = b
;   returns r1 = sum a[i]*b[i], increasing i, one accumulator
;   clobbers r3-r8
mth_lanczos_256_dot:
  ADD r3,r1,r0
  ADD r4,r2,r0
  MOVI r5,0
  MOVI r6,0
mth_lanczos_256_dot_loop:
  ADD r7,r3,r5
  LD r7,r7,0
  ADD r8,r4,r5
  LD r8,r8,0
  FMUL r7,r7,r8
  FADD r6,r6,r7
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_lanczos_256_dot_loop
  ADD r1,r6,r0
  RET

; Static save frame. Non-reentrant and non-concurrent.
mth_lanczos_256_frame:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0
; Single saved r15 link word.
mth_lanczos_256_return:
.word 0
