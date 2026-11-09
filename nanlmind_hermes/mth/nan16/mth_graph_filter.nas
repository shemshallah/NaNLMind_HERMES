; MTH bounded degree-2 polynomial graph filter, Q16.16/I32 v1 — NaN16 only.
;
; Applies the declared polynomial
;
;   y_i = c0*x_i + c1*(Lx)_i + c2*(L^2 x)_i
;
; to a 256-word signal x, where L is the exact unweighted symmetric simple
; graph Laplacian represented by canonical CSR.  The 65,536 x 65,536
; Kronecker/pair-space matrix is never formed; only 256-word vectors and the
; bounded CSR edge list are touched.
;
; ABI (r13 = 1):
;   r1 = CSR row_offsets, 257 U32 words; offsets[0]=0, offsets[256]=edge_count
;   r2 = CSR neighbors, edge_count U32 words; each endpoint in [0,255]
;   r3 = x, 256 signed I32 Q16.16 words, each in [-4096,+4096]
;   r4 = coefficients, 3 signed I32 Q16.16 words, each in [-65536,+65536]
;   r5 = output y, 256 signed I32 Q16.16 words
;   r6 = caller-owned scratch_lx, 256 signed I32 words
;   r7 = caller-owned scratch_l2x, 256 signed I32 words
;   r8 = edge_count, 0..65280 (each undirected edge stored twice)
;   r9 = caller-owned 16-word U32 control scratch frame
;   returns r1=1 success,
;           r1=0 invalid count/state/coefficient,
;           r1=2 invalid CSR offsets/degree,
;           r1=3 invalid endpoint/order/self-loop
;   clobbers r1-r12, r14; preserves r13.  Leaf graph computation with two
;   internal CALL helpers; no static scratch and no host arithmetic.
;
; Numeric contract:
;   |x_i| <= 2^12.  Each row has at most 255 neighbors, so
;     |(Lx)_i|   <= 255 * 2^13 = 2,088,960
;     |(L^2x)_i| <= 255 * 2 * 2,088,960 = 255^2 * 2^14
;                = 1,065,369,600 < 2^31.
;   Both Laplacian applications are exact signed I32 arithmetic with no
;   rounding, saturation, or intermediate overflow.
;
;   Coefficients are Q16.16 in [-1,+1].  Each product c_k * v_k is formed as
;   an exact signed 64-bit two's-complement integer with a shift-and-add
;   loop.  The three Q32.32 products sum to at most
;     3 * 2^16 * 1,065,369,600 < 2^48,
;   which fits signed I64.  The sum is shifted right by 16 with arithmetic
;   truncation toward negative infinity and stored as I32 Q16.16.  The
;   declared bounds prove the final value is in
;     [-1,067,462,656, +1,067,462,656] < 2^31,
;   so no saturation or wrap is possible.
;
; CSR contract:
;   Canonical symmetric simple graph: reciprocal entries and an immutable
;   graph digest validated by the caller/runner.  This kernel validates count
;   bounds, offset[0]=0, monotone offsets, final edge count, row degree <=255,
;   endpoint range, strictly sorted unique neighbors, no self-loops, every x
;   word, and all three coefficients.  All validation finishes before the
;   first vector-scratch or output write, so a rejected descriptor leaves the
;   output and both vector scratch pages unchanged.  The 16-word control frame
;   is written first and may be clobbered on rejection.  Symmetry is bound by
;   the graph digest.  Input, output, and scratch spans must not overlap.
;
; Control scratch frame layout (r9):
;   1=row_offsets, 2=neighbors, 3=x, 4=coefficients, 5=output,
;   6=scratch_lx, 7=scratch_l2x, 8=edge_count, 9=loop index.
; Words 0, 10..15 are unused and need not be initialized by the caller.

mth_graph_filter_poly2_q16_16:
  ADD r14,r15,r0

  ST r1,r9,1
  ST r2,r9,2
  ST r3,r9,3
  ST r4,r9,4
  ST r5,r9,5
  ST r6,r9,6
  ST r7,r9,7
  ST r8,r9,8

  ; Edge-count bound.  A symmetric simple graph stores each edge twice.
  CMP r8,r0
  JMP[LT] @mth_graph_filter_invalid_state
  MOVI r10,65280
  CMP r8,r10
  JMP[GT] @mth_graph_filter_invalid_state

  ; Validate every x word: signed Q16.16 in [-4096,+4096].
  ADD r6,r3,r0
  MOVI r7,256
mth_graph_filter_validate_x:
  LD r8,r6,0
  SHR r10,r8,12
  CMP r10,r0
  JMP[EQ] @mth_graph_filter_x_valid
  MOVI r11,1
  CMP r10,r11
  JMP[NE] @mth_graph_filter_x_check_negative
  MOVI r11,4096
  CMP r8,r11
  JMP[EQ] @mth_graph_filter_x_valid
  JMP @mth_graph_filter_invalid_state
mth_graph_filter_x_check_negative:
  ; Negative values in [-4096,-1] have unsigned SHR 12 == 0x000FFFFF.
  MOVI r11,0xFFFF
  MOVHI r11,0x000F
  CMP r10,r11
  JMP[EQ] @mth_graph_filter_x_valid
  JMP @mth_graph_filter_invalid_state
mth_graph_filter_x_valid:
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_graph_filter_validate_x

  ; Validate all three coefficients: signed Q16.16 in [-65536,+65536].
  ADD r6,r4,r0
  MOVI r7,3
mth_graph_filter_validate_coeff:
  LD r8,r6,0
  SHR r10,r8,31
  CMP r10,r0
  JMP[NE] @mth_graph_filter_coeff_check_negative
  MOVI r11,0
  MOVHI r11,1
  CMP r8,r11
  JMP[GT] @mth_graph_filter_invalid_state
  JMP @mth_graph_filter_coeff_valid
mth_graph_filter_coeff_check_negative:
  ; Negative values in [-65536,-1] have unsigned SHR 16 == 0xFFFF.
  SHR r10,r8,16
  MOVI r11,0xFFFF
  CMP r10,r11
  JMP[NE] @mth_graph_filter_invalid_state
mth_graph_filter_coeff_valid:
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_graph_filter_validate_coeff

  ; Validate all 257 CSR offsets before any scratch/output write.
  ADD r6,r1,r0
  LD r10,r6,0
  CMP r10,r0
  JMP[NE] @mth_graph_filter_invalid_offsets
  MOVI r7,257
  MOVI r8,0
mth_graph_filter_validate_offsets:
  LD r10,r6,0
  CMP r10,r8
  JMP[LT] @mth_graph_filter_invalid_offsets
  LD r11,r9,8
  CMP r10,r11
  JMP[GT] @mth_graph_filter_invalid_offsets
  SUB r11,r10,r8
  MOVI r12,255
  CMP r11,r12
  JMP[GT] @mth_graph_filter_invalid_offsets
  ADD r8,r10,r0
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_graph_filter_validate_offsets
  LD r11,r9,8
  CMP r8,r11
  JMP[NE] @mth_graph_filter_invalid_offsets

  ; Validate every row's endpoints, strict ordering, uniqueness, and no
  ; self-loop.  Empty rows (isolated vertices) are accepted.
  MOVI r7,0
mth_graph_filter_validate_row:
  LD r8,r9,1
  ADD r8,r8,r7
  LD r10,r8,0
  LD r11,r8,1
  CMP r10,r11
  JMP[EQ] @mth_graph_filter_next_row
  LD r8,r9,2
  ADD r12,r8,r10
  LD r8,r12,0
  SHR r1,r8,8
  CMP r1,r0
  JMP[NE] @mth_graph_filter_invalid_neighbors
  MOVI r1,256
  CMP r8,r1
  JMP[GE] @mth_graph_filter_invalid_neighbors
  CMP r8,r7
  JMP[EQ] @mth_graph_filter_invalid_neighbors
  ADD r1,r8,r0
  ADD r12,r12,r13
  ADD r10,r10,r13
mth_graph_filter_validate_edge:
  CMP r10,r11
  JMP[EQ] @mth_graph_filter_next_row
  LD r8,r12,0
  SHR r2,r8,8
  CMP r2,r0
  JMP[NE] @mth_graph_filter_invalid_neighbors
  MOVI r2,256
  CMP r8,r2
  JMP[GE] @mth_graph_filter_invalid_neighbors
  CMP r8,r7
  JMP[EQ] @mth_graph_filter_invalid_neighbors
  CMP r8,r1
  JMP[LE] @mth_graph_filter_invalid_neighbors
  ADD r1,r8,r0
  ADD r12,r12,r13
  ADD r10,r10,r13
  JMP @mth_graph_filter_validate_edge
mth_graph_filter_next_row:
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_graph_filter_validate_row

  ; All inputs are valid.  Lx and L^2 x are exact I32 vectors.
  LD r1,r9,1
  LD r2,r9,2
  LD r3,r9,3
  LD r4,r9,6
  LD r5,r9,8
  CALL @mth_graph_filter_apply_l
  LD r1,r9,1
  LD r2,r9,2
  LD r3,r9,6
  LD r4,r9,7
  LD r5,r9,8
  CALL @mth_graph_filter_apply_l

  ; Evaluate c0*x + c1*Lx + c2*L^2x in signed 64-bit Q32.32.
  MOVI r8,0
  ST r8,r9,9
mth_graph_filter_poly_loop:
  LD r8,r9,9
  MOVI r7,256
  CMP r8,r7
  JMP[GE] @mth_graph_filter_success
  MOVI r10,0
  MOVI r11,0

  LD r6,r9,4
  LD r1,r6,0
  LD r6,r9,3
  ADD r6,r6,r8
  LD r2,r6,0
  CALL @mth_graph_filter_mul_signed
  ADD r6,r10,r0
  ADD r10,r10,r3
  CMP r10,r6
  JMP[LT] @mth_graph_filter_t0_carry
  ADD r11,r11,r4
  JMP @mth_graph_filter_t1
mth_graph_filter_t0_carry:
  ADD r11,r11,r4
  ADD r11,r11,r13

mth_graph_filter_t1:
  LD r8,r9,9
  LD r6,r9,4
  LD r1,r6,1
  LD r6,r9,6
  ADD r6,r6,r8
  LD r2,r6,0
  CALL @mth_graph_filter_mul_signed
  ADD r6,r10,r0
  ADD r10,r10,r3
  CMP r10,r6
  JMP[LT] @mth_graph_filter_t1_carry
  ADD r11,r11,r4
  JMP @mth_graph_filter_t2
mth_graph_filter_t1_carry:
  ADD r11,r11,r4
  ADD r11,r11,r13

mth_graph_filter_t2:
  LD r8,r9,9
  LD r6,r9,4
  LD r1,r6,2
  LD r6,r9,7
  ADD r6,r6,r8
  LD r2,r6,0
  CALL @mth_graph_filter_mul_signed
  ADD r6,r10,r0
  ADD r10,r10,r3
  CMP r10,r6
  JMP[LT] @mth_graph_filter_t2_carry
  ADD r11,r11,r4
  JMP @mth_graph_filter_store
mth_graph_filter_t2_carry:
  ADD r11,r11,r4
  ADD r11,r11,r13

mth_graph_filter_store:
  ; Arithmetic (two's-complement) shift right by 16: truncation toward -inf.
  LD r8,r9,9
  SHR r6,r10,16
  SHL r7,r11,16
  BOOL.14 r6,r6,r7
  LD r7,r9,5
  ADD r7,r7,r8
  ST r6,r7,0
  ADD r8,r8,r13
  ST r8,r9,9
  JMP @mth_graph_filter_poly_loop

mth_graph_filter_success:
  ADD r15,r14,r0
  MOVI r1,1
  RET

mth_graph_filter_invalid_state:
  ADD r15,r14,r0
  MOVI r1,0
  RET
mth_graph_filter_invalid_offsets:
  ADD r15,r14,r0
  MOVI r1,2
  RET
mth_graph_filter_invalid_neighbors:
  ADD r15,r14,r0
  MOVI r1,3
  RET


; Apply the unweighted Laplacian to one validated 256-word I32 vector.
;   r1=row_offsets, r2=neighbors, r3=input, r4=output, r5=edge_count, r13=1
; Clobbers r1-r8, r10-r12; preserves r9 and r14.  No validation: the caller
; has already proved the CSR and state bounds.
mth_graph_filter_apply_l:
  ADD r6,r1,r0
  ADD r7,r2,r0
  ADD r8,r3,r0
  ADD r12,r3,r0       ; fixed input base for neighbor lookups
  MOVI r10,256
mth_graph_filter_apply_row:
  LD r1,r6,0
  LD r2,r6,1
  SUB r3,r2,r1
  ADD r11,r7,r1
  LD r2,r8,0          ; x_i
  MOVI r5,0
mth_graph_filter_apply_edge:
  CMP r3,r0
  JMP[EQ] @mth_graph_filter_apply_store
  LD r1,r11,0
  ADD r1,r12,r1
  LD r1,r1,0          ; x_j
  SUB r1,r2,r1
  ADD r5,r5,r1
  ADD r11,r11,r13
  SUB r3,r3,r13
  JMP @mth_graph_filter_apply_edge
mth_graph_filter_apply_store:
  ST r5,r4,0
  ADD r6,r6,r13
  ADD r8,r8,r13
  ADD r4,r4,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_filter_apply_row
  RET


; Exact signed 32x32 -> 64 product for a bounded coefficient and vector word.
;   r1=signed coefficient c, r2=signed value v, r13=1
;   returns r3=low32(c*v), r4=high32(c*v)
; Both |c| <= 65536 and |v| < 2^31, so the magnitude fits in 17 + 31 bits.
; The loop uses the absolute value of c and restores the sign at the end.
; Clobbers r1-r8, r12; preserves r9-r11 and r14-r15.
mth_graph_filter_mul_signed:
  MOVI r12,0
  SHR r8,r1,31
  CMP r8,r0
  JMP[EQ] @mth_graph_filter_mul_c_abs
  SUB r1,r0,r1
  MOVI r12,1
mth_graph_filter_mul_c_abs:
  SHR r8,r2,31
  CMP r8,r0
  JMP[EQ] @mth_graph_filter_mul_v_abs
  SUB r2,r0,r2
  MOVI r8,1
  BOOL.6 r12,r12,r8
mth_graph_filter_mul_v_abs:
  MOVI r3,0
  MOVI r4,0
  MOVI r5,0
  MOVI r7,17
mth_graph_filter_mul_loop:
  MOVI r8,1
  BOOL.8 r6,r1,r8
  CMP r6,r0
  JMP[EQ] @mth_graph_filter_mul_shift
  ADD r6,r3,r0
  ADD r3,r3,r2
  CMP r3,r6
  JMP[LT] @mth_graph_filter_mul_carry
  ADD r4,r4,r5
  JMP @mth_graph_filter_mul_shift
mth_graph_filter_mul_carry:
  ADD r4,r4,r5
  ADD r4,r4,r13
mth_graph_filter_mul_shift:
  SHR r8,r2,31
  SHL r2,r2,1
  SHL r5,r5,1
  BOOL.14 r5,r5,r8
  SHR r1,r1,1
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_graph_filter_mul_loop
  CMP r12,r0
  JMP[EQ] @mth_graph_filter_mul_done
  MOVI r8,0
  CMP r3,r8
  JMP[EQ] @mth_graph_filter_mul_neg_high
  SUB r3,r8,r3
  MOVI r6,1
  SUB r4,r8,r4
  SUB r4,r4,r6
  JMP @mth_graph_filter_mul_done
mth_graph_filter_mul_neg_high:
  SUB r4,r8,r4
mth_graph_filter_mul_done:
  RET
