; MTH weighted symmetric CSR Laplacian v2 — NaN16 only.
; L_w x[i] = sum_j w[i,j] * (x[i] - x[j]).
;
; ABI: r1 offsets, r2 neighbors, r3 weights, r4 x, r5 output,
;      r6 edge_count, r7 caller-owned 16-word scratch frame, r13=1.
; Output is 256 signed I64 Q16.16 values, low U32 then high U32.
; Weights are U8 integer multipliers stored in U32 words.
; Returns r1=1 success, 0 count/state, 2 offsets, 3 endpoints,
;         4 weights. Clobbers r1-r10,r12; r11 retains scratch base.
; Requires non-overlapping caller-owned spans and validated symmetric CSR.
;
; Per-edge |w*(xi-xj)| <= 255*2^23 < 2^31.
; Per-row |Lx| <= 255^2*2^23 < 2^40, so I64 accumulation is exact.
; The helper's 32x32->64 multiply is used; valid products have high=0.

.include "mth_multiply.nas"

mth_sparse_weighted_graph_laplacian_q16_16_i64_v2:
  ADD r11,r7,r0
  ST r1,r11,0
  ST r2,r11,1
  ST r3,r11,2
  ST r4,r11,3
  ST r5,r11,4
  ST r6,r11,5

  CMP r6,r0
  JMP[LT] @mth_weighted_v2_invalid_state
  MOVI r8,65280
  CMP r6,r8
  JMP[GT] @mth_weighted_v2_invalid_state

  ; Preflight all 256 signed Q16.16 states in [-2^22,+2^22].
  ADD r8,r4,r0
  MOVI r9,256
mth_weighted_v2_validate_x:
  LD r10,r8,0
  SHR r12,r10,22
  CMP r12,r0
  JMP[EQ] @mth_weighted_v2_x_ok
  MOVI r1,1
  CMP r12,r1
  JMP[NE] @mth_weighted_v2_check_negative
  MOVI r1,0
  MOVHI r1,0x0040
  CMP r10,r1
  JMP[EQ] @mth_weighted_v2_x_ok
  JMP @mth_weighted_v2_invalid_state
mth_weighted_v2_check_negative:
  MOVI r1,1023
  CMP r12,r1
  JMP[EQ] @mth_weighted_v2_x_ok
  JMP @mth_weighted_v2_invalid_state
mth_weighted_v2_x_ok:
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_weighted_v2_validate_x

  ; Validate 257 monotone offsets, degree <=255, final == edge_count.
  LD r8,r11,0
  LD r12,r8,0
  CMP r12,r0
  JMP[NE] @mth_weighted_v2_invalid_offsets
  MOVI r9,257
  MOVI r10,0
mth_weighted_v2_validate_offsets:
  LD r12,r8,0
  CMP r12,r10
  JMP[LT] @mth_weighted_v2_invalid_offsets
  LD r1,r11,5
  CMP r12,r1
  JMP[GT] @mth_weighted_v2_invalid_offsets
  SUB r1,r12,r10
  MOVI r2,255
  CMP r1,r2
  JMP[GT] @mth_weighted_v2_invalid_offsets
  ADD r10,r12,r0
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_weighted_v2_validate_offsets
  LD r1,r11,5
  CMP r10,r1
  JMP[NE] @mth_weighted_v2_invalid_offsets

  ; Validate endpoint words are canonical U8 indices.
  LD r8,r11,1
  LD r9,r11,5
mth_weighted_v2_validate_neighbors:
  CMP r9,r0
  JMP[EQ] @mth_weighted_v2_validate_weights_setup
  LD r10,r8,0
  SHR r12,r10,8
  CMP r12,r0
  JMP[NE] @mth_weighted_v2_invalid_neighbors
  MOVI r1,256
  CMP r10,r1
  JMP[LT] @mth_weighted_v2_neighbor_valid
  JMP @mth_weighted_v2_invalid_neighbors
mth_weighted_v2_neighbor_valid:
  ADD r8,r8,r13
  SUB r9,r9,r13
  JMP @mth_weighted_v2_validate_neighbors

  ; Validate U8 weights are nonzero and have no upper bits set.
mth_weighted_v2_validate_weights_setup:
  LD r8,r11,2
  LD r9,r11,5
mth_weighted_v2_validate_weights:
  CMP r9,r0
  JMP[EQ] @mth_weighted_v2_apply_setup
  LD r10,r8,0
  SHR r12,r10,8
  CMP r12,r0
  JMP[NE] @mth_weighted_v2_invalid_weights
  CMP r10,r0
  JMP[EQ] @mth_weighted_v2_invalid_weights
  ADD r8,r8,r13
  SUB r9,r9,r13
  JMP @mth_weighted_v2_validate_weights

  ; All inputs are valid. Each output row accumulates in signed I64.
mth_weighted_v2_apply_setup:
  MOVI r8,0
  ST r8,r11,6
mth_weighted_v2_apply_row:
  LD r8,r11,6
  LD r1,r11,0
  ADD r1,r1,r8
  LD r9,r1,0
  ADD r1,r1,r13
  LD r12,r1,0
  ST r9,r11,7
  ST r12,r11,8
  LD r1,r11,3
  ADD r1,r1,r8
  LD r10,r1,0
  ST r10,r11,9
  MOVI r1,0
  ST r1,r11,10
  ST r1,r11,11

mth_weighted_v2_apply_edge:
  LD r8,r11,7
  LD r9,r11,8
  CMP r8,r9
  JMP[EQ] @mth_weighted_v2_store_row

  LD r1,r11,1
  ADD r1,r1,r8
  LD r10,r1,0
  LD r2,r11,2
  ADD r2,r2,r8
  LD r2,r2,0
  ST r2,r11,13
  LD r1,r11,3
  ADD r1,r1,r10
  LD r3,r1,0
  LD r4,r11,9
  SUB r1,r4,r3
  SHR r2,r1,31
  ST r2,r11,14
  CMP r2,r0
  JMP[EQ] @mth_weighted_v2_abs_ready
  SUB r1,r0,r1
mth_weighted_v2_abs_ready:
  LD r2,r11,13
  ; Inline unsigned 32x32->64 shift/add multiply. NaN16 CALL is
  ; single-link and this routine is already a callee, so no nested CALL.
  MOVI r3,0
  MOVI r4,0
  MOVI r5,0
  ADD r6,r1,r0
  ADD r7,r2,r0
  MOVI r8,32
mth_weighted_v2_mul_loop:
  MOVI r10,1
  BOOL.8 r9,r7,r10
  CMP r9,r0
  JMP[EQ] @mth_weighted_v2_mul_shift
  ADD r10,r4,r0
  ADD r4,r4,r6
  CMP r4,r10
  JMP[LT] @mth_weighted_v2_mul_carry
  ADD r3,r3,r5
  JMP @mth_weighted_v2_mul_shift
mth_weighted_v2_mul_carry:
  ADD r3,r3,r5
  ADD r3,r3,r13
mth_weighted_v2_mul_shift:
  SHR r9,r6,31
  SHL r6,r6,1
  SHL r5,r5,1
  BOOL.14 r5,r5,r9
  SHR r7,r7,1
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_weighted_v2_mul_loop
  ADD r1,r4,r0
  ADD r2,r3,r0
  ; Product is exact signed-magnitude <= 2,139,095,040; high word is zero.
  LD r3,r11,10
  LD r4,r11,11
  LD r5,r11,14
  CMP r5,r0
  JMP[NE] @mth_weighted_v2_acc_subtract

  ; Positive product: add low word and propagate carry into high word.
  ADD r6,r3,r0
  ADD r3,r3,r1
  CMP r3,r6
  JMP[LT] @mth_weighted_v2_acc_add_carry
  JMP @mth_weighted_v2_acc_store
mth_weighted_v2_acc_add_carry:
  ADD r4,r4,r13
  JMP @mth_weighted_v2_acc_store

mth_weighted_v2_acc_subtract:
  ; Negative product: subtract low word and propagate borrow.
  ADD r6,r3,r0
  SUB r3,r3,r1
  CMP r6,r1
  JMP[LT] @mth_weighted_v2_acc_sub_borrow
  JMP @mth_weighted_v2_acc_store
mth_weighted_v2_acc_sub_borrow:
  SUB r4,r4,r13

mth_weighted_v2_acc_store:
  ST r3,r11,10
  ST r4,r11,11
  LD r8,r11,7
  ADD r8,r8,r13
  ST r8,r11,7
  JMP @mth_weighted_v2_apply_edge

mth_weighted_v2_store_row:
  LD r1,r11,4
  LD r8,r11,6
  SHL r8,r8,1
  LD r3,r11,10
  LD r4,r11,11
  ADD r1,r1,r8
  ST r3,r1,0
  ADD r1,r1,r13
  ST r4,r1,0
  LD r8,r11,6
  ADD r8,r8,r13
  ST r8,r11,6
  MOVI r9,256
  CMP r8,r9
  JMP[LT] @mth_weighted_v2_apply_row
  MOVI r1,1
  RET

mth_weighted_v2_invalid_state:
  MOVI r1,0
  RET
mth_weighted_v2_invalid_offsets:
  MOVI r1,2
  RET
mth_weighted_v2_invalid_neighbors:
  MOVI r1,3
  RET
mth_weighted_v2_invalid_weights:
  MOVI r1,4
  RET
