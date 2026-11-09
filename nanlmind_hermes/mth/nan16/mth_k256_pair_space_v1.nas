; MTH K256 matrix-free pair-space operator — native NaN16.
; For L = 256I - J, L_pair(U) = LU + UL^T = 512U - colsum(U) - rowsum(U).
; No 65,536 x 65,536 Kronecker matrix is formed.
;
; ABI:
;   r1 = input U, 65,536 signed I32 Q16.16 words (row-major)
;   r2 = output Y, 65,536 signed I32 Q16.16 words
;   r3 = row_sums workspace, 256 signed I32 words
;   r4 = col_sums workspace, 256 signed I32 words
;   r5 = caller-owned 16-word scratch
;   r13 = 1
; Returns r1=1 success, r1=0 if any input state is outside [-2^20,+2^20].
; Input/output/workspace spans must be disjoint.
;
; Bound: |Uij| <= 2^20. Then |row/col sum| <= 2^28,
; |512*Uij| <= 2^29, and |Yij| <= 2^30. All I32 arithmetic is exact.
; This is the exact K256 special case; it does not claim a generic CSR pair kernel.

mth_k256_pair_space_q16_16:
  ADD r11,r5,r0
  ST r1,r11,0
  ST r2,r11,1
  ST r3,r11,2
  ST r4,r11,3

  ; Full matrix preflight before writing output or workspaces.
  ADD r10,r1,r0
  MOVI r8,0
  MOVI r9,0
mth_pair_validate:
  LD r12,r10,0
  SHR r1,r12,20
  CMP r1,r0
  JMP[EQ] @mth_pair_state_ok
  MOVI r2,1
  CMP r1,r2
  JMP[EQ] @mth_pair_positive_limit
  MOVI r2,4095
  CMP r1,r2
  JMP[EQ] @mth_pair_state_ok
  JMP @mth_pair_bad_state
mth_pair_positive_limit:
  MOVI r2,0
  MOVHI r2,0x0010
  CMP r12,r2
  JMP[NE] @mth_pair_bad_state
mth_pair_state_ok:
  ADD r10,r10,r13
  ADD r9,r9,r13
  MOVI r1,256
  CMP r9,r1
  JMP[LT] @mth_pair_validate
  MOVI r9,0
  ADD r8,r8,r13
  CMP r8,r1
  JMP[LT] @mth_pair_validate

  ; Zero column sums.
  LD r10,r11,3
  MOVI r8,256
mth_pair_zero_cols:
  ST r0,r10,0
  ADD r10,r10,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_pair_zero_cols

  ; One matrix pass computes all row and column sums.
  LD r10,r11,0
  MOVI r8,0
mth_pair_sum_row:
  MOVI r6,0
  MOVI r9,0
mth_pair_sum_col:
  LD r12,r10,0
  ADD r6,r6,r12
  LD r1,r11,3
  ADD r1,r1,r9
  LD r2,r1,0
  ADD r2,r2,r12
  ST r2,r1,0
  ADD r10,r10,r13
  ADD r9,r9,r13
  MOVI r1,256
  CMP r9,r1
  JMP[LT] @mth_pair_sum_col
  LD r1,r11,2
  ADD r1,r1,r8
  ST r6,r1,0
  ADD r8,r8,r13
  ; Compare the completed row count against 256.
  MOVI r2,256
  CMP r8,r2
  JMP[LT] @mth_pair_sum_row

  ; Apply Yij = 512*Uij - row_sum[i] - col_sum[j].
  LD r10,r11,0
  LD r12,r11,1
  MOVI r8,0
mth_pair_apply_row:
  LD r1,r11,2
  ADD r1,r1,r8
  LD r6,r1,0
  MOVI r9,0
mth_pair_apply_col:
  LD r1,r10,0
  SHL r1,r1,9
  SUB r1,r1,r6
  LD r2,r11,3
  ADD r2,r2,r9
  LD r2,r2,0
  SUB r1,r1,r2
  ST r1,r12,0
  ADD r10,r10,r13
  ADD r12,r12,r13
  ADD r9,r9,r13
  MOVI r2,256
  CMP r9,r2
  JMP[LT] @mth_pair_apply_col
  ADD r8,r8,r13
  CMP r8,r2
  JMP[LT] @mth_pair_apply_row

  MOVI r1,1
  RET

mth_pair_bad_state:
  MOVI r1,0
  RET
