; NaN16 16x16 pair-space K-block accumulator, float32.
; Accumulates one 16-wide K slice into two persistent 16x16 partial sums.
;
; For one output tile:
;   acc_LU[p,q] += sum(k=0..15) L_IK[p,k] * U_KJ[k,q]
;   acc_ULt[p,q] += sum(k=0..15) U_IK[p,k] * L_JK[q,k]
; Caller invokes K blocks in increasing global-k order (0,16,...,240).
;
; ABI:
;   r1=L_IK, r2=U_KJ, r3=U_IK, r4=L_JK
;   r5=acc_LU, r6=acc_ULt, r7=caller-owned 6-word scratch, r13=1
; Each matrix/block is 16x16 row-major float32 (256 NaN16 words).
; Returns r1=1. Inputs are immutable; accumulators are updated in place.
; All spans are valid and pairwise disjoint.
; No host math, no FMA, no reordering. The K order within each call is 0..15.
; Two independent accumulators are retained across calls and finalized later.
;
; Working set: 4 input blocks + 2 accum blocks + scratch = 1,542 NaN16 words.
; This is 12,336 bytes at 8 bytes per NaN16 word, well below 512 KiB.

mth_graph_pair_tile16_k16_accum_f32:
  ADD r11,r7,r0
  ST r1,r11,0
  ST r2,r11,1
  ST r3,r11,2
  ST r4,r11,3
  ST r5,r11,4
  ST r6,r11,5

  MOVI r4,0
mth_pair_k16_row:
  MOVI r5,0
mth_pair_k16_col:
  ; Load the existing partial sums for output cell p*16+q.
  LD r1,r11,4
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r5
  ADD r1,r1,r2
  LD r7,r1,0

  LD r1,r11,5
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r5
  ADD r1,r1,r2
  LD r8,r1,0

  MOVI r6,0
mth_pair_k16_k:
  ; L_IK[p,k] * U_KJ[k,q]
  LD r1,r11,0
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r6
  ADD r1,r1,r2
  LD r9,r1,0

  LD r1,r11,1
  ADD r2,r6,r0
  SHL r2,r2,4
  ADD r2,r2,r5
  ADD r1,r1,r2
  LD r10,r1,0
  FMUL r9,r9,r10
  FADD r7,r7,r9

  ; U_IK[p,k] * L_JK[q,k]
  LD r1,r11,2
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r6
  ADD r1,r1,r2
  LD r9,r1,0

  LD r1,r11,3
  ADD r2,r5,r0
  SHL r2,r2,4
  ADD r2,r2,r6
  ADD r1,r1,r2
  LD r10,r1,0
  FMUL r9,r9,r10
  FADD r8,r8,r9

  ADD r6,r6,r13
  MOVI r12,16
  CMP r6,r12
  JMP[LT] @mth_pair_k16_k

  ; Store both partial sums without combining them.
  LD r1,r11,4
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r5
  ADD r1,r1,r2
  ST r7,r1,0

  LD r1,r11,5
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r2,r2,r5
  ADD r1,r1,r2
  ST r8,r1,0

  ADD r5,r5,r13
  MOVI r12,16
  CMP r5,r12
  JMP[LT] @mth_pair_k16_col

  ADD r4,r4,r13
  MOVI r12,16
  CMP r4,r12
  JMP[LT] @mth_pair_k16_row

  MOVI r1,1
  RET

; Finalize the two 16x16 partial-sum tiles into Ytile = acc_LU + acc_ULt.
; ABI: r1=acc_LU, r2=acc_ULt, r3=Ytile, r13=1. Returns r1=1.
mth_graph_pair_tile16_finalize_f32:
  MOVI r4,0
mth_pair_finalize_loop:
  LD r5,r1,0
  LD r6,r2,0
  FADD r5,r5,r6
  ST r5,r3,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  ADD r3,r3,r13
  ADD r4,r4,r13
  MOVI r7,256
  CMP r4,r7
  JMP[LT] @mth_pair_finalize_loop
  MOVI r1,1
  RET
