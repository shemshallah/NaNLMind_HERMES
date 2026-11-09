; MTH dense 256x256 pair-space action, one 16x16 output tile — NaN16 float32.
; Y[i,j] = sum_k L[i,k]*U[k,j] + sum_k U[i,k]*L[j,k].
;
; ABI: r1=U full matrix, r2=L_I 16x256 rows, r3=L_J 16x256 rows,
;      r4=Y tile, r5=i_base, r6=j_base, r7=caller-owned 6-word scratch,
;      r13=1. Bases are multiples of 16 in [0,240].
; Returns r1=1. Caller invokes all 256 tile pairs to cover Y.
; L_I and L_J are immutable row windows from the same L version.
; All spans are valid and pairwise disjoint.
;
; For each output cell, two independent +0 accumulators execute k=0..255
; in increasing order. Each term is one rounded FMUL then one FADD.
; The final FADD combines the completed sums. No FMA or reordering.
;
; Working set: U=256 KiB, L_I=16 KiB, L_J=16 KiB, Ytile=1 KiB,
; scratch=24 bytes. Below the 512 KiB guest ceiling.
; No 65,536x65,536 Kronecker operator is materialized.

mth_graph_pair_tile16_f32:
  ADD r11,r7,r0
  ST r1,r11,0
  ST r2,r11,1
  ST r3,r11,2
  ST r4,r11,3
  ST r5,r11,4
  ST r6,r11,5
  MOVI r4,0
mth_pair_tile_row:
  MOVI r5,0
mth_pair_tile_col:
  MOVI r6,0
  MOVI r7,0
  MOVI r8,0
mth_pair_tile_k:
  LD r1,r11,1
  ADD r2,r4,r0
  SHL r2,r2,8
  ADD r1,r1,r2
  ADD r1,r1,r6
  LD r9,r1,0
  LD r1,r11,0
  ADD r2,r6,r0
  SHL r2,r2,8
  ADD r1,r1,r2
  LD r2,r11,5
  ADD r2,r2,r5
  ADD r1,r1,r2
  LD r10,r1,0
  FMUL r9,r9,r10
  FADD r7,r7,r9
  LD r1,r11,0
  LD r2,r11,4
  ADD r2,r2,r4
  SHL r2,r2,8
  ADD r1,r1,r2
  ADD r1,r1,r6
  LD r9,r1,0
  LD r1,r11,2
  ADD r2,r5,r0
  SHL r2,r2,8
  ADD r1,r1,r2
  ADD r1,r1,r6
  LD r10,r1,0
  FMUL r9,r9,r10
  FADD r8,r8,r9
  ADD r6,r6,r13
  MOVI r12,256
  CMP r6,r12
  JMP[LT] @mth_pair_tile_k
  FADD r7,r7,r8
  LD r1,r11,3
  ADD r2,r4,r0
  SHL r2,r2,4
  ADD r1,r1,r2
  ADD r1,r1,r5
  ST r7,r1,0
  ADD r5,r5,r13
  MOVI r12,16
  CMP r5,r12
  JMP[LT] @mth_pair_tile_col
  ADD r4,r4,r13
  MOVI r12,16
  CMP r4,r12
  JMP[LT] @mth_pair_tile_row
  MOVI r1,1
  RET
