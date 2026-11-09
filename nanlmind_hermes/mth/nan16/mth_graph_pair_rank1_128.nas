; MTH rank-1 pair-space action on a 128-head ring — NaN16 float32 only.
;
; For a C128 ring Laplacian L and rank-1 factors u,v:
;   U = L (u v^T) + (u v^T) L^T
;     = (Lu) v^T + u (Lv)^T
;   U[i,j] = Lu[i]*v[j] + u[i]*Lv[j]
;
; ABI:
;   r1 = u, 128 float32
;   r2 = v, 128 float32
;   r3 = output U, 128x128 float32, row-major (16,384 words)
;   r4 = caller-owned scratch, 256 float32 (Lu[0..127], Lv[0..127])
;   r13 = 1
;   returns r1=1 success
;   clobbers r1-r12; no CALLs or static slots.
;
; This is the bounded 128-head prototype. Full 256x256 pair-space state is
; paged; it is not materialized in one guest.

mth_graph_pair_rank1_128:
  ; Boundary 0: 2*x0 - x1 - x127 for both u and v.
  LD r5,r1,0
  FADD r5,r5,r5
  ADD r6,r1,r13
  LD r6,r6,0
  FSUB r5,r5,r6
  MOVI r7,127
  ADD r6,r1,r7
  LD r6,r6,0
  FSUB r5,r5,r6
  ST r5,r4,0

  LD r5,r2,0
  FADD r5,r5,r5
  ADD r6,r2,r13
  LD r6,r6,0
  FSUB r5,r5,r6
  MOVI r7,127
  ADD r6,r2,r7
  LD r6,r6,0
  FSUB r5,r5,r6
  MOVI r7,128
  ADD r6,r4,r7
  ST r5,r6,0

  ; Interior 1..126.
  MOVI r5,1
mth_pair128_interior:
  MOVI r7,126
  CMP r5,r7
  JMP[GT] @mth_pair128_boundary_last

  ; Lu[k].
  ADD r6,r1,r5
  LD r7,r6,0
  FADD r7,r7,r7
  SUB r6,r5,r13
  ADD r6,r1,r6
  LD r6,r6,0
  FSUB r7,r7,r6
  ADD r6,r5,r13
  ADD r6,r1,r6
  LD r6,r6,0
  FSUB r7,r7,r6
  ADD r6,r4,r5
  ST r7,r6,0

  ; Lv[k].
  ADD r6,r2,r5
  LD r7,r6,0
  FADD r7,r7,r7
  SUB r6,r5,r13
  ADD r6,r2,r6
  LD r6,r6,0
  FSUB r7,r7,r6
  ADD r6,r5,r13
  ADD r6,r2,r6
  LD r6,r6,0
  FSUB r7,r7,r6
  MOVI r8,128
  ADD r6,r4,r8
  ADD r6,r6,r5
  ST r7,r6,0

  ADD r5,r5,r13
  JMP @mth_pair128_interior

mth_pair128_boundary_last:
  ; Boundary 127: 2*x127 - x126 - x0.
  MOVI r5,127
  ADD r6,r1,r5
  LD r7,r6,0
  FADD r7,r7,r7
  MOVI r8,126
  ADD r6,r1,r8
  LD r6,r6,0
  FSUB r7,r7,r6
  LD r6,r1,0
  FSUB r7,r7,r6
  ADD r6,r4,r5
  ST r7,r6,0

  ADD r6,r2,r5
  LD r7,r6,0
  FADD r7,r7,r7
  MOVI r8,126
  ADD r6,r2,r8
  LD r6,r6,0
  FSUB r7,r7,r6
  LD r6,r2,0
  FSUB r7,r7,r6
  MOVI r8,128
  ADD r6,r4,r8
  ADD r6,r6,r5
  ST r7,r6,0

  ; Outer-product accumulation: U[i,j] = Lu[i]*v[j] + u[i]*Lv[j].
  MOVI r5,0
  ADD r6,r3,r0
mth_pair128_row:
  ADD r12,r4,r5
  LD r8,r12,0
  ADD r12,r1,r5
  LD r9,r12,0
  MOVI r7,0
mth_pair128_col:
  ADD r12,r2,r7
  LD r10,r12,0
  MOVI r11,128
  ADD r12,r4,r11
  ADD r12,r12,r7
  LD r11,r12,0
  FMUL r12,r8,r10
  FMUL r3,r9,r11
  FADD r12,r12,r3
  ST r12,r6,0
  ADD r6,r6,r13
  ADD r7,r7,r13
  MOVI r3,128
  CMP r7,r3
  JMP[LT] @mth_pair128_col
  ADD r5,r5,r13
  MOVI r3,128
  CMP r5,r3
  JMP[LT] @mth_pair128_row

  MOVI r1,1
  RET
