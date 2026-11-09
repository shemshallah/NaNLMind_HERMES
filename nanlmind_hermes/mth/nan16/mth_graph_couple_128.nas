; MTH two-region coupling primitive for tunneling experiments — NaN16 float32.
;
; Symmetric convex coupling of 128 inner and 128 outer positions:
;   inner_next[i] = (1-eta)*inner[i] + eta*outer[i]
;   outer_next[i] = eta*inner[i] + (1-eta)*outer[i]
;
; ABI:
;   r1 = inner, 128 float32
;   r2 = outer, 128 float32
;   r3 = eta, one float32 in [0,1]
;   r4 = inner_next, 128 float32
;   r5 = outer_next, 128 float32
;   r13 = 1
;   returns r1=1 success, r1=0 invalid eta or layout
;   clobbers r6-r12; no CALLs or static slots.
;
; eta is a computational mixing coefficient, not a physical tunneling
; probability. This is the R7 block-coupling primitive; the actual transmission
; metric is a separate experiment. Preflight checks eta in [0,1] and finite,
; and rejects overlapping spans before writing output.

mth_graph_couple_128_f32:
  ; eta >= 0.
  MOVI r6,0
  FCMP r3,r6
  JMP[LT] @mth_graph_couple_invalid
  ; eta <= 1.
  MOVI r6,0
  MOVHI r6,0x3F80
  FCMP r3,r6
  JMP[GT] @mth_graph_couple_invalid
  ; eta finite: exponent bits 0xFF are rejected.
  MOVI r7,0
  MOVHI r7,0x7F80
  FCMP r3,r7
  JMP[GE] @mth_graph_couple_invalid

  ; Reject output spans that overlap inputs or each other.
  MOVI r8,127
  ADD r9,r1,r8
  ADD r10,r4,r8
  CMP r1,r10
  JMP[GT] @mth_couple_ov_1
  CMP r4,r9
  JMP[LE] @mth_graph_couple_invalid
mth_couple_ov_1:
  ADD r9,r2,r8
  CMP r2,r10
  JMP[GT] @mth_couple_ov_2
  CMP r4,r9
  JMP[LE] @mth_graph_couple_invalid
mth_couple_ov_2:
  ADD r10,r5,r8
  CMP r1,r10
  JMP[GT] @mth_couple_ov_3
  CMP r5,r9
  JMP[LE] @mth_graph_couple_invalid
mth_couple_ov_3:
  ADD r9,r1,r8
  CMP r2,r10
  JMP[GT] @mth_couple_ov_4
  CMP r5,r9
  JMP[LE] @mth_graph_couple_invalid
mth_couple_ov_4:
  ADD r9,r4,r8
  CMP r5,r9
  JMP[GT] @mth_couple_ok
  ADD r9,r5,r8
  CMP r4,r9
  JMP[LE] @mth_graph_couple_invalid

mth_couple_ok:
  MOVI r8,0
  MOVHI r8,0x3F80
  FSUB r8,r8,r3              ; 1-eta
  MOVI r6,0                  ; i
mth_couple_loop:
  ADD r9,r1,r6
  LD r9,r9,0                 ; inner
  ADD r10,r2,r6
  LD r10,r10,0               ; outer
  FMUL r11,r8,r9
  FMUL r12,r3,r10
  FADD r11,r11,r12           ; inner_next
  ADD r12,r4,r6
  ST r11,r12,0
  FMUL r11,r3,r9
  FMUL r12,r8,r10
  FADD r11,r11,r12           ; outer_next
  ADD r12,r5,r6
  ST r11,r12,0
  ADD r6,r6,r13
  MOVI r7,128
  CMP r6,r7
  JMP[LT] @mth_couple_loop
  MOVI r1,1
  RET

mth_graph_couple_invalid:
  MOVI r1,0
  RET
