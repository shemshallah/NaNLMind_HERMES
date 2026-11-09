; MTH native graph fold/unfold vector transforms — NaN16 only.
; All payloads are IEEE binary32 in 32-bit NaN16 word payloads.
;
; ABI (r13=1):
;   mth_graph_fold_256_vector:
;     r1 = input x[256] float32
;     r2 = plus output p[128] float32
;     r3 = minus output m[128] float32
;     p[i] = x[i] + x[255-i], m[i] = x[i] - x[255-i], i=0..127
;     returns r1=1; clobbers r1-r10; requires non-overlapping spans.
;
;   mth_graph_fold_255_vector:
;     r1 = input x[255] float32
;     r2 = plus output p[128] float32
;     r3 = minus output m[127] float32
;     p[i] = x[i] + x[254-i], m[i] = x[i] - x[254-i], i=0..126
;     p[127] = x[127]
;     returns r1=1; clobbers r1-r10; requires non-overlapping spans.
;
;   mth_graph_fold_256_unfold:
;     r1 = plus p[128] float32
;     r2 = minus m[128] float32
;     r3 = output x[256] float32
;     x[i]     = (p[i] + m[i]) * 0.5, i=0..127
;     x[255-i] = (p[i] - m[i]) * 0.5, i=0..127
;     returns r1=1; clobbers r1-r12; requires non-overlapping spans.
;
; This is the UNNORMALIZED q+/- basis. The normalized basis uses
; q_i+/- = (e_i +/- e_{255-i}) / sqrt(2); its inverse needs the sqrt(1/2)
; multiplier, not 0.5. The 0.5 here is the exact binary32 constant
; 0x3F000000, so the unnormalized round trip is bit-exact whenever the
; forward sum and difference are exactly representable in binary32.
;
; No validation pass, allocation, CALL, or static slot is used. The caller
; owns every span. The transforms are readout algebra only; see
; docs/MTH_GRAPH_FOLD_ABI.md for the authorization caveat.

mth_graph_fold_256_vector:
  ADD r4,r1,r0
  MOVI r5,255
  ADD r5,r4,r5
  MOVI r7,128

mth_graph_fold_256_vector_loop:
  LD r8,r4,0
  LD r9,r5,0
  FADD r10,r8,r9
  ST r10,r2,0
  FSUB r10,r8,r9
  ST r10,r3,0
  ADD r4,r4,r13
  SUB r5,r5,r13
  ADD r2,r2,r13
  ADD r3,r3,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[NE] @mth_graph_fold_256_vector_loop

  MOVI r1,1
  RET

mth_graph_fold_255_vector:
  ADD r4,r1,r0
  MOVI r5,254
  ADD r5,r4,r5
  MOVI r7,127

mth_graph_fold_255_vector_loop:
  LD r8,r4,0
  LD r9,r5,0
  FADD r10,r8,r9
  ST r10,r2,0
  FSUB r10,r8,r9
  ST r10,r3,0
  ADD r4,r4,r13
  SUB r5,r5,r13
  ADD r2,r2,r13
  ADD r3,r3,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[NE] @mth_graph_fold_255_vector_loop

  ; The loop leaves r4 at input[127] and r2 at plus[127].
  LD r8,r4,0
  ST r8,r2,0

  MOVI r1,1
  RET

mth_graph_fold_256_unfold:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,255
  ADD r7,r6,r7
  MOVI r8,128
  MOVI r12,0
  MOVHI r12,0x3F00

mth_graph_fold_256_unfold_loop:
  LD r9,r4,0
  LD r10,r5,0
  FADD r11,r9,r10
  FMUL r11,r11,r12
  ST r11,r6,0
  FSUB r11,r9,r10
  FMUL r11,r11,r12
  ST r11,r7,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  ADD r6,r6,r13
  SUB r7,r7,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[NE] @mth_graph_fold_256_unfold_loop

  MOVI r1,1
  RET
