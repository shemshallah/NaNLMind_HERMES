; MTH 256-element float32 vector subtraction — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = y, 256 float32 payloads
;   r3 = output, 256 float32 payloads
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r8
;
; out[i] = x[i] - y[i]. Element order is increasing i with exactly one FSUB
; per element. No host math, no reordering, no fused multiply-add.
; The caller supplies readable input spans and a writable output span.
; Exact output aliasing with x or y is safe; partial overlap is not.

mth_vector_sub256_f32:
  MOVI r4,0
mth_vector_sub256_loop:
  ADD r5,r1,r4
  LD r5,r5,0
  ADD r6,r2,r4
  LD r6,r6,0
  FSUB r7,r5,r6
  ADD r8,r3,r4
  ST r7,r8,0
  ADD r4,r4,r13
  MOVI r6,256
  CMP r4,r6
  JMP[LT] @mth_vector_sub256_loop
  MOVI r1,1
  RET
