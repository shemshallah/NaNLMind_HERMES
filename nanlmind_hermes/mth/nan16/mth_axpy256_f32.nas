; MTH 256-element float32 axpy — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = y, 256 float32 payloads, updated in place
;   r3 = alpha, one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r8
;
; y[i] = y[i] + alpha * x[i]. Accumulation order is increasing i.
; No host math, no reordering, no fused multiply-add.

mth_axpy256_f32:
  LD r4,r3,0
  MOVI r5,0
  MOVI r6,0
mth_axpy256_loop:
  ADD r7,r1,r6
  LD r7,r7,0
  FMUL r7,r7,r4
  ADD r8,r2,r6
  LD r5,r8,0
  FADD r5,r5,r7
  ST r5,r8,0
  ADD r6,r6,r13
  MOVI r5,256
  CMP r6,r5
  JMP[LT] @mth_axpy256_loop
  MOVI r1,1
  RET
