; MTH 256-element float32 dot product — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = y, 256 float32 payloads
;   r3 = output address, one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r8
;
; The caller supplies non-overlapping spans. Accumulation order is increasing i.
; No host math, no reordering, no fused multiply-add.

mth_dot256_f32:
  MOVI r4,0
  MOVI r5,0
  MOVI r6,0
mth_dot256_loop:
  ADD r7,r1,r6
  LD r7,r7,0
  ADD r8,r2,r6
  LD r8,r8,0
  FMUL r7,r7,r8
  FADD r4,r4,r7
  ADD r6,r6,r13
  MOVI r5,256
  CMP r6,r5
  JMP[LT] @mth_dot256_loop
  ST r4,r3,0
  MOVI r1,1
  RET
