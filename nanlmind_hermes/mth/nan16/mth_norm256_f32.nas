; MTH 256-element float32 Euclidean norm — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = output address, one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r3-r8
;
; norm = sqrt(sum_i x[i]*x[i]). Accumulation order is increasing i.
; No host math, no reordering, no fused multiply-add.

mth_norm256_f32:
  MOVI r3,0
  MOVI r4,0
  MOVI r5,0
mth_norm256_loop:
  ADD r6,r1,r5
  LD r6,r6,0
  FMUL r6,r6,r6
  FADD r3,r3,r6
  ADD r5,r5,r13
  MOVI r4,256
  CMP r5,r4
  JMP[LT] @mth_norm256_loop
  FSQRT r3,r3
  ST r3,r2,0
  MOVI r1,1
  RET
