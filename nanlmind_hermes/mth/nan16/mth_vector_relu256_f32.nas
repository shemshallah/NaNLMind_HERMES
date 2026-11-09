; MTH 256-element float32 ReLU-by-sign — NaN16 only.
;
; ABI:
;   r1 = input, 256 float32 payloads
;   r2 = output, 256 float32 payloads
;   r13 = 1
;   returns r1=1
;   clobbers r3-r7
;
; out[i] = in[i] when the sign bit is clear, else +0.0.
; This is an exact sign test, not a host float comparison. Positive NaN is
; preserved; negative NaN becomes +0.0 under the declared sign-only policy.

mth_vector_relu256_f32:
  MOVI r4,0
mth_vector_relu256_loop:
  ADD r5,r1,r4
  LD r6,r5,0
  SHR r7,r6,31
  CMP r7,r0
  JMP[EQ] @mth_vector_relu256_store
  MOVI r6,0
mth_vector_relu256_store:
  ADD r5,r2,r4
  ST r6,r5,0
  ADD r4,r4,r13
  MOVI r7,256
  CMP r4,r7
  JMP[LT] @mth_vector_relu256_loop
  MOVI r1,1
  RET
