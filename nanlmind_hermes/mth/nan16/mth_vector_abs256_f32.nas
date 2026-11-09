; MTH 256-element float32 absolute value — NaN16 only.
;
; ABI:
;   r1 = input, 256 float32 payloads
;   r2 = output, 256 float32 payloads
;   r13 = 1
;   returns r1=1
;   clobbers r3-r7
;
; out[i] = abs(in[i]) by exact IEEE sign-bit clear (XOR 0x80000000).
; Positive values pass unchanged; -0.0 becomes +0.0; NaN payload/sign handling
; follows the same sign-bit rule (positive NaN unchanged, negative NaN becomes
; positive NaN). No arithmetic or host math is performed.

mth_vector_abs256_f32:
  MOVI r3,0
  MOVHI r3,0x8000            ; r3 = 0x80000000
  MOVI r4,0
mth_vector_abs256_loop:
  ADD r5,r1,r4
  LD r6,r5,0
  BOOL.6 r6,r6,r3            ; XOR: clear only the sign bit
  ADD r5,r2,r4
  ST r6,r5,0
  ADD r4,r4,r13
  MOVI r7,256
  CMP r4,r7
  JMP[LT] @mth_vector_abs256_loop
  MOVI r1,1
  RET
