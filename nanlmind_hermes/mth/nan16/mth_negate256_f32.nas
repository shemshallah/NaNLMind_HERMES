; MTH 256-element float32 negation — NaN16 only.
;
; ABI:
;   r1 = input, 256 float32 payloads
;   r2 = output, 256 float32 payloads
;   r13 = 1
;   returns r1=1 success
;   clobbers r3-r7
;
; out[i] = -in[i]. Negation is an exact IEEE sign-bit flip (XOR 0x80000000),
; not an arithmetic subtraction. It is applied to the payload word directly,
; so signed zeros invert (including -0.0 -> +0.0) and NaN sign bits flip
; without touching the payload. No host math, no reordering, leaf routine.

mth_negate256_f32:
  MOVI r3,0
  MOVHI r3,0x8000            ; r3 = 0x80000000
  MOVI r4,0
mth_negate256_loop:
  ADD r5,r1,r4
  LD r6,r5,0
  BOOL.6 r6,r6,r3            ; XOR: flip only the sign bit
  ADD r5,r2,r4
  ST r6,r5,0
  ADD r4,r4,r13
  MOVI r7,256
  CMP r4,r7
  JMP[LT] @mth_negate256_loop
  MOVI r1,1
  RET
