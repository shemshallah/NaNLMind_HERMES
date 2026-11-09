; MTH 256-element float32 vector scale — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = scalar, one float32 payload
;   r3 = output, 256 float32 payloads
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r7
;
; out[i] = x[i] * scalar. The scalar payload is loaded once before the loop.
; Exactly one FMUL is executed per element in increasing i. No host math, no
; reordering, no fused multiply-add.
;
; NaN16 float32 rules for FMUL with A=x[i] and B=scalar:
;   1. If A is a NaN, the result is A | 0x00400000 (A wins).
;   2. Else if B is a NaN, the result is B | 0x00400000 (B is quieted).
;   3. Else compute IEEE binary32 multiplication with round-to-nearest-even,
;      with no flush-to-zero and no denormals-are-zero.
;   4. If that result is NaN (0 * inf), use the canonical invalid NaN
;      0xFFC00000.
; Signed zero follows IEEE sign XOR. For example +0 * -2 = -0 and
; -0 * -2 = +0. Any finite product rounds to nearest, ties to even; the
; fixture's finite cases are exact except for the declared rounding case.
;
; The caller supplies readable input spans and a writable output span.
; Exact output aliasing with x is safe because each element is loaded before
; its result is stored; partial overlap is not.

mth_vector_scale256_f32:
  LD r4,r2,0
  MOVI r5,0
mth_vector_scale256_loop:
  ADD r6,r1,r5
  LD r6,r6,0
  FMUL r6,r6,r4
  ADD r7,r3,r5
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r6,256
  CMP r5,r6
  JMP[LT] @mth_vector_scale256_loop
  MOVI r1,1
  RET
