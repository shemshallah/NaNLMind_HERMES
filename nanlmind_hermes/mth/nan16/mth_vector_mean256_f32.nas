; MTH 256-element float32 arithmetic mean — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32 payloads
;   r2 = output address, one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r3-r6
;
; mean = (sum_i x[i]) / 256. The accumulator starts at +0.0 and elements are
; added in increasing i with exactly one FADD per element. The final division
; is one FDIV by the float32 constant 256.0 (0x43800000), not a multiplication
; by 1/256. Both FADD and FDIV use the NaN16 float32 rules: IEEE round-to-
; nearest-even for non-NaN operands, quieted first-NaN propagation, and
; 0xFFC00000 for an invalid NaN result. No host math, no reordering, no
; compensation, and no fused multiply-add.

mth_vector_mean256_f32:
  MOVI r3,0
  MOVI r4,0
mth_vector_mean256_loop:
  ADD r5,r1,r4
  LD r5,r5,0
  FADD r3,r3,r5
  ADD r4,r4,r13
  MOVI r6,256
  CMP r4,r6
  JMP[LT] @mth_vector_mean256_loop
  MOVI r6,0
  MOVHI r6,0x4380            ; 256.0f
  FDIV r3,r3,r6
  ST r3,r2,0
  MOVI r1,1
  RET
