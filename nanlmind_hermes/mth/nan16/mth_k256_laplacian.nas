; MTH exact K256 graph Laplacian — NaN16 only.
; Operator: L = 256 I - J, so (Lx)[i] = 256*x[i] - sum(x).
;
; ABI:
;   r1 = input address, 256 unsigned 16-bit values in word payloads
;   r2 = output address, 256 signed two's-complement 32-bit payloads
;   returns r1=1 on success, r1=0 if any input exceeds 0xFFFF
;   clobbers r3-r10; does not call helpers or allocate memory.
;
; Preconditions:
;   input and output spans each contain 256 words and do not overlap.
;   Each input payload is in [0,65535].
; Exact range proof:
;   0 <= sum(x) <= 256*65535 = 16,776,960.
;   0 <= 256*x[i] <= 16,776,960.
;   Therefore every output is in [-16,776,960,16,776,960], safely inside
;   signed 32-bit. No rounding, scaling, floating point or saturation occurs.
;
; Validation is a complete first pass. Invalid input leaves output unchanged.
; ST writes canonical +q/+q result words.

mth_k256_laplacian_u16:
  ADD r3,r1,r0
  MOVI r4,256
  MOVI r5,0

mth_k256_validate:
  CMP r4,r0
  JMP[EQ] @mth_k256_apply_setup
  LD r7,r3,0
  SHR r8,r7,16
  CMP r8,r0
  JMP[NE] @mth_k256_invalid
  ADD r5,r5,r7
  ADD r3,r3,r13
  SUB r4,r4,r13
  JMP @mth_k256_validate

mth_k256_apply_setup:
  ADD r3,r1,r0
  ADD r6,r2,r0
  MOVI r4,256

mth_k256_apply:
  CMP r4,r0
  JMP[EQ] @mth_k256_success
  LD r7,r3,0
  SHL r7,r7,8
  SUB r7,r7,r5
  ST r7,r6,0
  ADD r3,r3,r13
  ADD r6,r6,r13
  SUB r4,r4,r13
  JMP @mth_k256_apply

mth_k256_success:
  MOVI r1,1
  RET

mth_k256_invalid:
  MOVI r1,0
  RET
