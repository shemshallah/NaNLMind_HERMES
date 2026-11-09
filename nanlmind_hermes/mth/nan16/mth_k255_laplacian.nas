; MTH exact K255 head-only graph Laplacian — NaN16 only.
; Operator: L = 255 I - J, so (Lx)[i] = 255*x[i] - sum(x) over 255 heads.
;
; ABI:
;   r1 = input address, 255 unsigned 16-bit values in word payloads
;   r2 = output address, 255 signed two's-complement 32-bit payloads
;   returns r1=1 on success, r1=0 if any input exceeds 0xFFFF
;   clobbers r3-r10; does not call helpers or allocate memory.
;
; Preconditions:
;   input and output spans each contain 255 words and do not overlap.
;   Each input payload is in [0,65535].
; Exact range proof:
;   0 <= sum(x) <= 255*65535 = 16,711,425.
;   0 <= 255*x[i] <= 16,711,425.
;   Every output is in [-16,711,425,16,711,425], safely inside
;   signed 32-bit. No rounding, scaling, floating point or saturation occurs.
;
; Validation is a complete first pass. Invalid input leaves output unchanged.
; ST writes canonical +q/+q result words.

mth_k255_laplacian_u16:
  ADD r3,r1,r0
  MOVI r4,255
  MOVI r5,0

mth_k255_validate:
  CMP r4,r0
  JMP[EQ] @mth_k255_apply_setup
  LD r7,r3,0
  SHR r8,r7,16
  CMP r8,r0
  JMP[NE] @mth_k255_invalid
  ADD r5,r5,r7
  ADD r3,r3,r13
  SUB r4,r4,r13
  JMP @mth_k255_validate

mth_k255_apply_setup:
  ADD r3,r1,r0
  ADD r6,r2,r0
  MOVI r4,255

mth_k255_apply:
  CMP r4,r0
  JMP[EQ] @mth_k255_success
  LD r7,r3,0
  ADD r8,r7,r0
  SHL r7,r7,8
  SUB r7,r7,r8
  SUB r7,r7,r5
  ST r7,r6,0
  ADD r3,r3,r13
  ADD r6,r6,r13
  SUB r4,r4,r13
  JMP @mth_k255_apply

mth_k255_success:
  MOVI r1,1
  RET

mth_k255_invalid:
  MOVI r1,0
  RET
