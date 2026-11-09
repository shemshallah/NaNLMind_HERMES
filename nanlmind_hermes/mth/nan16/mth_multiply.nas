; MTH full-width unsigned 32x32 -> 64-bit multiplication, NaN16 only.
; Inputs: r1=a, r2=b.
; Outputs: r1=low32(product), r2=high32(product).
; Uses shift-and-add so the mathematical implementation does not depend on a
; host multiply. Clobbers r3..r10; requires r13=1.

mth_u32_mul_wide:
  MOVI r3,0              ; accumulator high
  MOVI r4,0              ; accumulator low
  MOVI r5,0              ; shifted multiplicand high
  ADD r6,r1,r0           ; shifted multiplicand low
  ADD r7,r2,r0           ; multiplier
  MOVI r8,32             ; exactly 32 multiplier bits

mth_mul_loop:
  MOVI r10,1
  BOOL.8 r9,r7,r10       ; multiplier bit 0
  CMP r9,r0
  JMP[EQ] @mth_mul_shift

  ; Add the 64-bit shifted multiplicand to the 64-bit accumulator.
  ADD r10,r4,r0          ; old accumulator low
  ADD r4,r4,r6
  CMP r4,r10
  JMP[LT] @mth_mul_carry
  ADD r3,r3,r5
  JMP @mth_mul_shift

mth_mul_carry:
  ADD r3,r3,r5
  ADD r3,r3,r13

mth_mul_shift:
  ; Shift the 64-bit multiplicand left by one.
  SHR r9,r6,31
  SHL r6,r6,1
  SHL r5,r5,1
  BOOL.14 r5,r5,r9

  SHR r7,r7,1
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_mul_loop

  ADD r1,r4,r0
  ADD r2,r3,r0
  RET
