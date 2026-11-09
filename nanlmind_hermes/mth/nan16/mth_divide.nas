; MTH unsigned 32-bit division and remainder — restoring algorithm in NaN16.
; mth_u32_divmod inputs r1=dividend,r2=divisor.
; outputs r1=quotient,r2=remainder,r3=status (1 valid, 0 divide-by-zero).
; mth_u32_div returns r1=quotient,r2=status.
; mth_u32_mod returns r1=remainder,r2=status.
; All arithmetic and control flow below are NaN16 guest instructions.

mth_u32_divmod:
  CMP r2,r0
  JMP[EQ] @mth_div_zero
  ADD r6,r2,r0           ; divisor
  ADD r5,r1,r0           ; shifting quotient / dividend
  MOVI r3,0              ; 33rd remainder bit
  MOVI r4,0              ; remainder low
  MOVI r7,32

mth_div_loop:
  SHR r8,r5,31           ; next dividend bit
  SHL r5,r5,1
  SHR r9,r4,31           ; carry from remainder low
  SHL r4,r4,1
  BOOL.14 r4,r4,r8       ; append next dividend bit
  ADD r3,r9,r0           ; remainder high bit is this carry

  CMP r3,r0
  JMP[NE] @mth_div_subtract
  CMP r4,r6
  JMP[LT] @mth_div_no_subtract

mth_div_subtract:
  ADD r10,r4,r0          ; old low remainder
  SUB r4,r4,r6
  CMP r4,r10
  JMP[GT] @mth_div_borrow
  JMP @mth_div_set_quotient_bit
mth_div_borrow:
  SUB r3,r3,r13
mth_div_set_quotient_bit:
  ADD r5,r5,r13

mth_div_no_subtract:
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_div_loop

  ADD r1,r5,r0
  ADD r2,r4,r0
  MOVI r3,1
  RET

mth_div_zero:
  MOVI r1,0
  MOVI r2,0
  MOVI r3,0
  RET

mth_u32_div:
  ADD r11,r15,r0
  CALL @mth_u32_divmod
  ADD r15,r11,r0
  ADD r2,r3,r0
  RET

mth_u32_mod:
  ADD r11,r15,r0
  CALL @mth_u32_divmod
  ADD r1,r2,r0
  ADD r2,r3,r0
  ADD r15,r11,r0
  RET
