; MTH unsigned 32-bit integer primitives — native NaN16.
; Inputs: r1=a, r2=b.
; mth_u32_add_checked -> r1=sum modulo 2^32, r2=1 if exact/no overflow else 0.
; mth_u32_sub_checked -> r1=difference, r2=1 if a>=b else 0 (underflow returns 0).
; Clobbers r3 and condition flags. No host arithmetic.

mth_u32_add_checked:
  ADD r3,r1,r0
  ADD r1,r1,r2
  CMP r1,r3
  JMP[LT] @mth_u32_add_overflow
  MOVI r2,1
  RET
mth_u32_add_overflow:
  MOVI r2,0
  RET

mth_u32_sub_checked:
  CMP r1,r2
  JMP[LT] @mth_u32_sub_underflow
  SUB r1,r1,r2
  MOVI r2,1
  RET
mth_u32_sub_underflow:
  MOVI r1,0
  MOVI r2,0
  RET
