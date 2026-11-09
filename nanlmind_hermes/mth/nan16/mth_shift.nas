; MTH variable-count 32-bit logical shifts — native NaN16.
; Inputs: r1=value, r2=count. Count 0..31 is valid.
; Output: r1=shifted value, r2=1 on success; invalid count returns 0,0.
; Left shift is explicitly modulo 2^32; right shift is logical (zero-fill).

mth_u32_shl:
  MOVI r4,31
  CMP r2,r4
  JMP[GT] @mth_u32_shift_bad
  ADD r3,r2,r0
mth_u32_shl_loop:
  CMP r3,r0
  JMP[EQ] @mth_u32_shift_ok
  SHL r1,r1,1
  SUB r3,r3,r13
  JMP @mth_u32_shl_loop

mth_u32_shr:
  MOVI r4,31
  CMP r2,r4
  JMP[GT] @mth_u32_shift_bad
  ADD r3,r2,r0
mth_u32_shr_loop:
  CMP r3,r0
  JMP[EQ] @mth_u32_shift_ok
  SHR r1,r1,1
  SUB r3,r3,r13
  JMP @mth_u32_shr_loop

mth_u32_shift_ok:
  MOVI r2,1
  RET

mth_u32_shift_bad:
  MOVI r1,0
  MOVI r2,0
  RET
