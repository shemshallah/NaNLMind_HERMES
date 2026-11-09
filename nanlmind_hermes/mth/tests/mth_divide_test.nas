.org 0
JMP @mth_div_test_main
.include "../nan16/mth_divide.nas"

mth_div_test_main:
  MOVI r13,1

  ; 100 / 7 = 14 remainder 2.
  MOVI r1,100
  MOVI r2,7
  CALL @mth_u32_divmod
  MOVI r4,14
  CMP r1,r4
  JMP[NE] @mth_div_fail
  MOVI r4,2
  CMP r2,r4
  JMP[NE] @mth_div_fail
  CMP r3,r13
  JMP[NE] @mth_div_fail

  ; Quotient-only and remainder-only wrappers preserve the caller link.
  MOVI r1,100
  MOVI r2,7
  CALL @mth_u32_div
  MOVI r4,14
  CMP r1,r4
  JMP[NE] @mth_div_fail
  CMP r2,r13
  JMP[NE] @mth_div_fail

  MOVI r1,100
  MOVI r2,7
  CALL @mth_u32_mod
  MOVI r4,2
  CMP r1,r4
  JMP[NE] @mth_div_fail
  CMP r2,r13
  JMP[NE] @mth_div_fail

  ; UINT32_MAX / 3 = 0x55555555 remainder 0.
  MOVI r1,0xFFFF
  MOVHI r1,0xFFFF
  MOVI r2,3
  CALL @mth_u32_divmod
  MOVI r4,0x5555
  MOVHI r4,0x5555
  CMP r1,r4
  JMP[NE] @mth_div_fail
  CMP r2,r0
  JMP[NE] @mth_div_fail
  CMP r3,r13
  JMP[NE] @mth_div_fail

  ; Division by zero has an explicit status, never a guessed quotient.
  MOVI r1,123
  MOVI r2,0
  CALL @mth_u32_divmod
  CMP r1,r0
  JMP[NE] @mth_div_fail
  CMP r2,r0
  JMP[NE] @mth_div_fail
  CMP r3,r0
  JMP[NE] @mth_div_fail

  MOVI r1,@mth_div_pass
  CALL @mth_div_puts
  HALT

mth_div_fail:
  MOVI r1,@mth_div_bad
  CALL @mth_div_puts
  HALT

mth_div_puts:
  MOVI r4,255
mth_div_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_div_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_div_puts_byte
  ADD r1,r1,r13
  JMP @mth_div_puts_word

mth_div_pass:
.str "MTH DIVMOD PASS\n"
mth_div_bad:
.str "MTH DIVMOD FAIL\n"
