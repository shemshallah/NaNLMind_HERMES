.org 0
JMP @mth_int_test_main
.include "../nan16/mth_integer.nas"

mth_int_test_main:
  MOVI r13,1

  ; 7 + 9 = 16, exact.
  MOVI r1,7
  MOVI r2,9
  CALL @mth_u32_add_checked
  MOVI r3,16
  CMP r1,r3
  JMP[NE] @mth_int_fail
  CMP r2,r13
  JMP[NE] @mth_int_fail

  ; UINT32_MAX + 1 wraps to zero but reports overflow.
  MOVI r1,0xFFFF
  MOVHI r1,0xFFFF
  MOVI r2,1
  CALL @mth_u32_add_checked
  CMP r1,r0
  JMP[NE] @mth_int_fail
  CMP r2,r0
  JMP[NE] @mth_int_fail

  ; 9 - 7 = 2, exact.
  MOVI r1,9
  MOVI r2,7
  CALL @mth_u32_sub_checked
  MOVI r3,2
  CMP r1,r3
  JMP[NE] @mth_int_fail
  CMP r2,r13
  JMP[NE] @mth_int_fail

  ; 0 - 1 is an underflow and returns explicit status 0.
  MOVI r1,0
  MOVI r2,1
  CALL @mth_u32_sub_checked
  CMP r1,r0
  JMP[NE] @mth_int_fail
  CMP r2,r0
  JMP[NE] @mth_int_fail

  MOVI r1,@mth_int_pass
  CALL @mth_int_puts
  HALT

mth_int_fail:
  MOVI r1,@mth_int_bad
  CALL @mth_int_puts
  HALT

mth_int_puts:
  MOVI r4,255
mth_int_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_int_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_int_puts_byte
  ADD r1,r1,r13
  JMP @mth_int_puts_word

mth_int_pass:
.str "MTH INT32 CHECKED PASS\n"
mth_int_bad:
.str "MTH INT32 CHECKED FAIL\n"
