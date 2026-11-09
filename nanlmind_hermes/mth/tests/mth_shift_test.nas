.org 0
JMP @mth_shift_test_main
.include "../nan16/mth_shift.nas"

mth_shift_test_main:
  MOVI r13,1

  ; 3 << 2 = 12.
  MOVI r1,3
  MOVI r2,2
  CALL @mth_u32_shl
  MOVI r3,12
  CMP r1,r3
  JMP[NE] @mth_shift_fail
  CMP r2,r13
  JMP[NE] @mth_shift_fail

  ; 0x80000000 >> 31 = 1 (logical).
  MOVI r1,0
  MOVHI r1,0x8000
  MOVI r2,31
  CALL @mth_u32_shr
  CMP r1,r13
  JMP[NE] @mth_shift_fail
  CMP r2,r13
  JMP[NE] @mth_shift_fail

  ; Count 32 is invalid, not silently masked.
  MOVI r1,1
  MOVI r2,32
  CALL @mth_u32_shl
  CMP r1,r0
  JMP[NE] @mth_shift_fail
  CMP r2,r0
  JMP[NE] @mth_shift_fail

  MOVI r1,@mth_shift_pass
  CALL @mth_shift_puts
  HALT

mth_shift_fail:
  MOVI r1,@mth_shift_bad
  CALL @mth_shift_puts
  HALT

mth_shift_puts:
  MOVI r4,255
mth_shift_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_shift_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_shift_puts_byte
  ADD r1,r1,r13
  JMP @mth_shift_puts_word

mth_shift_pass:
.str "MTH SHIFT PASS\n"
mth_shift_bad:
.str "MTH SHIFT FAIL\n"
