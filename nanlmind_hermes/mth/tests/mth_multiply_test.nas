.org 0
JMP @mth_mul_test_main
.include "../nan16/mth_multiply.nas"

mth_mul_test_main:
  MOVI r13,1

  ; 3 * 7 = 21.
  MOVI r1,3
  MOVI r2,7
  CALL @mth_u32_mul_wide
  MOVI r3,21
  CMP r1,r3
  JMP[NE] @mth_mul_fail
  CMP r2,r0
  JMP[NE] @mth_mul_fail

  ; (2^32-1) * 2 = 0x00000001FFFFFFFE.
  MOVI r1,0xFFFF
  MOVHI r1,0xFFFF
  MOVI r2,2
  CALL @mth_u32_mul_wide
  MOVI r3,0xFFFE
  MOVHI r3,0xFFFF
  CMP r1,r3
  JMP[NE] @mth_mul_fail
  CMP r2,r13
  JMP[NE] @mth_mul_fail

  ; (2^32-1)^2 = 0xFFFFFFFE00000001.
  MOVI r1,0xFFFF
  MOVHI r1,0xFFFF
  MOVI r2,0xFFFF
  MOVHI r2,0xFFFF
  CALL @mth_u32_mul_wide
  MOVI r3,1
  CMP r1,r3
  JMP[NE] @mth_mul_fail
  MOVI r3,0xFFFE
  MOVHI r3,0xFFFF
  CMP r2,r3
  JMP[NE] @mth_mul_fail

  MOVI r1,@mth_mul_pass
  CALL @mth_mul_puts
  HALT

mth_mul_fail:
  MOVI r1,@mth_mul_bad
  CALL @mth_mul_puts
  HALT

mth_mul_puts:
  MOVI r4,255
mth_mul_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_mul_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_mul_puts_byte
  ADD r1,r1,r13
  JMP @mth_mul_puts_word

mth_mul_pass:
.str "MTH MUL WIDE PASS\n"
mth_mul_bad:
.str "MTH MUL WIDE FAIL\n"
