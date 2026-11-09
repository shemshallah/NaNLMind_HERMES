.org 0
JMP @mth_bit_test_main
.include "../nan16/mth_bitwise.nas"

mth_bit_test_main:
  MOVI r13,1
  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F

  CALL @mth_u32_and
  CMP r1,r0
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F
  CALL @mth_u32_or
  MOVI r3,0xFFFF
  MOVHI r3,0xFFFF
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F
  CALL @mth_u32_xor
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F
  CALL @mth_u32_nand
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F
  CALL @mth_u32_nor
  CMP r1,r0
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  MOVI r2,0x0F0F
  MOVHI r2,0x0F0F
  CALL @mth_u32_xnor
  CMP r1,r0
  JMP[NE] @mth_bit_fail

  MOVI r1,0xF0F0
  MOVHI r1,0xF0F0
  CALL @mth_u32_not
  MOVI r3,0x0F0F
  MOVHI r3,0x0F0F
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,7
  MOVI r2,9
  CALL @mth_u32_min
  MOVI r3,7
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,7
  MOVI r2,9
  CALL @mth_u32_max
  MOVI r3,9
  CMP r1,r3
  JMP[NE] @mth_bit_fail

  MOVI r1,7
  MOVI r2,9
  CALL @mth_u32_lt
  CMP r1,r13
  JMP[NE] @mth_bit_fail

  MOVI r1,7
  MOVI r2,7
  CALL @mth_u32_eq
  CMP r1,r13
  JMP[NE] @mth_bit_fail

  MOVI r1,@mth_bit_pass
  CALL @mth_bit_puts
  HALT

mth_bit_fail:
  MOVI r1,@mth_bit_bad
  CALL @mth_bit_puts
  HALT

mth_bit_puts:
  MOVI r4,255
mth_bit_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_bit_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_bit_puts_byte
  ADD r1,r1,r13
  JMP @mth_bit_puts_word

mth_bit_pass:
.str "MTH BITWISE/COMPARE PASS\n"
mth_bit_bad:
.str "MTH BITWISE/COMPARE FAIL\n"
