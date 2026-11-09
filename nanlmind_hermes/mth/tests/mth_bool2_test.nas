.org 0
JMP @mth_bool_test_main
.include "../nan16/mth_bool.nas"

mth_bool_test_main:
  MOVI r13,1
  MOVI r12,0

mth_bool_loop:
  ADD r1,r12,r0
  LD r10,r12,@mth_bool_expected
  LD r5,r0,@mth_bool_a
  LD r6,r0,@mth_bool_b
  CALL @mth_bool2
  CMP r4,r10
  JMP[NE] @mth_bool_fail

  ADD r12,r12,r13
  MOVI r8,16
  CMP r12,r8
  JMP[LT] @mth_bool_loop

  MOVI r1,@mth_bool_pass
  CALL @mth_bool_puts
  HALT

mth_bool_fail:
  MOVI r1,@mth_bool_bad
  CALL @mth_bool_puts
  HALT

mth_bool_puts:
  MOVI r4,255
mth_bool_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_bool_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_bool_puts_byte
  ADD r1,r1,r13
  JMP @mth_bool_puts_word

mth_bool_a:
.word 0xCCCCCCCC
mth_bool_b:
.word 0xAAAAAAAA
mth_bool_expected:
.word 0x00000000
.word 0x11111111
.word 0x22222222
.word 0x33333333
.word 0x44444444
.word 0x55555555
.word 0x66666666
.word 0x77777777
.word 0x88888888
.word 0x99999999
.word 0xAAAAAAAA
.word 0xBBBBBBBB
.word 0xCCCCCCCC
.word 0xDDDDDDDD
.word 0xEEEEEEEE
.word 0xFFFFFFFF
mth_bool_pass:
.str "MTH BOOL2 ALL 16 PASS\n"
mth_bool_bad:
.str "MTH BOOL2 FAIL\n"
