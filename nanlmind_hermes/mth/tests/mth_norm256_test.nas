.org 0
JMP @mth_norm256_test_main
.include "../nan16/mth_norm256_f32.nas"

mth_norm256_test_main:
  MOVI r13,1

  ; x = 3.0 for all 256 -> norm = sqrt(2304) = 48.0 (0x42400000).
  MOVI r1,@mth_norm256_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4040
mth_norm256_fill:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_norm256_fill

  MOVI r1,@mth_norm256_x
  MOVI r2,@mth_norm256_out
  CALL @mth_norm256_f32
  CMP r1,r13
  JMP[NE] @mth_norm256_fail

  MOVI r4,@mth_norm256_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4240
  CMP r5,r6
  JMP[NE] @mth_norm256_fail

  ; Zero vector -> zero norm.
  MOVI r1,@mth_norm256_x
  MOVI r6,256
mth_norm256_zero:
  ST r0,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_norm256_zero

  MOVI r1,@mth_norm256_x
  MOVI r2,@mth_norm256_out
  CALL @mth_norm256_f32
  CMP r1,r13
  JMP[NE] @mth_norm256_fail
  MOVI r4,@mth_norm256_out
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_norm256_fail

  MOVI r1,@mth_norm256_pass
  CALL @mth_norm256_puts
  HALT

mth_norm256_fail:
  MOVI r1,@mth_norm256_bad
  CALL @mth_norm256_puts
  HALT

mth_norm256_puts:
  MOVI r4,255
mth_norm256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_norm256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_norm256_puts_byte
  ADD r1,r1,r13
  JMP @mth_norm256_puts_word

.org 1024
mth_norm256_x:
.word 0
.org 2048
mth_norm256_out:
.word 77
.org 2560
mth_norm256_pass:
.str "MTH NORM256 PASS\n"
mth_norm256_bad:
.str "MTH NORM256 FAIL\n"
