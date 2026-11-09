.org 0
JMP @mth_dot256_test_main
.include "../nan16/mth_dot256_f32.nas"

mth_dot256_test_main:
  MOVI r13,1

  ; x = 2.0, y = 3.0 for all 256 -> dot = 1536.0 (0x44C00000).
  MOVI r1,@mth_dot256_x
  MOVI r2,@mth_dot256_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_dot256_fill_x:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_dot256_fill_x

  MOVI r2,@mth_dot256_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4040
mth_dot256_fill_y:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_dot256_fill_y

  MOVI r1,@mth_dot256_x
  MOVI r2,@mth_dot256_y
  MOVI r3,@mth_dot256_out
  CALL @mth_dot256_f32
  CMP r1,r13
  JMP[NE] @mth_dot256_fail

  MOVI r4,@mth_dot256_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x44C0
  CMP r5,r6
  JMP[NE] @mth_dot256_fail

  ; Zero x must produce zero.
  MOVI r1,@mth_dot256_x
  MOVI r6,256
mth_dot256_zero:
  ST r0,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_dot256_zero

  MOVI r1,@mth_dot256_x
  MOVI r2,@mth_dot256_y
  MOVI r3,@mth_dot256_out
  CALL @mth_dot256_f32
  CMP r1,r13
  JMP[NE] @mth_dot256_fail
  MOVI r4,@mth_dot256_out
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_dot256_fail

  MOVI r1,@mth_dot256_pass
  CALL @mth_dot256_puts
  HALT

mth_dot256_fail:
  MOVI r1,@mth_dot256_bad
  CALL @mth_dot256_puts
  HALT

mth_dot256_puts:
  MOVI r4,255
mth_dot256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_dot256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_dot256_puts_byte
  ADD r1,r1,r13
  JMP @mth_dot256_puts_word

.org 1024
mth_dot256_x:
.word 0
.org 2048
mth_dot256_y:
.word 0
.org 3072
mth_dot256_out:
.word 77
.org 3584
mth_dot256_pass:
.str "MTH DOT256 PASS\n"
mth_dot256_bad:
.str "MTH DOT256 FAIL\n"
