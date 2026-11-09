.org 0
JMP @mth_cosine_test_main
.include "../nan16/mth_cosine256_f32.nas"

mth_cosine_test_main:
  MOVI r13,1

  ; x=y=2.0 for all 256 -> cosine 1.0.
  MOVI r1,@mth_cosine_x
  MOVI r2,@mth_cosine_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_cosine_fill:
  ST r7,r1,0
  ST r7,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cosine_fill

  MOVI r1,@mth_cosine_x
  MOVI r2,@mth_cosine_y
  MOVI r3,@mth_cosine_out
  MOVI r4,@mth_cosine_scratch
  CALL @mth_cosine256_f32
  CMP r1,r13
  JMP[NE] @mth_cosine_fail
  MOVI r5,@mth_cosine_out
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x3F80
  CMP r6,r7
  JMP[NE] @mth_cosine_fail

  ; Orthogonal unit impulse: x[0]=1, y[1]=1 -> cosine 0.
  MOVI r1,@mth_cosine_x
  MOVI r2,@mth_cosine_y
  MOVI r6,256
mth_cosine_zero_fill:
  ST r0,r1,0
  ST r0,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cosine_zero_fill

  MOVI r1,@mth_cosine_x
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r1,0
  MOVI r2,@mth_cosine_y
  ADD r2,r2,r13
  ST r7,r2,0

  MOVI r1,@mth_cosine_x
  MOVI r2,@mth_cosine_y
  MOVI r3,@mth_cosine_out
  MOVI r4,@mth_cosine_scratch
  CALL @mth_cosine256_f32
  CMP r1,r13
  JMP[NE] @mth_cosine_fail
  MOVI r5,@mth_cosine_out
  LD r6,r5,0
  CMP r6,r0
  JMP[NE] @mth_cosine_fail

  MOVI r1,@mth_cosine_pass
  CALL @mth_cosine_puts
  HALT

mth_cosine_fail:
  MOVI r1,@mth_cosine_bad
  CALL @mth_cosine_puts
  HALT

mth_cosine_puts:
  MOVI r4,255
mth_cosine_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_cosine_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_cosine_puts_byte
  ADD r1,r1,r13
  JMP @mth_cosine_puts_word

.org 1024
mth_cosine_x:
.word 0
.org 2048
mth_cosine_y:
.word 0
.org 3072
mth_cosine_out:
.word 0
.org 3136
mth_cosine_scratch:
.word 0
.org 3200
mth_cosine_pass:
.str "MTH COSINE256 PASS\n"
mth_cosine_bad:
.str "MTH COSINE256 FAIL\n"
