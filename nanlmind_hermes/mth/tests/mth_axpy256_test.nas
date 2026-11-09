.org 0
JMP @mth_axpy256_test_main
.include "../nan16/mth_axpy256_f32.nas"

mth_axpy256_test_main:
  MOVI r13,1

  ; x=2.0, y=1.0, alpha=0.5 -> y=2.0.
  MOVI r1,@mth_axpy256_x
  MOVI r2,@mth_axpy256_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_axpy256_fill_x:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_axpy256_fill_x

  MOVI r2,@mth_axpy256_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_axpy256_fill_y:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_axpy256_fill_y

  MOVI r1,@mth_axpy256_x
  MOVI r2,@mth_axpy256_y
  MOVI r3,@mth_axpy256_alpha
  CALL @mth_axpy256_f32
  CMP r1,r13
  JMP[NE] @mth_axpy256_fail

  MOVI r4,@mth_axpy256_y
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_axpy256_fail

  MOVI r6,255
  ADD r6,r4,r6
  LD r5,r6,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_axpy256_fail

  ; alpha=0 leaves y unchanged.
  MOVI r3,@mth_axpy256_alpha
  ST r0,r3,0
  MOVI r1,@mth_axpy256_x
  MOVI r2,@mth_axpy256_y
  CALL @mth_axpy256_f32
  CMP r1,r13
  JMP[NE] @mth_axpy256_fail
  MOVI r4,@mth_axpy256_y
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_axpy256_fail

  MOVI r1,@mth_axpy256_pass
  CALL @mth_axpy256_puts
  HALT

mth_axpy256_fail:
  MOVI r1,@mth_axpy256_bad
  CALL @mth_axpy256_puts
  HALT

mth_axpy256_puts:
  MOVI r4,255
mth_axpy256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_axpy256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_axpy256_puts_byte
  ADD r1,r1,r13
  JMP @mth_axpy256_puts_word

.org 1024
mth_axpy256_x:
.word 0
.org 2048
mth_axpy256_y:
.word 0
.org 3072
mth_axpy256_alpha:
.word 0x3F000000
.org 3584
mth_axpy256_pass:
.str "MTH AXPY256 PASS\n"
mth_axpy256_bad:
.str "MTH AXPY256 FAIL\n"
