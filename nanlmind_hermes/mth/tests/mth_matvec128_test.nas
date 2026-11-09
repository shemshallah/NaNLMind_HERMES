.org 0
JMP @mth_matvec128_test_main
.include "../nan16/mth_matvec128_f32.nas"

mth_matvec128_test_main:
  MOVI r13,1

  ; Zero A (16,384 words), then set diagonal entries to 2.0.
  MOVI r4,@mth_matvec128_a
  MOVI r6,16384
mth_matvec128_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec128_zero_a

  MOVI r5,@mth_matvec128_a
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x4000
mth_matvec128_diag:
  ST r7,r5,0
  MOVI r8,129
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec128_diag

  ; x = 1.0 for all 128.
  MOVI r2,@mth_matvec128_x
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_matvec128_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec128_fill_x

  MOVI r1,@mth_matvec128_a
  MOVI r2,@mth_matvec128_x
  MOVI r3,@mth_matvec128_y
  CALL @mth_matvec128_f32
  CMP r1,r13
  JMP[NE] @mth_matvec128_fail

  MOVI r4,@mth_matvec128_y
  MOVI r6,0
  MOVHI r6,0x4000
  MOVI r7,0
mth_matvec128_check:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec128_fail
  ADD r7,r7,r13
  MOVI r8,128
  CMP r7,r8
  JMP[LT] @mth_matvec128_check

  MOVI r1,@mth_matvec128_pass
  CALL @mth_matvec128_puts
  HALT

mth_matvec128_fail:
  MOVI r1,@mth_matvec128_bad
  CALL @mth_matvec128_puts
  HALT

mth_matvec128_puts:
  MOVI r4,255
mth_matvec128_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_matvec128_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_matvec128_puts_byte
  ADD r1,r1,r13
  JMP @mth_matvec128_puts_word

.org 512
mth_matvec128_a:
.word 0
.org 17024
mth_matvec128_x:
.word 0
.org 17280
mth_matvec128_y:
.word 0
.org 17536
mth_matvec128_pass:
.str "MTH MATVEC128 PASS\n"
mth_matvec128_bad:
.str "MTH MATVEC128 FAIL\n"
