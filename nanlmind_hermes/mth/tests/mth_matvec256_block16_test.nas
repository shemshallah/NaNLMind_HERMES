.org 0
JMP @mth_matvec256_block16_test_main
.include "../nan16/mth_matvec256_block16.nas"

mth_matvec256_block16_test_main:
  MOVI r13,1

  ; Zero the 16x256 A block (4,096 words).
  MOVI r4,@mth_matvec256_block16_a
  MOVI r6,4096
mth_matvec256_block16_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_block16_zero_a

  ; A[k,k] = 2.0 for k=0..15; row stride 256 plus 1 => 257.
  MOVI r5,@mth_matvec256_block16_a
  MOVI r6,16
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_matvec256_block16_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_block16_diag

  ; x = 1.0 for all 256; expect y[k] = 2.0 for all 16.
  MOVI r2,@mth_matvec256_block16_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_matvec256_block16_fill_x_pos:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_block16_fill_x_pos

  MOVI r1,@mth_matvec256_block16_a
  MOVI r2,@mth_matvec256_block16_x
  MOVI r3,@mth_matvec256_block16_y
  CALL @mth_matvec256_block16
  CMP r1,r13
  JMP[NE] @mth_matvec256_block16_fail

  MOVI r4,@mth_matvec256_block16_y
  MOVI r6,0
  MOVHI r6,0x4000
  MOVI r7,0
mth_matvec256_block16_check_pos:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_block16_fail
  ADD r7,r7,r13
  MOVI r8,16
  CMP r7,r8
  JMP[LT] @mth_matvec256_block16_check_pos

  ; x = -1.0 for all 256; expect y[k] = -2.0 for all 16.
  MOVI r2,@mth_matvec256_block16_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0xBF80
mth_matvec256_block16_fill_x_neg:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_block16_fill_x_neg

  MOVI r1,@mth_matvec256_block16_a
  MOVI r2,@mth_matvec256_block16_x
  MOVI r3,@mth_matvec256_block16_y
  CALL @mth_matvec256_block16
  CMP r1,r13
  JMP[NE] @mth_matvec256_block16_fail

  MOVI r4,@mth_matvec256_block16_y
  MOVI r6,0
  MOVHI r6,0xC000
  MOVI r7,0
mth_matvec256_block16_check_neg:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_block16_fail
  ADD r7,r7,r13
  MOVI r8,16
  CMP r7,r8
  JMP[LT] @mth_matvec256_block16_check_neg

  ; x = +0.0 for all 256; expect y[k] = +0.0 for all 16.
  MOVI r2,@mth_matvec256_block16_x
  MOVI r6,256
  MOVI r7,0
mth_matvec256_block16_fill_x_zero:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_block16_fill_x_zero

  MOVI r1,@mth_matvec256_block16_a
  MOVI r2,@mth_matvec256_block16_x
  MOVI r3,@mth_matvec256_block16_y
  CALL @mth_matvec256_block16
  CMP r1,r13
  JMP[NE] @mth_matvec256_block16_fail

  MOVI r4,@mth_matvec256_block16_y
  MOVI r6,0
  MOVI r7,0
mth_matvec256_block16_check_zero:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_block16_fail
  ADD r7,r7,r13
  MOVI r8,16
  CMP r7,r8
  JMP[LT] @mth_matvec256_block16_check_zero

  MOVI r1,@mth_matvec256_block16_pass
  CALL @mth_matvec256_block16_puts
  HALT

mth_matvec256_block16_fail:
  MOVI r1,@mth_matvec256_block16_bad
  CALL @mth_matvec256_block16_puts
  HALT

mth_matvec256_block16_puts:
  MOVI r4,255
mth_matvec256_block16_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_matvec256_block16_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_matvec256_block16_puts_byte
  ADD r1,r1,r13
  JMP @mth_matvec256_block16_puts_word

.org 512
mth_matvec256_block16_a:
.word 0
.org 4608
mth_matvec256_block16_x:
.word 0
.org 4864
mth_matvec256_block16_y:
.word 0
.org 5120
mth_matvec256_block16_pass:
.str "MTH MATVEC256 BLOCK16 PASS\n"
mth_matvec256_block16_bad:
.str "MTH MATVEC256 BLOCK16 FAIL\n"
