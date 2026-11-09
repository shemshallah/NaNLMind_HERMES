.org 0
JMP @mth_matvec256_full_test_main
.include "../nan16/mth_matvec256_full.nas"

mth_matvec256_full_test_main:
  MOVI r13,1

  ; Zero A (65,536 words). MOVI r6,0 + MOVHI r6,0x0001 builds 65,536.
  LA r4,@mth_matvec256_full_a
  MOVI r6,0
  MOVHI r6,0x0001
mth_matvec256_full_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_full_zero_a

  ; A[i,i] = 2.0 for i=0..255; row stride 256 plus 1 => 257.
  LA r5,@mth_matvec256_full_a
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_matvec256_full_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_full_diag

  ; x = 1.0 for all 256; expect y[i] = 2.0 for all 256.
  LA r2,@mth_matvec256_full_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_matvec256_full_fill_x_pos:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_full_fill_x_pos

  LA r1,@mth_matvec256_full_a
  LA r2,@mth_matvec256_full_x
  LA r3,@mth_matvec256_full_y
  CALL @mth_matvec256_full
  CMP r1,r13
  JMP[NE] @mth_matvec256_full_fail

  LA r4,@mth_matvec256_full_y
  MOVI r6,0
  MOVHI r6,0x4000
  MOVI r7,0
mth_matvec256_full_check_pos:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_matvec256_full_check_pos

  ; x = -1.0 for all 256; expect y[i] = -2.0 for all 256.
  LA r2,@mth_matvec256_full_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0xBF80
mth_matvec256_full_fill_x_neg:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_full_fill_x_neg

  LA r1,@mth_matvec256_full_a
  LA r2,@mth_matvec256_full_x
  LA r3,@mth_matvec256_full_y
  CALL @mth_matvec256_full
  CMP r1,r13
  JMP[NE] @mth_matvec256_full_fail

  LA r4,@mth_matvec256_full_y
  MOVI r6,0
  MOVHI r6,0xC000
  MOVI r7,0
mth_matvec256_full_check_neg:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_matvec256_full_check_neg

  ; x = +0.0 for all 256; expect y[i] = +0.0 for all 256.
  LA r2,@mth_matvec256_full_x
  MOVI r6,256
  MOVI r7,0
mth_matvec256_full_fill_x_zero:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_matvec256_full_fill_x_zero

  LA r1,@mth_matvec256_full_a
  LA r2,@mth_matvec256_full_x
  LA r3,@mth_matvec256_full_y
  CALL @mth_matvec256_full
  CMP r1,r13
  JMP[NE] @mth_matvec256_full_fail

  LA r4,@mth_matvec256_full_y
  MOVI r6,0
  MOVI r7,0
mth_matvec256_full_check_zero:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_matvec256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_matvec256_full_check_zero

  ; Non-uniform x ramp: x[i] = float(i+1). With A=2I, y[i] = 2*x[i].
  ; This catches any column/row alignment error that a uniform vector hides.
  LA r2,@mth_matvec256_full_x
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x3F80
mth_matvec256_full_fill_x_ramp:
  CVTIF r8,r6
  FADD r8,r8,r7
  ADD r9,r2,r6
  ST r8,r9,0
  ADD r6,r6,r13
  MOVI r9,256
  CMP r6,r9
  JMP[LT] @mth_matvec256_full_fill_x_ramp

  LA r1,@mth_matvec256_full_a
  LA r2,@mth_matvec256_full_x
  LA r3,@mth_matvec256_full_y
  CALL @mth_matvec256_full
  CMP r1,r13
  JMP[NE] @mth_matvec256_full_fail

  LA r4,@mth_matvec256_full_y
  LA r5,@mth_matvec256_full_x
  MOVI r6,0
  MOVHI r6,0x4000
  MOVI r7,0
mth_matvec256_full_check_ramp:
  ADD r8,r4,r7
  LD r9,r8,0
  ADD r8,r5,r7
  LD r10,r8,0
  FMUL r10,r10,r6
  CMP r9,r10
  JMP[NE] @mth_matvec256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_matvec256_full_check_ramp

  LA r1,@mth_matvec256_full_pass
  CALL @mth_matvec256_full_puts
  HALT

mth_matvec256_full_fail:
  LA r1,@mth_matvec256_full_bad
  CALL @mth_matvec256_full_puts
  HALT

mth_matvec256_full_puts:
  MOVI r4,255
mth_matvec256_full_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_matvec256_full_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_matvec256_full_puts_byte
  ADD r1,r1,r13
  JMP @mth_matvec256_full_puts_word

.org 512
mth_matvec256_full_a:
.word 0
.org 66048
mth_matvec256_full_x:
.word 0
.org 66304
mth_matvec256_full_y:
.word 0
.org 66560
mth_matvec256_full_pass:
.str "MTH MATVEC256 FULL PASS\n"
mth_matvec256_full_bad:
.str "MTH MATVEC256 FULL FAIL\n"
