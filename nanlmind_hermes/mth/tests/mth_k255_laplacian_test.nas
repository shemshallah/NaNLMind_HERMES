.org 0
JMP @mth_k255_test_main
.include "../nan16/mth_k255_laplacian.nas"

mth_k255_test_main:
  MOVI r13,1

  ; Impulse x=[1,0,...,0]. For K255, Lx=[254,-1,...,-1].
  MOVI r1,@mth_k255_input
  MOVI r2,@mth_k255_output
  CALL @mth_k255_laplacian_u16
  CMP r1,r13
  JMP[NE] @mth_k255_test_fail

  MOVI r3,@mth_k255_output
  LD r4,r3,0
  MOVI r5,254
  CMP r4,r5
  JMP[NE] @mth_k255_test_fail

  LD r4,r3,1
  MOVI r5,0xFFFF
  MOVHI r5,0xFFFF
  CMP r4,r5
  JMP[NE] @mth_k255_test_fail

  LD r4,r3,254
  CMP r4,r5
  JMP[NE] @mth_k255_test_fail

  ; Put a value outside the U16 input contract at input[1].
  ; The preflight must reject it without changing output[0].
  MOVI r6,@mth_k255_input
  ADD r6,r6,r13
  MOVI r7,0
  MOVHI r7,1
  ST r7,r6,0
  MOVI r7,77
  ST r7,r3,0

  MOVI r1,@mth_k255_input
  MOVI r2,@mth_k255_output
  CALL @mth_k255_laplacian_u16
  CMP r1,r0
  JMP[NE] @mth_k255_test_fail

  ; The kernel clobbers r3-r10 by ABI; reload the output base.
  MOVI r3,@mth_k255_output
  LD r4,r3,0
  MOVI r5,77
  CMP r4,r5
  JMP[NE] @mth_k255_test_fail

  MOVI r1,@mth_k255_test_pass
  CALL @mth_k255_test_puts
  HALT

mth_k255_test_fail:
  MOVI r1,@mth_k255_test_bad
  CALL @mth_k255_test_puts
  HALT

mth_k255_test_puts:
  MOVI r4,255
mth_k255_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_k255_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_k255_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_k255_test_puts_word

 ; Fixed, non-overlapping 255-word test regions.
.org 512
mth_k255_input:
.word 1
.org 1024
mth_k255_output:
.word 77
.org 2048
mth_k255_test_pass:
.str "MTH K255 LAPLACIAN PASS\n"
mth_k255_test_bad:
.str "MTH K255 LAPLACIAN FAIL\n"
