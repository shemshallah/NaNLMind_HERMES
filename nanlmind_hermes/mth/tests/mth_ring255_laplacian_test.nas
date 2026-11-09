.org 0
JMP @mth_ring255_test_main
.include "../nan16/mth_ring255_laplacian.nas"

mth_ring255_test_main:
  MOVI r13,1

  ; Impulse at vertex 0: [2,-1,0,...,0,-1].
  MOVI r1,@mth_ring255_input
  MOVI r2,@mth_ring255_output
  CALL @mth_ring255_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring255_fail

  MOVI r3,@mth_ring255_output
  LD r4,r3,0
  MOVI r5,0
  MOVHI r5,0x4000
  CMP r4,r5
  JMP[NE] @mth_ring255_fail

  LD r4,r3,1
  MOVI r5,0
  MOVHI r5,0xBF80
  CMP r4,r5
  JMP[NE] @mth_ring255_fail

  LD r4,r3,2
  CMP r4,r0
  JMP[NE] @mth_ring255_fail

  MOVI r6,254
  ADD r6,r3,r6
  LD r4,r6,0
  MOVI r5,0
  MOVHI r5,0xBF80
  CMP r4,r5
  JMP[NE] @mth_ring255_fail

  ; Constant state must map to exactly zero.
  MOVI r6,@mth_ring255_input
  MOVI r8,255
  MOVI r7,0
  MOVHI r7,0x4000
mth_ring255_fill:
  ST r7,r6,0
  ADD r6,r6,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring255_fill

  MOVI r1,@mth_ring255_input
  MOVI r2,@mth_ring255_output
  CALL @mth_ring255_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring255_fail
  MOVI r3,@mth_ring255_output
  MOVI r8,255
mth_ring255_check_zero:
  LD r4,r3,0
  CMP r4,r0
  JMP[NE] @mth_ring255_fail
  ADD r3,r3,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring255_check_zero

  MOVI r1,@mth_ring255_pass
  CALL @mth_ring255_puts
  HALT

mth_ring255_fail:
  MOVI r1,@mth_ring255_bad
  CALL @mth_ring255_puts
  HALT

mth_ring255_puts:
  MOVI r4,255
mth_ring255_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_ring255_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_ring255_puts_byte
  ADD r1,r1,r13
  JMP @mth_ring255_puts_word

.org 512
mth_ring255_input:
.word 0x3F800000
.org 1024
mth_ring255_output:
.word 77
.org 2048
mth_ring255_pass:
.str "MTH RING255 LAPLACIAN PASS\n"
mth_ring255_bad:
.str "MTH RING255 LAPLACIAN FAIL\n"
