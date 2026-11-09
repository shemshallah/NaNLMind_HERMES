.org 0
JMP @mth_ring256_test_main
.include "../nan16/mth_ring256_laplacian.nas"

mth_ring256_test_main:
  MOVI r13,1

  ; Case 1: impulse at vertex 0 -> [2,-1,0,...,0,-1].
  MOVI r6,@mth_ring256_input
  MOVI r8,256
mth_ring256_zero_input:
  ST r0,r6,0
  ADD r6,r6,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring256_zero_input
  MOVI r6,@mth_ring256_input
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r6,0

  MOVI r1,@mth_ring256_input
  MOVI r2,@mth_ring256_output
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring256_fail

  MOVI r3,@mth_ring256_output
  LD r4,r3,0
  MOVI r5,0
  MOVHI r5,0x4000            ; 2.0
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  LD r4,r3,1
  MOVI r5,0
  MOVHI r5,0xBF80            ; -1.0
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  LD r4,r3,2
  CMP r4,r0
  JMP[NE] @mth_ring256_fail

  LD r4,r3,254
  CMP r4,r0
  JMP[NE] @mth_ring256_fail

  MOVI r6,255
  ADD r6,r3,r6
  LD r4,r6,0
  MOVI r5,0
  MOVHI r5,0xBF80            ; -1.0 at wrapped neighbour 255
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  ; Case 2: constant 1.0 -> exactly zero everywhere.
  MOVI r6,@mth_ring256_input
  MOVI r8,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_ring256_fill_one:
  ST r7,r6,0
  ADD r6,r6,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring256_fill_one

  MOVI r1,@mth_ring256_input
  MOVI r2,@mth_ring256_output
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring256_fail
  MOVI r3,@mth_ring256_output
  MOVI r8,256
mth_ring256_check_zero:
  LD r4,r3,0
  CMP r4,r0
  JMP[NE] @mth_ring256_fail
  ADD r3,r3,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring256_check_zero

  ; Case 3: impulse at interior vertex 128 -> neighbours 127 and 129.
  MOVI r6,@mth_ring256_input
  MOVI r8,256
mth_ring256_zero_input2:
  ST r0,r6,0
  ADD r6,r6,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_ring256_zero_input2
  MOVI r6,@mth_ring256_input
  MOVI r7,128
  ADD r6,r6,r7
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r6,0

  MOVI r1,@mth_ring256_input
  MOVI r2,@mth_ring256_output
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring256_fail

  MOVI r3,@mth_ring256_output
  MOVI r6,128
  ADD r6,r3,r6
  LD r4,r6,0
  MOVI r5,0
  MOVHI r5,0x4000
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  LD r4,r3,127
  MOVI r5,0
  MOVHI r5,0xBF80
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  LD r4,r3,129
  MOVI r5,0
  MOVHI r5,0xBF80
  CMP r4,r5
  JMP[NE] @mth_ring256_fail

  LD r4,r3,0
  CMP r4,r0
  JMP[NE] @mth_ring256_fail

  MOVI r1,@mth_ring256_pass
  CALL @mth_ring256_puts
  HALT

mth_ring256_fail:
  MOVI r1,@mth_ring256_bad
  CALL @mth_ring256_puts
  HALT

mth_ring256_puts:
  MOVI r4,255
mth_ring256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_ring256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_ring256_puts_byte
  ADD r1,r1,r13
  JMP @mth_ring256_puts_word

.org 512
mth_ring256_input:
.word 0
.org 1024
mth_ring256_output:
.word 77
.org 2048
mth_ring256_pass:
.str "MTH RING256 LAPLACIAN PASS\n"
mth_ring256_bad:
.str "MTH RING256 LAPLACIAN FAIL\n"
