.org 0
JMP @mth_adapter_residual_256_block16_test_main
.include "../nan16/mth_adapter_residual_256_block16.nas"

mth_adapter_residual_256_block16_test_main:
  MOVI r13,1

  ; Zero the 16x256 C block (4,096 words).
  MOVI r4,@mth_adapter_residual_256_block16_c
  MOVI r6,4096
mth_adapter_residual_256_block16_zero_c:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_block16_zero_c

  ; C = 2I on the first 16 rows: C[k,k]=2.0 at 257*k.
  MOVI r5,@mth_adapter_residual_256_block16_c
  MOVI r6,16
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_adapter_residual_256_block16_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_block16_diag

  ; h = 1.0 for all 16.
  MOVI r1,@mth_adapter_residual_256_block16_h
  MOVI r6,16
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_256_block16_fill_h:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_block16_fill_h

  ; u = 1.0 for all 256.
  MOVI r3,@mth_adapter_residual_256_block16_u
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_256_block16_fill_u:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_block16_fill_u

  ; beta=0.5, h=u=1, C=2I -> out = 1 + 0.5*2 = 2 for all 16.
  MOVI r1,@mth_adapter_residual_256_block16_h
  MOVI r2,@mth_adapter_residual_256_block16_c
  MOVI r3,@mth_adapter_residual_256_block16_u
  MOVI r4,0
  MOVHI r4,0x3F00
  MOVI r5,@mth_adapter_residual_256_block16_out
  CALL @mth_adapter_residual_256_block16
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_256_block16_fail

  ; Check all 16 outputs equal 2.0.
  MOVI r5,@mth_adapter_residual_256_block16_out
  MOVI r6,0
  MOVI r7,16
  MOVI r10,0
  MOVHI r10,0x4000
mth_adapter_residual_256_block16_check_out:
  ADD r8,r5,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_adapter_residual_256_block16_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_adapter_residual_256_block16_check_out

  ; Poison out with 77.0 to prove beta=0 rewrites every element.
  MOVI r5,@mth_adapter_residual_256_block16_out
  MOVI r6,16
  MOVI r7,0
  MOVHI r7,0x429A
mth_adapter_residual_256_block16_poison_out:
  ST r7,r5,0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_block16_poison_out

  ; beta=0 -> out = h = 1.0 for all 16.
  MOVI r1,@mth_adapter_residual_256_block16_h
  MOVI r2,@mth_adapter_residual_256_block16_c
  MOVI r3,@mth_adapter_residual_256_block16_u
  MOVI r4,0
  MOVI r5,@mth_adapter_residual_256_block16_out
  CALL @mth_adapter_residual_256_block16
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_256_block16_fail

  ; Check all 16 outputs equal h = 1.0.
  MOVI r5,@mth_adapter_residual_256_block16_out
  MOVI r6,0
  MOVI r7,16
  MOVI r10,0
  MOVHI r10,0x3F80
mth_adapter_residual_256_block16_check_zero:
  ADD r8,r5,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_adapter_residual_256_block16_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_adapter_residual_256_block16_check_zero

  MOVI r1,@mth_adapter_residual_256_block16_pass
  CALL @mth_adapter_residual_256_block16_puts
  HALT

mth_adapter_residual_256_block16_fail:
  MOVI r1,@mth_adapter_residual_256_block16_bad
  CALL @mth_adapter_residual_256_block16_puts
  HALT

mth_adapter_residual_256_block16_puts:
  MOVI r4,255
mth_adapter_residual_256_block16_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_adapter_residual_256_block16_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_adapter_residual_256_block16_puts_byte
  ADD r1,r1,r13
  JMP @mth_adapter_residual_256_block16_puts_word

.org 1024
mth_adapter_residual_256_block16_c:
.word 0
.org 5120
mth_adapter_residual_256_block16_u:
.word 0
.org 5376
mth_adapter_residual_256_block16_h:
.word 0
.org 5392
mth_adapter_residual_256_block16_out:
.word 0
.org 5408
mth_adapter_residual_256_block16_pass:
.str "MTH ADAPTER RESIDUAL 256 BLOCK16 PASS\n"
mth_adapter_residual_256_block16_bad:
.str "MTH ADAPTER RESIDUAL 256 BLOCK16 FAIL\n"
