.org 0
JMP @mth_adapter_residual_test_main
.include "../nan16/mth_adapter_residual_128.nas"

mth_adapter_residual_test_main:
  MOVI r13,1

  ; C = 0 over all 16,384 words.
  MOVI r4,@mth_adapter_residual_c
  MOVI r6,16384
mth_adapter_residual_zero_c:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_zero_c

  ; C = 2I: diagonal entries at stride 129.
  MOVI r5,@mth_adapter_residual_c
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x4000
mth_adapter_residual_diag:
  ST r7,r5,0
  MOVI r8,129
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_diag

  ; h = 1.0 for all 128.
  MOVI r1,@mth_adapter_residual_h
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_fill_h:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_fill_h

  ; u = 1.0 for all 128.
  MOVI r3,@mth_adapter_residual_u
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_fill_u:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_fill_u

  ; beta = 0.5, h = u = 1, C = 2I -> scratch = 2, out = 2.
  MOVI r1,@mth_adapter_residual_h
  MOVI r2,@mth_adapter_residual_c
  MOVI r3,@mth_adapter_residual_u
  MOVI r4,0
  MOVHI r4,0x3F00
  MOVI r5,@mth_adapter_residual_out
  MOVI r6,@mth_adapter_residual_scratch
  CALL @mth_adapter_residual_128
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_fail

  ; scratch = C u: check a sample (0, 64) and the endpoint (127).
  MOVI r7,@mth_adapter_residual_scratch
  LD r8,r7,0
  MOVI r9,0
  MOVHI r9,0x4000
  CMP r8,r9
  JMP[NE] @mth_adapter_residual_fail
  MOVI r8,64
  ADD r7,r7,r8
  LD r8,r7,0
  CMP r8,r9
  JMP[NE] @mth_adapter_residual_fail
  MOVI r7,@mth_adapter_residual_scratch
  MOVI r8,127
  ADD r7,r7,r8
  LD r8,r7,0
  CMP r8,r9
  JMP[NE] @mth_adapter_residual_fail

  ; out = 2.0 for all 128.
  MOVI r5,@mth_adapter_residual_out
  MOVI r6,0
  MOVI r7,128
  MOVI r10,0
  MOVHI r10,0x4000
mth_adapter_residual_check_out:
  ADD r8,r5,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_adapter_residual_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_adapter_residual_check_out

  ; Poison out with 77.0 to prove beta=0 rewrites every element.
  MOVI r5,@mth_adapter_residual_out
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x429A
mth_adapter_residual_poison_out:
  ST r7,r5,0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_poison_out

  ; beta = 0 -> out = h = 1.0 for all 128.
  MOVI r1,@mth_adapter_residual_h
  MOVI r2,@mth_adapter_residual_c
  MOVI r3,@mth_adapter_residual_u
  MOVI r4,0
  MOVI r5,@mth_adapter_residual_out
  MOVI r6,@mth_adapter_residual_scratch
  CALL @mth_adapter_residual_128
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_fail

  MOVI r5,@mth_adapter_residual_out
  MOVI r6,0
  MOVI r7,128
  MOVI r10,0
  MOVHI r10,0x3F80
mth_adapter_residual_check_zero:
  ADD r8,r5,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_adapter_residual_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_adapter_residual_check_zero

  MOVI r1,@mth_adapter_residual_pass
  CALL @mth_adapter_residual_puts
  HALT

mth_adapter_residual_fail:
  MOVI r1,@mth_adapter_residual_bad
  CALL @mth_adapter_residual_puts
  HALT

mth_adapter_residual_puts:
  MOVI r4,255
mth_adapter_residual_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_adapter_residual_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_adapter_residual_puts_byte
  ADD r1,r1,r13
  JMP @mth_adapter_residual_puts_word

.org 1024
mth_adapter_residual_c:
.word 0
.org 17408
mth_adapter_residual_h:
.word 0
.org 17536
mth_adapter_residual_u:
.word 0
.org 17664
mth_adapter_residual_out:
.word 0
.org 17792
mth_adapter_residual_scratch:
.word 0
.org 18048
mth_adapter_residual_pass:
.str "MTH ADAPTER RESIDUAL 128 PASS\n"
mth_adapter_residual_bad:
.str "MTH ADAPTER RESIDUAL 128 FAIL\n"
