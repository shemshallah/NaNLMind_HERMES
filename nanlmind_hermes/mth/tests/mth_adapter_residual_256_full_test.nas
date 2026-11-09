.org 0
JMP @mth_adapter_residual_256_full_test_main
.include "../nan16/mth_adapter_residual_256_full.nas"

mth_adapter_residual_256_full_test_main:
  MOVI r13,1

  ; Zero W (65,536 words). MOVI r6,0 + MOVHI r6,0x0001 builds 65,536.
  LA r4,@mth_adapter_residual_256_full_w
  MOVI r6,0
  MOVHI r6,0x0001
mth_adapter_residual_256_full_zero_w:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_zero_w

  ; W = 2I: W[i,i] = 2.0 for i=0..255; row stride 256 plus 1 => 257.
  LA r5,@mth_adapter_residual_256_full_w
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_adapter_residual_256_full_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_diag

  ; h = 1.0 for all 256.
  LA r1,@mth_adapter_residual_256_full_h
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_256_full_fill_h:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_fill_h

  ; u = 1.0 for all 256.
  LA r3,@mth_adapter_residual_256_full_u
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_adapter_residual_256_full_fill_u:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_fill_u

  ; beta=0.5, h=u=1, W=2I -> y = 1 + 0.5*2 = 2 for all 256.
  LA r1,@mth_adapter_residual_256_full_w
  LA r2,@mth_adapter_residual_256_full_h
  LA r3,@mth_adapter_residual_256_full_u
  MOVI r4,0
  MOVHI r4,0x3F00
  LA r5,@mth_adapter_residual_256_full_y
  CALL @mth_adapter_residual_256_full
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_256_full_fail

  ; Check all 256 outputs equal 2.0.
  LA r4,@mth_adapter_residual_256_full_y
  MOVI r6,0
  MOVHI r6,0x4000
  MOVI r7,0
mth_adapter_residual_256_full_check_half:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_adapter_residual_256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_adapter_residual_256_full_check_half

  ; Poison y with 77.0 to prove beta=0 rewrites every element.
  LA r5,@mth_adapter_residual_256_full_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x429A
mth_adapter_residual_256_full_poison_y:
  ST r7,r5,0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_poison_y

  ; beta=0 -> y = h = 1.0 for all 256.
  LA r1,@mth_adapter_residual_256_full_w
  LA r2,@mth_adapter_residual_256_full_h
  LA r3,@mth_adapter_residual_256_full_u
  MOVI r4,0
  LA r5,@mth_adapter_residual_256_full_y
  CALL @mth_adapter_residual_256_full
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_256_full_fail

  ; Check all 256 outputs equal h = 1.0.
  LA r4,@mth_adapter_residual_256_full_y
  MOVI r6,0
  MOVHI r6,0x3F80
  MOVI r7,0
mth_adapter_residual_256_full_check_zero_beta:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_adapter_residual_256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_adapter_residual_256_full_check_zero_beta

  ; W = 0 over all 65,536 words; keep h=1, u=1.
  LA r4,@mth_adapter_residual_256_full_w
  MOVI r6,0
  MOVHI r6,0x0001
mth_adapter_residual_256_full_zero_w_again:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_zero_w_again

  ; Poison y again so the W=0 call must rewrite every element.
  LA r5,@mth_adapter_residual_256_full_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x429A
mth_adapter_residual_256_full_poison_y_again:
  ST r7,r5,0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_adapter_residual_256_full_poison_y_again

  ; beta=0.5 with W=0 -> y = h = 1.0 for all 256.
  LA r1,@mth_adapter_residual_256_full_w
  LA r2,@mth_adapter_residual_256_full_h
  LA r3,@mth_adapter_residual_256_full_u
  MOVI r4,0
  MOVHI r4,0x3F00
  LA r5,@mth_adapter_residual_256_full_y
  CALL @mth_adapter_residual_256_full
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_256_full_fail

  ; Check all 256 outputs equal h = 1.0.
  LA r4,@mth_adapter_residual_256_full_y
  MOVI r6,0
  MOVHI r6,0x3F80
  MOVI r7,0
mth_adapter_residual_256_full_check_zero_w:
  ADD r8,r4,r7
  LD r9,r8,0
  CMP r9,r6
  JMP[NE] @mth_adapter_residual_256_full_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_adapter_residual_256_full_check_zero_w

  LA r1,@mth_adapter_residual_256_full_pass
  CALL @mth_adapter_residual_256_full_puts
  HALT

mth_adapter_residual_256_full_fail:
  LA r1,@mth_adapter_residual_256_full_bad
  CALL @mth_adapter_residual_256_full_puts
  HALT

mth_adapter_residual_256_full_puts:
  MOVI r4,255
mth_adapter_residual_256_full_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_adapter_residual_256_full_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_adapter_residual_256_full_puts_byte
  ADD r1,r1,r13
  JMP @mth_adapter_residual_256_full_puts_word

.org 1024
mth_adapter_residual_256_full_w:
.word 0
.org 66560
mth_adapter_residual_256_full_h:
.word 0
.org 66816
mth_adapter_residual_256_full_u:
.word 0
.org 67072
mth_adapter_residual_256_full_y:
.word 0
.org 67328
mth_adapter_residual_256_full_pass:
.str "MTH ADAPTER RESIDUAL 256 FULL PASS\n"
mth_adapter_residual_256_full_bad:
.str "MTH ADAPTER RESIDUAL 256 FULL FAIL\n"
