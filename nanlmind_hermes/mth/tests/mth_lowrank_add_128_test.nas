.org 0
JMP @mth_lowrank_test_main
.include "../nan16/mth_lowrank_add_128.nas"

; Fixture for mth_lowrank_add_128_f32.
;
; Phase 1 (the canonical constant-field case):
;   zero U, b=c=1, alpha=2 -> sample positions all 2.0;
;   alpha=-1 -> every U=1.0 (one additive negative update);
;   alpha=-1 again -> every U=0.0 (the update cancels the +2.0 term).
;   The two -1 calls are required because the operation is additive:
;   2.0 + (-1.0) = 1.0 and 1.0 + (-1.0) = 0.0. The final state requested
;   by the fixture contract (alpha=-1 leaves every U=0) is the state after
;   the complete cancellation, checked over all 16,384 elements.
;
; Phase 2 (index mapping):
;   zero U, b[i]=i+1, c[j]=j+1, alpha=1 -> U[i,j]=(i+1)*(j+1).
;   This verifies that i selects the row (b) and j the column (c), and
;   exercises a nontrivial outer product. All chosen products are exactly
;   representable float32 values.

mth_lowrank_test_main:
  MOVI r13,1

  ; ---- Phase 1: b=c=1.0 ----
  MOVI r1,@mth_lowrank_u
  CALL @mth_lowrank_zero_u
  MOVI r1,@mth_lowrank_b
  CALL @mth_lowrank_fill_ones
  MOVI r1,@mth_lowrank_c
  CALL @mth_lowrank_fill_ones

  ; alpha = +2.0.
  MOVI r4,@mth_lowrank_alpha
  MOVI r6,0
  MOVHI r6,0x4000
  ST r6,r4,0

  MOVI r1,@mth_lowrank_u
  MOVI r2,@mth_lowrank_b
  MOVI r3,@mth_lowrank_c
  MOVI r4,@mth_lowrank_alpha
  CALL @mth_lowrank_add_128_f32
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; Sample check: U[0,0], U[0,127], U[63,64], U[127,0], U[127,127] = 2.0.
  MOVI r3,0
  MOVHI r3,0x4000
  MOVI r1,@mth_lowrank_u
  MOVI r2,0
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail
  MOVI r1,@mth_lowrank_u
  MOVI r2,127
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail
  MOVI r1,@mth_lowrank_u
  MOVI r2,8128
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail
  MOVI r1,@mth_lowrank_u
  MOVI r2,16256
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail
  MOVI r1,@mth_lowrank_u
  MOVI r2,16383
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; alpha = -1.0; one additive update leaves every U=1.0.
  MOVI r4,@mth_lowrank_alpha
  MOVI r6,0
  MOVHI r6,0xBF80
  ST r6,r4,0

  MOVI r1,@mth_lowrank_u
  MOVI r2,@mth_lowrank_b
  MOVI r3,@mth_lowrank_c
  MOVI r4,@mth_lowrank_alpha
  CALL @mth_lowrank_add_128_f32
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  MOVI r1,@mth_lowrank_u
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,16384
  CALL @mth_lowrank_check_all
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; alpha = -1.0 again; every U=0.0.
  MOVI r1,@mth_lowrank_u
  MOVI r2,@mth_lowrank_b
  MOVI r3,@mth_lowrank_c
  MOVI r4,@mth_lowrank_alpha
  CALL @mth_lowrank_add_128_f32
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  MOVI r1,@mth_lowrank_u
  MOVI r2,0
  MOVI r3,16384
  CALL @mth_lowrank_check_all
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; ---- Phase 2: b[i]=i+1, c[j]=j+1, alpha=1.0 ----
  MOVI r1,@mth_lowrank_u
  CALL @mth_lowrank_zero_u
  MOVI r1,@mth_lowrank_b
  CALL @mth_lowrank_fill_ramp
  MOVI r1,@mth_lowrank_c
  CALL @mth_lowrank_fill_ramp

  MOVI r4,@mth_lowrank_alpha
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r4,0

  MOVI r1,@mth_lowrank_u
  MOVI r2,@mth_lowrank_b
  MOVI r3,@mth_lowrank_c
  MOVI r4,@mth_lowrank_alpha
  CALL @mth_lowrank_add_128_f32
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[0,0] = 1.
  MOVI r3,0
  MOVHI r3,0x3F80
  MOVI r1,@mth_lowrank_u
  MOVI r2,0
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[0,127] = 128.
  MOVI r3,0
  MOVHI r3,0x4300
  MOVI r1,@mth_lowrank_u
  MOVI r2,127
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[127,0] = 128.
  MOVI r1,@mth_lowrank_u
  MOVI r2,16256
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[10,20] = 11*21 = 231.
  MOVI r3,0
  MOVHI r3,0x4367
  MOVI r1,@mth_lowrank_u
  MOVI r2,1300
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[63,64] = 64*65 = 4160.
  MOVI r3,0
  MOVHI r3,0x4582
  MOVI r1,@mth_lowrank_u
  MOVI r2,8128
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  ; U[127,127] = 128*128 = 16384.
  MOVI r3,0
  MOVHI r3,0x4680
  MOVI r1,@mth_lowrank_u
  MOVI r2,16383
  CALL @mth_lowrank_check_word
  CMP r1,r13
  JMP[NE] @mth_lowrank_fail

  MOVI r1,@mth_lowrank_pass
  CALL @mth_lowrank_puts
  HALT

mth_lowrank_fail:
  MOVI r1,@mth_lowrank_bad
  CALL @mth_lowrank_puts
  HALT

; Zero 16,384 words at r1. Clobbers r2-r4; preserves r1, r13.
mth_lowrank_zero_u:
  ADD r2,r1,r0
  MOVI r3,16384
mth_lowrank_zero_u_loop:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lowrank_zero_u_loop
  RET

; Fill 128 words at r1 with 1.0. Clobbers r2-r4; preserves r1, r13.
mth_lowrank_fill_ones:
  ADD r2,r1,r0
  MOVI r3,128
  MOVI r4,0
  MOVHI r4,0x3F80
mth_lowrank_fill_ones_loop:
  ST r4,r2,0
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lowrank_fill_ones_loop
  RET

; Fill 128 words at r1 with float(k+1), k=0..127. Clobbers r2-r6; preserves r13.
mth_lowrank_fill_ramp:
  MOVI r2,0
  MOVI r4,0
  MOVHI r4,0x3F80
mth_lowrank_fill_ramp_loop:
  CVTIF r5,r2
  FADD r5,r5,r4
  ADD r6,r1,r2
  ST r5,r6,0
  ADD r2,r2,r13
  MOVI r6,128
  CMP r2,r6
  JMP[LT] @mth_lowrank_fill_ramp_loop
  RET

; r1=base, r2=word offset, r3=expected payload.
; Returns r1=1 on match, 0 on mismatch. Clobbers r4.
mth_lowrank_check_word:
  ADD r4,r1,r2
  LD r4,r4,0
  CMP r4,r3
  JMP[NE] @mth_lowrank_check_word_bad
  MOVI r1,1
  RET
mth_lowrank_check_word_bad:
  MOVI r1,0
  RET

; r1=base, r2=expected payload, r3=count.
; Returns r1=1 if every word matches, 0 otherwise. Clobbers r4-r5.
mth_lowrank_check_all:
  ADD r4,r1,r0
mth_lowrank_check_all_loop:
  LD r5,r4,0
  CMP r5,r2
  JMP[NE] @mth_lowrank_check_all_bad
  ADD r4,r4,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lowrank_check_all_loop
  MOVI r1,1
  RET
mth_lowrank_check_all_bad:
  MOVI r1,0
  RET

mth_lowrank_puts:
  MOVI r4,255
mth_lowrank_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_lowrank_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_lowrank_puts_byte
  ADD r1,r1,r13
  JMP @mth_lowrank_puts_word

.org 512
mth_lowrank_u:
.word 0
.org 16896
mth_lowrank_b:
.word 0
.org 17024
mth_lowrank_c:
.word 0
.org 17152
mth_lowrank_alpha:
.word 0x40000000
.org 17184
mth_lowrank_pass:
.str "MTH LOWRANK ADD 128 PASS\n"
.org 17248
mth_lowrank_bad:
.str "MTH LOWRANK ADD 128 FAIL\n"
