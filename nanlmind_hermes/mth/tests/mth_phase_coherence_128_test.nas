.org 0
JMP @mth_phase_test_main
.include "../nan16/mth_phase_coherence_128.nas"

mth_phase_test_main:
  MOVI r13,1

  ; Case 1: all z = 1 + 0i -> sum z = 128, mag_sum = 128 -> R = 1 exactly.
  MOVI r4,@mth_phase_re
  MOVI r5,@mth_phase_im
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_phase_fill_ones:
  ST r7,r4,0
  ST r0,r5,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_phase_fill_ones

  MOVI r1,@mth_phase_re
  MOVI r2,@mth_phase_im
  MOVI r3,@mth_phase_out
  CALL @mth_phase_coherence_128
  CMP r1,r13
  JMP[NE] @mth_phase_fail
  MOVI r4,@mth_phase_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_phase_fail

  ; Case 2: 64 z = 1 + 0i, then 64 z = -1 + 0i.
  ; Vector sum is 0, mag_sum is 128 -> R = 0 exactly.
  MOVI r4,@mth_phase_re
  MOVI r6,64
  MOVI r7,0
  MOVHI r7,0x3F80
mth_phase_fill_pos:
  ST r7,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_phase_fill_pos

  MOVI r6,64
  MOVI r7,0
  MOVHI r7,0xBF80
mth_phase_fill_neg:
  ST r7,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_phase_fill_neg

  MOVI r1,@mth_phase_re
  MOVI r2,@mth_phase_im
  MOVI r3,@mth_phase_out
  CALL @mth_phase_coherence_128
  CMP r1,r13
  JMP[NE] @mth_phase_fail
  MOVI r4,@mth_phase_out
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_phase_fail

  ; Case 2b: all z = 0 + 1i -> sum z = 128i, mag_sum = 128 -> R = 1 exactly.
  ; This exercises the imaginary accumulation and the im^2 term.
  MOVI r4,@mth_phase_re
  MOVI r5,@mth_phase_im
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_phase_fill_imag:
  ST r0,r4,0
  ST r7,r5,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_phase_fill_imag

  MOVI r1,@mth_phase_re
  MOVI r2,@mth_phase_im
  MOVI r3,@mth_phase_out
  CALL @mth_phase_coherence_128
  CMP r1,r13
  JMP[NE] @mth_phase_fail
  MOVI r4,@mth_phase_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_phase_fail

  ; Case 3: all z = 0 + 0i -> mag_sum = 0, zero policy -> R = 0 exactly.
  MOVI r4,@mth_phase_re
  MOVI r5,@mth_phase_im
  MOVI r6,128
mth_phase_fill_zero:
  ST r0,r4,0
  ST r0,r5,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_phase_fill_zero

  MOVI r1,@mth_phase_re
  MOVI r2,@mth_phase_im
  MOVI r3,@mth_phase_out
  CALL @mth_phase_coherence_128
  CMP r1,r13
  JMP[NE] @mth_phase_fail
  MOVI r4,@mth_phase_out
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_phase_fail

  MOVI r1,@mth_phase_pass
  CALL @mth_phase_puts
  HALT

mth_phase_fail:
  MOVI r1,@mth_phase_bad
  CALL @mth_phase_puts
  HALT

mth_phase_puts:
  MOVI r4,255
mth_phase_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_phase_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_phase_puts_byte
  ADD r1,r1,r13
  JMP @mth_phase_puts_word

.org 1024
mth_phase_re:
.word 0
.org 2048
mth_phase_im:
.word 0
.org 3072
mth_phase_out:
.word 0
.org 3136
mth_phase_pass:
.str "MTH PHASE COHERENCE 128 PASS\n"
mth_phase_bad:
.str "MTH PHASE COHERENCE 128 FAIL\n"
