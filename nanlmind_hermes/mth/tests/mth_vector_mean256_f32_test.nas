; MTH vector mean256 float32 fixture — NaN16 only.
;
; Phase 1 checks the exact mean of x[i] = float32(i), i=0..255: 127.5.
; Phase 2 checks the declared sequential round-to-nearest-even absorption of
; 255 copies of 2^-25 after 1.0: sum=1.0, mean=2^-8.
; Phase 3 checks signed zero: an all -0.0 vector means +0.0 because the
; accumulator starts at +0.0.
; Phase 4 checks infinity propagation.
; Phase 5 checks quiet-NaN propagation and bit preservation.

.org 0
JMP @mth_vec_mean_main
.include "../nan16/mth_vector_mean256_f32.nas"

mth_vec_mean_main:
  MOVI r13,1

  ; x[i] = (float32)i for i=0..255.
  MOVI r8,@mth_mean_x
  MOVI r9,0
  MOVI r10,256
mth_mean_fill_ints:
  CVTIF r7,r9
  ST r7,r8,0
  ADD r8,r8,r13
  ADD r9,r9,r13
  CMP r9,r10
  JMP[LT] @mth_mean_fill_ints

  MOVI r1,@mth_mean_x
  MOVI r2,@mth_mean_out
  CALL @mth_vector_mean256_f32
  CMP r1,r13
  JMP[NE] @mth_mean_fail
  MOVI r4,@mth_mean_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x42FF           ; 127.5f
  CMP r5,r6
  JMP[NE] @mth_mean_fail

  ; x = 2^-25 everywhere, then x[0] = 1.0. Each later FADD rounds back to
  ; 1.0, so sum=1.0 and mean=2^-8 (0x3B800000).
  MOVI r1,@mth_mean_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3300           ; 2^-25
mth_mean_fill_tiny:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_mean_fill_tiny

  MOVI r1,@mth_mean_x
  MOVI r7,0
  MOVHI r7,0x3F80           ; 1.0
  ST r7,r1,0

  MOVI r1,@mth_mean_x
  MOVI r2,@mth_mean_out
  CALL @mth_vector_mean256_f32
  CMP r1,r13
  JMP[NE] @mth_mean_fail
  MOVI r4,@mth_mean_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3B80           ; 2^-8
  CMP r5,r6
  JMP[NE] @mth_mean_fail

  ; All -0.0 -> +0.0 mean (accumulator starts at +0.0).
  MOVI r1,@mth_mean_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x8000
mth_mean_fill_negzero:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_mean_fill_negzero

  MOVI r1,@mth_mean_x
  MOVI r2,@mth_mean_out
  CALL @mth_vector_mean256_f32
  CMP r1,r13
  JMP[NE] @mth_mean_fail
  MOVI r4,@mth_mean_out
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_mean_fail

  ; All +inf -> +inf mean.
  MOVI r1,@mth_mean_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x7F80
mth_mean_fill_posinf:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_mean_fill_posinf

  MOVI r1,@mth_mean_x
  MOVI r2,@mth_mean_out
  CALL @mth_vector_mean256_f32
  CMP r1,r13
  JMP[NE] @mth_mean_fail
  MOVI r4,@mth_mean_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x7F80
  CMP r5,r6
  JMP[NE] @mth_mean_fail

  ; x[0] = quiet NaN 0x7FC00123, all other elements +0.0.
  MOVI r1,@mth_mean_x
  MOVI r6,256
mth_mean_fill_zero:
  ST r0,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_mean_fill_zero

  MOVI r1,@mth_mean_x
  MOVI r7,0x0123
  MOVHI r7,0x7FC0
  ST r7,r1,0

  MOVI r1,@mth_mean_x
  MOVI r2,@mth_mean_out
  CALL @mth_vector_mean256_f32
  CMP r1,r13
  JMP[NE] @mth_mean_fail
  MOVI r4,@mth_mean_out
  LD r5,r4,0
  MOVI r6,0x0123
  MOVHI r6,0x7FC0
  CMP r5,r6
  JMP[NE] @mth_mean_fail

  MOVI r1,@mth_mean_pass
  CALL @mth_mean_puts
  HALT

mth_mean_fail:
  MOVI r1,@mth_mean_bad
  CALL @mth_mean_puts
  HALT

mth_mean_puts:
  MOVI r4,255
mth_mean_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_mean_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_mean_puts_byte
  ADD r1,r1,r13
  JMP @mth_mean_puts_word

.org 1024
mth_mean_x:
.word 0
.org 2048
mth_mean_out:
.word 77
.org 2560
mth_mean_pass:
.str "MTH VECTOR MEAN256 F32 PASS\n"
mth_mean_bad:
.str "MTH VECTOR MEAN256 F32 FAIL\n"
