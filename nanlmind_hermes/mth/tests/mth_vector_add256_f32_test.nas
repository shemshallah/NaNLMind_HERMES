; MTH vector add256 float32 fixture — NaN16 only.
;
; Phase 1 checks all 256 elements of 2.0 + 3.0 = 5.0.
; Phase 2 checks the declared NaN/Inf/signed-zero bit rules on eight cases.

.org 0
JMP @mth_vec_add_main
.include "../nan16/mth_vector_add256_f32.nas"

mth_vec_add_main:
  MOVI r13,1

  ; x = 2.0 for all 256 elements.
  MOVI r1,@mth_add_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_add_fill_x:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_add_fill_x

  ; y = 3.0 for all 256 elements.
  MOVI r2,@mth_add_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4040
mth_add_fill_y:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_add_fill_y

  MOVI r1,@mth_add_x
  MOVI r2,@mth_add_y
  MOVI r3,@mth_add_out
  CALL @mth_vector_add256_f32
  CMP r1,r13
  JMP[NE] @mth_add_fail

  ; All 256 outputs must be exactly 5.0 (0x40A00000).
  MOVI r4,@mth_add_out
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x40A0
mth_add_check_all:
  LD r5,r4,0
  CMP r5,r7
  JMP[NE] @mth_add_fail
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_add_check_all

  ; Copy the eight policy cases into x[0..7] and y[0..7].
  MOVI r9,0
  MOVI r12,@mth_add_x
  MOVI r14,@mth_add_y
mth_add_copy_cases:
  MOVI r10,@mth_add_case_x
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r12,r9
  ST r11,r10,0

  MOVI r10,@mth_add_case_y
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r14,r9
  ST r11,r10,0

  ADD r9,r9,r13
  MOVI r10,8
  CMP r9,r10
  JMP[LT] @mth_add_copy_cases

  MOVI r1,@mth_add_x
  MOVI r2,@mth_add_y
  MOVI r3,@mth_add_out
  CALL @mth_vector_add256_f32
  CMP r1,r13
  JMP[NE] @mth_add_fail

  ; Compare out[0..7] against the declared bit results.
  MOVI r9,0
  MOVI r12,@mth_add_out
  MOVI r14,@mth_add_case_out
mth_add_check_cases:
  ADD r10,r14,r9
  LD r11,r10,0
  ADD r10,r12,r9
  LD r10,r10,0
  CMP r10,r11
  JMP[NE] @mth_add_fail
  ADD r9,r9,r13
  MOVI r10,8
  CMP r9,r10
  JMP[LT] @mth_add_check_cases

  MOVI r1,@mth_add_pass
  CALL @mth_add_puts
  HALT

mth_add_fail:
  MOVI r1,@mth_add_bad
  CALL @mth_add_puts
  HALT

mth_add_puts:
  MOVI r4,255
mth_add_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_add_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_add_puts_byte
  ADD r1,r1,r13
  JMP @mth_add_puts_word

.org 1024
mth_add_x:
.word 0
.org 2048
mth_add_y:
.word 0
.org 3072
mth_add_out:
.word 0
.org 4096
mth_add_case_x:
.word 0x7F800000        ; +inf + -inf -> invalid
.word 0xFF800000        ; -inf + -inf -> -inf
.word 0x00000000        ; +0 + -0 -> +0 (RNE)
.word 0x80000000        ; -0 + -0 -> -0
.word 0x7F800001        ; signalling NaN, first operand
.word 0x3F800000        ; 1.0 + negative signalling NaN
.word 0x7FC00003        ; quiet NaN A wins over quiet NaN B
.word 0x7F800000        ; +inf + 1.0 -> +inf
mth_add_case_y:
.word 0xFF800000
.word 0xFF800000
.word 0x80000000
.word 0x80000000
.word 0x3F800000
.word 0xFF800002
.word 0xFFC00004
.word 0x3F800000
mth_add_case_out:
.word 0xFFC00000
.word 0xFF800000
.word 0x00000000
.word 0x80000000
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0x7F800000
.org 4608
mth_add_pass:
.str "MTH VECTOR ADD256 F32 PASS\n"
mth_add_bad:
.str "MTH VECTOR ADD256 F32 FAIL\n"
