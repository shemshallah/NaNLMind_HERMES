; MTH vector sub256 float32 fixture — NaN16 only.
;
; Phase 1 checks all 256 elements of 5.0 - 1.5 = 3.5.
; Phase 2 checks the declared NaN/Inf/signed-zero bit rules on ten cases.

.org 0
JMP @mth_vec_sub_main
.include "../nan16/mth_vector_sub256_f32.nas"

mth_vec_sub_main:
  MOVI r13,1

  ; x = 5.0 for all 256 elements.
  MOVI r1,@mth_sub_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x40A0
mth_sub_fill_x:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_sub_fill_x

  ; y = 1.5 for all 256 elements.
  MOVI r2,@mth_sub_y
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3FC0
mth_sub_fill_y:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_sub_fill_y

  MOVI r1,@mth_sub_x
  MOVI r2,@mth_sub_y
  MOVI r3,@mth_sub_out
  CALL @mth_vector_sub256_f32
  CMP r1,r13
  JMP[NE] @mth_sub_fail

  ; All 256 outputs must be exactly 3.5 (0x40600000).
  MOVI r4,@mth_sub_out
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4060
mth_sub_check_all:
  LD r5,r4,0
  CMP r5,r7
  JMP[NE] @mth_sub_fail
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_sub_check_all

  ; Copy the ten policy cases into x[0..9] and y[0..9].
  MOVI r9,0
  MOVI r12,@mth_sub_x
  MOVI r14,@mth_sub_y
mth_sub_copy_cases:
  MOVI r10,@mth_sub_case_x
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r12,r9
  ST r11,r10,0

  MOVI r10,@mth_sub_case_y
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r14,r9
  ST r11,r10,0

  ADD r9,r9,r13
  MOVI r10,10
  CMP r9,r10
  JMP[LT] @mth_sub_copy_cases

  MOVI r1,@mth_sub_x
  MOVI r2,@mth_sub_y
  MOVI r3,@mth_sub_out
  CALL @mth_vector_sub256_f32
  CMP r1,r13
  JMP[NE] @mth_sub_fail

  ; Compare out[0..9] against the declared bit results.
  MOVI r9,0
  MOVI r12,@mth_sub_out
  MOVI r14,@mth_sub_case_out
mth_sub_check_cases:
  ADD r10,r14,r9
  LD r11,r10,0
  ADD r10,r12,r9
  LD r10,r10,0
  CMP r10,r11
  JMP[NE] @mth_sub_fail
  ADD r9,r9,r13
  MOVI r10,10
  CMP r9,r10
  JMP[LT] @mth_sub_check_cases

  MOVI r1,@mth_sub_pass
  CALL @mth_sub_puts
  HALT

mth_sub_fail:
  MOVI r1,@mth_sub_bad
  CALL @mth_sub_puts
  HALT

mth_sub_puts:
  MOVI r4,255
mth_sub_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_sub_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_sub_puts_byte
  ADD r1,r1,r13
  JMP @mth_sub_puts_word

.org 1024
mth_sub_x:
.word 0
.org 2048
mth_sub_y:
.word 0
.org 3072
mth_sub_out:
.word 0
.org 4096
mth_sub_case_x:
.word 0x7F800000        ; +inf - +inf -> invalid
.word 0x7F800000        ; +inf - 1.0 -> +inf
.word 0x00000000        ; +0 - +0 -> +0
.word 0x00000000        ; +0 - -0 -> +0
.word 0x80000000        ; -0 - +0 -> -0
.word 0x80000000        ; -0 - -0 -> +0 (RNE)
.word 0x7F800001        ; signalling NaN, first operand
.word 0x40400000        ; 3.0 - negative signalling NaN
.word 0x7FC00003        ; quiet NaN A wins over quiet NaN B
.word 0xFF800000        ; -inf - 1.0 -> -inf
mth_sub_case_y:
.word 0x7F800000
.word 0x3F800000
.word 0x00000000
.word 0x80000000
.word 0x00000000
.word 0x80000000
.word 0x40000000
.word 0xFF800002
.word 0xFFC00004
.word 0x3F800000
mth_sub_case_out:
.word 0xFFC00000
.word 0x7F800000
.word 0x00000000
.word 0x00000000
.word 0x80000000
.word 0x00000000
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFF800000
.org 4608
mth_sub_pass:
.str "MTH VECTOR SUB256 F32 PASS\n"
mth_sub_bad:
.str "MTH VECTOR SUB256 F32 FAIL\n"
