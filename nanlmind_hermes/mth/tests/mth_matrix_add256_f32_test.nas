; MTH 256x256 float32 matrix add fixture — NaN16 only.
;
; The two inputs are built with runtime loops, not a 65,536-line source.
; A and B are zeroed, then each row gets one diagonal and one off-diagonal
; nonzero generated from i+1 with CVTIF (and an exact sign-bit flip):
;
;   A[i,i]              = +(i+1)
;   A[i,(i+1) mod 256]  = -(i+1)
;   B[i,i]              = +(i+1)
;   B[i,(i+2) mod 256]  = +(i+1)
;
; so the expected sum is sparse:
;
;   C[i,i]              = 2*(i+1)
;   C[i,(i+1) mod 256]  = -(i+1)
;   C[i,(i+2) mod 256]  = +(i+1)
;   C[i,j]              = +0.0 elsewhere
;
; Eight NaN/Inf/signed-zero pairs overwrite A[0,0..7] and B[0,0..7] and are
; checked against a hard-coded expected bit table. The remaining 248 cells of
; row 0 are checked as +0.0, and every cell of rows 1..255 is checked against
; the sparse pattern. Together those passes compare all 65,536 output words.
; No host arithmetic computes an input value or an expected output.

.org 0
JMP @mth_matrix_add_test_main
.include "../nan16/mth_matrix_add256_f32.nas"

mth_matrix_add_test_main:
  MOVI r13,1

  ; Zero both dense input matrices.
  LA r1,@mth_matrix_add_test_a
  CALL @mth_matrix_add_test_zero64k
  LA r1,@mth_matrix_add_test_b
  CALL @mth_matrix_add_test_zero64k

  ; Build the sparse diagonal + rotation patterns.
  LA r1,@mth_matrix_add_test_a
  LA r2,@mth_matrix_add_test_b
  CALL @mth_matrix_add_test_fill_pattern

  ; Overwrite the first eight elements of each input with special cases.
  LA r1,@mth_matrix_add_test_a
  LA r2,@mth_matrix_add_test_b
  CALL @mth_matrix_add_test_copy_cases

  ; Phase 1: C = A + B.
  LA r1,@mth_matrix_add_test_a
  LA r2,@mth_matrix_add_test_b
  LA r3,@mth_matrix_add_test_c
  CALL @mth_matrix_add256_f32
  CMP r1,r13
  JMP[NE] @mth_matrix_add_test_fail

  ; Phase 2: check the eight special output words C[0..7].
  LA r1,@mth_matrix_add_test_c
  CALL @mth_matrix_add_test_check_cases
  CMP r1,r13
  JMP[NE] @mth_matrix_add_test_fail

  ; Phase 3: check the sparse pattern in every remaining output word.
  LA r1,@mth_matrix_add_test_c
  CALL @mth_matrix_add_test_check_pattern
  CMP r1,r13
  JMP[NE] @mth_matrix_add_test_fail

  LA r1,@mth_matrix_add_test_pass
  CALL @mth_matrix_add_test_puts
  HALT

mth_matrix_add_test_fail:
  LA r1,@mth_matrix_add_test_bad
  CALL @mth_matrix_add_test_puts
  HALT

; Zero 65,536 words at r1. Clobbers r2-r3; preserves r1 and r13.
mth_matrix_add_test_zero64k:
  ADD r2,r1,r0
  MOVI r3,0
  MOVHI r3,0x0001
mth_matrix_add_test_zero64k_loop:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_matrix_add_test_zero64k_loop
  RET

; Fill A at r1 and B at r2 with the sparse row pattern.
; Clobbers r2-r12; preserves r13.
mth_matrix_add_test_fill_pattern:
  ADD r4,r1,r0          ; A row base
  ADD r5,r2,r0          ; B row base
  MOVI r6,0             ; i
  MOVI r2,0
  MOVHI r2,0x8000       ; sign bit
  MOVI r3,255
  MOVI r12,256
mth_matrix_add_test_fill_pattern_row:
  ADD r7,r6,r13         ; i+1
  CVTIF r8,r7           ; +(i+1)
  BOOL.6 r9,r8,r2       ; -(i+1), exact sign flip
  ADD r10,r4,r6         ; &A[i,i]
  ST r8,r10,0
  BOOL.8 r7,r7,r3       ; (i+1) mod 256
  ADD r10,r4,r7         ; &A[i,(i+1) mod 256]
  ST r9,r10,0
  ADD r10,r5,r6         ; &B[i,i]
  ST r8,r10,0
  ADD r11,r6,r13
  ADD r11,r11,r13       ; i+2
  BOOL.8 r11,r11,r3     ; (i+2) mod 256
  ADD r10,r5,r11        ; &B[i,(i+2) mod 256]
  ST r8,r10,0
  ADD r4,r4,r12         ; next A row
  ADD r5,r5,r12         ; next B row
  ADD r6,r6,r13
  CMP r6,r12
  JMP[LT] @mth_matrix_add_test_fill_pattern_row
  RET

; Copy the eight special pairs into A[0..7] and B[0..7].
; Clobbers r9-r11; preserves r1-r8 and r13.
mth_matrix_add_test_copy_cases:
  MOVI r9,0
mth_matrix_add_test_copy_cases_loop:
  LA r10,@mth_matrix_add_test_case_x
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r1,r9
  ST r11,r10,0

  LA r10,@mth_matrix_add_test_case_y
  ADD r10,r10,r9
  LD r11,r10,0
  ADD r10,r2,r9
  ST r11,r10,0

  ADD r9,r9,r13
  MOVI r10,8
  CMP r9,r10
  JMP[LT] @mth_matrix_add_test_copy_cases_loop
  RET

; Check C[0..7] at r1 against the declared special-case results.
; Returns r1=1 on exact match, 0 otherwise. Clobbers r4-r7.
mth_matrix_add_test_check_cases:
  MOVI r4,0
  ADD r5,r1,r0
mth_matrix_add_test_check_cases_loop:
  ADD r6,r5,r4
  LD r6,r6,0
  LA r7,@mth_matrix_add_test_case_out
  ADD r7,r7,r4
  LD r7,r7,0
  CMP r6,r7
  JMP[NE] @mth_matrix_add_test_check_cases_bad
  ADD r4,r4,r13
  MOVI r7,8
  CMP r4,r7
  JMP[LT] @mth_matrix_add_test_check_cases_loop
  MOVI r1,1
  RET
mth_matrix_add_test_check_cases_bad:
  MOVI r1,0
  RET

; Check the sparse sum pattern in C at r1. Row 0 columns 0..7 are skipped
; because check_cases already compared them; every other word of all 256 rows
; is compared here. Returns r1=1 on exact match, 0 otherwise.
; Clobbers r2-r14; preserves r13.
mth_matrix_add_test_check_pattern:
  ADD r4,r1,r0          ; current row base
  MOVI r5,0             ; r
  MOVI r12,256
mth_matrix_add_test_check_pattern_row:
  MOVI r3,255
  ADD r6,r5,r13         ; r+1
  CVTIF r7,r6           ; +(r+1)
  MOVI r2,0
  MOVHI r2,0x8000       ; sign bit
  BOOL.6 r8,r7,r2       ; -(r+1)
  ADD r10,r6,r6
  CVTIF r10,r10         ; 2*(r+1)
  BOOL.8 r6,r6,r3       ; off1 = (r+1) mod 256
  ADD r11,r6,r13
  BOOL.8 r11,r11,r3     ; off2 = (r+2) mod 256
  ADD r9,r4,r0          ; word cursor
  MOVI r14,0            ; c
mth_matrix_add_test_check_pattern_col:
  CMP r5,r0
  JMP[NE] @mth_matrix_add_test_check_pattern_eval
  MOVI r2,8
  CMP r14,r2
  JMP[GE] @mth_matrix_add_test_check_pattern_eval
  JMP @mth_matrix_add_test_check_pattern_next
mth_matrix_add_test_check_pattern_eval:
  MOVI r2,0             ; default +0.0
  CMP r14,r5
  JMP[EQ] @mth_matrix_add_test_check_pattern_diag
  CMP r14,r6
  JMP[EQ] @mth_matrix_add_test_check_pattern_off1
  CMP r14,r11
  JMP[EQ] @mth_matrix_add_test_check_pattern_off2
  JMP @mth_matrix_add_test_check_pattern_cmp
mth_matrix_add_test_check_pattern_diag:
  ADD r2,r10,r0
  JMP @mth_matrix_add_test_check_pattern_cmp
mth_matrix_add_test_check_pattern_off1:
  ADD r2,r8,r0
  JMP @mth_matrix_add_test_check_pattern_cmp
mth_matrix_add_test_check_pattern_off2:
  ADD r2,r7,r0
mth_matrix_add_test_check_pattern_cmp:
  LD r3,r9,0
  CMP r3,r2
  JMP[NE] @mth_matrix_add_test_check_pattern_bad
mth_matrix_add_test_check_pattern_next:
  ADD r9,r9,r13
  ADD r14,r14,r13
  CMP r14,r12
  JMP[LT] @mth_matrix_add_test_check_pattern_col
  ADD r4,r4,r12
  ADD r5,r5,r13
  CMP r5,r12
  JMP[LT] @mth_matrix_add_test_check_pattern_row
  MOVI r1,1
  RET
mth_matrix_add_test_check_pattern_bad:
  MOVI r1,0
  RET

mth_matrix_add_test_puts:
  MOVI r4,255
mth_matrix_add_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_matrix_add_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_matrix_add_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_matrix_add_test_puts_word

.org 512
mth_matrix_add_test_a:
.word 0
.org 66048
mth_matrix_add_test_b:
.word 0
.org 131584
mth_matrix_add_test_c:
.word 0
.org 197120
mth_matrix_add_test_pass:
.str "MTH MATRIX ADD256 F32 PASS\n"
.org 197152
mth_matrix_add_test_bad:
.str "MTH MATRIX ADD256 F32 FAIL\n"
.org 197184
mth_matrix_add_test_case_x:
.word 0x7F800000        ; +inf + -inf -> invalid
.word 0xFF800000        ; -inf + -inf -> -inf
.word 0x00000000        ; +0 + -0 -> +0 (RNE)
.word 0x80000000        ; -0 + -0 -> -0
.word 0x7F800001        ; signalling NaN, first operand
.word 0x3F800000        ; 1.0 + negative signalling NaN
.word 0x7FC00003        ; quiet NaN A wins over quiet NaN B
.word 0x7F800000        ; +inf + 1.0 -> +inf
mth_matrix_add_test_case_y:
.word 0xFF800000
.word 0xFF800000
.word 0x80000000
.word 0x80000000
.word 0x3F800000
.word 0xFF800002
.word 0xFFC00004
.word 0x3F800000
mth_matrix_add_test_case_out:
.word 0xFFC00000
.word 0xFF800000
.word 0x00000000
.word 0x80000000
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0x7F800000
