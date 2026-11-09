; MTH 256-element diagonal extraction fixture — NaN16 only.
;
; The input is built with 256 runtime diagonal writes plus 11 runtime
; off-diagonal sentinel writes, not a 65,536-line source.
;
;   A[i,i] = +(i+1)                  for i = 0..255
;   A[r,c] = -(r+1)                  at 11 selected off-diagonal positions
;   A[r,c] = +0.0                    elsewhere (assembler padding)
;
; The 11 off-diagonal values are negative, while every expected diagonal value
; is positive. The table below is used for both the source pre-check and the
; input fill, so the fixture proves those off-diagonals are present before the
; kernel runs.
;
; Phase 1 pre-fills the 256-word output with 0xDEAD0000, then checks that all
; 256 source sentinels were actually written as negative values.
; Phase 2 calls the kernel and checks all 256 output words:
;   D[r] = +(r+1)                    exact diagonal mapping
; Phase 3 scans all 256 output words and requires the sign bit to be clear,
; proving that none of the negative off-diagonal sentinels was selected.
;
; All values and comparisons are produced in the guest with CVTIF and exact
; sign-bit flips (BOOL.6); no host math is used.

.org 0
JMP @mth_diag_test_main
.include "../nan16/mth_matrix_diagonal256_f32.nas"

mth_diag_test_main:
  MOVI r13,1

  ; Pre-fill D with a distinctive sentinel so an unwritten output word fails.
  LA r1,@mth_diag_test_d
  CALL @mth_diag_test_fill_output

  ; Build the diagonal A[i,i] = +(i+1).
  MOVI r1,@mth_diag_test_a
  CALL @mth_diag_test_fill_diagonal

  ; Build the selected off-diagonal negative sentinels.
  LA r1,@mth_diag_test_offdiag_table
  CALL @mth_diag_test_fill_offdiag

  ; Prove the off-diagonal sentinels are present before the kernel runs.
  LA r1,@mth_diag_test_offdiag_table
  CALL @mth_diag_test_check_offdiag
  CMP r1,r13
  JMP[NE] @mth_diag_test_fail

  ; D = diagonal(A).
  MOVI r1,@mth_diag_test_a
  LA r2,@mth_diag_test_d
  CALL @mth_matrix_diagonal256_f32
  CMP r1,r13
  JMP[NE] @mth_diag_test_fail

  ; Check all 256 expected diagonal values exactly.
  LA r1,@mth_diag_test_d
  CALL @mth_diag_test_check_output
  CMP r1,r13
  JMP[NE] @mth_diag_test_fail

  ; No selected off-diagonal is negative, so no output may have its sign bit.
  LA r1,@mth_diag_test_d
  CALL @mth_diag_test_check_positive
  CMP r1,r13
  JMP[NE] @mth_diag_test_fail

  LA r1,@mth_diag_test_pass
  CALL @mth_diag_test_puts
  HALT

mth_diag_test_fail:
  LA r1,@mth_diag_test_bad
  CALL @mth_diag_test_puts
  HALT

; Fill 256 words at r1 with 0xDEAD0000. Clobbers r3-r6; preserves r1 and r13.
mth_diag_test_fill_output:
  ADD r4,r1,r0
  MOVI r3,0
  MOVHI r3,0xDEAD
  MOVI r5,0
  MOVI r6,256
mth_diag_test_fill_output_loop:
  ST r3,r4,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  CMP r5,r6
  JMP[LT] @mth_diag_test_fill_output_loop
  RET

; Fill A at r1 with A[i,i] = +(i+1). Clobbers r4-r8 and r12; preserves r13.
mth_diag_test_fill_diagonal:
  ADD r4,r1,r0          ; input diagonal cursor
  MOVI r5,0             ; i
  MOVI r6,256           ; extent
  MOVI r12,257          ; next diagonal stride
mth_diag_test_fill_diagonal_loop:
  ADD r7,r5,r13         ; i+1
  CVTIF r8,r7           ; +(i+1)
  ST r8,r4,0            ; A[i,i]
  ADD r4,r4,r12
  ADD r5,r5,r13
  CMP r5,r6
  JMP[LT] @mth_diag_test_fill_diagonal_loop
  RET

; Fill selected off-diagonals from a table of {offset, value} pairs.
; r1 = table pointer; a zero offset terminates the table.
; A[offset] = -float32(value). Clobbers r4-r11; preserves r13.
mth_diag_test_fill_offdiag:
  ADD r4,r1,r0          ; table cursor
  MOVI r5,@mth_diag_test_a
mth_diag_test_fill_offdiag_next:
  LD r6,r4,0            ; word offset
  CMP r6,r0
  RET[EQ]
  ADD r7,r4,r13
  LD r8,r7,0            ; positive value
  ADD r9,r5,r6          ; &A[offset]
  CVTIF r10,r8          ; +value
  MOVI r11,0
  MOVHI r11,0x8000      ; sign bit
  BOOL.6 r10,r10,r11    ; -value, exact sign flip
  ST r10,r9,0
  ADD r4,r4,r13
  ADD r4,r4,r13         ; next pair
  JMP @mth_diag_test_fill_offdiag_next

; Check that every table entry was written as its negative sentinel.
; r1 = table pointer. Returns r1=1 on exact match, 0 otherwise.
; Clobbers r4-r12; preserves r13.
mth_diag_test_check_offdiag:
  ADD r4,r1,r0          ; table cursor
  MOVI r5,@mth_diag_test_a
  MOVI r12,0
  MOVHI r12,0x8000      ; sign bit
mth_diag_test_check_offdiag_next:
  LD r6,r4,0            ; word offset
  CMP r6,r0
  JMP[EQ] @mth_diag_test_check_offdiag_ok
  ADD r7,r4,r13
  LD r8,r7,0            ; positive value
  ADD r9,r5,r6          ; &A[offset]
  LD r10,r9,0           ; stored value
  CVTIF r11,r8          ; +value
  BOOL.6 r11,r11,r12    ; expected -value
  CMP r10,r11
  JMP[NE] @mth_diag_test_check_offdiag_bad
  ADD r4,r4,r13
  ADD r4,r4,r13         ; next pair
  JMP @mth_diag_test_check_offdiag_next
mth_diag_test_check_offdiag_ok:
  MOVI r1,1
  RET
mth_diag_test_check_offdiag_bad:
  MOVI r1,0
  RET

; Check all 256 outputs at r1 against D[r] = +(r+1).
; Returns r1=1 on exact match, 0 otherwise. Clobbers r4-r8; preserves r13.
mth_diag_test_check_output:
  ADD r4,r1,r0          ; output cursor
  MOVI r5,0             ; r
  MOVI r6,256           ; extent
mth_diag_test_check_output_loop:
  LD r7,r4,0            ; D[r]
  ADD r8,r5,r13
  CVTIF r8,r8           ; expected +(r+1)
  CMP r7,r8
  JMP[NE] @mth_diag_test_check_output_bad
  ADD r4,r4,r13
  ADD r5,r5,r13
  CMP r5,r6
  JMP[LT] @mth_diag_test_check_output_loop
  MOVI r1,1
  RET
mth_diag_test_check_output_bad:
  MOVI r1,0
  RET

; Check all 256 outputs at r1 have the sign bit clear.
; Returns r1=1 if all are non-negative, 0 otherwise. Clobbers r4-r8.
mth_diag_test_check_positive:
  ADD r4,r1,r0          ; output cursor
  MOVI r5,0             ; r
  MOVI r6,256           ; extent
  MOVI r7,0
  MOVHI r7,0x8000       ; sign bit
mth_diag_test_check_positive_loop:
  LD r8,r4,0
  BOOL.8 r8,r8,r7       ; isolate sign bit
  CMP r8,r0
  JMP[NE] @mth_diag_test_check_positive_bad
  ADD r4,r4,r13
  ADD r5,r5,r13
  CMP r5,r6
  JMP[LT] @mth_diag_test_check_positive_loop
  MOVI r1,1
  RET
mth_diag_test_check_positive_bad:
  MOVI r1,0
  RET

mth_diag_test_puts:
  MOVI r4,255
mth_diag_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_diag_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_diag_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_diag_test_puts_word

.org 512
mth_diag_test_a:
.word 0
.org 66048
mth_diag_test_d:
.word 0
.org 66304
mth_diag_test_pass:
.str "MTH MATRIX DIAGONAL256 F32 PASS\n"
.org 66313
mth_diag_test_bad:
.str "MTH MATRIX DIAGONAL256 F32 FAIL\n"
.org 66322
mth_diag_test_offdiag_table:
.word 1,1
.word 255,1
.word 256,2
.word 258,2
.word 32638,128
.word 32640,128
.word 32895,129
.word 32897,129
.word 65279,255
.word 65280,256
.word 65534,256
.word 0,0
