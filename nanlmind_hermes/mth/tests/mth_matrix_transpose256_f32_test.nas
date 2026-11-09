; MTH 256x256 float32 transpose fixture — NaN16 only.
;
; The input is built with 256 runtime writes, not a 65,536-line source.
; A[i,i] = -(i+1) and A[i,(i+1) mod 256] = +(i+1), so every input row has
; one diagonal and one off-diagonal nonzero.
;
; Phase 1 checks every one of the 65,536 output words of B = transpose(A):
;   B[r,r]              = -(r+1)                  (diagonal mapping)
;   B[r,(r-1) mod 256]  = ((r-1) mod 256) + 1     (off-diagonal mapping)
;   all other B[r,c]    = 0
; The off-diagonal value for r=0 is 256 because source i=255.
;
; Phase 2 transposes B back into A and checks all 65,536 words of A against
; the original source pattern, proving transpose(transpose(A)) = A for this
; dense image.

.org 0
JMP @mth_transpose_test_main
.include "../nan16/mth_matrix_transpose256_f32.nas"

mth_transpose_test_main:
  MOVI r13,1

  ; Zero both dense matrices.
  MOVI r1,@mth_transpose_test_a
  CALL @mth_transpose_test_zero64k
  LA r1,@mth_transpose_test_b
  CALL @mth_transpose_test_zero64k

  ; Build the sparse diagonal + rotation pattern in A.
  MOVI r1,@mth_transpose_test_a
  CALL @mth_transpose_test_fill_pattern

  ; Phase 1: A -> B.
  MOVI r1,@mth_transpose_test_a
  LA r2,@mth_transpose_test_b
  CALL @mth_matrix_transpose256_f32
  CMP r1,r13
  JMP[NE] @mth_transpose_test_fail

  LA r1,@mth_transpose_test_b
  CALL @mth_transpose_test_check_transposed
  CMP r1,r13
  JMP[NE] @mth_transpose_test_fail

  ; Phase 2 (involution): B -> A must restore the original pattern.
  LA r1,@mth_transpose_test_b
  MOVI r2,@mth_transpose_test_a
  CALL @mth_matrix_transpose256_f32
  CMP r1,r13
  JMP[NE] @mth_transpose_test_fail

  MOVI r1,@mth_transpose_test_a
  CALL @mth_transpose_test_check_source
  CMP r1,r13
  JMP[NE] @mth_transpose_test_fail

  LA r1,@mth_transpose_test_pass
  CALL @mth_transpose_test_puts
  HALT

mth_transpose_test_fail:
  LA r1,@mth_transpose_test_bad
  CALL @mth_transpose_test_puts
  HALT

; Zero 65,536 words at r1. Clobbers r2-r3; preserves r1 and r13.
mth_transpose_test_zero64k:
  ADD r2,r1,r0
  MOVI r3,0
  MOVHI r3,0x0001
mth_transpose_test_zero64k_loop:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_transpose_test_zero64k_loop
  RET

; Fill A at r1 with A[i,i] = -(i+1), A[i,(i+1)&255] = +(i+1).
; Clobbers r1-r10; preserves r13.
mth_transpose_test_fill_pattern:
  ADD r4,r1,r0          ; row base
  MOVI r5,0             ; i
  MOVI r1,0
  MOVHI r1,0x3F80       ; 1.0
  MOVI r2,0
  MOVHI r2,0x8000       ; sign bit
  MOVI r3,255
  MOVI r10,256
mth_transpose_test_fill_row:
  ADD r6,r5,r13         ; i+1
  CVTIF r7,r6           ; +(i+1)
  MOVI r8,0
  BOOL.6 r8,r7,r2       ; -(i+1), exact sign flip
  ADD r9,r4,r5          ; &A[i,i]
  ST r8,r9,0
  BOOL.8 r6,r6,r3       ; (i+1) mod 256
  ADD r9,r4,r6          ; &A[i,(i+1) mod 256]
  ST r7,r9,0
  ADD r4,r4,r10         ; next row
  ADD r5,r5,r13
  CMP r5,r10
  JMP[LT] @mth_transpose_test_fill_row
  RET

; Check 65,536 words at r1 against the original source pattern:
;   M[r,r] = -(r+1), M[r,(r+1)&255] = r+1, and zero elsewhere.
; Returns r1=1 on exact match, 0 otherwise. Clobbers r1-r12 and r14.
mth_transpose_test_check_source:
  ADD r4,r1,r0          ; row base
  MOVI r5,0             ; r
  MOVI r1,0
  MOVHI r1,0x3F80       ; 1.0
  MOVI r2,0
  MOVHI r2,0x8000       ; sign bit
  MOVI r3,255
  MOVI r12,256
mth_transpose_test_check_source_row:
  MOVI r6,0
  ADD r7,r5,r13
  CVTIF r7,r7
  BOOL.6 r6,r7,r2       ; diagonal value -(r+1)
  ADD r8,r5,r13
  BOOL.8 r8,r8,r3       ; off-diagonal column (r+1) mod 256
  ADD r10,r4,r0         ; word cursor
  MOVI r9,0             ; c
mth_transpose_test_check_source_col:
  MOVI r11,0
  CMP r9,r5
  JMP[EQ] @mth_transpose_test_check_source_diag
  CMP r9,r8
  JMP[EQ] @mth_transpose_test_check_source_off
  JMP @mth_transpose_test_check_source_cmp
mth_transpose_test_check_source_diag:
  ADD r11,r6,r0
  JMP @mth_transpose_test_check_source_cmp
mth_transpose_test_check_source_off:
  ADD r11,r7,r0
mth_transpose_test_check_source_cmp:
  LD r14,r10,0
  CMP r14,r11
  JMP[NE] @mth_transpose_test_check_bad
  ADD r10,r10,r13
  ADD r9,r9,r13
  CMP r9,r12
  JMP[LT] @mth_transpose_test_check_source_col
  ADD r4,r4,r12
  ADD r5,r5,r13
  CMP r5,r12
  JMP[LT] @mth_transpose_test_check_source_row
  MOVI r1,1
  RET

; Check 65,536 words at r1 against the transposed pattern:
;   M[r,r] = -(r+1), M[r,(r-1)&255] = ((r-1)&255)+1, zero elsewhere.
; Returns r1=1 on exact match, 0 otherwise. Clobbers r1-r12 and r14.
mth_transpose_test_check_transposed:
  ADD r4,r1,r0          ; row base
  MOVI r5,0             ; r
  MOVI r1,0
  MOVHI r1,0x3F80       ; 1.0
  MOVI r2,0
  MOVHI r2,0x8000       ; sign bit
  MOVI r3,255
  MOVI r12,256
mth_transpose_test_check_transposed_row:
  CVTIF r6,r5
  FADD r6,r6,r1
  BOOL.6 r6,r6,r2       ; diagonal value -(r+1)
  ADD r7,r5,r3
  BOOL.8 r7,r7,r3       ; off-diagonal column (r-1) mod 256
  ADD r8,r7,r13
  CVTIF r8,r8           ; off-diagonal value
  ADD r10,r4,r0         ; word cursor
  MOVI r9,0             ; c
mth_transpose_test_check_transposed_col:
  MOVI r11,0
  CMP r9,r5
  JMP[EQ] @mth_transpose_test_check_transposed_diag
  CMP r9,r7
  JMP[EQ] @mth_transpose_test_check_transposed_off
  JMP @mth_transpose_test_check_transposed_cmp
mth_transpose_test_check_transposed_diag:
  ADD r11,r6,r0
  JMP @mth_transpose_test_check_transposed_cmp
mth_transpose_test_check_transposed_off:
  ADD r11,r8,r0
mth_transpose_test_check_transposed_cmp:
  LD r14,r10,0
  CMP r14,r11
  JMP[NE] @mth_transpose_test_check_bad
  ADD r10,r10,r13
  ADD r9,r9,r13
  CMP r9,r12
  JMP[LT] @mth_transpose_test_check_transposed_col
  ADD r4,r4,r12
  ADD r5,r5,r13
  CMP r5,r12
  JMP[LT] @mth_transpose_test_check_transposed_row
  MOVI r1,1
  RET

mth_transpose_test_check_bad:
  MOVI r1,0
  RET

mth_transpose_test_puts:
  MOVI r4,255
mth_transpose_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_transpose_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_transpose_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_transpose_test_puts_word

.org 512
mth_transpose_test_a:
.word 0
.org 66048
mth_transpose_test_b:
.word 0
.org 131584
mth_transpose_test_pass:
.str "MTH MATRIX TRANSPOSE256 F32 PASS\n"
.org 131600
mth_transpose_test_bad:
.str "MTH MATRIX TRANSPOSE256 F32 FAIL\n"
