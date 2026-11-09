; MTH full 255x255 centrosymmetry fixture — NaN16 only.
;
; The fixture builds a genuinely non-trivial centrosymmetric 255x255 matrix
; in guest memory and then exercises the full-matrix certificate:
;
;   1. initial symmetric matrix: kernel returns 1;
;   2. one word changed in a generic two-cell orbit: kernel returns 0;
;   3. the same word restored: kernel returns 1;
;   4. one word changed on the fixed row 127: kernel returns 0;
;   5. restore, one word changed on the fixed column 127: kernel returns 0;
;   6. restore, one word changed at the far corner (254,0): kernel returns 0;
;   7. restore, final full call: kernel returns 1.
;
; Every rejection changes exactly one 32-bit payload word. The value is
; bumped by +1 on the guest side, so the mirror partner is untouched.
;
; Fill order: the 127 upper-half rows are filled first, each j=0..254 orbit
; writing both A[i,j] and A[254-i,254-j] with the same deterministic payload.
; Row 127 is then filled as the fixed-row palindrome pairs j and 254-j, and
; finally the centre cell (127,127). Payload k is 0x3F800001 + k; values are
; distinct, finite, and non-zero, so a zero-filled or pivot-permuted matrix
; cannot pass. The direct word checks below pin down generic, fixed-row,
; fixed-column, and centre coverage before the kernel is called.

.org 0
JMP @mth_centrosymmetry_255_full_test_main
.include "../nan16/mth_centrosymmetry_255_full.nas"

mth_centrosymmetry_255_full_test_main:
  MOVI r13,1

  ; ---- Zero all 65,025 matrix words. ----
  LA r4,@mth_centrosymmetry_255_full_a
  MOVI r6,65025
mth_centro255_full_zero:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_full_zero

  ; ---- Fill the 127 upper-half row pairs. ----
  ; r4 = A[i,0], r5 = A[254-i,254], payload r7 = 0x3F800001 + orbit.
  LA r4,@mth_centrosymmetry_255_full_a
  MOVI r5,65024
  ADD r5,r4,r5              ; A[254,254]
  MOVI r7,1
  MOVHI r7,0x3F80           ; first payload 0x3F800001
  MOVI r6,127               ; paired row count
mth_centro255_full_fill_row:
  ADD r9,r4,r0              ; forward pointer
  ADD r10,r5,r0             ; backward pointer
  MOVI r8,255
mth_centro255_full_fill_col:
  ST r7,r9,0                ; A[i,j]
  ST r7,r10,0               ; A[254-i,254-j]
  ADD r7,r7,r13
  ADD r9,r9,r13
  SUB r10,r10,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_centro255_full_fill_col
  MOVI r11,255
  ADD r4,r4,r11             ; next row i+1
  SUB r5,r5,r11             ; previous mirror row 253-i
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_full_fill_row

  ; After 127 rows, r4 = A[127,0] and r5 = A[127,254].
  ; ---- Fill fixed row 127: pairs j and 254-j. ----
  MOVI r6,127
mth_centro255_full_fill_mid:
  ST r7,r4,0
  ST r7,r5,0
  ADD r7,r7,r13
  ADD r4,r4,r13
  SUB r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_full_fill_mid

  ; ---- Centre cell (127,127). ----
  ST r7,r4,0

  ; ---- Spot-check the constructed matrix. ----
  ; Generic corner orbit.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,1
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,65024
  ADD r1,r1,r2
  MOVI r2,1
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; Generic near-end orbit: A[0,254] and A[254,0] share 0x3F8000FF.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,254
  ADD r1,r1,r2
  MOVI r2,255
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,64770
  ADD r1,r1,r2
  MOVI r2,255
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; Generic interior orbit: A[3,17] and A[251,237] share 0x3F80030F.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,782
  ADD r1,r1,r2
  MOVI r2,0x030F
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,64242
  ADD r1,r1,r2
  MOVI r2,0x030F
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; Fixed row 127, deepest pair: A[127,126] and A[127,128] share 0x3F807F00.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,32511
  ADD r1,r1,r2
  MOVI r2,0x7F00
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,32513
  ADD r1,r1,r2
  MOVI r2,0x7F00
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; Fixed column 127: A[20,127] and A[234,127] share 0x3F80146C.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,5227
  ADD r1,r1,r2
  MOVI r2,0x146C
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,59797
  ADD r1,r1,r2
  MOVI r2,0x146C
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; Centre cell (127,127): 0x3F807F01.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,32512
  ADD r1,r1,r2
  MOVI r2,0x7F01
  MOVHI r2,0x3F80
  CALL @mth_centro255_full_expect_word

  ; ---- 1. Exact matrix: certificate returns 1. ----
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_255_full_fail

  ; ---- 2. One generic word changed: reject. ----
  ; A[3,17] at 3*255 + 17 = 782, partner A[251,237] stays put.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,782
  ADD r1,r1,r2
  LD r3,r1,0
  ADD r3,r3,r13
  ST r3,r1,0
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r0
  JMP[NE] @mth_centrosymmetry_255_full_fail
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,782
  ADD r1,r1,r2
  LD r3,r1,0
  SUB r3,r3,r13
  ST r3,r1,0

  ; ---- 3. Restored: certificate returns 1. ----
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_255_full_fail

  ; ---- 4. Fixed row 127, one word changed: reject. ----
  ; A[127,126], partner A[127,128] stays put. This is the deepest
  ; non-self orbit in the scan: it is compared just before the centre.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,32511
  ADD r1,r1,r2
  LD r3,r1,0
  ADD r3,r3,r13
  ST r3,r1,0
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r0
  JMP[NE] @mth_centrosymmetry_255_full_fail
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,32511
  ADD r1,r1,r2
  LD r3,r1,0
  SUB r3,r3,r13
  ST r3,r1,0

  ; ---- 5. Fixed column 127, one word changed: reject. ----
  ; A[20,127], partner A[234,127] stays put.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,5227
  ADD r1,r1,r2
  LD r3,r1,0
  ADD r3,r3,r13
  ST r3,r1,0
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r0
  JMP[NE] @mth_centrosymmetry_255_full_fail
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,5227
  ADD r1,r1,r2
  LD r3,r1,0
  SUB r3,r3,r13
  ST r3,r1,0

  ; ---- 6. Far corner, one word changed: reject. ----
  ; A[254,0] at 254*255 = 64,770, partner A[0,254] stays put.
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,64770
  ADD r1,r1,r2
  LD r3,r1,0
  ADD r3,r3,r13
  ST r3,r1,0
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r0
  JMP[NE] @mth_centrosymmetry_255_full_fail
  LA r1,@mth_centrosymmetry_255_full_a
  MOVI r2,64770
  ADD r1,r1,r2
  LD r3,r1,0
  SUB r3,r3,r13
  ST r3,r1,0

  ; ---- 7. Final exact matrix: certificate returns 1. ----
  LA r1,@mth_centrosymmetry_255_full_a
  CALL @mth_centrosymmetry_255_full
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_255_full_fail

  LA r1,@mth_centrosymmetry_255_full_pass
  CALL @mth_centro255_full_puts
  HALT

mth_centrosymmetry_255_full_fail:
  LA r1,@mth_centrosymmetry_255_full_bad
  CALL @mth_centro255_full_puts
  HALT

; r1 = matrix word address, r2 = expected payload.
mth_centro255_full_expect_word:
  LD r4,r1,0
  CMP r4,r2
  JMP[NE] @mth_centrosymmetry_255_full_fail
  RET

mth_centro255_full_puts:
  MOVI r4,255
mth_centro255_full_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_centro255_full_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_centro255_full_puts_byte
  ADD r1,r1,r13
  JMP @mth_centro255_full_puts_word

; The matrix occupies words 512..65,536 (65,025 words). The strings start at
; 65,537, so the loaded image is larger than the historical 65,536-word guest.
.org 512
mth_centrosymmetry_255_full_a:
.word 0
.org 65537
mth_centrosymmetry_255_full_pass:
.str "MTH CENTROSYMMETRY 255 FULL PASS\n"
mth_centrosymmetry_255_full_bad:
.str "MTH CENTROSYMMETRY 255 FULL FAIL\n"
