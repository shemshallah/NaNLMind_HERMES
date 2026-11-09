; MTH 15x255 block certificate fixture for the odd 255-head matrix — NaN16 only.
;
; Base 120 makes the block global rows 120..134, so the fixed middle row 127
; is both A row 7 and mirror row 7. The caller-supplied mirror block is the
; exact row reversal of A:
;   mirror[k,c] = A[14-k,c].
; Column 127 is fixed by the 255-head reversal. The fixture therefore covers
; the fixed middle row (a column-palindrome check at k=7) and the fixed middle
; column (j=127 compares with mirror[k,127]) without any kernel special case.
;
; The fixture also makes the middle cell (127,127) a NaN payload, and puts a
; second, different NaN payload on an ordinary two-cell orbit. Equality is
; integer CMP on the payload, so equal payloads pass and payload differences
; fail.
.org 0
JMP @mth_centrosymmetry255_block15_test_main
.include "../nan16/mth_centrosymmetry255_block15.nas"

mth_centrosymmetry255_block15_test_main:
  MOVI r13,1

  ; Zero both 3,825-word blocks.
  MOVI r4,@mth_centrosymmetry255_block15_a
  MOVI r6,3825
mth_centro255_zero_a:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_zero_a

  MOVI r4,@mth_centrosymmetry255_block15_mirror
  MOVI r6,3825
mth_centro255_zero_b:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_zero_b

  ; ---- A nonzeros. A row k is global row 120+k. ----
  MOVI r4,@mth_centrosymmetry255_block15_a

  ; A[0,0] = 1.0 and its partner A[14,254] = 1.0.
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r4,0
  MOVI r7,3824
  ADD r7,r4,r7
  ST r6,r7,0

  ; A[3,17] = 2.5 and its partner A[11,237] = 2.5.
  MOVI r6,0
  MOVHI r6,0x4020
  MOVI r7,782
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,3042
  ADD r7,r4,r7
  ST r6,r7,0

  ; Middle row 127: A[7,20] and A[7,234] are a palindrome pair (pi float).
  MOVI r6,0x0FDB
  MOVHI r6,0x4049
  MOVI r7,1805
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,2019
  ADD r7,r4,r7
  ST r6,r7,0

  ; Fixed column 127: A[3,127] and A[11,127] are a palindrome pair (-2.0).
  MOVI r6,0
  MOVHI r6,0xC000
  MOVI r7,892
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,2932
  ADD r7,r4,r7
  ST r6,r7,0

  ; Middle cell (127,127): NaN payload 0x7FC00001.
  MOVI r6,1
  MOVHI r6,0x7FC0
  MOVI r7,1912
  ADD r7,r4,r7
  ST r6,r7,0

  ; Ordinary pair with a different NaN payload 0x7FC00002:
  ; A[2,64] and its partner A[12,190].
  MOVI r6,2
  MOVHI r6,0x7FC0
  MOVI r7,574
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,3250
  ADD r7,r4,r7
  ST r6,r7,0

  ; ---- Build the exact mirror block: mirror[k,c] = A[14-k,c]. ----
  MOVI r4,@mth_centrosymmetry255_block15_a
  MOVI r7,3570
  ADD r4,r4,r7              ; A row 14
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  MOVI r6,15
mth_centro255_build_row:
  ADD r8,r4,r0
  ADD r9,r5,r0
  MOVI r7,255
mth_centro255_build_col:
  LD r10,r8,0
  ST r10,r9,0
  ADD r8,r8,r13
  ADD r9,r9,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_centro255_build_col
  MOVI r7,255
  SUB r4,r4,r7
  ADD r5,r5,r7
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro255_build_row

  ; ---- Spot-check the constructed blocks. ----
  ; These prove the middle-row, fixed-column, centre, and generic nonzeros
  ; are genuinely present, so the 0-mismatch call is not a zero-field pass.
  ; A[3,17] = 2.5.
  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,782
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0x4020
  CALL @mth_centro255_expect_word

  ; Middle row 127: A[7,20] = pi and mirror[7,234] = A[7,234] = pi.
  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,1805
  ADD r1,r1,r2
  MOVI r2,0x0FDB
  MOVHI r2,0x4049
  CALL @mth_centro255_expect_word

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,2019
  ADD r1,r1,r2
  MOVI r2,0x0FDB
  MOVHI r2,0x4049
  CALL @mth_centro255_expect_word

  ; Fixed column 127: A[3,127] = -2.0 and mirror[11,127] = A[3,127].
  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,892
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0xC000
  CALL @mth_centro255_expect_word

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,2932
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0xC000
  CALL @mth_centro255_expect_word

  ; Centre cell (127,127) carries NaN payload 0x7FC00001 in both blocks.
  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,1912
  ADD r1,r1,r2
  MOVI r2,1
  MOVHI r2,0x7FC0
  CALL @mth_centro255_expect_word

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,1912
  ADD r1,r1,r2
  MOVI r2,1
  MOVHI r2,0x7FC0
  CALL @mth_centro255_expect_word

  ; Generic row-reversed pair: mirror[3,237] = A[11,237] = 2.5.
  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,1002
  ADD r1,r1,r2
  MOVI r2,0
  MOVHI r2,0x4020
  CALL @mth_centro255_expect_word

  ; ---- Exact block: 0 mismatches in both orientations. ----
  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  ; ---- One asymmetric matrix entry in the self-opposite block. ----
  ; Base 120 contains both cells of the orbit. Set the global cell
  ; M[131,237] = 3.0, which appears as A[11,237] and mirror[3,237], while
  ; the partner M[123,17] stays 2.5 at A[3,17] and mirror[11,17].
  ; This single asymmetric entry must produce 2 mismatches in one call.
  MOVI r6,0
  MOVHI r6,0x4040           ; 3.0f
  MOVI r7,3042
  MOVI r4,@mth_centrosymmetry255_block15_a
  ADD r7,r4,r7
  ST r6,r7,0                ; A[11,237]
  MOVI r7,1002
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0                ; mirror[3,237]

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_two

  ; Restore the cell to 2.5 in both copies.
  MOVI r6,0
  MOVHI r6,0x4020
  MOVI r7,3042
  MOVI r4,@mth_centrosymmetry255_block15_a
  ADD r7,r4,r7
  ST r6,r7,0                ; A[11,237]
  MOVI r7,1002
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0                ; mirror[3,237]

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  ; ---- Perturb one mirror payload: mirror[3,237] = 0x40200001. ----
  ; Forward orientation sees 1 failure. The reversed orientation sees the same
  ; corrupted cell from the other side: 1 more, so the two-call probe total is
  ; 2. Base 120 is self-opposite: the real 17-call sweep calls this block once,
  ; and the coherent single-entry check above is the sweep-faithful 2 count.
  MOVI r6,1
  MOVHI r6,0x4020
  MOVI r7,1002
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  ST r2,r5,0

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry255_block15_fail

  ; Repair mirror[3,237] to 2.5.
  MOVI r6,0
  MOVHI r6,0x4020
  MOVI r7,1002
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  ; ---- Fixed middle row 127: perturb mirror[7,234] = 0x40490FDC. ----
  MOVI r6,0x0FDC
  MOVHI r6,0x4049
  MOVI r7,2019
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  ST r2,r5,0

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry255_block15_fail

  ; Restore the middle-row pair.
  MOVI r6,0x0FDB
  MOVHI r6,0x4049
  MOVI r7,2019
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  ; ---- Fixed middle column 127: perturb mirror[11,127] to 0xC0000001. ----
  MOVI r6,1
  MOVHI r6,0xC000
  MOVI r7,2932
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  ST r2,r5,0

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry255_block15_fail

  ; Restore the fixed-column pair.
  MOVI r6,0
  MOVHI r6,0xC000
  MOVI r7,2932
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  ; ---- NaN payload pair: perturb mirror[2,190] to 0x7FC00003. ----
  MOVI r6,3
  MOVHI r6,0x7FC0
  MOVI r7,700
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  ST r2,r5,0

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_one
  MOVI r5,@mth_centrosymmetry255_block15_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry255_block15_fail

  ; Restore the NaN payload pair.
  MOVI r6,2
  MOVHI r6,0x7FC0
  MOVI r7,700
  MOVI r5,@mth_centrosymmetry255_block15_mirror
  ADD r7,r5,r7
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry255_block15_a
  MOVI r2,@mth_centrosymmetry255_block15_mirror
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  MOVI r1,@mth_centrosymmetry255_block15_mirror
  MOVI r2,@mth_centrosymmetry255_block15_a
  CALL @mth_centrosymmetry255_block15
  CALL @mth_centro255_expect_zero

  MOVI r1,@mth_centrosymmetry255_block15_pass
  CALL @mth_centrosymmetry255_block15_puts
  HALT

mth_centrosymmetry255_block15_fail:
  MOVI r1,@mth_centrosymmetry255_block15_bad
  CALL @mth_centrosymmetry255_block15_puts
  HALT

; Leaf check helpers: r1 must be 1 and r2 must be 0, 1, or 2.
mth_centro255_expect_word:
  LD r4,r1,0
  CMP r4,r2
  JMP[NE] @mth_centrosymmetry255_block15_fail
  RET

mth_centro255_expect_zero:
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry255_block15_fail
  CMP r2,r0
  JMP[NE] @mth_centrosymmetry255_block15_fail
  RET

mth_centro255_expect_one:
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry255_block15_fail
  CMP r2,r13
  JMP[NE] @mth_centrosymmetry255_block15_fail
  RET

mth_centro255_expect_two:
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry255_block15_fail
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry255_block15_fail
  RET

mth_centrosymmetry255_block15_puts:
  MOVI r4,255
mth_centrosymmetry255_block15_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_centrosymmetry255_block15_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_centrosymmetry255_block15_puts_byte
  ADD r1,r1,r13
  JMP @mth_centrosymmetry255_block15_puts_word

.org 512
mth_centrosymmetry255_block15_a:
.word 0
.org 4608
mth_centrosymmetry255_block15_mirror:
.word 0
.org 8704
mth_centrosymmetry255_block15_count:
.word 0
.org 8768
mth_centrosymmetry255_block15_pass:
.str "MTH CENTROSYMMETRY 255 BLOCK15 PASS\n"
mth_centrosymmetry255_block15_bad:
.str "MTH CENTROSYMMETRY 255 BLOCK15 FAIL\n"
