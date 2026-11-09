.org 0
JMP @mth_centrosymmetry_block16_test_main
.include "../nan16/mth_centrosymmetry_block16.nas"

mth_centrosymmetry_block16_test_main:
  MOVI r13,1

  ; Both blocks are row-major. A row k is global row base+k; mirror row k is
  ; global row 255-(base+k) in natural column order. The kernel performs the
  ; column reversal when it compares A[k,j] with mirror[k,255-j].

  ; Zero both 4,096-word blocks. They are contiguous in this fixture.
  MOVI r4,@mth_centrosymmetry_block16_a
  MOVI r6,8192
mth_centrosymmetry_block16_zero:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centrosymmetry_block16_zero

  MOVI r4,@mth_centrosymmetry_block16_a
  MOVI r5,@mth_centrosymmetry_block16_mirror

  ; A[0,0] = 1.0 and mirror[0,255] = 1.0.
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r4,0
  MOVI r7,255
  ADD r7,r5,r7
  ST r6,r7,0

  ; A[3,17] = 2.5 (offset 785) and mirror[3,238] = 2.5 (offset 1006).
  MOVI r6,0
  MOVHI r6,0x4020
  MOVI r7,785
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,1006
  ADD r7,r5,r7
  ST r6,r7,0

  ; A[8,100] = 0x40490FDB (offset 2148), mirror[8,155] same (offset 2203).
  MOVI r6,0x0FDB
  MOVHI r6,0x4049
  MOVI r7,2148
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,2203
  ADD r7,r5,r7
  ST r6,r7,0

  ; A[15,255] = -2.0 (offset 4095) and mirror[15,0] = -2.0 (offset 3840).
  MOVI r6,0
  MOVHI r6,0xC000
  MOVI r7,4095
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,3840
  ADD r7,r5,r7
  ST r6,r7,0

  ; A[12,64] = 0x7FC00001 (offset 3136), mirror[12,191] same (offset 3263).
  ; Equal NaN payloads must pass because the comparison is integer CMP.
  MOVI r6,1
  MOVHI r6,0x7FC0
  MOVI r7,3136
  ADD r7,r4,r7
  ST r6,r7,0
  MOVI r7,3263
  ADD r7,r5,r7
  ST r6,r7,0

  ; Exact mirror image -> 0 mismatches.
  MOVI r1,@mth_centrosymmetry_block16_a
  MOVI r2,@mth_centrosymmetry_block16_mirror
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  CMP r2,r0
  JMP[NE] @mth_centrosymmetry_block16_fail

  ; Perturb one mirror payload: mirror[3,238] becomes 0x40200001, one bit
  ; away from 2.5 at A[3,17].
  MOVI r4,@mth_centrosymmetry_block16_mirror
  MOVI r7,1006
  ADD r7,r4,r7
  MOVI r6,1
  MOVHI r6,0x4020
  ST r6,r7,0

  ; Forward orientation sees exactly one failed comparison.
  MOVI r1,@mth_centrosymmetry_block16_a
  MOVI r2,@mth_centrosymmetry_block16_mirror
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,1
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,@mth_centrosymmetry_block16_count
  ST r2,r5,0

  ; Reverse orientation visits the same asymmetric pair from the other side.
  ; Forward + reverse = 2 mismatches, the streaming composition's count.
  MOVI r1,@mth_centrosymmetry_block16_mirror
  MOVI r2,@mth_centrosymmetry_block16_a
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,@mth_centrosymmetry_block16_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry_block16_fail

  ; Repair the entry; both orientations certify zero again.
  MOVI r4,@mth_centrosymmetry_block16_mirror
  MOVI r7,1006
  ADD r7,r4,r7
  MOVI r6,0
  MOVHI r6,0x4020
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry_block16_a
  MOVI r2,@mth_centrosymmetry_block16_mirror
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  CMP r2,r0
  JMP[NE] @mth_centrosymmetry_block16_fail

  MOVI r1,@mth_centrosymmetry_block16_mirror
  MOVI r2,@mth_centrosymmetry_block16_a
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  CMP r2,r0
  JMP[NE] @mth_centrosymmetry_block16_fail

  ; Perturb the NaN payload instead: mirror[12,191] = 0x7FC00002.
  ; This must also produce 2 mismatches over the two orientations.
  MOVI r4,@mth_centrosymmetry_block16_mirror
  MOVI r7,3263
  ADD r7,r4,r7
  MOVI r6,2
  MOVHI r6,0x7FC0
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry_block16_a
  MOVI r2,@mth_centrosymmetry_block16_mirror
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,1
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,@mth_centrosymmetry_block16_count
  ST r2,r5,0

  MOVI r1,@mth_centrosymmetry_block16_mirror
  MOVI r2,@mth_centrosymmetry_block16_a
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  MOVI r5,@mth_centrosymmetry_block16_count
  LD r6,r5,0
  ADD r2,r2,r6
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centrosymmetry_block16_fail

  ; Restore the NaN payload and check the forward orientation once more.
  MOVI r4,@mth_centrosymmetry_block16_mirror
  MOVI r7,3263
  ADD r7,r4,r7
  MOVI r6,1
  MOVHI r6,0x7FC0
  ST r6,r7,0

  MOVI r1,@mth_centrosymmetry_block16_a
  MOVI r2,@mth_centrosymmetry_block16_mirror
  CALL @mth_centrosymmetry_block16
  CMP r1,r13
  JMP[NE] @mth_centrosymmetry_block16_fail
  CMP r2,r0
  JMP[NE] @mth_centrosymmetry_block16_fail

  MOVI r1,@mth_centrosymmetry_block16_pass
  CALL @mth_centrosymmetry_block16_puts
  HALT

mth_centrosymmetry_block16_fail:
  MOVI r1,@mth_centrosymmetry_block16_bad
  CALL @mth_centrosymmetry_block16_puts
  HALT

mth_centrosymmetry_block16_puts:
  MOVI r4,255
mth_centrosymmetry_block16_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_centrosymmetry_block16_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_centrosymmetry_block16_puts_byte
  ADD r1,r1,r13
  JMP @mth_centrosymmetry_block16_puts_word

.org 512
mth_centrosymmetry_block16_a:
.word 0
.org 4608
mth_centrosymmetry_block16_mirror:
.word 0
.org 8704
mth_centrosymmetry_block16_count:
.word 0
.org 8768
mth_centrosymmetry_block16_pass:
.str "MTH CENTROSYMMETRY BLOCK16 PASS\n"
mth_centrosymmetry_block16_bad:
.str "MTH CENTROSYMMETRY BLOCK16 FAIL\n"
