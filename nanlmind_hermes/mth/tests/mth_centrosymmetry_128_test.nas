.org 0
JMP @mth_centro_test_main
.include "../nan16/mth_centrosymmetry_128.nas"

mth_centro_test_main:
  MOVI r13,1

  ; Zero matrix -> 0 mismatches.
  MOVI r4,@mth_centro_matrix
  MOVI r6,16384
mth_centro_zero:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_centro_zero

  MOVI r1,@mth_centro_matrix
  CALL @mth_centrosymmetry_certify_128
  CMP r1,r13
  JMP[NE] @mth_centro_fail
  CMP r2,r0
  JMP[NE] @mth_centro_fail

  ; One asymmetric pair: set M[0,1]=1.0, leave M[127,126]=0.
  MOVI r4,@mth_centro_matrix
  ADD r4,r4,r13
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,0

  MOVI r1,@mth_centro_matrix
  CALL @mth_centrosymmetry_certify_128
  CMP r1,r13
  JMP[NE] @mth_centro_fail
  MOVI r5,2
  CMP r2,r5
  JMP[NE] @mth_centro_fail

  ; Repair the mirror entry at M[127,126] = 127*128 + 126 = 16382.
  MOVI r4,@mth_centro_matrix
  MOVI r5,16382
  ADD r4,r4,r5
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,0

  MOVI r1,@mth_centro_matrix
  CALL @mth_centrosymmetry_certify_128
  CMP r1,r13
  JMP[NE] @mth_centro_fail
  CMP r2,r0
  JMP[NE] @mth_centro_fail

  MOVI r1,@mth_centro_pass
  CALL @mth_centro_puts
  HALT

mth_centro_fail:
  MOVI r1,@mth_centro_bad
  CALL @mth_centro_puts
  HALT

mth_centro_puts:
  MOVI r4,255
mth_centro_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_centro_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_centro_puts_byte
  ADD r1,r1,r13
  JMP @mth_centro_puts_word

.org 512
mth_centro_matrix:
.word 0
.org 17408
mth_centro_pass:
.str "MTH CENTROSYMMETRY 128 PASS\n"
mth_centro_bad:
.str "MTH CENTROSYMMETRY 128 FAIL\n"
