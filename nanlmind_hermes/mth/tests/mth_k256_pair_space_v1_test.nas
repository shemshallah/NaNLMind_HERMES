.org 0
JMP @mth_pair_test_main
.include "../nan16/mth_k256_pair_space_v1.nas"

mth_pair_test_main:
  MOVI r13,1
  MOVI r1,@mth_pair_input
  LA r2,@mth_pair_output
  LA r3,@mth_pair_rows
  LA r4,@mth_pair_cols
  LA r5,@mth_pair_scratch
  CALL @mth_k256_pair_space_q16_16
  CMP r1,r13
  JMP[NE] @mth_pair_fail

  ; U[0,0]=1, all other entries zero.
  ; Y[0,0]=510, Y[0,1]=-1, Y[1,0]=-1, Y[1,1]=0.
  LA r6,@mth_pair_output
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x01FE
  CMP r7,r8
  JMP[NE] @mth_pair_fail
  LD r7,r6,1
  MOVI r8,0
  MOVHI r8,0xFFFF
  CMP r7,r8
  JMP[NE] @mth_pair_fail
  LD r7,r6,256
  CMP r7,r8
  JMP[NE] @mth_pair_fail
  LD r7,r6,257
  CMP r7,r0
  JMP[NE] @mth_pair_fail

  ; Independent row/column sums are both exactly 1.0 in Q16.16.
  LA r6,@mth_pair_rows
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,1
  CMP r7,r8
  JMP[NE] @mth_pair_fail
  LA r6,@mth_pair_cols
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @mth_pair_fail

  LA r1,@mth_pair_pass
  CALL @mth_pair_puts
  HALT
mth_pair_fail:
  LA r1,@mth_pair_bad
  CALL @mth_pair_puts
  HALT
mth_pair_puts:
  MOVI r4,255
mth_pair_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pair_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pair_puts_byte
  ADD r1,r1,r13
  JMP @mth_pair_puts_word

.org 1024
mth_pair_input:
.word 65536
.org 66560
mth_pair_output:
.word 0
.org 132096
mth_pair_rows:
.word 0
.org 132352
mth_pair_cols:
.word 0
.org 132608
mth_pair_scratch:
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.org 140000
.word 0
.org 140100
mth_pair_pass:
.str "MTH K256 PAIR SPACE V1 PASS\n"
mth_pair_bad:
.str "MTH K256 PAIR SPACE V1 FAIL\n"
