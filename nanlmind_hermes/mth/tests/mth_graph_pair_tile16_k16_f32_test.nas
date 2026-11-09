.org 0
JMP @mth_pair_k16_test_main
.include "../nan16/mth_graph_pair_tile16_k16_f32.nas"

mth_pair_k16_test_main:
  MOVI r13,1
  MOVI r5,0
  MOVHI r5,0x4000

  ; K block 16..31: L_IK[1,1]=2 and U_KJ[1,2]=2.
  LA r4,@mth_pair_k16_li
  MOVI r6,17
  ADD r4,r4,r6
  ST r5,r4,0
  LA r4,@mth_pair_k16_ukj
  MOVI r6,18
  ADD r4,r4,r6
  ST r5,r4,0

  LA r1,@mth_pair_k16_li
  LA r2,@mth_pair_k16_ukj
  LA r3,@mth_pair_k16_uik
  LA r4,@mth_pair_k16_lj
  LA r5,@mth_pair_k16_acc_lu
  LA r6,@mth_pair_k16_acc_ult
  LA r7,@mth_pair_k16_scratch
  CALL @mth_graph_pair_tile16_k16_accum_f32
  MOVI r8,1
  CMP r1,r8
  JMP[NE] @mth_pair_k16_fail

  ; After K=16..31, only acc_LU[1,2] is 4.0.
  LA r4,@mth_pair_k16_acc_lu
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x4080
  CMP r7,r8
  JMP[NE] @mth_pair_k16_fail
  LA r4,@mth_pair_k16_acc_ult
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  CMP r7,r0
  JMP[NE] @mth_pair_k16_fail

  ; K block 32..47: clear first-term inputs, then U_IK[1,2]=2 and L_JK[2,2]=2.
  MOVI r5,0
  MOVHI r5,0x4000
  LA r4,@mth_pair_k16_li
  MOVI r6,17
  ADD r4,r4,r6
  ST r0,r4,0
  LA r4,@mth_pair_k16_ukj
  MOVI r6,18
  ADD r4,r4,r6
  ST r0,r4,0
  LA r4,@mth_pair_k16_uik
  MOVI r6,18
  ADD r4,r4,r6
  ST r5,r4,0
  LA r4,@mth_pair_k16_lj
  MOVI r6,34
  ADD r4,r4,r6
  ST r5,r4,0

  LA r1,@mth_pair_k16_li
  LA r2,@mth_pair_k16_ukj
  LA r3,@mth_pair_k16_uik
  LA r4,@mth_pair_k16_lj
  LA r5,@mth_pair_k16_acc_lu
  LA r6,@mth_pair_k16_acc_ult
  LA r7,@mth_pair_k16_scratch
  CALL @mth_graph_pair_tile16_k16_accum_f32
  MOVI r8,1
  CMP r1,r8
  JMP[NE] @mth_pair_k16_fail

  ; Both independent partial sums are now 4.0 at local [1,2].
  LA r4,@mth_pair_k16_acc_lu
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x4080
  CMP r7,r8
  JMP[NE] @mth_pair_k16_fail
  LA r4,@mth_pair_k16_acc_ult
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  CMP r7,r8
  JMP[NE] @mth_pair_k16_fail

  ; Finalize and verify the complete 16x16 output tile: Y=4U.
  LA r1,@mth_pair_k16_acc_lu
  LA r2,@mth_pair_k16_acc_ult
  LA r3,@mth_pair_k16_y
  CALL @mth_graph_pair_tile16_finalize_f32
  MOVI r8,1
  CMP r1,r8
  JMP[NE] @mth_pair_k16_fail

  LA r4,@mth_pair_k16_y
  MOVI r5,0
mth_pair_k16_check:
  LD r6,r4,0
  MOVI r7,18
  CMP r5,r7
  JMP[EQ] @mth_pair_k16_expect_eight
  CMP r6,r0
  JMP[NE] @mth_pair_k16_fail
  JMP @mth_pair_k16_check_next
mth_pair_k16_expect_eight:
  MOVI r7,0
  MOVHI r7,0x4100
  CMP r6,r7
  JMP[NE] @mth_pair_k16_fail
mth_pair_k16_check_next:
  ADD r4,r4,r13
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_pair_k16_check

  LA r1,@mth_pair_k16_pass
  CALL @mth_pair_k16_puts
  HALT

mth_pair_k16_fail:
  LA r1,@mth_pair_k16_bad
  CALL @mth_pair_k16_puts
  HALT

mth_pair_k16_puts:
  MOVI r4,255
mth_pair_k16_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pair_k16_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pair_k16_puts_byte
  ADD r1,r1,r13
  JMP @mth_pair_k16_puts_word

.org 512
mth_pair_k16_li:
.word 0
.org 768
mth_pair_k16_ukj:
.word 0
.org 1024
mth_pair_k16_uik:
.word 0
.org 1280
mth_pair_k16_lj:
.word 0
.org 1536
mth_pair_k16_acc_lu:
.word 0
.org 1792
mth_pair_k16_acc_ult:
.word 0
.org 2048
mth_pair_k16_y:
.word 0
.org 2304
mth_pair_k16_scratch:
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.org 2320
mth_pair_k16_pass:
.str "MTH PAIR TILE16 K16 ACCUM PASS\n"
mth_pair_k16_bad:
.str "MTH PAIR TILE16 K16 ACCUM FAIL\n"
