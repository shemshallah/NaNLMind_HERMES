.org 0
JMP @mth_pair_tile_test_main
.include "../nan16/mth_graph_pair_tile16_f32.nas"

mth_pair_tile_test_main:
  MOVI r13,1
  LA r4,@mth_pair_tile_li
  MOVI r5,0
  MOVI r6,0
  MOVHI r6,0x4000
mth_pair_tile_fill_li:
  ADD r7,r5,r0
  SHL r7,r7,8
  ADD r7,r7,r5
  ADD r7,r4,r7
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r8,16
  CMP r5,r8
  JMP[LT] @mth_pair_tile_fill_li
  LA r4,@mth_pair_tile_lj
  MOVI r5,0
mth_pair_tile_fill_lj:
  ADD r7,r5,r0
  SHL r7,r7,8
  ADD r7,r7,r5
  ADD r7,r4,r7
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r8,16
  CMP r5,r8
  JMP[LT] @mth_pair_tile_fill_lj
  LA r4,@mth_pair_tile_u
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,0
  MOVI r5,258
  ADD r4,r4,r5
  MOVI r5,0
  MOVHI r5,0x4000
  ST r5,r4,0
  LA r1,@mth_pair_tile_u
  LA r2,@mth_pair_tile_li
  LA r3,@mth_pair_tile_lj
  LA r4,@mth_pair_tile_y
  MOVI r5,0
  MOVI r6,0
  LA r7,@mth_pair_tile_scratch
  CALL @mth_graph_pair_tile16_f32
  MOVI r8,1
  CMP r1,r8
  JMP[NE] @mth_pair_tile_fail
  LA r4,@mth_pair_tile_y
  MOVI r5,0
mth_pair_tile_check:
  LD r6,r4,0
  CMP r5,r0
  JMP[EQ] @mth_pair_tile_expect_four
  MOVI r7,18
  CMP r5,r7
  JMP[EQ] @mth_pair_tile_expect_eight
  CMP r6,r0
  JMP[NE] @mth_pair_tile_fail
  JMP @mth_pair_tile_check_next
mth_pair_tile_expect_four:
  MOVI r7,0
  MOVHI r7,0x4080
  CMP r6,r7
  JMP[NE] @mth_pair_tile_fail
  JMP @mth_pair_tile_check_next
mth_pair_tile_expect_eight:
  MOVI r7,0
  MOVHI r7,0x4100
  CMP r6,r7
  JMP[NE] @mth_pair_tile_fail
mth_pair_tile_check_next:
  ADD r4,r4,r13
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_pair_tile_check

  ; Verify nonzero global tile bases: i_base=16, j_base=32.
  ; Rebuild the two 16-row windows as rows 16..31 and 32..47 of L=2I.
  LA r4,@mth_pair_tile_li
  MOVI r5,4096
  MOVI r6,0
mth_pair_tile_clear_li:
  ST r6,r4,0
  ADD r4,r4,r13
  SUB r5,r5,r13
  CMP r5,r0
  JMP[GT] @mth_pair_tile_clear_li

  LA r4,@mth_pair_tile_lj
  MOVI r5,4096
mth_pair_tile_clear_lj:
  ST r6,r4,0
  ADD r4,r4,r13
  SUB r5,r5,r13
  CMP r5,r0
  JMP[GT] @mth_pair_tile_clear_lj

  LA r4,@mth_pair_tile_li
  MOVI r5,0
  MOVI r6,0
  MOVHI r6,0x4000
mth_pair_tile_fill_li_offset:
  ADD r7,r5,r0
  SHL r7,r7,8
  ADD r7,r7,r5
  MOVI r8,16
  ADD r7,r7,r8
  ADD r7,r4,r7
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r8,16
  CMP r5,r8
  JMP[LT] @mth_pair_tile_fill_li_offset

  LA r4,@mth_pair_tile_lj
  MOVI r5,0
mth_pair_tile_fill_lj_offset:
  ADD r7,r5,r0
  SHL r7,r7,8
  ADD r7,r7,r5
  MOVI r8,32
  ADD r7,r7,r8
  ADD r7,r4,r7
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r8,16
  CMP r5,r8
  JMP[LT] @mth_pair_tile_fill_lj_offset

  ; Add U[17,34]=2.0 at full-matrix word offset 4386.
  LA r4,@mth_pair_tile_u
  MOVI r5,4386
  ADD r4,r4,r5
  MOVI r5,0
  MOVHI r5,0x4000
  ST r5,r4,0

  LA r1,@mth_pair_tile_u
  LA r2,@mth_pair_tile_li
  LA r3,@mth_pair_tile_lj
  LA r4,@mth_pair_tile_y
  MOVI r5,16
  MOVI r6,32
  LA r7,@mth_pair_tile_scratch
  CALL @mth_graph_pair_tile16_f32
  MOVI r8,1
  CMP r1,r8
  JMP[NE] @mth_pair_tile_fail

  ; The second tile also satisfies Y=4U: only local [1,2] is 8.0.
  LA r4,@mth_pair_tile_y
  MOVI r5,0
mth_pair_tile_check_offset:
  LD r6,r4,0
  MOVI r7,18
  CMP r5,r7
  JMP[EQ] @mth_pair_tile_expect_offset_eight
  CMP r6,r0
  JMP[NE] @mth_pair_tile_fail
  JMP @mth_pair_tile_check_offset_next
mth_pair_tile_expect_offset_eight:
  MOVI r7,0
  MOVHI r7,0x4100
  CMP r6,r7
  JMP[NE] @mth_pair_tile_fail
mth_pair_tile_check_offset_next:
  ADD r4,r4,r13
  ADD r5,r5,r13
  MOVI r7,256
  CMP r5,r7
  JMP[LT] @mth_pair_tile_check_offset

  LA r1,@mth_pair_tile_pass
  CALL @mth_pair_tile_puts
  HALT
mth_pair_tile_fail:
  LA r1,@mth_pair_tile_bad
  CALL @mth_pair_tile_puts
  HALT
mth_pair_tile_puts:
  MOVI r4,255
mth_pair_tile_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pair_tile_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pair_tile_puts_byte
  ADD r1,r1,r13
  JMP @mth_pair_tile_puts_word
.org 512
mth_pair_tile_u:
.word 0
.org 66048
mth_pair_tile_li:
.word 0
.org 70144
mth_pair_tile_lj:
.word 0
.org 74240
mth_pair_tile_y:
.word 0
.org 74496
mth_pair_tile_scratch:
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.org 74512
mth_pair_tile_pass:
.str "MTH PAIR TILE16 F32 PASS\n"
mth_pair_tile_bad:
.str "MTH PAIR TILE16 F32 FAIL\n"
