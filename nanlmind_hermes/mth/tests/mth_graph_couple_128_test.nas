.org 0
JMP @mth_couple128_test_main
.include "../nan16/mth_graph_couple_128.nas"

mth_couple128_test_main:
  MOVI r13,1

  ; inner=1.0, outer=3.0, eta=0.5 -> both 2.0.
  MOVI r1,@mth_couple128_inner
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_couple128_fill_inner:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_couple128_fill_inner

  MOVI r2,@mth_couple128_outer
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x4040
mth_couple128_fill_outer:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_couple128_fill_outer

  MOVI r1,@mth_couple128_inner
  MOVI r2,@mth_couple128_outer
  MOVI r3,@mth_couple128_eta
  LD r3,r3,0
  MOVI r4,@mth_couple128_inner_out
  MOVI r5,@mth_couple128_outer_out
  CALL @mth_graph_couple_128_f32
  CMP r1,r13
  JMP[NE] @mth_couple128_fail

  MOVI r6,@mth_couple128_inner_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4000
  CMP r7,r8
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_outer_out
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @mth_couple128_fail

  ; eta=0 -> inner=1, outer=3.
  MOVI r3,@mth_couple128_eta
  ST r0,r3,0
  LD r3,r3,0
  MOVI r1,@mth_couple128_inner
  MOVI r2,@mth_couple128_outer
  MOVI r4,@mth_couple128_inner_out
  MOVI r5,@mth_couple128_outer_out
  CALL @mth_graph_couple_128_f32
  CMP r1,r13
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_inner_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  CMP r7,r8
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_outer_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4040
  CMP r7,r8
  JMP[NE] @mth_couple128_fail

  ; eta=1 -> inner=3, outer=1.
  MOVI r3,@mth_couple128_eta
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r3,0
  LD r3,r3,0
  MOVI r1,@mth_couple128_inner
  MOVI r2,@mth_couple128_outer
  MOVI r4,@mth_couple128_inner_out
  MOVI r5,@mth_couple128_outer_out
  CALL @mth_graph_couple_128_f32
  CMP r1,r13
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_inner_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4040
  CMP r7,r8
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_outer_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  CMP r7,r8
  JMP[NE] @mth_couple128_fail

  ; Invalid eta=1.5 rejects and preserves outputs.
  MOVI r3,@mth_couple128_eta
  MOVI r7,0
  MOVHI r7,0x3FC0
  ST r7,r3,0
  LD r3,r3,0
  MOVI r1,@mth_couple128_inner
  MOVI r2,@mth_couple128_outer
  MOVI r4,@mth_couple128_inner_out
  MOVI r5,@mth_couple128_outer_out
  MOVI r6,@mth_couple128_inner_out
  MOVI r7,77
  ST r7,r6,0
  CALL @mth_graph_couple_128_f32
  CMP r1,r0
  JMP[NE] @mth_couple128_fail
  MOVI r6,@mth_couple128_inner_out
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_couple128_fail

  MOVI r1,@mth_couple128_pass
  CALL @mth_couple128_puts
  HALT

mth_couple128_fail:
  MOVI r1,@mth_couple128_bad
  CALL @mth_couple128_puts
  HALT

mth_couple128_puts:
  MOVI r4,255
mth_couple128_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_couple128_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_couple128_puts_byte
  ADD r1,r1,r13
  JMP @mth_couple128_puts_word

.org 512
mth_couple128_inner:
.word 0
.org 1024
mth_couple128_outer:
.word 0
.org 1536
mth_couple128_inner_out:
.word 0
.org 2048
mth_couple128_outer_out:
.word 0
.org 2560
mth_couple128_eta:
.word 0x3F000000
.org 3072
mth_couple128_pass:
.str "MTH GRAPH COUPLE 128 PASS\n"
mth_couple128_bad:
.str "MTH GRAPH COUPLE 128 FAIL\n"
