.org 0
JMP @mth_transmission_test_main
.include "../nan16/mth_graph_transmission_128.nas"

mth_transmission_test_main:
  MOVI r13,1

  ; inner=outer=1.0 -> E_in=E_out=128, T=0.5.
  MOVI r1,@mth_transmission_inner
  MOVI r2,@mth_transmission_outer
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_transmission_fill:
  ST r7,r1,0
  ST r7,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_transmission_fill

  MOVI r1,@mth_transmission_inner
  MOVI r2,@mth_transmission_outer
  MOVI r3,@mth_transmission_out
  CALL @mth_graph_transmission_128_f32
  CMP r1,r13
  JMP[NE] @mth_transmission_fail

  MOVI r4,@mth_transmission_out
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4300
  CMP r5,r6
  JMP[NE] @mth_transmission_fail
  ADD r4,r4,r13
  LD r5,r4,0
  CMP r5,r6
  JMP[NE] @mth_transmission_fail
  ADD r4,r4,r13
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F00
  CMP r5,r6
  JMP[NE] @mth_transmission_fail

  ; outer=0 -> T=0.
  MOVI r2,@mth_transmission_outer
  MOVI r6,128
mth_transmission_zero_outer:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_transmission_zero_outer

  MOVI r1,@mth_transmission_inner
  MOVI r2,@mth_transmission_outer
  MOVI r3,@mth_transmission_out
  CALL @mth_graph_transmission_128_f32
  CMP r1,r13
  JMP[NE] @mth_transmission_fail
  MOVI r4,@mth_transmission_out
  MOVI r6,2
  ADD r4,r4,r6
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_transmission_fail

  ; both zero -> T=0.
  MOVI r1,@mth_transmission_inner
  MOVI r6,128
mth_transmission_zero_inner:
  ST r0,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_transmission_zero_inner

  MOVI r1,@mth_transmission_inner
  MOVI r2,@mth_transmission_outer
  MOVI r3,@mth_transmission_out
  CALL @mth_graph_transmission_128_f32
  CMP r1,r13
  JMP[NE] @mth_transmission_fail
  MOVI r4,@mth_transmission_out
  MOVI r6,2
  ADD r4,r4,r6
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_transmission_fail

  MOVI r1,@mth_transmission_pass
  CALL @mth_transmission_puts
  HALT

mth_transmission_fail:
  MOVI r1,@mth_transmission_bad
  CALL @mth_transmission_puts
  HALT

mth_transmission_puts:
  MOVI r4,255
mth_transmission_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_transmission_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_transmission_puts_byte
  ADD r1,r1,r13
  JMP @mth_transmission_puts_word

.org 512
mth_transmission_inner:
.word 0
.org 1024
mth_transmission_outer:
.word 0
.org 1536
mth_transmission_out:
.word 77
.org 2048
mth_transmission_pass:
.str "MTH GRAPH TRANSMISSION 128 PASS\n"
mth_transmission_bad:
.str "MTH GRAPH TRANSMISSION 128 FAIL\n"
