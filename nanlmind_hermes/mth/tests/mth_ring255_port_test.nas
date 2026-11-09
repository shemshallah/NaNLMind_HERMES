.org 0
JMP @mth_ring255_port_test_main
.include "../nan16/mth_ring255_port.nas"

mth_ring255_port_test_main:
  MOVI r13,1

  ; Fill heads with 1.0; port = 7.0.
  MOVI r1,@mth_ring255_port_heads
  MOVI r6,255
  MOVI r7,0
  MOVHI r7,0x3F80
mth_ring255_port_fill:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_ring255_port_fill

  MOVI r1,@mth_ring255_port_heads
  MOVI r2,@mth_ring255_port_scalar
  MOVI r3,@mth_ring255_port_readout
  CALL @mth_ring255_port_assemble_f32
  CMP r1,r13
  JMP[NE] @mth_ring255_port_fail

  MOVI r4,@mth_ring255_port_readout
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_ring255_port_fail

  MOVI r6,254
  ADD r6,r4,r6
  LD r5,r6,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_ring255_port_fail

  MOVI r6,255
  ADD r6,r4,r6
  LD r5,r6,0
  MOVI r6,0
  MOVHI r6,0x40E0
  CMP r5,r6
  JMP[NE] @mth_ring255_port_fail

  MOVI r1,@mth_ring255_port_pass
  CALL @mth_ring255_port_puts
  HALT

mth_ring255_port_fail:
  MOVI r1,@mth_ring255_port_bad
  CALL @mth_ring255_port_puts
  HALT

mth_ring255_port_puts:
  MOVI r4,255
mth_ring255_port_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_ring255_port_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_ring255_port_puts_byte
  ADD r1,r1,r13
  JMP @mth_ring255_port_puts_word

.org 512
mth_ring255_port_heads:
.word 0
.org 1024
mth_ring255_port_scalar:
.word 0x40E00000
.org 1536
mth_ring255_port_readout:
.word 77
.org 2560
mth_ring255_port_pass:
.str "MTH RING255 PORT ASSEMBLE PASS\n"
mth_ring255_port_bad:
.str "MTH RING255 PORT ASSEMBLE FAIL\n"
