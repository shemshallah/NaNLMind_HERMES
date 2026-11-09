.org 0
JMP @mth_step_energy_test_main
.include "../nan16/mth_graph_step_energy_256.nas"

mth_step_energy_test_main:
  MOVI r13,1

  ; u=1, v=2, a=-3 for all 256:
  ; kinetic=0.5*256*4=512; potential=0.5*256*3=384; total=896.
  MOVI r1,@mth_step_energy_u
  MOVI r2,@mth_step_energy_v
  MOVI r3,@mth_step_energy_a
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_step_energy_u_fill:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_step_energy_u_fill

  MOVI r2,@mth_step_energy_v
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_step_energy_v_fill:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_step_energy_v_fill

  MOVI r3,@mth_step_energy_a
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0xC040
mth_step_energy_a_fill:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_step_energy_a_fill

  MOVI r1,@mth_step_energy_u
  MOVI r2,@mth_step_energy_v
  MOVI r3,@mth_step_energy_a
  MOVI r4,@mth_step_energy_out
  CALL @mth_graph_step_energy_256_f32
  CMP r1,r13
  JMP[NE] @mth_step_energy_fail

  MOVI r5,@mth_step_energy_out
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x4400
  CMP r6,r7
  JMP[NE] @mth_step_energy_fail
  ADD r5,r5,r13
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x43C0
  CMP r6,r7
  JMP[NE] @mth_step_energy_fail
  ADD r5,r5,r13
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x4460
  CMP r6,r7
  JMP[NE] @mth_step_energy_fail

  MOVI r1,@mth_step_energy_pass
  CALL @mth_step_energy_puts
  HALT

mth_step_energy_fail:
  MOVI r1,@mth_step_energy_bad
  CALL @mth_step_energy_puts
  HALT

mth_step_energy_puts:
  MOVI r4,255
mth_step_energy_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_step_energy_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_step_energy_puts_byte
  ADD r1,r1,r13
  JMP @mth_step_energy_puts_word

.org 512
mth_step_energy_u:
.word 0
.org 1024
mth_step_energy_v:
.word 0
.org 1536
mth_step_energy_a:
.word 0
.org 2048
mth_step_energy_out:
.word 77
.org 2560
mth_step_energy_pass:
.str "MTH GRAPH STEP ENERGY PASS\n"
mth_step_energy_bad:
.str "MTH GRAPH STEP ENERGY FAIL\n"
