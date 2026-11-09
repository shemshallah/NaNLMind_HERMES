.org 0
JMP @mth_pair128_test_main
.include "../nan16/mth_graph_pair_rank1_128.nas"

mth_pair128_test_main:
  MOVI r13,1

  ; u[0]=1.0, all other u=0; v=1.0 everywhere.
  MOVI r1,@mth_pair128_u
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r1,0

  MOVI r2,@mth_pair128_v
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_pair128_fill_v:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_pair128_fill_v

  MOVI r1,@mth_pair128_u
  MOVI r2,@mth_pair128_v
  MOVI r3,@mth_pair128_out
  MOVI r4,@mth_pair128_scratch
  CALL @mth_graph_pair_rank1_128
  CMP r1,r13
  JMP[NE] @mth_pair128_fail

  MOVI r5,@mth_pair128_out

  ; U[0,0] = 2.0, U[0,1] = 2.0.
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x4000
  CMP r6,r7
  JMP[NE] @mth_pair128_fail
  ADD r8,r5,r13
  LD r6,r8,0
  CMP r6,r7
  JMP[NE] @mth_pair128_fail

  ; U[1,0] = -1.0.
  MOVI r8,128
  ADD r8,r5,r8
  LD r6,r8,0
  MOVI r7,0
  MOVHI r7,0xBF80
  CMP r6,r7
  JMP[NE] @mth_pair128_fail

  ; U[127,0] = -1.0 and U[127,1] = -1.0.
  MOVI r8,16256
  ADD r8,r5,r8
  LD r6,r8,0
  CMP r6,r7
  JMP[NE] @mth_pair128_fail
  ADD r8,r8,r13
  LD r6,r8,0
  CMP r6,r7
  JMP[NE] @mth_pair128_fail

  ; U[2,0] = 0.0.
  MOVI r8,256
  ADD r8,r5,r8
  LD r6,r8,0
  CMP r6,r0
  JMP[NE] @mth_pair128_fail

  MOVI r1,@mth_pair128_pass
  CALL @mth_pair128_puts
  HALT

mth_pair128_fail:
  MOVI r1,@mth_pair128_bad
  CALL @mth_pair128_puts
  HALT

mth_pair128_puts:
  MOVI r4,255
mth_pair128_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pair128_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pair128_puts_byte
  ADD r1,r1,r13
  JMP @mth_pair128_puts_word

.org 512
mth_pair128_u:
.word 0
.org 1024
mth_pair128_v:
.word 0
.org 4096
mth_pair128_out:
.word 77
.org 20480
mth_pair128_scratch:
.word 0
.org 20736
mth_pair128_pass:
.str "MTH GRAPH PAIR RANK1 128 PASS\n"
mth_pair128_bad:
.str "MTH GRAPH PAIR RANK1 128 FAIL\n"
