.org 0
JMP @mth_pair_test_main
.include "../nan16/mth_graph_pair.nas"

; Fixture for the full dense 128x128 pair-space action.
;
; L = 2*I is built programmatically: zero all 16,384 words, then write
; 2.0 at the 128 diagonal positions. U = 0 except for five 1.0 entries.
; With L = 2I:
;   U' = L U + U L^T = 2U + 2U = 4U
; so the five seeded entries must become 4.0 and every other entry 0.0.
; The fixture then zeros U and checks that the output is all zero.
;
; Memory: L at 1024, U at 17408, output at 33792. Each is 16,384 words;
; the last word is 50,175, so the three-array footprint is 49,152 words.

mth_pair_test_main:
  MOVI r13,1
  MOVI r0,0

  ; ---- L = 0 (16,384 words) ----
  MOVI r1,@mth_pair_L
  MOVI r2,16384
  MOVI r3,0
mth_pair_zero_l:
  ST r3,r1,0
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_pair_zero_l

  ; ---- L = 2*I: diagonal stride is 128+1 = 129 ----
  MOVI r1,@mth_pair_L
  MOVI r2,128
  MOVI r4,0
  MOVHI r4,0x4000           ; 2.0f
mth_pair_diag:
  ST r4,r1,0
  ADDI r1,r1,129
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_pair_diag

  ; ---- Spot-check the constructed L ----
  MOVI r1,@mth_pair_L
  LD r5,r1,0
  CMP r5,r4
  JMP[NE] @mth_pair_fail
  MOVI r6,1
  ADD r5,r1,r6
  LD r5,r5,0
  CMP r5,r0
  JMP[NE] @mth_pair_fail
  MOVI r6,129
  ADD r5,r1,r6
  LD r5,r5,0
  CMP r5,r4
  JMP[NE] @mth_pair_fail
  MOVI r6,16383
  ADD r5,r1,r6
  LD r5,r5,0
  CMP r5,r4
  JMP[NE] @mth_pair_fail

  ; ---- U = 0 (16,384 words) ----
  MOVI r1,@mth_pair_U
  MOVI r2,16384
  MOVI r3,0
mth_pair_zero_u:
  ST r3,r1,0
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_pair_zero_u

  ; ---- Seed five 1.0 entries: (0,0), (1,2), (17,5), (64,64), (127,127) ----
  MOVI r1,@mth_pair_U
  MOVI r4,0
  MOVHI r4,0x3F80           ; 1.0f
  ST r4,r1,0                ; offset 0
  MOVI r6,130
  ADD r5,r1,r6
  ST r4,r5,0                ; offset 1*128+2 = 130
  MOVI r6,2181
  ADD r5,r1,r6
  ST r4,r5,0                ; offset 17*128+5 = 2181
  MOVI r6,8256
  ADD r5,r1,r6
  ST r4,r5,0                ; offset 64*128+64 = 8256
  MOVI r6,16383
  ADD r5,r1,r6
  ST r4,r5,0                ; offset 127*128+127 = 16383

  ; ---- First action: U' = 4U ----
  MOVI r1,@mth_pair_L
  MOVI r2,@mth_pair_U
  MOVI r3,@mth_pair_OUT
  CALL @mth_graph_pair
  CMP r1,r13
  JMP[NE] @mth_pair_fail

  MOVI r5,@mth_pair_OUT
  MOVI r6,0
  MOVHI r6,0x4080           ; 4.0f

  LD r7,r5,0
  CMP r7,r6
  JMP[NE] @mth_pair_fail
  MOVI r8,130
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r6
  JMP[NE] @mth_pair_fail
  MOVI r8,2181
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r6
  JMP[NE] @mth_pair_fail
  MOVI r8,8256
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r6
  JMP[NE] @mth_pair_fail
  MOVI r8,16383
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r6
  JMP[NE] @mth_pair_fail

  ; Untouched entries must be +0.0f.
  MOVI r8,1
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r0
  JMP[NE] @mth_pair_fail
  MOVI r8,258
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r0
  JMP[NE] @mth_pair_fail
  MOVI r8,16256
  ADD r7,r5,r8
  LD r7,r7,0
  CMP r7,r0
  JMP[NE] @mth_pair_fail

  ; ---- Zero U; zero U must give zero output ----
  MOVI r1,@mth_pair_U
  MOVI r2,16384
  MOVI r3,0
mth_pair_zero_u2:
  ST r3,r1,0
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_pair_zero_u2

  MOVI r1,@mth_pair_L
  MOVI r2,@mth_pair_U
  MOVI r3,@mth_pair_OUT
  CALL @mth_graph_pair
  CMP r1,r13
  JMP[NE] @mth_pair_fail

  MOVI r1,@mth_pair_OUT
  MOVI r2,16384
mth_pair_check_zero:
  LD r3,r1,0
  CMP r3,r0
  JMP[NE] @mth_pair_fail
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_pair_check_zero

  MOVI r1,@mth_pair_pass
  CALL @mth_pair_puts
  HALT

mth_pair_fail:
  MOVI r1,@mth_pair_bad
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

; Three dense 128x128 float32 arrays: 16,384 words each.
.org 1024
mth_pair_L:
.word 0
.org 17408
mth_pair_U:
.word 0
.org 33792
mth_pair_OUT:
.word 77
.org 50176
mth_pair_pass:
.str "MTH GRAPH PAIR FULL 128 PASS\n"
mth_pair_bad:
.str "MTH GRAPH PAIR FULL 128 FAIL\n"
