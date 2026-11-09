.org 0
JMP @mth_pair_driver_test_main
.include "../nan16/mth_graph_pair_tile16_k16_f32.nas"
.include "../nan16/mth_graph_pair_tile16_k16_driver_f32.nas"

; Two independent tile patterns exercise the full K=0,16,...,240 loop.
; Pattern 1: slice 0 gives acc_LU[1,2] += 2*2, slice 15 gives
;            acc_ULt[1,2] += 2*2, so Y1[1,2] = 4 + 4 = 8.
; Pattern 2: slice 0 gives acc_LU[2,5] += 1*2, slice 15 gives
;            acc_ULt[2,5] += 2*3, so Y2[2,5] = 2 + 6 = 8.
mth_pair_driver_test_main:
  MOVI r13,1

  ; ---- Pattern 1, slice 0: L[1,3]=2, U[3,2]=2 ----
  MOVI r5,0
  MOVHI r5,0x4000
  LA r4,@mth_driver_li
  MOVI r6,19
  ADD r4,r4,r6
  ST r5,r4,0
  LA r4,@mth_driver_ukj
  MOVI r6,50
  ADD r4,r4,r6
  ST r5,r4,0

  ; Verify the staged source word, then fill slice 15 from the static image.
  LA r4,@mth_driver_li
  MOVI r6,19
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x4000
  CMP r7,r8
  JMP[NE] @mth_pair_driver_test_fail

  ; ---- Pattern 1, slice 15: U[1,5]=2, L[2,5]=2 (static .word below) ----
  LA r1,@mth_driver_li
  LA r2,@mth_driver_ukj
  LA r3,@mth_driver_uik
  LA r4,@mth_driver_ljk
  LA r5,@mth_driver_acc_lu
  LA r6,@mth_driver_acc_ult
  LA r7,@mth_driver_scratch
  LA r8,@mth_driver_y1
  CALL @mth_graph_pair_tile16_k16_driver_f32
  MOVI r2,1
  CMP r1,r2
  JMP[NE] @mth_pair_driver_test_fail

  ; Both partial sums are 4.0 at local [1,2] after all sixteen slices.
  LA r4,@mth_driver_acc_lu
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x4080
  CMP r7,r8
  JMP[NE] @mth_pair_driver_test_fail
  LA r4,@mth_driver_acc_ult
  MOVI r6,18
  ADD r4,r4,r6
  LD r7,r4,0
  CMP r7,r8
  JMP[NE] @mth_pair_driver_test_fail

  ; Y1[1,2] must be 8.0 and every other cell must be zero.
  LA r1,@mth_driver_y1
  MOVI r2,18
  MOVI r3,0
  MOVHI r3,0x4100
  CALL @mth_pair_driver_check_tile
  MOVI r2,1
  CMP r1,r2
  JMP[NE] @mth_pair_driver_test_fail

  ; ---- Pattern 2: clear the four pattern-1 products ----
  LA r4,@mth_driver_li
  MOVI r6,19
  ADD r4,r4,r6
  ST r0,r4,0
  LA r4,@mth_driver_ukj
  MOVI r6,50
  ADD r4,r4,r6
  ST r0,r4,0
  LA r4,@mth_driver_uik
  MOVI r6,3861
  ADD r4,r4,r6
  ST r0,r4,0
  LA r4,@mth_driver_ljk
  MOVI r6,3877
  ADD r4,r4,r6
  ST r0,r4,0

  ; Slice 0: L[2,4]=1, U[4,5]=2.
  LA r4,@mth_driver_li
  MOVI r6,36
  ADD r4,r4,r6
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r4,0
  LA r4,@mth_driver_ukj
  MOVI r6,69
  ADD r4,r4,r6
  MOVI r5,0
  MOVHI r5,0x4000
  ST r5,r4,0
  ; Slice 15: U[2,7]=2, L[5,7]=3.
  LA r4,@mth_driver_uik
  MOVI r6,3879
  ADD r4,r4,r6
  ST r5,r4,0
  LA r4,@mth_driver_ljk
  MOVI r6,3927
  ADD r4,r4,r6
  MOVI r7,0
  MOVHI r7,0x4040
  ST r7,r4,0

  ; Fresh accumulators for the second tile.
  LA r1,@mth_driver_acc_lu
  CALL @mth_pair_driver_zero256
  LA r1,@mth_driver_acc_ult
  CALL @mth_pair_driver_zero256

  LA r1,@mth_driver_li
  LA r2,@mth_driver_ukj
  LA r3,@mth_driver_uik
  LA r4,@mth_driver_ljk
  LA r5,@mth_driver_acc_lu
  LA r6,@mth_driver_acc_ult
  LA r7,@mth_driver_scratch
  LA r8,@mth_driver_y2
  CALL @mth_graph_pair_tile16_k16_driver_f32
  MOVI r2,1
  CMP r1,r2
  JMP[NE] @mth_pair_driver_test_fail

  ; Pattern 2 partial sums: acc_LU[2,5]=2.0, acc_ULt[2,5]=6.0.
  LA r4,@mth_driver_acc_lu
  MOVI r6,37
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x4000
  CMP r7,r8
  JMP[NE] @mth_pair_driver_test_fail
  LA r4,@mth_driver_acc_ult
  MOVI r6,37
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,0
  MOVHI r8,0x40C0
  CMP r7,r8
  JMP[NE] @mth_pair_driver_test_fail

  ; Y2[2,5] must be 8.0 and every other cell must be zero.
  LA r1,@mth_driver_y2
  MOVI r2,37
  MOVI r3,0
  MOVHI r3,0x4100
  CALL @mth_pair_driver_check_tile
  MOVI r2,1
  CMP r1,r2
  JMP[NE] @mth_pair_driver_test_fail

  LA r1,@mth_pair_driver_pass
  CALL @mth_pair_driver_puts
  HALT

mth_pair_driver_test_fail:
  LA r1,@mth_pair_driver_bad
  CALL @mth_pair_driver_puts
  HALT

; Check one 16x16 tile: r1=base, r2=nonzero index, r3=expected value.
; Returns r1=1 when every other word is zero and base[r2] == expected,
; else r1=0. r13 must be 1.
mth_pair_driver_check_tile:
  MOVI r4,0
mth_pair_driver_check_loop:
  LD r5,r1,0
  CMP r4,r2
  JMP[EQ] @mth_pair_driver_check_hit
  CMP r5,r0
  JMP[NE] @mth_pair_driver_check_bad
  JMP @mth_pair_driver_check_next
mth_pair_driver_check_hit:
  CMP r5,r3
  JMP[NE] @mth_pair_driver_check_bad
mth_pair_driver_check_next:
  ADD r1,r1,r13
  ADD r4,r4,r13
  MOVI r6,256
  CMP r4,r6
  JMP[LT] @mth_pair_driver_check_loop
  MOVI r1,1
  RET
mth_pair_driver_check_bad:
  MOVI r1,0
  RET

; Zero one 256-word accumulator: r1=base, r13=1.
mth_pair_driver_zero256:
  MOVI r2,0
  MOVI r4,256
mth_pair_driver_zero_loop:
  ST r2,r1,0
  ADD r1,r1,r13
  SUB r4,r4,r13
  JMP[NE] @mth_pair_driver_zero_loop
  RET

mth_pair_driver_puts:
  MOVI r4,255
mth_pair_driver_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pair_driver_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pair_driver_puts_byte
  ADD r1,r1,r13
  JMP @mth_pair_driver_puts_word

; Four 16-slice strips = 4 * 4096 words.
.org 1024
mth_driver_li:
.org 1043
.word 0x40000000
.org 5120
mth_driver_ukj:
.org 5170
.word 0x40000000
.org 9216
mth_driver_uik:
.org 13077
.word 0x40000000
.org 13312
mth_driver_ljk:
.org 17189
.word 0x40000000
.org 17408
mth_driver_acc_lu:
.word 0
.org 17664
mth_driver_acc_ult:
.word 0
.org 17920
mth_driver_y1:
.word 0
.org 18176
mth_driver_y2:
.word 0
.org 18432
mth_driver_scratch:
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.org 18512
mth_pair_driver_pass:
.str "MTH PAIR K16 DRIVER PASS\n"
mth_pair_driver_bad:
.str "MTH PAIR K16 DRIVER FAIL\n"
