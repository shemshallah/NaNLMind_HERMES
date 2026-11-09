; NaN16 full 16-slice driver for one 16x16 pair-space output tile.
;
; ABI:
;   r1..r4 = four strip bases, each 16 contiguous 16x16 K blocks in increasing
;            global-K order (K=0,16,...,240). Each strip is 4096 words.
;   r5=acc_LU (256), r6=acc_ULt (256), r7=6-word K16 scratch
;   r8=output tile (256), r13=1
; Returns r1=1 after all sixteen K blocks and the single finalize succeed;
; returns r1=0 on the first callee failure.
;
; The driver calls mth_graph_pair_tile16_k16_accum_f32 for K=0,16,...,240,
; then calls mth_graph_pair_tile16_finalize_f32 exactly once on the saved r8
; output tile.
; The caller must link mth_graph_pair_tile16_k16_f32.nas (which supplies both
; callees) into the same image.
;
; r15 is the NaN16 link register. Every nested CALL overwrites it, so the
; driver saves its own return link in the private static frame and restores it
; before RET on both the success and failure paths. r14 is saved as well so a
; caller may treat it as callee-saved. r12 is recomputed after every callee
; call; it is scratch, not state.
;
; The driver is not reentrant: the state frame below is private static guest
; scratch. A caller embedding it must keep code and data out of words
; 400..411 (12 words).

mth_graph_pair_tile16_k16_driver_f32:
  LA r12,@mth_pair_driver_state
  ST r1,r12,0              ; L_IK strip base
  ST r2,r12,1              ; U_KJ strip base
  ST r3,r12,2              ; U_IK strip base
  ST r4,r12,3              ; L_JK strip base
  ST r5,r12,4              ; acc_LU
  ST r6,r12,5              ; acc_ULt
  ST r7,r12,6              ; K16 scratch
  ST r8,r12,7              ; output tile
  ST r0,r12,8              ; K offset in words: 0,256,...,3840
  ST r0,r12,9              ; completed K-slice count: 0..16
  ST r14,r12,10
  ST r15,r12,11            ; caller return link

mth_pair_driver_slice:
  ; Build the four input pointers for the current 256-word K block.
  LA r12,@mth_pair_driver_state
  LD r1,r12,0
  LD r2,r12,8
  ADD r1,r1,r2
  LA r12,@mth_pair_driver_state
  LD r2,r12,1
  LD r3,r12,8
  ADD r2,r2,r3
  LA r12,@mth_pair_driver_state
  LD r3,r12,2
  LD r4,r12,8
  ADD r3,r3,r4
  LA r12,@mth_pair_driver_state
  LD r4,r12,3
  LD r9,r12,8
  ADD r4,r4,r9
  LA r12,@mth_pair_driver_state
  LD r5,r12,4
  LD r6,r12,5
  LD r7,r12,6
  MOVI r13,1
  CALL @mth_graph_pair_tile16_k16_accum_f32
  MOVI r10,1
  CMP r1,r10
  JMP[NE] @mth_pair_driver_fail

  ; Advance to the next K slice. There are exactly sixteen.
  LA r12,@mth_pair_driver_state
  LD r1,r12,8
  MOVI r2,256
  ADD r1,r1,r2
  ST r1,r12,8
  LD r1,r12,9
  MOVI r2,1
  ADD r1,r1,r2
  ST r1,r12,9
  MOVI r2,16
  CMP r1,r2
  JMP[LT] @mth_pair_driver_slice

  ; Finalize once, only after all sixteen slices returned successfully.
  LA r12,@mth_pair_driver_state
  LD r1,r12,4
  LD r2,r12,5
  LD r3,r12,7
  MOVI r13,1
  CALL @mth_graph_pair_tile16_finalize_f32
  MOVI r2,1
  CMP r1,r2
  JMP[NE] @mth_pair_driver_fail
  MOVI r1,1
  JMP @mth_pair_driver_return

mth_pair_driver_fail:
  MOVI r1,0

mth_pair_driver_return:
  LA r12,@mth_pair_driver_state
  LD r14,r12,10
  LD r15,r12,11
  RET

; Private 12-word frame; r12 is recomputed after each callee call.
; 0..3 strips, 4..5 accumulators, 6 scratch, 7 output,
; 8 K word offset, 9 slice count, 10 saved r14, 11 saved r15.
.org 400
mth_pair_driver_state:
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
