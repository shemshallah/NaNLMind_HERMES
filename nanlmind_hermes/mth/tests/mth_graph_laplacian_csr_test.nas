; MTH canonical-CSR graph Laplacian native fixture — NaN16 only.
;
; Cases:
;   1. Unit-weight path 0--1--2--3, x=[0,1,4,9] Q16.16:
;      Lx=[-1,-2,-2,+5].
;   2. Constant x=1 on the same path: Lx=0 (constant nullspace).
;   3. Disconnected graph (0--1)+(2--3): Lx=[-1,+1,-5,+5].
;   4. Triangle 0--1--2, x=[0,1,4]: Lx=[-5,-2,+7,0].
;   5. Empty graph: zero output.
;   6. Out-of-range state rejects before changing output.
;   7. Weight 2 rejects (this is the unweighted operator).
;   8. Zero weight rejects.
;   9. Weight word with nonzero high 24 bits rejects.
;  10. Duplicate endpoint rejects.
;  11. Self-loop rejects.
;  12. Neighbor word with nonzero high 24 bits rejects.
;  13. Nonmonotone offsets and nonzero offsets[0] reject.
;  14. Missing reciprocal edge (asymmetric CSR) rejects.
;  15. edge_count above 65,280 rejects.
;
; Every rejection case pre-fills output with a sentinel and checks both the
; first and last output words are unchanged.

.org 0
JMP @mth_lap_csr_test_main
.include "../nan16/mth_graph_laplacian_csr.nas"

mth_lap_csr_test_main:
  MOVI r13,1

  ; ---- Case 1: unit-weight path -----------------------------------------
  MOVI r1,@mth_lap_csr_msg_path
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,6
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r13
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  MOVI r2,@mth_lap_csr_expected_path
  CALL @mth_lap_csr_check4
  MOVI r1,@mth_lap_csr_out
  MOVI r2,4
  ADD r1,r1,r2
  CALL @mth_lap_csr_check_zero_tail
  MOVI r1,@mth_lap_csr_msg_path_ok
  CALL @mth_lap_csr_puts

  ; ---- Case 2: constant vector nullspace --------------------------------
  MOVI r1,@mth_lap_csr_msg_const
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVHI r2,1                    ; 65536
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,6
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r13
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_all_zero
  MOVI r1,@mth_lap_csr_msg_const_ok
  CALL @mth_lap_csr_puts

  ; ---- Case 3: disconnected graph ---------------------------------------
  MOVI r1,@mth_lap_csr_msg_disc
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_disc
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,4
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r13
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  MOVI r2,@mth_lap_csr_expected_disc
  CALL @mth_lap_csr_check4
  MOVI r1,@mth_lap_csr_out
  MOVI r2,4
  ADD r1,r1,r2
  CALL @mth_lap_csr_check_zero_tail
  MOVI r1,@mth_lap_csr_msg_disc_ok
  CALL @mth_lap_csr_puts

  ; ---- Case 4: triangle -------------------------------------------------
  MOVI r1,@mth_lap_csr_msg_tri
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_triangle
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,6
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r13
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  MOVI r2,@mth_lap_csr_expected_triangle
  CALL @mth_lap_csr_check4
  MOVI r1,@mth_lap_csr_out
  MOVI r2,4
  ADD r1,r1,r2
  CALL @mth_lap_csr_check_zero_tail
  MOVI r1,@mth_lap_csr_msg_tri_ok
  CALL @mth_lap_csr_puts

  ; ---- Case 5: empty graph ----------------------------------------------
  MOVI r1,@mth_lap_csr_msg_empty
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_empty
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,0
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r13
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_all_zero
  MOVI r1,@mth_lap_csr_msg_empty_ok
  CALL @mth_lap_csr_puts

  ; ---- Case 6: out-of-range state ---------------------------------------
  MOVI r1,@mth_lap_csr_msg_bad_x
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_x
  MOVI r2,1
  MOVHI r2,0x0040                ; +2^22 + 1
  ST r2,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  CMP r1,r0
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 7: non-unit weight ------------------------------------------
  MOVI r1,@mth_lap_csr_msg_weight
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,2
  ST r2,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,4
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 8: zero weight ----------------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_weights
  ST r0,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,4
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 9: weight high bits -----------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,1
  MOVHI r2,0x0100                ; 0x01000001
  ST r2,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,4
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 10: duplicate endpoint --------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_neighbors
  ST r0,r1,2                    ; row 1 becomes [0,0]
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,3
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 11: self-loop ------------------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_neighbors
  MOVI r2,1
  ST r2,r1,1                    ; row 1 becomes [1,2]
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,3
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 12: neighbor high bits --------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_neighbors
  MOVI r2,1
  MOVHI r2,0x0100                ; 0x01000001
  ST r2,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,3
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 13: nonmonotone offsets -------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_offsets
  ST r0,r1,2                    ; offsets[1]=1, offsets[2]=0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,2
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 14: nonzero offsets[0] --------------------------------------
  CALL @mth_lap_csr_setup_path
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,1
  ST r2,r1,0
  CALL @mth_lap_csr_fill_sentinel
  CALL @mth_lap_csr_call_path
  MOVI r2,2
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 15: missing reciprocal edge ---------------------------------
  MOVI r1,@mth_lap_csr_msg_sym
  CALL @mth_lap_csr_puts
  CALL @mth_lap_csr_setup_asym
  CALL @mth_lap_csr_fill_sentinel
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,1
  CALL @mth_graph_laplacian_csr_q16_16
  MOVI r2,5
  CMP r1,r2
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  ; ---- Case 16: edge_count above limit ----------------------------------
  CALL @mth_lap_csr_setup_path
  CALL @mth_lap_csr_fill_sentinel
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,65281
  CALL @mth_graph_laplacian_csr_q16_16
  CMP r1,r0
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r1,@mth_lap_csr_out
  CALL @mth_lap_csr_check_unchanged

  MOVI r1,@mth_lap_csr_msg_pass
  CALL @mth_lap_csr_puts
  HALT

mth_lap_csr_test_fail:
  MOVI r1,@mth_lap_csr_msg_fail
  CALL @mth_lap_csr_puts
  HALT


; ---- fixture helpers (all leaf except setup_* wrappers) -----------------

; r1=base, r2=value, r3=count.  Clobbers r3 only.
mth_lap_csr_fill_count:
  CMP r3,r0
  JMP[EQ] @mth_lap_csr_fill_done
mth_lap_csr_fill_loop:
  ST r2,r1,0
  ADD r1,r1,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lap_csr_fill_loop
mth_lap_csr_fill_done:
  RET

; r1=base, r2=start index, r3=value.  Fills offsets[r2..256].
mth_lap_csr_fill_offset_tail:
  MOVI r4,257
mth_lap_csr_offset_tail_loop:
  CMP r2,r4
  JMP[GE] @mth_lap_csr_offset_tail_done
  ADD r5,r1,r2
  ST r3,r5,0
  ADD r2,r2,r13
  JMP @mth_lap_csr_offset_tail_loop
mth_lap_csr_offset_tail_done:
  RET

; r1=base.  Writes [1,0,2,1,3,2].
mth_lap_csr_write_neighbors_path:
  MOVI r2,1
  ST r2,r1,0
  ST r0,r1,1
  MOVI r2,2
  ST r2,r1,2
  MOVI r2,1
  ST r2,r1,3
  MOVI r2,3
  ST r2,r1,4
  MOVI r2,2
  ST r2,r1,5
  RET

; r1=base.  Writes [0,65536,262144,589824] into the first four words.
mth_lap_csr_write_path_x:
  MOVI r2,0
  MOVHI r2,1                    ; 65536
  ST r2,r1,1
  MOVI r2,0
  MOVHI r2,4                    ; 262144
  ST r2,r1,2
  MOVI r2,0
  MOVHI r2,9                    ; 589824
  ST r2,r1,3
  RET

mth_lap_csr_setup_path:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,5
  ST r2,r1,3
  MOVI r2,4
  MOVI r3,6
  CALL @mth_lap_csr_fill_offset_tail
  MOVI r1,@mth_lap_csr_neighbors
  CALL @mth_lap_csr_write_neighbors_path
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,1
  MOVI r3,6
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  CALL @mth_lap_csr_write_path_x
  ADD r15,r14,r0
  RET

mth_lap_csr_setup_disc:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,2
  ST r2,r1,2
  MOVI r2,3
  ST r2,r1,3
  MOVI r2,4
  ST r2,r1,4
  MOVI r2,5
  MOVI r3,4
  CALL @mth_lap_csr_fill_offset_tail
  MOVI r1,@mth_lap_csr_neighbors
  MOVI r2,1
  ST r2,r1,0
  ST r0,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,2
  ST r2,r1,3
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,1
  MOVI r3,4
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  CALL @mth_lap_csr_write_path_x
  ADD r15,r14,r0
  RET

mth_lap_csr_setup_triangle:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_offsets
  ST r0,r1,0
  MOVI r2,2
  ST r2,r1,1
  MOVI r2,4
  ST r2,r1,2
  MOVI r2,6
  ST r2,r1,3
  MOVI r2,4
  MOVI r3,6
  CALL @mth_lap_csr_fill_offset_tail
  MOVI r1,@mth_lap_csr_neighbors
  MOVI r2,1
  ST r2,r1,0
  MOVI r2,2
  ST r2,r1,1
  ST r0,r1,2
  MOVI r2,2
  ST r2,r1,3
  ST r0,r1,4
  MOVI r2,1
  ST r2,r1,5
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,1
  MOVI r3,6
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  CALL @mth_lap_csr_write_path_x
  ADD r15,r14,r0
  RET

mth_lap_csr_setup_empty:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,0
  MOVI r3,0
  CALL @mth_lap_csr_fill_offset_tail
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  ADD r15,r14,r0
  RET

; Asymmetric CSR: offsets=[0,1,1,...,1], neighbors=[1], edge_count=1.
mth_lap_csr_setup_asym:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,2
  MOVI r3,1
  CALL @mth_lap_csr_fill_offset_tail
  MOVI r1,@mth_lap_csr_neighbors
  MOVI r2,1
  ST r2,r1,0
  MOVI r1,@mth_lap_csr_weights
  MOVI r2,1
  MOVI r3,1
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  MOVI r2,0
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  MOVI r1,@mth_lap_csr_x
  CALL @mth_lap_csr_write_path_x
  ADD r15,r14,r0
  RET

mth_lap_csr_fill_sentinel:
  ADD r14,r15,r0
  MOVI r1,@mth_lap_csr_out
  MOVI r2,0x5678
  MOVHI r2,0x1234
  MOVI r3,256
  CALL @mth_lap_csr_fill_count
  ADD r15,r14,r0
  RET

; Call the path kernel with the standard path ABI and edge_count 6.
; The kernel clobbers r14, so the wrapper's return link lives in memory.
mth_lap_csr_call_path:
  ST r15,r0,@mth_lap_csr_link_save
  MOVI r1,@mth_lap_csr_offsets
  MOVI r2,@mth_lap_csr_neighbors
  MOVI r3,@mth_lap_csr_weights
  MOVI r4,@mth_lap_csr_x
  MOVI r5,@mth_lap_csr_out
  MOVI r6,6
  CALL @mth_graph_laplacian_csr_q16_16
  LD r15,r0,@mth_lap_csr_link_save
  RET

; r1=actual, r2=expected.  Four exact word comparisons; jumps to FAIL.
mth_lap_csr_check4:
  MOVI r3,4
mth_lap_csr_check4_loop:
  LD r4,r1,0
  LD r5,r2,0
  CMP r4,r5
  JMP[NE] @mth_lap_csr_test_fail
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lap_csr_check4_loop
  RET

; r1=start of 252-word tail; every word must be zero.
mth_lap_csr_check_zero_tail:
  MOVI r3,252
mth_lap_csr_check_zero_tail_loop:
  LD r4,r1,0
  CMP r4,r0
  JMP[NE] @mth_lap_csr_test_fail
  ADD r1,r1,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lap_csr_check_zero_tail_loop
  RET

; r1=base; all 256 words must be zero.
mth_lap_csr_check_all_zero:
  MOVI r3,256
mth_lap_csr_check_all_zero_loop:
  LD r4,r1,0
  CMP r4,r0
  JMP[NE] @mth_lap_csr_test_fail
  ADD r1,r1,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_lap_csr_check_all_zero_loop
  RET

; r1=output base; first and last words must still be the sentinel.
mth_lap_csr_check_unchanged:
  LD r4,r1,0
  MOVI r5,0x5678
  MOVHI r5,0x1234
  CMP r4,r5
  JMP[NE] @mth_lap_csr_test_fail
  MOVI r4,255
  ADD r1,r1,r4
  LD r4,r1,0
  CMP r4,r5
  JMP[NE] @mth_lap_csr_test_fail
  RET

mth_lap_csr_puts:
  MOVI r4,255
mth_lap_csr_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_lap_csr_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_lap_csr_puts_byte
  ADD r1,r1,r13
  JMP @mth_lap_csr_puts_word


.org 1024
mth_lap_csr_offsets:
.org 1288
mth_lap_csr_neighbors:
.org 1400
mth_lap_csr_weights:
.org 1456
mth_lap_csr_link_save:
.word 0
.org 1600
mth_lap_csr_x:
.org 2048
mth_lap_csr_out:
.org 2304
mth_lap_csr_expected_path:
.word -65536
.word -131072
.word -131072
.word 327680
mth_lap_csr_expected_disc:
.word -65536
.word 65536
.word -327680
.word 327680
mth_lap_csr_expected_triangle:
.word -327680
.word -131072
.word 458752
.word 0

.org 2500
mth_lap_csr_msg_path:
.str "PATH "
mth_lap_csr_msg_path_ok:
.str "PATH OK\n"
mth_lap_csr_msg_const:
.str "CONST "
mth_lap_csr_msg_const_ok:
.str "CONST OK\n"
mth_lap_csr_msg_disc:
.str "DISC "
mth_lap_csr_msg_disc_ok:
.str "DISC OK\n"
mth_lap_csr_msg_tri:
.str "TRI "
mth_lap_csr_msg_tri_ok:
.str "TRI OK\n"
mth_lap_csr_msg_empty:
.str "EMPTY "
mth_lap_csr_msg_empty_ok:
.str "EMPTY OK\n"
mth_lap_csr_msg_bad_x:
.str "BADX "
mth_lap_csr_msg_weight:
.str "WEIGHT "
mth_lap_csr_msg_sym:
.str "SYM "
mth_lap_csr_msg_pass:
.str "MTH GRAPH LAPLACIAN CSR PASS\n"
mth_lap_csr_msg_fail:
.str "MTH GRAPH LAPLACIAN CSR FAIL\n"
