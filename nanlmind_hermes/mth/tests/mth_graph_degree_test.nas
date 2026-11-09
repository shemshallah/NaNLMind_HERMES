; MTH graph degree / weighted-degree native fixture — NaN16 only.
;
; Cases:
;   1. Symmetric path 0--1--2--3 with U8 weights [2,2,3,3,5,5] in CSR order.
;      Degrees [1,2,2,1] and weighted degrees [2,5,8,5]; vertices 4..255 are
;      isolated and must have degree 0 and weighted degree 0.
;   2. The same topology with unit weights; weighted degree must equal degree
;      for all 256 vertices.
;   3. Empty graph (edge_count 0); every degree and weighted degree is zero.
;   4. Zero weight rejects before changing either output span.
;   5. Nonzero offsets[0] rejects before changing either output span.

.org 0
JMP @mth_degree_test_main
.include "../nan16/mth_graph_degree.nas"

mth_degree_test_main:
  MOVI r13,1

  ; offsets = [0,1,3,5,6,6,...,6].
  MOVI r1,@mth_degree_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,5
  ST r2,r1,3
  MOVI r2,4
mth_degree_fill_offsets:
  ADD r3,r1,r2
  MOVI r4,6
  ST r4,r3,0
  ADD r2,r2,r13
  MOVI r3,257
  CMP r2,r3
  JMP[LT] @mth_degree_fill_offsets

  ; Weighted path case.
  MOVI r1,@mth_degree_offsets
  MOVI r2,@mth_degree_neighbors
  MOVI r3,@mth_degree_weights
  MOVI r4,6
  MOVI r5,@mth_degree_degree_out
  MOVI r6,@mth_degree_weighted_out
  CALL @mth_graph_degree_256
  CMP r1,r13
  JMP[NE] @mth_degree_test_fail

  MOVI r7,@mth_degree_degree_out
  LD r8,r7,0
  MOVI r9,1
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,1
  MOVI r9,2
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,2
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,3
  MOVI r9,1
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,4
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail
  MOVI r9,255
  ADD r7,r7,r9
  LD r8,r7,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail

  MOVI r7,@mth_degree_weighted_out
  LD r8,r7,0
  MOVI r9,2
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,1
  MOVI r9,5
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,2
  MOVI r9,8
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,3
  MOVI r9,5
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  LD r8,r7,4
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail
  MOVI r9,255
  ADD r7,r7,r9
  LD r8,r7,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail

  ; Unit-weight case: weighted degree must equal degree for every vertex.
  MOVI r1,@mth_degree_offsets
  MOVI r2,@mth_degree_neighbors
  MOVI r3,@mth_degree_unit_weights
  MOVI r4,6
  MOVI r5,@mth_degree_degree_out
  MOVI r6,@mth_degree_weighted_out
  CALL @mth_graph_degree_256
  CMP r1,r13
  JMP[NE] @mth_degree_test_fail
  MOVI r7,0
mth_degree_compare_unit:
  ADD r8,r5,r7
  LD r9,r8,0
  ADD r8,r6,r7
  LD r10,r8,0
  CMP r9,r10
  JMP[NE] @mth_degree_test_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_degree_compare_unit

  ; Empty graph: all 257 offsets are zero and edge_count is zero.
  MOVI r1,@mth_degree_offsets
  MOVI r7,0
mth_degree_zero_offsets:
  ADD r8,r1,r7
  ST r0,r8,0
  ADD r7,r7,r13
  MOVI r8,257
  CMP r7,r8
  JMP[LT] @mth_degree_zero_offsets
  MOVI r2,@mth_degree_neighbors
  MOVI r3,@mth_degree_weights
  MOVI r4,0
  MOVI r5,@mth_degree_degree_out
  MOVI r6,@mth_degree_weighted_out
  CALL @mth_graph_degree_256
  CMP r1,r13
  JMP[NE] @mth_degree_test_fail
  LD r8,r5,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail
  MOVI r9,255
  ADD r8,r5,r9
  LD r8,r8,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail
  LD r8,r6,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail
  ADD r8,r6,r9
  LD r8,r8,0
  CMP r8,r0
  JMP[NE] @mth_degree_test_fail

  ; Rebuild the path offsets for the two rejection-atomicity cases.
  MOVI r1,@mth_degree_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,5
  ST r2,r1,3
  MOVI r2,4
mth_degree_refill_offsets:
  ADD r3,r1,r2
  MOVI r4,6
  ST r4,r3,0
  ADD r2,r2,r13
  MOVI r3,257
  CMP r2,r3
  JMP[LT] @mth_degree_refill_offsets

  ; Zero weight rejects with both output spans unchanged.
  MOVI r1,@mth_degree_weights
  ST r0,r1,0
  MOVI r1,@mth_degree_degree_out
  MOVI r2,@mth_degree_weighted_out
  MOVI r3,0x5678
  MOVHI r3,0x1234
  MOVI r7,0
mth_degree_fill_degree_sentinel:
  ADD r8,r1,r7
  ST r3,r8,0
  ADD r8,r2,r7
  ST r3,r8,0
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_degree_fill_degree_sentinel
  MOVI r1,@mth_degree_offsets
  MOVI r2,@mth_degree_neighbors
  MOVI r3,@mth_degree_weights
  MOVI r4,6
  MOVI r5,@mth_degree_degree_out
  MOVI r6,@mth_degree_weighted_out
  CALL @mth_graph_degree_256
  CMP r1,r0
  JMP[NE] @mth_degree_test_fail
  MOVI r7,@mth_degree_degree_out
  LD r8,r7,0
  MOVI r9,0x5678
  MOVHI r9,0x1234
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail
  MOVI r7,@mth_degree_weighted_out
  MOVI r10,255
  ADD r7,r7,r10
  LD r8,r7,0
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail

  ; Restore the first weight, then reject nonzero offsets[0].
  MOVI r1,@mth_degree_weights
  MOVI r2,2
  ST r2,r1,0
  MOVI r1,@mth_degree_offsets
  MOVI r2,1
  ST r2,r1,0
  MOVI r1,@mth_degree_degree_out
  MOVI r2,77
  ST r2,r1,0
  MOVI r1,@mth_degree_offsets
  MOVI r2,@mth_degree_neighbors
  MOVI r3,@mth_degree_weights
  MOVI r4,6
  MOVI r5,@mth_degree_degree_out
  MOVI r6,@mth_degree_weighted_out
  CALL @mth_graph_degree_256
  CMP r1,r0
  JMP[NE] @mth_degree_test_fail
  MOVI r7,@mth_degree_degree_out
  LD r8,r7,0
  MOVI r9,77
  CMP r8,r9
  JMP[NE] @mth_degree_test_fail

  MOVI r1,@mth_degree_test_pass
  CALL @mth_degree_test_puts
  HALT

mth_degree_test_fail:
  MOVI r1,@mth_degree_test_bad
  CALL @mth_degree_test_puts
  HALT

mth_degree_test_puts:
  MOVI r4,255
mth_degree_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_degree_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_degree_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_degree_test_puts_word

.org 512
mth_degree_offsets:
.org 1536
mth_degree_neighbors:
.word 1
.word 0
.word 2
.word 1
.word 3
.word 2
.org 1568
mth_degree_weights:
.word 2
.word 2
.word 3
.word 3
.word 5
.word 5
.org 1600
mth_degree_unit_weights:
.word 1
.word 1
.word 1
.word 1
.word 1
.word 1
.org 1632
mth_degree_degree_out:
.org 2656
mth_degree_weighted_out:
.org 3680
mth_degree_test_pass:
.str "MTH GRAPH DEGREE 256 PASS\n"
mth_degree_test_bad:
.str "MTH GRAPH DEGREE 256 FAIL\n"
