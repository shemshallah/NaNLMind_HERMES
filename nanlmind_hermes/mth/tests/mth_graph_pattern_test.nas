; MTH graph pattern signature native fixture — NaN16 only.
;
; Signature layout:
;   bits 31..24 = shell (BFS distance clamped to 254; 255 = unreachable)
;   bits 23..16 = degree
;   bits 15..0  = sum of neighbor degrees
;
; Cases:
;   1. Path 0--1--2--3 plus 252 isolates, source 0.
;      Expected signatures: 0x00010002, 0x01020003, 0x02020003, 0x03010002;
;      every isolated vertex must be exactly 0xFF000000 (bounded top byte).
;   2. Same path, source 2. This changes shell bits while degrees and
;      neighbor sums stay fixed: 0x02010002, 0x01020003, 0x00020003,
;      0x01010002 for vertices 0..3.
;   3. Triangle 0--1--2 plus isolates, source 0. Every triangle vertex has
;      degree 2 and neighbor-degree sum 4; shells are 0,1,1.
;      Expected: 0x00020004, 0x01020004, 0x01020004, 0xFF000000.
;   4. Source 256 rejects before changing the output page.
;   5. Nonzero offsets[0] rejects before changing the output page.

.org 0
JMP @mth_pattern_test_main
.include "../nan16/mth_graph_pattern.nas"

mth_pattern_test_main:
  MOVI r13,1

  ; Path offsets = [0,1,3,5,6,6,...,6].
  MOVI r1,@mth_pattern_path_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,5
  ST r2,r1,3
  MOVI r2,4
mth_pattern_fill_path_offsets:
  ADD r3,r1,r2
  MOVI r4,6
  ST r4,r3,0
  ADD r2,r2,r13
  MOVI r3,257
  CMP r2,r3
  JMP[LT] @mth_pattern_fill_path_offsets

  ; Path case, source 0.
  MOVI r1,@mth_pattern_path_offsets
  MOVI r2,@mth_pattern_path_neighbors
  MOVI r3,6
  MOVI r4,0
  MOVI r5,@mth_pattern_out
  MOVI r6,@mth_pattern_degree_scratch
  MOVI r7,@mth_pattern_queue_scratch
  CALL @mth_graph_pattern_256
  CMP r1,r13
  JMP[NE] @mth_pattern_test_fail

  MOVI r8,@mth_pattern_out
  LD r9,r8,0
  MOVI r10,0x0002
  MOVHI r10,0x0001
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,1
  MOVI r10,0x0003
  MOVHI r10,0x0102
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,2
  MOVI r10,0x0003
  MOVHI r10,0x0202
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,3
  MOVI r10,0x0002
  MOVHI r10,0x0301
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail

  ; Vertices 4..255 are isolated and unreachable: exact 0xFF000000.
  MOVI r7,4
  MOVI r10,0
  MOVHI r10,0xFF00
mth_pattern_check_isolates:
  ADD r8,r5,r7
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_pattern_check_isolates

  ; Same path, source 2: shell bits change, degree/neighborhood fields do not.
  MOVI r1,@mth_pattern_path_offsets
  MOVI r2,@mth_pattern_path_neighbors
  MOVI r3,6
  MOVI r4,2
  MOVI r5,@mth_pattern_out
  MOVI r6,@mth_pattern_degree_scratch
  MOVI r7,@mth_pattern_queue_scratch
  CALL @mth_graph_pattern_256
  CMP r1,r13
  JMP[NE] @mth_pattern_test_fail

  MOVI r8,@mth_pattern_out
  LD r9,r8,0
  MOVI r10,0x0002
  MOVHI r10,0x0201
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,1
  MOVI r10,0x0003
  MOVHI r10,0x0102
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,2
  MOVI r10,0x0003
  MOVHI r10,0x0002
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,3
  MOVI r10,0x0002
  MOVHI r10,0x0101
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,4
  MOVI r10,0
  MOVHI r10,0xFF00
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail

  ; Triangle offsets = [0,2,4,6,6,...,6].
  MOVI r1,@mth_pattern_triangle_offsets
  ST r0,r1,0
  MOVI r2,2
  ST r2,r1,1
  MOVI r2,4
  ST r2,r1,2
  MOVI r2,6
  ST r2,r1,3
  MOVI r2,4
mth_pattern_fill_triangle_offsets:
  ADD r3,r1,r2
  MOVI r4,6
  ST r4,r3,0
  ADD r2,r2,r13
  MOVI r3,257
  CMP r2,r3
  JMP[LT] @mth_pattern_fill_triangle_offsets

  ; Triangle case, source 0.
  MOVI r1,@mth_pattern_triangle_offsets
  MOVI r2,@mth_pattern_triangle_neighbors
  MOVI r3,6
  MOVI r4,0
  MOVI r5,@mth_pattern_out
  MOVI r6,@mth_pattern_degree_scratch
  MOVI r7,@mth_pattern_queue_scratch
  CALL @mth_graph_pattern_256
  CMP r1,r13
  JMP[NE] @mth_pattern_test_fail

  MOVI r8,@mth_pattern_out
  LD r9,r8,0
  MOVI r10,0x0004
  MOVHI r10,0x0002
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,1
  MOVI r10,0x0004
  MOVHI r10,0x0102
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,2
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail
  LD r9,r8,3
  MOVI r10,0
  MOVHI r10,0xFF00
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail

  ; Source 256 rejects without changing output[0].
  MOVI r8,@mth_pattern_out
  MOVI r9,77
  ST r9,r8,0
  MOVI r1,@mth_pattern_triangle_offsets
  MOVI r2,@mth_pattern_triangle_neighbors
  MOVI r3,6
  MOVI r4,256
  MOVI r5,@mth_pattern_out
  MOVI r6,@mth_pattern_degree_scratch
  MOVI r7,@mth_pattern_queue_scratch
  CALL @mth_graph_pattern_256
  CMP r1,r0
  JMP[NE] @mth_pattern_test_fail
  MOVI r8,@mth_pattern_out
  LD r9,r8,0
  MOVI r10,77
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail

  ; Nonzero offsets[0] rejects without changing output[0].
  MOVI r1,@mth_pattern_triangle_offsets
  MOVI r2,1
  ST r2,r1,0
  MOVI r1,@mth_pattern_triangle_offsets
  MOVI r2,@mth_pattern_triangle_neighbors
  MOVI r3,6
  MOVI r4,0
  MOVI r5,@mth_pattern_out
  MOVI r6,@mth_pattern_degree_scratch
  MOVI r7,@mth_pattern_queue_scratch
  CALL @mth_graph_pattern_256
  CMP r1,r0
  JMP[NE] @mth_pattern_test_fail
  MOVI r8,@mth_pattern_out
  LD r9,r8,0
  MOVI r10,77
  CMP r9,r10
  JMP[NE] @mth_pattern_test_fail

  MOVI r1,@mth_pattern_test_pass
  CALL @mth_pattern_test_puts
  HALT

mth_pattern_test_fail:
  MOVI r1,@mth_pattern_test_bad
  CALL @mth_pattern_test_puts
  HALT

mth_pattern_test_puts:
  MOVI r4,255
mth_pattern_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pattern_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pattern_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_pattern_test_puts_word

.org 512
mth_pattern_path_offsets:
.org 1536
mth_pattern_path_neighbors:
.word 1
.word 0
.word 2
.word 1
.word 3
.word 2
.org 1568
mth_pattern_triangle_offsets:
.org 1856
mth_pattern_triangle_neighbors:
.word 1
.word 2
.word 0
.word 2
.word 0
.word 1
.org 1888
mth_pattern_out:
.org 2144
mth_pattern_degree_scratch:
.org 2400
mth_pattern_queue_scratch:
.org 2656
mth_pattern_test_pass:
.str "MTH GRAPH PATTERN 256 PASS\n"
mth_pattern_test_bad:
.str "MTH GRAPH PATTERN 256 FAIL\n"
