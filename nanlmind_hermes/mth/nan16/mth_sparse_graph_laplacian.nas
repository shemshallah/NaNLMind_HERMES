; MTH sparse unweighted graph Laplacian, Q16.16/I32 v1 — NaN16 only.
;
; Lx[i] = sum_{j in N(i)} (x[i] - x[j]) for a symmetric simple CSR graph.
;
; ABI:
;   r1 = CSR row_offsets, 257 U32 words, offsets[0]=0, offsets[256]=edge_count
;   r2 = CSR neighbors, edge_count U32 words, each endpoint in [0,255]
;   r3 = x, 256 signed I32 Q16.16 words
;   r4 = output, 256 signed I32 Q16.16 words
;   r5 = edge_count, 0..65280 (each undirected edge is stored twice)
;   returns r1=1 success, 0 invalid state/count, 2 invalid offsets, 3 invalid endpoints
;   requires r13=1; clobbers r1-r12; leaf routine, no static scratch.
;
; Numeric contract:
;   input raw values are in [-2^22,+2^22] (real range [-64,+64]).
;   each difference is bounded by 2^23; each row has <=255 neighbors;
;   |Lx[i]| <= 255*2^23 = 2,139,095,040 < 2^31.
;   Thus all arithmetic is exact signed I32 Q16.16: no rounding,
;   saturation, or intermediate overflow. Output has the same Q format.
;
; CSR must be a canonical symmetric simple graph: sorted unique neighbors,
; no self-loops, reciprocal entries, and immutable graph digest validated by
; the caller/runner. This kernel preflights state range, offset monotonicity,
; final edge count, and endpoint range before writing any output. The graph
; digest binds the stronger symmetry/simple-graph invariant.
; Input/output spans must not overlap.

mth_sparse_graph_laplacian_q16_16:
  CMP r5,r0
  JMP[LT] @mth_sparse_graph_invalid_input
  MOVI r6,65280
  CMP r5,r6
  JMP[GT] @mth_sparse_graph_invalid_input

  ; Complete state-range preflight.
  ADD r6,r3,r0
  MOVI r7,256
mth_sparse_graph_validate_x:
  LD r10,r6,0
  ; Unsigned shift extracts the ten high sign/range bits. Valid Q16.16
  ; values in [-2^22,+2^22] yield 0, 1, or 1023 respectively.
  SHR r11,r10,22
  MOVI r12,0
  CMP r11,r12
  JMP[EQ] @mth_sparse_graph_x_valid
  MOVI r12,1
  CMP r11,r12
  JMP[NE] @mth_sparse_graph_check_negative
  ; The only accepted positive value with high bits 1 is +2^22 exactly.
  MOVI r12,0
  MOVHI r12,0x0040
  CMP r10,r12
  JMP[EQ] @mth_sparse_graph_x_valid
  JMP @mth_sparse_graph_invalid_input
mth_sparse_graph_check_negative:
  MOVI r12,1023
  CMP r11,r12
  JMP[EQ] @mth_sparse_graph_x_valid
  JMP @mth_sparse_graph_invalid_input
mth_sparse_graph_x_valid:
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_sparse_graph_validate_x

  ; Validate all 257 CSR offsets before output is touched.
  ADD r6,r1,r0
  ; CSR canonical form requires offsets[0] == 0.
  LD r10,r6,0
  CMP r10,r0
  JMP[NE] @mth_sparse_graph_invalid_offsets
  MOVI r7,257
  MOVI r8,0
mth_sparse_graph_validate_offsets:
  LD r10,r6,0
  CMP r10,r8
  JMP[LT] @mth_sparse_graph_invalid_offsets
  CMP r10,r5
  JMP[GT] @mth_sparse_graph_invalid_offsets
  SUB r11,r10,r8
  MOVI r12,255
  CMP r11,r12
  JMP[GT] @mth_sparse_graph_invalid_offsets
  ADD r8,r10,r0
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_sparse_graph_validate_offsets
  CMP r8,r5
  JMP[NE] @mth_sparse_graph_invalid_offsets

  ; Validate every endpoint is a canonical U8 value.
  ADD r6,r2,r0
  ADD r7,r5,r0
  MOVI r8,256
mth_sparse_graph_validate_neighbors:
  CMP r7,r0
  JMP[EQ] @mth_sparse_graph_apply_setup
  LD r9,r6,0
  SHR r10,r9,8
  CMP r10,r0
  JMP[NE] @mth_sparse_graph_invalid_neighbors
  CMP r9,r8
  JMP[LT] @mth_sparse_graph_neighbor_ok
  JMP @mth_sparse_graph_invalid_neighbors
mth_sparse_graph_neighbor_ok:
  ADD r6,r6,r13
  SUB r7,r7,r13
  JMP @mth_sparse_graph_validate_neighbors

  ; All failure conditions have been checked. Apply one row at a time.
mth_sparse_graph_apply_setup:
  ADD r6,r1,r0
  ADD r7,r2,r0
  ADD r8,r3,r0
  ADD r12,r3,r0
  ADD r9,r4,r0
  MOVI r10,256
mth_sparse_graph_apply_row:
  LD r1,r6,0
  LD r2,r6,1
  SUB r3,r2,r1
  ADD r11,r7,r1
  LD r4,r8,0
  MOVI r5,0
mth_sparse_graph_apply_edge:
  CMP r3,r0
  JMP[EQ] @mth_sparse_graph_store_row
  LD r1,r11,0
  ADD r2,r12,r1
  LD r1,r2,0
  SUB r1,r4,r1
  ADD r5,r5,r1
  ADD r11,r11,r13
  SUB r3,r3,r13
  JMP @mth_sparse_graph_apply_edge
mth_sparse_graph_store_row:
  ST r5,r9,0
  ADD r6,r6,r13
  ADD r8,r8,r13
  ADD r9,r9,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_sparse_graph_apply_row
  MOVI r1,1
  RET

mth_sparse_graph_invalid_input:
  MOVI r1,0
  RET
mth_sparse_graph_invalid_offsets:
  MOVI r1,2
  RET
mth_sparse_graph_invalid_neighbors:
  MOVI r1,3
  RET
