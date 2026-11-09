; MTH graph degree / weighted-degree for 256-vertex canonical CSR — NaN16 only.
;
; degree[i]        = number of neighbors in row i.
; weighted_degree[i] = sum of the U8 integer weights in row i.
;
; ABI (r13 = 1):
;   r1 = row_offsets, 257 U32 words; offsets[0]=0, offsets[256]=edge_count
;   r2 = neighbors, edge_count U32 words; each endpoint in [0,255]
;   r3 = weights, edge_count U32 storage words; only the low 8 bits are used,
;        the upper 24 bits must be zero, and the low byte must be in [1,255]
;   r4 = edge_count, 0..65280 (each undirected edge is stored twice)
;   r5 = degree_out, 256 U32 words
;   r6 = weighted_out, 256 U32 words
;   returns r1=1 success, r1=0 malformed count/CSR/weight descriptor
;   clobbers r1-r12, r14; leaf routine, no static scratch, no CALLs.
;
; Numeric contract:
;   Every row has at most 255 entries and every weight is at most 255, so
;   0 <= weighted_degree[i] <= 255*255 = 65,025. Both outputs are exact U32
;   integers. There is no rounding, saturation, or floating-point operation.
;
; CSR contract:
;   Canonical symmetric simple graph: sorted unique neighbors, no self-loops,
;   reciprocal entries, and an immutable graph digest validated by the runner.
;   This kernel preflights count bounds, offset[0]=0, monotone offsets, final
;   edge count, degree bounds, endpoint range, sorted uniqueness, no self-loops,
;   and every weight word. All validation finishes before the first output
;   write, so a rejected descriptor leaves both output spans unchanged.
;   Symmetry and reciprocal-weight equality are bound by the graph digest.
;   Input and output spans must not overlap. For an unweighted graph the
;   caller supplies an all-ones weights page; weighted_degree then equals
;   degree exactly.

mth_graph_degree_256:
  ; Count bounds. A symmetric simple graph stores each undirected edge twice.
  CMP r4,r0
  JMP[LT] @mth_degree_invalid
  MOVI r8,65280
  CMP r4,r8
  JMP[GT] @mth_degree_invalid

  ; offset[0] must be exactly zero.
  LD r8,r1,0
  CMP r8,r0
  JMP[NE] @mth_degree_invalid

  ; Validate all 257 offsets and every row degree before any output write.
  ADD r8,r1,r0
  MOVI r9,257
  MOVI r10,0
mth_degree_validate_offsets:
  LD r11,r8,0
  CMP r11,r10
  JMP[LT] @mth_degree_invalid
  CMP r11,r4
  JMP[GT] @mth_degree_invalid
  SUB r12,r11,r10
  MOVI r14,255
  CMP r12,r14
  JMP[GT] @mth_degree_invalid
  ADD r10,r11,r0
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_degree_validate_offsets
  CMP r10,r4
  JMP[NE] @mth_degree_invalid

  ; Validate every row: endpoint U8 range, no self-loop, strictly sorted
  ; unique neighbors. Empty rows (isolated vertices) are accepted.
  MOVI r7,0
mth_degree_validate_row:
  ADD r8,r1,r7
  LD r9,r8,0
  LD r10,r8,1
  CMP r9,r10
  JMP[EQ] @mth_degree_next_row
  ADD r11,r2,r9
  LD r8,r11,0
  SHR r12,r8,8
  CMP r12,r0
  JMP[NE] @mth_degree_invalid
  MOVI r12,256
  CMP r8,r12
  JMP[GE] @mth_degree_invalid
  CMP r8,r7
  JMP[EQ] @mth_degree_invalid
  ADD r12,r8,r0
  ADD r11,r11,r13
  ADD r9,r9,r13
mth_degree_validate_edge:
  CMP r9,r10
  JMP[EQ] @mth_degree_next_row
  LD r8,r11,0
  SHR r14,r8,8
  CMP r14,r0
  JMP[NE] @mth_degree_invalid
  MOVI r14,256
  CMP r8,r14
  JMP[GE] @mth_degree_invalid
  CMP r8,r7
  JMP[EQ] @mth_degree_invalid
  CMP r8,r12
  JMP[LE] @mth_degree_invalid
  ADD r12,r8,r0
  ADD r11,r11,r13
  ADD r9,r9,r13
  JMP @mth_degree_validate_edge
mth_degree_next_row:
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_degree_validate_row

  ; Validate every weight word: upper 24 bits zero and low U8 nonzero.
  ADD r8,r3,r0
  ADD r9,r4,r0
  MOVI r14,255
mth_degree_validate_weights:
  CMP r9,r0
  JMP[EQ] @mth_degree_apply_setup
  LD r10,r8,0
  SHR r11,r10,8
  CMP r11,r0
  JMP[NE] @mth_degree_invalid
  BOOL.8 r11,r10,r14
  CMP r11,r0
  JMP[EQ] @mth_degree_invalid
  ADD r8,r8,r13
  SUB r9,r9,r13
  JMP @mth_degree_validate_weights

  ; All descriptor checks passed. Reduce each row independently.
mth_degree_apply_setup:
  MOVI r7,0
mth_degree_apply_row:
  ADD r8,r1,r7
  LD r9,r8,0
  LD r10,r8,1
  SUB r11,r10,r9
  ADD r8,r5,r7
  ST r11,r8,0

  MOVI r14,0
  ADD r11,r2,r9
  ADD r12,r2,r10
  ADD r8,r3,r9
mth_degree_apply_edge:
  CMP r11,r12
  JMP[EQ] @mth_degree_store_weighted
  LD r10,r8,0
  ADD r14,r14,r10
  ADD r8,r8,r13
  ADD r11,r11,r13
  JMP @mth_degree_apply_edge
mth_degree_store_weighted:
  ADD r8,r6,r7
  ST r14,r8,0
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_degree_apply_row

  MOVI r1,1
  RET

mth_degree_invalid:
  MOVI r1,0
  RET
