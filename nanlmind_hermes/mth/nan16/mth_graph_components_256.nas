; MTH connected-component labels for canonical 256-vertex CSR — NaN16 only.
; Topology kernel: positive edges are treated as present; weights are validated
; separately by the weighted-CSR descriptor gate.
;
; ABI: r1=row_offsets, r2=neighbors, r3=edge_count,
;      r4=labels_out (256 U32), r5=parent_scratch (256 U32), r13=1.
; Returns r1=1 success, r2=component_count; r1=0 malformed CSR.
; Clobbers r1-r12; caller-owned scratch only; no CALLs or static slots.
;
; Requires canonical symmetric simple CSR. Kernel preflights offset[0],
; monotonicity, degree, final count, endpoint U8/range, no self-loops,
; and sorted unique row entries before writing labels or scratch.

mth_graph_components_256:
  CMP r3,r0
  JMP[LT] @mth_components_invalid
  MOVI r6,65280
  CMP r3,r6
  JMP[GT] @mth_components_invalid

  ; offset[0] must be exactly zero.
  LD r6,r1,0
  CMP r6,r0
  JMP[NE] @mth_components_invalid

  ; Validate all 257 offsets and row degrees.
  ADD r6,r1,r0
  MOVI r7,257
  MOVI r8,0
mth_components_validate_offsets:
  LD r9,r6,0
  CMP r9,r8
  JMP[LT] @mth_components_invalid
  CMP r9,r3
  JMP[GT] @mth_components_invalid
  SUB r10,r9,r8
  MOVI r11,255
  CMP r10,r11
  JMP[GT] @mth_components_invalid
  ADD r8,r9,r0
  ADD r6,r6,r13
  SUB r7,r7,r13
  CMP r7,r0
  JMP[GT] @mth_components_validate_offsets
  CMP r8,r3
  JMP[NE] @mth_components_invalid

  ; Validate every row's endpoints, no self-loops, sorted uniqueness.
  MOVI r6,0
mth_components_validate_row:
  ADD r12,r1,r6
  LD r7,r12,0
  ADD r12,r12,r13
  LD r8,r12,0
  ADD r9,r7,r0
  MOVI r10,0
mth_components_validate_edge:
  CMP r9,r8
  JMP[EQ] @mth_components_next_validate_row
  ADD r12,r2,r9
  LD r11,r12,0
  SHR r12,r11,8
  CMP r12,r0
  JMP[NE] @mth_components_invalid
  MOVI r12,256
  CMP r11,r12
  JMP[LT] @mth_components_endpoint_range_ok
  JMP @mth_components_invalid
mth_components_endpoint_range_ok:
  CMP r11,r6
  JMP[EQ] @mth_components_invalid
  CMP r9,r7
  JMP[EQ] @mth_components_first_neighbor
  CMP r11,r10
  JMP[GT] @mth_components_neighbor_sorted
  JMP @mth_components_invalid
mth_components_neighbor_sorted:
mth_components_first_neighbor:
  ADD r10,r11,r0
  ADD r9,r9,r13
  JMP @mth_components_validate_edge
mth_components_next_validate_row:
  ADD r6,r6,r13
  MOVI r12,256
  CMP r6,r12
  JMP[LT] @mth_components_validate_row

  ; Initialize union-find parent[i]=i only after all preflight succeeds.
  MOVI r6,0
mth_components_init_parent:
  ADD r12,r5,r6
  ST r6,r12,0
  ADD r6,r6,r13
  MOVI r12,256
  CMP r6,r12
  JMP[LT] @mth_components_init_parent

  ; Union each undirected edge once (neighbor > row index).
  MOVI r6,0
mth_components_union_row:
  ADD r12,r1,r6
  LD r7,r12,0
  ADD r12,r12,r13
  LD r8,r12,0
  ADD r9,r7,r0
mth_components_union_edge:
  CMP r9,r8
  JMP[EQ] @mth_components_next_union_row
  ADD r12,r2,r9
  LD r11,r12,0
  CMP r11,r6
  JMP[GT] @mth_components_union_pair
  JMP @mth_components_union_next_edge

mth_components_union_pair:
  ; Find root(row) without recursion or helper calls.
  ADD r10,r6,r0
mth_components_find_root_i:
  ADD r12,r5,r10
  LD r12,r12,0
  CMP r12,r10
  JMP[EQ] @mth_components_root_i_found
  ADD r10,r12,r0
  JMP @mth_components_find_root_i
mth_components_root_i_found:
  ADD r7,r10,r0

  ; Find root(neighbor).
  ADD r10,r11,r0
mth_components_find_root_j:
  ADD r12,r5,r10
  LD r12,r12,0
  CMP r12,r10
  JMP[EQ] @mth_components_root_j_found
  ADD r10,r12,r0
  JMP @mth_components_find_root_j
mth_components_root_j_found:
  CMP r7,r10
  JMP[EQ] @mth_components_union_next_edge
  ADD r12,r5,r7
  ST r10,r12,0

mth_components_union_next_edge:
  ADD r9,r9,r13
  JMP @mth_components_union_edge

mth_components_next_union_row:
  ADD r6,r6,r13
  MOVI r12,256
  CMP r6,r12
  JMP[LT] @mth_components_union_row

  ; Compress labels and count roots. Root representative is deterministic.
  MOVI r6,0
  MOVI r8,0
mth_components_emit_labels:
  ADD r10,r6,r0
mth_components_find_final_root:
  ADD r12,r5,r10
  LD r12,r12,0
  CMP r12,r10
  JMP[EQ] @mth_components_final_root_found
  ADD r10,r12,r0
  JMP @mth_components_find_final_root
mth_components_final_root_found:
  ADD r12,r4,r6
  ST r10,r12,0
  CMP r10,r6
  JMP[NE] @mth_components_label_next
  ADD r8,r8,r13
mth_components_label_next:
  ADD r6,r6,r13
  MOVI r12,256
  CMP r6,r12
  JMP[LT] @mth_components_emit_labels

  MOVI r1,1
  ADD r2,r8,r0
  RET

mth_components_invalid:
  MOVI r1,0
  MOVI r2,0
  RET
