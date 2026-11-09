; MTH canonical-CSR unweighted graph Laplacian, Q16.16/I32 v1 — NaN16 only.
;
; Mathematical operator (docs/GRAPH_SPECTRAL_MTH_PLAN.md §2.2):
;
;   (Lx)_i = d_i*x_i - sum_{j in N(i)} x_j
;          = sum_{j in N(i)} (x_i - x_j)
;
; for a symmetric simple graph with unit edge weights.  The 65,536 x 65,536
; dense Laplacian/Kronecker matrix is never formed: the kernel touches only
; the 257-word offset page, the bounded CSR edge lists, one 256-word state
; vector, and one 256-word output vector.
;
; ABI (r13 = 1):
;   r1 = row_offsets, 257 U32 words; offsets[0]=0, offsets[256]=edge_count
;   r2 = neighbors, edge_count U32 words; each endpoint in [0,255]
;   r3 = weights, edge_count U32 storage words; canonical U8 value exactly 1
;   r4 = x, 256 signed I32 Q16.16 words, each in [-2^22,+2^22]
;   r5 = output, 256 signed I32 Q16.16 words
;   r6 = edge_count, 0..65280 (each undirected edge stored twice)
;   returns r1=1 success,
;           r1=0 invalid edge count or invalid state value,
;           r1=2 invalid CSR offsets/degree,
;           r1=3 invalid endpoint encoding, row order, duplicate, or self-loop,
;           r1=4 invalid weight encoding or non-unit (weighted) graph,
;           r1=5 non-reciprocal/asymmetric edge list.
;   clobbers r1-r12 and r14; preserves r13 and r15.  Leaf routine:
;   no static scratch and no CALLs.
;
; Numeric contract:
;   |x_i| <= 2^22.  Each row has at most 255 neighbors, so
;     |(Lx)_i| <= 255 * 2^23 = 2,139,095,040 < 2^31.
;   Every subtraction and row accumulation is exact signed I32
;   two's-complement arithmetic.  There is no rounding, saturation, wrap,
;   or floating-point operation.  Output carries the same Q16.16 scale.
;
; Canonical CSR contract:
;   * offsets[0] is exactly zero, offsets are monotone nondecreasing,
;     every row degree is at most 255, and offsets[256] equals edge_count;
;   * every neighbor word is a canonical U8 index (upper 24 bits zero);
;   * every row is strictly increasing, duplicate-free, and self-loop-free;
;   * every weight word is canonical U8 and exactly 1, because this operator
;     is the *unweighted* Laplacian; a weighted descriptor must use the
;     separately versioned weighted-CSR ABI;
;   * every stored directed edge has its reciprocal entry (full symmetry
;     check over the immutable edge list).  Strict ordering makes the search
;     terminate at the first non-smaller endpoint.
;   * all validation finishes before the first output write, so a rejected
;     descriptor leaves the 256-word output page unchanged.
;   Input and output spans must not overlap.  The directed semantic relation
;   rails and W/W_s/W_a are not read or modified; this CSR view is a derived
;   coupling operator only.

mth_graph_laplacian_csr_q16_16:
  ; Edge-count bound.  A symmetric simple graph stores every edge twice.
  CMP r6,r0
  JMP[LT] @mth_lap_csr_invalid_state
  MOVI r8,65280
  CMP r6,r8
  JMP[GT] @mth_lap_csr_invalid_state

  ; Preflight all 256 signed Q16.16 states in [-2^22,+2^22].
  ADD r8,r4,r0
  MOVI r9,256
mth_lap_csr_validate_x:
  LD r10,r8,0
  ; Unsigned shift extracts the ten high sign/range bits.  Valid values
  ; yield 0, 1 (only +2^22), or 1023 (negative range).
  SHR r11,r10,22
  CMP r11,r0
  JMP[EQ] @mth_lap_csr_x_ok
  MOVI r12,1
  CMP r11,r12
  JMP[NE] @mth_lap_csr_x_check_negative
  MOVI r12,0
  MOVHI r12,0x0040
  CMP r10,r12
  JMP[EQ] @mth_lap_csr_x_ok
  JMP @mth_lap_csr_invalid_state
mth_lap_csr_x_check_negative:
  MOVI r12,1023
  CMP r11,r12
  JMP[EQ] @mth_lap_csr_x_ok
  JMP @mth_lap_csr_invalid_state
mth_lap_csr_x_ok:
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_lap_csr_validate_x

  ; Validate all 257 CSR offsets before any output write.
  ADD r8,r1,r0
  LD r10,r8,0
  CMP r10,r0
  JMP[NE] @mth_lap_csr_invalid_offsets
  MOVI r9,257
  MOVI r7,0
mth_lap_csr_validate_offsets:
  LD r10,r8,0
  CMP r10,r7
  JMP[LT] @mth_lap_csr_invalid_offsets
  CMP r10,r6
  JMP[GT] @mth_lap_csr_invalid_offsets
  SUB r11,r10,r7
  MOVI r12,255
  CMP r11,r12
  JMP[GT] @mth_lap_csr_invalid_offsets
  ADD r7,r10,r0
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_lap_csr_validate_offsets
  CMP r7,r6
  JMP[NE] @mth_lap_csr_invalid_offsets

  ; Validate every row: canonical U8 endpoints, no self-loop, and strictly
  ; increasing unique neighbors.  Empty rows (isolated vertices) are valid.
  MOVI r7,0
mth_lap_csr_validate_row:
  ADD r8,r1,r7
  LD r9,r8,0
  LD r10,r8,1
  CMP r9,r10
  JMP[EQ] @mth_lap_csr_next_row
  ADD r11,r2,r9
  LD r12,r11,0
  SHR r14,r12,8
  CMP r14,r0
  JMP[NE] @mth_lap_csr_invalid_neighbors
  MOVI r14,256
  CMP r12,r14
  JMP[GE] @mth_lap_csr_invalid_neighbors
  CMP r12,r7
  JMP[EQ] @mth_lap_csr_invalid_neighbors
  ADD r9,r9,r13
mth_lap_csr_validate_edge:
  CMP r9,r10
  JMP[EQ] @mth_lap_csr_next_row
  ADD r11,r2,r9
  LD r14,r11,0
  SHR r11,r14,8
  CMP r11,r0
  JMP[NE] @mth_lap_csr_invalid_neighbors
  MOVI r11,256
  CMP r14,r11
  JMP[GE] @mth_lap_csr_invalid_neighbors
  CMP r14,r7
  JMP[EQ] @mth_lap_csr_invalid_neighbors
  CMP r14,r12
  JMP[LE] @mth_lap_csr_invalid_neighbors
  ADD r12,r14,r0
  ADD r9,r9,r13
  JMP @mth_lap_csr_validate_edge
mth_lap_csr_next_row:
  ADD r7,r7,r13
  MOVI r14,256
  CMP r7,r14
  JMP[LT] @mth_lap_csr_validate_row

  ; Validate every weight word: upper 24 bits zero and low U8 exactly one.
  ADD r8,r3,r0
  ADD r9,r6,r0
mth_lap_csr_validate_weights:
  CMP r9,r0
  JMP[EQ] @mth_lap_csr_validate_symmetry_setup
  LD r10,r8,0
  SHR r11,r10,8
  CMP r11,r0
  JMP[NE] @mth_lap_csr_invalid_weights
  MOVI r11,1
  CMP r10,r11
  JMP[NE] @mth_lap_csr_invalid_weights
  ADD r8,r8,r13
  SUB r9,r9,r13
  JMP @mth_lap_csr_validate_weights

  ; Full reciprocal-edge check.  For each row i, only neighbors j > i need a
  ; search in row j; entries j < i were verified when row j was processed.
  ; Row j is strictly increasing, so stop as soon as an endpoint is > i.
mth_lap_csr_validate_symmetry_setup:
  MOVI r7,0
mth_lap_csr_symmetry_row:
  ADD r8,r1,r7
  LD r9,r8,0
  LD r10,r8,1
mth_lap_csr_symmetry_edge:
  CMP r9,r10
  JMP[EQ] @mth_lap_csr_symmetry_next_row
  ADD r11,r2,r9
  LD r12,r11,0
  CMP r12,r7
  JMP[LE] @mth_lap_csr_symmetry_skip
  ADD r8,r1,r12
  LD r14,r8,0
  LD r8,r8,1
mth_lap_csr_symmetry_search:
  CMP r14,r8
  JMP[EQ] @mth_lap_csr_invalid_symmetry
  ADD r11,r2,r14
  LD r11,r11,0
  CMP r11,r7
  JMP[EQ] @mth_lap_csr_symmetry_skip
  CMP r11,r7
  JMP[GT] @mth_lap_csr_invalid_symmetry
  ADD r14,r14,r13
  JMP @mth_lap_csr_symmetry_search
mth_lap_csr_symmetry_skip:
  ADD r9,r9,r13
  JMP @mth_lap_csr_symmetry_edge
mth_lap_csr_symmetry_next_row:
  ADD r7,r7,r13
  MOVI r14,256
  CMP r7,r14
  JMP[LT] @mth_lap_csr_symmetry_row

  ; All descriptor and numeric checks passed.  Apply one row at a time.
  ; r1/r2/r4 stay as immutable bases; r5-r12,r14 are working registers.
  ADD r8,r1,r0
  ADD r10,r4,r0
  ADD r11,r5,r0
  MOVI r12,256
mth_lap_csr_apply_row:
  LD r7,r8,0
  LD r14,r8,1
  ADD r7,r2,r7
  ADD r14,r2,r14
  LD r9,r10,0
  MOVI r6,0
mth_lap_csr_apply_edge:
  CMP r7,r14
  JMP[EQ] @mth_lap_csr_apply_store
  LD r3,r7,0
  ADD r3,r4,r3
  LD r3,r3,0
  SUB r3,r9,r3
  ADD r6,r6,r3
  ADD r7,r7,r13
  JMP @mth_lap_csr_apply_edge
mth_lap_csr_apply_store:
  ST r6,r11,0
  ADD r8,r8,r13
  ADD r10,r10,r13
  ADD r11,r11,r13
  SUB r12,r12,r13
  CMP r12,r0
  JMP[GT] @mth_lap_csr_apply_row

  MOVI r1,1
  RET

mth_lap_csr_invalid_state:
  MOVI r1,0
  RET
mth_lap_csr_invalid_offsets:
  MOVI r1,2
  RET
mth_lap_csr_invalid_neighbors:
  MOVI r1,3
  RET
mth_lap_csr_invalid_weights:
  MOVI r1,4
  RET
mth_lap_csr_invalid_symmetry:
  MOVI r1,5
  RET
