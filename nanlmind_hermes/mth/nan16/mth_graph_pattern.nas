; MTH deterministic degree/shell/neighborhood signature for 256-vertex CSR.
;
; For each vertex i the kernel emits one bounded U32 signature:
;
;   signature[i] = (shell[i] << 24) | (degree[i] << 16) | neighbor_degree_sum[i]
;
;   degree[i]            = row length, 0..255
;   neighbor_degree_sum[i] = sum of degree[j] over all neighbors j, 0..65025
;   shell[i]             = BFS distance from the caller-supplied source,
;                          clamped to 0..254; 255 means unreachable
;
; The signature is deterministic for identical CSR pages, source vertex,
; kernel version and digest. It is a lossy pattern fingerprint: shell values
; >=254 are intentionally collapsed, and the signature never replaces the
; exact source relations or graph digest.
;
; ABI (r13 = 1):
;   r1 = row_offsets, 257 U32 words; offsets[0]=0, offsets[256]=edge_count
;   r2 = neighbors, edge_count U32 words; each endpoint in [0,255]
;   r3 = edge_count, 0..65280 (each undirected edge is stored twice)
;   r4 = source vertex, 0..255
;   r5 = signature_out, 256 U32 words
;   r6 = degree_scratch, 256 U32 words (caller-owned)
;   r7 = queue_scratch, 256 U32 words (caller-owned)
;   returns r1=1 success, r1=0 malformed count/CSR/source
;   clobbers r1-r12, r14; leaf routine, no static scratch, no CALLs.
;
; CSR contract:
;   Canonical symmetric simple graph: sorted unique neighbors, no self-loops,
;   reciprocal entries, and an immutable graph digest validated by the runner.
;   This kernel preflights count bounds, source range, offset[0]=0, monotone
;   offsets, final edge count, degree bounds, endpoint range, sorted uniqueness,
;   and no self-loops before touching signature_out. Symmetry is bound by the
;   graph digest. Input, scratch and output spans must not overlap.
;
; Workspace: 512 caller-owned U32 words (4,096 bytes), far below the 2 MiB
; MTH guest ceiling. Shell BFS marks each vertex when enqueued, so the queue
; holds at most 256 entries and never overflows for a validated simple graph.
;
; Numeric contract: integer-only NaN16 operations. The neighbor degree sum is
; at most 255*255 = 65,025, so it occupies exactly 16 bits. The three packed
; fields do not overlap; every signature is <= 0xFFFFFFFF.

mth_graph_pattern_256:
  ; Source range and edge-count bounds.
  MOVI r8,256
  CMP r4,r8
  JMP[GE] @mth_pattern_invalid
  CMP r3,r0
  JMP[LT] @mth_pattern_invalid
  MOVI r8,65280
  CMP r3,r8
  JMP[GT] @mth_pattern_invalid

  ; offset[0] must be exactly zero.
  LD r8,r1,0
  CMP r8,r0
  JMP[NE] @mth_pattern_invalid

  ; Validate all 257 offsets and row degrees.
  ADD r8,r1,r0
  MOVI r9,257
  MOVI r10,0
mth_pattern_validate_offsets:
  LD r11,r8,0
  CMP r11,r10
  JMP[LT] @mth_pattern_invalid
  CMP r11,r3
  JMP[GT] @mth_pattern_invalid
  SUB r12,r11,r10
  MOVI r14,255
  CMP r12,r14
  JMP[GT] @mth_pattern_invalid
  ADD r10,r11,r0
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_pattern_validate_offsets
  CMP r10,r3
  JMP[NE] @mth_pattern_invalid

  ; Validate every row: endpoint U8 range, no self-loop, sorted uniqueness.
  MOVI r14,0
mth_pattern_validate_row:
  ADD r8,r1,r14
  LD r9,r8,0
  LD r10,r8,1
  CMP r9,r10
  JMP[EQ] @mth_pattern_next_row
  ADD r11,r2,r9
  LD r8,r11,0
  SHR r12,r8,8
  CMP r12,r0
  JMP[NE] @mth_pattern_invalid
  MOVI r12,256
  CMP r8,r12
  JMP[GE] @mth_pattern_invalid
  CMP r8,r14
  JMP[EQ] @mth_pattern_invalid
  ADD r12,r8,r0
  ADD r11,r11,r13
  ADD r9,r9,r13
mth_pattern_validate_edge:
  CMP r9,r10
  JMP[EQ] @mth_pattern_next_row
  LD r8,r11,0
  SHR r3,r8,8
  CMP r3,r0
  JMP[NE] @mth_pattern_invalid
  MOVI r3,256
  CMP r8,r3
  JMP[GE] @mth_pattern_invalid
  CMP r8,r14
  JMP[EQ] @mth_pattern_invalid
  CMP r8,r12
  JMP[LE] @mth_pattern_invalid
  ADD r12,r8,r0
  ADD r11,r11,r13
  ADD r9,r9,r13
  JMP @mth_pattern_validate_edge
mth_pattern_next_row:
  ADD r14,r14,r13
  MOVI r8,256
  CMP r14,r8
  JMP[LT] @mth_pattern_validate_row

  ; degree_scratch[i] = offsets[i+1] - offsets[i].
  MOVI r14,0
mth_pattern_degree_loop:
  ADD r8,r1,r14
  LD r9,r8,0
  LD r10,r8,1
  SUB r11,r10,r9
  ADD r8,r6,r14
  ST r11,r8,0
  ADD r14,r14,r13
  MOVI r8,256
  CMP r14,r8
  JMP[LT] @mth_pattern_degree_loop

  ; signature_out starts with the unreachable sentinel 0xFFFFFFFF.
  MOVI r8,0xFFFF
  MOVHI r8,0xFFFF
  MOVI r14,0
mth_pattern_init_shell:
  ADD r9,r5,r14
  ST r8,r9,0
  ADD r14,r14,r13
  MOVI r9,256
  CMP r14,r9
  JMP[LT] @mth_pattern_init_shell

  ; BFS shell distances from source. The output page doubles as the distance
  ; page until the final signature pass.
  ADD r8,r5,r4
  ST r0,r8,0
  ST r4,r7,0
  MOVI r8,0
  MOVI r9,1
  MOVI r4,0xFFFF
  MOVHI r4,0xFFFF
mth_pattern_bfs_loop:
  CMP r8,r9
  JMP[GE] @mth_pattern_finalize
  ADD r10,r7,r8
  LD r10,r10,0
  ADD r8,r8,r13
  ADD r12,r5,r10
  LD r11,r12,0
  ADD r3,r11,r13
  MOVI r12,254
  CMP r3,r12
  JMP[LE] @mth_pattern_shell_bounded
  ADD r3,r12,r0
mth_pattern_shell_bounded:
  ADD r12,r1,r10
  LD r14,r12,0
  LD r10,r12,1
  ADD r12,r2,r14
  ADD r14,r2,r10
mth_pattern_bfs_edge:
  CMP r12,r14
  JMP[GE] @mth_pattern_bfs_loop
  LD r11,r12,0
  ADD r10,r5,r11
  LD r10,r10,0
  CMP r10,r4
  JMP[NE] @mth_pattern_bfs_next
  ADD r10,r5,r11
  ST r3,r10,0
  ADD r10,r7,r9
  ST r11,r10,0
  ADD r9,r9,r13
mth_pattern_bfs_next:
  ADD r12,r12,r13
  JMP @mth_pattern_bfs_edge

  ; Pack shell, degree and neighbor-degree sum into one bounded U32 word.
mth_pattern_finalize:
  MOVI r14,0
mth_pattern_pack_loop:
  ADD r8,r5,r14
  LD r9,r8,0
  CMP r9,r4
  JMP[NE] @mth_pattern_shell_ready
  MOVI r9,255
mth_pattern_shell_ready:
  ADD r8,r6,r14
  LD r10,r8,0
  ADD r8,r1,r14
  LD r11,r8,0
  LD r12,r8,1
  ADD r11,r2,r11
  ADD r12,r2,r12
  MOVI r8,0
mth_pattern_neighbor_sum:
  CMP r11,r12
  JMP[GE] @mth_pattern_pack
  LD r3,r11,0
  ADD r3,r6,r3
  LD r3,r3,0
  ADD r8,r8,r3
  ADD r11,r11,r13
  JMP @mth_pattern_neighbor_sum
mth_pattern_pack:
  SHL r9,r9,24
  SHL r10,r10,16
  BOOL.14 r9,r9,r10
  BOOL.14 r9,r9,r8
  ADD r3,r5,r14
  ST r9,r3,0
  ADD r14,r14,r13
  MOVI r3,256
  CMP r14,r3
  JMP[LT] @mth_pattern_pack_loop

  MOVI r1,1
  RET

mth_pattern_invalid:
  MOVI r1,0
  RET
