; MTH R5 observability/reachability primitive - NaN16 integer BFS.
;
; ABI:
;   r1  = A, 65,536 float32 words, row-major 256x256 adjacency
;   r2  = source vertex index, 0..255
;   r3  = visited_out, 256 float32 words (0.0 / 1.0)
;   r4  = queue_scratch, at least 256 float32-width words
;   r13 = constant 1
;   returns r1 = reachable count in 1..256, or 0 for invalid source/overflow
;   clobbers r1, r5-r12, r14, flags
;   preserves r2, r3, r4, r13, r15
;
; A[i,j] != 0 means the directed edge i -> j is present. The test is a
; numeric-zero test on the float32 payload: SHL by one drops the sign bit, so
; +0.0 and -0.0 are absent, while every other payload (including NaN/Inf and
; subnormals) is present. Only integer instructions and the native word load
; are used; there is no host math, no float arithmetic, and no FMA.
;
; BFS is deterministic: mark source visited and enqueue it; while the FIFO is
; non-empty, dequeue i, scan j=0..255 in increasing order, and enqueue every
; unvisited j with A[i,j] != 0. A vertex is marked visited when it is
; enqueued, so every vertex enters the queue at most once and the queue can
; never hold more than 256 entries. queue_scratch[0..count-1] is left holding
; the deterministic enqueue order; entries at or beyond count are untouched.
;
; The source is validated before visited_out is touched, so an invalid source
; is rejected atomically. The overflow guards are defensive: for a valid
; non-aliasing caller they are unreachable. visited_out and queue_scratch must
; not overlap A or each other. The kernel is a leaf routine with no static
; scratch and does not use r15.

mth_reachability_256:
  ; Reject any source outside 0..255 before the first output write.
  MOVI r5,256
  CMP r2,r5
  JMP[GE] @mth_reachability_256_invalid

  ; visited_out[0..255] = +0.0.
  MOVI r6,0
mth_reachability_256_clear_visited:
  ADD r7,r3,r6
  ST r0,r7,0
  ADD r6,r6,r13
  CMP r6,r5
  JMP[LT] @mth_reachability_256_clear_visited

  ; visited_out[source] = 1.0; queue[0] = source.
  MOVI r11,0
  MOVHI r11,0x3F80
  ADD r7,r3,r2
  ST r11,r7,0
  ST r2,r4,0

  ; head = 0, tail = 1, reachable count = 1.
  MOVI r6,0
  MOVI r7,1
  MOVI r14,1

mth_reachability_256_bfs:
  ; Empty FIFO -> done.
  CMP r6,r7
  JMP[GE] @mth_reachability_256_done

  ; i = queue[head].
  ADD r8,r4,r6
  LD r8,r8,0
  CMP r8,r5
  JMP[GE] @mth_reachability_256_overflow

  ; Row pointer A + i*256, then scan j = 0..255.
  SHL r10,r8,8
  ADD r10,r1,r10
  MOVI r9,0

mth_reachability_256_scan:
  ; Numeric nonzero test: (±0.0) << 1 == 0, every other payload != 0.
  ADD r11,r10,r9
  LD r11,r11,0
  SHL r11,r11,1
  CMP r11,r0
  JMP[EQ] @mth_reachability_256_scan_next

  ; Skip a neighbor that is already visited.
  ADD r12,r3,r9
  LD r12,r12,0
  CMP r12,r0
  JMP[NE] @mth_reachability_256_scan_next

  ; Defensive queue-capacity guard. Unreachable for valid buffers because
  ; each of the 256 vertices is enqueued at most once.
  CMP r7,r5
  JMP[GE] @mth_reachability_256_overflow

  ; Enqueue j, then mark visited_out[j] = 1.0.
  ADD r12,r4,r7
  ST r9,r12,0
  ADD r7,r7,r13
  MOVI r12,0
  MOVHI r12,0x3F80
  ADD r11,r3,r9
  ST r12,r11,0
  ADD r14,r14,r13

mth_reachability_256_scan_next:
  ADD r9,r9,r13
  CMP r9,r5
  JMP[LT] @mth_reachability_256_scan

  ADD r6,r6,r13
  JMP @mth_reachability_256_bfs

mth_reachability_256_done:
  ADD r1,r14,r0
  RET

mth_reachability_256_overflow:
mth_reachability_256_invalid:
  MOVI r1,0
  RET
