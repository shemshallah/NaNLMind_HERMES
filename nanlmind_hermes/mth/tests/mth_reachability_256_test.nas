; MTH R5 reachability 256 fixture - NaN16 only.
; Exercises directed chain, middle source, FIFO/increasing-j order, a
; disconnected cycle plus an isolated vertex, a self-loop, the valid upper
; source boundary, and atomic rejection of source 256.
;
; Layout:
;   code + puts helper   word 0
;   A                    65,536 words at 512
;   visited_out          256 words at 66,048
;   queue_scratch        256 words at 66,304
;   PASS/FAIL strings    word 66,560
; A starts zeroed; each graph case clears A and writes only its edges.

.org 0
JMP @mth_reachability_256_test_main
.include "../nan16/mth_reachability_256.nas"

mth_reachability_256_test_main:
  MOVI r13,1

  ; ---------------------------------------------------------------
  ; Case 1: directed chain 0->1->...->255, source 0.
  ; Expect count 256, every visited flag 1.0, queue = 0..255.
  ; ---------------------------------------------------------------
  CALL @mth_reachability_256_test_clear_a
  CALL @mth_reachability_256_test_set_chain

  LA r1,@mth_reachability_256_test_a
  MOVI r2,0
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  MOVI r9,256
  CMP r1,r9
  JMP[NE] @mth_reachability_256_test_fail

  ; Every visited_out word must be exactly 1.0 (0x3F800000).
  LA r5,@mth_reachability_256_test_visited
  MOVI r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  MOVI r9,256
mth_reachability_256_test_case1_visited:
  LD r7,r5,0
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case1_visited

  ; FIFO + increasing-j on a chain leaves queue[i] == i.
  LA r5,@mth_reachability_256_test_queue
  MOVI r6,0
  MOVI r9,256
mth_reachability_256_test_case1_queue:
  LD r7,r5,0
  CMP r7,r6
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case1_queue

  ; ---------------------------------------------------------------
  ; Case 2: same chain, source 128. Directed reachability is
  ; 128..255 only, so the count must be 128.
  ; ---------------------------------------------------------------
  LA r1,@mth_reachability_256_test_a
  MOVI r2,128
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  MOVI r9,128
  CMP r1,r9
  JMP[NE] @mth_reachability_256_test_fail

  ; visited_out[0..127] == 0.0.
  LA r5,@mth_reachability_256_test_visited
  MOVI r6,0
  MOVI r9,128
mth_reachability_256_test_case2_zero:
  LD r7,r5,0
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case2_zero

  ; visited_out[128..255] == 1.0.
  MOVI r6,0
  MOVI r9,128
  MOVI r8,0
  MOVHI r8,0x3F80
mth_reachability_256_test_case2_one:
  LD r7,r5,0
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case2_one

  ; ---------------------------------------------------------------
  ; Case 2b: source 255 (valid upper boundary). Expect count 1.
  ; ---------------------------------------------------------------
  LA r1,@mth_reachability_256_test_a
  MOVI r2,255
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  CMP r1,r13
  JMP[NE] @mth_reachability_256_test_fail
  LA r5,@mth_reachability_256_test_visited
  LD r7,r5,254
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail
  LD r7,r5,255
  MOVI r8,0
  MOVHI r8,0x3F80
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail

  ; ---------------------------------------------------------------
  ; Case 3: branching graph for FIFO order.
  ;   0->1, 0->2, 1->3, 1->4, 2->5
  ; Increasing-j BFS enqueues 0,1,2,3,4,5. A decreasing-j scan
  ; would leave 0,2,1,5,4,3, so this checks the specified order.
  ; ---------------------------------------------------------------
  CALL @mth_reachability_256_test_clear_a
  LA r5,@mth_reachability_256_test_a
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r5,1
  ST r7,r5,2
  ST r7,r5,259
  ST r7,r5,260
  ST r7,r5,517

  LA r1,@mth_reachability_256_test_a
  MOVI r2,0
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  MOVI r9,6
  CMP r1,r9
  JMP[NE] @mth_reachability_256_test_fail

  ; visited_out[0..5] == 1.0.
  LA r5,@mth_reachability_256_test_visited
  MOVI r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  MOVI r9,6
mth_reachability_256_test_case3_one:
  LD r7,r5,0
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case3_one

  ; visited_out[6..255] == 0.0.
  MOVI r6,0
  MOVI r9,250
mth_reachability_256_test_case3_zero:
  LD r7,r5,0
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case3_zero

  ; queue_scratch[0..5] == 0,1,2,3,4,5.
  LA r5,@mth_reachability_256_test_queue
  MOVI r6,0
  MOVI r9,6
mth_reachability_256_test_case3_queue:
  LD r7,r5,0
  CMP r7,r6
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case3_queue

  ; ---------------------------------------------------------------
  ; Case 4: disconnected directed cycle 4->5->6->7->4 and an
  ; isolated vertex 9. Source 4 => count 4, 9 stays unvisited.
  ; Source 9 => count 1 and the cycle stays unvisited.
  ; ---------------------------------------------------------------
  CALL @mth_reachability_256_test_clear_a
  LA r5,@mth_reachability_256_test_a
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r5,1029
  ST r7,r5,1286
  ST r7,r5,1543
  ST r7,r5,1796

  LA r1,@mth_reachability_256_test_a
  MOVI r2,4
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  MOVI r9,4
  CMP r1,r9
  JMP[NE] @mth_reachability_256_test_fail

  ; visited_out[4..7] == 1.0.
  LA r5,@mth_reachability_256_test_visited
  ADDI r5,r5,4
  MOVI r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  MOVI r9,4
mth_reachability_256_test_case4_cycle:
  LD r7,r5,0
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case4_cycle

  ; The isolated vertex 9 is not reachable from 4.
  LA r5,@mth_reachability_256_test_visited
  LD r7,r5,9
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail

  ; Source 9 => count 1; visited_out[9] == 1.0, visited_out[4] == 0.0.
  LA r1,@mth_reachability_256_test_a
  MOVI r2,9
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  CMP r1,r13
  JMP[NE] @mth_reachability_256_test_fail
  LA r5,@mth_reachability_256_test_visited
  LD r7,r5,9
  MOVI r8,0
  MOVHI r8,0x3F80
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  LD r7,r5,4
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail

  ; ---------------------------------------------------------------
  ; Case 5: self-loop only at vertex 3. Expect count 1 and no
  ; duplicate enqueue from A[3,3] != 0.
  ; ---------------------------------------------------------------
  CALL @mth_reachability_256_test_clear_a
  LA r5,@mth_reachability_256_test_a
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r5,771

  LA r1,@mth_reachability_256_test_a
  MOVI r2,3
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  CMP r1,r13
  JMP[NE] @mth_reachability_256_test_fail

  LA r5,@mth_reachability_256_test_visited
  LD r7,r5,3
  MOVI r8,0
  MOVHI r8,0x3F80
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail
  LD r7,r5,2
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail
  LD r7,r5,4
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail

  ; queue[0] == 3 and exactly one enqueue.
  LA r5,@mth_reachability_256_test_queue
  LD r7,r5,0
  MOVI r8,3
  CMP r7,r8
  JMP[NE] @mth_reachability_256_test_fail

  ; ---------------------------------------------------------------
  ; Case 5b: -0.0 is numerically zero and must not be an edge.
  ; A[3,4] = 0x80000000; source 3 must still reach only itself.
  ; ---------------------------------------------------------------
  CALL @mth_reachability_256_test_clear_a
  LA r5,@mth_reachability_256_test_a
  MOVI r7,0
  MOVHI r7,0x8000
  ST r7,r5,772

  LA r1,@mth_reachability_256_test_a
  MOVI r2,3
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  CMP r1,r13
  JMP[NE] @mth_reachability_256_test_fail
  LA r5,@mth_reachability_256_test_visited
  LD r7,r5,4
  CMP r7,r0
  JMP[NE] @mth_reachability_256_test_fail

  ; ---------------------------------------------------------------
  ; Case 6: invalid source 256. Poison visited_out and queue_scratch
  ; first; the rejection must return 0 without touching either.
  ; ---------------------------------------------------------------
  LA r5,@mth_reachability_256_test_visited
  MOVI r6,0
  MOVI r7,77
  MOVI r9,256
mth_reachability_256_test_case6_poison_visited:
  ST r7,r5,0
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case6_poison_visited

  LA r5,@mth_reachability_256_test_queue
  MOVI r6,0
  MOVI r7,99
mth_reachability_256_test_case6_poison_queue:
  ST r7,r5,0
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case6_poison_queue

  LA r1,@mth_reachability_256_test_a
  MOVI r2,256
  LA r3,@mth_reachability_256_test_visited
  LA r4,@mth_reachability_256_test_queue
  CALL @mth_reachability_256
  CMP r1,r0
  JMP[NE] @mth_reachability_256_test_fail

  ; Every visited_out word is still 77.
  LA r5,@mth_reachability_256_test_visited
  MOVI r6,0
  MOVI r7,77
  MOVI r9,256
mth_reachability_256_test_case6_check_visited:
  LD r8,r5,0
  CMP r8,r7
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case6_check_visited

  ; Every queue_scratch word is still 99.
  LA r5,@mth_reachability_256_test_queue
  MOVI r6,0
  MOVI r7,99
mth_reachability_256_test_case6_check_queue:
  LD r8,r5,0
  CMP r8,r7
  JMP[NE] @mth_reachability_256_test_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  CMP r6,r9
  JMP[LT] @mth_reachability_256_test_case6_check_queue

  LA r1,@mth_reachability_256_test_pass
  CALL @mth_reachability_256_test_puts
  HALT

mth_reachability_256_test_fail:
  LA r1,@mth_reachability_256_test_bad
  CALL @mth_reachability_256_test_puts
  HALT

; ---------------------------------------------------------------
; Fixture helpers.
; ---------------------------------------------------------------
mth_reachability_256_test_clear_a:
  LA r5,@mth_reachability_256_test_a
  MOVI r6,0
  MOVHI r6,0x0001
mth_reachability_256_test_clear_a_loop:
  ST r0,r5,0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_reachability_256_test_clear_a_loop
  RET

mth_reachability_256_test_set_chain:
  ; A[i,i+1] = 1.0 for i=0..254, flat offsets 1,258,...,65279.
  LA r5,@mth_reachability_256_test_a
  MOVI r6,255
  MOVI r7,1
  MOVI r8,0
  MOVHI r8,0x3F80
mth_reachability_256_test_set_chain_loop:
  ADD r9,r5,r7
  ST r8,r9,0
  ADDI r7,r7,257
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_reachability_256_test_set_chain_loop
  RET

mth_reachability_256_test_puts:
  MOVI r4,255
mth_reachability_256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_reachability_256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_reachability_256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_reachability_256_test_puts_word

; ---------------------------------------------------------------
; Storage. A occupies 512..66,047; visited 66,048..66,303;
; queue 66,304..66,559; strings from 66,560.
; ---------------------------------------------------------------
.org 512
mth_reachability_256_test_a:
.word 0

.org 66048
mth_reachability_256_test_visited:
.word 0

.org 66304
mth_reachability_256_test_queue:
.word 0

.org 66560
mth_reachability_256_test_pass:
.str "MTH REACHABILITY 256 PASS\n"
mth_reachability_256_test_bad:
.str "MTH REACHABILITY 256 FAIL\n"
