; MTH degree-2 polynomial graph filter native fixture — NaN16 only.
;
; Graph: path 0--1--2--3 plus isolated vertices 4..255.
; Signal x = [0, 4096, 2048, -4096, 0, ...] Q16.16.
;
; Exact hand-checkable vectors:
;   Lx   = [ -4096,  6144, 4096, -6144, 0, ...]
;   L^2x = [-10240, 12288, 8192, -10240, 0, ...]
;
; Cases:
;   1. c=[+1, +1/2, -1/4] -> y=[512, 4096, 2048, -4608, 0, ...].
;   2. c=[+1, 0, 0]       -> y=x.
;   3. c=[0, +1, 0]       -> y=Lx.
;   4. x[0]=4097 rejects before changing output or either scratch vector.
;   5. coefficient 65537 rejects before changing output.
;   6. decreasing CSR offset rejects with r1=2 and output unchanged.
;   7. duplicate endpoint rejects with r1=3 and output unchanged.
;   8. edge_count 65281 rejects before changing output.

.org 0
JMP @mth_filter_test_main
.include "../nan16/mth_graph_filter.nas"

mth_filter_test_main:
  MOVI r13,1

  ; Build the path CSR offsets [0,1,3,5,6,6,...,6].
  MOVI r1,@mth_filter_offsets
  ST r0,r1,0
  MOVI r2,1
  ST r2,r1,1
  MOVI r2,3
  ST r2,r1,2
  MOVI r2,5
  ST r2,r1,3
  MOVI r2,6
  ST r2,r1,4
  MOVI r2,5
mth_filter_fill_offsets:
  ADD r3,r1,r2
  MOVI r4,6
  ST r4,r3,0
  ADD r2,r2,r13
  MOVI r3,257
  CMP r2,r3
  JMP[LT] @mth_filter_fill_offsets

  ; Case 1: c=[+1, +1/2, -1/4].
  MOVI r1,@mth_filter_coeffs
  MOVI r2,0
  MOVHI r2,1
  ST r2,r1,0
  MOVI r2,32768
  ST r2,r1,1
  MOVI r2,0xC000
  MOVHI r2,0xFFFF
  ST r2,r1,2
  CALL @mth_filter_call
  CMP r1,r13
  JMP[NE] @mth_filter_test_fail

  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,512
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,1
  MOVI r8,4096
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,2
  MOVI r8,2048
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,3
  MOVI r8,0xEE00
  MOVHI r8,0xFFFF
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  CALL @mth_filter_check_zero_tail
  MOVI r1,@mth_filter_poly_ok
  CALL @mth_filter_test_puts

  ; Case 2: c=[+1, 0, 0] is the identity.
  MOVI r1,@mth_filter_coeffs
  MOVI r2,0
  MOVHI r2,1
  ST r2,r1,0
  MOVI r2,0
  ST r2,r1,1
  ST r2,r1,2
  CALL @mth_filter_call
  CMP r1,r13
  JMP[NE] @mth_filter_test_fail

  MOVI r6,@mth_filter_output
  LD r7,r6,0
  CMP r7,r0
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,1
  MOVI r8,4096
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,2
  MOVI r8,2048
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,3
  MOVI r8,0xF000
  MOVHI r8,0xFFFF
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  CALL @mth_filter_check_zero_tail
  MOVI r1,@mth_filter_identity_ok
  CALL @mth_filter_test_puts

  ; Case 3: c=[0, +1, 0] is Lx.
  MOVI r1,@mth_filter_coeffs
  MOVI r2,0
  ST r2,r1,0
  MOVI r2,0
  MOVHI r2,1
  ST r2,r1,1
  MOVI r2,0
  ST r2,r1,2
  CALL @mth_filter_call
  CMP r1,r13
  JMP[NE] @mth_filter_test_fail

  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,0xF000
  MOVHI r8,0xFFFF
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,1
  MOVI r8,6144
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,2
  MOVI r8,4096
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  LD r7,r6,3
  MOVI r8,0xE800
  MOVHI r8,0xFFFF
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  CALL @mth_filter_check_zero_tail
  MOVI r1,@mth_filter_laplacian_ok
  CALL @mth_filter_test_puts

  ; Case 4: out-of-range x rejects before output/scratch writes.
  MOVI r6,@mth_filter_output
  MOVI r7,77
  ST r7,r6,0
  MOVI r8,255
  ADD r8,r6,r8
  ST r7,r8,0
  MOVI r6,@mth_filter_lx
  ST r7,r6,0
  MOVI r6,@mth_filter_l2x
  ST r7,r6,0
  MOVI r6,@mth_filter_x
  MOVI r7,4097
  ST r7,r6,0
  CALL @mth_filter_call
  CMP r1,r0
  JMP[NE] @mth_filter_test_fail

  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r9,255
  ADD r6,r6,r9
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_lx
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_l2x
  LD r7,r6,0
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_x
  ST r0,r6,0

  ; Case 5: coefficient 65537 rejects before output writes.
  MOVI r6,@mth_filter_coeffs
  MOVI r7,1
  MOVHI r7,1
  ST r7,r6,0
  MOVI r6,@mth_filter_output
  MOVI r7,77
  ST r7,r6,0
  CALL @mth_filter_call
  CMP r1,r0
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail

  ; Restore valid identity coefficients for the CSR rejection cases.
  MOVI r6,@mth_filter_coeffs
  MOVI r7,0
  MOVHI r7,1
  ST r7,r6,0
  MOVI r7,0
  ST r7,r6,1
  ST r7,r6,2

  ; Case 6: decreasing offsets[1]=4 > offsets[2]=3 rejects with r1=2.
  MOVI r6,@mth_filter_offsets
  MOVI r7,4
  ST r7,r6,1
  MOVI r6,@mth_filter_output
  MOVI r7,77
  ST r7,r6,0
  CALL @mth_filter_call
  MOVI r7,2
  CMP r1,r7
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_offsets
  MOVI r7,1
  ST r7,r6,1

  ; Case 7: duplicate endpoint rejects with r1=3.
  MOVI r6,@mth_filter_neighbors
  MOVI r7,1
  ST r7,r6,1
  MOVI r6,@mth_filter_output
  MOVI r7,77
  ST r7,r6,0
  CALL @mth_filter_call
  MOVI r7,3
  CMP r1,r7
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_neighbors
  ST r0,r6,1

  ; Case 8: edge_count 65281 rejects before output writes.
  MOVI r6,@mth_filter_output
  MOVI r7,77
  ST r7,r6,0
  MOVI r1,@mth_filter_offsets
  MOVI r2,@mth_filter_neighbors
  MOVI r3,@mth_filter_x
  MOVI r4,@mth_filter_coeffs
  MOVI r5,@mth_filter_output
  MOVI r6,@mth_filter_lx
  MOVI r7,@mth_filter_l2x
  MOVI r8,65281
  MOVI r9,@mth_filter_scratch
  CALL @mth_graph_filter_poly2_q16_16
  CMP r1,r0
  JMP[NE] @mth_filter_test_fail
  MOVI r6,@mth_filter_output
  LD r7,r6,0
  MOVI r8,77
  CMP r7,r8
  JMP[NE] @mth_filter_test_fail

  MOVI r1,@mth_filter_test_pass
  CALL @mth_filter_test_puts
  HALT

mth_filter_test_fail:
  MOVI r1,@mth_filter_test_bad
  CALL @mth_filter_test_puts
  HALT

; Call the filter with the standard fixture pages and edge_count=6.
; The filter clobbers r14, so the wrapper's own return address is saved in
; control-scratch word 0 (the filter never touches word 0).
mth_filter_call:
  MOVI r14,@mth_filter_scratch
  ST r15,r14,0
  MOVI r1,@mth_filter_offsets
  MOVI r2,@mth_filter_neighbors
  MOVI r3,@mth_filter_x
  MOVI r4,@mth_filter_coeffs
  MOVI r5,@mth_filter_output
  MOVI r6,@mth_filter_lx
  MOVI r7,@mth_filter_l2x
  MOVI r8,6
  MOVI r9,@mth_filter_scratch
  CALL @mth_graph_filter_poly2_q16_16
  MOVI r14,@mth_filter_scratch
  LD r15,r14,0
  RET

; Verify output[4..255] are exactly zero, leaving r13=1.
mth_filter_check_zero_tail:
  MOVI r6,@mth_filter_output
  MOVI r7,4
mth_filter_check_tail_loop:
  ADD r8,r6,r7
  LD r9,r8,0
  CMP r9,r0
  JMP[NE] @mth_filter_test_fail
  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_filter_check_tail_loop
  RET

mth_filter_test_puts:
  MOVI r4,255
mth_filter_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_filter_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_filter_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_filter_test_puts_word

.org 2048
mth_filter_offsets:
.org 2560
mth_filter_neighbors:
.word 1
.word 0
.word 2
.word 1
.word 3
.word 2
.org 3072
mth_filter_x:
.word 0
.word 4096
.word 2048
.word 0xFFFFF000
.org 3584
mth_filter_coeffs:
.word 0
.word 0
.word 0
.org 3840
mth_filter_output:
.org 4352
mth_filter_lx:
.org 5120
mth_filter_l2x:
.org 6144
mth_filter_scratch:
.org 6400
mth_filter_test_pass:
.str "MTH GRAPH FILTER Q16.16 PASS\n"
mth_filter_test_bad:
.str "MTH GRAPH FILTER Q16.16 FAIL\n"
mth_filter_poly_ok:
.str "POLY PATH OK\n"
mth_filter_identity_ok:
.str "IDENTITY OK\n"
mth_filter_laplacian_ok:
.str "LAPLACIAN OK\n"
