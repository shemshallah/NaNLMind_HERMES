; MTH regularized void potential native fixture — NaN16 only.
;
; Covers the regularized core-potential contract:
;   1. alpha=1, eps=1 over 256 radii: V(0)=1, V(1)=0.5, V(2)=0.2 (exact bits).
;      All other radii are 0 and therefore also produce 1.0.
;   2. alpha=0 produces exact +0.0 in all 256 outputs.
;   3. epsilon=0 rejects with the full output span unchanged.
;   4. a negative radius rejects with the full output span unchanged.
;   5. non-finite radius/alpha/epsilon rejects without changing output.
;
; Fixed non-overlapping regions: radii at word 512, output at word 768
; (256 words each), strings at word 1024. r13=1 throughout.

.org 0
JMP @mth_graph_void_test_main
.include "../nan16/mth_graph_void.nas"

mth_graph_void_test_main:
  MOVI r13,1

  ; Fill all 256 radii with +0.0.
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  MOVI r10,256
mth_graph_void_test_init_radii:
  ST r6,r5,0
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_init_radii

  ; radii[1]=1.0, radii[2]=2.0.
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r5,1
  MOVI r6,0
  MOVHI r6,0x4000
  ST r6,r5,2

  ; Case 1: alpha=1.0, epsilon=1.0.
  MOVI r1,@mth_graph_void_radii
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x3F80
  MOVI r4,@mth_graph_void_out
  CALL @mth_graph_void_potential
  CMP r1,r13
  JMP[NE] @mth_graph_void_test_fail

  MOVI r5,@mth_graph_void_out
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x3F80
  CMP r6,r7
  JMP[NE] @mth_graph_void_test_fail

  LD r6,r5,1
  MOVI r7,0
  MOVHI r7,0x3F00
  CMP r6,r7
  JMP[NE] @mth_graph_void_test_fail

  LD r6,r5,2
  MOVI r7,0xCCCD
  MOVHI r7,0x3E4C
  CMP r6,r7
  JMP[NE] @mth_graph_void_test_fail

  ; Outputs 3..255 used r=0 and must be exactly 1.0.
  MOVI r8,3
  ADD r5,r5,r8
  MOVI r10,253
mth_graph_void_test_check_ones:
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x3F80
  CMP r6,r7
  JMP[NE] @mth_graph_void_test_fail
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_check_ones

  ; Case 2: alpha=0.0 produces exact +0.0 everywhere.
  MOVI r1,@mth_graph_void_radii
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F80
  MOVI r4,@mth_graph_void_out
  CALL @mth_graph_void_potential
  CMP r1,r13
  JMP[NE] @mth_graph_void_test_fail
  MOVI r5,@mth_graph_void_out
  MOVI r10,256
mth_graph_void_test_check_zero:
  LD r6,r5,0
  CMP r6,r0
  JMP[NE] @mth_graph_void_test_fail
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_check_zero

  ; Case 3: epsilon=0 rejection, output fully atomic.
  ; Fill the output span with a sentinel that no valid result can imitate.
  MOVI r5,@mth_graph_void_out
  MOVI r6,0x5678
  MOVHI r6,0x1234
  MOVI r10,256
mth_graph_void_test_fill_sentinel:
  ST r6,r5,0
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_fill_sentinel

  MOVI r1,@mth_graph_void_radii
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVI r4,@mth_graph_void_out
  CALL @mth_graph_void_potential
  CMP r1,r0
  JMP[NE] @mth_graph_void_test_fail

  MOVI r5,@mth_graph_void_out
  MOVI r6,0x5678
  MOVHI r6,0x1234
  MOVI r10,256
mth_graph_void_test_check_sentinel_eps:
  LD r7,r5,0
  CMP r7,r6
  JMP[NE] @mth_graph_void_test_fail
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_check_sentinel_eps

  ; Case 4: negative radius rejection, output fully atomic.
  ; The output is still the verified sentinel from case 3.
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  MOVHI r6,0xBF80
  ST r6,r5,0

  MOVI r1,@mth_graph_void_radii
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x3F80
  MOVI r4,@mth_graph_void_out
  CALL @mth_graph_void_potential
  CMP r1,r0
  JMP[NE] @mth_graph_void_test_fail

  MOVI r5,@mth_graph_void_out
  MOVI r6,0x5678
  MOVHI r6,0x1234
  MOVI r10,256
mth_graph_void_test_check_sentinel_negative:
  LD r7,r5,0
  CMP r7,r6
  JMP[NE] @mth_graph_void_test_fail
  ADD r5,r5,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_graph_void_test_check_sentinel_negative

  ; Case 5: non-finite radius inspection.
  ; radii[0]=+infinity.
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  MOVHI r6,0x7F80
  ST r6,r5,0
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_graph_void_test_expect_reject

  ; radii[0]=quiet NaN (0x7FC00000).
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  MOVHI r6,0x7FC0
  ST r6,r5,0
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_graph_void_test_expect_reject

  ; Restore radii[0]=0.0. alpha=+infinity is invalid.
  MOVI r5,@mth_graph_void_radii
  MOVI r6,0
  ST r6,r5,0
  MOVI r2,0
  MOVHI r2,0x7F80
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_graph_void_test_expect_reject

  ; alpha=NaN is invalid.
  MOVI r2,0
  MOVHI r2,0x7FC0
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_graph_void_test_expect_reject

  ; alpha=-1.0 is invalid.
  MOVI r2,0
  MOVHI r2,0xBF80
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_graph_void_test_expect_reject

  ; epsilon=+infinity is invalid.
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x7F80
  CALL @mth_graph_void_test_expect_reject

  ; epsilon=NaN is invalid.
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0x7FC0
  CALL @mth_graph_void_test_expect_reject

  ; epsilon=-1.0 is invalid.
  MOVI r2,0
  MOVHI r2,0x3F80
  MOVI r3,0
  MOVHI r3,0xBF80
  CALL @mth_graph_void_test_expect_reject

  MOVI r1,@mth_graph_void_test_pass
  CALL @mth_graph_void_test_puts
  HALT

; Expect the kernel to reject invalid parameters/radius. The output span is
; already the verified sentinel from cases 3/4; this checks the first word,
; while cases 3/4 check all 256 words. alpha and epsilon arrive in r2/r3.
mth_graph_void_test_expect_reject:
  MOVI r5,@mth_graph_void_test_link
  ST r15,r5,0
  MOVI r1,@mth_graph_void_radii
  MOVI r4,@mth_graph_void_out
  CALL @mth_graph_void_potential
  CMP r1,r0
  JMP[NE] @mth_graph_void_test_fail
  MOVI r5,@mth_graph_void_out
  LD r6,r5,0
  MOVI r7,0x5678
  MOVHI r7,0x1234
  CMP r6,r7
  JMP[NE] @mth_graph_void_test_fail
  MOVI r5,@mth_graph_void_test_link
  LD r15,r5,0
  RET

mth_graph_void_test_fail:
  MOVI r1,@mth_graph_void_test_bad
  CALL @mth_graph_void_test_puts
  HALT

mth_graph_void_test_puts:
  MOVI r4,255
mth_graph_void_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_graph_void_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_graph_void_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_graph_void_test_puts_word

; Fixed, non-overlapping 256-word test regions; strings begin at word 1024.
; The link word holds the helper's return address while the kernel's CALL/RET
; reuses r15.
.org 508
mth_graph_void_test_link:
.word 0
.org 512
mth_graph_void_radii:
.word 0
.org 768
mth_graph_void_out:
.word 0
.org 1024
mth_graph_void_test_pass:
.str "MTH GRAPH VOID POTENTIAL PASS\n"
mth_graph_void_test_bad:
.str "MTH GRAPH VOID POTENTIAL FAIL\n"
