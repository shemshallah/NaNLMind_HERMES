; MTH vector argmax256 float32 fixture — NaN16 only.
;
; Phases:
;   1. x[i] = (float)(i - 128), distinct maximum at index 255 (127.0).
;   2. maximum tie at indices 10, 77, 200 (5.0); index 10 must win.
;   3. maximum tie among negative values at 3, 8, 100 (-2.0); index 3 wins.
;   4. all-equal positive vector (3.25); index 0 wins.
;   5. all-equal negative vector (-2.5); index 0 wins.
;   6. signed zero, -0.0 first at index 3, +0.0 at 100; index 3 and the
;      exact -0.0 payload must be returned.
;   7. signed zero, +0.0 first at index 3, -0.0 at 100; index 3 and the
;      exact +0.0 payload must be returned.
;   8. signaling NaN at the last index: rejection, outputs untouched.
;   9. negative NaN at index 0: rejection, outputs untouched.
;  10. +inf at index 42: rejection, outputs untouched.
;  11. -inf at index 128: rejection, outputs untouched.
;  12. restoring the distinct vector recovers the index-255 result.
;
; Expected values are IEEE binary32 payload constants, and every check is a
; native integer CMP. No host math is used.

.org 0
JMP @mth_argmax256_test_main
.include "../nan16/mth_vector_argmax256_f32.nas"

mth_argmax256_test_main:
  MOVI r13,1

  ; Phase 1: distinct maximum at index 255, value 127.0.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,255
  MOVI r2,0
  MOVHI r2,0x42FE            ; 127.0
  CALL @mth_argmax256_test_check_result

  ; Phase 2: three-way maximum tie at 10, 77, 200; first index wins.
  MOVI r1,0
  MOVHI r1,0xBF80            ; -1.0 baseline
  CALL @mth_argmax256_test_fill_value
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0x40A0            ; 5.0
  ST r6,r5,10
  ST r6,r5,77
  ST r6,r5,200
  MOVI r6,0
  MOVHI r6,0x4080            ; 4.0, below the tie
  ST r6,r5,255
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,10
  MOVI r2,0
  MOVHI r2,0x40A0            ; 5.0
  CALL @mth_argmax256_test_check_result

  ; Phase 3: negative maximum tie at 3, 8, 100; first index wins.
  MOVI r1,0
  MOVHI r1,0xC110            ; -9.0 baseline
  CALL @mth_argmax256_test_fill_value
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0xC000            ; -2.0
  ST r6,r5,3
  ST r6,r5,8
  ST r6,r5,100
  MOVI r6,0
  MOVHI r6,0xC040            ; -3.0, below the tie
  ST r6,r5,200
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,3
  MOVI r2,0
  MOVHI r2,0xC000            ; -2.0
  CALL @mth_argmax256_test_check_result

  ; Phase 4: all elements 3.25; index 0 and x[0] win.
  MOVI r1,0
  MOVHI r1,0x4050            ; 3.25
  CALL @mth_argmax256_test_fill_value
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,0
  MOVI r2,0
  MOVHI r2,0x4050            ; 3.25
  CALL @mth_argmax256_test_check_result

  ; Phase 5: all elements -2.5; index 0 and x[0] win.
  MOVI r1,0
  MOVHI r1,0xC020            ; -2.5
  CALL @mth_argmax256_test_fill_value
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,0
  MOVI r2,0
  MOVHI r2,0xC020            ; -2.5
  CALL @mth_argmax256_test_check_result

  ; Phase 6: -1.0 baseline, -0.0 at 3 and +0.0 at 100. The zeros are equal
  ; and greater than -1.0, so index 3 wins and its -0.0 payload is preserved.
  MOVI r1,0
  MOVHI r1,0xBF80            ; -1.0 baseline
  CALL @mth_argmax256_test_fill_value
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0x8000            ; -0.0
  ST r6,r5,3
  ST r0,r5,100               ; +0.0
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,3
  MOVI r2,0
  MOVHI r2,0x8000            ; -0.0
  CALL @mth_argmax256_test_check_result

  ; Phase 7: swap the zero signs. +0.0 at 3 is still the first zero, so it
  ; wins and its +0.0 payload is preserved.
  LA r5,@mth_argmax256_test_x
  ST r0,r5,3                 ; +0.0
  MOVI r6,0
  MOVHI r6,0x8000            ; -0.0
  ST r6,r5,100
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,3
  MOVI r2,0
  CALL @mth_argmax256_test_check_result

  ; Phase 8: signaling NaN at the last index. Reject atomically.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_fill_sentinels
  LA r5,@mth_argmax256_test_x
  MOVI r6,1
  MOVHI r6,0x7F80            ; 0x7F800001 sNaN
  ST r6,r5,255
  CALL @mth_argmax256_test_call
  CMP r1,r0
  JMP[NE] @mth_argmax256_test_fail
  CALL @mth_argmax256_test_check_sentinels

  ; Phase 9: negative quiet NaN at the first index. Reject atomically.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_fill_sentinels
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0xFFC0            ; 0xFFC00000 qNaN
  ST r6,r5,0
  CALL @mth_argmax256_test_call
  CMP r1,r0
  JMP[NE] @mth_argmax256_test_fail
  CALL @mth_argmax256_test_check_sentinels

  ; Phase 10: +inf at index 42. Reject atomically.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_fill_sentinels
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0x7F80            ; +inf
  ST r6,r5,42
  CALL @mth_argmax256_test_call
  CMP r1,r0
  JMP[NE] @mth_argmax256_test_fail
  CALL @mth_argmax256_test_check_sentinels

  ; Phase 11: -inf at index 128. Reject atomically.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_fill_sentinels
  LA r5,@mth_argmax256_test_x
  MOVI r6,0
  MOVHI r6,0xFF80            ; -inf
  ST r6,r5,128
  CALL @mth_argmax256_test_call
  CMP r1,r0
  JMP[NE] @mth_argmax256_test_fail
  CALL @mth_argmax256_test_check_sentinels

  ; Phase 12: recovery after rejection.
  CALL @mth_argmax256_test_fill_iota
  CALL @mth_argmax256_test_call
  CMP r1,r13
  JMP[NE] @mth_argmax256_test_fail
  MOVI r1,255
  MOVI r2,0
  MOVHI r2,0x42FE            ; 127.0
  CALL @mth_argmax256_test_check_result

  LA r1,@mth_argmax256_test_pass
  CALL @mth_argmax256_test_puts
  HALT

mth_argmax256_test_fail:
  LA r1,@mth_argmax256_test_bad
  CALL @mth_argmax256_test_puts
  HALT

; Call the argmax kernel on the fixture x/output spans and return its r1
; status. Saves and restores the link register through r14; the kernel
; preserves r14. Clobbers r1-r3, r14 and the flags.
mth_argmax256_test_call:
  ADD r14,r15,r0
  LA r1,@mth_argmax256_test_x
  LA r2,@mth_argmax256_test_index
  LA r3,@mth_argmax256_test_value
  CALL @mth_vector_argmax256_f32
  ADD r15,r14,r0
  RET

; r1 = expected index, r2 = expected value payload. Jumps to the fail label
; on mismatch. Clobbers r5-r7 and the flags.
mth_argmax256_test_check_result:
  LA r5,@mth_argmax256_test_index
  LD r6,r5,0
  CMP r6,r1
  JMP[NE] @mth_argmax256_test_fail
  LA r5,@mth_argmax256_test_value
  LD r6,r5,0
  CMP r6,r2
  JMP[NE] @mth_argmax256_test_fail
  RET

; Fill x[0..255] with the payload in r1. Preserves r1; clobbers r2-r4.
mth_argmax256_test_fill_value:
  LA r2,@mth_argmax256_test_x
  MOVI r3,0
  MOVI r4,256
mth_argmax256_test_fill_value_loop:
  ST r1,r2,0
  ADD r2,r2,r13
  ADD r3,r3,r13
  CMP r3,r4
  JMP[LT] @mth_argmax256_test_fill_value_loop
  RET

; Fill x[i] = (float32)(i - 128) for i=0..255. Clobbers r1-r4.
mth_argmax256_test_fill_iota:
  LA r1,@mth_argmax256_test_x
  MOVI r2,0
mth_argmax256_test_fill_iota_loop:
  MOVI r3,128
  SUB r4,r2,r3
  CVTIF r4,r4
  ST r4,r1,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  MOVI r3,256
  CMP r2,r3
  JMP[LT] @mth_argmax256_test_fill_iota_loop
  RET

; Fill both output words with 0xDEADBEEF. Clobbers r5-r6.
mth_argmax256_test_fill_sentinels:
  MOVI r5,0xDEAD
  MOVHI r5,0xDEAD
  LA r6,@mth_argmax256_test_index
  ST r5,r6,0
  LA r6,@mth_argmax256_test_value
  ST r5,r6,0
  RET

; Require both output words to remain 0xDEADBEEF. Clobbers r5-r7.
mth_argmax256_test_check_sentinels:
  MOVI r5,0xDEAD
  MOVHI r5,0xDEAD
  LA r6,@mth_argmax256_test_index
  LD r7,r6,0
  CMP r7,r5
  JMP[NE] @mth_argmax256_test_fail
  LA r6,@mth_argmax256_test_value
  LD r7,r6,0
  CMP r7,r5
  JMP[NE] @mth_argmax256_test_fail
  RET

mth_argmax256_test_puts:
  MOVI r4,255
mth_argmax256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_argmax256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_argmax256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_argmax256_test_puts_word

.org 1024
mth_argmax256_test_x:
.word 0
.org 1280
mth_argmax256_test_index:
.word 0
.org 1281
mth_argmax256_test_value:
.word 0
.org 1536
mth_argmax256_test_pass:
.str "MTH VECTOR ARGMAX256 F32 PASS\n"
mth_argmax256_test_bad:
.str "MTH VECTOR ARGMAX256 F32 FAIL\n"
