; MTH vector scale256 float32 fixture — NaN16 only.
;
; Every all-element phase checks all 256 output words with integer CMP.
; Phase 1: x=2.0, scalar=3.0 -> 6.0 everywhere.
; Phase 2: x=2.0, scalar=-3.0 -> -6.0 everywhere.
; Phase 3: x=2.0, scalar=+0.0 -> +0.0 everywhere.
; Phase 4: x=2.0, scalar=-0.0 -> -0.0 everywhere.
; Phase 5: x=-2.0, scalar=+0.0 -> -0.0 everywhere.
; Phase 6: x=-2.0, scalar=-0.0 -> +0.0 everywhere.
; Phase 7: infinity products -> +inf/-inf everywhere.
; Phase 8: x=+0.0 and x=-0.0 with scalar=+inf -> canonical 0xFFC00000.
; Phase 9: all-256 NaN checks, once from x and once from the scalar.
; Phase 10: seven scalar scenarios x eight x[i] policy cases, including
;           x-NaN priority, scalar-NaN quieting, signed infinities, signed
;           zeros, and 0*inf invalid results.
; Phase 11: x[i]=1+2^-23, scalar=1+2^-23 -> 1+2^-22 everywhere (RNE).
;
; Expected values are literal IEEE binary32 payloads. No host math is used by
; the guest or by this fixture.

.org 0
JMP @mth_scale_main
.include "../nan16/mth_vector_scale256_f32.nas"

mth_scale_main:
  MOVI r13,1

  ; ---- Phase 1: 2.0 * 3.0 = 6.0, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x4000          ; 2.0
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x4040          ; 3.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x40C0          ; 6.0
  CALL @mth_scale_check

  ; ---- Phase 2: 2.0 * -3.0 = -6.0, all 256 elements.
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0xC040          ; -3.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xC0C0          ; -6.0
  CALL @mth_scale_check

  ; ---- Phase 3: 2.0 * +0.0 = +0.0, all 256 elements.
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  CALL @mth_scale_check

  ; ---- Phase 4: 2.0 * -0.0 = -0.0, all 256 elements.
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x8000          ; -0.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x8000          ; -0.0
  CALL @mth_scale_check

  ; ---- Phase 5: -2.0 * +0.0 = -0.0, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xC000          ; -2.0
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  ST r4,r1,0               ; +0.0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x8000          ; -0.0
  CALL @mth_scale_check

  ; ---- Phase 6: -2.0 * -0.0 = +0.0, all 256 elements.
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x8000          ; -0.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  CALL @mth_scale_check

  ; ---- Phase 7a: +inf * +2.0 = +inf, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x7F80          ; +inf
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x4000          ; 2.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x7F80          ; +inf
  CALL @mth_scale_check

  ; ---- Phase 7b: -inf * +2.0 = -inf, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xFF80          ; -inf
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xFF80          ; -inf
  CALL @mth_scale_check

  ; ---- Phase 8a: +0.0 * +inf = canonical invalid NaN, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x7F80          ; +inf
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xFFC0          ; 0xFFC00000
  CALL @mth_scale_check

  ; ---- Phase 8b: -0.0 * +inf = canonical invalid NaN, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x8000          ; -0.0
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0xFFC0          ; 0xFFC00000
  CALL @mth_scale_check

  ; ---- Phase 9a: x=0x7FC00123 (qNaN), scalar=2.0 -> qNaN bits preserved.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0x0123
  MOVHI r4,0x7FC0          ; 0x7FC00123
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,0
  MOVHI r4,0x4000          ; 2.0
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,0x0123
  MOVHI r4,0x7FC0          ; 0x7FC00123
  CALL @mth_scale_check

  ; ---- Phase 9b: x=2.0, scalar=0x7F800001 (sNaN) -> quieted scalar NaN.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,0
  MOVHI r4,0x4000          ; 2.0
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,1
  MOVHI r4,0x7F80          ; 0x7F800001
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,1
  MOVHI r4,0x7FC0          ; 0x7FC00001
  CALL @mth_scale_check

  ; ---- Phase 10: eight x[i] policy words under seven scalar scenarios.
  MOVI r1,@mth_scale_x
  MOVI r5,@mth_scale_case_x
  MOVI r2,8
  CALL @mth_scale_copy

  MOVI r8,@mth_scale_scenario_scalars
  MOVI r9,@mth_scale_scenario_expects
  MOVI r10,7
mth_scale_scenario_loop:
  LD r4,r8,0
  MOVI r1,@mth_scale_scalar
  ST r4,r1,0

  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail

  MOVI r1,@mth_scale_out
  LD r5,r9,0
  MOVI r2,8
  CALL @mth_scale_check_cases

  ADD r8,r8,r13
  ADD r9,r9,r13
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_scale_scenario_loop

  ; ---- Phase 11: RNE rounding, (1+2^-23)^2 -> 1+2^-22, all 256 elements.
  MOVI r1,@mth_scale_x
  MOVI r2,256
  MOVI r4,1
  MOVHI r4,0x3F80          ; 0x3F800001 = 1 + 2^-23
  CALL @mth_scale_fill
  MOVI r1,@mth_scale_scalar
  MOVI r4,1
  MOVHI r4,0x3F80          ; 0x3F800001
  ST r4,r1,0
  MOVI r1,@mth_scale_x
  MOVI r2,@mth_scale_scalar
  MOVI r3,@mth_scale_out
  CALL @mth_vector_scale256_f32
  CMP r1,r13
  JMP[NE] @mth_scale_fail
  MOVI r1,@mth_scale_out
  MOVI r2,256
  MOVI r4,2
  MOVHI r4,0x3F80          ; 0x3F800002 = 1 + 2^-22
  CALL @mth_scale_check

  MOVI r1,@mth_scale_pass
  CALL @mth_scale_puts
  HALT

mth_scale_fail:
  MOVI r1,@mth_scale_bad
  CALL @mth_scale_puts
  HALT

; r1 = destination, r2 = count, r4 = payload. Clobbers r1, r2.
mth_scale_fill:
  ST r4,r1,0
  ADD r1,r1,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_scale_fill
  RET

; r1 = output, r2 = count, r4 = expected payload. Clobbers r1, r2, r5.
mth_scale_check:
  LD r5,r1,0
  CMP r5,r4
  JMP[NE] @mth_scale_fail
  ADD r1,r1,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_scale_check
  RET

; r1 = destination, r5 = source, r2 = count. Clobbers r1, r2, r5, r6.
mth_scale_copy:
  LD r6,r5,0
  ST r6,r1,0
  ADD r1,r1,r13
  ADD r5,r5,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_scale_copy
  RET

; r1 = output, r5 = expected array, r2 = count. Clobbers r1, r2, r5-r7.
mth_scale_check_cases:
  LD r6,r1,0
  LD r7,r5,0
  CMP r6,r7
  JMP[NE] @mth_scale_fail
  ADD r1,r1,r13
  ADD r5,r5,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_scale_check_cases
  RET

mth_scale_puts:
  MOVI r4,255
mth_scale_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_scale_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_scale_puts_byte
  ADD r1,r1,r13
  JMP @mth_scale_puts_word

.org 1024
mth_scale_x:
.word 0
.org 2048
mth_scale_out:
.word 0
.org 3072
mth_scale_scalar:
.word 0
.org 4096
mth_scale_case_x:
.word 0x7F800001          ; +signalling NaN
.word 0xFF800002          ; -signalling NaN
.word 0x7FC00003          ; +quiet NaN
.word 0xFFC00004          ; -quiet NaN
.word 0x7F800000          ; +inf
.word 0xFF800000          ; -inf
.word 0x3F800001          ; 1 + 2^-23
.word 0x80000000          ; -0.0
.org 4128
mth_scale_scenario_scalars:
.word 0x40000000          ; +2.0
.word 0x7F800001          ; +signalling NaN
.word 0xFF800002          ; -signalling NaN
.word 0x7F800000          ; +inf
.word 0xFF800000          ; -inf
.word 0x00000000          ; +0.0
.word 0x80000000          ; -0.0
.org 4160
mth_scale_scenario_expects:
.word @mth_scale_case_out_a
.word @mth_scale_case_out_b
.word @mth_scale_case_out_c
.word @mth_scale_case_out_d
.word @mth_scale_case_out_e
.word @mth_scale_case_out_f
.word @mth_scale_case_out_g
.org 4352
mth_scale_case_out_a:       ; scalar +2.0
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0x7F800000
.word 0xFF800000
.word 0x40000001
.word 0x80000000
.org 4384
mth_scale_case_out_b:       ; scalar +signalling NaN
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0x7FC00001
.word 0x7FC00001
.word 0x7FC00001
.word 0x7FC00001
.org 4416
mth_scale_case_out_c:       ; scalar -signalling NaN
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0xFFC00002
.word 0xFFC00002
.word 0xFFC00002
.word 0xFFC00002
.org 4448
mth_scale_case_out_d:       ; scalar +inf
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0x7F800000
.word 0xFF800000
.word 0x7F800000
.word 0xFFC00000
.org 4480
mth_scale_case_out_e:       ; scalar -inf
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0xFF800000
.word 0x7F800000
.word 0xFF800000
.word 0xFFC00000
.org 4512
mth_scale_case_out_f:       ; scalar +0.0
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0xFFC00000
.word 0xFFC00000
.word 0x00000000
.word 0x80000000
.org 4544
mth_scale_case_out_g:       ; scalar -0.0
.word 0x7FC00001
.word 0xFFC00002
.word 0x7FC00003
.word 0xFFC00004
.word 0xFFC00000
.word 0xFFC00000
.word 0x80000000
.word 0x00000000
.org 4608
mth_scale_pass:
.str "MTH VECTOR SCALE256 F32 PASS\n"
mth_scale_bad:
.str "MTH VECTOR SCALE256 F32 FAIL\n"
