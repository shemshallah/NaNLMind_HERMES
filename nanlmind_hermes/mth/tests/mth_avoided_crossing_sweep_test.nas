; MTH avoided-crossing sweep fixture — native NaN16 float32 only.
;
; Case 1: a=1, b=3, couplings [0, 1, 2], n=3.
;   c=0: lower=1, upper=3, gap=2 exactly.
;   c=1: lower=2-sqrt(2), upper=2+sqrt(2), gap=2*sqrt(2);
;        checked within 1e-5 (0x3727C5AC) of the float32 nearest values.
;   c=2: lower=2-sqrt(5), upper=2+sqrt(5), gap=2*sqrt(5);
;        checked within the same tolerance.
;
; Case 2: a=b=2, couplings [0, 0.5, 1, 2], n=4.
;   Zero coupling gives lower=upper=2, gap=0 exactly.
;   For c>0 the gap is 2c exactly: 1, 2, and 4.
;
; Case 3: a=b=2, couplings [0,1,...,255], n=256 (maximum sweep).
;   Every c is exact and c*c is an exact float32 integer, so
;   lower=2-i, upper=2+i, gap=2i exactly for every index.
;
; Step budget used by the runner: 1,000,000.

.org 0
JMP @mth_avoided_test_main
.include "../nan16/mth_avoided_crossing_sweep.nas"

mth_avoided_test_main:
  MOVI r13,1

  ; --- Case 1: a=1, b=3, couplings [0, 1, 2]. ---
  MOVI r4,@mth_avoided_couplings
  ST r0,r4,0
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,1                  ; 1.0
  MOVI r5,0
  MOVHI r5,0x4000
  ST r5,r4,2                  ; 2.0

  MOVI r1,@mth_avoided_a13
  MOVI r2,@mth_avoided_b13
  MOVI r3,@mth_avoided_couplings
  MOVI r4,3
  MOVI r5,@mth_avoided_gap
  MOVI r6,@mth_avoided_lower
  MOVI r7,@mth_avoided_upper
  CALL @mth_avoided_crossing_sweep
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=0: lower == 1 exactly.
  MOVI r4,@mth_avoided_lower
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_avoided_fail

  ; c=0: upper == 3 exactly.
  MOVI r4,@mth_avoided_upper
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_avoided_fail

  ; c=0: gap == 2 exactly.
  MOVI r4,@mth_avoided_gap
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_avoided_fail

  ; c=1: |lower - (2-sqrt(2))| <= 1e-5.
  MOVI r4,@mth_avoided_lower
  LD r1,r4,1
  MOVI r2,0xF61A
  MOVHI r2,0x3F15
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=1: |upper - (2+sqrt(2))| <= 1e-5.
  MOVI r4,@mth_avoided_upper
  LD r1,r4,1
  MOVI r2,0x827A
  MOVHI r2,0x405A
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=1: |gap - 2*sqrt(2)| <= 1e-5.
  MOVI r4,@mth_avoided_gap
  LD r1,r4,1
  MOVI r2,0x04F3
  MOVHI r2,0x4035
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=2: |lower - (2-sqrt(5))| <= 1e-5.
  MOVI r4,@mth_avoided_lower
  LD r1,r4,2
  MOVI r2,0xBBCE
  MOVHI r2,0xBE71
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=2: |upper - (2+sqrt(5))| <= 1e-5.
  MOVI r4,@mth_avoided_upper
  LD r1,r4,2
  MOVI r2,0x8DDE
  MOVHI r2,0x4087
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; c=2: |gap - 2*sqrt(5)| <= 1e-5.
  MOVI r4,@mth_avoided_gap
  LD r1,r4,2
  MOVI r2,0x1BBD
  MOVHI r2,0x408F
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_avoided_close
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; --- Case 2: a=b=2, couplings [0, 0.5, 1, 2]. ---
  MOVI r4,@mth_avoided_couplings
  ST r0,r4,0                  ; 0.0
  MOVI r5,0
  MOVHI r5,0x3F00
  ST r5,r4,1                  ; 0.5
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,2                  ; 1.0
  MOVI r5,0
  MOVHI r5,0x4000
  ST r5,r4,3                  ; 2.0

  MOVI r1,@mth_avoided_a22
  MOVI r2,@mth_avoided_b22
  MOVI r3,@mth_avoided_couplings
  MOVI r4,4
  MOVI r5,@mth_avoided_gap
  MOVI r6,@mth_avoided_lower
  MOVI r7,@mth_avoided_upper
  CALL @mth_avoided_crossing_sweep
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; Zero coupling: lower == upper == 2, gap == 0.
  MOVI r4,@mth_avoided_lower
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_avoided_fail
  MOVI r4,@mth_avoided_upper
  LD r5,r4,0
  CMP r5,r6
  JMP[NE] @mth_avoided_fail
  MOVI r4,@mth_avoided_gap
  LD r5,r4,0
  CMP r5,r0
  JMP[NE] @mth_avoided_fail

  ; c=0.5: lower == 1.5, upper == 2.5, gap == 1 exactly.
  MOVI r4,@mth_avoided_lower
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3FC0
  CMP r5,r6
  JMP[NE] @mth_avoided_fail
  MOVI r4,@mth_avoided_upper
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x4020
  CMP r5,r6
  JMP[NE] @mth_avoided_fail

  ; c=0.5, 1, 2: gap == 2c exactly (1, 2, 4).
  MOVI r4,@mth_avoided_gap
  LD r5,r4,1
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_avoided_fail
  LD r5,r4,2
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_avoided_fail
  LD r5,r4,3
  MOVI r6,0
  MOVHI r6,0x4080
  CMP r5,r6
  JMP[NE] @mth_avoided_fail

  ; --- Case 3: maximum sweep, a=b=2, couplings [0..255]. ---
  MOVI r1,@mth_avoided_couplings
  MOVI r2,0
  MOVI r3,256
mth_avoided_fill_couplings:
  CVTIF r4,r2
  ADD r5,r1,r2
  ST r4,r5,0
  ADD r2,r2,r13
  CMP r2,r3
  JMP[LT] @mth_avoided_fill_couplings

  MOVI r1,@mth_avoided_a22
  MOVI r2,@mth_avoided_b22
  MOVI r3,@mth_avoided_couplings
  MOVI r4,256
  MOVI r5,@mth_avoided_gap
  MOVI r6,@mth_avoided_lower
  MOVI r7,@mth_avoided_upper
  CALL @mth_avoided_crossing_sweep
  CMP r1,r13
  JMP[NE] @mth_avoided_fail

  ; Check every index: gap=2i, lower=2-i, upper=2+i exactly.
  MOVI r8,@mth_avoided_gap
  MOVI r9,@mth_avoided_lower
  MOVI r10,@mth_avoided_upper
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x4000             ; 2.0f
  MOVI r4,256
mth_avoided_check_sweep:
  CVTIF r5,r2                 ; i as float32
  FADD r6,r5,r5               ; expected gap = 2i
  ADD r11,r8,r2
  LD r11,r11,0
  CMP r11,r6
  JMP[NE] @mth_avoided_fail

  FSUB r12,r3,r5              ; expected lower = 2-i
  ADD r11,r9,r2
  LD r11,r11,0
  CMP r11,r12
  JMP[NE] @mth_avoided_fail

  FADD r12,r3,r5              ; expected upper = 2+i
  ADD r11,r10,r2
  LD r11,r11,0
  CMP r11,r12
  JMP[NE] @mth_avoided_fail

  ADD r2,r2,r13
  CMP r2,r4
  JMP[LT] @mth_avoided_check_sweep

  MOVI r1,@mth_avoided_pass
  CALL @mth_avoided_puts
  HALT

mth_avoided_fail:
  MOVI r1,@mth_avoided_bad
  CALL @mth_avoided_puts
  HALT

; r1=actual, r2=expected, r3=tolerance; returns r1=1 if
; |actual-expected| <= tolerance, else 0. Native float32 only.
mth_avoided_close:
  FSUB r4,r1,r2
  FCMP r4,r0
  JMP[GE] @mth_avoided_close_abs
  FSUB r4,r0,r4
mth_avoided_close_abs:
  FCMP r4,r3
  JMP[GT] @mth_avoided_close_bad
  MOVI r1,1
  RET
mth_avoided_close_bad:
  MOVI r1,0
  RET

mth_avoided_puts:
  MOVI r4,255
mth_avoided_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_avoided_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_avoided_puts_byte
  ADD r1,r1,r13
  JMP @mth_avoided_puts_word

.org 512
mth_avoided_a13:
.word 0x3F800000
.org 576
mth_avoided_b13:
.word 0x40400000
.org 640
mth_avoided_a22:
.word 0x40000000
.org 704
mth_avoided_b22:
.word 0x40000000
.org 1024
mth_avoided_couplings:
.word 0
.org 1536
mth_avoided_gap:
.word 0
.org 2048
mth_avoided_lower:
.word 0
.org 2560
mth_avoided_upper:
.word 0
.org 3584
mth_avoided_pass:
.str "MTH AVOIDED CROSSING SWEEP PASS\n"
mth_avoided_bad:
.str "MTH AVOIDED CROSSING SWEEP FAIL\n"
