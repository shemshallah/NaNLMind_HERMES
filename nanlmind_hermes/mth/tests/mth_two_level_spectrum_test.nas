; MTH two-level avoided-crossing spectrum fixture — native NaN16 float32 only.
;
; H(eta) = [[a, eta*c], [eta*c, b]].
;
; Case 1: a=1, b=3, c=1, eta=0
;   E_low = 1, E_high = 3, gap = 2, required exactly.
;
; Case 2: a=1, b=3, c=1, eta=1
;   E_low  = 2 - sqrt(2)  ~= 0.5857864618301392 (0x3F15F61A)
;   E_high = 2 + sqrt(2)  ~= 3.4142136573791504 (0x405A827A)
;   gap    = 2*sqrt(2)    ~= 2.8284270763397217 (0x403504F3)
;   each checked within 1e-5 (0x3727C5AC); E_low+E_high checked
;   against 4 within the same tolerance.
;
; Step budget: 1,000,000.

.org 0
JMP @mth_two_level_test_main
.include "../nan16/mth_two_level_spectrum.nas"

mth_two_level_test_main:
  MOVI r13,1

  ; --- Case eta=0: inputs are fixed in the data section. ---
  MOVI r1,@mth_two_level_a
  MOVI r2,@mth_two_level_b
  MOVI r3,@mth_two_level_c
  MOVI r4,@mth_two_level_eta
  MOVI r5,@mth_two_level_low
  MOVI r6,@mth_two_level_high
  MOVI r7,@mth_two_level_gap
  CALL @mth_two_level_spectrum
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  ; E_low == 1.0 exactly.
  MOVI r4,@mth_two_level_low
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_two_level_fail

  ; E_high == 3.0 exactly.
  MOVI r4,@mth_two_level_high
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4040
  CMP r5,r6
  JMP[NE] @mth_two_level_fail

  ; gap == 2.0 exactly.
  MOVI r4,@mth_two_level_gap
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_two_level_fail

  ; --- Case eta=1. ---
  MOVI r4,@mth_two_level_eta
  MOVI r5,0
  MOVHI r5,0x3F80
  ST r5,r4,0

  MOVI r1,@mth_two_level_a
  MOVI r2,@mth_two_level_b
  MOVI r3,@mth_two_level_c
  MOVI r4,@mth_two_level_eta
  MOVI r5,@mth_two_level_low
  MOVI r6,@mth_two_level_high
  MOVI r7,@mth_two_level_gap
  CALL @mth_two_level_spectrum
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  ; |E_low - (2-sqrt(2))| <= 1e-5.
  MOVI r4,@mth_two_level_low
  LD r1,r4,0
  MOVI r2,0xF61A
  MOVHI r2,0x3F15
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_two_level_close
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  ; |E_high - (2+sqrt(2))| <= 1e-5.
  MOVI r4,@mth_two_level_high
  LD r1,r4,0
  MOVI r2,0x827A
  MOVHI r2,0x405A
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_two_level_close
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  ; |gap - 2*sqrt(2)| <= 1e-5.
  MOVI r4,@mth_two_level_gap
  LD r1,r4,0
  MOVI r2,0x04F3
  MOVHI r2,0x4035
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_two_level_close
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  ; |E_low + E_high - 4| <= 1e-5.
  MOVI r4,@mth_two_level_low
  LD r1,r4,0
  MOVI r4,@mth_two_level_high
  LD r5,r4,0
  FADD r1,r1,r5
  MOVI r2,0
  MOVHI r2,0x4080
  MOVI r3,0xC5AC
  MOVHI r3,0x3727
  CALL @mth_two_level_close
  CMP r1,r13
  JMP[NE] @mth_two_level_fail

  MOVI r1,@mth_two_level_pass
  CALL @mth_two_level_puts
  HALT

mth_two_level_fail:
  MOVI r1,@mth_two_level_bad
  CALL @mth_two_level_puts
  HALT

; r1=actual, r2=expected, r3=tolerance; returns r1=1 if
; |actual-expected| <= tolerance, else 0. Native float32 only.
mth_two_level_close:
  FSUB r4,r1,r2
  FCMP r4,r0
  JMP[GE] @mth_two_level_close_abs
  FSUB r4,r0,r4
mth_two_level_close_abs:
  FCMP r4,r3
  JMP[GT] @mth_two_level_close_bad
  MOVI r1,1
  RET
mth_two_level_close_bad:
  MOVI r1,0
  RET

mth_two_level_puts:
  MOVI r4,255
mth_two_level_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_two_level_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_two_level_puts_byte
  ADD r1,r1,r13
  JMP @mth_two_level_puts_word

.org 512
mth_two_level_a:
.word 0x3F800000
.org 1024
mth_two_level_b:
.word 0x40400000
.org 1536
mth_two_level_c:
.word 0x3F800000
.org 2048
mth_two_level_eta:
.word 0x00000000
.org 2560
mth_two_level_low:
.word 0
.org 3072
mth_two_level_high:
.word 0
.org 3584
mth_two_level_gap:
.word 0
.org 4096
mth_two_level_pass:
.str "MTH TWO LEVEL SPECTRUM PASS\n"
mth_two_level_bad:
.str "MTH TWO LEVEL SPECTRUM FAIL\n"
