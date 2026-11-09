.org 0
JMP @mth_radial_energy_test_main
.include "../nan16/mth_radial_energy.nas"

; Fixture for mth_radial_energy.
;
; Every case uses diag/offdiag/x spans at fixed, non-overlapping addresses,
; calls the kernel once, checks the return value, and then checks both scalar
; outputs. Integer-valued expectations are exact float32 payload comparisons;
; the non-power-of-two localization ratios are checked with a native float32
; absolute tolerance of 2^-20.
;
; Case 1 note: for the path operator diag=2, offdiag=-1, x=1, every interior
; row contributes zero to x^T T x, but the two boundary rows contribute 1
; each. The correct quadratic energy is therefore 2, not 0. A periodic ring
; would need a 256th off-diagonal and is outside this 255-coupling ABI.

mth_radial_energy_test_main:
  MOVI r13,1

  ; ---- Case 1: diag=2, offdiag=-1, x=1 -> energy=2, localization=1/256. ----
  LA r1,@mth_radial_energy_test_diag
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x4000
  CALL @mth_radial_energy_test_fill

  LA r1,@mth_radial_energy_test_offdiag
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0xBF80
  CALL @mth_radial_energy_test_fill

  LA r1,@mth_radial_energy_test_x
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_radial_energy_test_fill

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4000          ; 2.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3B80          ; 1/256
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  ; ABI preservation after the first call: r2-r5 must still hold the four
  ; input/output pointers and r13 must still be 1.
  LA r6,@mth_radial_energy_test_offdiag
  CMP r2,r6
  JMP[NE] @mth_radial_energy_test_fail
  LA r6,@mth_radial_energy_test_x
  CMP r3,r6
  JMP[NE] @mth_radial_energy_test_fail
  LA r6,@mth_radial_energy_test_energy_out
  CMP r4,r6
  JMP[NE] @mth_radial_energy_test_fail
  LA r6,@mth_radial_energy_test_loc_out
  CMP r5,r6
  JMP[NE] @mth_radial_energy_test_fail
  CMP r13,r1               ; r1 is the verified success value 1
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 2: x=e0 -> energy=2, localization=1. ----
  LA r1,@mth_radial_energy_test_x
  MOVI r2,256
  MOVI r3,0
  CALL @mth_radial_energy_test_fill

  LA r6,@mth_radial_energy_test_x
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r6,0

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4000          ; 2.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3F80          ; 1.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 3: x=e128 -> energy=2, localization=1. ----
  LA r1,@mth_radial_energy_test_x
  MOVI r2,256
  MOVI r3,0
  CALL @mth_radial_energy_test_fill

  LA r6,@mth_radial_energy_test_x
  MOVI r8,128
  ADD r6,r6,r8
  MOVI r7,0
  MOVHI r7,0x3F80
  ST r7,r6,0

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x4000          ; 2.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3F80          ; 1.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 4: x=0 -> energy=0, localization=0 (zero-denominator path). ----
  LA r1,@mth_radial_energy_test_x
  MOVI r2,256
  MOVI r3,0
  CALL @mth_radial_energy_test_fill

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  CMP r7,r0
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r7,r6,0
  CMP r7,r0
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 5: deterministic sawtooth x[i]=(i mod 7)-3.
  ; Sums: sum x^2 = 1022, max x^2 = 9, sum x[i]*x[i+1] = 260.
  ; energy = 2*1022 + 2*(-1)*260 = 1524; localization = 9/1022.
  LA r4,@mth_radial_energy_test_x
  MOVI r5,0
  MOVHI r5,0xC040          ; -3.0
  MOVI r6,0
  MOVHI r6,0xC000          ; -2.0
  MOVI r7,0
  MOVHI r7,0xBF80          ; -1.0
  MOVI r8,0                ;  0.0
  MOVI r9,0
  MOVHI r9,0x3F80          ; +1.0
  MOVI r10,0
  MOVHI r10,0x4000         ; +2.0
  MOVI r11,0
  MOVHI r11,0x4040         ; +3.0
  MOVI r12,36
  MOVI r3,7
mth_radial_energy_test_saw_loop:
  ST r5,r4,0
  ST r6,r4,1
  ST r7,r4,2
  ST r8,r4,3
  ST r9,r4,4
  ST r10,r4,5
  ST r11,r4,6
  ADD r4,r4,r3
  SUB r12,r12,r13
  CMP r12,r0
  JMP[GT] @mth_radial_energy_test_saw_loop
  ST r5,r4,0
  ST r6,r4,1
  ST r7,r4,2
  ST r8,r4,3

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,1524
  CVTIF r8,r8
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r1,r6,0               ; actual
  MOVI r2,9
  CVTIF r2,r2
  MOVI r3,1022
  CVTIF r3,r3
  FDIV r2,r2,r3            ; expected 9/1022
  MOVI r3,0
  MOVHI r3,0x3580          ; tolerance 2^-20
  CALL @mth_radial_energy_test_close
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 6: linear ramp x[i]=i+1 with the same path operator.
  ; Every interior row cancels exactly; y[0]=0 and y[255]=257, so the
  ; quadratic energy is x[255]*y[255] = 256*257 = 65792.
  LA r5,@mth_radial_energy_test_x
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x3F80          ; 1.0
mth_radial_energy_test_ramp_loop:
  CVTIF r8,r6
  FADD r8,r8,r7
  ST r8,r5,0
  ADD r6,r6,r13
  ADD r5,r5,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_radial_energy_test_ramp_loop

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,256
  CVTIF r8,r8
  MOVI r9,257
  CVTIF r9,r9
  FMUL r8,r8,r9            ; 65792.0
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r1,r6,0               ; actual
  MOVI r2,0
  MOVHI r2,0x0001          ; 65536 = 0x00010000
  CVTIF r2,r2
  MOVI r3,0xD580
  MOVHI r3,0x0055          ; 5625216 = 0x0055D580
  CVTIF r3,r3
  FDIV r2,r2,r3            ; expected 65536/5625216
  MOVI r3,0
  MOVHI r3,0x3580          ; tolerance 2^-20
  CALL @mth_radial_energy_test_close
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  ; ---- Case 7: non-constant diag and offdiag, x=1.
  ; diag[i]=i+1; offdiag[i]=-1 for even i and -2 for odd i.
  ; sum diag = 256*257/2 = 32896.
  ; sum offdiag = 128*(-1) + 127*(-2) = -382.
  ; energy = 32896 + 2*(-382) = 32132; localization = 1/256.
  LA r5,@mth_radial_energy_test_diag
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x3F80          ; 1.0
mth_radial_energy_test_diag_ramp_loop:
  CVTIF r8,r6
  FADD r8,r8,r7
  ST r8,r5,0
  ADD r6,r6,r13
  ADD r5,r5,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_radial_energy_test_diag_ramp_loop

  LA r5,@mth_radial_energy_test_offdiag
  MOVI r6,255
  MOVI r7,0
  MOVHI r7,0xBF80          ; -1.0
  MOVI r8,0
  MOVHI r8,0xC000          ; -2.0
mth_radial_energy_test_off_alt_loop:
  ST r7,r5,0
  ADD r9,r7,r0
  ADD r7,r8,r0
  ADD r8,r9,r0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_radial_energy_test_off_alt_loop

  LA r1,@mth_radial_energy_test_x
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_radial_energy_test_fill

  LA r1,@mth_radial_energy_test_diag
  LA r2,@mth_radial_energy_test_offdiag
  LA r3,@mth_radial_energy_test_x
  LA r4,@mth_radial_energy_test_energy_out
  LA r5,@mth_radial_energy_test_loc_out
  CALL @mth_radial_energy
  CMP r1,r13
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_energy_out
  LD r7,r6,0
  MOVI r8,32132
  CVTIF r8,r8
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r6,@mth_radial_energy_test_loc_out
  LD r7,r6,0
  MOVI r8,0
  MOVHI r8,0x3B80          ; 1/256
  CMP r7,r8
  JMP[NE] @mth_radial_energy_test_fail

  LA r1,@mth_radial_energy_test_pass
  CALL @mth_radial_energy_test_puts
  HALT

mth_radial_energy_test_fail:
  LA r1,@mth_radial_energy_test_bad
  CALL @mth_radial_energy_test_puts
  HALT

; Fill r2 words at r1 with the float32 payload in r3.
mth_radial_energy_test_fill:
  CMP r2,r0
  RET[EQ]
mth_radial_energy_test_fill_loop:
  ST r3,r1,0
  ADD r1,r1,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_radial_energy_test_fill_loop
  RET

; r1=actual, r2=expected, r3=tolerance; returns r1=1 if
; |actual-expected| <= tolerance, else 0. Native float32 only.
mth_radial_energy_test_close:
  FSUB r6,r1,r2
  FCMP r6,r0
  JMP[GE] @mth_radial_energy_test_close_abs
  FSUB r6,r0,r6
mth_radial_energy_test_close_abs:
  FCMP r6,r3
  JMP[GT] @mth_radial_energy_test_close_bad
  MOVI r1,1
  RET
mth_radial_energy_test_close_bad:
  MOVI r1,0
  RET

mth_radial_energy_test_puts:
  MOVI r4,255
mth_radial_energy_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_radial_energy_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_radial_energy_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_radial_energy_test_puts_word

.org 1024
mth_radial_energy_test_diag:
.word 0
.org 1280
mth_radial_energy_test_offdiag:
.word 0
.org 1536
mth_radial_energy_test_x:
.word 0
.org 1792
mth_radial_energy_test_energy_out:
.word 77
mth_radial_energy_test_loc_out:
.word 77
.org 2048
mth_radial_energy_test_pass:
.str "MTH RADIAL ENERGY PASS\n"
mth_radial_energy_test_bad:
.str "MTH RADIAL ENERGY FAIL\n"
