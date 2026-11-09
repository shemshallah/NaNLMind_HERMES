; MTH 256-point direct DFT reference fixture — NaN16 only.
; Exercises exact DC/Nyquist bins, all-bin impulse responses, and the
; caller-supplied twiddle table. The twiddle table is generated in-guest
; from W^1 by the complex recurrence, then bins 0/64/128/192 are forced
; to their exact float32 values.
.org 0
JMP @mth_dft_test_main
.include "../nan16/mth_graph_dft256.nas"

mth_dft_test_main:
  MOVI r13,1
  CALL @mth_dft_build_twiddles

  ; ---- impulse at n=0: X[k] = 1+0i for every k ----
  MOVI r1,@mth_dft_xr
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xi
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xr
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,0

  MOVI r1,@mth_dft_xr
  MOVI r2,@mth_dft_xi
  MOVI r3,@mth_dft_twr
  MOVI r4,@mth_dft_twi
  MOVI r5,@mth_dft_Xr
  MOVI r6,@mth_dft_Xi
  CALL @mth_graph_dft256

  MOVI r7,@mth_dft_Xr
  MOVI r8,@mth_dft_Xi
  MOVI r9,256
  MOVI r10,0
  MOVHI r10,0x3F80
mth_dft_imp0_loop:
  LD r1,r7,0
  CMP r1,r10
  JMP[NE] @mth_dft_fail
  LD r1,r8,0
  CMP r1,r0
  JMP[NE] @mth_dft_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,-1
  CMP r9,r0
  JMP[GT] @mth_dft_imp0_loop

  ; ---- impulse at n=1: X[k] equals the supplied W^k exactly ----
  MOVI r1,@mth_dft_xr
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xi
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xr
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,1

  MOVI r1,@mth_dft_xr
  MOVI r2,@mth_dft_xi
  MOVI r3,@mth_dft_twr
  MOVI r4,@mth_dft_twi
  MOVI r5,@mth_dft_Xr
  MOVI r6,@mth_dft_Xi
  CALL @mth_graph_dft256

  MOVI r7,@mth_dft_Xr
  MOVI r8,@mth_dft_Xi
  MOVI r9,@mth_dft_twr
  MOVI r10,@mth_dft_twi
  MOVI r11,256
mth_dft_imp1_loop:
  LD r1,r7,0
  LD r2,r9,0
  CMP r1,r2
  JMP[NE] @mth_dft_fail
  LD r1,r8,0
  LD r2,r10,0
  CMP r1,r2
  JMP[NE] @mth_dft_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,1
  ADDI r10,r10,1
  ADDI r11,r11,-1
  CMP r11,r0
  JMP[GT] @mth_dft_imp1_loop

  ; ---- DC: X[0]=256, exact zeros at 64/128/192, rest near zero ----
  MOVI r1,@mth_dft_xr
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xi
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill

  MOVI r1,@mth_dft_xr
  MOVI r2,@mth_dft_xi
  MOVI r3,@mth_dft_twr
  MOVI r4,@mth_dft_twi
  MOVI r5,@mth_dft_Xr
  MOVI r6,@mth_dft_Xi
  CALL @mth_graph_dft256

  MOVI r1,0
  MOVI r2,0
  MOVHI r2,0x4380
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,64
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,128
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,192
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,1
  MOVI r2,255
  MOVI r3,0xD70A
  MOVHI r3,0x3C23
  CALL @mth_dft_check_range_tol

  ; ---- Nyquist: x[n]=(-1)^n, X[128]=256, rest near zero ----
  MOVI r1,@mth_dft_xr
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_dft_fill
  MOVI r1,@mth_dft_xr
  ADDI r1,r1,1
  MOVI r2,128
  MOVI r3,0
  MOVHI r3,0xBF80
mth_dft_nyq_fill:
  ST r3,r1,0
  ADDI r1,r1,2
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_dft_nyq_fill
  MOVI r1,@mth_dft_xi
  MOVI r2,256
  MOVI r3,0
  CALL @mth_dft_fill

  MOVI r1,@mth_dft_xr
  MOVI r2,@mth_dft_xi
  MOVI r3,@mth_dft_twr
  MOVI r4,@mth_dft_twi
  MOVI r5,@mth_dft_Xr
  MOVI r6,@mth_dft_Xi
  CALL @mth_graph_dft256

  MOVI r1,128
  MOVI r2,0
  MOVHI r2,0x4380
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,0
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,64
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,192
  MOVI r2,0
  MOVI r3,0
  CALL @mth_dft_expect
  MOVI r1,0
  MOVI r2,128
  MOVI r3,0xD70A
  MOVHI r3,0x3C23
  CALL @mth_dft_check_range_tol
  MOVI r1,129
  MOVI r2,127
  MOVI r3,0xD70A
  MOVHI r3,0x3C23
  CALL @mth_dft_check_range_tol

  MOVI r1,@mth_dft_pass
  CALL @mth_dft_puts
  HALT

mth_dft_fail:
  MOVI r1,@mth_dft_bad
  CALL @mth_dft_puts
  HALT

; ---- fill r2 words at r1 with bits r3 ----
mth_dft_fill:
  CMP r2,r0
  RET[EQ]
mth_dft_fill_loop:
  ST r3,r1,0
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_dft_fill_loop
  RET

; ---- exact check: X[r1] must equal (r2,r3) bit-for-bit ----
mth_dft_expect:
  MOVI r4,@mth_dft_Xr
  ADD r4,r4,r1
  LD r5,r4,0
  CMP r5,r2
  JMP[NE] @mth_dft_fail
  MOVI r4,@mth_dft_Xi
  ADD r4,r4,r1
  LD r5,r4,0
  CMP r5,r3
  JMP[NE] @mth_dft_fail
  RET

; ---- |Re| and |Im| <= r3 for r2 bins starting at r1 ----
mth_dft_check_range_tol:
  MOVI r7,@mth_dft_Xr
  ADD r7,r7,r1
  MOVI r8,@mth_dft_Xi
  ADD r8,r8,r1
  ADD r9,r2,r0
  ADD r10,r3,r0
mth_dft_range_tol_loop:
  CMP r9,r0
  RET[EQ]
  LD r1,r7,0
  SHL r4,r1,1
  SHR r4,r4,1
  CMP r4,r10
  JMP[GT] @mth_dft_fail
  LD r1,r8,0
  SHL r4,r1,1
  SHR r4,r4,1
  CMP r4,r10
  JMP[GT] @mth_dft_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,-1
  JMP @mth_dft_range_tol_loop

; ---- build W^k table by recurrence, then force exact quarter turns ----
mth_dft_build_twiddles:
  MOVI r1,@mth_dft_twr
  MOVI r2,@mth_dft_twi
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,0
  MOVI r3,0
  ST r3,r2,0
  ADDI r1,r1,1
  ADDI r2,r2,1

  ; W^1 = cos(2*pi/256) - i*sin(2*pi/256), rounded to float32.
  MOVI r4,0xEC43
  MOVHI r4,0x3F7F
  MOVI r5,0x0AB0
  MOVHI r5,0xBCC9
  MOVI r6,0xEC43
  MOVHI r6,0x3F7F
  MOVI r7,0x0AB0
  MOVHI r7,0xBCC9
  MOVI r8,255
mth_dft_tw_loop:
  ST r4,r1,0
  ST r5,r2,0
  FMUL r9,r4,r6
  FMUL r10,r5,r7
  FSUB r9,r9,r10
  FMUL r10,r4,r7
  FMUL r11,r5,r6
  FADD r10,r10,r11
  ADD r4,r9,r0
  ADD r5,r10,r0
  ADDI r1,r1,1
  ADDI r2,r2,1
  ADDI r8,r8,-1
  CMP r8,r0
  JMP[GT] @mth_dft_tw_loop

  MOVI r1,@mth_dft_twr
  MOVI r2,@mth_dft_twi
  MOVI r3,0
  MOVI r4,0
  MOVHI r4,0xBF80
  ST r3,r1,64
  ST r4,r2,64
  MOVI r5,0
  MOVHI r5,0xBF80
  ST r5,r1,128
  ST r3,r2,128
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r3,r1,192
  ST r6,r2,192
  RET

mth_dft_puts:
  MOVI r4,255
mth_dft_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_dft_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_dft_puts_byte
  ADD r1,r1,r13
  JMP @mth_dft_puts_word

mth_dft_pass:
.str "MTH GRAPH DFT256 PASS\n"
mth_dft_bad:
.str "MTH GRAPH DFT256 FAIL\n"

; Six distinct 256-word float32 arrays: 1,536 words.
.org 4096
mth_dft_twr:
.org 4352
mth_dft_twi:
.org 4608
mth_dft_xr:
.org 4864
mth_dft_xi:
.org 5120
mth_dft_Xr:
.org 5376
mth_dft_Xi:
.org 5632
