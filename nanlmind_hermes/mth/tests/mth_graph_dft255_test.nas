; MTH 255-point direct DFT reference fixture — NaN16 only.
; Exercises exact DC/impulse responses and the caller-supplied twiddle
; table. The twiddle table is generated in-guest from W^1 by the complex
; recurrence for k=1..127, then completed by conjugate symmetry
; W^k = conj(W^(255-k)) for k=128..254. This keeps recurrence drift out
; of the upper half of the table while using only FADD/FSUB/FMUL.
.org 0
JMP @mth_dft255_test_main
.include "../nan16/mth_graph_dft255.nas"

mth_dft255_test_main:
  MOVI r0,0
  MOVI r13,1
  CALL @mth_dft255_build_twiddles

  ; ---- impulse at n=0: X[k] = 1+0i for every k ----
  MOVI r1,@mth_dft255_xr
  MOVI r2,255
  MOVI r3,0
  CALL @mth_dft255_fill
  MOVI r1,@mth_dft255_xi
  MOVI r2,255
  MOVI r3,0
  CALL @mth_dft255_fill
  MOVI r1,@mth_dft255_xr
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,0

  MOVI r1,@mth_dft255_xr
  MOVI r2,@mth_dft255_xi
  MOVI r3,@mth_dft255_twr
  MOVI r4,@mth_dft255_twi
  MOVI r5,@mth_dft255_Xr
  MOVI r6,@mth_dft255_Xi
  CALL @mth_graph_dft255

  MOVI r7,@mth_dft255_Xr
  MOVI r8,@mth_dft255_Xi
  MOVI r9,255
  MOVI r10,0
  MOVHI r10,0x3F80
mth_dft255_imp0_loop:
  LD r1,r7,0
  CMP r1,r10
  JMP[NE] @mth_dft255_fail
  LD r1,r8,0
  CMP r1,r0
  JMP[NE] @mth_dft255_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,-1
  CMP r9,r0
  JMP[GT] @mth_dft255_imp0_loop

  ; ---- impulse at n=1: X[k] equals the supplied W^k exactly ----
  MOVI r1,@mth_dft255_xr
  MOVI r2,255
  MOVI r3,0
  CALL @mth_dft255_fill
  MOVI r1,@mth_dft255_xi
  MOVI r2,255
  MOVI r3,0
  CALL @mth_dft255_fill
  MOVI r1,@mth_dft255_xr
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,1

  MOVI r1,@mth_dft255_xr
  MOVI r2,@mth_dft255_xi
  MOVI r3,@mth_dft255_twr
  MOVI r4,@mth_dft255_twi
  MOVI r5,@mth_dft255_Xr
  MOVI r6,@mth_dft255_Xi
  CALL @mth_graph_dft255

  MOVI r7,@mth_dft255_Xr
  MOVI r8,@mth_dft255_Xi
  MOVI r9,@mth_dft255_twr
  MOVI r10,@mth_dft255_twi
  MOVI r11,255
mth_dft255_imp1_loop:
  LD r1,r7,0
  LD r2,r9,0
  CMP r1,r2
  JMP[NE] @mth_dft255_fail
  LD r1,r8,0
  LD r2,r10,0
  CMP r1,r2
  JMP[NE] @mth_dft255_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,1
  ADDI r10,r10,1
  ADDI r11,r11,-1
  CMP r11,r0
  JMP[GT] @mth_dft255_imp1_loop

  ; ---- DC: X[0]=255, all other bins near zero (|Re|,|Im| <= 0.01) ----
  MOVI r1,@mth_dft255_xr
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_dft255_fill
  MOVI r1,@mth_dft255_xi
  MOVI r2,255
  MOVI r3,0
  CALL @mth_dft255_fill

  MOVI r1,@mth_dft255_xr
  MOVI r2,@mth_dft255_xi
  MOVI r3,@mth_dft255_twr
  MOVI r4,@mth_dft255_twi
  MOVI r5,@mth_dft255_Xr
  MOVI r6,@mth_dft255_Xi
  CALL @mth_graph_dft255

  MOVI r1,0
  MOVI r2,0
  MOVHI r2,0x437F           ; 255.0f
  MOVI r3,0
  CALL @mth_dft255_expect
  MOVI r1,1
  MOVI r2,254
  MOVI r3,0xD70A
  MOVHI r3,0x3C23           ; ~0.00999999f
  CALL @mth_dft255_check_range_tol

  MOVI r1,@mth_dft255_pass
  CALL @mth_dft255_puts
  HALT

mth_dft255_fail:
  MOVI r1,@mth_dft255_bad
  CALL @mth_dft255_puts
  HALT

; ---- fill r2 words at r1 with bits r3 ----
mth_dft255_fill:
  CMP r2,r0
  RET[EQ]
mth_dft255_fill_loop:
  ST r3,r1,0
  ADDI r1,r1,1
  ADDI r2,r2,-1
  CMP r2,r0
  JMP[GT] @mth_dft255_fill_loop
  RET

; ---- exact check: X[r1] must equal (r2,r3) bit-for-bit ----
mth_dft255_expect:
  MOVI r4,@mth_dft255_Xr
  ADD r4,r4,r1
  LD r5,r4,0
  CMP r5,r2
  JMP[NE] @mth_dft255_fail
  MOVI r4,@mth_dft255_Xi
  ADD r4,r4,r1
  LD r5,r4,0
  CMP r5,r3
  JMP[NE] @mth_dft255_fail
  RET

; ---- |Re| and |Im| <= r3 for r2 bins starting at r1 ----
mth_dft255_check_range_tol:
  MOVI r7,@mth_dft255_Xr
  ADD r7,r7,r1
  MOVI r8,@mth_dft255_Xi
  ADD r8,r8,r1
  ADD r9,r2,r0
  ADD r10,r3,r0
mth_dft255_range_tol_loop:
  CMP r9,r0
  RET[EQ]
  LD r1,r7,0
  SHL r4,r1,1
  SHR r4,r4,1
  CMP r4,r10
  JMP[GT] @mth_dft255_fail
  LD r1,r8,0
  SHL r4,r1,1
  SHR r4,r4,1
  CMP r4,r10
  JMP[GT] @mth_dft255_fail
  ADDI r7,r7,1
  ADDI r8,r8,1
  ADDI r9,r9,-1
  JMP @mth_dft255_range_tol_loop

; ---- build W^k: recurrence for k=1..127, then conjugate mirror ----
mth_dft255_build_twiddles:
  MOVI r1,@mth_dft255_twr
  MOVI r2,@mth_dft255_twi
  MOVI r3,0
  MOVHI r3,0x3F80
  ST r3,r1,0                ; W^0 = 1 + 0i
  MOVI r3,0
  ST r3,r2,0
  ADDI r1,r1,1
  ADDI r2,r2,1

  ; W^1 = cos(2*pi/255) - i*sin(2*pi/255), rounded to float32.
  MOVI r4,0xEC1B
  MOVHI r4,0x3F7F
  MOVI r5,0xD47A
  MOVHI r5,0xBCC9
  MOVI r6,0xEC1B
  MOVHI r6,0x3F7F
  MOVI r7,0xD47A
  MOVHI r7,0xBCC9
  MOVI r8,127
mth_dft255_tw_loop:
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
  JMP[GT] @mth_dft255_tw_loop

  ; W^k = conj(W^(255-k)) for k=128..254, source k=127..1.
  MOVI r1,@mth_dft255_twr
  ADDI r1,r1,127
  MOVI r2,@mth_dft255_twi
  ADDI r2,r2,127
  MOVI r3,@mth_dft255_twr
  ADDI r3,r3,128
  MOVI r4,@mth_dft255_twi
  ADDI r4,r4,128
  MOVI r8,127
mth_dft255_tw_mirror:
  LD r5,r1,0
  ST r5,r3,0
  LD r5,r2,0
  MOVI r6,0
  FSUB r5,r6,r5
  ST r5,r4,0
  ADDI r1,r1,-1
  ADDI r2,r2,-1
  ADDI r3,r3,1
  ADDI r4,r4,1
  ADDI r8,r8,-1
  CMP r8,r0
  JMP[GT] @mth_dft255_tw_mirror
  RET

mth_dft255_puts:
  MOVI r4,255
mth_dft255_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_dft255_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_dft255_puts_byte
  ADD r1,r1,r13
  JMP @mth_dft255_puts_word

mth_dft255_pass:
.str "MTH GRAPH DFT255 PASS\n"
mth_dft255_bad:
.str "MTH GRAPH DFT255 FAIL\n"

; Six distinct, adjacent 255-word float32 arrays.
.org 4096
mth_dft255_twr:
.org 4351
mth_dft255_twi:
.org 4606
mth_dft255_xr:
.org 4861
mth_dft255_xi:
.org 5116
mth_dft255_Xr:
.org 5371
mth_dft255_Xi:
.org 5626
