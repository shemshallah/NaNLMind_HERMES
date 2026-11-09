.org 0
JMP @mth_radial_tridiagonal_test_main
.include "../nan16/mth_radial_tridiagonal.nas"

mth_radial_tridiagonal_test_main:
  MOVI r13,1

  ; ---- Case 1: diag=2, offdiag=-1, x=1 -> y=[1,0,...,0,1]. ----
  LA r1,@mth_radial_tridiagonal_diag
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x4000
  CALL @mth_radial_tridiagonal_fill

  LA r1,@mth_radial_tridiagonal_offdiag
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0xBF80
  CALL @mth_radial_tridiagonal_fill

  LA r1,@mth_radial_tridiagonal_x
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_radial_tridiagonal_fill

  LA r1,@mth_radial_tridiagonal_diag
  LA r2,@mth_radial_tridiagonal_offdiag
  LA r3,@mth_radial_tridiagonal_x
  LA r4,@mth_radial_tridiagonal_y
  CALL @mth_radial_tridiagonal
  CMP r1,r13
  JMP[NE] @mth_radial_tridiagonal_fail

  LA r5,@mth_radial_tridiagonal_y
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x3F80
  CMP r6,r7
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  MOVI r6,254
mth_radial_tridiagonal_check_one_inner:
  LD r7,r5,0
  CMP r7,r0
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_radial_tridiagonal_check_one_inner
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x3F80
  CMP r6,r7
  JMP[NE] @mth_radial_tridiagonal_fail

  ; ---- Case 2: ramp x[i]=i+1 -> y=[0,0,...,0,257]. ----
  LA r5,@mth_radial_tridiagonal_x
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x3F80
mth_radial_tridiagonal_fill_ramp:
  CVTIF r8,r6
  FADD r8,r8,r7
  ST r8,r5,0
  ADD r6,r6,r13
  ADD r5,r5,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_radial_tridiagonal_fill_ramp

  LA r1,@mth_radial_tridiagonal_diag
  LA r2,@mth_radial_tridiagonal_offdiag
  LA r3,@mth_radial_tridiagonal_x
  LA r4,@mth_radial_tridiagonal_y
  CALL @mth_radial_tridiagonal
  CMP r1,r13
  JMP[NE] @mth_radial_tridiagonal_fail

  LA r5,@mth_radial_tridiagonal_y
  LD r6,r5,0
  CMP r6,r0
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  MOVI r6,254
mth_radial_tridiagonal_check_ramp_inner:
  LD r7,r5,0
  CMP r7,r0
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_radial_tridiagonal_check_ramp_inner
  LD r6,r5,0
  MOVI r7,0x8000
  MOVHI r7,0x4380
  CMP r6,r7
  JMP[NE] @mth_radial_tridiagonal_fail

  ; ---- Case 3: alternating x=+1,-1,... -> interior y[i]=4*x[i]. ----
  LA r5,@mth_radial_tridiagonal_x
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
  MOVI r8,0
  MOVHI r8,0xBF80
mth_radial_tridiagonal_fill_alt:
  ST r7,r5,0
  ADD r9,r7,r0
  ADD r7,r8,r0
  ADD r8,r9,r0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_radial_tridiagonal_fill_alt

  LA r1,@mth_radial_tridiagonal_diag
  LA r2,@mth_radial_tridiagonal_offdiag
  LA r3,@mth_radial_tridiagonal_x
  LA r4,@mth_radial_tridiagonal_y
  CALL @mth_radial_tridiagonal
  CMP r1,r13
  JMP[NE] @mth_radial_tridiagonal_fail

  LA r5,@mth_radial_tridiagonal_y
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x4040
  CMP r6,r7
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  MOVI r6,254
  MOVI r8,0
  MOVHI r8,0xC080
  MOVI r9,0
  MOVHI r9,0x4080
mth_radial_tridiagonal_check_alt_inner:
  LD r7,r5,0
  CMP r7,r8
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r10,r8,r0
  ADD r8,r9,r0
  ADD r9,r10,r0
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_radial_tridiagonal_check_alt_inner
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0xC040
  CMP r6,r7
  JMP[NE] @mth_radial_tridiagonal_fail

  ; ---- Case 4: offdiag=0, diag[i]=i+2, x[i]=i+1 -> y[i]=(i+2)(i+1). ----
  LA r1,@mth_radial_tridiagonal_offdiag
  MOVI r2,255
  MOVI r3,0
  CALL @mth_radial_tridiagonal_fill

  LA r5,@mth_radial_tridiagonal_diag
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x4000
mth_radial_tridiagonal_fill_diag:
  CVTIF r8,r6
  FADD r8,r8,r7
  ST r8,r5,0
  ADD r6,r6,r13
  ADD r5,r5,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_radial_tridiagonal_fill_diag

  LA r5,@mth_radial_tridiagonal_x
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x3F80
mth_radial_tridiagonal_refill_ramp:
  CVTIF r8,r6
  FADD r8,r8,r7
  ST r8,r5,0
  ADD r6,r6,r13
  ADD r5,r5,r13
  MOVI r8,256
  CMP r6,r8
  JMP[LT] @mth_radial_tridiagonal_refill_ramp

  LA r1,@mth_radial_tridiagonal_diag
  LA r2,@mth_radial_tridiagonal_offdiag
  LA r3,@mth_radial_tridiagonal_x
  LA r4,@mth_radial_tridiagonal_y
  CALL @mth_radial_tridiagonal
  CMP r1,r13
  JMP[NE] @mth_radial_tridiagonal_fail

  LA r5,@mth_radial_tridiagonal_y
  MOVI r6,0
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
  MOVHI r8,0x3F80
mth_radial_tridiagonal_check_diag:
  CVTIF r9,r6
  FADD r9,r9,r7
  CVTIF r10,r6
  FADD r10,r10,r8
  FMUL r9,r9,r10
  LD r10,r5,0
  CMP r10,r9
  JMP[NE] @mth_radial_tridiagonal_fail
  ADD r5,r5,r13
  ADD r6,r6,r13
  MOVI r10,256
  CMP r6,r10
  JMP[LT] @mth_radial_tridiagonal_check_diag

  LA r1,@mth_radial_tridiagonal_pass
  CALL @mth_radial_tridiagonal_puts
  HALT

mth_radial_tridiagonal_fail:
  LA r1,@mth_radial_tridiagonal_bad
  CALL @mth_radial_tridiagonal_puts
  HALT

; Fill r2 words at r1 with the float32 payload in r3.
mth_radial_tridiagonal_fill:
  CMP r2,r0
  RET[EQ]
mth_radial_tridiagonal_fill_loop:
  ST r3,r1,0
  ADD r1,r1,r13
  SUB r2,r2,r13
  CMP r2,r0
  JMP[GT] @mth_radial_tridiagonal_fill_loop
  RET

mth_radial_tridiagonal_puts:
  MOVI r4,255
mth_radial_tridiagonal_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_radial_tridiagonal_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_radial_tridiagonal_puts_byte
  ADD r1,r1,r13
  JMP @mth_radial_tridiagonal_puts_word

.org 512
mth_radial_tridiagonal_diag:
.word 0
.org 768
mth_radial_tridiagonal_offdiag:
.word 0
.org 1024
mth_radial_tridiagonal_x:
.word 0
.org 1280
mth_radial_tridiagonal_y:
.word 0
.org 1536
mth_radial_tridiagonal_pass:
.str "MTH RADIAL TRIDIAGONAL PASS\n"
mth_radial_tridiagonal_bad:
.str "MTH RADIAL TRIDIAGONAL FAIL\n"
