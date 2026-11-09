; MTH 128-element complex phase-coherence metric — NaN16 float32 only.
;
; For z_i = re[i] + i*im[i]:
;
;   R = |sum_i z_i| / sum_i |z_i|
;     = sqrt(sum_re^2 + sum_im^2) / mag_sum
;
; with sum_re = sum_i re[i], sum_im = sum_i im[i], and
; mag_sum = sum_i sqrt(re[i]^2 + im[i]^2). R is 0 when mag_sum == 0.
;
; ABI:
;   r1 = re, 128 float32
;   r2 = im, 128 float32
;   r3 = output, one float32
;   r13 = 1
;   returns r1=1
;   clobbers r4-r12
;
; Accumulation order is increasing i for all three sums. Only
; FADD/FSUB/FMUL/FSQRT/FDIV are used for float arithmetic; no host math,
; no reordering, no fused multiply-add. The inputs are read-only.

mth_phase_coherence_128:
  ADD r4,r1,r0              ; re base
  ADD r5,r2,r0              ; im base
  ADD r6,r3,r0              ; output
  MOVI r7,0                 ; i
  MOVI r8,0                 ; sum_re = +0.0
  MOVI r9,0                 ; sum_im = +0.0
  MOVI r10,0                ; mag_sum = +0.0
mth_phase_loop:
  ADD r11,r4,r7
  LD r11,r11,0              ; re[i]
  ADD r12,r5,r7
  LD r12,r12,0              ; im[i]
  FADD r8,r8,r11            ; sum_re += re
  FADD r9,r9,r12            ; sum_im += im
  FMUL r11,r11,r11          ; re^2
  FMUL r12,r12,r12          ; im^2
  FADD r11,r11,r12          ; re^2 + im^2
  FSQRT r11,r11             ; |z_i|
  FADD r10,r10,r11          ; mag_sum += |z_i|
  ADD r7,r7,r13
  MOVI r12,128
  CMP r7,r12
  JMP[LT] @mth_phase_loop

  FMUL r8,r8,r8             ; sum_re^2
  FMUL r9,r9,r9             ; sum_im^2
  FADD r8,r8,r9
  FSQRT r8,r8               ; |sum_i z_i|

  FCMP r10,r0
  JMP[EQ] @mth_phase_zero   ; mag_sum == 0 -> R = 0
  FDIV r8,r8,r10
  ST r8,r6,0
  MOVI r1,1
  RET

mth_phase_zero:
  ST r0,r6,0
  MOVI r1,1
  RET
