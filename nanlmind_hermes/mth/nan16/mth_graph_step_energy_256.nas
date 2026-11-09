; MTH step energy metric — NaN16 float32 only.
;
; Given u, v, and a = -H u, compute:
;   kinetic    = 0.5 * sum v[i]^2
;   potential  = 0.5 * sum u[i] * (-a[i])
;   total      = kinetic + potential
;
; ABI:
;   r1 = u, 256 float32
;   r2 = v, 256 float32
;   r3 = a, 256 float32 (already -H u)
;   r4 = output, three float32: kinetic, potential, total
;   r13 = 1
;   returns r1=1
;   clobbers r5-r12
;
; This is the discrete quadratic energy analogue for the wave step. It does not
; prove a full-run conservation law; it makes the per-step metric inspectable.
; No host math; increasing-index accumulation.

mth_graph_step_energy_256_f32:
  MOVI r5,0
  MOVI r6,0
  MOVI r7,0
mth_step_energy_loop:
  ADD r8,r2,r7
  LD r8,r8,0
  FMUL r8,r8,r8
  FADD r5,r5,r8

  ADD r8,r1,r7
  LD r8,r8,0
  ADD r9,r3,r7
  LD r9,r9,0
  FMUL r8,r8,r9
  FSUB r8,r0,r8
  FADD r6,r6,r8

  ADD r7,r7,r13
  MOVI r8,256
  CMP r7,r8
  JMP[LT] @mth_step_energy_loop

  MOVI r8,0
  MOVHI r8,0x3F00
  FMUL r5,r5,r8
  FMUL r6,r6,r8
  FADD r7,r5,r6
  ST r5,r4,0
  ST r6,r4,1
  ST r7,r4,2
  MOVI r1,1
  RET
