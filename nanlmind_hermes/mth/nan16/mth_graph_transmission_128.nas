; MTH two-region energy and transmission metric — NaN16 float32 only.
;
; ABI:
;   r1 = inner, 128 float32
;   r2 = outer, 128 float32
;   r3 = output, three float32: inner_energy, outer_energy, transmission
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r12
;
; Computes:
;   E_in  = sum inner[i]^2
;   E_out = sum outer[i]^2
;   T     = E_out / (E_in + E_out), or 0 if the denominator is zero.
; No host math; increasing-index accumulation; one accumulator per region.
; This is an energy-ratio analogue, not a physical tunneling probability.

mth_graph_transmission_128_f32:
  MOVI r4,0
  MOVI r5,0
  MOVI r6,0
mth_transmission_loop:
  ADD r7,r1,r6
  LD r7,r7,0
  FMUL r7,r7,r7
  FADD r4,r4,r7
  ADD r7,r2,r6
  LD r7,r7,0
  FMUL r7,r7,r7
  FADD r5,r5,r7
  ADD r6,r6,r13
  MOVI r7,128
  CMP r6,r7
  JMP[LT] @mth_transmission_loop

  FADD r8,r4,r5
  MOVI r9,0
  FCMP r8,r9
  JMP[EQ] @mth_transmission_zero
  FDIV r8,r5,r8
  JMP @mth_transmission_store
mth_transmission_zero:
  MOVI r8,0
mth_transmission_store:
  ST r4,r3,0
  ST r5,r3,1
  ST r8,r3,2
  MOVI r1,1
  RET
