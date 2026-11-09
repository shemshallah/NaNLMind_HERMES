; MTH diagonal harmonic acceleration — NaN16 float32 only.
; a[i] = -k * u[i], for the 256-element integrator/energy fixture.
;
; ABI: r1=u 256 float32, r2=k one float32, r3=a 256 float32, r13=1.
; Returns r1=1. Leaf routine; no CALLs, no static slots.

mth_harmonic_accel_256_f32:
  LD r4,r2,0
  MOVI r5,0
mth_harmonic_accel_loop:
  ADD r6,r1,r5
  LD r6,r6,0
  FMUL r6,r6,r4
  FSUB r6,r0,r6
  ADD r7,r3,r5
  ST r6,r7,0
  ADD r5,r5,r13
  MOVI r6,256
  CMP r5,r6
  JMP[LT] @mth_harmonic_accel_loop
  MOVI r1,1
  RET
