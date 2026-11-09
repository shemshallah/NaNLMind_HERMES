; MTH graph wave velocity-Verlet step -- native NaN16 float32 only.
;
; One step of the damped velocity-Verlet integrator for
;   u'' + 2*gamma_ode*u' + H u = 0,
; given the already-computed acceleration a = -H u for the old state u.
; The ABI's gamma is the discrete velocity-scaling coefficient:
;   gamma_ABI = 2*gamma_ode.
; Per element i = 0..255:
;   v_half = v + 0.5*dt*a
;   u_next = u + dt*v_half
;   v_next = (v_half + 0.5*dt*a) * (1 - gamma_ABI*dt)
;
; ABI (r13 = 1):
;   r1 = u       pointer to 256 float32 words
;   r2 = v       pointer to 256 float32 words
;   r3 = a       pointer to 256 float32 words, already = -H u
;   r4 = u_next  pointer to 256 float32 words (output)
;   r5 = v_next  pointer to 256 float32 words (output)
;   r6 = dt      float32, must be finite and > 0
;   r7 = gamma   float32 damping, must be finite and >= 0
;   returns r1=1 on success
;           r1=0 for an invalid dt/gamma/damping factor
;           r1=2 for an out-of-range or overlapping buffer layout
;
; Preconditions enforced before the first output write:
;   * every one of the five 256-word spans lies fully inside the NaN16
;     address space without wrapping (base <= 0x3FFF00);
;   * all five spans are pairwise disjoint;
;   * dt has a finite exponent, gamma has a finite exponent;
;   * dt > 0 and gamma >= 0;
;   * the per-step damping factor d = 1 - gamma*dt is >= 0.
;
; Damping policy: a damping factor < 0 is REJECTED, not clamped. The
; kernel returns r1=0 and leaves both output spans unchanged. A negative
; gamma is likewise rejected. gamma = -0.0 compares equal to zero and is
; accepted because its damping factor is exactly 1.0.
;
; Numeric requirements:
;   * per element: three FADDs and three FMULs; the only FSUB is the
;     preflight damping subtraction. All are IEEE binary32 ops executed
;     by the NaN16 float unit; no host math;
;   * half_dt = 0.5*dt is computed once; t = half_dt*a is computed once
;     and reused for both half-kicks (bit-identical to computing
;     0.5*dt*a twice);
;   * u_next is written before v_next, which is safe because all spans
;     are proven disjoint in preflight;
;   * a[] is trusted input. Non-finite or NaN a[] values propagate through
;     IEEE float arithmetic; this kernel does not validate a[].
;
; Energy semantics:
;   This kernel does not compute or assert an energy value. For gamma=0
;   the velocity-Verlet map is symplectic for a fixed symmetric H, so the
;   supplied a=-H u participates in a second-order shadow-energy method;
;   float32 roundoff can still drift the measured energy. For gamma>0 the
;   velocity is multiplied by d in [0,1] at the end of each step, so the
;   kinetic term cannot increase during that multiplication, but total
;   discrete wave energy is a property of a full run and is not proven by
;   this single-step fixture. The fixture checks exact kinematics and
;   rejection atomicity only.
;
; No static scratch, no shared return slot, no helper calls.

mth_graph_wave_step_f32:
  ADD r8,r1,r0
  ADD r9,r2,r0
  ADD r10,r3,r0
  ADD r11,r4,r0
  ADD r12,r5,r0

  ; Reject spans that cannot fit 256 words without wrapping at the top.
  MOVI r1,0xFF00
  MOVHI r1,0x3F              ; r1 = 0x003FFF00 = 0x3FFFFF - 255
  CMP r8,r1
  JMP[GT] @mth_wave_bad_layout
  CMP r9,r1
  JMP[GT] @mth_wave_bad_layout
  CMP r10,r1
  JMP[GT] @mth_wave_bad_layout
  CMP r11,r1
  JMP[GT] @mth_wave_bad_layout
  CMP r12,r1
  JMP[GT] @mth_wave_bad_layout

  ; Pairwise disjointness. Two inclusive spans [A,A+255], [B,B+255]
  ; overlap iff A <= B+255 and B <= A+255.
  MOVI r2,255

  ADD r3,r9,r2               ; v+255
  CMP r8,r3
  JMP[GT] @mth_wave_ov_uv_ok
  ADD r3,r8,r2               ; u+255
  CMP r9,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_uv_ok:

  ADD r3,r10,r2              ; a+255
  CMP r8,r3
  JMP[GT] @mth_wave_ov_ua_ok
  ADD r3,r8,r2               ; u+255
  CMP r10,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_ua_ok:

  ADD r3,r11,r2              ; u_next+255
  CMP r8,r3
  JMP[GT] @mth_wave_ov_uun_ok
  ADD r3,r8,r2               ; u+255
  CMP r11,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_uun_ok:

  ADD r3,r12,r2              ; v_next+255
  CMP r8,r3
  JMP[GT] @mth_wave_ov_uvn_ok
  ADD r3,r8,r2               ; u+255
  CMP r12,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_uvn_ok:

  ADD r3,r10,r2              ; a+255
  CMP r9,r3
  JMP[GT] @mth_wave_ov_va_ok
  ADD r3,r9,r2               ; v+255
  CMP r10,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_va_ok:

  ADD r3,r11,r2              ; u_next+255
  CMP r9,r3
  JMP[GT] @mth_wave_ov_vun_ok
  ADD r3,r9,r2               ; v+255
  CMP r11,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_vun_ok:

  ADD r3,r12,r2              ; v_next+255
  CMP r9,r3
  JMP[GT] @mth_wave_ov_vvn_ok
  ADD r3,r9,r2               ; v+255
  CMP r12,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_vvn_ok:

  ADD r3,r11,r2              ; u_next+255
  CMP r10,r3
  JMP[GT] @mth_wave_ov_aun_ok
  ADD r3,r10,r2              ; a+255
  CMP r11,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_aun_ok:

  ADD r3,r12,r2              ; v_next+255
  CMP r10,r3
  JMP[GT] @mth_wave_ov_avn_ok
  ADD r3,r10,r2              ; a+255
  CMP r12,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_avn_ok:

  ADD r3,r12,r2              ; v_next+255
  CMP r11,r3
  JMP[GT] @mth_wave_ov_unvn_ok
  ADD r3,r11,r2              ; u_next+255
  CMP r12,r3
  JMP[LE] @mth_wave_bad_layout
mth_wave_ov_unvn_ok:

  ; dt and gamma must have finite exponent fields. Shifting left once
  ; discards the sign bit; shifting right 24 leaves exactly the 8 exponent
  ; bits. 0xFF means infinity or NaN.
  ADD r1,r6,r0
  SHL r1,r1,1
  SHR r1,r1,24
  MOVI r2,255
  CMP r1,r2
  JMP[EQ] @mth_wave_bad_param

  ADD r1,r7,r0
  SHL r1,r1,1
  SHR r1,r1,24
  CMP r1,r2
  JMP[EQ] @mth_wave_bad_param

  ; dt > 0. FCMP with a NaN would set only flag bit 3, but the exponent
  ; test above has already excluded NaN. A -0.0 dt compares equal here.
  MOVI r2,0
  FCMP r6,r2
  JMP[LE] @mth_wave_bad_param

  ; gamma >= 0. -0.0 is accepted; NaN was excluded above.
  FCMP r7,r2
  JMP[LT] @mth_wave_bad_param

  ; Damping factor d = 1 - gamma*dt. gamma>=0 and dt>0 imply the product
  ; is >=0, so only an overflow to +inf can make d negative. Reject d<0.
  MOVI r1,0
  MOVHI r1,0x3F80            ; 1.0
  FMUL r2,r7,r6              ; gamma*dt
  FSUB r1,r1,r2              ; d = 1 - gamma*dt
  MOVI r2,0
  FCMP r1,r2
  JMP[LT] @mth_wave_bad_param
  ADD r7,r1,r0               ; r7 = d

  ; half_dt = 0.5*dt, computed once and reused.
  MOVI r1,0
  MOVHI r1,0x3F00            ; 0.5
  FMUL r3,r1,r6              ; r3 = 0.5*dt

  MOVI r1,256
mth_wave_apply:
  LD r14,r8,0                ; u_i
  LD r2,r10,0                ; a_i
  FMUL r2,r3,r2              ; t = 0.5*dt*a_i
  LD r4,r9,0                 ; v_i
  FADD r4,r4,r2              ; v_half
  FMUL r5,r6,r4              ; dt*v_half
  FADD r5,r14,r5             ; u_next
  ST r5,r11,0
  FADD r4,r4,r2              ; v_half + 0.5*dt*a_i
  FMUL r4,r4,r7              ; v_next = ... * d
  ST r4,r12,0
  ADD r8,r8,r13
  ADD r9,r9,r13
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r12,r12,r13
  SUB r1,r1,r13
  CMP r1,r0
  JMP[GT] @mth_wave_apply

mth_wave_success:
  MOVI r1,1
  RET

mth_wave_bad_param:
  MOVI r1,0
  RET

mth_wave_bad_layout:
  MOVI r1,2
  RET
