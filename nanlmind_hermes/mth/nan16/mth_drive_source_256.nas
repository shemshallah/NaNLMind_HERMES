; MTH R4 bounded DRIVE source — NaN16 float32 only.
;
; ABI:
;   r1 = state, 256 float32 payloads
;   r2 = source, 256 float32 payloads
;   r3 = dt, one float32 payload
;   r4 = out, 256 float32 payloads
;   r13 = 1
;   returns r1=1 success
;   clobbers r5-r9, flags
;
; out[i] = state[i] + dt * source[i], increasing i.
; One FMUL followed by one FADD per element; no fused multiply-add, no
; reassociation, no host math. Leaf routine: no CALL, no static scratch.
;
; The source is the bounded forward DRIVE input f(t). state and out may alias
; elementwise; the source span must not overlap out. The energy, rate, and ACL
; budget remain the caller's contract; this kernel performs only the fixed
; 256-element forward update.

mth_drive_source_256:
  LD r5,r3,0
  MOVI r6,0
mth_drive_source_256_loop:
  ADD r7,r1,r6
  LD r8,r7,0
  ADD r7,r2,r6
  LD r9,r7,0
  FMUL r9,r9,r5
  FADD r8,r8,r9
  ADD r7,r4,r6
  ST r8,r7,0
  ADD r6,r6,r13
  MOVI r7,256
  CMP r6,r7
  JMP[LT] @mth_drive_source_256_loop
  MOVI r1,1
  RET
