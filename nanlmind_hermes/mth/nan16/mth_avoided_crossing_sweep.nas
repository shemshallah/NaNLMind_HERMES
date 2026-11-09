; MTH avoided-crossing sweep kernel — NaN16 float32 only.
;
; R7 sweep primitive: for each coupling c[i], form the real symmetric 2x2
; Hamiltonian
;   H = [[a, c[i]], [c[i], b]]
; and emit its lower eigenvalue, upper eigenvalue, and gap.
;
; ABI:
;   r1 = a, pointer to one float32 payload (diagonal A)
;   r2 = b, pointer to one float32 payload (diagonal B)
;   r3 = couplings, n float32 in increasing index order
;   r4 = n, integer count 1..256
;   r5 = gap_out, n float32
;   r6 = lower_out, n float32
;   r7 = upper_out, n float32
;   r13 = 1
;   returns r1=1 success
;   clobbers r8-r12 and r14; r1 becomes the return value
;
; Fixed per-element computation order (same stable formula as
; mth_two_level_spectrum with eta=1; no reordering, no fused multiply-add):
;   mean   = (a + b) * 0.5
;   half   = (a - b) * 0.5
;   rad    = sqrt(half*half + c*c)
;   lower  = mean - rad
;   upper  = mean + rad
;   gap    = upper - lower
;
; mean and half depend only on the diagonal and are computed once. The loop
; then visits i = 0,1,...,n-1 in increasing order. Division by two is
; multiplication by the exactly representable 0.5f (0x3F000000), built with
; MOVI+MOVHI. The gap is the literal E_high - E_low difference, not a
; separate 2*rad shortcut.
;
; Leaf routine: no CALL, no static slots, no host math. The caller supplies
; non-overlapping readable inputs and writable outputs. Finite float32
; inputs and n in 1..256 are assumed; no NaN/Inf, range, or aliasing check
; is performed. Input pointers r2-r7 are preserved.

mth_avoided_crossing_sweep:
  LD r8,r1,0                  ; a
  LD r9,r2,0                  ; b
  MOVI r10,0
  MOVHI r10,0x3F00            ; 0.5f
  FADD r11,r8,r9              ; a + b
  FMUL r11,r11,r10            ; mean = (a+b)/2
  FSUB r12,r8,r9              ; a - b
  FMUL r12,r12,r10            ; half = (a-b)/2
  MOVI r8,0                   ; i = 0

mth_avoided_crossing_sweep_loop:
  ADD r9,r3,r8
  LD r9,r9,0                  ; c = couplings[i]
  FMUL r9,r9,r9               ; c*c
  FMUL r10,r12,r12            ; half*half
  FADD r10,r10,r9             ; half*half + c*c
  FSQRT r10,r10               ; rad
  FSUB r9,r11,r10             ; lower = mean - rad
  FADD r14,r11,r10            ; upper = mean + rad
  FSUB r10,r14,r9             ; gap = upper - lower

  ADD r1,r6,r8
  ST r9,r1,0                  ; lower_out[i]
  ADD r1,r7,r8
  ST r14,r1,0                 ; upper_out[i]
  ADD r1,r5,r8
  ST r10,r1,0                 ; gap_out[i]

  ADD r8,r8,r13
  CMP r8,r4
  JMP[LT] @mth_avoided_crossing_sweep_loop

  MOVI r1,1
  RET
