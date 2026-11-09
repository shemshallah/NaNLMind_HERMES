; MTH two-level avoided-crossing spectrum kernel — NaN16 float32 only.
;
; H(eta) = [[a, eta*c], [eta*c, b]], real symmetric 2x2.
;
; ABI:
;   r1 = a, pointer to one float32 payload
;   r2 = b, pointer to one float32 payload
;   r3 = c, pointer to one float32 payload
;   r4 = eta, pointer to one float32 payload
;   r5 = E_low output, pointer to one float32 payload
;   r6 = E_high output, pointer to one float32 payload
;   r7 = gap output, pointer to one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r8-r12 and r14
;
; Computation order (fixed; no reordering, no fused multiply-add):
;   mean   = (a + b) * 0.5
;   half   = (a - b) * 0.5
;   ec     = eta * c
;   rad    = sqrt(half*half + ec*ec)
;   E_low  = mean - rad
;   E_high = mean + rad
;   gap    = E_high - E_low
;
; Division by two is multiply by the exactly representable 0.5f
; (0x3F000000), built with MOVI+MOVHI. The radicand is nonnegative for
; finite inputs. The last gap operation is the literal definition
; E_high - E_low, not a separate 2*rad shortcut.
;
; Leaf routine: no CALL, no static slots, no host math. The caller supplies
; non-overlapping readable inputs and writable outputs. Finite float32
; inputs are assumed; no NaN/Inf validation or range check is performed.
; The eta factor scales only the off-diagonal coupling, so eta=0 yields the
; decoupled diagonal pair (a,b) exactly (up to the 0.5 scaling).

mth_two_level_spectrum:
  LD r8,r1,0                  ; a
  LD r9,r2,0                  ; b
  LD r10,r3,0                 ; c
  LD r11,r4,0                 ; eta
  MOVI r12,0
  MOVHI r12,0x3F00            ; 0.5f
  FADD r14,r8,r9              ; a + b
  FMUL r14,r14,r12            ; mean = (a+b)/2
  FSUB r8,r8,r9               ; a - b
  FMUL r8,r8,r12              ; half = (a-b)/2
  FMUL r11,r11,r10            ; ec = eta*c
  FMUL r8,r8,r8               ; half*half
  FMUL r11,r11,r11            ; ec*ec
  FADD r8,r8,r11              ; half*half + ec*ec
  FSQRT r8,r8                 ; rad
  FSUB r9,r14,r8              ; E_low = mean - rad
  FADD r14,r14,r8             ; E_high = mean + rad
  FSUB r8,r14,r9              ; gap = E_high - E_low
  ST r9,r5,0
  ST r14,r6,0
  ST r8,r7,0
  MOVI r1,1
  RET
