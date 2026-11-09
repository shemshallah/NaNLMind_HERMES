; MTH radial-block quadratic energy and localization — NaN16 float32 only.
;
; ABI:
;   r1 = diag, 256 float32 words (input)
;   r2 = offdiag, 255 float32 words (input); offdiag[i] couples x[i] and x[i+1]
;   r3 = x, 256 float32 words (input)
;   r4 = energy_out, one float32 word (output)
;   r5 = localization_out, one float32 word (output)
;   r13 = 1
;   returns r1=1
;   clobbers r1 (return value), r6-r12, r14
;   preserves r0, r2-r5, r13, r15
;
; energy       = sum_{i=0}^{255} diag[i]*x[i]^2
;              + 2 * sum_{i=0}^{254} offdiag[i]*x[i]*x[i+1]
; localization = max_{0<=i<=255} x[i]^2 / sum_{i=0}^{255} x[i]^2
;              = +0.0 when the denominator is zero.
;
; This is the scalar functional companion of mth_radial_tridiagonal. It reads
; the same diag/offdiag/x layout and inherits the same boundary convention
; (the caller has already folded inner/outer boundary coefficients into
; diag[0], offdiag[0], diag[255], and offdiag[254]). It does not form y = T x,
; does not need a y span, and contains no CALL.
;
; Numeric contract:
;   - Pass 1 visits i = 0..255 in increasing order. It maintains one energy
;     accumulator, one sum-of-squares accumulator, and one running maximum.
;     Each i contributes FMUL(x[i],x[i]), FADD(sumsq,x[i]^2),
;     FMUL(diag[i],x[i]^2), and FADD(energy,term). The maximum is updated by
;     an ordered FCMP/GT branch.
;   - Pass 2 visits i = 0..254 in increasing order and continues the same
;     energy accumulator. Each i contributes FMUL(offdiag[i],x[i]),
;     FMUL(...,x[i+1]), FADD(term,term) (exact doubling), and
;     FADD(energy,doubled). Thus the diagonal sum is accumulated first and
;     the coupling sum second.
;   - Every arithmetic operation is one rounded float32 FMUL/FADD/FDIV.
;     There is no FMA, reassociation, hidden precision, host math, or
;     tabulated approximation. Results are deterministic for a given input
;     bit pattern but are not bitwise identical to an interleaved or FMA
;     implementation.
;   - FDIV is used once for localization. A zero denominator stores +0.0
;     instead of dividing. NaN/Inf input policy is caller-side: the kernel
;     propagates what the float unit produces and does not validate inputs.
;
; Inputs are read-only; the five spans must be distinct and non-overlapping
; (caller precondition). Working set is 256+255+256+1+1 = 769 words.

mth_radial_energy:
  ADD r6,r1,r0        ; r6 -> diag[0]
  ADD r7,r3,r0        ; r7 -> x[0]
  MOVI r8,0           ; energy accumulator
  MOVI r9,0           ; sum_i x[i]^2
  MOVI r10,0          ; max_i x[i]^2
  MOVI r11,256        ; remaining diagonal terms

mth_radial_energy_diag_loop:
  LD r12,r7,0         ; x[i]
  FMUL r12,r12,r12    ; x[i]^2
  FADD r9,r9,r12      ; sumsq += x[i]^2
  FCMP r12,r10        ; x[i]^2 > max?
  JMP[GT] @mth_radial_energy_new_max
mth_radial_energy_diag_after_max:
  LD r14,r6,0         ; diag[i]
  FMUL r14,r14,r12    ; diag[i]*x[i]^2
  FADD r8,r8,r14      ; energy += diag[i]*x[i]^2
  ADD r6,r6,r13
  ADD r7,r7,r13
  SUB r11,r11,r13
  CMP r11,r0
  JMP[GT] @mth_radial_energy_diag_loop

  ; Pass 2: coupling terms continue the same energy accumulator.
  ADD r6,r2,r0        ; r6 -> offdiag[0]
  ADD r7,r3,r0        ; r7 -> x[0]
  MOVI r11,255        ; 255 coupling terms

mth_radial_energy_off_loop:
  LD r12,r6,0         ; offdiag[i]
  LD r14,r7,0         ; x[i]
  FMUL r12,r12,r14    ; offdiag[i]*x[i]
  LD r14,r7,1         ; x[i+1]
  FMUL r12,r12,r14    ; offdiag[i]*x[i]*x[i+1]
  FADD r12,r12,r12    ; exact doubling: 2*term
  FADD r8,r8,r12      ; energy += 2*offdiag[i]*x[i]*x[i+1]
  ADD r6,r6,r13
  ADD r7,r7,r13
  SUB r11,r11,r13
  CMP r11,r0
  JMP[GT] @mth_radial_energy_off_loop

  ST r8,r4,0          ; energy_out
  FCMP r9,r0          ; zero denominator?
  JMP[EQ] @mth_radial_energy_zero_loc
  FDIV r10,r10,r9     ; localization = max(x^2)/sum(x^2)
  ST r10,r5,0
  MOVI r1,1
  RET

mth_radial_energy_new_max:
  ADD r10,r12,r0
  JMP @mth_radial_energy_diag_after_max

mth_radial_energy_zero_loc:
  ST r0,r5,0          ; +0.0 for an all-zero (or underflowed) x
  MOVI r1,1
  RET
