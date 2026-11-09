; MTH symmetric tridiagonal radial action — NaN16 float32 only.
;
; ABI:
;   r1 = diag, 256 float32
;   r2 = offdiag, 255 float32; offdiag[i] couples x[i] and x[i+1]
;   r3 = x, 256 float32
;   r4 = y, 256 float32 output
;   r13 = 1
;   returns r1=1
;   clobbers r1 (return value) and r5-r12; r2-r4 and r13-r15 are preserved
;
; y[0]   = diag[0]*x[0] + offdiag[0]*x[1]
; y[i]   = offdiag[i-1]*x[i-1] + diag[i]*x[i] + offdiag[i]*x[i+1],
;          1 <= i <= 254
; y[255] = offdiag[254]*x[254] + diag[255]*x[255]
;
; Leaf: no CALL. Increasing i. One accumulator per row; terms accumulate in
; increasing column order (lower off-diagonal, diagonal, upper off-diagonal).
; One rounded FMUL and one rounded FADD per term. No FMA, reassociation,
; hidden precision, host math, or boundary branch.
;
; Boundary convention (descriptor-level, not a second operator): the caller
; encodes a homogeneous inner Dirichlet/Robin row in diag[0] and offdiag[0],
; and the corresponding outer row in diag[255] and offdiag[254]. The kernel
; always evaluates the same symmetric tridiagonal action; it never adds a
; boundary term of its own and never inspects a boundary kind. Inhomogeneous
; boundary data is a caller-side right-hand-side adjustment.
;
; Inputs are read-only; the four spans must be distinct and non-overlapping
; (caller precondition). Working set is 256+255+256+256 = 1,023 words.

mth_radial_tridiagonal:
  ADD r5,r1,r0        ; r5 -> diag[0]
  ADD r6,r2,r0        ; r6 -> offdiag[0]
  ADD r7,r3,r0        ; r7 -> x[0]
  ADD r8,r4,r0        ; r8 -> y[0]

  ; Row 0: diag[0]*x[0] + offdiag[0]*x[1].
  LD r9,r5,0
  LD r10,r7,0
  FMUL r9,r9,r10
  LD r11,r6,0
  LD r12,r7,1
  FMUL r11,r11,r12
  FADD r9,r9,r11
  ST r9,r8,0

  ; Row 1 setup: diag[1], offdiag[0], x[0], y[1].
  ADD r5,r5,r13
  ADD r8,r8,r13

  MOVI r9,254
mth_radial_tridiagonal_inner:
  ; r5=diag[i], r6=offdiag[i-1], r7=x[i-1], r8=y[i].
  LD r10,r6,0         ; offdiag[i-1]
  LD r11,r7,0         ; x[i-1]
  FMUL r10,r10,r11    ; acc = offdiag[i-1]*x[i-1]
  LD r11,r5,0         ; diag[i]
  LD r12,r7,1         ; x[i]
  FMUL r11,r11,r12
  FADD r10,r10,r11
  LD r11,r6,1         ; offdiag[i]
  LD r12,r7,2         ; x[i+1]
  FMUL r11,r11,r12
  FADD r10,r10,r11
  ST r10,r8,0

  ADD r5,r5,r13
  ADD r6,r6,r13
  ADD r7,r7,r13
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_radial_tridiagonal_inner

  ; Row 255: offdiag[254]*x[254] + diag[255]*x[255].
  LD r10,r6,0
  LD r11,r7,0
  FMUL r10,r10,r11
  LD r11,r5,0
  LD r12,r7,1
  FMUL r11,r11,r12
  FADD r10,r10,r11
  ST r10,r8,0

  MOVI r1,1
  RET
