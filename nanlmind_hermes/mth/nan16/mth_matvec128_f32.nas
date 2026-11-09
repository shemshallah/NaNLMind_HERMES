; MTH dense 128x128 matrix-vector product — NaN16 float32 only.
;
; ABI:
;   r1 = A, 16,384 float32 payloads, row-major 128x128
;   r2 = x, 128 float32
;   r3 = y, 128 float32 output
;   r13 = 1
;   returns r1=1
;   clobbers r4-r12, r14
;
; y[i] = sum_{j=0}^{127} A[i,j] * x[j], increasing j, one accumulator.
; No host math, no FMA, no reordering. Inputs are read-only; spans must be
; distinct and non-overlapping (caller precondition).

mth_matvec128_f32:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_matvec128_row:
  MOVI r9,0
  ADD r10,r4,r0
  ADD r11,r5,r0
  MOVI r8,0
mth_matvec128_col:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,128
  CMP r8,r12
  JMP[LT] @mth_matvec128_col
  ST r9,r6,0
  MOVI r12,128
  ADD r4,r4,r12
  ADD r6,r6,r13
  ADD r7,r7,r13
  MOVI r12,128
  CMP r7,r12
  JMP[LT] @mth_matvec128_row
  MOVI r1,1
  RET
