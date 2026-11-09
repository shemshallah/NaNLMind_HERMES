; MTH dense 256x256 matrix-vector product — NaN16 float32 only.
;
; ABI:
;   r1 = A, 65,536 float32 payloads, row-major 256x256
;   r2 = x, 256 float32
;   r3 = y, 256 float32 output
;   r13 = 1
;   returns r1=1
;   clobbers r4-r12, r14
;
; y[i] = sum_{j=0}^{255} A[i,j] * x[j], increasing j, one accumulator.
; Leaf: no CALL. No host math, no FMA, no reordering. Inputs are read-only;
; spans must be distinct and non-overlapping (caller precondition).
;
; Working set: 65,536 + 256 + 256 = 66,048 NaN16 words, leaving 65,024 words
; of the 1 MiB (131,072-word) guest budget for code and scratch. This image
; does not fit the historical 512 KiB (65,536-word) guest budget.

mth_matvec256_full:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_matvec256_full_row:
  MOVI r9,0
  ADD r10,r4,r0
  ADD r11,r5,r0
  MOVI r8,0
mth_matvec256_full_col:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,256
  CMP r8,r12
  JMP[LT] @mth_matvec256_full_col
  ST r9,r6,0
  MOVI r12,256
  ADD r4,r4,r12
  ADD r6,r6,r13
  ADD r7,r7,r13
  MOVI r12,256
  CMP r7,r12
  JMP[LT] @mth_matvec256_full_row
  MOVI r1,1
  RET
