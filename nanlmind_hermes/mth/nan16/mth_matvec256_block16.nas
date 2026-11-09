; MTH paged 16x256 block of a 256x256 float32 matrix-vector product — NaN16 only.
;
; ABI:
;   r1 = A block, 4,096 float32 payloads, row-major 16 rows x 256 columns
;   r2 = x, 256 float32 payloads
;   r3 = y block, 16 float32 output payloads
;   r13 = 1
;   returns r1=1
;   clobbers r4-r12, r14
;
; y[k] = sum_{j=0}^{255} A[k,j] * x[j], increasing j, one accumulator.
; Caller drives a full 256x256 matvec as 16 calls: for block b=0..15 pass
; A + b*16*256 and y + b*16. No host math, no FMA, no reordering.
; Inputs are read-only; spans must be distinct and non-overlapping (caller
; precondition).

mth_matvec256_block16:
  ADD r4,r1,r0
  ADD r5,r2,r0
  ADD r6,r3,r0
  MOVI r7,0
mth_matvec256_block16_row:
  MOVI r9,0
  ADD r10,r4,r0
  ADD r11,r5,r0
  MOVI r8,0
mth_matvec256_block16_col:
  LD r12,r10,0
  LD r14,r11,0
  FMUL r12,r12,r14
  FADD r9,r9,r12
  ADD r10,r10,r13
  ADD r11,r11,r13
  ADD r8,r8,r13
  MOVI r12,256
  CMP r8,r12
  JMP[LT] @mth_matvec256_block16_col
  ST r9,r6,0
  MOVI r12,256
  ADD r4,r4,r12
  ADD r6,r6,r13
  ADD r7,r7,r13
  MOVI r12,16
  CMP r7,r12
  JMP[LT] @mth_matvec256_block16_row
  MOVI r1,1
  RET
