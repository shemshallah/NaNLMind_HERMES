; MTH 16x256 block certificate for a 256x256 centrosymmetric matrix — NaN16 only.
;
; A 256x256 float32 matrix is 65,536 NaN16 words, which is the declared guest
; working set before code and scratch. This kernel certifies one 16-row strip
; at a time. The caller walks base = 0,16,...,240 in order.
;
; ABI:
;   r1  = A block, 4,096 float32 payloads, row-major 16 rows x 256 columns
;   r2  = mirror block, 4,096 float32 payloads, row-major 16 rows x 256 columns
;   r13 = 1
;   returns r1 = 1
;   returns r2 = mismatch count
;   clobbers r1-r12; preserves r13
;
; For block row k (global row i = base+k), the caller supplies the global
; row 255-i as mirror row k, in natural column order:
;   mirror[k,c] = M[255-i,c].
; The kernel compares, for k = 0..15 and j = 0..255,
;   A[k,j] == mirror[k,255-j] == M[255-i,255-j]
; with integer CMP on the 32-bit payload. This is bitwise equality, not host
; float equality: +0.0 and -0.0 differ, and differing NaN payloads differ.
;
; One call reports one mismatch for each failed comparison. A full 256x256
; sweep visits both orientations of every off-diagonal pair, so across the
; 16 calls one asymmetric matrix entry contributes two mismatches. The caller
; owns the accumulator and the block pointers; this kernel allocates nothing.
;
; Inputs are read-only. The two 4,096-word spans must be readable and may
; alias or overlap only if the caller accepts that the comparison is then
; not a certificate (no aliasing check is performed).

mth_centrosymmetry_block16:
  ADD r4,r1,r0              ; A block base
  ADD r5,r2,r0              ; mirror block base
  MOVI r2,0                 ; mismatch count
  ADD r6,r4,r0              ; A row pointer
  ADD r7,r5,r0              ; mirror row pointer
  MOVI r8,16                ; rows remaining

mth_centro_block16_row:
  ADD r9,r6,r0              ; A column pointer, starts at j=0
  MOVI r3,255
  ADD r10,r7,r3             ; mirror column pointer, starts at j=255
  MOVI r3,256               ; columns remaining

mth_centro_block16_col:
  LD r11,r9,0               ; A[k,j]
  LD r12,r10,0              ; mirror[k,255-j]
  CMP r11,r12
  JMP[EQ] @mth_centro_block16_equal
  ADD r2,r2,r13             ; payload mismatch
mth_centro_block16_equal:
  ADD r9,r9,r13             ; next A column
  SUB r10,r10,r13           ; previous mirror column
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_centro_block16_col

  MOVI r3,256
  ADD r6,r6,r3              ; next A row
  ADD r7,r7,r3              ; next mirror row
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_centro_block16_row

  MOVI r1,1
  RET
