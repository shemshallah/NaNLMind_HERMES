; MTH 128x128 centrosymmetry certificate — NaN16 float32 words only.
;
; A matrix M is centrosymmetric when M[i,j] == M[127-i,127-j] for every
; entry. This kernel scans the matrix in place and returns a mismatch count.
;
; ABI:
;   r1 = M, 16,384 float32 payloads, row-major 128x128
;   r13 = 1
;   returns r1=1 success, r2=mismatch count
;   clobbers r3-r12
;
; Comparison is bitwise payload equality (integer CMP), which is exact for
; finite float32 and also detects NaN payload differences. No host math.
; The matrix is read-only; no output array is allocated.

mth_centrosymmetry_certify_128:
  ADD r4,r1,r0              ; matrix base
  MOVI r5,0                 ; i
  MOVI r2,0                 ; mismatch count
mth_centro_row:
  SHL r6,r5,7               ; i*128
  ADD r6,r4,r6              ; row A base
  MOVI r3,127
  SUB r3,r3,r5
  SHL r3,r3,7
  ADD r7,r4,r3              ; row B base
  MOVI r8,0                 ; j
mth_centro_col:
  ADD r9,r6,r8
  LD r11,r9,0               ; M[i,j]
  MOVI r3,127
  SUB r3,r3,r8
  ADD r10,r7,r3
  LD r12,r10,0              ; M[127-i,127-j]
  CMP r11,r12
  JMP[EQ] @mth_centro_match
  ADD r2,r2,r13
mth_centro_match:
  ADD r8,r8,r13
  MOVI r3,128
  CMP r8,r3
  JMP[LT] @mth_centro_col
  ADD r5,r5,r13
  MOVI r3,128
  CMP r5,r3
  JMP[LT] @mth_centro_row
  MOVI r1,1
  RET
