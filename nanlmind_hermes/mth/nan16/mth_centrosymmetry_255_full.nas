; MTH full 255x255 centrosymmetry certificate — NaN16 float32 only.
;
; A 255x255 float32 matrix A is centrosymmetric when
;
;   A[i,j] == A[254-i,254-j]   for every i,j in 0..254.
;
; The odd 255-head reversal has one fixed index, 127: row 127 maps to itself,
; column 127 maps to itself, and the centre cell (127,127) is the only
; one-cell orbit. The kernel has no special case for the fixed lines; the
; generic comparison covers them, including the self-comparison at the centre.
;
; ABI:
;   r1  = A, 65,025 float32 payloads, row-major 255 rows x 255 columns
;   r13 = 1
;   returns r1 = 1 iff every A[i,j] is bitwise equal to A[254-i,254-j]
;   returns r1 = 0 on the first mismatch
;   clobbers r2-r12; preserves r13, r14, r15
;
; Leaf: no CALL, no static slots, no output array. A is read-only; the kernel
; writes no guest memory. Comparison is integer CMP on the raw 32-bit payload,
; which is bitwise equality: +0.0 and -0.0 differ, and differing NaN payloads
; differ. No host math, tolerance, rounding, or canonicalization is used.
;
; Scan order is increasing row-major over the A side: i = 0,1,...,254 and,
; within each row, j = 0,1,...,254. The mirror pointer starts at column 254
; and decrements, so comparison k of the scan is
;
;   A[i,j] vs A[254-i,254-j]
;
; and the scan stops at the first failing comparison. On a pass all 65,025
; directed comparisons are made; every two-cell orbit is therefore compared
; twice and the centre cell once.
;
; Working set: 65,025 NaN16 words (520,200 bytes) for A alone. The historical
; 512 KiB (65,536-word) guest cannot hold this matrix together with code and
; fixture data; this kernel targets the 1 MiB (131,072-word) guest budget.

mth_centrosymmetry_255_full:
  ADD r4,r1,r0              ; A row i pointer, starts at row 0
  MOVI r5,64770             ; 254*255 = 64,770
  ADD r5,r1,r5              ; A row 254 pointer
  MOVI r10,255              ; rows remaining
mth_centrosymmetry_255_full_row:
  ADD r6,r4,r0              ; A[i,0] pointer
  MOVI r3,254
  ADD r7,r5,r3              ; A[254-i,254] pointer
  MOVI r11,255              ; columns remaining
mth_centrosymmetry_255_full_col:
  LD r8,r6,0                ; A[i,j]
  LD r9,r7,0                ; A[254-i,254-j]
  CMP r8,r9
  JMP[NE] @mth_centrosymmetry_255_full_mismatch
  ADD r6,r6,r13             ; next A column
  SUB r7,r7,r13             ; previous mirror column
  SUB r11,r11,r13
  CMP r11,r0
  JMP[GT] @mth_centrosymmetry_255_full_col
  MOVI r3,255
  ADD r4,r4,r3              ; next A row
  SUB r5,r5,r3              ; previous mirror row
  SUB r10,r10,r13
  CMP r10,r0
  JMP[GT] @mth_centrosymmetry_255_full_row
  MOVI r1,1
  RET
mth_centrosymmetry_255_full_mismatch:
  MOVI r1,0
  RET
