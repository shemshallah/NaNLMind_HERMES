; MTH 15x255 block certificate for a 255x255 odd centrosymmetric matrix — NaN16 only.
;
; A 255x255 float32 matrix is 65,025 NaN16 words. The head-only aperture
; therefore certifies one 15-row strip at a time. 255 = 17*15, so exactly 17
; block calls tile the full matrix with no padding row.
;
; ABI:
;   r1  = A block, 3,825 float32 payloads, row-major 15 rows x 255 columns
;   r2  = mirror block, 3,825 float32 payloads, row-major 15 rows x 255 columns
;   r13 = 1
;   returns r1 = 1
;   returns r2 = mismatch count
;   clobbers r1-r12; preserves r13
;
; For block row k (global row i = base+k), the caller supplies global row
; 254-i as mirror row k, in natural column order:
;   mirror[k,c] = M[254-(base+k),c].
; The kernel compares, for k = 0..14 and j = 0..254,
;   A[k,j] == mirror[k,254-j] == M[254-(base+k),254-j]
; with integer CMP on the 32-bit payload. This is bitwise equality, not host
; float equality: +0.0 and -0.0 differ, and differing NaN payloads differ.
;
; The 255-head reversal i <-> 254-i is odd: row 127 and column 127 are fixed.
; There is no special path here. The caller handles them through the supplied
; mirror block:
;   - for base=120 the block contains row 127 as A row 7, and the caller must
;     also supply row 127 as mirror row 7; the kernel then checks that fixed
;     row against its own column reversal (a palindrome test);
;   - for every k, column j=127 is compared against mirror[k,127], so the
;     fixed column is checked across the paired rows by the same loop.
;
; One call reports one mismatch for each failed comparison. Across the 17-call
; full sweep (base = 0,15,...,240) every two-cell reversal orbit is compared
; twice, so one asymmetric off-centre entry contributes two mismatches. The
; centre cell (127,127) is the only one-cell orbit and is compared once.
; The caller owns the accumulator and the block pointers; this kernel
; allocates nothing.
;
; Inputs are read-only. The two 3,825-word spans must be readable and may
; alias or overlap only if the caller accepts that the comparison is then
; not a certificate (no aliasing check is performed).

mth_centrosymmetry255_block15:
  ADD r4,r1,r0              ; A block base
  ADD r5,r2,r0              ; mirror block base
  MOVI r2,0                 ; mismatch count
  ADD r6,r4,r0              ; A row pointer
  ADD r7,r5,r0              ; mirror row pointer
  MOVI r8,15                ; rows remaining

mth_centro255_block15_row:
  ADD r9,r6,r0              ; A column pointer, starts at j=0
  MOVI r3,254
  ADD r10,r7,r3             ; mirror column pointer, starts at j=254
  MOVI r3,255               ; columns remaining

mth_centro255_block15_col:
  LD r11,r9,0               ; A[k,j]
  LD r12,r10,0              ; mirror[k,254-j]
  CMP r11,r12
  JMP[EQ] @mth_centro255_block15_equal
  ADD r2,r2,r13             ; payload mismatch
mth_centro255_block15_equal:
  ADD r9,r9,r13             ; next A column
  SUB r10,r10,r13           ; previous mirror column
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_centro255_block15_col

  MOVI r3,255
  ADD r6,r6,r3              ; next A row
  ADD r7,r7,r3              ; next mirror row
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_centro255_block15_row

  MOVI r1,1
  RET
