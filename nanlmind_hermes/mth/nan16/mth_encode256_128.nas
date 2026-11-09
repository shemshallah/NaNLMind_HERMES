; MTH dense 128x256 encode map (256 -> 128) — NaN16 float32 only.
;
; ABI:
;   r1 = W, 32,768 float32 payloads, row-major 128 rows x 256 columns
;   r2 = x, 256 float32
;   r3 = b, 128 float32 bias
;   r4 = y, 128 float32 output
;   r13 = 1
;   returns r1=1
;   clobbers r1-r12, r14; preserves r13 and r15
;
; y[i] = b[i] + sum_{j=0}^{255} W[i,j] * x[j]
;
; Rows are visited in increasing i and columns in increasing j. One
; accumulator per row is seeded with b[i] and then receives one rounded
; FMUL followed by one rounded FADD per column, so the evaluation order is
; b[i] + W[i,0]x[0] + ... + W[i,255]x[255]. No FMA, no reordering, no host
; math. Inputs are read-only; W, x, b, and y must be distinct non-overlapping
; spans (caller precondition). This is a prototype dense kernel: no training,
; no sparsity, no overlap/NaN/Inf validation.

mth_encode256_128_f32:
  ADD r5,r1,r0             ; W row base cursor
  ADD r6,r2,r0             ; x base
  ADD r7,r3,r0             ; b cursor
  ADD r8,r4,r0             ; y cursor
  MOVI r9,0                ; i
mth_encode256_128_row:
  LD r10,r7,0              ; acc = b[i]
  ADD r11,r5,r0            ; &W[i,0]
  ADD r12,r6,r0            ; &x[0]
  MOVI r14,0               ; j
mth_encode256_128_col:
  LD r1,r11,0              ; W[i,j]
  LD r2,r12,0              ; x[j]
  FMUL r3,r1,r2
  FADD r10,r10,r3          ; acc = acc + W[i,j]*x[j]
  ADD r11,r11,r13
  ADD r12,r12,r13
  ADD r14,r14,r13
  MOVI r3,256
  CMP r14,r3
  JMP[LT] @mth_encode256_128_col
  ST r10,r8,0              ; y[i]
  MOVI r3,256              ; row stride in words
  ADD r5,r5,r3
  ADD r7,r7,r13
  ADD r8,r8,r13
  ADD r9,r9,r13
  MOVI r3,128
  CMP r9,r3
  JMP[LT] @mth_encode256_128_row
  MOVI r1,1
  RET
