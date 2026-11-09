; MTH dense 256x128 decode map (128 -> 256) — NaN16 float32 only.
;
; ABI:
;   r1 = W, 32,768 float32 payloads, row-major 256 rows x 128 columns
;   r2 = z, 128 float32
;   r3 = b, 256 float32 bias
;   r4 = y, 256 float32 output
;   r13 = 1
;   returns r1=1
;   clobbers r1-r12, r14; preserves r13 and r15
;
; y[i] = b[i] + sum_{j=0}^{127} W[i,j] * z[j]
;
; Rows are visited in increasing i and columns in increasing j. One
; accumulator per row is seeded with b[i] and then receives one rounded
; FMUL followed by one rounded FADD per column, so the evaluation order is
; b[i] + W[i,0]z[0] + ... + W[i,127]z[127]. No FMA, no reordering, no host
; math. Inputs are read-only; W, z, b, and y must be distinct non-overlapping
; spans (caller precondition). This is a prototype dense kernel: no training,
; no sparsity, no overlap/NaN/Inf validation.

mth_decode128_256_f32:
  ADD r5,r1,r0             ; W row base cursor
  ADD r6,r2,r0             ; z base
  ADD r7,r3,r0             ; b cursor
  ADD r8,r4,r0             ; y cursor
  MOVI r9,0                ; i
mth_decode128_256_row:
  LD r10,r7,0              ; acc = b[i]
  ADD r11,r5,r0            ; &W[i,0]
  ADD r12,r6,r0            ; &z[0]
  MOVI r14,0               ; j
mth_decode128_256_col:
  LD r1,r11,0              ; W[i,j]
  LD r2,r12,0              ; z[j]
  FMUL r3,r1,r2
  FADD r10,r10,r3          ; acc = acc + W[i,j]*z[j]
  ADD r11,r11,r13
  ADD r12,r12,r13
  ADD r14,r14,r13
  MOVI r3,128
  CMP r14,r3
  JMP[LT] @mth_decode128_256_col
  ST r10,r8,0              ; y[i]
  MOVI r3,128              ; row stride in words
  ADD r5,r5,r3
  ADD r7,r7,r13
  ADD r8,r8,r13
  ADD r9,r9,r13
  MOVI r3,256
  CMP r9,r3
  JMP[LT] @mth_decode128_256_row
  MOVI r1,1
  RET
