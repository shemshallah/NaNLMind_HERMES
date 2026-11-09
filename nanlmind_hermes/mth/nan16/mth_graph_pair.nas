; MTH dense 128x128 pair-space action — NaN16 float32 only.
;
;   U' = L U + U L^T
;
; L and U are dense 128x128 row-major float32 matrices. Both terms are
; evaluated in float32 with FADD/FSUB/FMUL only; there is no host math,
; no FMA, no reassociation, and no fast-math mode.
;
; ABI (r13=1):
;   r1  = L, 16,384 float32 payloads (row-major 128x128)
;   r2  = U, 16,384 float32 payloads (row-major 128x128)
;   r3  = output U', 16,384 float32 payloads (row-major 128x128)
;   r13 = 1
;   returns r1 = 1
;
; Preconditions:
;   - L, U, and output are pairwise non-overlapping. Inputs are read-only.
;     This kernel does not detect overlap.
;   - Output need not be initialized; every word is written.
;   - r0 = 0 on entry. The kernel restores r0 = 0 on return.
;
; Register effects: r0-r12 and r14 are clobbered; r13 stays 1; r15 is
; preserved so ordinary CALL/RET works. No static slots are used.
;
; Memory bound: 3 * 16,384 = 49,152 words, inside the 65,536-word guest.
; A full 256x256 pair space needs four 65,536-word matrices (L, U, output,
; plus temporaries) and cannot be materialized in one guest; it must be
; paged.
;
; Accumulation order (deterministic, increasing k):
;   Pass 1: P[i,j] = sum_{k=0}^{127} L[i,k] * U[k,j]
;           one accumulator initialized to +0.0f, k = 0..127 in order,
;           one rounded FMUL and one rounded FADD per k; P is stored to
;           the output array.
;   Pass 2: Q[i,j] = sum_{k=0}^{127} U[i,k] * L[j,k]
;           one accumulator initialized to +0.0f, k = 0..127 in order,
;           one rounded FMUL and one rounded FADD per k; then
;           output[i,j] = P[i,j] + Q[i,j] with one final rounded FADD.
;
; The two term sums are the "two accumulators" of the specification; they
; are not interleaved, but each is accumulated in increasing k and their
; only combination is the single final FADD. No product is reordered.

mth_graph_pair:
  MOVI r0,0

  ; ---- Pass 1: output = L * U ----
  MOVI r4,0                 ; i = 0
  ADD r6,r1,r0              ; L row 0 base
mth_pair_row1:
  ADD r7,r2,r0              ; U column 0 base for j = 0
  MOVI r5,0                 ; j = 0
  MOVI r12,128
  ADD r10,r6,r12            ; end of this L row
mth_pair_col1:
  ADD r8,r6,r0              ; pL = L row base
  ADD r9,r7,r0              ; pU = U column base
  MOVI r11,0                ; acc = +0.0f
mth_pair_k1:
  LD r12,r8,0               ; L[i,k]
  LD r14,r9,0               ; U[k,j]
  FMUL r12,r12,r14
  FADD r11,r11,r12
  ADDI r8,r8,1
  ADDI r9,r9,128
  CMP r8,r10
  JMP[LT] @mth_pair_k1
  ST r11,r3,0               ; P[i,j]
  ADDI r3,r3,1
  ADDI r7,r7,1              ; next U column
  ADDI r5,r5,1
  MOVI r12,128
  CMP r5,r12
  JMP[LT] @mth_pair_col1
  ADDI r6,r6,128            ; next L row
  ADDI r4,r4,1
  MOVI r12,128
  CMP r4,r12
  JMP[LT] @mth_pair_row1

  ; ---- Pass 2: output += U * L^T ----
  MOVI r12,16384
  SUB r3,r3,r12             ; rewind output to row 0
  MOVI r4,0                 ; i = 0
  ADD r6,r2,r0              ; U row 0 base
mth_pair_row2:
  ADD r7,r1,r0              ; L row 0 base for j = 0
  MOVI r5,0                 ; j = 0
  MOVI r12,128
  ADD r10,r6,r12            ; end of this U row
mth_pair_col2:
  ADD r8,r6,r0              ; pU = U row base
  ADD r9,r7,r0              ; pL = L row base
  MOVI r11,0                ; acc = +0.0f
mth_pair_k2:
  LD r12,r8,0               ; U[i,k]
  LD r14,r9,0               ; L[j,k]
  FMUL r12,r12,r14
  FADD r11,r11,r12
  ADDI r8,r8,1
  ADDI r9,r9,1
  CMP r8,r10
  JMP[LT] @mth_pair_k2
  LD r12,r3,0               ; P[i,j]
  FADD r11,r11,r12          ; P[i,j] + Q[i,j]
  ST r11,r3,0
  ADDI r3,r3,1
  ADDI r7,r7,128            ; next L row for L[j,k]
  ADDI r5,r5,1
  MOVI r12,128
  CMP r5,r12
  JMP[LT] @mth_pair_col2
  ADDI r6,r6,128            ; next U row
  ADDI r4,r4,1
  MOVI r12,128
  CMP r4,r12
  JMP[LT] @mth_pair_row2

  MOVI r0,0
  MOVI r1,1
  RET
