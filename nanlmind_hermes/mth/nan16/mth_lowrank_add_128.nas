; MTH rank-1 additive coupling on a dense 128x128 float32 matrix — NaN16 only.
;
; U[i,j] = U[i,j] + alpha * b[i] * c[j]
;
; ABI:
;   r1 = U, 16,384 float32 payloads, row-major 128x128, updated in place
;   r2 = b, 128 float32
;   r3 = c, 128 float32
;   r4 = alpha, one float32 payload
;   r13 = 1
;   returns r1=1 success
;   clobbers r2-r12, r14; no CALLs, no static slots.
;
; Iteration is increasing i (row), then increasing j (column). Each row
; computes s = round(alpha * b[i]) with one FMUL and reuses s for all 128
; columns. Each element then performs one rounded FMUL t = round(s * c[j])
; followed by one rounded FADD U[i,j] + t. No FMA and no reordering.
;
; U must be writable and disjoint from b and c (caller precondition; this
; routine performs no overlap, aliasing, NaN, or Inf validation).
;
; One call applies one rank-1 term. A rank-R update is R calls; the caller
; owns the loop and accumulates in the same fixed order.

mth_lowrank_add_128_f32:
  LD r14,r4,0            ; alpha
  ADD r5,r2,r0           ; b cursor
  ADD r6,r3,r0           ; c base
  MOVI r2,128            ; row and column bound
  ADD r10,r1,r0          ; U row cursor
  MOVI r7,0              ; i
mth_lowrank_add_128_row:
  LD r8,r5,0             ; b[i]
  FMUL r8,r8,r14         ; s = alpha*b[i], rounded once per row
  MOVI r9,0              ; j
mth_lowrank_add_128_col:
  ADD r11,r10,r9         ; &U[i,j]
  LD r12,r11,0           ; U[i,j]
  ADD r3,r6,r9           ; &c[j]
  LD r3,r3,0             ; c[j]
  FMUL r3,r3,r8          ; s*c[j]
  FADD r12,r12,r3        ; U[i,j] + s*c[j]
  ST r12,r11,0
  ADD r9,r9,r13
  CMP r9,r2
  JMP[LT] @mth_lowrank_add_128_col
  ADD r5,r5,r13          ; next b element
  ADD r10,r10,r2         ; next U row (128 words)
  ADD r7,r7,r13
  CMP r7,r2
  JMP[LT] @mth_lowrank_add_128_row
  MOVI r1,1
  RET
