; MTH blockwise 256-dimensional adapter residual — NaN16 float32 only.
;
; Computes 16 rows of the adapter residual
;   out[k] = h[k] + beta * sum_{j=0..255} C[k,j] * u[j]     for k = 0..15
; where h is a 16-float32 block of the frozen base state, C is one 16x256
; row-major block of a dense typed decode map, u is the 256-float32 bounded
; readout state, and beta is the residual scale.
;
; Arithmetic order: for each row k, one accumulator is seeded with h[k], then
; for increasing j = 0..255 the kernel evaluates
;   acc = acc + (beta * C[k,j]) * u[j]
; i.e. the left-to-right form
;   h[k] + beta*C[k,0]*u[0] + beta*C[k,1]*u[1] + ...
; with two FMULs followed by one FADD per term. No FMA, no reordering, no
; host math. Because beta is applied per term before accumulation, this order
; is not in general bitwise identical to h[k] + beta*(sum_j C[k,j]*u[j]).
;
; ABI:
;   r1  = h block, 16 float32
;   r2  = C block, 4,096 float32, row-major 16 rows x 256 columns
;   r3  = u, 256 float32
;   r4  = beta, one float32, passed by value
;   r5  = out block, 16 float32
;   r13 = 1
;   returns r1 = 1
;   clobbers r6-r12, r14; preserves r13 and r15
;   leaf: no CALL, no static slots, no self-modification.
;
; A full 256x256 adapter residual is a caller-driven sequence of 16 calls:
; for block b = 0..15 pass h + b*16, C + b*16*256, and out + b*16, keeping the
; same u and beta. The full C matrix (65,536 words) is never resident; only
; one 4,096-word row block is. Inputs are read-only; the three spans must be
; distinct and non-overlapping (caller precondition). beta's [0,1] bound and
; finiteness are contract preconditions from the adapter descriptor; this
; arithmetic kernel does not validate them, and does not validate overlap,
; NaN, or Inf.

mth_adapter_residual_256_block16:
  ADD r6,r4,r0               ; r6 = beta
  MOVI r9,0                  ; row k
mth_adapter_residual_256_block16_row:
  LD r11,r1,0                ; accumulator seeded with h[k]
  ADD r8,r3,r0               ; u pointer, reset for this row
  MOVI r10,0                 ; column j
mth_adapter_residual_256_block16_col:
  LD r12,r2,0                ; C[k,j]
  LD r14,r8,0                ; u[j]
  FMUL r12,r6,r12            ; beta * C[k,j]
  FMUL r12,r12,r14           ; (beta * C[k,j]) * u[j]
  FADD r11,r11,r12           ; acc = acc + term
  ADD r2,r2,r13
  ADD r8,r8,r13
  ADD r10,r10,r13
  MOVI r12,256
  CMP r10,r12
  JMP[LT] @mth_adapter_residual_256_block16_col
  ST r11,r5,0                ; out[k]
  ADD r1,r1,r13
  ADD r5,r5,r13
  ADD r9,r9,r13
  MOVI r12,16
  CMP r9,r12
  JMP[LT] @mth_adapter_residual_256_block16_row
  MOVI r1,1
  RET
