; MTH full 256-dimensional adapter residual — NaN16 float32 only.
;
; Computes the complete dense adapter residual
;   y[i] = h[i] + beta * sum_{j=0..255} W[i,j] * u[j]     for i = 0..255
; where h is the frozen base state, u is the bounded 256-float32 readout
; state, W is a dense typed decode map, and beta is the residual scale.
;
; Arithmetic order: for each row i, one accumulator is seeded with h[i], then
; for increasing j = 0..255 the kernel evaluates
;   t1  = beta * W[i,j]
;   t2  = t1 * u[j]
;   acc = acc + t2
; i.e. the left-to-right form
;   h[i] + beta*W[i,0]*u[0] + beta*W[i,1]*u[1] + ...
; with two FMULs followed by one FADD per term. No FMA, no reordering, no
; host math. Because beta is applied per term before accumulation, this order
; is not in general bitwise identical to h[i] + beta*(sum_j W[i,j]*u[j]).
;
; ABI:
;   r1  = W, 65,536 float32, row-major 256 rows x 256 columns
;   r2  = h, 256 float32
;   r3  = u, 256 float32
;   r4  = beta, one float32, passed by value
;   r5  = y, 256 float32 output
;   r13 = 1
;   returns r1 = 1
;   clobbers r1, r2, r5-r12, r14; preserves r3, r4, r13, r15
;   leaf: no CALL, no static slots, no self-modification.
;
; Working set: 65,536 (W) + 256 (h) + 256 (u) + 256 (y) = 66,304 words, which
; fits the 1 MiB (131,072-word) guest budget but not the historical 512 KiB
; (65,536-word) budget. This is the monolithic form of the paged
; mth_adapter_residual_256_block16 kernel: one call computes all 256 rows, so
; the caller does not perform the 16-block composition. W, h, and u are
; read-only; only y is written. The four spans must be distinct and
; non-overlapping (caller precondition). beta's [0,1] bound and finiteness are
; contract preconditions from the adapter descriptor; this arithmetic kernel
; does not validate them, and does not validate overlap, NaN, or Inf.

mth_adapter_residual_256_full:
  ADD r6,r4,r0               ; r6 = beta
  MOVI r9,0                  ; row i
mth_adapter_residual_256_full_row:
  LD r11,r2,0                ; accumulator seeded with h[i]
  ADD r8,r3,r0               ; u pointer, reset for this row
  MOVI r10,0                 ; column j
mth_adapter_residual_256_full_col:
  LD r12,r1,0                ; W[i,j]
  LD r14,r8,0                ; u[j]
  FMUL r12,r6,r12            ; beta * W[i,j]
  FMUL r12,r12,r14           ; (beta * W[i,j]) * u[j]
  FADD r11,r11,r12           ; acc = acc + term
  ADD r1,r1,r13
  ADD r8,r8,r13
  ADD r10,r10,r13
  MOVI r12,256
  CMP r10,r12
  JMP[LT] @mth_adapter_residual_256_full_col
  ST r11,r5,0                ; y[i]
  ADD r2,r2,r13
  ADD r5,r5,r13
  ADD r9,r9,r13
  MOVI r12,256
  CMP r9,r12
  JMP[LT] @mth_adapter_residual_256_full_row
  MOVI r1,1
  RET
