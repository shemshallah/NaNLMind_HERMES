; MTH 256-point complex DFT reference — native NaN16 float32 only.
;
; Direct O(N^2) evaluation of
;   X[k] = sum_{n=0}^{255} x[n] * (W^k)^n,   W = exp(-2*pi*i/256)
; with a caller-supplied complex twiddle table. The kernel performs no
; host sin/cos and does not build a twiddle table of its own.
;
; ABI (r13=1):
;   r1 = real input      x_re[256]  (float32)
;   r2 = imaginary input x_im[256]  (float32)
;   r3 = twiddle real    W_re[256]  (float32), W_re[k]=cos(2*pi*k/256)
;   r4 = twiddle imag    W_im[256]  (float32), W_im[k]=-sin(2*pi*k/256)
;   r5 = real output     X_re[256]  (float32)
;   r6 = imaginary output X_im[256] (float32)
; The six arrays must be distinct and non-overlapping. Inputs are read-only.
; Output need not be initialized; every X[k] is fully overwritten.
;
; Returns r1=1. There is no malformed-input status and no preflight; the
; caller owns the array extents. r7-r12 and r14 are clobbered. r0 is used
; as a temporary but restored to 0, and r13 is preserved at 1. r15 is not
; touched, so ordinary CALL/RET works.
;
; Recurrence and accumulation order (float32, round-to-nearest-even):
;   for k = 0..255:
;     acc = 0 + 0i
;     w   = 1 + 0i
;     for n = 0..255:
;       acc_re += x_re[n]*w_re
;       acc_re -= x_im[n]*w_im
;       acc_im += x_re[n]*w_im
;       acc_im += x_im[n]*w_re
;       w = w * W^k                    ; four FMUL, one FSUB, one FADD
;     X[k] = acc
; The two real additions in each component are separate rounded operations;
; no FMA is used. The power w is advanced only by repeated complex
; multiplication and is never renormalized. See docs/MTH_GRAPH_DFT_ABI.md
; for the precision contract and its measured limits.
;
; Implementation detail: the real output slot X_re[k] is used as a one-word
; temporary for the new w_re during each recurrence step, then overwritten
; with the final accumulators after the n loop. This is why outputs must be
; writable and distinct from inputs and twiddles.

mth_graph_dft256:
  MOVI r7,0                  ; k = 0
  ADDI r8,r1,256             ; end of one real-input pass
mth_dft_k_loop:
  MOVI r9,0                  ; acc_re = +0.0f
  MOVI r10,0                 ; acc_im = +0.0f
  MOVI r11,0
  MOVHI r11,0x3F80           ; w_re = 1.0f
  MOVI r12,0                 ; w_im = +0.0f
mth_dft_n_loop:
  ; acc_re += x_re*w_re - x_im*w_im
  LD r14,r1,0
  FMUL r0,r14,r11
  FADD r9,r9,r0
  LD r14,r2,0
  FMUL r0,r14,r12
  FSUB r9,r9,r0
  ; acc_im += x_re*w_im + x_im*w_re
  LD r14,r1,0
  FMUL r0,r14,r12
  FADD r10,r10,r0
  LD r14,r2,0
  FMUL r0,r14,r11
  FADD r10,r10,r0

  ; w <- w * W^k:
  ;   new_w_re = w_re*W_re[k] - w_im*W_im[k]
  ;   new_w_im = w_re*W_im[k] + w_im*W_re[k]
  ADD r0,r3,r7
  LD r0,r0,0                 ; W_re[k]
  ADD r14,r4,r7
  LD r14,r14,0               ; W_im[k]
  FMUL r0,r11,r0             ; w_re*W_re[k]
  FMUL r14,r12,r14           ; w_im*W_im[k]
  FSUB r0,r0,r14             ; new_w_re
  ADD r14,r5,r7
  ST r0,r14,0                ; transient X_re[k] scratch
  ADD r14,r4,r7
  LD r14,r14,0               ; W_im[k]
  FMUL r14,r11,r14           ; w_re*W_im[k]
  ADD r0,r3,r7
  LD r0,r0,0                 ; W_re[k]
  FMUL r0,r12,r0             ; w_im*W_re[k]
  FADD r12,r14,r0            ; new_w_im
  ADD r0,r5,r7
  LD r11,r0,0                ; recover new_w_re

  ADDI r1,r1,1
  ADDI r2,r2,1
  CMP r1,r8
  JMP[LT] @mth_dft_n_loop

  ; Commit this bin only after the full n loop.
  ADD r0,r5,r7
  ST r9,r0,0
  ADD r0,r6,r7
  ST r10,r0,0

  ; Rewind the two input cursors and advance k.
  ADDI r1,r1,-256
  ADDI r2,r2,-256
  ADDI r7,r7,1
  MOVI r14,256
  CMP r7,r14
  JMP[LT] @mth_dft_k_loop

  MOVI r0,0                  ; restore the r0=0 convention
  MOVI r1,1                  ; success
  RET
