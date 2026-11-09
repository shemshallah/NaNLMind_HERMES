; MTH R9 bounded LM-adapter residual composition — NaN16 float32 only.
;
; Computes the R9 adapter residual
;   scratch = C u
;   out[i]  = h[i] + beta * scratch[i]        for i = 0..127
; where h is the frozen base state, u is the bounded readout state, C is a
; dense typed decode map, and beta is the residual scale.
;
; ABI:
;   r1 = h, 128 float32 base state
;   r2 = C, 128x128 float32 row-major decode map (16,384 words)
;   r3 = u, 128 float32 readout state
;   r4 = beta, one float32, passed by value
;   r5 = out, 128 float32 output
;   r6 = scratch, 128 float32 caller-owned workspace
;   r13 = 1
;   returns r1=1 success, r1=0 if the matvec primitive reports failure
;   clobbers r1-r12, r14; preserves r13 and r15
;
; The matvec is the verified mth_matvec128_f32 kernel, included and called
; unchanged. The residual pass is increasing i, one FMUL followed by one FADD;
; no FMA, reordering, or host math. beta's [0,1] bound is a caller/contract
; precondition (docs/MTH_LM_ADAPTER_ABI.md); this arithmetic kernel does not
; validate it. h, u, C, out, and scratch must be distinct non-overlapping
; spans. The kernel is not reentrant: it saves r15 and call arguments in the
; static slots below, so concurrent or nested use requires separate images.

.include "mth_matvec128_f32.nas"

mth_adapter_residual_128:
  ; Preserve the return link and the arguments that the matvec clobbers.
  ST r15,r0,@mth_adapter_residual_save_ret
  ST r1,r0,@mth_adapter_residual_save_h
  ST r4,r0,@mth_adapter_residual_save_beta
  ST r5,r0,@mth_adapter_residual_save_out
  ST r6,r0,@mth_adapter_residual_save_scratch

  ; scratch = C u (verified primitive; its ABI is r1=A, r2=x, r3=y).
  ADD r1,r2,r0
  ADD r2,r3,r0
  ADD r3,r6,r0
  CALL @mth_matvec128_f32

  LD r15,r0,@mth_adapter_residual_save_ret
  CMP r1,r13
  JMP[NE] @mth_adapter_residual_fail

  LD r3,r0,@mth_adapter_residual_save_scratch
  LD r4,r0,@mth_adapter_residual_save_beta
  LD r5,r0,@mth_adapter_residual_save_out
  LD r6,r0,@mth_adapter_residual_save_h

  MOVI r7,0
mth_adapter_residual_loop:
  ADD r9,r6,r7
  LD r10,r9,0                ; h[i]
  ADD r9,r3,r7
  LD r11,r9,0                ; scratch[i] = (C u)[i]
  FMUL r11,r4,r11            ; beta * scratch[i]
  FADD r10,r10,r11           ; h[i] + beta * scratch[i]
  ADD r9,r5,r7
  ST r10,r9,0
  ADD r7,r7,r13
  MOVI r9,128
  CMP r7,r9
  JMP[LT] @mth_adapter_residual_loop

  MOVI r1,1
  RET

mth_adapter_residual_fail:
  MOVI r1,0
  RET

; Static slots: link plus five saved words. Words 768..772 are reserved; a
; caller embedding this kernel must keep code and data out of that range.
.org 768
mth_adapter_residual_save_ret:
.word 0
mth_adapter_residual_save_h:
.word 0
mth_adapter_residual_save_beta:
.word 0
mth_adapter_residual_save_out:
.word 0
mth_adapter_residual_save_scratch:
.word 0
