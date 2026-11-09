; MTH 256-element float32 argmin — NaN16 only.
;
; ABI:
;   r1  = x, 256 float32 payloads (read-only)
;   r2  = index_out, one word (written on success only)
;   r3  = value_out, one float32 payload (written on success only)
;   r13 = 1
;   returns r1 = 1 on success, r1 = 0 on rejection
;   clobbers r1, r4-r8 and the flags
;   preserves r0, r2, r3, r9-r15
;
; Result:
;   index_out = smallest i such that x[i] is the minimum of x[0..255]
;   value_out = the exact 32-bit payload of x[index_out]
;
; DOMAIN AND FAIL-CLOSED POLICY:
;   Every input element must be finite: neither NaN nor +inf nor -inf.
;   A preflight over all 256 words tests
;
;     (payload & 0x7F800000) == 0x7F800000
;
;   which is exactly the binary32 exponent-all-ones class (NaN and both
;   infinities). The first such element rejects the call with r1=0. The
;   whole preflight finishes before the first output store, so on rejection
;   index_out and value_out are untouched. This is a finite-domain contract:
;   the kernel never returns a result for a non-finite input, and it does not
;   attempt NaN ordering or IEEE maxNum/minNum semantics.
;
; TIE RULE:
;   The scan visits i = 0..255 in increasing order. The running best is
;   replaced only when candidate < best (strictly less). An equal value
;   never replaces the current best, so the first index among equal minima
;   wins. -0.0 and +0.0 compare equal under float32 FCMP, so the first of
;   them wins that tie and value_out preserves that occurrence's payload bits.
;   With all 256 values equal, index_out=0 and value_out=x[0].
;
; No host math, no CALL, no static storage, and no allocation. The kernel
; uses only integer address arithmetic, one exact float32 FCMP per scanned
; element after the preflight, and bit copies.

mth_vector_argmin256_f32:
  ; Preflight: reject any non-finite payload before touching the outputs.
  MOVI r5,0
  MOVHI r5,0x7F80            ; r5 = 0x7F800000 exponent mask
  MOVI r4,0
mth_vector_argmin256_preflight:
  ADD r6,r1,r4
  LD r6,r6,0
  BOOL.8 r6,r6,r5            ; exponent bits only
  CMP r6,r5
  JMP[EQ] @mth_vector_argmin256_invalid
  ADD r4,r4,r13
  MOVI r6,256
  CMP r4,r6
  JMP[LT] @mth_vector_argmin256_preflight

  ; Seed with x[0]. The preflight proved it finite, so every later FCMP is
  ; ordered and the strict-predicate scan is total.
  LD r6,r1,0                 ; best value payload
  MOVI r7,0                  ; best index
  MOVI r4,1                  ; i = 1
mth_vector_argmin256_scan:
  ADD r5,r1,r4
  LD r8,r5,0                 ; candidate
  FCMP r8,r6
  JMP[LT] @mth_vector_argmin256_take
  JMP @mth_vector_argmin256_next
mth_vector_argmin256_take:
  ADD r6,r8,r0               ; best = candidate (exact payload copy)
  ADD r7,r4,r0               ; best_index = i
mth_vector_argmin256_next:
  ADD r4,r4,r13
  MOVI r5,256
  CMP r4,r5
  JMP[LT] @mth_vector_argmin256_scan

  ; Publish only after every element was validated and scanned.
  ST r7,r2,0
  ST r6,r3,0
  MOVI r1,1
  RET

mth_vector_argmin256_invalid:
  MOVI r1,0
  RET
