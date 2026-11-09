; MTH contiguous unsigned-32 vector addition — NaN16 only.
; No host arithmetic and no hidden allocation.
;
; ABI (r13=1):
;   r1 = address of first U32 input span
;   r2 = address of second U32 input span
;   r3 = address of output U32 span
;   r4 = element count
;   r12 = caller-owned three-word scratch frame
;   returns r1=1 on success, r1=0 on unsigned overflow.
;
; The routine performs a complete validation pass before writing output.
; Therefore overflow leaves the output span unchanged. The three spans must
; be valid, writable/readable for count words, and non-overlapping. Count zero
; is valid and performs no memory access. This is the raw contiguous-span
; primitive; typed shape/stride descriptors are a later layer.

mth_u32_span_add_checked:
  STW r1,r12,r0
  ADD r10,r12,r13
  STW r2,r10,r0
  ADD r10,r10,r13
  STW r3,r10,r0

  ADD r5,r4,r0
mth_u32_span_add_validate:
  CMP r5,r0
  JMP[EQ] @mth_u32_span_add_commit
  LD r6,r1,0
  LD r7,r2,0
  ADD r8,r6,r7
  CMP r8,r6
  JMP[LT] @mth_u32_span_add_overflow
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r5,r5,r13
  JMP @mth_u32_span_add_validate

mth_u32_span_add_commit:
  LD r1,r12,0
  LD r2,r12,1
  LD r3,r12,2
  ADD r5,r4,r0
mth_u32_span_add_write:
  CMP r5,r0
  JMP[EQ] @mth_u32_span_add_ok
  LD r6,r1,0
  LD r7,r2,0
  ADD r8,r6,r7
  STW r8,r3,r0
  ADD r1,r1,r13
  ADD r2,r2,r13
  ADD r3,r3,r13
  SUB r5,r5,r13
  JMP @mth_u32_span_add_write

mth_u32_span_add_ok:
  MOVI r1,1
  RET

mth_u32_span_add_overflow:
  MOVI r1,0
  RET
