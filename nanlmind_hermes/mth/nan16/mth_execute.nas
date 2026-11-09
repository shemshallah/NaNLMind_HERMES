; Native NaN16 execution dispatcher.
;
; ABI:
;   r1 = candidate descriptor
;   r2 = authoritative rule descriptor
;   r3 = left typed leaf descriptor
;   r4 = right typed leaf descriptor
;   r5 = exact canonical 10-word MC0 contract pointer
;   r6 = available capability bits
;   r7 = output typed leaf descriptor
;   r8 = program ID returned by the canonical :: operator linker
;   r9 = operator ID returned by the canonical :: operator linker
;   r12 = caller scratch base (57 words)
;   r13 = constant 1 (set internally)
;
; Allowlist:
;   program 900 + operator 77 -> mth.vector.dot256.f32
;   program 901 + operator 78 -> mth.vector.cosine256.f32
;   program 902 + operator 79 -> mth.vector.add256.f32
;   program 903 + operator 80 -> mth.vector.sub256.f32
; Admission runs before either numerical kernel; denial never reaches a kernel.
; Unsupported or mismatched program/operator pairs are rejected before
; admission/output metadata writes. Return 0 on success, pass through
; mth_admit denials, 64 for unsupported pairs, 80 for kernel ABI failure.
;
; Scratch words 0..47 belong to mth_admit. This wrapper uses 48..56.
; Cosine uses words 52..55 as its four-word caller-owned nested scratch frame.
; r14 is used as the preserved scratch-base register and is clobbered.


mth_execute:
  ; Preserve the canonical operator ID before r9 is reused for program lookup.
  ADD r11,r9,r0
  MOVI r13,1
  MOVI r9,900
  CMP r8,r9
  JMP[EQ] @mth_execute_check_dot
  MOVI r9,901
  CMP r8,r9
  JMP[EQ] @mth_execute_check_cosine
  MOVI r9,902
  CMP r8,r9
  JMP[EQ] @mth_execute_check_add
  MOVI r9,903
  CMP r8,r9
  JMP[EQ] @mth_execute_check_sub
  JMP @mth_execute_unsupported

mth_execute_check_dot:
  ; Program 900 is bound only to canonical operator 77 (vector dot).
  ; r9 is the operator ID returned by the canonical expression linker.
  MOVI r10,77
  CMP r11,r10
  JMP[NE] @mth_execute_unsupported
  JMP @mth_execute_supported

mth_execute_check_cosine:
  ; Program 901 is bound only to canonical operator 78 (vector cosine).
  MOVI r10,78
  CMP r11,r10
  JMP[NE] @mth_execute_unsupported
  JMP @mth_execute_supported

mth_execute_check_add:
  MOVI r10,79
  CMP r11,r10
  JMP[NE] @mth_execute_unsupported
  JMP @mth_execute_supported

mth_execute_check_sub:
  MOVI r10,80
  CMP r11,r10
  JMP[NE] @mth_execute_unsupported

mth_execute_supported:
  ADD r14,r12,r0
  ST r15,r14,56
  ST r1,r14,48
  ST r2,r14,49
  ST r3,r14,50
  ST r4,r14,51
  ST r5,r14,52
  ST r6,r14,53
  ST r7,r14,54
  ST r8,r14,55

  ; The exact registry-owned contract pointer and typed operands enter both gates.
  CALL @mth_admit
  ADD r12,r14,r0
  CMP r1,r0
  JMP[NE] @mth_execute_return

  ; Select the already-checked program. Payload ranges and dimensions were
  ; admitted; the kernels receive only the exact registered operand buffers.
  LD r8,r14,55
  MOVI r9,900
  CMP r8,r9
  JMP[EQ] @mth_execute_dot
  MOVI r9,901
  CMP r8,r9
  JMP[EQ] @mth_execute_cosine
  MOVI r9,902
  CMP r8,r9
  JMP[EQ] @mth_execute_add
  JMP @mth_execute_sub

mth_execute_add:
  LD r1,r14,50
  LD r1,r1,5
  LD r2,r14,51
  LD r2,r2,5
  LD r3,r14,54
  LD r3,r3,5
  CALL @mth_vector_add256_f32
  JMP @mth_execute_kernel_status

mth_execute_sub:
  LD r1,r14,50
  LD r1,r1,5
  LD r2,r14,51
  LD r2,r2,5
  LD r3,r14,54
  LD r3,r3,5
  CALL @mth_vector_sub256_f32
  JMP @mth_execute_kernel_status

mth_execute_cosine:
  LD r1,r14,50
  LD r1,r1,5
  LD r2,r14,51
  LD r2,r2,5
  LD r3,r14,54
  LD r3,r3,5
  MOVI r4,52
  ADD r4,r14,r4
  CALL @mth_cosine256_f32
  JMP @mth_execute_kernel_status

mth_execute_dot:
  LD r1,r14,50
  LD r1,r1,5
  LD r2,r14,51
  LD r2,r2,5
  LD r3,r14,54
  LD r3,r3,5
  CALL @mth_dot256_f32

mth_execute_kernel_status:
  MOVI r9,1
  CMP r1,r9
  JMP[NE] @mth_execute_kernel_error
  MOVI r1,0
  JMP @mth_execute_return

mth_execute_unsupported:
  MOVI r1,64
  RET

mth_execute_kernel_error:
  MOVI r1,80

mth_execute_return:
  ADD r12,r14,r0
  LD r15,r12,56
  RET
