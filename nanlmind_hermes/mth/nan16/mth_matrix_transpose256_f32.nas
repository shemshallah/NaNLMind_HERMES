; MTH dense 256x256 float32 matrix transpose — NaN16 float32 only.
;
; ABI:
;   r1 = input, 65,536 float32 payloads, row-major 256x256
;   r2 = output, 65,536 float32 payloads, row-major 256x256
;   r13 = 1
;   returns r1=1
;   clobbers r4-r10, r12 (r11 and r14 are preserved)
;
; B[j,i] = A[i,j], with increasing i (input row) and increasing j (input
; column). Each element is one LD followed by one ST; no arithmetic, no host
; math, no reordering. This is a leaf: no CALL and no static workspace.
;
; Input is read-only; input and output spans must be distinct and
; non-overlapping (caller precondition). In-place transposition is not
; supported because each output element is overwritten by a later row visit.
;
; Working set: 65,536 input words + 65,536 output words = 131,072 words
; (1 MiB as 8-byte NaN16 guest words), leaving 131,072 words of the 2 MiB
; (262,144-word) guest budget for code, fixture data, and scratch.

mth_matrix_transpose256_f32:
  ADD r4,r1,r0          ; input row base
  ADD r5,r2,r0          ; output column base
  MOVI r6,0             ; i
  MOVI r12,256          ; row/column extent
mth_matrix_transpose256_row:
  ADD r7,r4,r0          ; input cursor = &A[i,0]
  ADD r8,r5,r0          ; output cursor = &B[0,i]
  MOVI r9,0             ; j
mth_matrix_transpose256_col:
  LD r10,r7,0           ; A[i,j]
  ST r10,r8,0           ; B[j,i]
  ADD r7,r7,r13         ; next input column
  ADD r8,r8,r12         ; next output row (256 words)
  ADD r9,r9,r13
  CMP r9,r12
  JMP[LT] @mth_matrix_transpose256_col
  ADD r4,r4,r12         ; next input row
  ADD r5,r5,r13         ; next output column
  ADD r6,r6,r13
  CMP r6,r12
  JMP[LT] @mth_matrix_transpose256_row
  MOVI r1,1
  RET
