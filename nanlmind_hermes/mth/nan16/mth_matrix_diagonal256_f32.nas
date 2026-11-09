; MTH dense 256x256 float32 matrix diagonal extraction — NaN16 float32 only.
;
; ABI:
;   r1 = input, 65,536 float32 payloads, row-major 256x256
;   r2 = output, 256 float32 payloads, D[i] = A[i,i]
;   r13 = 1
;   returns r1=1
;   clobbers r4-r8, r12 (r2, r3, r9-r11, r13, and r14 are preserved)
;
; The input cursor starts at A[0,0] and advances by 257 words for each next
; diagonal element (one row plus one column). The output cursor advances by
; one word. Each diagonal element is one LD followed by one ST; no arithmetic
; on the payload, no host math, no reordering. This is a leaf: no CALL and no
; static workspace.
;
; Input is read-only. The output span of 256 words must be distinct from and
; non-overlapping with the input span (caller precondition), because a store to
; an overlapping output could overwrite a diagonal element not yet loaded.
;
; Working set: 65,536 input words + 256 output words = 65,792 words
; (526,336 bytes as 8-byte NaN16 guest words), leaving 196,352 words of the
; 2 MiB (262,144-word) guest budget for code, fixture data, and scratch.

mth_matrix_diagonal256_f32:
  ADD r4,r1,r0          ; input cursor = &A[0,0]
  ADD r5,r2,r0          ; output cursor = &D[0]
  MOVI r6,0             ; i
  MOVI r7,256           ; extent
  MOVI r12,257          ; next diagonal stride (256 + 1 words)
mth_matrix_diagonal256_loop:
  LD r8,r4,0            ; A[i,i]
  ST r8,r5,0            ; D[i]
  ADD r4,r4,r12         ; next input diagonal element
  ADD r5,r5,r13         ; next output element
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_matrix_diagonal256_loop
  MOVI r1,1
  RET
