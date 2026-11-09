; MTH dense 256x256 float32 matrix addition — NaN16 float32 only.
;
; ABI:
;   r1 = A, 65,536 float32 payloads, row-major 256x256
;   r2 = B, 65,536 float32 payloads, row-major 256x256
;   r3 = C, 65,536 float32 payloads, row-major 256x256
;   r13 = 1
;   returns r1=1
;   clobbers r4-r8 (r9-r12, r14 and r15 are preserved)
;
; C[k] = A[k] + B[k] for k = 0..65,535 in increasing linear element order.
; Each element executes exactly one FADD from the two loaded payloads; there is
; no host math, no reordering, no fused multiply-add, and no reassociation.
; This is a leaf: no CALL and no static workspace.
;
; The caller supplies readable input spans and a writable output span. Exact
; aliasing of C with A or B is safe because each element is loaded before its
; result is stored. Partial overlap between any two spans is not safe.

mth_matrix_add256_f32:
  MOVI r4,0             ; remaining elements
  MOVHI r4,0x0001       ; 65,536 = 0x00010000
  MOVI r5,0             ; linear word offset k
mth_matrix_add256_loop:
  ADD r6,r1,r5
  LD r6,r6,0            ; A[k]
  ADD r7,r2,r5
  LD r7,r7,0            ; B[k]
  FADD r8,r6,r7         ; C[k] = A[k] + B[k]
  ADD r6,r3,r5
  ST r8,r6,0
  ADD r5,r5,r13         ; next element
  SUB r4,r4,r13
  CMP r4,r0
  JMP[GT] @mth_matrix_add256_loop
  MOVI r1,1
  RET
