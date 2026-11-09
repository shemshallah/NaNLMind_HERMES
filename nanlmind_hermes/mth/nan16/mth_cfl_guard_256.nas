; MTH R4 CFL / lambda_max guard — NaN16 float32 only.
;
; ABI:
;   r1 = A, 65,536 float32 payloads, row-major 256x256
;   r2 = dt, one float32 payload
;   r3 = lambda_out, one float32 payload
;   r13 = 1
;   returns r1=1 iff dt*lambda <= 2.0, else 0
;   clobbers r4-r12, r14, flags
;
; lambda = max_i sum_j |A[i,j]|, rows and columns in increasing order, one
; row accumulator. |x| clears the float32 sign bit with BOOL.8 against
; 0x7FFFFFFF because NaN16 has no FABS; the mask is exact for every float32
; bit pattern and is not host math. FCMP then implements the inclusive
; dt*lambda <= 2.0 guard, so the exact boundary 2.0 accepts and an unordered
; (NaN) product rejects.
;
; One row accumulator exists at any time; no 256-element scalar array is
; materialized. lambda_out is written on both accept and reject paths.
;
; Working set: 65,536 + 2 scalar words = 65,538 NaN16 words, leaving 65,534
; words of the 1 MiB (131,072-word) guest budget for code and scratch.
; Finite inputs are assumed; NaN/Inf domain policy is caller-side.

mth_cfl_guard_256:
  LD r4,r2,0                ; dt
  MOVI r5,0                 ; lambda bits = +0.0
  MOVI r11,0
  MOVHI r11,0x7FFF          ; r11 = 0x7FFFFFFF sign mask
  MOVI r14,256
  ADD r6,r1,r0              ; row base = A
  MOVI r7,0                 ; i
mth_cfl_guard_256_row:
  MOVI r8,0                 ; row sum = +0.0
  ADD r10,r6,r0             ; element pointer
  MOVI r9,0                 ; j
mth_cfl_guard_256_col:
  LD r12,r10,0
  BOOL.8 r12,r12,r11        ; |A[i,j]| = payload & 0x7FFFFFFF
  FADD r8,r8,r12
  ADD r10,r10,r13
  ADD r9,r9,r13
  CMP r9,r14
  JMP[LT] @mth_cfl_guard_256_col
  FCMP r8,r5
  JMP[GT] @mth_cfl_guard_256_take
  JMP @mth_cfl_guard_256_next
mth_cfl_guard_256_take:
  ADD r5,r8,r0              ; max = row sum (bit copy)
mth_cfl_guard_256_next:
  MOVI r12,256
  ADD r6,r6,r12
  ADD r7,r7,r13
  CMP r7,r14
  JMP[LT] @mth_cfl_guard_256_row
  ST r5,r3,0                ; lambda_out
  FMUL r5,r5,r4             ; dt*lambda
  MOVI r6,0
  MOVHI r6,0x4000           ; r6 = 2.0
  FCMP r5,r6
  MOVI r1,1
  JMP[LE] @mth_cfl_guard_256_done
  MOVI r1,0
mth_cfl_guard_256_done:
  RET
