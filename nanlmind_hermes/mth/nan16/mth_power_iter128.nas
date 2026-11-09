; MTH dense 128x128 float32 symmetric power iteration — NaN16 only.
;
; ABI:
;   r1 = A, 16,384 float32 payloads, row-major 128x128 symmetric
;   r2 = x, 128 float32 input direction
;   r3 = iterations, integer 1..64
;   r4 = lambda output, one float32
;   r5 = v output, 128 float32
;   r6 = scratch, 128 float32 (A*v workspace)
;   r13 = 1
;   returns r1=1 success
;   returns r1=0 for a zero input vector, nonpositive iterations, an
;          out-of-range iteration count, or a degenerate A*v=0 iterate
;   clobbers r1-r12, r14
;
; The verified mth_matvec128_f32 kernel is used by include. v is first
; normalized from x, then each bounded iteration computes scratch=A*v,
; norm=sqrt(sum scratch^2), and v=scratch/norm. The final Rayleigh
; quotient is lambda = dot(v,A*v)/dot(v,v), accumulated with increasing
; index. Dividing by dot(v,v) keeps the quotient scale invariant when the
; float32 normalization leaves ||v||^2 one ulp away from 1.0.
;
; A small private save frame keeps caller pointers and the loop counter
; across matvec calls (matvec clobbers r4-r12,r14). The frame makes this
; kernel non-reentrant and non-concurrent; it uses no host math.

.include "mth_matvec128_f32.nas"

mth_power_iter128:
  ; Save the single-link return address before any nested matvec call.
  MOVI r7,@mth_power_iter128_return
  ST r15,r7,0

  ; Bounded iteration count: reject 0 and anything above 64.
  CMP r3,r0
  JMP[EQ] @mth_power_iter128_invalid
  MOVI r7,64
  CMP r3,r7
  JMP[GT] @mth_power_iter128_invalid

  ; Save caller-owned pointers and the iteration counter.
  MOVI r7,@mth_power_iter128_frame
  ST r1,r7,0
  ST r2,r7,1
  ST r3,r7,2
  ST r4,r7,3
  ST r5,r7,4
  ST r6,r7,5

  ; norm = sqrt(sum x[i]^2), increasing i. A zero vector is invalid.
  MOVI r8,0
  MOVI r9,0
mth_power_iter128_xnorm:
  ADD r10,r2,r8
  LD r10,r10,0
  FMUL r10,r10,r10
  FADD r9,r9,r10
  ADD r8,r8,r13
  MOVI r10,128
  CMP r8,r10
  JMP[LT] @mth_power_iter128_xnorm
  FSQRT r9,r9
  FCMP r9,r0
  JMP[EQ] @mth_power_iter128_invalid

  ; v = x/||x||.
  MOVI r8,0
mth_power_iter128_normalize_x:
  ADD r10,r2,r8
  LD r10,r10,0
  FDIV r10,r10,r9
  ADD r11,r5,r8
  ST r10,r11,0
  ADD r8,r8,r13
  MOVI r10,128
  CMP r8,r10
  JMP[LT] @mth_power_iter128_normalize_x

mth_power_iter128_iter:
  ; scratch = A*v, using the verified dense matvec.
  MOVI r7,@mth_power_iter128_frame
  LD r1,r7,0
  LD r2,r7,4
  LD r3,r7,5
  CALL @mth_matvec128_f32
  CMP r1,r13
  JMP[NE] @mth_power_iter128_invalid

  MOVI r7,@mth_power_iter128_frame
  LD r8,r7,4                 ; v
  LD r9,r7,5                 ; scratch

  ; norm = sqrt(sum scratch[i]^2), increasing i.
  MOVI r10,0
  MOVI r11,0
mth_power_iter128_wnorm:
  ADD r12,r9,r10
  LD r12,r12,0
  FMUL r12,r12,r12
  FADD r11,r11,r12
  ADD r10,r10,r13
  MOVI r12,128
  CMP r10,r12
  JMP[LT] @mth_power_iter128_wnorm
  FSQRT r11,r11
  FCMP r11,r0
  JMP[EQ] @mth_power_iter128_invalid

  ; v = scratch/norm.
  MOVI r10,0
mth_power_iter128_normalize_w:
  ADD r12,r9,r10
  LD r12,r12,0
  FDIV r12,r12,r11
  ADD r14,r8,r10
  ST r12,r14,0
  ADD r10,r10,r13
  MOVI r12,128
  CMP r10,r12
  JMP[LT] @mth_power_iter128_normalize_w

  ; Bounded loop: stop when the saved counter reaches zero.
  LD r10,r7,2
  SUB r10,r10,r13
  ST r10,r7,2
  CMP r10,r0
  JMP[GT] @mth_power_iter128_iter

  ; Final scratch = A*v at the normalized v.
  MOVI r7,@mth_power_iter128_frame
  LD r1,r7,0
  LD r2,r7,4
  LD r3,r7,5
  CALL @mth_matvec128_f32
  CMP r1,r13
  JMP[NE] @mth_power_iter128_invalid

  MOVI r7,@mth_power_iter128_frame
  LD r8,r7,4                 ; v
  LD r9,r7,5                 ; scratch = A*v
  LD r6,r7,3                 ; lambda output

  ; num = sum v[i]*scratch[i], den = sum v[i]*v[i], increasing i.
  MOVI r10,0
  MOVI r11,0
  MOVI r12,0
mth_power_iter128_dot:
  ADD r14,r8,r10
  LD r14,r14,0
  ADD r3,r9,r10
  LD r3,r3,0
  FMUL r3,r14,r3
  FADD r11,r11,r3
  FMUL r14,r14,r14
  FADD r12,r12,r14
  ADD r10,r10,r13
  MOVI r3,128
  CMP r10,r3
  JMP[LT] @mth_power_iter128_dot

  FDIV r11,r11,r12
  ST r11,r6,0
  MOVI r7,@mth_power_iter128_return
  LD r15,r7,0
  MOVI r1,1
  RET

mth_power_iter128_invalid:
  MOVI r7,@mth_power_iter128_return
  LD r15,r7,0
  MOVI r1,0
  RET

; Private save frame: A, x, iterations, lambda, v, scratch.
mth_power_iter128_frame:
.word 0,0,0,0,0,0
mth_power_iter128_return:
.word 0
