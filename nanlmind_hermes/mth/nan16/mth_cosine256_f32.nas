; MTH 256-element float32 cosine similarity — NaN16 only.
;
; ABI:
;   r1 = x, 256 float32
;   r2 = y, 256 float32
;   r3 = output, one float32
;   r4 = caller scratch, three float32 (dot, norm_x, norm_y)
;   r13 = 1
;   returns r1=1
;
; cosine = dot(x,y) / (||x||*||y||), or 0 when either norm is zero.
; Uses the verified dot/norm primitives; no host math.

.include "mth_dot256_f32.nas"
.include "mth_norm256_f32.nas"

mth_cosine256_f32:
  ADD r9,r1,r0
  ADD r10,r2,r0
  ADD r11,r3,r0
  ADD r12,r4,r0
  ST r15,r12,3

  ADD r1,r9,r0
  ADD r2,r10,r0
  ADD r3,r12,r0
  CALL @mth_dot256_f32

  ADD r1,r9,r0
  ADD r2,r12,r13
  CALL @mth_norm256_f32

  ADD r1,r10,r0
  MOVI r5,2
  ADD r2,r12,r5
  CALL @mth_norm256_f32

  LD r5,r12,0
  LD r6,r12,1
  LD r7,r12,2
  FMUL r6,r6,r7
  FCMP r6,r0
  JMP[EQ] @mth_cosine_zero
  FDIV r5,r5,r6
  ST r5,r11,0
  LD r15,r12,3
  MOVI r1,1
  RET

mth_cosine_zero:
  ST r0,r11,0
  LD r15,r12,3
  MOVI r1,1
  RET
