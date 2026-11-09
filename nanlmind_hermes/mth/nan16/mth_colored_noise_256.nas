; MTH 256-element deterministic first-order colored noise — NaN16 only.
;
; ABI:
;   r1 = seed, one 32-bit integer payload (all 2^32 values accepted)
;   r2 = out, 256 float32 payloads
;   r3 = alpha, one float32 payload in [0,1)
;   r13 = 1
;   returns r1=1
;   clobbers r2-r11
;
; Sequence (all arithmetic is NaN16 integer or float32; no host math):
;   state_0 = seed
;   out[0]  = U(state_0)
;   for i = 1..255:
;     state_i = (1664525 * state_{i-1} + 1013904223) mod 2^32
;     out[i]  = alpha*out[i-1] + (1-alpha)*U(state_i)
;
; U(s) maps the LCG state to an exact float32 in [-1,1):
;   u24   = s >> 8                         ; logical right shift, 0..2^24-1
;   s24   = int32(u24 - 2^23)              ; [-2^23, 2^23-1], exactly 24 bits
;   U(s)  = CVTIF(s24) * 2^-23             ; exact conversion and exact power
;
; The LCG is the Numerical Recipes 32-bit LCG. MUL keeps the low 32 bits and
; ADD wraps modulo 2^32, so the state transition is exactly the document's.
; The float combination evaluates alpha*z first, then (1-alpha)*u, then one
; FADD. There is no fused multiply-add, reassociation, clamping, or host math.
; The written order is i=0,1,...,255, with out[0] written before the loop.
;
; Output bounds: U is in [-1,1), so out[0] is in [-1,1). If out[i-1] is in
; [-1,1], the exact convex combination with alpha in [0,1), U in [-1,1) is in
; [-1,1); round-to-nearest cannot leave that interval. alpha=0 reduces to the
; white sequence U(state_i); alpha close to 1 retains strong lag-1 correlation.

mth_colored_noise_256:
  ADD r4,r1,r0              ; state = seed
  ADD r1,r2,r0              ; out cursor = out
  LD r5,r3,0                ; alpha
  MOVI r6,0
  MOVHI r6,0x3F80           ; 1.0
  FSUB r6,r6,r5             ; beta = 1.0 - alpha

  MOVI r9,0x660D
  MOVHI r9,0x0019           ; 1664525 = 0x0019660D
  MOVI r2,0xF35F
  MOVHI r2,0x3C6E           ; 1013904223 = 0x3C6EF35F
  MOVI r10,0
  MOVHI r10,0x0080          ; 2^23 = 0x00800000
  MOVI r3,0
  MOVHI r3,0x3400           ; 2^-23 = 0x34000000

  ; out[0] = U(seed)
  SHR r8,r4,8
  SUB r8,r8,r10
  CVTIF r8,r8
  FMUL r8,r8,r3
  ADD r7,r8,r0
  ST r7,r1,0
  ADD r1,r1,r13
  MOVI r11,255

mth_colored_noise_loop:
  MUL r4,r4,r9              ; low 32 bits of state*1664525
  ADD r4,r4,r2              ; + 1013904223, modulo 2^32
  SHR r8,r4,8
  SUB r8,r8,r10
  CVTIF r8,r8
  FMUL r8,r8,r3             ; u = U(state)
  FMUL r7,r7,r5             ; alpha*z
  FMUL r8,r8,r6             ; (1-alpha)*u
  FADD r7,r7,r8             ; z = alpha*z + (1-alpha)*u
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r11,r11,r13
  CMP r11,r0
  JMP[GT] @mth_colored_noise_loop

  MOVI r1,1
  RET
