; MTH exact C256 ring Laplacian — NaN16 float32 only.
; The ring is the undirected cycle 0-1-...-255-0 over all 256 positions.
;
; ABI:
;   r1 = input address, 256 float32 payloads
;   r2 = output address, 256 float32 payloads
;   returns r1=1 on success
;   r13 = 1
;   clobbers r3-r8; requires r13=1; caller supplies non-overlapping spans.
;
; (Lx)[i] = 2*x[i] - x[(i-1) mod 256] - x[(i+1) mod 256].
; No rounding beyond the native float32 operations; no host math.
; Leaf routine: no CALLs, no static scratch, no shared return slot.

mth_ring256_laplacian_f32:
  ; Boundary vertex 0: 2*x0 - x1 - x255.
  LD r3,r1,0
  ADD r4,r1,r13
  LD r5,r4,0
  MOVI r6,255
  ADD r4,r1,r6
  LD r7,r4,0
  FADD r3,r3,r3
  FSUB r3,r3,r5
  FSUB r3,r3,r7
  ST r3,r2,0

  ; Interior vertices 1..254.
  MOVI r6,1
mth_ring256_interior:
  MOVI r8,254
  CMP r6,r8
  JMP[GT] @mth_ring256_boundary_last

  ADD r4,r1,r6
  LD r3,r4,0
  FADD r3,r3,r3
  SUB r8,r6,r13
  ADD r5,r1,r8
  LD r5,r5,0
  FSUB r3,r3,r5
  ADD r8,r6,r13
  ADD r5,r1,r8
  LD r5,r5,0
  FSUB r3,r3,r5

  ADD r4,r2,r6
  ST r3,r4,0
  ADD r6,r6,r13
  JMP @mth_ring256_interior

mth_ring256_boundary_last:
  ; Boundary vertex 255: 2*x255 - x254 - x0.
  MOVI r6,255
  ADD r4,r1,r6
  LD r3,r4,0
  FADD r3,r3,r3
  MOVI r8,254
  ADD r5,r1,r8
  LD r5,r5,0
  FSUB r3,r3,r5
  LD r5,r1,0
  FSUB r3,r3,r5
  ADD r4,r2,r6
  ST r3,r4,0

  MOVI r1,1
  RET
