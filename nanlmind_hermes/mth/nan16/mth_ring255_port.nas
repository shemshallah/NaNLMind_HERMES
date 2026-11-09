; MTH 255-head + NANNET-port readout assembly — NaN16 float32 only.
; Maps a 255-head state and one NANNET port scalar into 256 ring positions.
;
; ABI:
;   r1 = head state, 255 float32 payloads
;   r2 = NANNET port scalar, one float32 payload
;   r3 = ring readout, 256 float32 payloads
;   r13 = 1
;   returns r1=1 success
;   clobbers r4-r8
;
; Layout:
;   ring[0..254] = head[0..254]
;   ring[255]    = port
; The port is a position, not a head. No coupling arithmetic is performed here;
; graph/port coupling is a separate versioned operator.

mth_ring255_port_assemble_f32:
  ADD r4,r1,r0
  ADD r5,r3,r0
  MOVI r6,255
mth_ring255_port_copy_heads:
  LD r7,r4,0
  ST r7,r5,0
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_ring255_port_copy_heads

  LD r7,r2,0
  ST r7,r5,0
  MOVI r1,1
  RET
