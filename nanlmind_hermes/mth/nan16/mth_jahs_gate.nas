; MTH JAHS handoff seal gate — native NaN16 only.
; JAHS = "Just A Handoff Seal".
; ABI: r1 = seal word (the 32-bit NaN16 payload of the canonical 64-bit
;       data word 0x000000004A414853, i.e. ASCII "JAHS" in bits 31..0).
; Returns r1=1 iff the seal is exactly 0x4A414853; otherwise r1=0.
; Leaf routine: no CALL, no host math. Clobbers r1, r2, and the flags.

mth_jahs_gate:
  MOVI r2,0x4853
  MOVHI r2,0x4A41
  CMP r1,r2
  MOVI r1,0
  RET[NE]
  MOVI r1,1
  RET
