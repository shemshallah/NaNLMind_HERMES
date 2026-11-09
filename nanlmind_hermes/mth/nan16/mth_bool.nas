; MTH generic two-input Boolean function, entirely NaN16.
; Inputs: r1 = truth-table class 0..15, r5 = operand A, r6 = operand B.
; Output: r4 = BOOL.class(A,B).
; Clobbers r7,r8. Uses the established NaN16 runtime BOOL-word patch pattern.
; Caller must provide a writable/executable NaN16 code region for the slot.

mth_bool2:
  MOVI r7,6
  MOVHI r7,0x0045
  MOVI r8,@mth_bool2_slot
  STW r7,r8,r1
mth_bool2_slot:
  BOOL.0 r4,r5,r6
  RET
