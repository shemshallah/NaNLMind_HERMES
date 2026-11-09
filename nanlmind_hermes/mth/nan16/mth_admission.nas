; Composed NaN16 MTH admission gate.
; The rule/candidate gate runs first; typed object applicability runs second.
; No numerical kernel is called here.
;
; ABI:
;   r1 candidate[22], r2 authoritative_rule[17]
;   r3 left_object[8], r4 right_object[8], r5 operator_contract[10]
;   r6 available_capability_bits, r7 caller_reserved_output[8]
;   r12 caller-owned scratch[48], r13=1
; Return r1:
;   0       both gates passed
;   16 + s  rule/candidate gate denied (s=1..9)
;   32 + s  typed applicability gate denied (s=1..9)
;
; Scratch [40..47] are wrapper-private; typed preflight owns [0..38].
mth_admit:
  ST r15,r12,47
  ST r1,r12,40
  ST r2,r12,41
  ST r3,r12,42
  ST r4,r12,43
  ST r5,r12,44
  ST r6,r12,45
  ST r7,r12,46

  ; Bind first gate's explicit result word to scratch[39].
  MOVI r8,39
  ADD r3,r12,r8
  LD r1,r12,40
  LD r2,r12,41
  CALL @mth_preflight_gate
  CMP r1,r0
  JMP[NE] @mth_admit_rule_denied

  ; Only after exact rule snapshot admission do typed applicability checks.
  LD r1,r12,42
  LD r2,r12,43
  LD r3,r12,44
  LD r4,r12,45
  LD r5,r12,46
  CALL @mth_preflight
  CMP r1,r0
  JMP[NE] @mth_admit_typed_denied

  MOVI r1,0
  LD r15,r12,47
  RET

mth_admit_rule_denied:
  MOVI r8,16
  ADD r1,r1,r8
  LD r15,r12,47
  RET

mth_admit_typed_denied:
  MOVI r8,32
  ADD r1,r1,r8
  LD r15,r12,47
  RET
