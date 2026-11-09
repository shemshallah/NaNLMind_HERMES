.org 0
JMP @mth_link_pf_main
.include "../nan16/mth_preflight_gate.nas"
.include "../nan16/mth_preflight.nas"
.include "../nan16/mth_admission.nas"
.include "../nan16/mth_cosine256_f32.nas"
.include "../nan16/mth_vector_add256_f32.nas"
.include "../nan16/mth_vector_sub256_f32.nas"
.include "../nan16/mth_execute.nas"
.include "../../nan0s/src/nannet/route_address.nas"
.include "../../nan0s/src/nannet/object_key_decode.nas"
.include "../../nan0s/src/nannet/math_registry.nas"
.include "../../nan0s/src/nannet/route_object_resolve.nas"
.include "../../nan0s/src/nannet/vector_ops.nas"
.include "../../nan0s/src/nannet/mc0db_scratch.nas"
.include "../../nan0s/src/nannet/math_expression.nas"

mth_link_pf_main:
  MOVI r13,1
  MOVI r12,@mth_link_pf_scratch

  ; Register two leaf objects at distinct logical HW0 points.
  MOVI r1,123
  MOVI r2,4
  MOVI r3,1
  MOVI r4,0x1234
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail
  MOVI r1,234
  MOVI r2,4
  MOVI r3,2
  MOVI r4,0x5678
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail
  MOVI r1,123
  MOVI r2,@mth_link_pf_left_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail
  MOVI r1,234
  MOVI r2,@mth_link_pf_right_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail

  ; Existing shape metadata is used by the legacy route-expression parser.
  MOVI r1,123
  MOVI r2,1
  MOVI r3,256
  MOVI r4,1
  MOVI r5,256
  CALL @nmr_shape_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_shape_fail
  MOVI r1,234
  MOVI r2,1
  MOVI r3,256
  MOVI r4,1
  MOVI r5,256
  CALL @nmr_shape_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail

  ; Register the operator once, then attach its immutable typed contract.
  MOVI r1,77
  MOVI r2,4
  MOVI r3,4
  MOVI r4,3
  MOVI r5,900
  CALL @nmr_op_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail
  MOVI r1,77
  MOVI r2,1
  MOVI r3,1
  MOVI r4,0
  MOVI r5,2
  MOVI r6,1
  MOVI r7,1
  MOVI r8,0
  CALL @nmr_contract_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail

  ; The literal ::op: expression is parsed and resolved by the canonical
  ; NaN0S linker. Its r10 is the registered 10-word contract descriptor.
  MOVI r1,@mth_link_pf_expression
  SHL r1,r1,2
  CALL @nmr_expr_resolve
  ; Preserve the resolver's register ABI: diagnostic output uses/clobbers
  ; argument registers, so do not print between resolve and its assertions.
  CMP r8,r0
  JMP[NE] @mth_link_pf_resolve_fail
  MOVI r6,123
  CMP r1,r6
  JMP[NE] @mth_link_pf_fail
  MOVI r6,234
  CMP r2,r6
  JMP[NE] @mth_link_pf_fail
  MOVI r6,77
  CMP r3,r6
  JMP[NE] @mth_link_pf_fail
  ST r3,r0,@mth_link_pf_operator_id
  MOVI r6,3
  CMP r4,r6
  JMP[NE] @mth_link_pf_fail
  MOVI r6,900
  CMP r5,r6
  JMP[NE] @mth_link_pf_fail
  ST r5,r0,@mth_link_pf_program_id
  MOVI r6,256
  CMP r9,r6
  JMP[NE] @mth_link_pf_fail
  CMP r10,r0
  JMP[EQ] @mth_link_pf_fail
  ST r10,r0,@mth_link_pf_contract

  ; mth_admit performs both gates; pass the exact registry contract pointer.
  MOVI r12,@mth_link_pf_scratch

  ; Reuse the exact registry-owned contract pointer through composed admission.
  ; This binds the authoritative rule snapshot first, then concrete typed leaves.
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_admit
  CMP r1,r0
  JMP[EQ] @mth_link_pf_admit_ok
  MOVI r8,32
  CMP r1,r8
  JMP[GE] @mth_link_pf_admit_typed_fail
  MOVI r1,@mth_link_pf_admit_rule_bad
  CALL @mth_link_pf_puts
  HALT
mth_link_pf_admit_typed_fail:
  MOVI r1,@mth_link_pf_admit_typed_bad
  CALL @mth_link_pf_puts
  HALT
mth_link_pf_admit_ok:

  ; A stale rule epoch is rejected before typed applicability.
  MOVI r8,@mth_link_pf_candidate
  MOVI r9,6
  ST r9,r8,9
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_admit
  MOVI r9,18
  CMP r1,r9
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_candidate
  MOVI r9,7
  ST r9,r8,9

  ; Concrete dimension mismatch is rejected by the second gate and leaves
  ; the caller's output descriptor untouched.
  MOVI r8,@mth_link_pf_right
  MOVI r9,2
  ST r9,r8,2
  MOVI r8,@mth_link_pf_output
  MOVI r9,55
  ST r9,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_admit
  MOVI r9,38
  CMP r1,r9
  JMP[NE] @mth_link_pf_fail
  ; Restore the admitted 256-vector shape and execute the allowlisted dot kernel.
  MOVI r8,@mth_link_pf_right
  MOVI r9,256
  ST r9,r8,2
  MOVI r8,@mth_link_pf_output_payload
  MOVI r9,0
  ST r9,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  LD r8,r0,@mth_link_pf_program_id
  LD r9,r0,@mth_link_pf_operator_id
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_execute
  CMP r1,r0
  JMP[NE] @mth_link_pf_dispatch_status_fail
  MOVI r8,@mth_link_pf_output_payload
  LD r9,r8,0
  MOVI r10,0x4200
  SHL r10,r10,16
  CMP r9,r10
  JMP[NE] @mth_link_pf_fail
  ; Preserve the exact dot contract pointer for the negative-path checks.
  LD r10,r0,@mth_link_pf_contract
  ST r10,r0,@mth_link_pf_dot_contract

  ; A stale epoch must stop before the numerical kernel writes its output.
  MOVI r8,@mth_link_pf_candidate
  MOVI r9,6
  ST r9,r8,9
  MOVI r8,@mth_link_pf_output_payload
  MOVI r9,0x1234
  SHL r9,r9,16
  MOVI r10,0x5678
  ADD r9,r9,r10
  ST r9,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  LD r8,r0,@mth_link_pf_program_id
  LD r9,r0,@mth_link_pf_operator_id
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_execute
  MOVI r9,18
  CMP r1,r9
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_output_payload
  LD r9,r8,0
  MOVI r10,0x1234
  SHL r10,r10,16
  MOVI r11,0x5678
  ADD r10,r10,r11
  CMP r9,r10
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_candidate
  MOVI r9,7
  ST r9,r8,9

  ; A mismatched operator/program pair is rejected before admission.
  ; Keep the admitted rule epoch intact; pass the wrong linker operator ID.
  MOVI r8,@mth_link_pf_output_payload
  MOVI r9,0x1234
  SHL r9,r9,16
  MOVI r10,0x5678
  ADD r9,r9,r10
  ST r9,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  LD r8,r0,@mth_link_pf_program_id
  MOVI r9,78
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_execute
  MOVI r9,64
  CMP r1,r9
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_output_payload
  LD r9,r8,0
  MOVI r10,0x1234
  SHL r10,r10,16
  MOVI r11,0x5678
  ADD r10,r10,r11
  CMP r9,r10
  JMP[NE] @mth_link_pf_fail
  ; Unknown program IDs are rejected before preflight or output publication.
  MOVI r8,@mth_link_pf_output_payload
  MOVI r9,0x1234
  SHL r9,r9,16
  MOVI r10,0x5678
  ADD r9,r9,r10
  ST r9,r8,0
  MOVI r8,@mth_link_pf_output
  MOVI r9,55
  ST r9,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  MOVI r8,902
  MOVI r9,77
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_execute
  MOVI r9,64
  CMP r1,r9
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_output_payload
  LD r9,r8,0
  MOVI r10,0x1234
  SHL r10,r10,16
  MOVI r11,0x5678
  ADD r10,r10,r11
  CMP r9,r10
  JMP[NE] @mth_link_pf_fail
  MOVI r8,@mth_link_pf_output
  LD r9,r8,0
  MOVI r10,55
  CMP r9,r10
  JMP[NE] @mth_link_pf_fail

  ; Register the second allowlisted operator: cosine uses program 901.
  MOVI r1,78
  MOVI r2,4
  MOVI r3,4
  MOVI r4,3
  MOVI r5,901
  CALL @nmr_op_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail
  MOVI r1,78
  MOVI r2,1
  MOVI r3,1
  MOVI r4,0
  MOVI r5,2
  MOVI r6,1
  MOVI r7,1
  MOVI r8,0
  CALL @nmr_contract_put
  CMP r3,r0
  JMP[NE] @mth_link_pf_fail

  ; Make the two operands equal, resolve ::op:78, and execute native cosine.
  MOVI r8,@mth_link_pf_right_payload
  MOVI r9,0x3F80
  SHL r9,r9,16
  ST r9,r8,0
  MOVI r9,0x4000
  SHL r9,r9,16
  ST r9,r8,1
  MOVI r9,0x4040
  SHL r9,r9,16
  ST r9,r8,2
  MOVI r1,@mth_link_pf_expression_cosine
  SHL r1,r1,2
  CALL @nmr_expr_resolve
  CMP r8,r0
  JMP[NE] @mth_link_pf_resolve_fail
  MOVI r6,78
  CMP r3,r6
  JMP[NE] @mth_link_pf_fail
  ST r3,r0,@mth_link_pf_operator_id
  MOVI r6,901
  CMP r5,r6
  JMP[NE] @mth_link_pf_fail
  MOVI r6,256
  CMP r9,r6
  JMP[NE] @mth_link_pf_fail
  CMP r10,r0
  JMP[EQ] @mth_link_pf_fail
  ST r10,r0,@mth_link_pf_contract
  ST r5,r0,@mth_link_pf_program_id
  MOVI r8,@mth_link_pf_output_payload
  ST r0,r8,0
  MOVI r1,@mth_link_pf_candidate
  MOVI r2,@mth_link_pf_rule
  MOVI r3,@mth_link_pf_left
  MOVI r4,@mth_link_pf_right
  LD r5,r0,@mth_link_pf_contract
  MOVI r6,1
  MOVI r7,@mth_link_pf_output
  LD r8,r0,@mth_link_pf_program_id
  LD r9,r0,@mth_link_pf_operator_id
  MOVI r12,@mth_link_pf_scratch
  CALL @mth_execute
  CMP r1,r0
  JMP[NE] @mth_link_pf_dispatch_status_fail
  MOVI r8,@mth_link_pf_output_payload
  LD r9,r8,0
  ; Float32 cosine of equal nonzero vectors is 1.0 within a tight
  ; 0.001 ULP-scale envelope; sqrt/divide need not round to exact 1.0.
  MOVI r10,0x3F7F
  SHL r10,r10,16
  MOVI r11,0xF000
  ADD r10,r10,r11
  CMP r9,r10
  JMP[LT] @mth_link_pf_fail
  MOVI r10,0x3F80
  SHL r10,r10,16
  MOVI r11,0x1000
  ADD r10,r10,r11
  CMP r9,r10
  JMP[GT] @mth_link_pf_fail

  ; Restore dot operator/rule/contract for the negative-path checks below.
  MOVI r9,77
  ST r9,r0,@mth_link_pf_operator_id
  LD r10,r0,@mth_link_pf_dot_contract
  ST r10,r0,@mth_link_pf_contract
  MOVI r8,900
  ST r8,r0,@mth_link_pf_program_id

  MOVI r1,@mth_link_pf_pass
  CALL @mth_link_pf_puts
  MOVI r1,@mth_link_pf_admit_pass
  CALL @mth_link_pf_puts
  MOVI r1,@mth_link_pf_dispatch_pass
  CALL @mth_link_pf_puts
  HALT

mth_link_pf_resolve_fail:
  MOVI r1,@mth_link_pf_resolve_bad
  CALL @mth_link_pf_puts
  HALT

mth_link_pf_shape_fail:
  MOVI r1,@mth_link_pf_shape_bad
  CALL @mth_link_pf_puts
  HALT

mth_link_pf_dispatch_status_fail:
  MOVI r1,@mth_link_pf_dispatch_status_bad
  CALL @mth_link_pf_puts
  HALT

mth_link_pf_fail:
  MOVI r1,@mth_link_pf_bad
  CALL @mth_link_pf_puts
  HALT

mth_link_pf_puts:
  MOVI r4,255
mth_link_pf_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_link_pf_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_link_pf_puts_byte
  ADD r1,r1,r13
  JMP @mth_link_pf_puts_word

.org 30000
mth_link_pf_left:
.word 4,1,256,1,2,@mth_link_pf_left_payload,256,1
mth_link_pf_right:
.word 4,1,256,1,2,@mth_link_pf_right_payload,256,1
mth_link_pf_output:
.word 0,0,0,0,0,@mth_link_pf_output_payload,1,0
mth_link_pf_candidate:
.word 1,7,4,4,3,2,2,256,256,7,1,1,2,1,100,101,102,103,104,105,106,107
mth_link_pf_rule:
.word 1,7,4,4,3,2,1,1,1,100,101,102,103,104,105,106,107

.org 30256
mth_link_pf_left_payload:
.word 1065353216,1073741824,1077936128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
mth_link_pf_right_payload:
.word 1082130432,1084227584,1086324736,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
mth_link_pf_output_payload:
.word 0
mth_link_pf_scratch:
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
.word 0
mth_link_pf_contract:
.word 0
mth_link_pf_dot_contract:
.word 0
mth_link_pf_program_id:
.word 0
mth_link_pf_operator_id:
.word 0
mth_link_pf_expression:
.str "nan://GM0:SC0:MC0:BS0:HW0::HW0:1::vector123::op:77::nan://GM0:SC0:MC0:BS0:HW0::HW0:2::vector234"
mth_link_pf_expression_cosine:
.str "nan://GM0:SC0:MC0:BS0:HW0::HW0:1::vector123::op:78::nan://GM0:SC0:MC0:BS0:HW0::HW0:2::vector234"
mth_link_pf_pass:
.str "MTH LINKER PREFLIGHT PASS\n"
mth_link_pf_admit_pass:
.str "MTH LINKER COMPOSED ADMISSION PASS\n"
mth_link_pf_admit_rule_bad:
.str "MTH LINKER RULE ADMISSION DENIED\n"
mth_link_pf_admit_typed_bad:
.str "MTH LINKER TYPED ADMISSION DENIED\n"
mth_link_pf_dispatch_pass:
.str "MTH NATIVE DISPATCH PASS\n"
mth_link_pf_bad:
.str "MTH LINKER PREFLIGHT FAIL\n"
mth_link_pf_dispatch_status_bad:
.str "MTH NATIVE DISPATCH STATUS FAIL\n"
mth_link_pf_shape_bad:
.str "MTH SHAPE REGISTRATION FAIL\n"
mth_link_pf_resolve_bad:
.str "MTH LINK RESOLVE STATUS FAIL\n"
