.org 0
JMP @mth_pf_test_main
.include "../nan16/mth_preflight.nas"
.include "../../nan0s/src/nannet/math_registry.nas"

mth_pf_test_main:
  MOVI r13,1
  MOVI r12,@mth_pf_scratch

  ; Register vector leaves and bind their NaN16 payloads.
  MOVI r1,123
  MOVI r2,4
  MOVI r3,1
  MOVI r4,0x1234
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_1
  MOVI r1,234
  MOVI r2,4
  MOVI r3,1
  MOVI r4,0x5678
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_2
  MOVI r1,123
  MOVI r2,@mth_pf_left_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_pf_fail_3
  MOVI r1,234
  MOVI r2,@mth_pf_right_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_pf_fail_4

  ; The existing registry owns the operator identity and its typed contract.
  MOVI r1,77
  MOVI r2,4
  MOVI r3,4
  MOVI r4,3
  MOVI r5,900
  CALL @nmr_op_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_5
  MOVI r1,77
  MOVI r2,1
  MOVI r3,1
  MOVI r4,0
  MOVI r5,1
  MOVI r6,1
  MOVI r7,1
  MOVI r8,2
  CALL @nmr_contract_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_6

  ; Resolve through the real object/operator registry; r6 is its contract.
  MOVI r1,123
  MOVI r2,234
  MOVI r3,77
  CALL @nmr_resolve
  CMP r3,r0
  JMP[NE] @mth_pf_fail_7
  MOVI r7,3
  CMP r1,r7
  JMP[NE] @mth_pf_fail_8
  MOVI r7,900
  CMP r2,r7
  JMP[NE] @mth_pf_fail_9
  CMP r6,r0
  JMP[EQ] @mth_pf_fail
  ST r6,r0,@mth_pf_contract
  ST r4,r0,@mth_pf_left_data
  ST r5,r0,@mth_pf_right_data

  ; VECTOR[3] dot VECTOR[3] -> SCALAR passes native preflight.
  MOVI r6,@mth_pf_left
  LD r7,r0,@mth_pf_left_data
  ST r7,r6,5
  MOVI r6,@mth_pf_right
  LD r7,r0,@mth_pf_right_data
  ST r7,r6,5
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,1
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  CMP r1,r0
  JMP[NE] @mth_pf_fail_10
  LD r6,r5,0
  MOVI r7,3
  CMP r6,r7
  JMP[NE] @mth_pf_fail_11
  LD r6,r5,1
  CMP r6,r0
  JMP[NE] @mth_pf_fail_12
  LD r6,r5,4
  MOVI r7,1
  CMP r6,r7
  JMP[NE] @mth_pf_fail_13
  LD r6,r5,5
  MOVI r7,@mth_pf_output_payload
  CMP r6,r7
  JMP[NE] @mth_pf_fail_14

  ; Dimension mismatch rejects and leaves output metadata untouched.
  MOVI r6,@mth_pf_right
  MOVI r7,2
  ST r7,r6,2
  MOVI r6,@mth_pf_output
  MOVI r7,99
  ST r7,r6,0
  ST r7,r6,1
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,1
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  MOVI r7,6
  CMP r1,r7
  JMP[NE] @mth_pf_fail_15
  LD r6,r5,0
  MOVI r7,99
  CMP r6,r7
  JMP[NE] @mth_pf_fail_16

  ; Restore shape. Missing capability is a resource rejection, not NULL.
  MOVI r6,@mth_pf_right
  MOVI r7,3
  ST r7,r6,2
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,0
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  MOVI r7,7
  CMP r1,r7
  JMP[NE] @mth_pf_fail_17

  ; Register scalar division contract with an explicit nonzero precondition.
  MOVI r1,9
  MOVI r2,3
  MOVI r3,1
  MOVI r4,0x2222
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_18
  MOVI r1,10
  MOVI r2,3
  MOVI r3,1
  MOVI r4,0x3333
  CALL @nmr_obj_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_19
  MOVI r1,9
  MOVI r2,@mth_pf_scalar_left_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_pf_fail_20
  MOVI r1,10
  MOVI r2,@mth_pf_scalar_right_payload
  CALL @nmr_obj_bind_data
  CMP r3,r0
  JMP[NE] @mth_pf_fail_21
  MOVI r1,78
  MOVI r2,3
  MOVI r3,3
  MOVI r4,3
  MOVI r5,901
  CALL @nmr_op_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_22
  MOVI r1,78
  MOVI r2,0
  MOVI r3,0
  MOVI r4,0
  MOVI r5,1
  MOVI r6,4
  MOVI r7,1
  MOVI r8,1
  CALL @nmr_contract_put
  CMP r3,r0
  JMP[NE] @mth_pf_fail_23
  MOVI r1,9
  MOVI r2,10
  MOVI r3,78
  CALL @nmr_resolve
  CMP r3,r0
  JMP[NE] @mth_pf_fail_24
  ST r6,r0,@mth_pf_scalar_contract
  ST r4,r0,@mth_pf_scalar_left_data
  ST r5,r0,@mth_pf_scalar_right_data

  ; Zero divisor fails the contract precondition without publishing output.
  MOVI r6,@mth_pf_scalar_right_payload
  ST r0,r6,0
  MOVI r6,@mth_pf_scalar_left
  LD r7,r0,@mth_pf_scalar_left_data
  ST r7,r6,5
  MOVI r6,@mth_pf_scalar_right
  LD r7,r0,@mth_pf_scalar_right_data
  ST r7,r6,5
  MOVI r6,@mth_pf_scalar_output
  MOVI r7,55
  ST r7,r6,0
  MOVI r1,@mth_pf_scalar_left
  MOVI r2,@mth_pf_scalar_right
  LD r3,r0,@mth_pf_scalar_contract
  MOVI r4,1
  MOVI r5,@mth_pf_scalar_output
  CALL @mth_preflight
  MOVI r7,8
  CMP r1,r7
  JMP[NE] @mth_pf_fail_25
  LD r6,r5,0
  MOVI r7,55
  CMP r6,r7
  JMP[NE] @mth_pf_fail_26

  ; A nonzero divisor passes the same registered contract.
  MOVI r6,@mth_pf_scalar_right_payload
  MOVI r7,5
  ST r7,r6,0
  MOVI r1,@mth_pf_scalar_left
  MOVI r2,@mth_pf_scalar_right
  LD r3,r0,@mth_pf_scalar_contract
  MOVI r4,1
  MOVI r5,@mth_pf_scalar_output
  CALL @mth_preflight
  CMP r1,r0
  JMP[NE] @mth_pf_fail_27
  LD r6,r5,0
  MOVI r7,3
  CMP r6,r7
  JMP[NE] @mth_pf_fail_28

  ; Output capacity is a distinct rejection and remains atomic.
  MOVI r6,@mth_pf_output
  MOVI r7,0
  ST r7,r6,6
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,1
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  MOVI r7,9
  CMP r1,r7
  JMP[NE] @mth_pf_fail_29

  ; Partial overlap must reject even when the payload base pointers differ.
  ; Give the output its full required capacity so capacity validation passes
  ; and the range-overlap predicate is the check under test.
  MOVI r6,@mth_pf_output
  MOVI r7,55
  ST r7,r6,0
  MOVI r7,@mth_pf_left_payload
  ADDI r7,r7,1
  ST r7,r6,5
  MOVI r7,3
  ST r7,r6,6
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,1
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  MOVI r7,8
  CMP r1,r7
  JMP[NE] @mth_pf_fail_30
  LD r6,r5,0
  MOVI r7,55
  CMP r6,r7
  JMP[NE] @mth_pf_fail_31

  ; The same partial-overlap rule applies to the right operand.
  MOVI r6,@mth_pf_output
  MOVI r7,55
  ST r7,r6,0
  MOVI r7,@mth_pf_right_payload
  ADDI r7,r7,1
  ST r7,r6,5
  MOVI r7,3
  ST r7,r6,6
  MOVI r1,@mth_pf_left
  MOVI r2,@mth_pf_right
  LD r3,r0,@mth_pf_contract
  MOVI r4,1
  MOVI r5,@mth_pf_output
  CALL @mth_preflight
  MOVI r7,8
  CMP r1,r7
  JMP[NE] @mth_pf_fail_32
  MOVI r6,@mth_pf_output
  LD r7,r6,0
  MOVI r8,55
  CMP r7,r8
  JMP[NE] @mth_pf_fail_33

  MOVI r1,@mth_pf_pass
  CALL @mth_pf_puts
  HALT

mth_pf_fail_1:
  MOVI r1,1
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_2:
  MOVI r1,2
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_3:
  MOVI r1,3
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_4:
  MOVI r1,4
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_5:
  MOVI r1,5
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_6:
  MOVI r1,6
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_7:
  MOVI r1,7
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_8:
  MOVI r1,8
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_9:
  MOVI r1,9
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_10:
  MOVI r1,10
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_11:
  MOVI r1,11
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_12:
  MOVI r1,12
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_13:
  MOVI r1,13
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_14:
  MOVI r1,14
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_15:
  MOVI r1,15
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_16:
  MOVI r1,16
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_17:
  MOVI r1,17
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_18:
  MOVI r1,18
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_19:
  MOVI r1,19
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_20:
  MOVI r1,20
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_21:
  MOVI r1,21
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_22:
  MOVI r1,22
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_23:
  MOVI r1,23
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_24:
  MOVI r1,24
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_25:
  MOVI r1,25
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_26:
  MOVI r1,26
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_27:
  MOVI r1,27
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_28:
  MOVI r1,28
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_29:
  MOVI r1,29
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_30:
  MOVI r1,30
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_31:
  MOVI r1,31
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_32:
  MOVI r1,32
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail_33:
  MOVI r1,33
  ST r1,r12,38
  JMP @mth_pf_fail
mth_pf_fail:
  MOVI r1,@mth_pf_bad
  CALL @mth_pf_puts
  HALT

mth_pf_puts:
  MOVI r4,255
mth_pf_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_pf_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_pf_puts_byte
  ADD r1,r1,r13
  JMP @mth_pf_puts_word

.org 30000
mth_pf_left:
.word 4,1,3,1,1,@mth_pf_left_payload,3,1
mth_pf_right:
.word 4,1,3,1,1,@mth_pf_right_payload,3,1
mth_pf_output:
.word 0,0,0,0,0,@mth_pf_output_payload,1,0
mth_pf_scalar_left:
.word 3,0,1,1,1,@mth_pf_scalar_left_payload,1,1
mth_pf_scalar_right:
.word 3,0,1,1,1,@mth_pf_scalar_right_payload,1,1
mth_pf_scalar_output:
.word 0,0,0,0,0,@mth_pf_scalar_output_payload,1,0

.org 30256
mth_pf_left_payload:
.word 1,2,3
mth_pf_right_payload:
.word 4,5,6
mth_pf_output_payload:
.word 0
mth_pf_scalar_left_payload:
.word 9
mth_pf_scalar_right_payload:
.word 0
mth_pf_scalar_output_payload:
.word 0
mth_pf_scratch:
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
mth_pf_contract:
.word 0
mth_pf_scalar_contract:
.word 0
mth_pf_left_data:
.word 0
mth_pf_right_data:
.word 0
mth_pf_scalar_left_data:
.word 0
mth_pf_scalar_right_data:
.word 0
mth_pf_pass:
.str "MTH TYPED PREFLIGHT PASS\n"
mth_pf_bad:
.str "MTH TYPED PREFLIGHT FAIL\n"
