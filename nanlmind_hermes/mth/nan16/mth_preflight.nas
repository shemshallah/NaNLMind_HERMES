; MTH typed-object preflight gate — NaN16 only.
; The operator contract is resolved upstream by the single canonical registry.
; This module validates concrete leaf descriptors against that contract; it
; does not create or duplicate a function/operator registry.
;
; Object descriptor (8 words):
;   [0] kind [1] rank [2] dim0 [3] dim1 [4] domain
;   [5] payload_word_address [6] payload_words [7] flags
; Contract descriptor (10 words):
;   [0] left_kind [1] right_kind [2] output_kind
;   [3] left_rank [4] right_rank [5] output_rank
;   [6] domain [7] shape_rule [8] required_capability_bits [9] precondition_bits
; Output descriptor uses the object layout; its payload address/capacity at
; [5]/[6] must be reserved by the caller before preflight.
;
; Shape rules: 0 equal-shape, 1 vector-dot, 2 matrix-vector, 3 matrix-matrix,
;              4 scalar-scalar.
; Preconditions: bit0 right first word nonzero; bit1 output does not alias
; either input; bit2 both input flags bit0 (finite); bit3 nonempty dimensions.
;
; ABI: r1=left descriptor, r2=right descriptor, r3=contract,
;      r4=available capability bits, r5=output descriptor,
;      r12=caller-owned 39-word scratch, r13=1.
; Returns r1=0 success; rejection codes:
;   1 descriptor, 2 missing payload, 3 kind, 4 rank, 5 domain,
;   6 shape, 7 capability, 8 precondition, 9 capacity.
; Output descriptor is unchanged on every rejection.
;
; This validates applicability only. Kernel execution failures and
; mathematical undefinedness remain separate result states.

mth_preflight:
  CMP r1,r0
  JMP[EQ] @mth_pf_bad_descriptor
  CMP r2,r0
  JMP[EQ] @mth_pf_bad_descriptor
  CMP r3,r0
  JMP[EQ] @mth_pf_bad_descriptor
  CMP r5,r0
  JMP[EQ] @mth_pf_bad_descriptor
  CMP r12,r0
  JMP[EQ] @mth_pf_bad_descriptor
  ST r1,r12,0
  ST r2,r12,1
  ST r3,r12,2
  ST r5,r12,3
  ST r4,r12,4

  ; Contract fields remain separate from concrete descriptor fields.
  LD r6,r3,0
  ST r6,r12,10
  LD r6,r3,1
  ST r6,r12,11
  LD r6,r3,2
  ST r6,r12,12
  LD r6,r3,3
  ST r6,r12,13
  LD r6,r3,4
  ST r6,r12,14
  LD r6,r3,5
  ST r6,r12,15
  LD r6,r3,6
  ST r6,r12,19
  LD r6,r3,7
  ST r6,r12,8
  LD r6,r3,8
  ST r6,r12,20
  LD r6,r3,9
  ST r6,r12,9

  ; Preserve caller-reserved output storage without modifying its descriptor.
  LD r6,r5,5
  LD r7,r5,6
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_payload
  ST r6,r12,32
  ST r7,r12,18

  ; Left descriptor.
  LD r1,r12,0
  LD r6,r1,0
  LD r7,r1,1
  LD r8,r1,2
  LD r9,r1,3
  LD r10,r1,4
  LD r11,r1,5
  LD r3,r1,6
  LD r4,r1,7
  ST r6,r12,21
  ST r7,r12,22
  ST r8,r12,5
  ST r9,r12,6
  ST r10,r12,33
  ST r11,r12,16
  ST r3,r12,26
  ST r4,r12,7
  CMP r11,r0
  JMP[EQ] @mth_pf_bad_payload

  ; Right descriptor.
  LD r1,r12,1
  LD r6,r1,0
  LD r7,r1,1
  LD r8,r1,2
  LD r9,r1,3
  LD r10,r1,4
  LD r11,r1,5
  LD r3,r1,6
  LD r4,r1,7
  ST r6,r12,34
  ST r7,r12,35
  ST r8,r12,23
  ST r9,r12,24
  ST r10,r12,36
  ST r11,r12,17
  ST r3,r12,27
  ST r4,r12,25
  CMP r11,r0
  JMP[EQ] @mth_pf_bad_payload

  ; Exact kind, rank, and domain compatibility.
  LD r6,r12,21
  LD r7,r12,10
  CMP r6,r7
  JMP[NE] @mth_pf_bad_kind
  LD r6,r12,34
  LD r7,r12,11
  CMP r6,r7
  JMP[NE] @mth_pf_bad_kind
  LD r6,r12,22
  LD r7,r12,13
  CMP r6,r7
  JMP[NE] @mth_pf_bad_rank
  LD r6,r12,35
  LD r7,r12,14
  CMP r6,r7
  JMP[NE] @mth_pf_bad_rank
  LD r6,r12,33
  LD r7,r12,19
  CMP r6,r7
  JMP[NE] @mth_pf_bad_domain
  LD r6,r12,36
  LD r7,r12,19
  CMP r6,r7
  JMP[NE] @mth_pf_bad_domain

  ; Capability subset: (available & required) == required.
  LD r6,r12,4
  LD r7,r12,20
  BOOL.8 r8,r6,r7
  CMP r8,r7
  JMP[NE] @mth_pf_bad_capability

  ; Compute the element count for each rank (0, 1, or 2).
  LD r6,r12,22
  CMP r6,r0
  JMP[EQ] @mth_pf_left_scalar
  MOVI r7,1
  CMP r6,r7
  JMP[EQ] @mth_pf_left_vector
  MOVI r7,2
  CMP r6,r7
  JMP[EQ] @mth_pf_left_matrix
  JMP @mth_pf_bad_rank
mth_pf_left_scalar:
  MOVI r6,1
  ST r6,r12,37
  JMP @mth_pf_left_done
mth_pf_left_vector:
  LD r6,r12,5
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_shape
  ST r6,r12,37
  JMP @mth_pf_left_done
mth_pf_left_matrix:
  LD r6,r12,5
  LD r7,r12,6
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_shape
  CMP r7,r0
  JMP[EQ] @mth_pf_bad_shape
  MUL r6,r6,r7
  ST r6,r12,37
mth_pf_left_done:
  LD r6,r12,35
  CMP r6,r0
  JMP[EQ] @mth_pf_right_scalar
  MOVI r7,1
  CMP r6,r7
  JMP[EQ] @mth_pf_right_vector
  MOVI r7,2
  CMP r6,r7
  JMP[EQ] @mth_pf_right_matrix
  JMP @mth_pf_bad_rank
mth_pf_right_scalar:
  MOVI r6,1
  ST r6,r12,38
  JMP @mth_pf_right_done
mth_pf_right_vector:
  LD r6,r12,23
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_shape
  ST r6,r12,38
  JMP @mth_pf_right_done
mth_pf_right_matrix:
  LD r6,r12,23
  LD r7,r12,24
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_shape
  CMP r7,r0
  JMP[EQ] @mth_pf_bad_shape
  MUL r6,r6,r7
  ST r6,r12,38
mth_pf_right_done:
  LD r6,r12,26
  LD r7,r12,37
  CMP r6,r7
  JMP[LT] @mth_pf_bad_capacity
  LD r6,r12,27
  LD r7,r12,38
  CMP r6,r7
  JMP[LT] @mth_pf_bad_capacity

  ; Shape-rule dispatch.
  LD r6,r12,8
  CMP r6,r0
  JMP[EQ] @mth_pf_shape_equal
  MOVI r7,1
  CMP r6,r7
  JMP[EQ] @mth_pf_shape_dot
  MOVI r7,2
  CMP r6,r7
  JMP[EQ] @mth_pf_shape_matvec
  MOVI r7,3
  CMP r6,r7
  JMP[EQ] @mth_pf_shape_matmul
  MOVI r7,4
  CMP r6,r7
  JMP[EQ] @mth_pf_shape_scalar
  JMP @mth_pf_bad_shape

mth_pf_shape_equal:
  LD r6,r12,22
  LD r7,r12,35
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,5
  LD r7,r12,23
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,6
  LD r7,r12,24
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,22
  ST r6,r12,28
  LD r6,r12,5
  ST r6,r12,29
  LD r6,r12,6
  ST r6,r12,30
  JMP @mth_pf_shape_done

mth_pf_shape_dot:
  LD r6,r12,22
  MOVI r7,1
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,35
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,5
  LD r7,r12,23
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  MOVI r6,0
  ST r6,r12,28
  MOVI r6,1
  ST r6,r12,29
  ST r6,r12,30
  JMP @mth_pf_shape_done

mth_pf_shape_matvec:
  LD r6,r12,22
  MOVI r7,2
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,35
  MOVI r7,1
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,6
  LD r7,r12,23
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  MOVI r6,1
  ST r6,r12,28
  LD r6,r12,5
  ST r6,r12,29
  MOVI r6,1
  ST r6,r12,30
  JMP @mth_pf_shape_done

mth_pf_shape_matmul:
  LD r6,r12,22
  MOVI r7,2
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,35
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,6
  LD r7,r12,23
  CMP r6,r7
  JMP[NE] @mth_pf_bad_shape
  MOVI r6,2
  ST r6,r12,28
  LD r6,r12,5
  ST r6,r12,29
  LD r6,r12,24
  ST r6,r12,30
  JMP @mth_pf_shape_done

mth_pf_shape_scalar:
  LD r6,r12,22
  CMP r6,r0
  JMP[NE] @mth_pf_bad_shape
  LD r6,r12,35
  CMP r6,r0
  JMP[NE] @mth_pf_bad_shape
  MOVI r6,0
  ST r6,r12,28
  MOVI r6,1
  ST r6,r12,29
  ST r6,r12,30

mth_pf_shape_done:
  LD r6,r12,28
  LD r7,r12,15
  CMP r6,r7
  JMP[NE] @mth_pf_bad_rank

  ; Verify the caller-reserved output payload can hold the derived shape.
  LD r6,r12,28
  CMP r6,r0
  JMP[EQ] @mth_pf_output_scalar
  MOVI r7,1
  CMP r6,r7
  JMP[EQ] @mth_pf_output_vector
  MOVI r7,2
  CMP r6,r7
  JMP[EQ] @mth_pf_output_matrix
  JMP @mth_pf_bad_rank
mth_pf_output_scalar:
  MOVI r6,1
  ST r6,r12,31
  JMP @mth_pf_output_count_done
mth_pf_output_vector:
  LD r6,r12,29
  ST r6,r12,31
  JMP @mth_pf_output_count_done
mth_pf_output_matrix:
  LD r6,r12,29
  LD r7,r12,30
  MUL r6,r6,r7
  ST r6,r12,31
mth_pf_output_count_done:
  LD r6,r12,18
  LD r7,r12,31
  CMP r6,r7
  JMP[LT] @mth_pf_bad_capacity

  ; bit 0: right operand must be nonzero.
  LD r6,r12,9
  MOVI r7,1
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_noalias
  LD r6,r12,17
  LD r6,r6,0
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_precondition
mth_pf_noalias:
  ; bit 1: output payload must not alias either input.
  LD r6,r12,9
  MOVI r7,2
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_finite
  ; Compare half-open payload ranges, not only equal base pointers.
  ; Reject address-plus-capacity wrap before interval comparisons.
  LD r6,r12,32
  LD r7,r12,18
  ADD r8,r6,r7
  CMP r8,r6
  JMP[LT] @mth_pf_bad_descriptor
  LD r9,r12,16
  LD r10,r12,26
  ADD r11,r9,r10
  CMP r11,r9
  JMP[LT] @mth_pf_bad_descriptor
  CMP r6,r11
  JMP[GE] @mth_pf_noalias_right
  CMP r9,r8
  JMP[GE] @mth_pf_noalias_right
  JMP @mth_pf_bad_precondition
mth_pf_noalias_right:
  LD r9,r12,17
  LD r10,r12,27
  ADD r11,r9,r10
  CMP r11,r9
  JMP[LT] @mth_pf_bad_descriptor
  CMP r6,r11
  JMP[GE] @mth_pf_finite
  CMP r9,r8
  JMP[GE] @mth_pf_finite
  JMP @mth_pf_bad_precondition
mth_pf_finite:
  ; bit 2: both descriptors must assert finite input values.
  LD r6,r12,9
  MOVI r7,4
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_nonempty
  LD r6,r12,7
  MOVI r7,1
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_bad_precondition
  LD r6,r12,25
  MOVI r7,1
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_bad_precondition
mth_pf_nonempty:
  ; bit 3: dimensions are nonzero for all non-scalar inputs.
  LD r6,r12,9
  MOVI r7,8
  BOOL.8 r8,r6,r7
  CMP r8,r0
  JMP[EQ] @mth_pf_commit
  LD r6,r12,22
  CMP r6,r0
  JMP[EQ] @mth_pf_nonempty_right
  LD r6,r12,5
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_precondition
mth_pf_nonempty_right:
  LD r6,r12,35
  CMP r6,r0
  JMP[EQ] @mth_pf_commit
  LD r6,r12,23
  CMP r6,r0
  JMP[EQ] @mth_pf_bad_precondition

mth_pf_commit:
  ; Commit output metadata only after all validation succeeds.
  LD r6,r12,3
  LD r7,r12,12
  ST r7,r6,0
  LD r7,r12,28
  ST r7,r6,1
  LD r7,r12,29
  ST r7,r6,2
  LD r7,r12,30
  ST r7,r6,3
  LD r7,r12,19
  ST r7,r6,4
  MOVI r7,0
  ST r7,r6,7
  MOVI r1,0
  MOVI r2,0
  RET

mth_pf_bad_descriptor:
  MOVI r1,1
  RET
mth_pf_bad_payload:
  MOVI r1,2
  RET
mth_pf_bad_kind:
  MOVI r1,3
  RET
mth_pf_bad_rank:
  MOVI r1,4
  RET
mth_pf_bad_domain:
  MOVI r1,5
  RET
mth_pf_bad_shape:
  MOVI r1,6
  RET
mth_pf_bad_capability:
  MOVI r1,7
  RET
mth_pf_bad_precondition:
  MOVI r1,8
  RET
mth_pf_bad_capacity:
  MOVI r1,9
  RET
