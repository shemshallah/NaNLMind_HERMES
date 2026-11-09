; MTH RSS typed operation descriptor/contract checker — NaN16 only.
;
; This module validates a typed RSS operation request against a caller-owned
; contract.  It is a pure admission check: it does not execute the operation,
; allocate pages, mutate either descriptor, or touch any payload.
;
; Descriptor and contract layouts are documented in
; docs/NATIVE_GRAPH_CONTRACT_OPS.md.  Both are 16 U32 words.
;
; Descriptor words:
;   0 version=1
;   1 operation_id (1=ENCODE, 2=QUERY, 3=MATCH, 4=FILTER, 5=DRIVE,
;                   6=FOLD, 7=PROPAGATE, 8=PROJECT, 9=DECODE)
;   2 input_kind (0=scalar, 1=vector, 2=matrix, 3=CSR, 4=descriptor)
;   3 output_kind
;   4 domain_id (0=abstract, 1=real, 2=Q16.16, 3=integer, 4=float32)
;   5 input_dim0        6 input_dim1
;   7 output_dim0       8 output_dim1
;   9 input_words      10 output_words
;  11 scale_policy (0=exact Q16.16, 1=exact integer scale 1,
;                   2=float32, 3=reserved)
;  12 policy_flags    13 input bound magnitude
;  14 output bound magnitude                15 reserved=0
;
; Contract words:
;   0 version=1              1 operation_id
;   2 input_kind             3 output_kind
;   4 domain_id              5 max_dim0
;   6 max_dim1               7 max_input_words
;   8 max_output_words       9 allowed scale-policy mask (bits 0..3)
;  10 required policy flags 11 forbidden policy flags
;  12 max bound magnitude   13 shape_rule
;  14 reserved=0            15 reserved=0
;
; Shape rules:
;   0 scalar: all dimensions zero, one input word, one output word
;   1 vector->vector: equal nonempty lengths, dim1 fields zero
;   2 matrix(m,n)->vector(m): n>0
;   3 matrix(m,n)->matrix(m,n): n>0
;   4 CSR(256 vertices, E edges)->vector(256): payload >= 257+2E words
;   5 bounded-any: rectangular dims and payloads only
;
; Policy flag bits (declared by the contract, not hard-coded here):
;   bit0 immutable input, bit1 atomic output, bit2 deterministic,
;   bit3 canonical CSR, bit4 no dense pair materialization,
;   bit5 bounded workspace.
; Contract required/forbidden flags must be disjoint and fit in 6 bits.
;
; ABI:
;   r1 = descriptor pointer, 16 U32 words
;   r2 = contract pointer, 16 U32 words
;   r13 = 1
; Returns r1=1 allow, r2=0
;         r1=0 deny,  r2=typed reason code:
;           1 null descriptor/contract
;           2 unsupported version (descriptor or contract)
;           3 unknown or mismatched operation id
;           4 kind mismatch
;           5 domain mismatch
;           6 dimension exceeds contract bound
;           7 shape rule violation
;           8 payload capacity violation
;           9 scale policy unsupported
;          10 policy flag violation
;          11 numeric bound violation
;          12 malformed descriptor/contract structure
; Clobbers r1-r12 and r14; preserves r13 and r15.  Leaf routine, no CALLs,
; no static scratch.

mth_rss_contract_check:
  ; Null pointers are structural failures; never dereference them.
  CMP r1,r0
  JMP[EQ] @mth_rss_deny_null
  CMP r2,r0
  JMP[EQ] @mth_rss_deny_null

  ; ---- contract structure -------------------------------------------------
  LD r3,r2,0
  MOVI r4,1
  CMP r3,r4
  JMP[NE] @mth_rss_deny_version

  LD r3,r2,1                    ; operation id
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_operation
  MOVI r4,9
  CMP r3,r4
  JMP[GT] @mth_rss_deny_operation

  MOVI r4,4
  LD r3,r2,2                    ; input kind
  CMP r3,r4
  JMP[GT] @mth_rss_deny_kind
  LD r3,r2,3                    ; output kind
  CMP r3,r4
  JMP[GT] @mth_rss_deny_kind
  LD r3,r2,4                    ; domain
  CMP r3,r4
  JMP[GT] @mth_rss_deny_domain

  LD r3,r2,5                    ; max_dim0
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_structure
  MOVI r4,65535
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure
  LD r3,r2,6                    ; max_dim1
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure

  LD r3,r2,7                    ; max_input_words
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_structure
  MOVI r4,0
  MOVHI r4,0x0010               ; 1,048,576 words
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure
  LD r3,r2,8                    ; max_output_words
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_structure
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure

  LD r3,r2,9                    ; allowed scale-policy mask
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_structure
  SHR r4,r3,4
  CMP r4,r0
  JMP[NE] @mth_rss_deny_structure

  MOVI r4,0x3F
  LD r3,r2,10                   ; required flags
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure
  LD r5,r2,11                   ; forbidden flags
  CMP r5,r4
  JMP[GT] @mth_rss_deny_structure
  BOOL.8 r6,r3,r5
  CMP r6,r0
  JMP[NE] @mth_rss_deny_structure

  LD r3,r2,12                   ; max bound magnitude
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_structure
  LD r3,r2,13                   ; shape rule
  MOVI r4,5
  CMP r3,r4
  JMP[GT] @mth_rss_deny_structure
  LD r3,r2,14                   ; reserved
  CMP r3,r0
  JMP[NE] @mth_rss_deny_structure
  LD r3,r2,15
  CMP r3,r0
  JMP[NE] @mth_rss_deny_structure

  ; ---- descriptor identity ------------------------------------------------
  LD r3,r1,0
  MOVI r4,1
  CMP r3,r4
  JMP[NE] @mth_rss_deny_version

  LD r3,r1,1
  LD r4,r2,1
  CMP r3,r4
  JMP[NE] @mth_rss_deny_operation
  LD r3,r1,2
  LD r4,r2,2
  CMP r3,r4
  JMP[NE] @mth_rss_deny_kind
  LD r3,r1,3
  LD r4,r2,3
  CMP r3,r4
  JMP[NE] @mth_rss_deny_kind
  LD r3,r1,4
  LD r4,r2,4
  CMP r3,r4
  JMP[NE] @mth_rss_deny_domain

  LD r3,r1,15                   ; descriptor reserved word
  CMP r3,r0
  JMP[NE] @mth_rss_deny_structure

  ; ---- dimension bounds ---------------------------------------------------
  LD r5,r2,5
  LD r3,r1,5
  CMP r3,r5
  JMP[GT] @mth_rss_deny_dimension
  LD r3,r1,7
  CMP r3,r5
  JMP[GT] @mth_rss_deny_dimension
  LD r5,r2,6
  LD r3,r1,6
  CMP r3,r5
  JMP[GT] @mth_rss_deny_dimension
  LD r3,r1,8
  CMP r3,r5
  JMP[GT] @mth_rss_deny_dimension

  ; ---- scale policy -------------------------------------------------------
  LD r3,r1,11
  MOVI r4,3
  CMP r3,r4
  JMP[GT] @mth_rss_deny_scale
  CMP r3,r0
  JMP[EQ] @mth_rss_scale_bit0
  MOVI r4,1
  CMP r3,r4
  JMP[EQ] @mth_rss_scale_bit1
  MOVI r4,2
  CMP r3,r4
  JMP[EQ] @mth_rss_scale_bit2
  MOVI r5,8
  JMP @mth_rss_scale_bit_ready
mth_rss_scale_bit2:
  MOVI r5,4
  JMP @mth_rss_scale_bit_ready
mth_rss_scale_bit1:
  MOVI r5,2
  JMP @mth_rss_scale_bit_ready
mth_rss_scale_bit0:
  MOVI r5,1
mth_rss_scale_bit_ready:
  LD r4,r2,9
  BOOL.8 r3,r4,r5
  CMP r3,r5
  JMP[NE] @mth_rss_deny_scale

  ; ---- policy flags -------------------------------------------------------
  LD r3,r1,12
  LD r4,r2,10
  BOOL.8 r5,r3,r4
  CMP r5,r4
  JMP[NE] @mth_rss_deny_policy
  LD r4,r2,11
  BOOL.8 r5,r3,r4
  CMP r5,r0
  JMP[NE] @mth_rss_deny_policy

  ; ---- numeric bounds -----------------------------------------------------
  LD r4,r2,12
  LD r3,r1,13
  CMP r3,r4
  JMP[GT] @mth_rss_deny_bounds
  LD r3,r1,14
  CMP r3,r4
  JMP[GT] @mth_rss_deny_bounds

  ; ---- shape dispatch -----------------------------------------------------
  LD r5,r2,13
  CMP r5,r0
  JMP[EQ] @mth_rss_shape_scalar
  MOVI r4,1
  CMP r5,r4
  JMP[EQ] @mth_rss_shape_vector
  MOVI r4,2
  CMP r5,r4
  JMP[EQ] @mth_rss_shape_matvec
  MOVI r4,3
  CMP r5,r4
  JMP[EQ] @mth_rss_shape_matmat
  MOVI r4,4
  CMP r5,r4
  JMP[EQ] @mth_rss_shape_csr
  JMP @mth_rss_shape_any

mth_rss_shape_scalar:
  LD r3,r1,2
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,3
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,5
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,6
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,7
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,8
  CMP r3,r0
  JMP[NE] @mth_rss_deny_shape
  MOVI r6,1
  MOVI r7,1
  JMP @mth_rss_capacity

mth_rss_shape_vector:
  MOVI r4,1
  LD r3,r1,2
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,3
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,5
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_shape
  LD r4,r1,7
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r4,r1,6
  CMP r4,r0
  JMP[NE] @mth_rss_deny_shape
  LD r4,r1,8
  CMP r4,r0
  JMP[NE] @mth_rss_deny_shape
  ADD r6,r3,r0
  ADD r7,r3,r0
  JMP @mth_rss_capacity

mth_rss_shape_matvec:
  MOVI r4,2
  LD r3,r1,2
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  MOVI r4,1
  LD r3,r1,3
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,5
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_shape
  LD r4,r1,6
  CMP r4,r0
  JMP[EQ] @mth_rss_deny_shape
  LD r5,r1,7
  CMP r5,r3
  JMP[NE] @mth_rss_deny_shape
  LD r5,r1,8
  CMP r5,r0
  JMP[NE] @mth_rss_deny_shape
  MUL r6,r3,r4
  ADD r7,r3,r0
  JMP @mth_rss_capacity

mth_rss_shape_matmat:
  MOVI r4,2
  LD r3,r1,2
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,3
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,5
  CMP r3,r0
  JMP[EQ] @mth_rss_deny_shape
  LD r4,r1,6
  CMP r4,r0
  JMP[EQ] @mth_rss_deny_shape
  LD r5,r1,7
  CMP r5,r3
  JMP[NE] @mth_rss_deny_shape
  LD r5,r1,8
  CMP r5,r4
  JMP[NE] @mth_rss_deny_shape
  MUL r6,r3,r4
  ADD r7,r6,r0
  JMP @mth_rss_capacity

mth_rss_shape_csr:
  MOVI r4,3
  LD r3,r1,2
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  MOVI r4,1
  LD r3,r1,3
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r3,r1,5
  MOVI r4,256
  CMP r3,r4
  JMP[NE] @mth_rss_deny_shape
  LD r4,r1,6                    ; edge_count
  MOVI r5,65280
  CMP r4,r5
  JMP[GT] @mth_rss_deny_dimension
  LD r5,r1,7
  CMP r5,r3
  JMP[NE] @mth_rss_deny_shape
  LD r5,r1,8
  CMP r5,r0
  JMP[NE] @mth_rss_deny_shape
  SHL r6,r4,1                   ; 2E
  MOVI r5,256
  ADD r6,r6,r5
  ADD r6,r6,r13                 ; 257 + 2E
  ADD r7,r3,r0                  ; 256 output words
  JMP @mth_rss_capacity

mth_rss_shape_any:
  LD r3,r1,5
  LD r4,r1,6
  CMP r4,r0
  JMP[EQ] @mth_rss_shape_any_vector
  MUL r6,r3,r4
  JMP @mth_rss_shape_any_output
mth_rss_shape_any_vector:
  ADD r6,r3,r0
mth_rss_shape_any_output:
  LD r3,r1,7
  LD r4,r1,8
  CMP r4,r0
  JMP[EQ] @mth_rss_shape_any_out_vector
  MUL r7,r3,r4
  JMP @mth_rss_capacity
mth_rss_shape_any_out_vector:
  ADD r7,r3,r0
  JMP @mth_rss_capacity

mth_rss_capacity:
  LD r3,r1,9
  CMP r3,r6
  JMP[LT] @mth_rss_deny_capacity
  LD r4,r2,7
  CMP r3,r4
  JMP[GT] @mth_rss_deny_capacity
  LD r3,r1,10
  CMP r3,r7
  JMP[LT] @mth_rss_deny_capacity
  LD r4,r2,8
  CMP r3,r4
  JMP[GT] @mth_rss_deny_capacity

  MOVI r1,1
  MOVI r2,0
  RET

mth_rss_deny_null:
  MOVI r1,0
  MOVI r2,1
  RET
mth_rss_deny_version:
  MOVI r1,0
  MOVI r2,2
  RET
mth_rss_deny_operation:
  MOVI r1,0
  MOVI r2,3
  RET
mth_rss_deny_kind:
  MOVI r1,0
  MOVI r2,4
  RET
mth_rss_deny_domain:
  MOVI r1,0
  MOVI r2,5
  RET
mth_rss_deny_dimension:
  MOVI r1,0
  MOVI r2,6
  RET
mth_rss_deny_shape:
  MOVI r1,0
  MOVI r2,7
  RET
mth_rss_deny_capacity:
  MOVI r1,0
  MOVI r2,8
  RET
mth_rss_deny_scale:
  MOVI r1,0
  MOVI r2,9
  RET
mth_rss_deny_policy:
  MOVI r1,0
  MOVI r2,10
  RET
mth_rss_deny_bounds:
  MOVI r1,0
  MOVI r2,11
  RET
mth_rss_deny_structure:
  MOVI r1,0
  MOVI r2,12
  RET
