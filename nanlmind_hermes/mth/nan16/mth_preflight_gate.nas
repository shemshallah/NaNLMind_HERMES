; NaN16 native mathematical preflight gate v1.
; This is the execution authorization boundary. It performs no host-language
; semantic evaluation. All decisions below are NaN16 guest instructions.
;
; ABI:
;   r1 = candidate descriptor (22 words)
;   r2 = authoritative rule descriptor (17 words)
;   r3 = result word (written on every exit)
;   returns r1 = status (0=ALLOW; nonzero=DENY)
;
; Candidate words:
;  0 abi, 1 rule_epoch, 2 lhs_type, 3 rhs_type, 4 out_type,
;  5 lhs_domain, 6 rhs_domain, 7 lhs_dim, 8 rhs_dim,
;  9 candidate_rule_epoch, 10 required_caps, 11 available_caps,
; 12 proof_level, 13 enabled, 14..21 rule_digest[0..7].
; Rule words:
;  0 abi, 1 epoch, 2 lhs_type, 3 rhs_type, 4 out_type,
;  5 domain, 6 shape_policy (0=any, 1=equal dimensions),
;  7 minimum_proof_level, 8 enabled, 9..16 digest[0..7].
;
; Status:
; 0 ALLOW, 1 ABI, 2 STALE_EPOCH, 3 SIGNATURE, 4 DOMAIN,
; 5 SHAPE, 6 CAPABILITY, 7 PROOF, 8 DISABLED, 9 RULE_DIGEST.
; Capability check is (required & available) == required.
; A successful preflight is scoped to the exact rule epoch and 8-word digest.
; The caller MUST bind the returned status and descriptor digest to its NMP
; packet/lease. This routine does not grant ACL authorization or runner capacity.

mth_preflight_gate:
  LD r4,r1,0
  LD r5,r2,0
  CMP r4,r5
  JMP[NE] @mth_pf_abi
  MOVI r6,1
  CMP r4,r6
  JMP[NE] @mth_pf_abi

  LD r4,r1,1
  LD r5,r2,1
  CMP r4,r5
  JMP[NE] @mth_pf_epoch
  LD r5,r1,9
  CMP r4,r5
  JMP[NE] @mth_pf_epoch

  LD r4,r1,2
  LD r5,r2,2
  CMP r4,r5
  JMP[NE] @mth_pf_signature
  LD r4,r1,3
  LD r5,r2,3
  CMP r4,r5
  JMP[NE] @mth_pf_signature
  LD r4,r1,4
  LD r5,r2,4
  CMP r4,r5
  JMP[NE] @mth_pf_signature

  LD r4,r1,5
  LD r5,r2,5
  CMP r4,r5
  JMP[NE] @mth_pf_domain
  LD r6,r1,6
  CMP r6,r5
  JMP[NE] @mth_pf_domain

  LD r6,r2,6
  CMP r6,r0
  JMP[EQ] @mth_pf_shape_ok
  MOVI r7,1
  CMP r6,r7
  JMP[NE] @mth_pf_shape
  LD r6,r1,7
  LD r7,r1,8
  CMP r6,r7
  JMP[NE] @mth_pf_shape
mth_pf_shape_ok:
  LD r6,r1,10
  LD r7,r1,11
  BOOL.8 r8,r6,r7
  CMP r8,r6
  JMP[NE] @mth_pf_capability

  LD r6,r1,12
  LD r7,r2,7
  CMP r6,r7
  JMP[LT] @mth_pf_proof

  LD r6,r1,13
  LD r7,r2,8
  CMP r6,r0
  JMP[EQ] @mth_pf_disabled
  CMP r7,r0
  JMP[EQ] @mth_pf_disabled

  LD r6,r1,14
  LD r7,r2,9
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,15
  LD r7,r2,10
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,16
  LD r7,r2,11
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,17
  LD r7,r2,12
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,18
  LD r7,r2,13
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,19
  LD r7,r2,14
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,20
  LD r7,r2,15
  CMP r6,r7
  JMP[NE] @mth_pf_digest
  LD r6,r1,21
  LD r7,r2,16
  CMP r6,r7
  JMP[NE] @mth_pf_digest

  MOVI r1,0
  JMP @mth_pf_return
mth_pf_abi:
  MOVI r1,1
  JMP @mth_pf_return
mth_pf_epoch:
  MOVI r1,2
  JMP @mth_pf_return
mth_pf_signature:
  MOVI r1,3
  JMP @mth_pf_return
mth_pf_domain:
  MOVI r1,4
  JMP @mth_pf_return
mth_pf_shape:
  MOVI r1,5
  JMP @mth_pf_return
mth_pf_capability:
  MOVI r1,6
  JMP @mth_pf_return
mth_pf_proof:
  MOVI r1,7
  JMP @mth_pf_return
mth_pf_disabled:
  MOVI r1,8
  JMP @mth_pf_return
mth_pf_digest:
  MOVI r1,9
mth_pf_return:
  ST r1,r3,0
  RET
