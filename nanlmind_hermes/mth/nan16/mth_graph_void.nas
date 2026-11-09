; MTH regularized inverse-square void potential, float32 v1 — NaN16 only.
;
; Operator (RSS.POT.CORE.v1 core-potential form):
;   V(r) = alpha / (r*r + eps*eps)
; for exactly 256 radial positions.
;
; ABI:
;   r1 = input radius array, 256 float32 payloads (IEEE-754 binary32)
;   r2 = alpha, float32 payload
;   r3 = epsilon, float32 payload
;   r4 = output array, 256 float32 payloads
;   r13 = constant 1
;   returns r1=1 success, r1=0 invalid parameter or radius
;   clobbers r2-r12; leaf routine, no static scratch.
;
; Validation (all before the first output write):
;   radius[i] >= 0 and finite for all i in [0,255]
;   alpha     >= 0 and finite
;   epsilon   >  0 and finite
; A finite nonnegative float32 is accepted iff FCMP value,+inf reports LT.
; FCMP is unordered for NaN, and every ordinary predicate (including LT)
; is skipped in that state, so a NaN also falls through to the invalid path.
; Negative values are caught by FCMP value,0 -> LT. Negative zero is accepted
; as zero. The complete radius pass finishes before any ST to the output.
;
; Numeric contract:
;   Input and output are raw binary32 payloads. The kernel performs the three
;   IEEE-754 binary32 operations r*r, r*r+eps*eps, and alpha/(...) with the
;   VM's round-to-nearest-even float unit. It performs no host math, table
;   lookup, or decimal conversion. Constants below are loaded as IEEE-754
;   bit patterns with MOVI (low 16 bits) + MOVHI (high 16 bits):
;     0.0f        = 0x00000000
;     +infinity   = 0x7F800000
;   Output is not guaranteed finite for every legal input: a legal positive
;   epsilon whose square underflows to zero, or an extreme denominator, can
;   still produce an infinity or a zero under IEEE-754 rules. Callers that
;   require finite outputs must bound the accepted parameter domain.
;
; Input and output spans must not overlap. The output span is untouched on
; every invalid return.

mth_graph_void_potential:
  ; alpha >= 0 and alpha < +infinity.
  MOVI r5,0
  FCMP r2,r5
  JMP[LT] @mth_graph_void_invalid
  MOVI r6,0
  MOVHI r6,0x7F80
  FCMP r2,r6
  JMP[LT] @mth_graph_void_alpha_ok
  JMP @mth_graph_void_invalid
mth_graph_void_alpha_ok:
  ; epsilon > 0.
  FCMP r3,r5
  JMP[GT] @mth_graph_void_epsilon_positive
  JMP @mth_graph_void_invalid
mth_graph_void_epsilon_positive:
  ; epsilon < +infinity.
  FCMP r3,r6
  JMP[LT] @mth_graph_void_epsilon_ok
  JMP @mth_graph_void_invalid
mth_graph_void_epsilon_ok:
  ; Complete radius preflight before touching output.
  ADD r7,r1,r0
  MOVI r5,256
mth_graph_void_validate_radius:
  LD r8,r7,0
  FCMP r8,r0
  JMP[LT] @mth_graph_void_invalid
  FCMP r8,r6
  JMP[LT] @mth_graph_void_radius_ok
  JMP @mth_graph_void_invalid
mth_graph_void_radius_ok:
  ADD r7,r7,r13
  SUB r5,r5,r13
  CMP r5,r0
  JMP[GT] @mth_graph_void_validate_radius

  ; All inputs are valid. Compute the regularizer once, then apply.
  FMUL r6,r3,r3                 ; eps*eps
  ADD r7,r1,r0                  ; radius cursor
  ADD r4,r4,r0                  ; output cursor
  MOVI r5,256
mth_graph_void_apply:
  LD r8,r7,0                    ; r
  FMUL r9,r8,r8                 ; r*r
  FADD r9,r9,r6                 ; r*r + eps*eps
  FDIV r10,r2,r9                ; alpha / (r*r + eps*eps)
  ST r10,r4,0
  ADD r7,r7,r13
  ADD r4,r4,r13
  SUB r5,r5,r13
  CMP r5,r0
  JMP[GT] @mth_graph_void_apply

  MOVI r1,1
  RET

mth_graph_void_invalid:
  MOVI r1,0
  RET
