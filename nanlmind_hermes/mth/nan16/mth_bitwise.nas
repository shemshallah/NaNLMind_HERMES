; MTH unsigned bitwise and comparison primitives — native NaN16 only.
; Binary functions take r1=a,r2=b and return r1.
; Predicates return r1=0 or 1. Comparisons are unsigned as defined by NaN16 CMP.

mth_u32_and:
  BOOL.8 r1,r1,r2
  RET

mth_u32_or:
  BOOL.14 r1,r1,r2
  RET

mth_u32_xor:
  BOOL.6 r1,r1,r2
  RET

mth_u32_xnor:
  BOOL.9 r1,r1,r2
  RET

mth_u32_nand:
  BOOL.7 r1,r1,r2
  RET

mth_u32_nor:
  BOOL.1 r1,r1,r2
  RET

mth_u32_not:
  BOOL.3 r1,r1,r0
  RET

mth_u32_min:
  CMP r1,r2
  JMP[LE] @mth_u32_min_done
  ADD r1,r2,r0
mth_u32_min_done:
  RET

mth_u32_max:
  CMP r1,r2
  JMP[GE] @mth_u32_max_done
  ADD r1,r2,r0
mth_u32_max_done:
  RET

mth_u32_lt:
  CMP r1,r2
  JMP[LT] @mth_u32_lt_yes
  MOVI r1,0
  RET
mth_u32_lt_yes:
  MOVI r1,1
  RET

mth_u32_eq:
  CMP r1,r2
  JMP[EQ] @mth_u32_eq_yes
  MOVI r1,0
  RET
mth_u32_eq_yes:
  MOVI r1,1
  RET
