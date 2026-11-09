; MTH checked U32 span addition fixture — NaN16 only.
.org 0
JMP @mth_vec_test_main
.include "../nan16/mth_vector.nas"

mth_vec_test_main:
  MOVI r13,1
  MOVI r12,@mth_vec_frame

  ; A = [1, UINT32_MAX, 3], B = [2, 1, 4].
  MOVI r1,@mth_vec_a
  MOVI r2,@mth_vec_b
  MOVI r3,@mth_vec_out
  MOVI r4,3
  CALL @mth_u32_span_add_checked
  CMP r1,r0
  JMP[NE] @mth_vec_fail

  ; Overflow must leave every output sentinel unchanged.
  MOVI r2,@mth_vec_out
  LD r5,r2,0
  MOVI r6,77
  CMP r5,r6
  JMP[NE] @mth_vec_fail
  LD r5,r2,1
  MOVI r6,88
  CMP r5,r6
  JMP[NE] @mth_vec_fail
  LD r5,r2,2
  MOVI r6,99
  CMP r5,r6
  JMP[NE] @mth_vec_fail

  ; Repair A[1], then expect [3,11,7].
  MOVI r2,@mth_vec_a
  ADD r2,r2,r13
  MOVI r1,10
  STW r1,r2,r0

  MOVI r1,@mth_vec_a
  MOVI r2,@mth_vec_b
  MOVI r3,@mth_vec_out
  MOVI r4,3
  CALL @mth_u32_span_add_checked
  MOVI r6,1
  CMP r1,r6
  JMP[NE] @mth_vec_fail

  MOVI r2,@mth_vec_out
  LD r5,r2,0
  MOVI r6,3
  CMP r5,r6
  JMP[NE] @mth_vec_fail
  LD r5,r2,1
  MOVI r6,11
  CMP r5,r6
  JMP[NE] @mth_vec_fail
  LD r5,r2,2
  MOVI r6,7
  CMP r5,r6
  JMP[NE] @mth_vec_fail

  ; Empty spans succeed without touching their addresses.
  MOVI r1,0
  MOVI r2,0
  MOVI r3,0
  MOVI r4,0
  MOVI r6,1
  CALL @mth_u32_span_add_checked
  CMP r1,r6
  JMP[NE] @mth_vec_fail

  MOVI r1,@mth_vec_pass
  CALL @mth_vec_puts
  HALT

mth_vec_fail:
  MOVI r1,@mth_vec_bad
  CALL @mth_vec_puts
  HALT

mth_vec_puts:
  MOVI r4,255
mth_vec_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_vec_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_vec_puts_byte
  ADD r1,r1,r13
  JMP @mth_vec_puts_word

mth_vec_frame:
.word 0
.word 0
.word 0
mth_vec_a:
.word 1
.word 0xFFFFFFFF
.word 3
mth_vec_b:
.word 2
.word 1
.word 4
mth_vec_out:
.word 77
.word 88
.word 99
mth_vec_pass:
.str "MTH U32 VECTOR ADD PASS\n"
mth_vec_bad:
.str "MTH U32 VECTOR ADD FAIL\n"
