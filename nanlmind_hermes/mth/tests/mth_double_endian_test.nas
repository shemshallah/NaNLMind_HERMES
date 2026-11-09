; MTH double-endian reentrancy and validation fixture — NaN16 only.
.org 0
JMP @mth_de_test_main
.include "../nan16/mth_double_endian.nas"

mth_de_test_main:
  MOVI r13,1

  ; Two independent caller-owned frames and destinations.
  MOVI r1,0x1234
  MOVI r2,4096
  MOVI r12,@mth_de_frame_a
  CALL @mth_de_encode

  MOVI r1,0xABCD
  MOVI r2,4100
  MOVI r12,@mth_de_frame_b
  CALL @mth_de_encode

  MOVI r1,4096
  MOVI r12,@mth_de_frame_a
  CALL @mth_de_decode
  MOVI r3,0x1234
  CMP r1,r3
  JMP[NE] @mth_de_fail
  CMP r2,r0
  JMP[EQ] @mth_de_fail

  MOVI r1,4100
  MOVI r12,@mth_de_frame_b
  CALL @mth_de_decode
  MOVI r3,0xABCD
  CMP r1,r3
  JMP[NE] @mth_de_fail
  CMP r2,r0
  JMP[EQ] @mth_de_fail

  ; Corrupt the low bit of the second member while retaining its class/tag.
  MOVI r2,4096
  LD r5,r2,1
  MOVI r7,1
  BOOL.6 r5,r5,r7
  LDN r6,r2,1
  ADD r2,r2,r13
  STW r5,r2,r6

  MOVI r1,4096
  MOVI r12,@mth_de_frame_a
  CALL @mth_de_decode
  CMP r2,r0
  JMP[NE] @mth_de_fail

  ; Wrong tag in the second member must be rejected.
  MOVI r2,4100
  LD r5,r2,1
  MOVI r6,0xFFFF
  BOOL.8 r5,r5,r6
  MOVI r7,2
  SHL r7,r7,16
  BOOL.14 r5,r5,r7
  LDN r6,r2,1
  ADD r2,r2,r13
  STW r5,r2,r6
  MOVI r1,4100
  MOVI r12,@mth_de_frame_b
  CALL @mth_de_decode
  CMP r2,r0
  JMP[NE] @mth_de_fail

  ; Wrong first-member class must be rejected.
  MOVI r2,4096
  MOVI r5,9
  MOVI r6,9
  STW r5,r2,r6
  MOVI r1,4096
  MOVI r12,@mth_de_frame_a
  CALL @mth_de_decode
  CMP r2,r0
  JMP[NE] @mth_de_fail

  ; Restore the first member, then reject a wrong second-member class.
  MOVI r2,4096
  MOVI r5,0
  MOVI r6,10
  STW r5,r2,r6

  MOVI r2,4096
  ADD r2,r2,r13
  MOVI r5,0
  MOVI r6,9
  STW r5,r2,r6
  MOVI r1,4096
  MOVI r12,@mth_de_frame_a
  CALL @mth_de_decode
  CMP r2,r0
  JMP[NE] @mth_de_fail

  MOVI r1,@mth_de_pass
  CALL @mth_de_puts
  HALT

mth_de_fail:
  MOVI r1,@mth_de_bad
  CALL @mth_de_puts
  HALT

mth_de_puts:
  MOVI r4,255
mth_de_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_de_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_de_puts_byte
  ADD r1,r1,r13
  JMP @mth_de_puts_word

mth_de_frame_a:
.word 0
.word 0
.word 0
mth_de_frame_b:
.word 0
.word 0
.word 0
mth_de_pass:
.str "MTH DOUBLE-ENDIAN REENTRANT PASS\n"
mth_de_bad:
.str "MTH DOUBLE-ENDIAN REENTRANT FAIL\n"
