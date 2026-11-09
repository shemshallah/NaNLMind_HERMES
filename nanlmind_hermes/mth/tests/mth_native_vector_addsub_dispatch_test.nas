; Exercise dynamic vector add/sub through the REAL NaN16 admission dispatcher.
; No host mathematics, no bypass of rule epoch, object shapes or operator IDs.
.org 0
JMP @mth_dispatch_addsub_main
.include "../nan16/mth_preflight_gate.nas"
.include "../nan16/mth_preflight.nas"
.include "../nan16/mth_admission.nas"
.include "../nan16/mth_cosine256_f32.nas"
.include "../nan16/mth_vector_add256_f32.nas"
.include "../nan16/mth_vector_sub256_f32.nas"
.include "../nan16/mth_execute.nas"

mth_dispatch_addsub_main:
  MOVI r13,1
  ; x[:] = 2.0 and y[:] = 3.0 in native binary32 payloads.
  MOVI r1,@mth_dispatch_x
  MOVI r2,@mth_dispatch_y
  MOVI r3,256
  MOVI r4,0
  MOVHI r4,0x4000
  MOVI r5,0
  MOVHI r5,0x4040
mth_dispatch_fill:
  ST r4,r1,0
  ST r5,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_dispatch_fill

  ; Program 902 / operator 79: validated vector + vector -> vector.
  MOVI r8,902
  MOVI r9,79
  CALL @mth_dispatch_call
  CMP r1,r0
  JMP[NE] @mth_dispatch_fail
  MOVI r4,0
  MOVHI r4,0x40A0
  CALL @mth_dispatch_check_all
  CMP r1,r0
  JMP[NE] @mth_dispatch_fail
  MOVI r3,@mth_dispatch_out
  LD r4,r3,0
  MOVI r5,0
  MOVHI r5,0x40A0
  CMP r4,r5
  JMP[NE] @mth_dispatch_fail

  ; Program 903 / operator 80: validated vector - vector -> vector.
  MOVI r8,903
  MOVI r9,80
  CALL @mth_dispatch_call
  CMP r1,r0
  JMP[NE] @mth_dispatch_fail
  MOVI r4,0
  MOVHI r4,0xBF80
  CALL @mth_dispatch_check_all
  CMP r1,r0
  JMP[NE] @mth_dispatch_fail

  ; Wrong program/operator pairing is rejected before guest output mutation.
  MOVI r3,@mth_dispatch_out
  MOVI r4,12345
  ST r4,r3,0
  MOVI r8,902
  MOVI r9,80
  CALL @mth_dispatch_call
  MOVI r10,64
  CMP r1,r10
  JMP[NE] @mth_dispatch_fail
  MOVI r3,@mth_dispatch_out
  LD r4,r3,0
  MOVI r10,12345
  CMP r4,r10
  JMP[NE] @mth_dispatch_fail

  ; Stale rule snapshot epoch is denied before executing FADD.
  MOVI r3,@mth_dispatch_candidate
  MOVI r4,6
  ST r4,r3,9
  MOVI r8,902
  MOVI r9,79
  CALL @mth_dispatch_call
  MOVI r10,18
  CMP r1,r10
  JMP[NE] @mth_dispatch_fail
  MOVI r3,@mth_dispatch_out
  LD r4,r3,0
  MOVI r10,12345
  CMP r4,r10
  JMP[NE] @mth_dispatch_fail
  MOVI r3,@mth_dispatch_candidate
  MOVI r4,7
  ST r4,r3,9

  ; Inadequate output capacity: typed gate denies with status 32+9.
  MOVI r3,@mth_dispatch_output
  MOVI r4,255
  ST r4,r3,6
  MOVI r8,903
  MOVI r9,80
  CALL @mth_dispatch_call
  MOVI r10,41
  CMP r1,r10
  JMP[NE] @mth_dispatch_fail
  MOVI r3,@mth_dispatch_out
  LD r4,r3,0
  MOVI r10,12345
  CMP r4,r10
  JMP[NE] @mth_dispatch_fail
  MOVI r1,@mth_dispatch_pass
  CALL @mth_dispatch_puts
  HALT

; Returns only the already allowlisted native MTH status in r1.
mth_dispatch_call:
  ST r15,r0,@mth_dispatch_saved_return
  MOVI r1,@mth_dispatch_candidate
  MOVI r2,@mth_dispatch_rule
  MOVI r3,@mth_dispatch_left
  MOVI r4,@mth_dispatch_right
  MOVI r5,@mth_dispatch_contract
  MOVI r6,1
  MOVI r7,@mth_dispatch_output
  MOVI r12,@mth_dispatch_scratch
  CALL @mth_execute
  LD r15,r0,@mth_dispatch_saved_return
  RET

; r4 holds the expected raw binary32 value; r1=0 means 256 exact matches.
mth_dispatch_check_all:
  MOVI r2,@mth_dispatch_out
  MOVI r3,256
mth_dispatch_check_loop:
  LD r5,r2,0
  CMP r4,r5
  JMP[NE] @mth_dispatch_check_bad
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_dispatch_check_loop
  MOVI r1,0
  RET
mth_dispatch_check_bad:
  MOVI r1,1
  RET

mth_dispatch_fail:
  MOVI r1,@mth_dispatch_bad
  CALL @mth_dispatch_puts
  HALT
mth_dispatch_puts:
  MOVI r4,255
mth_dispatch_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_dispatch_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_dispatch_puts_byte
  ADD r1,r1,r13
  JMP @mth_dispatch_puts_word

.org 40000
mth_dispatch_candidate:
.word 1,7,4,4,4,1,1,256,256,7,1,1,2,1,100,101,102,103,104,105,106,107
mth_dispatch_rule:
.word 1,7,4,4,4,1,1,1,1,100,101,102,103,104,105,106,107
mth_dispatch_left:
.word 4,1,256,1,1,@mth_dispatch_x,256,1
mth_dispatch_right:
.word 4,1,256,1,1,@mth_dispatch_y,256,1
mth_dispatch_contract:
.word 4,4,4,1,1,1,1,0,1,2
mth_dispatch_output:
.word 0,0,0,0,0,@mth_dispatch_out,256,0
mth_dispatch_pass:
.str "MTH NATIVE VECTOR ADDSUB DISPATCH PASS\n"
mth_dispatch_bad:
.str "MTH NATIVE VECTOR ADDSUB DISPATCH FAIL\n"
.org 41000
mth_dispatch_x:
.word 0
.org 42000
mth_dispatch_y:
.word 0
.org 43000
mth_dispatch_out:
.word 0
.org 44000
mth_dispatch_scratch:
.word 0
.org 45000
mth_dispatch_saved_return:
.word 0
