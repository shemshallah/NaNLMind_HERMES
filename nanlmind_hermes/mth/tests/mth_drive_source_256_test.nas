.org 0
JMP @mth_drive_source_256_test_main
.include "../nan16/mth_drive_source_256.nas"

mth_drive_source_256_test_main:
  MOVI r13,1

  ; state = 1.0 for all 256.
  LA r1,@mth_drive_source_256_state
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_drive_source_256_fill_state:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_drive_source_256_fill_state

  ; source = 2.0 for all 256.
  LA r2,@mth_drive_source_256_source
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_drive_source_256_fill_source:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_drive_source_256_fill_source

  ; dt = 0.5: out = 1.0 + 0.5*2.0 = 2.0 for every element.
  LA r9,@mth_drive_source_256_dt
  MOVI r10,0
  MOVHI r10,0x3F00
  ST r10,r9,0
  LA r1,@mth_drive_source_256_state
  LA r2,@mth_drive_source_256_source
  LA r3,@mth_drive_source_256_dt
  LA r4,@mth_drive_source_256_out
  CALL @mth_drive_source_256
  CMP r1,r13
  JMP[NE] @mth_drive_source_256_fail

  LA r4,@mth_drive_source_256_out
  MOVI r6,0
  MOVI r8,0
  MOVHI r8,0x4000
  MOVI r7,0
mth_drive_source_256_check_half:
  ADD r9,r4,r7
  LD r10,r9,0
  CMP r10,r8
  JMP[NE] @mth_drive_source_256_fail
  ADD r7,r7,r13
  MOVI r9,256
  CMP r7,r9
  JMP[LT] @mth_drive_source_256_check_half

  ; dt = 0: out = state = 1.0 for every element.
  LA r9,@mth_drive_source_256_dt
  ST r0,r9,0
  LA r1,@mth_drive_source_256_state
  LA r2,@mth_drive_source_256_source
  LA r3,@mth_drive_source_256_dt
  LA r4,@mth_drive_source_256_out
  CALL @mth_drive_source_256
  CMP r1,r13
  JMP[NE] @mth_drive_source_256_fail

  LA r4,@mth_drive_source_256_out
  MOVI r6,0
  MOVI r8,0
  MOVHI r8,0x3F80
  MOVI r7,0
mth_drive_source_256_check_zero_dt:
  ADD r9,r4,r7
  LD r10,r9,0
  CMP r10,r8
  JMP[NE] @mth_drive_source_256_fail
  ADD r7,r7,r13
  MOVI r9,256
  CMP r7,r9
  JMP[LT] @mth_drive_source_256_check_zero_dt

  ; Inputs must be read-only across both calls.
  LA r4,@mth_drive_source_256_state
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_drive_source_256_fail
  LA r4,@mth_drive_source_256_source
  LD r5,r4,0
  MOVI r6,0
  MOVHI r6,0x4000
  CMP r5,r6
  JMP[NE] @mth_drive_source_256_fail

  LA r1,@mth_drive_source_256_pass
  CALL @mth_drive_source_256_puts
  HALT

mth_drive_source_256_fail:
  LA r1,@mth_drive_source_256_bad
  CALL @mth_drive_source_256_puts
  HALT

mth_drive_source_256_puts:
  MOVI r4,255
mth_drive_source_256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_drive_source_256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_drive_source_256_puts_byte
  ADD r1,r1,r13
  JMP @mth_drive_source_256_puts_word

.org 512
mth_drive_source_256_state:
.word 0
.org 1024
mth_drive_source_256_source:
.word 0
.org 1536
mth_drive_source_256_dt:
.word 0
.org 1537
mth_drive_source_256_out:
.word 0
.org 2048
mth_drive_source_256_pass:
.str "MTH DRIVE SOURCE 256 PASS\n"
mth_drive_source_256_bad:
.str "MTH DRIVE SOURCE 256 FAIL\n"
