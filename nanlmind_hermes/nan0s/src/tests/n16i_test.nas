; n16i_test.nas - Stage 3 differential test. Host driver below word 8192, guest program at word 8192 (.org).
; mode word (patched by tools/ngate_n16i.sh): 0 = run guest natively, 1 = run it under n16i.
; gsel word: 0 = full guest suite, 1 = gate trap, 2 = bad-op trap.  Guest addresses are 8192+x; the guest
; window wraps (mask 8191) so they alias x, which is exactly what the interpreter must reproduce.
JMP @drv
.include "../tools/n16i.nas"
drv:
  MOVI r13,1
  LD r1,r0,@mode
  CMP r1,r0
  JMP[EQ] @gmain
  MOVI r1,@gmain
  MOVI r2,8192
  MOVI r3,8191
  CALL @n16i_run
  ST r1,r0,@res_steps
  HALT
mode:
.word 0
res_steps:
.word 0
.org 8192
gmain:
  MOVI r13,1
  TICK r3
  CALL @gph
  LD r1,r0,@gsel
  CMP r1,r13
  JMP[EQ] @g_gate
  MOVI r2,2
  CMP r1,r2
  JMP[EQ] @g_bad
  MOVI r1,52428
  MOVHI r1,52428
  MOVI r2,43690
  MOVHI r2,43690

  BOOL.0 r3,r1,r2
  CALL @gph
  BOOL.1 r3,r1,r2
  CALL @gph
  BOOL.2 r3,r1,r2
  CALL @gph
  BOOL.3 r3,r1,r2
  CALL @gph
  BOOL.4 r3,r1,r2
  CALL @gph
  BOOL.5 r3,r1,r2
  CALL @gph
  BOOL.6 r3,r1,r2
  CALL @gph
  BOOL.7 r3,r1,r2
  CALL @gph
  BOOL.8 r3,r1,r2
  CALL @gph
  BOOL.9 r3,r1,r2
  CALL @gph
  BOOL.10 r3,r1,r2
  CALL @gph
  BOOL.11 r3,r1,r2
  CALL @gph
  BOOL.12 r3,r1,r2
  CALL @gph
  BOOL.13 r3,r1,r2
  CALL @gph
  BOOL.14 r3,r1,r2
  CALL @gph
  BOOL.15 r3,r1,r2
  CALL @gph
  MOVI r1,0
  SUB r3,r1,r13
  CALL @gph
  ADD r3,r3,r13
  CALL @gph
  MOVI r1,65535
  MOVHI r1,32767
  ADD r3,r1,r13
  CALL @gph
  MOVI r1,1234
  SHL r3,r1,17
  CALL @gph
  SHR r3,r3,5
  CALL @gph
  SHL r3,r1,31
  SHL r3,r3,1
  CALL @gph

  MOVI r1,5
  MOVI r2,5
  CMP r1,r2
  MOVI r4,48
  MOVI[EQ] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[ALWAYS] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NEVER] r4,49
  PUTC r4
  MOVI r4,10
  PUTC r4
  MOVI r1,3
  MOVI r2,5
  CMP r1,r2
  MOVI r4,48
  MOVI[EQ] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[ALWAYS] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NEVER] r4,49
  PUTC r4
  MOVI r4,10
  PUTC r4
  MOVI r1,5
  MOVI r2,3
  CMP r1,r2
  MOVI r4,48
  MOVI[EQ] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[GT] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[LE] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[ALWAYS] r4,49
  PUTC r4
  MOVI r4,48
  MOVI[NEVER] r4,49
  PUTC r4
  MOVI r4,10
  PUTC r4
  MOVI r1,0
  SUB r1,r1,r13
  ADD r1,r1,r13
  MOVI r4,48
  MOVI.8 r4,49
  MOVI.15 r4,50
  PUTC r4
  MOVI r4,10
  PUTC r4
  LA r1,305419896
  ST r1,r0,@scr
  LD r3,r0,@scr
  CALL @gph
  MOVI r5,@scr
  MOVI r6,6
  STW r1,r5,r6
  LDN r3,r5,0
  CALL @gph
  BSWAPM r5,0
  LDN r3,r5,0
  CALL @gph
  LD r3,r5,0
  CALL @gph
  MOVI r6,13
  STW r1,r5,r6
  LDN r3,r5,0
  CALL @gph
  CALL @sub1
  CALL @gph
  MOVI r6,@jr1
  JMPR r6
  MOVI r3,222
jr1:
  MOVI r3,111
  CALL @gph
  LA r1,100000
  LA r2,70000
  MUL r3,r1,r2
  CALL @gph
  MULH r3,r1,r2
  CALL @gph
  MOVI r1,0
  SUB r1,r1,r13
  MULH r3,r1,r2
  CALL @gph
  MULHU r3,r1,r2
  CALL @gph
  MOVI r1,1000
  MOVI r2,0
  SUB r2,r2,r13
  MOVI r4,7
  DIV r3,r1,r4
  CALL @gph
  SUB r1,r0,r1
  DIV r3,r1,r4
  CALL @gph
  MOD r3,r1,r4
  CALL @gph
  DIVU r3,r2,r4
  CALL @gph
  MODU r3,r2,r4
  CALL @gph
  DIV r3,r1,r0
  CALL @gph
  MOVI r4,48
  MOVI.8 r4,49
  PUTC r4
  SAR r3,r1,3
  CALL @gph
  SAR r3,r2,31
  CALL @gph
  MOVI r1,100
  SAR r3,r1,2
  CALL @gph
  MOVI r1,3
  MOVI r2,9
  CMPS r2,r1
  MOVI r4,48
  MOVI[GT] r4,49
  PUTC r4
  MOVI r1,0
  SUB r1,r1,r13
  CMPS r1,r2
  MOVI r4,48
  MOVI[LT] r4,49
  PUTC r4
  MOVI r4,10
  PUTC r4
  MOVI r1,1000
  ADDI r3,r1,0
  ADDI r3,r1,65535
  CALL @gph
  ADDI r3,r1,300
  CALL @gph
  MOVI r1,@scr
  MOVI r3,22136
  MOVHI r3,43981
  ST r3,r0,@scr
  LDB r3,r1,0
  CALL @gph
  LDB r3,r1,1
  CALL @gph
  LDB r3,r1,2
  CALL @gph
  LDB r3,r1,3
  CALL @gph
  MOVI r2,255
  MOVI r6,256
  SHL r6,r1,2
  STB r2,r6,6
  STB r6,r6,9
  LD r3,r0,@scr
  CALL @gph
  MOVI r1,@bc_src
  MOVI r2,@bc_dst
  MOVI r4,4
  BCOPY r2,r1,r4
  LD r3,r0,@bc_dst
  CALL @gph
  LD r3,r2,3
  CALL @gph
  MOVI r1,@bc_src
  MOVI r2,@bc_src
  MOVI r5,1
  ADD r2,r2,r5
  MOVI r4,3
  BCOPY r2,r1,r4
  LD r3,r1,1
  CALL @gph
  LD r3,r1,3
  CALL @gph
  MOVI r1,@bc_src
  MOVI r4,6
  MOVI r5,@bc_src
  MOVI r2,3
  ADD r2,r5,r2
  BCOPY r5,r2,r4
  LD r3,r1,0
  CALL @gph
  MOVI r1,@bc_src
  LA r3,3735928559
  MOVI r4,3
  BFILL r1,r3,r4
  LD r3,r1,2
  CALL @gph
  MOVI r1,@scr
  CALLF @subf
  CALL @gph
  MOVI r1,@dis_sample
  MOVI r2,22
  CALL @dis_range
  MOVI r1,3
  MOVHI r1,786
  CALL @sweep
  MOVI r1,3
  MOVHI r1,18
  CALL @sweep
  MOVI r1,@s_done
  CALL @dis_puts
  HALT
g_gate:
  MOVI r3,1
  CALL @gph
  JMP @gbad
g_bad:
  MOVI r1,3
  MOVI r2,5
  CMP r1,r2
  JMP @gbadop
gbadop:
.word 469762048
gbad:
.word 0
subf:
  MOVI r3,77
  JMPF @subf2
  MOVI r3,88
subf2:
  RET
sub1:
  MOVI r3,321
  RET
gph:
  ADD r10,r3,r0
  MOVI r9,8
gph1:
  SHR r7,r10,28
  MOVI r8,10
  CMP r7,r8
  JMP[LT] @gph_d
  MOVI r8,55
  JMP @gph_p
gph_d:
  MOVI r8,48
gph_p:
  ADD r7,r7,r8
  PUTC r7
  SHL r10,r10,4
  SUB r9,r9,r13
  CMP r9,r0
  JMP[NE] @gph1
  MOVI r7,10
  PUTC r7
  RET
gsel:
.word 0
scr:
.word 0
bc_src:
.word 11,22,33,44,55,66,77,88
bc_dst:
.word 0,0,0,0,0,0
sweep:
  ST r15,r0,@sw_lnk
  MOVI r6,0
  MOVI r7,@buf
  MOVI r8,16
sw1:
  STW r1,r7,r6
  ADD r7,r7,r13
  ADD r6,r6,r13
  CMP r6,r8
  JMP[NE] @sw1
  MOVI r1,@buf
  MOVI r2,16
  CALL @dis_range
  LD r15,r0,@sw_lnk
  RET
sw_lnk:
.word 0
s_done:
.str "N16I GUEST DONE\n"
buf:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.include "../tools/disasm.nas"
dis_sample:
BOOL.6 r1,r2,r3
MOVI r1,42
MOVHI r2,4660
ADD r3,r1,r2
SUB[EQ] r4,r5,r6
SHL r1,r2,7
CMP r1,r2
LD[NE] r1,r2,16
ST r3,r0,512
JMP[LT] 100
CALL 200
RET
JMPR r15
NOP
BSWAPM r1,5
STW r1,r2,r3
DISPSET r3
OUTB r1,r2
JMPF 3000000
MUL r1,r2,r3
VSYNC
HALT
