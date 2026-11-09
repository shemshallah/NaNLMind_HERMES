; n16i.nas - a NaN16 interpreter written in NaN16 (r23, Stage 3 of NAN_MIGRATION_PLAN).
; Runs a guest program of NaN16 words (all 73 ops 0x00-0x48 + HALT) on top of the host machine.
;   entry:  r1 = guest start pc, r2 = host word address of guest word 0 (GB), r3 = guest mask (size-1)
;           CALL @n16i_run     -> returns r1 = guest steps executed (every fetch counts, as in the model)
; Host needs the integer ext ops (MUL/DIVU/...), r0=0, r13=1. Clobbers r1-r12,r14,r15.
; Semantics mirror tools/nan16sim.py (isa=1): both-face gate trap !Gpppp, bad-op trap !Opppp (pc after fetch),
; predication by guest flags (Z,L) in r11 = 2Z+L, BOOL runs unpredicated with n as the truth table.
; Guest I/O (PUTC KEY KEYQ VPOKE VPEEK DISPQ DISPSET BEEP FB* port ops floats) is passed through to the host.
; Guest memory words keep their class: LDN/STW/BCOPY/BSWAPM act on the real words at GB+(addr & mask).
; Register map while running: r10=GB  r11=guest fk  r12=guest pc  r13=1
;   per step: r2=class n  r3=payload  r4=op/result  r5=d  r6=a  r7=imm16  r8=r[a]  r9=r[b]
; Guest registers live in the 16 words at @ni_regs.

n16i_run:
  ST r15,r0,@ni_lnk
  ST r2,r0,@ni_gb
  ST r3,r0,@ni_mask
  ADD r12,r1,r0
  ADD r10,r2,r0
  MOVI r11,0
  ST r0,r0,@ni_steps
  MOVI r5,0
ni_z:
  ST r0,r5,@ni_regs
  ADD r5,r5,r13
  MOVI r4,16
  CMP r5,r4
  JMP[LT] @ni_z
; ---- fetch / gate / predicate / decode ---------------------------------------
ni_loop:
  LD r1,r0,@ni_mask
  BOOL.8 r1,r12,r1
  ADD r1,r1,r10
  ADD r12,r12,r13
  MOVI r4,65535
  BOOL.8 r12,r12,r4
  LD r4,r0,@ni_steps
  ADD r4,r4,r13
  ST r4,r0,@ni_steps
  LDN r2,r1,0
  MOVI r4,16
  CMP r2,r4
  JMP[GE] @ni_trapg
  LD r3,r1,0
  SHR r4,r3,24
  MOVI r5,255
  CMP r4,r5
  JMP[EQ] @ni_halt
  MOVI r5,72
  CMP r4,r5
  JMP[GT] @ni_trapo
  CMP r4,r0
  JMP[EQ] @ni_dec
  LD r5,r11,@ni_bit
  BOOL.8 r5,r5,r2
  CMP r5,r0
  JMP[EQ] @ni_loop
ni_dec:
  SHR r5,r3,20
  MOVI r6,15
  BOOL.8 r5,r5,r6
  SHR r6,r3,16
  MOVI r7,15
  BOOL.8 r6,r6,r7
  MOVI r7,65535
  BOOL.8 r7,r3,r7
  MOVI r8,15
  BOOL.8 r9,r7,r8
  LD r8,r6,@ni_regs
  LD r9,r9,@ni_regs
  LD r14,r4,@ni_optab
  JMPR r14
; ---- common tails ------------------------------------------------------------
ni_t_st:
  ST r4,r5,@ni_regs
  JMP @ni_loop
ni_t_fk:
  MOVI[GT] r11,0
  MOVI[LT] r11,1
  MOVI[EQ] r11,2
  MOVI.8 r11,3
  JMP @ni_loop
ni_t_stf:
  MOVI[GT] r11,0
  MOVI[LT] r11,1
  MOVI[EQ] r11,2
  MOVI.8 r11,3
  ST r4,r5,@ni_regs
  JMP @ni_loop
; ---- halt and traps ----------------------------------------------------------
ni_halt:
  LD r1,r0,@ni_steps
  LD r15,r0,@ni_lnk
  RET
ni_trapg:
  MOVI r1,33
  PUTC r1
  MOVI r1,71
  PUTC r1
  CALL @ni_hex4
  JMP @ni_halt
ni_trapo:
ni_h_bad:
  MOVI r1,33
  PUTC r1
  MOVI r1,79
  PUTC r1
  CALL @ni_hex4
  JMP @ni_halt
ni_hex4:
  ST r15,r0,@ni_hlnk
  SHR r1,r12,12
  MOVI r2,15
  BOOL.8 r1,r1,r2
  CALL @ni_hd
  SHR r1,r12,8
  BOOL.8 r1,r1,r2
  CALL @ni_hd
  SHR r1,r12,4
  BOOL.8 r1,r1,r2
  CALL @ni_hd
  BOOL.8 r1,r12,r2
  CALL @ni_hd
  LD r15,r0,@ni_hlnk
  RET
ni_hd:
  MOVI r3,10
  CMP r1,r3
  JMP[LT] @ni_hd_n
  MOVI r3,55
  ADD r1,r1,r3
  PUTC r1
  RET
ni_hd_n:
  MOVI r3,48
  ADD r1,r1,r3
  PUTC r1
  RET
; ---- address helper: r14 = host word addr of guest (r8+r7) ---------------------
ni_ea:
  ADD r14,r8,r7
  LD r1,r0,@ni_mask
  BOOL.8 r14,r14,r1
  ADD r14,r14,r10
  RET
; r14 = host word addr of guest address in r8 (no offset)
ni_ea0:
  LD r1,r0,@ni_mask
  BOOL.8 r14,r8,r1
  ADD r14,r14,r10
  RET
; ---- handlers: core ----------------------------------------------------------
ni_h_bool:
  MOVI r4,0
  MOVI r14,1
  BOOL.8 r14,r2,r14
  CMP r14,r0
  JMP[EQ] @ni_b2
  BOOL.1 r1,r8,r9
  BOOL.14 r4,r4,r1
ni_b2:
  MOVI r14,2
  BOOL.8 r14,r2,r14
  CMP r14,r0
  JMP[EQ] @ni_b3
  BOOL.2 r1,r8,r9
  BOOL.14 r4,r4,r1
ni_b3:
  MOVI r14,4
  BOOL.8 r14,r2,r14
  CMP r14,r0
  JMP[EQ] @ni_b4
  BOOL.4 r1,r8,r9
  BOOL.14 r4,r4,r1
ni_b4:
  MOVI r14,8
  BOOL.8 r14,r2,r14
  CMP r14,r0
  JMP[EQ] @ni_b5
  BOOL.8 r1,r8,r9
  BOOL.14 r4,r4,r1
ni_b5:
  CMP r4,r0
  JMP @ni_t_stf
ni_h_movi:
  ADD r4,r7,r0
  JMP @ni_t_st
ni_h_movhi:
  LD r4,r5,@ni_regs
  MOVI r14,65535
  BOOL.8 r4,r4,r14
  SHL r14,r7,16
  BOOL.14 r4,r4,r14
  JMP @ni_t_st
ni_h_add:
  ADD r4,r8,r9
  JMP @ni_t_stf
ni_h_sub:
  SUB r4,r8,r9
  JMP @ni_t_stf
ni_h_shl:
  MOVI r14,31
  BOOL.8 r14,r7,r14
  LD r14,r14,@ni_pow2
  MUL r4,r8,r14
  JMP @ni_t_stf
ni_h_shr:
  MOVI r14,31
  BOOL.8 r14,r7,r14
  LD r14,r14,@ni_pow2
  DIVU r4,r8,r14
  JMP @ni_t_stf
ni_h_cmp:
  CMP r8,r9
  JMP @ni_t_fk
ni_h_ld:
  CALL @ni_ea
  LD r4,r14,0
  JMP @ni_t_st
ni_h_st:
  CALL @ni_ea
  LD r4,r5,@ni_regs
  ST r4,r14,0
  JMP @ni_loop
ni_h_jmp:
  ADD r12,r7,r0
  JMP @ni_loop
ni_h_call:
  MOVI r14,15
  ST r12,r14,@ni_regs
  ADD r12,r7,r0
  JMP @ni_loop
ni_h_ret:
  MOVI r14,15
  LD r12,r14,@ni_regs
  MOVI r14,65535
  BOOL.8 r12,r12,r14
  JMP @ni_loop
ni_h_jmpr:
  MOVI r14,65535
  BOOL.8 r12,r8,r14
  JMP @ni_loop
ni_h_nop:
  JMP @ni_loop
ni_h_vpoke:
  LD r4,r5,@ni_regs
  VPOKE r4,r8
  JMP @ni_loop
ni_h_vpeek:
  VPEEK r4,r8
  JMP @ni_t_st
ni_h_key:
  KEY r4
  JMP @ni_t_st
ni_h_tick:
  LD r4,r0,@ni_steps
  MOVI r14,1000
  DIVU r4,r4,r14
  JMP @ni_t_st
ni_h_bswapm:
  CALL @ni_ea
  BSWAPM r14,0
  JMP @ni_loop
ni_h_putc:
  LD r4,r5,@ni_regs
  PUTC r4
  JMP @ni_loop
ni_h_stw:
  CALL @ni_ea0
  LD r4,r5,@ni_regs
  STW r4,r14,r9
  JMP @ni_loop
ni_h_ldn:
  CALL @ni_ea
  LDN r4,r14,0
  JMP @ni_t_st
ni_h_dispq:
  DISPQ r4
  JMP @ni_t_st
ni_h_dispset:
  LD r4,r5,@ni_regs
  DISPSET r4
  JMP @ni_loop
ni_h_keyq:
  KEYQ r4
  JMP @ni_t_st
ni_h_beep:
  LD r4,r5,@ni_regs
  BEEP r4
  JMP @ni_loop
; ---- handlers: integer ext -----------------------------------------------------
ni_h_mul:
  MUL r4,r8,r9
  JMP @ni_t_stf
ni_h_mulh:
  MULH r4,r8,r9
  JMP @ni_t_stf
ni_h_mulhu:
  MULHU r4,r8,r9
  JMP @ni_t_stf
ni_h_div:
  DIV r4,r8,r9
  JMP @ni_t_stf
ni_h_mod:
  MOD r4,r8,r9
  JMP @ni_t_stf
ni_h_divu:
  DIVU r4,r8,r9
  JMP @ni_t_stf
ni_h_modu:
  MODU r4,r8,r9
  JMP @ni_t_stf
ni_h_sar:
  MOVI r14,31
  BOOL.8 r14,r7,r14
  LD r14,r14,@ni_pow2
  SHR r2,r8,31
  CMP r2,r0
  JMP[NE] @ni_sar_n
  DIVU r4,r8,r14
  JMP @ni_sar_f
ni_sar_n:
  BOOL.3 r2,r8,r8
  DIVU r2,r2,r14
  BOOL.3 r4,r2,r2
ni_sar_f:
  CMP r4,r0
  JMP @ni_t_stf
ni_h_cmps:
  CMPS r8,r9
  JMP @ni_t_fk
ni_h_addi:
  MOVI r14,32768
  BOOL.8 r14,r7,r14
  CMP r14,r0
  JMP[EQ] @ni_ai
  MOVHI r7,65535
ni_ai:
  ADD r4,r8,r7
  JMP @ni_t_stf
ni_h_ldb:
  ADD r14,r8,r7
  SHR r2,r14,2
  LD r1,r0,@ni_mask
  BOOL.8 r2,r2,r1
  ADD r2,r2,r10
  LD r3,r2,0
  MOVI r1,3
  BOOL.8 r1,r14,r1
  LD r1,r1,@ni_bpow
  DIVU r4,r3,r1
  MOVI r14,255
  BOOL.8 r4,r4,r14
  JMP @ni_t_st
ni_h_stb:
  ADD r14,r8,r7
  SHR r2,r14,2
  LD r1,r0,@ni_mask
  BOOL.8 r2,r2,r1
  ADD r2,r2,r10
  MOVI r1,3
  BOOL.8 r1,r14,r1
  LD r1,r1,@ni_bpow
  LD r3,r2,0
  MOVI r4,255
  MUL r4,r4,r1
  BOOL.4 r3,r3,r4
  LD r4,r5,@ni_regs
  MOVI r14,255
  BOOL.8 r4,r4,r14
  MUL r4,r4,r1
  BOOL.14 r3,r3,r4
  ST r3,r2,0
  JMP @ni_loop
ni_h_callr:
  MOVI r14,15
  ST r12,r14,@ni_regs
  MOVI r14,65535
  BOOL.8 r12,r8,r14
  JMP @ni_loop
; BCOPY d,a,b : guest mem[r[d]..+r[b]) = mem[r[a]..+r[b])  memmove, classes preserved
ni_h_bcopy:
  LD r1,r5,@ni_regs
  CMP r9,r0
  JMP[EQ] @ni_loop
  CMP r1,r8
  JMP[GT] @ni_bc_back
  MOVI r2,0
ni_bc_f:
  ADD r3,r8,r2
  ADD r4,r1,r2
  CALL @ni_cp1
  ADD r2,r2,r13
  CMP r2,r9
  JMP[LT] @ni_bc_f
  JMP @ni_loop
ni_bc_back:
  ADD r2,r9,r0
ni_bc_b:
  SUB r2,r2,r13
  ADD r3,r8,r2
  ADD r4,r1,r2
  CALL @ni_cp1
  CMP r2,r0
  JMP[NE] @ni_bc_b
  JMP @ni_loop
; r3 = guest src addr, r4 = guest dst addr -> copy one word with its class (clobbers r3,r4,r14)
ni_cp1:
  LD r14,r0,@ni_mask
  BOOL.8 r3,r3,r14
  ADD r3,r3,r10
  BOOL.8 r4,r4,r14
  ADD r4,r4,r10
  LDN r14,r3,0
  LD r3,r3,0
  STW r3,r4,r14
  RET
; BFILL d,a,b : guest mem[r[d]..+r[b]) = +q/+q words, payload r[a]
ni_h_bfill:
  LD r1,r5,@ni_regs
  MOVI r2,0
  CMP r9,r0
  JMP[EQ] @ni_loop
ni_bf:
  ADD r4,r1,r2
  LD r14,r0,@ni_mask
  BOOL.8 r4,r4,r14
  ADD r4,r4,r10
  ST r8,r4,0
  ADD r2,r2,r13
  CMP r2,r9
  JMP[LT] @ni_bf
  JMP @ni_loop
; ---- handlers: float and hardware pass-through -----------------------------------
ni_h_fadd:
  FADD r4,r8,r9
  JMP @ni_t_stf
ni_h_fsub:
  FSUB r4,r8,r9
  JMP @ni_t_stf
ni_h_fmul:
  FMUL r4,r8,r9
  JMP @ni_t_stf
ni_h_fdiv:
  FDIV r4,r8,r9
  JMP @ni_t_stf
ni_h_fsqrt:
  FSQRT r4,r8
  JMP @ni_t_stf
ni_h_fcmp:
  FCMP r8,r9
  JMP @ni_t_fk
ni_h_cvtif:
  CVTIF r4,r8
  JMP @ni_t_stf
ni_h_cvtfi:
  CVTFI r4,r8
  JMP @ni_t_stf
ni_h_cpuid:
  MOVI r4,0
  JMP @ni_t_st
ni_h_rdtsc:
  LD r4,r0,@ni_steps
  JMP @ni_t_st
ni_h_fexp:
  FEXP r4,r8
  JMP @ni_t_stf
ni_h_ftanh:
  FTANH r4,r8
  JMP @ni_t_stf
ni_h_flog:
  FLOG r4,r8
  JMP @ni_t_stf
ni_h_nop2:
  JMP @ni_loop
ni_h_fbinfo:
  FBINFO r4
  JMP @ni_t_st
ni_h_vsync:
  VSYNC
  JMP @ni_loop
ni_h_fbcopy:
  LD r4,r5,@ni_regs
  FBCOPY r4,r8,r9
  JMP @ni_loop
ni_h_fbfill:
  LD r4,r5,@ni_regs
  FBFILL r4,r8,r9
  JMP @ni_loop
ni_h_outb:
  LD r4,r5,@ni_regs
  OUTB r4,r8
  JMP @ni_loop
ni_h_inb:
  INB r4,r8
  JMP @ni_t_st
ni_h_inw:
  INW r4,r8
  JMP @ni_t_st
ni_h_outw:
  LD r4,r5,@ni_regs
  OUTW r4,r8
  JMP @ni_loop
ni_h_outsw:
  LD r4,r5,@ni_regs
  OUTSW r4,r8,r9
  JMP @ni_loop
ni_h_outl:
  LD r4,r5,@ni_regs
  OUTL r4,r8
  JMP @ni_loop
; ---- state and tables ------------------------------------------------------------
ni_lnk:
.word 0
ni_hlnk:
.word 0
ni_gb:
.word 0
ni_mask:
.word 0
ni_steps:
.word 0
ni_regs:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
ni_bit:
.word 1,2,4,8
ni_bpow:
.word 1,256,65536,16777216
ni_pow2:
.word 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768
.word 65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824,2147483648
ni_optab:
.word @ni_h_bool,@ni_h_movi,@ni_h_movhi,@ni_h_add,@ni_h_sub,@ni_h_shl,@ni_h_shr,@ni_h_cmp
.word @ni_h_ld,@ni_h_st,@ni_h_jmp,@ni_h_call,@ni_h_ret,@ni_h_jmpr,@ni_h_bad,@ni_h_nop
.word @ni_h_vpoke,@ni_h_vpeek,@ni_h_key,@ni_h_tick,@ni_h_bswapm,@ni_h_putc,@ni_h_stw,@ni_h_ldn
.word @ni_h_dispq,@ni_h_dispset,@ni_h_keyq,@ni_h_beep,@ni_h_bad,@ni_h_bad,@ni_h_bad,@ni_h_bad
.word @ni_h_mul,@ni_h_mulh,@ni_h_mulhu,@ni_h_div,@ni_h_mod,@ni_h_divu,@ni_h_modu,@ni_h_sar
.word @ni_h_cmps,@ni_h_addi,@ni_h_ldb,@ni_h_stb,@ni_h_callr,@ni_h_bcopy,@ni_h_bfill,@ni_h_nop2
.word @ni_h_fadd,@ni_h_fsub,@ni_h_fmul,@ni_h_fdiv,@ni_h_fsqrt,@ni_h_fcmp,@ni_h_cvtif,@ni_h_cvtfi
.word @ni_h_jmp,@ni_h_call,@ni_h_fbinfo,@ni_h_vsync,@ni_h_fbcopy,@ni_h_fbfill,@ni_h_outb,@ni_h_inb
.word @ni_h_inw,@ni_h_outw,@ni_h_outsw,@ni_h_outl
.word @ni_h_cpuid,@ni_h_rdtsc,@ni_h_fexp,@ni_h_ftanh,@ni_h_flog
