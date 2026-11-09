; disasm.nas - the NaN16 disassembler, written in NaN16 (r22, Stage 2 of NAN_MIGRATION_PLAN).
; Prints, per word:  AAAA NNNN_PPPPPPPP_FFFF MNEM[PRED] operands
;   AAAA = word address, NNNN_PPPPPPPP_FFFF = the .nan source line (near head, payload,
;   far head as stored LE16), so a dump round-trips to a .nan file by eye.
; Gate failure (illegal head on either face) prints !GATE and moves on.
; API:  r1=start word, r2=count -> CALL @dis_range   (clobbers r1-r8,r15; keeps r0,r13)
;       r1=word addr           -> CALL @dis_word
; Uses only base NaN16 ops (no MUL/DIV): decimal by subtraction, hex by shifts.
; Convention: r0=0, r13=1. No stack: every routine that CALLs saves r15 in its own slot.

dis_range:
  ST r15,r0,@dr_lnk
  ST r1,r0,@dr_a
  ST r2,r0,@dr_c
dr_lp:
  LD r2,r0,@dr_c
  CMP r2,r0
  JMP[EQ] @dr_end
  LD r1,r0,@dr_a
  CALL @dis_word
  LD r1,r0,@dr_a
  ADD r1,r1,r13
  ST r1,r0,@dr_a
  LD r2,r0,@dr_c
  SUB r2,r2,r13
  ST r2,r0,@dr_c
  JMP @dr_lp
dr_end:
  LD r15,r0,@dr_lnk
  RET
dr_lnk:
.word 0
dr_a:
.word 0
dr_c:
.word 0

; ---- one word -------------------------------------------------------------
dis_word:
  ST r15,r0,@dw_lnk
  ST r1,r0,@dw_addr
  MOVI r2,4
  CALL @dis_hexq
  CALL @dis_putsp
  LD r1,r0,@dw_addr
  LDN r3,r1,0
  ST r3,r0,@dw_n
  MOVI r4,16
  CMP r3,r4
  JMP[LT] @dw_ok
  MOVI r1,@s_gate
  CALL @dis_puts
  JMP @dw_nl
dw_ok:
  SHR r4,r3,2
  LD r1,r4,@t_near
  MOVI r2,4
  CALL @dis_hexq
  MOVI r1,95
  PUTC r1
  LD r1,r0,@dw_addr
  LD r1,r1,0
  ST r1,r0,@dw_p
  MOVI r2,8
  CALL @dis_hexq
  MOVI r1,95
  PUTC r1
  LD r3,r0,@dw_n
  MOVI r4,3
  BOOL.8 r4,r3,r4
  LD r1,r4,@t_far
  MOVI r2,4
  CALL @dis_hexq
  CALL @dis_putsp
  LD r1,r0,@dw_p
  SHR r2,r1,24
  ST r2,r0,@dw_op
  CMP r2,r0
  JMP[NE] @dw_nb
  MOVI r1,@s_bool
  CALL @dis_puts
  LD r1,r0,@dw_n
  CALL @dis_dec
  MOVI r1,801
  ST r1,r0,@dw_form
  JMP @dw_ops
dw_nb:
  MOVI r5,@t_ops
dw_find:
  LD r6,r5,0
  MOVI r7,4096
  CMP r6,r7
  JMP[EQ] @dw_unk
  LD r2,r0,@dw_op
  CMP r6,r2
  JMP[EQ] @dw_found
  MOVI r6,4
  ADD r5,r5,r6
  JMP @dw_find
dw_unk:
  MOVI r1,@s_unk
  CALL @dis_puts
  JMP @dw_nl
dw_found:
  LD r1,r5,1
  ST r1,r0,@dw_form
  MOVI r6,2
  ADD r1,r5,r6
  CALL @dis_putn
  CALL @dis_putpred
dw_ops:
  MOVI r1,0
  ST r1,r0,@dw_first
do_lp:
  LD r1,r0,@dw_form
  MOVI r2,15
  BOOL.8 r3,r1,r2
  CMP r3,r0
  JMP[EQ] @dw_nl
  SHR r1,r1,4
  ST r1,r0,@dw_form
  ST r3,r0,@dw_kind
  LD r4,r0,@dw_first
  CMP r4,r0
  JMP[NE] @do_comma
  MOVI r4,1
  ST r4,r0,@dw_first
  CALL @dis_putsp
  JMP @do_val
do_comma:
  MOVI r1,44
  PUTC r1
do_val:
  LD r3,r0,@dw_kind
  LD r6,r0,@dw_p
  CALL @dis_field
  ST r1,r0,@dw_val
  CMP r2,r0
  JMP[EQ] @do_num
  MOVI r4,114
  PUTC r4
do_num:
  LD r1,r0,@dw_val
  CALL @dis_dec
  JMP @do_lp
dw_nl:
  MOVI r1,10
  PUTC r1
  LD r15,r0,@dw_lnk
  RET
dw_lnk:
.word 0
dw_addr:
.word 0
dw_n:
.word 0
dw_p:
.word 0
dw_op:
.word 0
dw_form:
.word 0
dw_first:
.word 0
dw_kind:
.word 0
dw_val:
.word 0

; field extract: r3=kind(1 d,2 a,3 b,4 imm16,5 j24) r6=payload -> r1=value r2=1 if register
dis_field:
  MOVI r7,15
  MOVI r2,1
  MOVI r5,1
  CMP r3,r5
  JMP[NE] @df2
  SHR r1,r6,20
  BOOL.8 r1,r1,r7
  RET
df2:
  MOVI r5,2
  CMP r3,r5
  JMP[NE] @df3
  SHR r1,r6,16
  BOOL.8 r1,r1,r7
  RET
df3:
  MOVI r5,3
  CMP r3,r5
  JMP[NE] @df4
  BOOL.8 r1,r6,r7
  RET
df4:
  MOVI r2,0
  MOVI r5,4
  CMP r3,r5
  JMP[NE] @df5
  MOVI r7,65535
  BOOL.8 r1,r6,r7
  RET
df5:
  SHL r1,r6,8
  SHR r1,r1,8
  RET

; predicate: prints [NAME] or [n] unless ALWAYS (15)
dis_putpred:
  ST r15,r0,@pp_lnk
  LD r3,r0,@dw_n
  MOVI r4,15
  CMP r3,r4
  JMP[EQ] @pp_out
  MOVI r1,91
  PUTC r1
  LD r1,r3,@t_pred
  CMP r1,r0
  JMP[EQ] @pp_num
  CALL @dis_puts
  JMP @pp_close
pp_num:
  MOVI r1,110
  PUTC r1
  LD r1,r0,@dw_n
  CALL @dis_dec
pp_close:
  MOVI r1,93
  PUTC r1
pp_out:
  LD r15,r0,@pp_lnk
  RET
pp_lnk:
.word 0

; ---- printing primitives ---------------------------------------------------
dis_putsp:
  MOVI r1,32
  PUTC r1
  RET

; puts: r1=ptr to NUL-terminated 4-chars/word LSB-first string
dis_puts:
  MOVI r4,255
dp1:
  LD r2,r1,0
  MOVI r3,4
dp2:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @dp2
  ADD r1,r1,r13
  JMP @dp1

; putn: r1=ptr to a 2-word space-padded name, stops at space or NUL
dis_putn:
  MOVI r4,255
  MOVI r6,32
pn_w:
  LD r2,r1,0
  MOVI r3,4
pn_c:
  BOOL.8 r5,r2,r4
  CMP r5,r6
  RET[EQ]
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @pn_c
  ADD r1,r1,r13
  JMP @pn_w

; hexn: r1=0..15 -> one hex digit
dis_hexn:
  MOVI r2,10
  CMP r1,r2
  JMP[LT] @hn_d
  MOVI r2,55
  JMP @hn_o
hn_d:
  MOVI r2,48
hn_o:
  ADD r2,r2,r1
  PUTC r2
  RET

; hexq: r1=value r2=digit count (1..8), MSD first
dis_hexq:
  ST r15,r0,@hq_lnk
  MOVI r3,8
  SUB r3,r3,r2
hq_sk:
  CMP r3,r0
  JMP[EQ] @hq_go
  SHL r1,r1,4
  SUB r3,r3,r13
  JMP @hq_sk
hq_go:
  ST r2,r0,@hq_cnt
  ST r1,r0,@hq_val
hq_lp:
  LD r1,r0,@hq_val
  SHR r4,r1,28
  SHL r1,r1,4
  ST r1,r0,@hq_val
  ADD r1,r4,r0
  CALL @dis_hexn
  LD r2,r0,@hq_cnt
  SUB r2,r2,r13
  ST r2,r0,@hq_cnt
  CMP r2,r0
  JMP[NE] @hq_lp
  LD r15,r0,@hq_lnk
  RET
hq_lnk:
.word 0
hq_cnt:
.word 0
hq_val:
.word 0

; dec: r1 -> unsigned decimal, no leading zeros (leaf, subtraction only)
dis_dec:
  MOVI r2,@t_pow
  MOVI r3,0
  MOVI r7,48
dd_p:
  LD r4,r2,0
  CMP r4,r0
  RET[EQ]
  MOVI r5,0
dd_s:
  CMP r1,r4
  JMP[LT] @dd_emit
  SUB r1,r1,r4
  ADD r5,r5,r13
  JMP @dd_s
dd_emit:
  CMP r5,r0
  JMP[NE] @dd_pr
  CMP r3,r0
  JMP[NE] @dd_pr
  MOVI r6,1
  CMP r4,r6
  JMP[NE] @dd_nx
dd_pr:
  MOVI r3,1
  ADD r6,r5,r7
  PUTC r6
dd_nx:
  ADD r2,r2,r13
  JMP @dd_p

; ---- tables ------------------------------------------------------------------
s_gate:
.str "!GATE"
s_bool:
.str "BOOL."
s_unk:
.str "???"
t_near:
.word 0x7FF1,0xFFF1,0x7FF8,0xFFF8
t_far:
.word 0xF17F,0xF1FF,0xF87F,0xF8FF
t_pow:
.word 1000000000,100000000,10000000,1000000,100000,10000,1000,100,10,1,0
p_never:
.str "NEVER"
p_gt:
.str "GT"
p_lt:
.str "LT"
p_ne:
.str "NE"
p_eq:
.str "EQ"
p_ge:
.str "GE"
p_le:
.str "LE"
t_pred:
.word @p_never,@p_gt,@p_lt,@p_ne,@p_eq,@p_ge,@p_le,0,0,0,0,0,0,0,0,0
; op, form (nibbles low-first: 1=d 2=a 3=b 4=imm16 5=j24), 7-char space-padded name
t_ops:
.word 1,65
.str "MOVI   "
.word 2,65
.str "MOVHI  "
.word 3,801
.str "ADD    "
.word 4,801
.str "SUB    "
.word 5,1057
.str "SHL    "
.word 6,1057
.str "SHR    "
.word 7,50
.str "CMP    "
.word 8,1057
.str "LD     "
.word 9,1057
.str "ST     "
.word 10,4
.str "JMP    "
.word 11,4
.str "CALL   "
.word 12,0
.str "RET    "
.word 13,2
.str "JMPR   "
.word 15,0
.str "NOP    "
.word 16,33
.str "VPOKE  "
.word 17,33
.str "VPEEK  "
.word 18,1
.str "KEY    "
.word 19,1
.str "TICK   "
.word 20,66
.str "BSWAPM "
.word 21,1
.str "PUTC   "
.word 22,801
.str "STW    "
.word 23,1057
.str "LDN    "
.word 24,1
.str "DISPQ  "
.word 25,1
.str "DISPSET"
.word 26,1
.str "KEYQ   "
.word 27,1
.str "BEEP   "
.word 32,801
.str "MUL    "
.word 33,801
.str "MULH   "
.word 34,801
.str "MULHU  "
.word 35,801
.str "DIV    "
.word 36,801
.str "MOD    "
.word 37,801
.str "DIVU   "
.word 38,801
.str "MODU   "
.word 39,1057
.str "SAR    "
.word 40,50
.str "CMPS   "
.word 41,1057
.str "ADDI   "
.word 42,1057
.str "LDB    "
.word 43,1057
.str "STB    "
.word 44,2
.str "CALLR  "
.word 45,801
.str "BCOPY  "
.word 46,801
.str "BFILL  "
.word 48,801
.str "FADD   "
.word 49,801
.str "FSUB   "
.word 50,801
.str "FMUL   "
.word 51,801
.str "FDIV   "
.word 52,33
.str "FSQRT  "
.word 53,50
.str "FCMP   "
.word 54,33
.str "CVTIF  "
.word 55,33
.str "CVTFI  "
.word 56,5
.str "JMPF   "
.word 57,5
.str "CALLF  "
.word 58,1
.str "FBINFO "
.word 59,0
.str "VSYNC  "
.word 60,801
.str "FBCOPY "
.word 61,801
.str "FBFILL "
.word 62,33
.str "OUTB   "
.word 63,33
.str "INB    "
.word 64,33
.str "INW    "
.word 65,33
.str "OUTW   "
.word 66,801
.str "OUTSW  "
.word 67,33
.str "OUTL   "
.word 255,0
.str "HALT   "
.word 4096,0
.str "END    "
