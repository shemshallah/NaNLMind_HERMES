; HERMES native point-0 bridge gate: SHA-256 vectors, NaN16DE words, N16B v2 round trip.
; Prints "SHA <len> <hex>" and "ENV <hex>" lines so a host oracle can compare byte-for-byte,
; and self-checks the round trip, tamper, slot and gate rejections.
.org 0
JMP @t_main
.include "hermes_n16bridge.nas"

; r1 = byte ptr, r2 = len : print lowercase hex then newline
t_hex:
ST r15,r0,@t_hex_ret
ST r1,r0,@t_hex_p
ST r2,r0,@t_hex_n
MOVI r6,0
t_hex_l:
LD r2,r0,@t_hex_n
CMP r6,r2
JMP[GE] @t_hex_end
LD r1,r0,@t_hex_p
ADD r1,r1,r6
LDB r3,r1,0
SHR r4,r3,4
CALL @t_nib
MOVI r4,15
BOOL.8 r4,r3,r4
CALL @t_nib
ADD r6,r6,r13
JMP @t_hex_l
t_hex_end:
MOVI r4,10
PUTC r4
LD r15,r0,@t_hex_ret
RET
t_nib:
MOVI r5,10
CMP r4,r5
JMP[LT] @t_nib_d
ADDI r4,r4,87
PUTC r4
RET
t_nib_d:
ADDI r4,r4,48
PUTC r4
RET
t_hex_ret:
.word 0
t_hex_p:
.word 0
t_hex_n:
.word 0

; r1 = number -> prints decimal then space
t_dec:
MOVI r2,10
MOVI r5,0
t_dec_l:
MODU r3,r1,r2
DIVU r1,r1,r2
ADDI r3,r3,48
MOVI r4,@t_dec_buf
ADD r4,r4,r5
ST r3,r4,0
ADD r5,r5,r13
CMP r1,r0
JMP[NE] @t_dec_l
t_dec_p:
SUB r5,r5,r13
MOVI r4,@t_dec_buf
ADD r4,r4,r5
LD r3,r4,0
PUTC r3
CMP r5,r0
JMP[NE] @t_dec_p
MOVI r3,32
PUTC r3
RET
t_dec_buf:
.word 0,0,0,0,0,0,0,0,0,0,0,0

; sha of pattern: r1 = len ; bytes are (i*7+3)&255 at byte 0xC000
t_sha_case:
ST r15,r0,@t_sc_ret
ST r1,r0,@t_sc_len
MOVI r6,0
MOVI r5,0
MOVHI r5,0
MOVI r7,0xC000
t_sc_fill:
LD r1,r0,@t_sc_len
CMP r6,r1
JMP[GE] @t_sc_go
MOVI r2,7
MUL r3,r6,r2
ADDI r3,r3,3
STB r3,r7,0
ADDI r7,r7,1
ADD r6,r6,r13
JMP @t_sc_fill
t_sc_go:
MOVI r1,0xC000
LD r2,r0,@t_sc_len
MOVI r3,0xC800
CALL @hb_sha256
MOVI r4,83
PUTC r4
MOVI r4,72
PUTC r4
MOVI r4,65
PUTC r4
MOVI r4,32
PUTC r4
LD r1,r0,@t_sc_len
CALL @t_dec
MOVI r1,0xC800
MOVI r2,32
CALL @t_hex
LD r15,r0,@t_sc_ret
RET
t_sc_ret:
.word 0
t_sc_len:
.word 0

t_main:
MOVI r13,1
; --- SHA-256 vectors across padding boundaries
MOVI r1,0
CALL @t_sha_case
MOVI r1,3
CALL @t_sha_case
MOVI r1,55
CALL @t_sha_case
MOVI r1,56
CALL @t_sha_case
MOVI r1,63
CALL @t_sha_case
MOVI r1,64
CALL @t_sha_case
MOVI r1,65
CALL @t_sha_case
MOVI r1,200
CALL @t_sha_case
; --- build a 3-word frame at word 0x3000
MOVI r5,0x3000
MOVI r1,0
MOVHI r1,0x0102
MOVI r2,0x0304
BOOL.14 r1,r1,r2
MOVI r6,10
STW r1,r5,r6
MOVI r1,0xBEEF
MOVHI r1,0xDEAD
ADDI r7,r5,1
MOVI r6,6
STW r1,r7,r6
MOVI r1,0
ADDI r7,r5,2
MOVI r6,3
STW r1,r7,r6
MOVI r1,0xF00D
MOVHI r1,0xCAFE
ADDI r7,r5,3
MOVI r6,15
STW r1,r7,r6
; slot 2, request 0x00000000_00C0FFEE... req_hi=0x01234567 req_lo=0x89ABCDEF, epoch 0x0000000A_00000007, status 0
MOVI r1,2
MOVI r2,0x4567
MOVHI r2,0x0123
MOVI r3,0xCDEF
MOVHI r3,0x89AB
MOVI r4,10
MOVI r5,7
MOVI r6,0
CALL @hb_env_set
MOVI r1,0x3000
MOVI r2,4
MOVI r3,0xC400
CALL @hb_n16b_encode
CMP r3,r0
JMP[NE] @t_fail
ST r1,r0,@t_envlen
MOVI r4,69
PUTC r4
MOVI r4,78
PUTC r4
MOVI r4,86
PUTC r4
MOVI r4,32
PUTC r4
MOVI r1,0xC400
LD r2,r0,@t_envlen
CALL @t_hex
; --- decode round trip into word 0x3100
MOVI r1,0xC400
LD r2,r0,@t_envlen
MOVI r3,0x3400
CALL @hb_n16b_decode
CMP r3,r0
JMP[NE] @t_fail
MOVI r4,4
CMP r1,r4
JMP[NE] @t_fail
; words must match nibble+payload
MOVI r6,0
t_cmp:
MOVI r4,0x3000
ADD r4,r4,r6
LDN r7,r4,0
LD r8,r4,0
MOVI r4,0x3400
ADD r4,r4,r6
LDN r9,r4,0
LD r10,r4,0
CMP r7,r9
JMP[NE] @t_fail
CMP r8,r10
JMP[NE] @t_fail
ADD r6,r6,r13
MOVI r4,4
CMP r6,r4
JMP[LT] @t_cmp
; envelope metadata restored
LD r4,r0,@hb_env_slot
MOVI r5,2
CMP r4,r5
JMP[NE] @t_fail
LD r4,r0,@hb_env_epl
MOVI r5,7
CMP r4,r5
JMP[NE] @t_fail
; --- tamper: flip one payload byte -> digest status 9
MOVI r4,0xC400
ADDI r4,r4,70
LDB r5,r4,0
MOVI r6,1
BOOL.6 r5,r5,r6
STB r5,r4,0
MOVI r1,0xC400
LD r2,r0,@t_envlen
MOVI r3,0x3500
CALL @hb_n16b_decode
MOVI r4,9
CMP r3,r4
JMP[NE] @t_fail
; restore and tamper header version -> 4
MOVI r4,0xC400
ADDI r4,r4,70
LDB r5,r4,0
MOVI r6,1
BOOL.6 r5,r5,r6
STB r5,r4,0
MOVI r4,0xC400
MOVI r5,3
STB r5,r4,4
MOVI r1,0xC400
LD r2,r0,@t_envlen
MOVI r3,0x3500
CALL @hb_n16b_decode
MOVI r4,4
CMP r3,r4
JMP[NE] @t_fail
; --- encode rejects slot 6 and a non-gated word
MOVI r1,6
MOVI r2,0
MOVI r3,0
MOVI r4,0
MOVI r5,0
MOVI r6,0
CALL @hb_env_set
MOVI r1,0x3000
MOVI r2,1
MOVI r3,0xC400
CALL @hb_n16b_encode
MOVI r4,6
CMP r3,r4
JMP[NE] @t_fail
MOVI r1,0
MOVI r2,0
MOVI r3,0
MOVI r4,0
MOVI r5,0
MOVI r6,0
CALL @hb_env_set
; word 0x3300 left virgin (zero heads) must trip the gate
MOVI r1,0x3600
MOVI r2,1
MOVI r3,0xC400
CALL @hb_n16b_encode
MOVI r4,2
CMP r3,r4
JMP[NE] @t_fail
MOVI r1,@t_pass
JMP @t_say
t_fail:
MOVI r1,@t_failmsg
t_say:
MOVI r11,255
t_say_loop:
LD r2,r1,0
MOVI r3,4
t_say_byte:
BOOL.8 r4,r2,r11
CMP r4,r0
JMP[EQ] @t_say_done
PUTC r4
SHR r2,r2,8
SUB r3,r3,r13
JMP[NE] @t_say_byte
ADDI r1,r1,1
JMP @t_say_loop
t_say_done:
HALT
t_envlen:
.word 0
t_pass:
.str "HERMES N16BRIDGE PASS\n"
t_failmsg:
.str "HERMES N16BRIDGE FAIL\n"