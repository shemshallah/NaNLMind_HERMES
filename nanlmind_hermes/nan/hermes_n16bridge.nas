; HERMES native point-0 bridge core — raw NaN16 only, no host math, no host hash.
;
;   L0  SHA-256 (streaming)            hb_sha_init / hb_sha_update / hb_sha_final / hb_sha256
;   L1  NaN16DE word codec             hb_word_to_wire / hb_wire_to_word / hb_opcode / hb_bool_eval
;   L2  N16B v2 point-0 envelope       hb_env_set / hb_n16b_encode / hb_n16b_decode
;
; Byte buffers live in guest memory packed 4 bytes per word (payload LE), exactly as
; LDB/STB address them: byte address = 4*word_index + lane.  Byte pointers passed in
; and out of this file are such byte addresses and MUST be 4-byte aligned for N16B.
; A NaN16 word on the wire is 8 bytes: far head BE | payload LE | near head LE
; (identical to nanlmind_hermes/nan16de.py NanWord.pack).  Envelope layout is
; identical to runner_riddle/point0_bridge.py BinaryEnvelope v2 (big-endian ints):
;   0 "N16B" | 4 ver=2 | 5 slot | 6 req u64 | 14 epoch u64 | 22 nwords u32 |
;   26 result_status u8 | 27 "BS0:0" | 32 sha256 | 64 payload (8*nwords)
; digest = SHA256(bytes[0..31] || payload).
;
; Conventions: r0=0, r13=1 (caller keeps), r15=link, statics saved with ST r15.
; Status codes (r3 unless noted): 0 OK, 1 EINVAL, 2 gate (bad NaN head),
;   3 bad magic, 4 bad version, 5 bad point-zero id, 6 slot>5, 7 bad result status,
;   8 length/count mismatch or too large, 9 digest mismatch.

; ---------------------------------------------------------------- SHA-256 leaf sigmas
hb_bsig0:
SHR r2,r1,2
SHL r3,r1,30
BOOL.14 r2,r2,r3
SHR r3,r1,13
SHL r4,r1,19
BOOL.14 r3,r3,r4
BOOL.6 r2,r2,r3
SHR r3,r1,22
SHL r4,r1,10
BOOL.14 r3,r3,r4
BOOL.6 r1,r2,r3
RET
hb_bsig1:
SHR r2,r1,6
SHL r3,r1,26
BOOL.14 r2,r2,r3
SHR r3,r1,11
SHL r4,r1,21
BOOL.14 r3,r3,r4
BOOL.6 r2,r2,r3
SHR r3,r1,25
SHL r4,r1,7
BOOL.14 r3,r3,r4
BOOL.6 r1,r2,r3
RET
hb_ssig0:
SHR r2,r1,7
SHL r3,r1,25
BOOL.14 r2,r2,r3
SHR r3,r1,18
SHL r4,r1,14
BOOL.14 r3,r3,r4
BOOL.6 r2,r2,r3
SHR r3,r1,3
BOOL.6 r1,r2,r3
RET
hb_ssig1:
SHR r2,r1,17
SHL r3,r1,15
BOOL.14 r2,r2,r3
SHR r3,r1,19
SHL r4,r1,13
BOOL.14 r3,r3,r4
BOOL.6 r2,r2,r3
SHR r3,r1,10
BOOL.6 r1,r2,r3
RET

; ---------------------------------------------------------------- compress one 64-byte block
; in r1 = byte address of block.  Updates hb_sha_h.  Clobbers r1-r12.
hb_sha_compress:
ST r15,r0,@hb_sc_ret
MOVI r5,0
MOVI r6,@hb_sha_w
hb_sc_ld:
LDB r2,r1,0
LDB r3,r1,1
LDB r4,r1,2
LDB r7,r1,3
SHL r2,r2,24
SHL r3,r3,16
SHL r4,r4,8
BOOL.14 r2,r2,r3
BOOL.14 r2,r2,r4
BOOL.14 r2,r2,r7
ADD r8,r6,r5
ST r2,r8,0
ADDI r1,r1,4
ADD r5,r5,r13
MOVI r8,16
CMP r5,r8
JMP[LT] @hb_sc_ld
MOVI r5,16
hb_sc_ex:
ADD r8,r6,r5
ADDI r9,r8,-2
LD r1,r9,0
CALL @hb_ssig1
ADD r10,r1,r0
ADDI r9,r8,-7
LD r2,r9,0
ADD r10,r10,r2
ADDI r9,r8,-15
LD r1,r9,0
CALL @hb_ssig0
ADD r10,r10,r1
ADDI r9,r8,-16
LD r2,r9,0
ADD r10,r10,r2
ST r10,r8,0
ADD r5,r5,r13
MOVI r9,64
CMP r5,r9
JMP[LT] @hb_sc_ex
; v = h
MOVI r4,@hb_sha_h
MOVI r9,@hb_sv
MOVI r5,0
hb_sc_cp:
ADD r2,r4,r5
LD r3,r2,0
ADD r2,r9,r5
ST r3,r2,0
ADD r5,r5,r13
MOVI r2,8
CMP r5,r2
JMP[LT] @hb_sc_cp
MOVI r5,0
hb_sc_rd:
MOVI r9,@hb_sv
LD r1,r9,4
CALL @hb_bsig1
ADD r10,r1,r0
LD r2,r9,4
LD r3,r9,5
LD r4,r9,6
BOOL.6 r3,r3,r4
BOOL.8 r3,r2,r3
BOOL.6 r3,r4,r3
ADD r10,r10,r3
LD r3,r9,7
ADD r10,r10,r3
MOVI r3,@hb_sha_k
ADD r3,r3,r5
LD r3,r3,0
ADD r10,r10,r3
MOVI r3,@hb_sha_w
ADD r3,r3,r5
LD r3,r3,0
ADD r10,r10,r3
LD r1,r9,0
CALL @hb_bsig0
ADD r11,r1,r0
LD r2,r9,0
LD r3,r9,1
LD r4,r9,2
BOOL.14 r1,r2,r3
BOOL.8 r1,r1,r4
BOOL.8 r2,r2,r3
BOOL.14 r1,r1,r2
ADD r11,r11,r1
LD r2,r9,3
ADD r2,r2,r10
ADD r1,r10,r11
LD r3,r9,6
ST r3,r9,7
LD r3,r9,5
ST r3,r9,6
LD r3,r9,4
ST r3,r9,5
ST r2,r9,4
LD r3,r9,2
ST r3,r9,3
LD r3,r9,1
ST r3,r9,2
LD r3,r9,0
ST r3,r9,1
ST r1,r9,0
ADD r5,r5,r13
MOVI r3,64
CMP r5,r3
JMP[LT] @hb_sc_rd
; h += v
MOVI r4,@hb_sha_h
MOVI r9,@hb_sv
MOVI r5,0
hb_sc_fin:
ADD r2,r4,r5
LD r3,r2,0
ADD r6,r9,r5
LD r7,r6,0
ADD r3,r3,r7
ST r3,r2,0
ADD r5,r5,r13
MOVI r2,8
CMP r5,r2
JMP[LT] @hb_sc_fin
LD r15,r0,@hb_sc_ret
RET
hb_sc_ret:
.word 0

; ---------------------------------------------------------------- streaming interface
hb_sha_init:
MOVI r4,@hb_sha_iv
MOVI r5,@hb_sha_h
MOVI r6,0
hb_si_cp:
ADD r2,r4,r6
LD r3,r2,0
ADD r2,r5,r6
ST r3,r2,0
ADD r6,r6,r13
MOVI r2,8
CMP r6,r2
JMP[LT] @hb_si_cp
ST r0,r0,@hb_sha_len
ST r0,r0,@hb_sha_bl
RET

; r1 = byte pointer, r2 = byte count.  Clobbers r1-r12.
hb_sha_update:
ST r15,r0,@hb_su_ret
ST r1,r0,@hb_su_ptr
ST r2,r0,@hb_su_rem
hb_su_loop:
LD r2,r0,@hb_su_rem
CMP r2,r0
JMP[EQ] @hb_su_done
LD r3,r0,@hb_sha_bl
CMP r3,r0
JMP[NE] @hb_su_byte
MOVI r4,64
CMP r2,r4
JMP[LT] @hb_su_byte
LD r1,r0,@hb_su_ptr
CALL @hb_sha_compress
LD r1,r0,@hb_su_ptr
ADDI r1,r1,64
ST r1,r0,@hb_su_ptr
LD r2,r0,@hb_su_rem
ADDI r2,r2,-64
ST r2,r0,@hb_su_rem
LD r2,r0,@hb_sha_len
ADDI r2,r2,64
ST r2,r0,@hb_sha_len
JMP @hb_su_loop
hb_su_byte:
LD r1,r0,@hb_su_ptr
LDB r4,r1,0
ADDI r1,r1,1
ST r1,r0,@hb_su_ptr
MOVI r5,@hb_sha_buf
SHL r5,r5,2
ADD r5,r5,r3
STB r4,r5,0
ADDI r3,r3,1
ST r3,r0,@hb_sha_bl
LD r2,r0,@hb_su_rem
ADDI r2,r2,-1
ST r2,r0,@hb_su_rem
LD r2,r0,@hb_sha_len
ADDI r2,r2,1
ST r2,r0,@hb_sha_len
MOVI r4,64
CMP r3,r4
JMP[LT] @hb_su_loop
MOVI r1,@hb_sha_buf
SHL r1,r1,2
CALL @hb_sha_compress
ST r0,r0,@hb_sha_bl
JMP @hb_su_loop
hb_su_done:
LD r15,r0,@hb_su_ret
RET
hb_su_ret:
.word 0
hb_su_ptr:
.word 0
hb_su_rem:
.word 0

; r1 = destination byte pointer for the 32-byte digest.  Clobbers r1-r12.
hb_sha_final:
ST r15,r0,@hb_sf_ret
ST r1,r0,@hb_sf_dst
LD r3,r0,@hb_sha_bl
MOVI r5,@hb_sha_buf
SHL r5,r5,2
ADD r6,r5,r3
MOVI r4,0x80
STB r4,r6,0
ADDI r3,r3,1
hb_sf_zero:
MOVI r4,56
CMP r3,r4
JMP[EQ] @hb_sf_len
MOVI r4,64
CMP r3,r4
JMP[EQ] @hb_sf_flush
ADD r6,r5,r3
STB r0,r6,0
ADDI r3,r3,1
JMP @hb_sf_zero
hb_sf_flush:
MOVI r1,@hb_sha_buf
SHL r1,r1,2
CALL @hb_sha_compress
MOVI r5,@hb_sha_buf
SHL r5,r5,2
MOVI r3,0
hb_sf_z2:
MOVI r4,56
CMP r3,r4
JMP[EQ] @hb_sf_len
ADD r6,r5,r3
STB r0,r6,0
ADDI r3,r3,1
JMP @hb_sf_z2
hb_sf_len:
MOVI r5,@hb_sha_buf
SHL r5,r5,2
LD r7,r0,@hb_sha_len
SHR r8,r7,29
SHL r9,r7,3
; high 32 bits at 56..59, low at 60..63, big-endian
SHR r4,r8,24
STB r4,r5,56
SHR r4,r8,16
STB r4,r5,57
SHR r4,r8,8
STB r4,r5,58
STB r8,r5,59
SHR r4,r9,24
STB r4,r5,60
SHR r4,r9,16
STB r4,r5,61
SHR r4,r9,8
STB r4,r5,62
STB r9,r5,63
MOVI r1,@hb_sha_buf
SHL r1,r1,2
CALL @hb_sha_compress
LD r3,r0,@hb_sf_dst
MOVI r5,@hb_sha_h
MOVI r6,0
hb_sf_out:
ADD r7,r5,r6
LD r7,r7,0
SHR r4,r7,24
STB r4,r3,0
SHR r4,r7,16
STB r4,r3,1
SHR r4,r7,8
STB r4,r3,2
STB r7,r3,3
ADDI r3,r3,4
ADD r6,r6,r13
MOVI r4,8
CMP r6,r4
JMP[LT] @hb_sf_out
LD r15,r0,@hb_sf_ret
RET
hb_sf_ret:
.word 0
hb_sf_dst:
.word 0

; one-shot: r1 = src byte ptr, r2 = len, r3 = dst byte ptr (32 bytes).
hb_sha256:
ST r15,r0,@hb_s1_ret
ST r3,r0,@hb_s1_dst
ST r1,r0,@hb_s1_src
ST r2,r0,@hb_s1_len
CALL @hb_sha_init
LD r1,r0,@hb_s1_src
LD r2,r0,@hb_s1_len
CALL @hb_sha_update
LD r1,r0,@hb_s1_dst
CALL @hb_sha_final
LD r15,r0,@hb_s1_ret
RET
hb_s1_ret:
.word 0
hb_s1_dst:
.word 0
hb_s1_src:
.word 0
hb_s1_len:
.word 0

; ---------------------------------------------------------------- NaN16DE word codec
; hb_word_to_wire: r1 = word index, r2 = dst byte ptr -> 8 bytes; r3 = status (0/2)
hb_word_to_wire:
LDN r4,r1,0
MOVI r5,15
CMP r5,r4
JMP[LT] @hb_w2w_bad
MOVI r12,255
MOVI r5,3
BOOL.8 r6,r4,r5
SHR r7,r4,2
MOVI r8,@hb_heads
ADD r9,r8,r6
LD r9,r9,0
SHR r10,r9,8
STB r10,r2,0
BOOL.8 r10,r9,r12
STB r10,r2,1
LD r10,r1,0
STB r10,r2,2
SHR r11,r10,8
STB r11,r2,3
SHR r11,r10,16
STB r11,r2,4
SHR r11,r10,24
STB r11,r2,5
ADD r9,r8,r7
LD r9,r9,0
BOOL.8 r10,r9,r12
STB r10,r2,6
SHR r10,r9,8
STB r10,r2,7
MOVI r3,0
RET
hb_w2w_bad:
MOVI r3,2
RET

; hb_head_class: r1 = head value -> r1 = class 0..3 or 0xFFFFFFFF
hb_head_class:
MOVI r2,0x7FF1
CMP r1,r2
MOVI r2,0
JMP[EQ] @hb_hc_ok
MOVI r2,0xFFF1
CMP r1,r2
MOVI r2,1
JMP[EQ] @hb_hc_ok
MOVI r2,0x7FF8
CMP r1,r2
MOVI r2,2
JMP[EQ] @hb_hc_ok
MOVI r2,0xFFF8
CMP r1,r2
MOVI r2,3
JMP[EQ] @hb_hc_ok
MOVI r1,0
MOVHI r1,0xFFFF
MOVI r2,0xFFFF
BOOL.14 r1,r1,r2
RET
hb_hc_ok:
ADD r1,r2,r0
RET

; hb_wire_to_word: r1 = src byte ptr (8 bytes), r2 = dst word index -> r3 status (0/2)
hb_wire_to_word:
ST r15,r0,@hb_ww_ret
ST r1,r0,@hb_ww_src
ST r2,r0,@hb_ww_dst
LDB r4,r1,0
SHL r4,r4,8
LDB r5,r1,1
BOOL.14 r1,r4,r5
CALL @hb_head_class
MOVI r4,3
CMP r4,r1
JMP[LT] @hb_ww_bad
ST r1,r0,@hb_ww_far
LD r6,r0,@hb_ww_src
LDB r4,r6,7
SHL r4,r4,8
LDB r5,r6,6
BOOL.14 r1,r4,r5
CALL @hb_head_class
MOVI r4,3
CMP r4,r1
JMP[LT] @hb_ww_bad
SHL r1,r1,2
LD r2,r0,@hb_ww_far
BOOL.14 r1,r1,r2
ST r1,r0,@hb_ww_nib
LD r6,r0,@hb_ww_src
LDB r4,r6,2
LDB r5,r6,3
SHL r5,r5,8
BOOL.14 r4,r4,r5
LDB r5,r6,4
SHL r5,r5,16
BOOL.14 r4,r4,r5
LDB r5,r6,5
SHL r5,r5,24
BOOL.14 r4,r4,r5
LD r7,r0,@hb_ww_dst
LD r8,r0,@hb_ww_nib
STW r4,r7,r8MOVI r3,0
LD r15,r0,@hb_ww_ret
RET
hb_ww_bad:
MOVI r3,2
LD r15,r0,@hb_ww_ret
RET
hb_ww_ret:
.word 0
hb_ww_src:
.word 0
hb_ww_dst:
.word 0
hb_ww_far:
.word 0
hb_ww_nib:
.word 0

; hb_opcode: r1 = word index -> r1 = class nibble (the Boolean opcode) or 0xFFFFFFFF
hb_opcode:
LDN r1,r1,0
RET
; hb_bool_eval: r1 = nibble, r2 = a bit, r3 = b bit -> r1 = (nibble >> (2a+b)) & 1
hb_bool_eval:
BOOL.8 r2,r2,r13
BOOL.8 r3,r3,r13
SHL r2,r2,1
ADD r2,r2,r3
SHR r4,r1,0
MOVI r5,0
hb_be_sh:
CMP r2,r0
JMP[EQ] @hb_be_done
SHR r4,r4,1
SUB r2,r2,r13
JMP @hb_be_sh
hb_be_done:
BOOL.8 r1,r4,r13
RET

; ---------------------------------------------------------------- N16B v2 envelope
; hb_env_set: r1=slot(0..5) r2=req_hi r3=req_lo r4=epoch_hi r5=epoch_lo r6=result status(0..10)
hb_env_set:
ST r1,r0,@hb_env_slot
ST r2,r0,@hb_env_reqh
ST r3,r0,@hb_env_reql
ST r4,r0,@hb_env_eph
ST r5,r0,@hb_env_epl
ST r6,r0,@hb_env_st
RET

; hb_put32be: r1 = value, r2 = byte ptr
hb_put32be:
SHR r3,r1,24
STB r3,r2,0
SHR r3,r1,16
STB r3,r2,1
SHR r3,r1,8
STB r3,r2,2
STB r1,r2,3
RET
; hb_get32be: r1 = byte ptr -> r1 = value
hb_get32be:
LDB r2,r1,0
SHL r2,r2,24
LDB r3,r1,1
SHL r3,r3,16
BOOL.14 r2,r2,r3
LDB r3,r1,2
SHL r3,r3,8
BOOL.14 r2,r2,r3
LDB r3,r1,3
BOOL.14 r1,r2,r3
RET

; hb_n16b_encode: r1 = src word index, r2 = word count, r3 = dst byte ptr (aligned)
;   -> r1 = total bytes, r3 = status.  Uses statics set by hb_env_set.
hb_n16b_encode:
ST r15,r0,@hb_ne_ret
ST r1,r0,@hb_ne_src
ST r2,r0,@hb_ne_cnt
ST r3,r0,@hb_ne_dst
MOVI r4,3
BOOL.8 r4,r3,r4
CMP r4,r0
JMP[NE] @hb_ne_bad
MOVI r4,4096
CMP r4,r2
JMP[LT] @hb_ne_big
LD r4,r0,@hb_env_slot
MOVI r5,5
CMP r5,r4
JMP[LT] @hb_ne_slot
LD r4,r0,@hb_env_st
MOVI r5,10
CMP r5,r4
JMP[LT] @hb_ne_stat
; header bytes 0..31
LD r3,r0,@hb_ne_dst
MOVI r4,0x4E
STB r4,r3,0
MOVI r4,0x31
STB r4,r3,1
MOVI r4,0x36
STB r4,r3,2
MOVI r4,0x42
STB r4,r3,3
MOVI r4,2
STB r4,r3,4
LD r4,r0,@hb_env_slot
STB r4,r3,5
LD r1,r0,@hb_env_reqh
ADDI r2,r3,6
CALL @hb_put32be
LD r3,r0,@hb_ne_dst
LD r1,r0,@hb_env_reql
ADDI r2,r3,10
CALL @hb_put32be
LD r3,r0,@hb_ne_dst
LD r1,r0,@hb_env_eph
ADDI r2,r3,14
CALL @hb_put32be
LD r3,r0,@hb_ne_dst
LD r1,r0,@hb_env_epl
ADDI r2,r3,18
CALL @hb_put32be
LD r3,r0,@hb_ne_dst
LD r1,r0,@hb_ne_cnt
ADDI r2,r3,22
CALL @hb_put32be
LD r3,r0,@hb_ne_dst
LD r4,r0,@hb_env_st
STB r4,r3,26
MOVI r4,0x42
STB r4,r3,27
MOVI r4,0x53
STB r4,r3,28
MOVI r4,0x30
STB r4,r3,29
MOVI r4,0x3A
STB r4,r3,30
MOVI r4,0x30
STB r4,r3,31
; payload words at dst+64
LD r3,r0,@hb_ne_dst
ADDI r3,r3,64
ST r3,r0,@hb_ne_pp
MOVI r5,0
ST r5,r0,@hb_ne_i
hb_ne_pl:
LD r5,r0,@hb_ne_i
LD r4,r0,@hb_ne_cnt
CMP r5,r4
JMP[GE] @hb_ne_hash
LD r1,r0,@hb_ne_src
ADD r1,r1,r5
LD r2,r0,@hb_ne_pp
CALL @hb_word_to_wire
CMP r3,r0
JMP[NE] @hb_ne_gate
LD r2,r0,@hb_ne_pp
ADDI r2,r2,8
ST r2,r0,@hb_ne_pp
LD r5,r0,@hb_ne_i
ADD r5,r5,r13
ST r5,r0,@hb_ne_i
JMP @hb_ne_pl
hb_ne_hash:
CALL @hb_sha_init
LD r1,r0,@hb_ne_dst
MOVI r2,32
CALL @hb_sha_update
LD r1,r0,@hb_ne_dst
ADDI r1,r1,64
LD r2,r0,@hb_ne_cnt
SHL r2,r2,3
CALL @hb_sha_update
LD r1,r0,@hb_ne_dst
ADDI r1,r1,32
CALL @hb_sha_final
LD r1,r0,@hb_ne_cnt
SHL r1,r1,3
ADDI r1,r1,64
MOVI r3,0
LD r15,r0,@hb_ne_ret
RET
hb_ne_bad:
MOVI r3,1
JMP @hb_ne_fail
hb_ne_gate:
MOVI r3,2
JMP @hb_ne_fail
hb_ne_slot:
MOVI r3,6
JMP @hb_ne_fail
hb_ne_stat:
MOVI r3,7
JMP @hb_ne_fail
hb_ne_big:
MOVI r3,8
hb_ne_fail:
MOVI r1,0
LD r15,r0,@hb_ne_ret
RET
hb_ne_ret:
.word 0
hb_ne_src:
.word 0
hb_ne_cnt:
.word 0
hb_ne_dst:
.word 0
hb_ne_pp:
.word 0
hb_ne_i:
.word 0

; hb_n16b_decode: r1 = src byte ptr, r2 = total bytes, r3 = dst word index
;   -> r1 = word count, r2 = result status code, r3 = status.  Fills hb_env_*.
hb_n16b_decode:
ST r15,r0,@hb_nd_ret
ST r1,r0,@hb_nd_src
ST r2,r0,@hb_nd_len
ST r3,r0,@hb_nd_dst
MOVI r4,64
CMP r2,r4
JMP[LT] @hb_nd_len_bad
LDB r4,r1,0
MOVI r5,0x4E
CMP r4,r5
JMP[NE] @hb_nd_magic
LDB r4,r1,1
MOVI r5,0x31
CMP r4,r5
JMP[NE] @hb_nd_magic
LDB r4,r1,2
MOVI r5,0x36
CMP r4,r5
JMP[NE] @hb_nd_magic
LDB r4,r1,3
MOVI r5,0x42
CMP r4,r5
JMP[NE] @hb_nd_magic
LDB r4,r1,4
MOVI r5,2
CMP r4,r5
JMP[NE] @hb_nd_ver
; point-zero identity "BS0:0"
LDB r4,r1,27
MOVI r5,0x42
CMP r4,r5
JMP[NE] @hb_nd_pz
LDB r4,r1,28
MOVI r5,0x53
CMP r4,r5
JMP[NE] @hb_nd_pz
LDB r4,r1,29
MOVI r5,0x30
CMP r4,r5
JMP[NE] @hb_nd_pz
LDB r4,r1,30
MOVI r5,0x3A
CMP r4,r5
JMP[NE] @hb_nd_pz
LDB r4,r1,31
MOVI r5,0x30
CMP r4,r5
JMP[NE] @hb_nd_pz
LDB r4,r1,5
MOVI r5,5
CMP r5,r4
JMP[LT] @hb_nd_slot
ST r4,r0,@hb_env_slot
LDB r4,r1,26
MOVI r5,10
CMP r5,r4
JMP[LT] @hb_nd_stat
ST r4,r0,@hb_env_st
; u64 fields as hi/lo u32
ADDI r1,r1,6
CALL @hb_get32be
ST r1,r0,@hb_env_reqh
LD r1,r0,@hb_nd_src
ADDI r1,r1,10
CALL @hb_get32be
ST r1,r0,@hb_env_reql
LD r1,r0,@hb_nd_src
ADDI r1,r1,14
CALL @hb_get32be
ST r1,r0,@hb_env_eph
LD r1,r0,@hb_nd_src
ADDI r1,r1,18
CALL @hb_get32be
ST r1,r0,@hb_env_epl
LD r1,r0,@hb_nd_src
ADDI r1,r1,22
CALL @hb_get32be
ST r1,r0,@hb_nd_cnt
MOVI r4,4096
CMP r4,r1
JMP[LT] @hb_nd_count
SHL r2,r1,3
ADDI r2,r2,64
LD r3,r0,@hb_nd_len
CMP r2,r3
JMP[NE] @hb_nd_count
; digest check: sha256(header[0..31] || payload) vs bytes 32..63
CALL @hb_sha_init
LD r1,r0,@hb_nd_src
MOVI r2,32
CALL @hb_sha_update
LD r1,r0,@hb_nd_src
ADDI r1,r1,64
LD r2,r0,@hb_nd_cnt
SHL r2,r2,3
CALL @hb_sha_update
MOVI r1,@hb_nd_dig
SHL r1,r1,2
CALL @hb_sha_final
MOVI r5,@hb_nd_dig
SHL r5,r5,2
LD r6,r0,@hb_nd_src
ADDI r6,r6,32
MOVI r7,0
hb_nd_cmp:
ADD r8,r5,r7
LDB r8,r8,0
ADD r9,r6,r7
LDB r9,r9,0
CMP r8,r9
JMP[NE] @hb_nd_digest
ADD r7,r7,r13
MOVI r8,32
CMP r7,r8
JMP[LT] @hb_nd_cmp
; words
LD r3,r0,@hb_nd_src
ADDI r3,r3,64
ST r3,r0,@hb_nd_pp
MOVI r5,0
ST r5,r0,@hb_nd_i
hb_nd_wl:
LD r5,r0,@hb_nd_i
LD r4,r0,@hb_nd_cnt
CMP r5,r4
JMP[GE] @hb_nd_ok
LD r1,r0,@hb_nd_pp
LD r2,r0,@hb_nd_dst
ADD r2,r2,r5
CALL @hb_wire_to_word
CMP r3,r0
JMP[NE] @hb_nd_gate
LD r1,r0,@hb_nd_pp
ADDI r1,r1,8
ST r1,r0,@hb_nd_pp
LD r5,r0,@hb_nd_i
ADD r5,r5,r13
ST r5,r0,@hb_nd_i
JMP @hb_nd_wl
hb_nd_ok:
LD r1,r0,@hb_nd_cnt
LD r2,r0,@hb_env_st
MOVI r3,0
LD r15,r0,@hb_nd_ret
RET
hb_nd_len_bad:
MOVI r3,8
JMP @hb_nd_fail
hb_nd_magic:
MOVI r3,3
JMP @hb_nd_fail
hb_nd_ver:
MOVI r3,4
JMP @hb_nd_fail
hb_nd_pz:
MOVI r3,5
JMP @hb_nd_fail
hb_nd_slot:
MOVI r3,6
JMP @hb_nd_fail
hb_nd_stat:
MOVI r3,7
JMP @hb_nd_fail
hb_nd_count:
MOVI r3,8
JMP @hb_nd_fail
hb_nd_digest:
MOVI r3,9
JMP @hb_nd_fail
hb_nd_gate:
MOVI r3,2
hb_nd_fail:
MOVI r1,0
MOVI r2,0
LD r15,r0,@hb_nd_ret
RET
hb_nd_ret:
.word 0
hb_nd_src:
.word 0
hb_nd_len:
.word 0
hb_nd_dst:
.word 0
hb_nd_cnt:
.word 0
hb_nd_pp:
.word 0
hb_nd_i:
.word 0
hb_nd_dig:
.word 0,0,0,0,0,0,0,0

; ---------------------------------------------------------------- statics
hb_env_slot:
.word 0
hb_env_reqh:
.word 0
hb_env_reql:
.word 0
hb_env_eph:
.word 0
hb_env_epl:
.word 0
hb_env_st:
.word 0
hb_sha_len:
.word 0
hb_sha_bl:
.word 0
hb_heads:
.word 0x7FF1,0xFFF1,0x7FF8,0xFFF8
hb_sha_iv:
.word 0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19
hb_sha_h:
.word 0,0,0,0,0,0,0,0
hb_sv:
.word 0,0,0,0,0,0,0,0
hb_sha_buf:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
hb_sha_w:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
hb_sha_k:
.word 0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5
.word 0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174
.word 0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA
.word 0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967
.word 0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85
.word 0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070
.word 0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3
.word 0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2