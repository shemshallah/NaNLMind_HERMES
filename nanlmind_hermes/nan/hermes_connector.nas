; HERMES native connector - raw NaN16 only, no host math, no host hash.
;
; Role: the only way a Hermes plugin (HW2:0) or an unallocated hardware node
; (HW5+) talks to the bootstrap universe BS0.  BS0 is c(256,2) with a single
; point BS0:0, the NANNET interface AND the runner ring.  BS0:0 is a void
; point: it carries no HW identity of its own.  Every frame leaves through it
; as an N16B v2 envelope (hermes_n16bridge.nas) with slot 0 / epoch 0, and the
; originating HWn identity travels only INSIDE the frame payload (HCF1).
;
; HW map (BS0 namespace, spec/point-map.toml in NaNROUTER):
;   HW0:0 NANNET runner-ring interface   reserved, never a connector origin
;   HW1:0 Android phone                  paired-device-only, own path
;   HW2:0 Hermes plugin #1               ALLOWED
;   HW3:0 NANNET clocked-rezolve app     own path
;   HW4:0 WordNet universe               reserved
;   HW5.. unallocated (5..255)           ALLOWED once leased
; Node point of HWn is n+1 (HW2 -> point 3), bound server-side in the lease.
;
; Uplink: two endpoints reached THROUGH BS0:0 - 0 = NaNROUTER /v1/point0,
; 1 = NANNET portal (Koyeb).  The host only moves bytes; link state and
; endpoint choice are decided here (hc_link_*).
;
; Requires (include before this file): hermes_lease.nas, hermes_nannet_route.nas,
; hermes_n16bridge.nas, hermes_tooltab.nas.  NMP wrappers additionally need
; runner_task.nas (NaN0S) and mth_jahs_gate.nas (NaNROUTER).  Caller keeps
; r0=0 and r13=1; r15 is the link register.
;
; ---- statuses (r3) ----
;  0 ok  1 einval  2 bad NaN head  3..9 as hermes_n16bridge.nas (3 magic/4 ver/
;  5 point-zero/6 slot/7 result status/8 length-count/9 digest)
;  10 hw not allowed as connector origin   11 lease/route rejected (native code
;  in hc_native_code)  12 unknown tool  13 tool not native-verified  14 span
;  violation  15 insufficient workspace  16 seq/replay/in-flight  17 link down
;  18 JAHS seal bad  19 task ledger rejected (native code in hc_native_code)
;  21 connector table full / hw already open  22 builder state  23 wrong op
;  24 wrong destination hw  26 task point != connector point  27 no window
;
; ---- layout (guest word indices) ----
;  window base 0x10000, 8 slots, stride = window words (default 2048)
;  frame builder 0x20000, reply buffer 0x30000   (caller byte buffers: 0x40000+)
;
; HCF1 frame (every word is a NaN16 word with class nibble 0, 32-bit payload)
;  CALL : 0 'HCF1' | 1 op=1<<24|src_hw<<16|dst_hw<<8 | 2 seq | 3 lease slot |
;         4 lease id | 5 universe | 6 generation | 7 owner tag | 8 head |
;         9 tool hash | 10..21 r1..r12 images | 22 nspans(0..4) |
;         23..38 four span descriptors (reg|flags<<8, off, len, 0) |
;         39.. data words of IN spans, in span order
;  REPLY: 0 'HCF1' | 1 op=2<<24|dst_hw<<16 | 2 seq | 3 kernel status |
;         4 tool hash | 5..16 r1..r12 | 17 nout | 18.. nout x (off, len, data)
;  HELLO: 0 'HCF1' | 1 op=3<<24|src_hw<<16 | 2 seq | 3 window words wanted
;  ACK  : 0 'HCF1' | 1 op=4<<24|dst_hw<<16 | 2 seq | 3 window words granted
; Span flags: 1 = IN (copied from local window[off..off+len) into the frame),
; 2 = OUT (reply data lands in local window[off..]).  A span's reg (1..12)
; documents which register image holds the window-relative pointer; 0 = none.

; ======================================================================= generic helpers
hc_zero:
CMP r2,r0
RET[EQ]
ST r0,r1,0
ADDI r1,r1,1
SUB r2,r2,r13
JMP @hc_zero

hc_fill:
CMP r2,r0
RET[EQ]
STW r0,r1,r0
ADDI r1,r1,1
SUB r2,r2,r13
JMP @hc_fill

; r1 = src word idx, r2 = dst word idx, r3 = count   (STW nibble 0)
hc_copyw:
CMP r3,r0
RET[EQ]
LD r4,r1,0
STW r4,r2,r0
ADDI r1,r1,1
ADDI r2,r2,1
SUB r3,r3,r13
JMP @hc_copyw

; r1 = value, r2 = offset into the frame builder buffer. clobbers r3
hc_fbw:
MOVI r3,@hc_cfg
LD r3,r3,1
ADD r3,r3,r2
STW r1,r3,r0
RET

; r2 = offset into the reply buffer -> r1 = payload. clobbers r3
hc_rbr:
MOVI r3,@hc_cfg
LD r3,r3,2
ADD r3,r3,r2
LD r1,r3,0
RET

; r1 = cslot -> r1 = record addr, r3 = status (cslot < 8 and open)
hc_rec:
MOVI r2,8
CMP r1,r2
JMP[GE] @hc_rec_bad
MOVI r2,12
MUL r2,r1,r2
MOVI r4,@hc_ctab
ADD r1,r2,r4
LD r2,r1,0
CMP r2,r13
JMP[NE] @hc_rec_bad
MOVI r3,0
RET
hc_rec_bad:
MOVI r3,1
RET

; r1 = cslot -> r1 = local window base word index (no open check)
hc_win_base:
MOVI r2,@hc_cfg
LD r3,r2,3
MUL r3,r1,r3
LD r4,r2,0
ADD r1,r3,r4
RET

; ======================================================================= init / config
hc_init:
ST r15,r0,@hc_init_ret
MOVI r1,0
MOVHI r1,1
MOVI r2,@hc_cfg
ST r1,r2,0
MOVI r1,0
MOVHI r1,2
ST r1,r2,1
MOVI r1,0
MOVHI r1,3
ST r1,r2,2
MOVI r1,2048
ST r1,r2,3
MOVI r1,@hc_ctab
MOVI r2,96
CALL @hc_zero
MOVI r1,@hc_ltab
MOVI r2,8
CALL @hc_zero
MOVI r1,@hc_bst
MOVI r2,8
CALL @hc_zero
MOVI r1,@hc_outregs
MOVI r2,12
CALL @hc_zero
MOVI r3,0
LD r15,r0,@hc_init_ret
RET
hc_init_ret:
.word 0

; hc_set_window: r1 = words per connector window (16..2048), before any hc_open
hc_set_window:
MOVI r2,16
CMP r1,r2
JMP[LT] @hc_sw_bad
MOVI r2,2048
CMP r2,r1
JMP[LT] @hc_sw_bad
MOVI r2,@hc_cfg
ST r1,r2,3
MOVI r3,0
RET
hc_sw_bad:
MOVI r3,1
RET

; ======================================================================= HW allocation policy
; hc_hw_class: r1 = hw -> r1 = class (0 ring,1 phone,2 hermes,3 clocked,4 wordnet,5 free), r3
hc_hw_class:
MOVI r2,255
CMP r1,r2
JMP[GT] @hc_hwc_bad
MOVI r2,5
CMP r1,r2
JMP[LT] @hc_hwc_ok
MOVI r1,5
hc_hwc_ok:
MOVI r3,0
RET
hc_hwc_bad:
MOVI r1,0
MOVI r3,1
RET

; hc_point_of_hw: r1 = hw -> r1 = node point (hw+1), r3
hc_point_of_hw:
MOVI r2,255
CMP r1,r2
JMP[GT] @hc_hwc_bad
ADDI r1,r1,1
MOVI r3,0
RET

; hc_hw_may_connect: r1 = hw -> r3 = 0 only for HW2 or HW5..255
hc_hw_may_connect:
ST r15,r0,@hc_mc_ret
CALL @hc_hw_class
CMP r3,r0
JMP[NE] @hc_mc_done
MOVI r2,2
CMP r1,r2
JMP[EQ] @hc_mc_ok
MOVI r2,5
CMP r1,r2
JMP[EQ] @hc_mc_ok
MOVI r3,10
JMP @hc_mc_done
hc_mc_ok:
MOVI r3,0
hc_mc_done:
LD r15,r0,@hc_mc_ret
RET
hc_mc_ret:
.word 0

; ======================================================================= connector registry
; hc_open: r1=hw r2=lease slot r3=lease id r4=universe r5=generation r6=owner tag r7=now
;   -> r1 = connector slot (0..7), r3 status.  Binds node point hw+1 through nhl_nannet_route.
hc_open:
ST r15,r0,@hc_open_ret
ST r1,r0,@hc_o_hw
ST r2,r0,@hc_o_lslot
ST r3,r0,@hc_o_lid
ST r4,r0,@hc_o_uni
ST r5,r0,@hc_o_gen
ST r6,r0,@hc_o_own
ST r7,r0,@hc_o_now
CALL @hc_hw_may_connect
CMP r3,r0
JMP[NE] @hc_open_out
LD r1,r0,@hc_o_lslot
LD r2,r0,@hc_o_lid
LD r3,r0,@hc_o_uni
LD r4,r0,@hc_o_gen
LD r5,r0,@hc_o_own
MOVI r6,0
LD r7,r0,@hc_o_now
LD r8,r0,@hc_o_hw
ADDI r8,r8,1
CALL @nhl_nannet_route
CMP r3,r0
JMP[EQ] @hc_open_alloc
ST r3,r0,@hc_native_code
MOVI r3,11
JMP @hc_open_out
hc_open_alloc:
MOVI r6,0
MOVI r10,8
hc_open_scan:
MOVI r2,8
CMP r6,r2
JMP[GE] @hc_open_pick
MOVI r2,12
MUL r2,r6,r2
MOVI r4,@hc_ctab
ADD r4,r4,r2
LD r5,r4,0
CMP r5,r13
JMP[NE] @hc_open_free
LD r5,r4,1
LD r7,r0,@hc_o_hw
CMP r5,r7
JMP[EQ] @hc_open_busy
ADDI r6,r6,1
JMP @hc_open_scan
hc_open_free:
MOVI r2,8
CMP r10,r2
JMP[NE] @hc_open_next
ADD r10,r6,r0
hc_open_next:
ADDI r6,r6,1
JMP @hc_open_scan
hc_open_busy:
MOVI r3,21
JMP @hc_open_out
hc_open_pick:
MOVI r2,8
CMP r10,r2
JMP[EQ] @hc_open_busy
MOVI r2,12
MUL r2,r10,r2
MOVI r4,@hc_ctab
ADD r4,r4,r2
MOVI r5,1
ST r5,r4,0
LD r5,r0,@hc_o_hw
ST r5,r4,1
ADDI r5,r5,1
ST r5,r4,2
LD r5,r0,@hc_o_lslot
ST r5,r4,3
LD r5,r0,@hc_o_lid
ST r5,r4,4
LD r5,r0,@hc_o_uni
ST r5,r4,5
LD r5,r0,@hc_o_gen
ST r5,r4,6
LD r5,r0,@hc_o_own
ST r5,r4,7
MOVI r5,1
ST r5,r4,8
ST r0,r4,9
ST r0,r4,10
MOVI r5,@hc_cfg
LD r5,r5,3
ST r5,r4,11
ADD r1,r10,r0
MOVI r3,0
hc_open_out:
LD r15,r0,@hc_open_ret
RET
hc_open_ret:
.word 0
hc_o_hw:
.word 0
hc_o_lslot:
.word 0
hc_o_lid:
.word 0
hc_o_uni:
.word 0
hc_o_gen:
.word 0
hc_o_own:
.word 0
hc_o_now:
.word 0
hc_native_code:
.word 0

; hc_close: r1 = cslot -> r3
hc_close:
ST r15,r0,@hc_cl_ret
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_cl_out
MOVI r2,12
hc_cl_z:
ST r0,r1,0
ADDI r1,r1,1
SUB r2,r2,r13
JMP[NE] @hc_cl_z
MOVI r3,0
hc_cl_out:
LD r15,r0,@hc_cl_ret
RET
hc_cl_ret:
.word 0

; hc_seq_set: r1 = cslot, r2 = next request sequence (>=1) -> r3.  The host persists seq.
hc_seq_set:
ST r15,r0,@hc_ss_ret
ST r2,r0,@hc_ss_seq
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_ss_out
LD r2,r0,@hc_ss_seq
CMP r2,r0
JMP[EQ] @hc_ss_bad
ST r2,r1,8
MOVI r3,0
JMP @hc_ss_out
hc_ss_bad:
MOVI r3,1
hc_ss_out:
LD r15,r0,@hc_ss_ret
RET
hc_ss_ret:
.word 0
hc_ss_seq:
.word 0

; ======================================================================= uplink state (through BS0:0)
; endpoints: 0 = NaNROUTER /v1/point0, 1 = NANNET portal (Koyeb).
; state 0 down, 1 probing, 2 up, 3 failed.  4 words per endpoint: state, fails, ok, last event.
; events: 1 probe_sent 2 probe_ok 3 probe_fail 4 request_ok 5 request_fail 6 reset
; hc_link_event: r1 = endpoint, r2 = event -> r1 = new state, r3
hc_link_event:
MOVI r4,2
CMP r1,r4
JMP[GE] @hc_le_bad
CMP r2,r0
JMP[EQ] @hc_le_bad
MOVI r4,6
CMP r4,r2
JMP[LT] @hc_le_bad
SHL r5,r1,2
MOVI r6,@hc_ltab
ADD r5,r5,r6
ST r2,r5,3
LD r6,r5,0
MOVI r4,1
CMP r2,r4
JMP[EQ] @hc_le_psent
MOVI r4,2
CMP r2,r4
JMP[EQ] @hc_le_pok
MOVI r4,3
CMP r2,r4
JMP[EQ] @hc_le_pfail
MOVI r4,4
CMP r2,r4
JMP[EQ] @hc_le_rok
MOVI r4,5
CMP r2,r4
JMP[EQ] @hc_le_rfail
ST r0,r5,0
ST r0,r5,1
MOVI r1,0
MOVI r3,0
RET
hc_le_psent:
MOVI r4,2
CMP r6,r4
JMP[EQ] @hc_le_keep
MOVI r1,1
ST r1,r5,0
JMP @hc_le_done
hc_le_keep:
ADD r1,r6,r0
JMP @hc_le_done
hc_le_pok:
MOVI r1,2
ST r1,r5,0
ST r0,r5,1
LD r4,r5,2
ADD r4,r4,r13
ST r4,r5,2
JMP @hc_le_done
hc_le_pfail:
LD r4,r5,1
ADD r4,r4,r13
ST r4,r5,1
MOVI r1,0
MOVI r7,3
CMP r4,r7
JMP[LT] @hc_le_setstate
MOVI r1,3
hc_le_setstate:
ST r1,r5,0
JMP @hc_le_done
hc_le_rok:
MOVI r4,2
CMP r6,r4
JMP[NE] @hc_le_state
ST r0,r5,1
LD r4,r5,2
ADD r4,r4,r13
ST r4,r5,2
MOVI r1,2
JMP @hc_le_done
hc_le_rfail:
LD r4,r5,1
ADD r4,r4,r13
ST r4,r5,1
ADD r1,r6,r0
MOVI r7,3
CMP r4,r7
JMP[LT] @hc_le_rkeep
MOVI r1,3
ST r1,r5,0
JMP @hc_le_done
hc_le_rkeep:
JMP @hc_le_done
hc_le_state:
MOVI r1,0
MOVI r3,22
RET
hc_le_done:
MOVI r3,0
RET
hc_le_bad:
MOVI r1,0
MOVI r3,1
RET

; hc_link_select: -> r1 = endpoint (0 router preferred, else 1 portal), r3 (17 if none up)
hc_link_select:
MOVI r4,@hc_ltab
LD r5,r4,0
MOVI r6,2
CMP r5,r6
JMP[EQ] @hc_ls_router
LD r5,r4,4
CMP r5,r6
JMP[EQ] @hc_ls_portal
MOVI r1,0
MOVI r3,17
RET
hc_ls_router:
MOVI r1,0
MOVI r3,0
RET
hc_ls_portal:
MOVI r1,1
MOVI r3,0
RET

; ======================================================================= tool table
; hc_tool_lookup: r1 = FNV-1a/32(kernel_id) -> r1 index, r2 workspace words, r3 status
hc_tool_lookup:
LD r4,r0,@hc_tool_count
MOVI r5,@hc_tooltab
MOVI r6,0
hc_tl_loop:
CMP r6,r4
JMP[GE] @hc_tl_none
LD r7,r5,0
CMP r7,r1
JMP[EQ] @hc_tl_found
ADDI r5,r5,4
ADDI r6,r6,1
JMP @hc_tl_loop
hc_tl_none:
MOVI r3,12
RET
hc_tl_found:
LD r2,r5,1
LD r8,r5,2
BOOL.8 r9,r8,r13
CMP r9,r0
JMP[EQ] @hc_tl_unver
ADD r1,r6,r0
MOVI r3,0
RET
hc_tl_unver:
MOVI r3,13
RET

; hc_tool_steps: r1 = index -> r1 = fixture step hint, r3
hc_tool_steps:
LD r4,r0,@hc_tool_count
CMP r1,r4
JMP[GE] @hc_ts_bad
SHL r1,r1,2
MOVI r5,@hc_tooltab
ADD r1,r1,r5
LD r1,r1,3
MOVI r3,0
RET
hc_ts_bad:
MOVI r1,0
MOVI r3,1
RET

; hc_fnv1a: r1 = byte ptr, r2 = len -> r1 = FNV-1a/32
hc_fnv1a:
MOVI r3,0x9DC5
MOVHI r3,0x811C
MOVI r4,0x0193
MOVHI r4,0x0100
hc_fnv_l:
CMP r2,r0
JMP[EQ] @hc_fnv_d
LDB r5,r1,0
BOOL.6 r3,r3,r5
MUL r3,r3,r4
ADDI r1,r1,1
SUB r2,r2,r13
JMP @hc_fnv_l
hc_fnv_d:
ADD r1,r3,r0
MOVI r3,0
RET

; ======================================================================= universal tool call (builder)
; builder state hc_bst: 0 active, 1 nspan, 2 data words, 3 seq, 4 cslot, 5 record addr
; hc_call_begin: r1=cslot r2=tool hash r3=head(0..254) r4=trusted now -> r1 = seq, r3
hc_call_begin:
ST r15,r0,@hc_cb_ret
ST r1,r0,@hc_cb_cs
ST r2,r0,@hc_cb_hash
ST r3,r0,@hc_cb_head
ST r4,r0,@hc_cb_now
MOVI r5,@hc_bst
ST r0,r5,0
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_cb_out
ST r1,r0,@hc_cb_rec
LD r2,r0,@hc_cb_head
MOVI r4,255
CMP r2,r4
JMP[GE] @hc_cb_inval
LD r2,r1,9
CMP r2,r0
JMP[NE] @hc_cb_seq
LD r1,r0,@hc_cb_hash
CALL @hc_tool_lookup
CMP r3,r0
JMP[NE] @hc_cb_out
LD r9,r0,@hc_cb_rec
LD r5,r9,11
CMP r5,r2
JMP[LT] @hc_cb_ws
LD r1,r9,3
LD r2,r9,4
LD r3,r9,5
LD r4,r9,6
LD r5,r9,7
LD r6,r0,@hc_cb_head
LD r7,r0,@hc_cb_now
LD r8,r9,2
CALL @nhl_nannet_route
CMP r3,r0
JMP[EQ] @hc_cb_route_ok
ST r3,r0,@hc_native_code
MOVI r3,11
JMP @hc_cb_out
hc_cb_route_ok:
LD r9,r0,@hc_cb_rec
LD r5,r9,8
ST r5,r9,9
ADDI r6,r5,1
ST r6,r9,8
LD r7,r0,@hc_cb_hash
ST r7,r9,10
MOVI r8,@hc_bst
MOVI r6,1
ST r6,r8,0
ST r0,r8,1
ST r0,r8,2
ST r5,r8,3
LD r6,r0,@hc_cb_cs
ST r6,r8,4
ST r9,r8,5
MOVI r1,@hc_cfg
LD r1,r1,1
MOVI r2,39
CALL @hc_fill
MOVI r1,0x4631
MOVHI r1,0x4843
MOVI r2,0
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r2,r9,1
SHL r2,r2,16
MOVI r1,0
MOVHI r1,0x0100
BOOL.14 r1,r1,r2
MOVI r2,1
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
MOVI r8,@hc_bst
LD r1,r8,3
MOVI r2,2
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r1,r9,3
MOVI r2,3
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r1,r9,4
MOVI r2,4
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r1,r9,5
MOVI r2,5
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r1,r9,6
MOVI r2,6
CALL @hc_fbw
LD r9,r0,@hc_cb_rec
LD r1,r9,7
MOVI r2,7
CALL @hc_fbw
LD r1,r0,@hc_cb_head
MOVI r2,8
CALL @hc_fbw
LD r1,r0,@hc_cb_hash
MOVI r2,9
CALL @hc_fbw
MOVI r8,@hc_bst
LD r1,r8,3
MOVI r3,0
JMP @hc_cb_out
hc_cb_inval:
MOVI r3,1
JMP @hc_cb_out
hc_cb_seq:
MOVI r3,16
JMP @hc_cb_out
hc_cb_ws:
MOVI r3,15
hc_cb_out:
LD r15,r0,@hc_cb_ret
RET
hc_cb_ret:
.word 0
hc_cb_cs:
.word 0
hc_cb_hash:
.word 0
hc_cb_head:
.word 0
hc_cb_now:
.word 0
hc_cb_rec:
.word 0

; hc_call_reg: r1 = register number 1..12, r2 = register image value -> r3
hc_call_reg:
ST r15,r0,@hc_cr_ret
MOVI r3,@hc_bst
LD r3,r3,0
CMP r3,r13
JMP[NE] @hc_cr_state
CMP r1,r0
JMP[EQ] @hc_cr_bad
MOVI r3,12
CMP r3,r1
JMP[LT] @hc_cr_bad
ADDI r3,r1,9
ADD r1,r2,r0
ADD r2,r3,r0
CALL @hc_fbw
MOVI r3,0
JMP @hc_cr_out
hc_cr_state:
MOVI r3,22
JMP @hc_cr_out
hc_cr_bad:
MOVI r3,1
hc_cr_out:
LD r15,r0,@hc_cr_ret
RET
hc_cr_ret:
.word 0

; hc_call_span: r1 = reg (0..12), r2 = window offset, r3 = length, r4 = flags (1 IN, 2 OUT, 3 both)
;   Bounds: len>=1, off+len <= connector window (no wrap), <=4 spans, IN data <= window.
hc_call_span:
ST r15,r0,@hc_sp_ret
ST r1,r0,@hc_sp_reg
ST r2,r0,@hc_sp_off
ST r3,r0,@hc_sp_len
ST r4,r0,@hc_sp_fl
MOVI r5,@hc_bst
LD r6,r5,0
CMP r6,r13
JMP[NE] @hc_sp_state
LD r6,r5,1
MOVI r7,4
CMP r6,r7
JMP[GE] @hc_sp_viol
MOVI r7,12
CMP r7,r1
JMP[LT] @hc_sp_viol
CMP r4,r0
JMP[EQ] @hc_sp_viol
MOVI r7,3
CMP r7,r4
JMP[LT] @hc_sp_viol
CMP r3,r0
JMP[EQ] @hc_sp_viol
LD r8,r5,5
LD r8,r8,11
CMP r8,r3
JMP[LT] @hc_sp_viol
SUB r9,r8,r3
CMP r9,r2
JMP[LT] @hc_sp_viol
BOOL.8 r9,r4,r13
CMP r9,r0
JMP[EQ] @hc_sp_desc
LD r9,r5,2
ADD r9,r9,r3
CMP r8,r9
JMP[LT] @hc_sp_viol
hc_sp_desc:
LD r6,r5,1
SHL r6,r6,2
ADDI r6,r6,23
ST r6,r0,@hc_sp_slot
LD r1,r0,@hc_sp_fl
SHL r1,r1,8
LD r2,r0,@hc_sp_reg
BOOL.14 r1,r1,r2
LD r2,r0,@hc_sp_slot
CALL @hc_fbw
LD r1,r0,@hc_sp_off
LD r2,r0,@hc_sp_slot
ADD r2,r2,r13
CALL @hc_fbw
LD r1,r0,@hc_sp_len
LD r2,r0,@hc_sp_slot
ADDI r2,r2,2
CALL @hc_fbw
MOVI r1,0
LD r2,r0,@hc_sp_slot
ADDI r2,r2,3
CALL @hc_fbw
LD r4,r0,@hc_sp_fl
BOOL.8 r9,r4,r13
CMP r9,r0
JMP[EQ] @hc_sp_done
MOVI r5,@hc_bst
LD r1,r5,4
CALL @hc_win_base
LD r2,r0,@hc_sp_off
ADD r1,r1,r2
MOVI r5,@hc_bst
LD r6,r5,2
MOVI r7,@hc_cfg
LD r7,r7,1
ADD r2,r7,r6
ADDI r2,r2,39
LD r3,r0,@hc_sp_len
CALL @hc_copyw
MOVI r5,@hc_bst
LD r6,r5,2
LD r7,r0,@hc_sp_len
ADD r6,r6,r7
ST r6,r5,2
hc_sp_done:
MOVI r5,@hc_bst
LD r6,r5,1
ADD r6,r6,r13
ST r6,r5,1
MOVI r3,0
JMP @hc_sp_out
hc_sp_state:
MOVI r3,22
JMP @hc_sp_out
hc_sp_viol:
MOVI r3,14
hc_sp_out:
LD r15,r0,@hc_sp_ret
RET
hc_sp_ret:
.word 0
hc_sp_reg:
.word 0
hc_sp_off:
.word 0
hc_sp_len:
.word 0
hc_sp_fl:
.word 0
hc_sp_slot:
.word 0

; hc_call_finish: r1 = dst byte ptr (4-aligned) -> r1 = total bytes, r3 status
hc_call_finish:
ST r15,r0,@hc_cf_ret
ST r1,r0,@hc_cf_dst
MOVI r5,@hc_bst
LD r6,r5,0
CMP r6,r13
JMP[NE] @hc_cf_state
LD r1,r5,1
MOVI r2,22
CALL @hc_fbw
MOVI r5,@hc_bst
LD r6,r5,2
ADDI r6,r6,39
ST r6,r0,@hc_cf_cnt
LD r3,r5,3
MOVI r1,0
MOVI r2,0
MOVI r4,0
MOVI r5,0
MOVI r6,0
CALL @hb_env_set_req
MOVI r1,@hc_cfg
LD r1,r1,1
LD r2,r0,@hc_cf_cnt
LD r3,r0,@hc_cf_dst
CALL @hb_n16b_encode
ST r3,r0,@hc_cf_st
MOVI r5,@hc_bst
ST r0,r5,0
LD r3,r0,@hc_cf_st
JMP @hc_cf_out
hc_cf_state:
MOVI r3,22
hc_cf_out:
LD r15,r0,@hc_cf_ret
RET
hc_cf_ret:
.word 0
hc_cf_dst:
.word 0
hc_cf_cnt:
.word 0
hc_cf_st:
.word 0

; hb_env_set wrapper: r3 = request id low word, others zero (slot 0, epoch 0, status ok)
hb_env_set_req:
ADD r3,r3,r0
MOVI r1,0
MOVI r2,0
MOVI r4,0
MOVI r5,0
MOVI r6,0
JMP @hb_env_set

; ======================================================================= reply parsing
; hc_reply_parse: r1 = src bytes, r2 = total bytes, r3 = cslot
;   -> r1 = kernel status (RB[3]), r2 = envelope result status, r3 = status.
;   OUT spans are copied into the local window; r1..r12 images land in hc_outregs.
hc_reply_parse:
ST r15,r0,@hc_rp_ret
ST r1,r0,@hc_rp_src
ST r2,r0,@hc_rp_tot
ST r3,r0,@hc_rp_cs
ADD r1,r3,r0
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_rp_out
ST r1,r0,@hc_rp_rec
LD r2,r1,9
CMP r2,r0
JMP[EQ] @hc_rp_seq
LD r1,r0,@hc_rp_src
LD r2,r0,@hc_rp_tot
MOVI r3,@hc_cfg
LD r3,r3,2
CALL @hb_n16b_decode
CMP r3,r0
JMP[NE] @hc_rp_out
ST r1,r0,@hc_rp_cnt
ST r2,r0,@hc_rp_env
MOVI r4,18
CMP r1,r4
JMP[LT] @hc_rp_len
MOVI r2,0
CALL @hc_rbr
MOVI r2,0x4631
MOVHI r2,0x4843
CMP r1,r2
JMP[NE] @hc_rp_magic
MOVI r2,1
CALL @hc_rbr
SHR r4,r1,24
MOVI r5,2
CMP r4,r5
JMP[NE] @hc_rp_op
SHR r4,r1,16
MOVI r5,255
BOOL.8 r4,r4,r5
LD r6,r0,@hc_rp_rec
LD r6,r6,1
CMP r4,r6
JMP[NE] @hc_rp_dst
MOVI r2,2
CALL @hc_rbr
LD r6,r0,@hc_rp_rec
LD r7,r6,9
CMP r1,r7
JMP[NE] @hc_rp_seq
MOVI r2,4
CALL @hc_rbr
LD r6,r0,@hc_rp_rec
LD r7,r6,10
CMP r1,r7
JMP[NE] @hc_rp_seq
MOVI r2,17
CALL @hc_rbr
ST r1,r0,@hc_rp_nout
MOVI r4,4
CMP r4,r1
JMP[LT] @hc_rp_span
MOVI r6,18
ST r6,r0,@hc_rp_pos
ST r0,r0,@hc_rp_i
hc_rp_sl:
LD r6,r0,@hc_rp_i
LD r7,r0,@hc_rp_nout
CMP r6,r7
JMP[GE] @hc_rp_spans_done
LD r2,r0,@hc_rp_pos
CALL @hc_rbr
ST r1,r0,@hc_rp_off
LD r2,r0,@hc_rp_pos
ADD r2,r2,r13
CALL @hc_rbr
ST r1,r0,@hc_rp_slen
LD r6,r0,@hc_rp_rec
LD r8,r6,11
LD r1,r0,@hc_rp_slen
CMP r1,r0
JMP[EQ] @hc_rp_span
CMP r8,r1
JMP[LT] @hc_rp_span
SUB r9,r8,r1
LD r2,r0,@hc_rp_off
CMP r9,r2
JMP[LT] @hc_rp_span
LD r6,r0,@hc_rp_pos
ADDI r6,r6,2
LD r7,r0,@hc_rp_slen
ADD r6,r6,r7
LD r7,r0,@hc_rp_cnt
CMP r7,r6
JMP[LT] @hc_rp_len
LD r1,r0,@hc_rp_cs
CALL @hc_win_base
LD r2,r0,@hc_rp_off
ADD r2,r1,r2
MOVI r1,@hc_cfg
LD r1,r1,2
LD r6,r0,@hc_rp_pos
ADD r1,r1,r6
ADDI r1,r1,2
LD r3,r0,@hc_rp_slen
CALL @hc_copyw
LD r6,r0,@hc_rp_pos
ADDI r6,r6,2
LD r7,r0,@hc_rp_slen
ADD r6,r6,r7
ST r6,r0,@hc_rp_pos
LD r6,r0,@hc_rp_i
ADD r6,r6,r13
ST r6,r0,@hc_rp_i
JMP @hc_rp_sl
hc_rp_spans_done:
LD r6,r0,@hc_rp_pos
LD r7,r0,@hc_rp_cnt
CMP r6,r7
JMP[NE] @hc_rp_len
MOVI r6,0
hc_rp_rl:
MOVI r7,12
CMP r6,r7
JMP[GE] @hc_rp_final
ADDI r2,r6,5
CALL @hc_rbr
MOVI r7,@hc_outregs
ADD r7,r7,r6
ST r1,r7,0
ADD r6,r6,r13
JMP @hc_rp_rl
hc_rp_final:
LD r6,r0,@hc_rp_rec
ST r0,r6,9
ST r0,r6,10
MOVI r2,3
CALL @hc_rbr
LD r2,r0,@hc_rp_env
MOVI r3,0
JMP @hc_rp_out
hc_rp_len:
MOVI r3,8
JMP @hc_rp_out
hc_rp_magic:
MOVI r3,3
JMP @hc_rp_out
hc_rp_op:
MOVI r3,23
JMP @hc_rp_out
hc_rp_dst:
MOVI r3,24
JMP @hc_rp_out
hc_rp_seq:
MOVI r3,16
JMP @hc_rp_out
hc_rp_span:
MOVI r3,14
hc_rp_out:
LD r15,r0,@hc_rp_ret
RET
hc_rp_ret:
.word 0
hc_rp_src:
.word 0
hc_rp_tot:
.word 0
hc_rp_cs:
.word 0
hc_rp_rec:
.word 0
hc_rp_cnt:
.word 0
hc_rp_env:
.word 0
hc_rp_nout:
.word 0
hc_rp_pos:
.word 0
hc_rp_i:
.word 0
hc_rp_off:
.word 0
hc_rp_slen:
.word 0

; ======================================================================= HELLO / window negotiation
; hc_hello_build: r1 = cslot, r2 = dst byte ptr -> r1 = total bytes, r3
hc_hello_build:
ST r15,r0,@hc_hb_ret
ST r2,r0,@hc_hb_dst
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_hb_out
ST r1,r0,@hc_hb_rec
LD r2,r1,9
CMP r2,r0
JMP[NE] @hc_hb_seq
LD r5,r1,8
ST r5,r1,9
ST r0,r1,10
ADDI r6,r5,1
ST r6,r1,8
ST r5,r0,@hc_hb_seqv
MOVI r1,@hc_cfg
LD r1,r1,1
MOVI r2,4
CALL @hc_fill
MOVI r1,0x4631
MOVHI r1,0x4843
MOVI r2,0
CALL @hc_fbw
LD r9,r0,@hc_hb_rec
LD r2,r9,1
SHL r2,r2,16
MOVI r1,0
MOVHI r1,0x0300
BOOL.14 r1,r1,r2
MOVI r2,1
CALL @hc_fbw
LD r1,r0,@hc_hb_seqv
MOVI r2,2
CALL @hc_fbw
LD r9,r0,@hc_hb_rec
LD r1,r9,11
MOVI r2,3
CALL @hc_fbw
LD r3,r0,@hc_hb_seqv
CALL @hb_env_set_req
MOVI r1,@hc_cfg
LD r1,r1,1
MOVI r2,4
LD r3,r0,@hc_hb_dst
CALL @hb_n16b_encode
JMP @hc_hb_out
hc_hb_seq:
MOVI r3,16
hc_hb_out:
LD r15,r0,@hc_hb_ret
RET
hc_hb_ret:
.word 0
hc_hb_dst:
.word 0
hc_hb_rec:
.word 0
hc_hb_seqv:
.word 0

; hc_hello_parse: r1 = src bytes, r2 = total, r3 = cslot -> r1 = granted window words, r3
hc_hello_parse:
ST r15,r0,@hc_hp_ret
ST r1,r0,@hc_hp_src
ST r2,r0,@hc_hp_tot
ST r3,r0,@hc_hp_cs
ADD r1,r3,r0
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_hp_out
ST r1,r0,@hc_hp_rec
LD r2,r1,9
CMP r2,r0
JMP[EQ] @hc_hp_seq
LD r1,r0,@hc_hp_src
LD r2,r0,@hc_hp_tot
MOVI r3,@hc_cfg
LD r3,r3,2
CALL @hb_n16b_decode
CMP r3,r0
JMP[NE] @hc_hp_out
MOVI r4,4
CMP r1,r4
JMP[NE] @hc_hp_len
MOVI r2,0
CALL @hc_rbr
MOVI r2,0x4631
MOVHI r2,0x4843
CMP r1,r2
JMP[NE] @hc_hp_magic
MOVI r2,1
CALL @hc_rbr
SHR r4,r1,24
MOVI r5,4
CMP r4,r5
JMP[NE] @hc_hp_op
SHR r4,r1,16
MOVI r5,255
BOOL.8 r4,r4,r5
LD r6,r0,@hc_hp_rec
LD r6,r6,1
CMP r4,r6
JMP[NE] @hc_hp_dst
MOVI r2,2
CALL @hc_rbr
LD r6,r0,@hc_hp_rec
LD r7,r6,9
CMP r1,r7
JMP[NE] @hc_hp_seq
MOVI r2,3
CALL @hc_rbr
CMP r1,r0
JMP[EQ] @hc_hp_nowin
LD r6,r0,@hc_hp_rec
LD r7,r6,11
CMP r1,r7
JMP[GE] @hc_hp_keep
ST r1,r6,11
hc_hp_keep:
ST r0,r6,9
LD r1,r6,11
MOVI r3,0
JMP @hc_hp_out
hc_hp_len:
MOVI r3,8
JMP @hc_hp_out
hc_hp_magic:
MOVI r3,3
JMP @hc_hp_out
hc_hp_op:
MOVI r3,23
JMP @hc_hp_out
hc_hp_dst:
MOVI r3,24
JMP @hc_hp_out
hc_hp_nowin:
MOVI r3,27
JMP @hc_hp_out
hc_hp_seq:
MOVI r3,16
hc_hp_out:
LD r15,r0,@hc_hp_ret
RET
hc_hp_ret:
.word 0
hc_hp_src:
.word 0
hc_hp_tot:
.word 0
hc_hp_cs:
.word 0
hc_hp_rec:
.word 0

; ======================================================================= NMP lifecycle (gated by the JAHS seal)
; Uses the existing runner_task ledger (NaN0S runner_task.nas).  Every verb needs the seal
; word 0x4A414853 ("JAHS") and the task's logical point must equal the connector's node point.
; hc_nmp_find: r1 = task id -> r1 = record addr (0 if none)
hc_nmp_find:
MOVI r4,@nmr_task_table
MOVI r5,16
MOVI r6,0
hc_nf_l:
LD r7,r4,0
CMP r7,r1
JMP[EQ] @hc_nf_hit
ADDI r4,r4,15
ADDI r6,r6,1
CMP r6,r5
JMP[LT] @hc_nf_l
MOVI r1,0
RET
hc_nf_hit:
ADD r1,r4,r0
RET

; hc_nmp_gate: r1 = cslot, r2 = seal -> r1 = record addr, r3 (0, 1, 18)
hc_nmp_gate:
ST r15,r0,@hc_ng_ret
ST r2,r0,@hc_ng_seal
CALL @hc_rec
CMP r3,r0
JMP[NE] @hc_ng_out
ST r1,r0,@hc_ng_rec
LD r1,r0,@hc_ng_seal
CALL @mth_jahs_gate
CMP r1,r0
JMP[EQ] @hc_ng_seal_bad
LD r1,r0,@hc_ng_rec
MOVI r3,0
JMP @hc_ng_out
hc_ng_seal_bad:
MOVI r3,18
hc_ng_out:
LD r15,r0,@hc_ng_ret
RET
hc_ng_ret:
.word 0
hc_ng_seal:
.word 0
hc_ng_rec:
.word 0

; hc_nmp_submit: r1 = cslot, r2 = JAHS seal, r3 = descriptor word idx (11 words, runner_task fields 0..10)
;   -> r1 = ledger slot, r3.  desc[4] (logical point) must equal the connector's node point.
hc_nmp_submit:
ST r15,r0,@hc_ns_ret
ST r3,r0,@hc_ns_desc
CALL @hc_nmp_gate
CMP r3,r0
JMP[NE] @hc_ns_out
LD r12,r0,@hc_ns_desc
LD r5,r12,4
LD r6,r1,2
CMP r5,r6
JMP[NE] @hc_ns_pt
LD r1,r12,0
LD r2,r12,1
LD r3,r12,2
LD r4,r12,3
LD r5,r12,4
LD r6,r12,5
LD r7,r12,6
LD r8,r12,7
LD r9,r12,8
LD r10,r12,9
LD r11,r12,10
CALL @nmr_task_submit
CMP r3,r0
JMP[EQ] @hc_ns_out
ST r3,r0,@hc_native_code
MOVI r3,19
JMP @hc_ns_out
hc_ns_pt:
MOVI r3,26
hc_ns_out:
LD r15,r0,@hc_ns_ret
RET
hc_ns_ret:
.word 0
hc_ns_desc:
.word 0

; hc_nmp_claim: r1 = cslot, r2 = seal, r3 = task id, r4 = runner id, r5 = epoch -> r3
hc_nmp_claim:
ST r15,r0,@hc_nc_ret
ST r3,r0,@hc_nc_id
ST r4,r0,@hc_nc_run
ST r5,r0,@hc_nc_ep
CALL @hc_nmp_gate
CMP r3,r0
JMP[NE] @hc_nc_out
ST r1,r0,@hc_nc_rec
LD r1,r0,@hc_nc_id
CALL @hc_nmp_find
CMP r1,r0
JMP[EQ] @hc_nc_none
LD r5,r1,4
LD r6,r0,@hc_nc_rec
LD r6,r6,2
CMP r5,r6
JMP[NE] @hc_nc_pt
LD r1,r0,@hc_nc_id
LD r2,r0,@hc_nc_run
LD r3,r0,@hc_nc_ep
CALL @nmr_task_claim
CMP r3,r0
JMP[EQ] @hc_nc_out
ST r3,r0,@hc_native_code
MOVI r3,19
JMP @hc_nc_out
hc_nc_none:
MOVI r3,19
MOVI r4,1
ST r4,r0,@hc_native_code
JMP @hc_nc_out
hc_nc_pt:
MOVI r3,26
hc_nc_out:
LD r15,r0,@hc_nc_ret
RET
hc_nc_ret:
.word 0
hc_nc_id:
.word 0
hc_nc_run:
.word 0
hc_nc_ep:
.word 0
hc_nc_rec:
.word 0

; hc_nmp_complete: r1 = cslot, r2 = seal, r3 = task id, r4 = runner, r5 = epoch,
;   r6 = output digest, r7 = trace digest -> r3
hc_nmp_complete:
ST r15,r0,@hc_nd_ret
ST r3,r0,@hc_nd_id
ST r4,r0,@hc_nd_run
ST r5,r0,@hc_nd_ep
ST r6,r0,@hc_nd_out
ST r7,r0,@hc_nd_trc
CALL @hc_nmp_gate
CMP r3,r0
JMP[NE] @hc_nd_exit
ST r1,r0,@hc_nd_rec
LD r1,r0,@hc_nd_id
CALL @hc_nmp_find
CMP r1,r0
JMP[EQ] @hc_nd_none
LD r5,r1,4
LD r6,r0,@hc_nd_rec
LD r6,r6,2
CMP r5,r6
JMP[NE] @hc_nd_pt
LD r1,r0,@hc_nd_id
LD r2,r0,@hc_nd_run
LD r3,r0,@hc_nd_ep
LD r4,r0,@hc_nd_out
LD r5,r0,@hc_nd_trc
CALL @nmr_task_finish
CMP r3,r0
JMP[EQ] @hc_nd_exit
ST r3,r0,@hc_native_code
MOVI r3,19
JMP @hc_nd_exit
hc_nd_none:
MOVI r3,19
MOVI r4,1
ST r4,r0,@hc_native_code
JMP @hc_nd_exit
hc_nd_pt:
MOVI r3,26
hc_nd_exit:
LD r15,r0,@hc_nd_ret
RET
hc_nd_ret:
.word 0
hc_nd_id:
.word 0
hc_nd_run:
.word 0
hc_nd_ep:
.word 0
hc_nd_out:
.word 0
hc_nd_trc:
.word 0
hc_nd_rec:
.word 0

; ======================================================================= state
hc_cfg:
.word 0,0,0,0
hc_bst:
.word 0,0,0,0,0,0,0,0
hc_outregs:
.word 0,0,0,0,0,0,0,0,0,0,0,0
hc_ltab:
.word 0,0,0,0,0,0,0,0
hc_ctab:
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0
