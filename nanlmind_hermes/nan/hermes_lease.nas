; HERMES per-lease universe registry — native NaN16DE/NaNVM0.
; Independent universes per lease, never aliases of HW0::BS0.
; 16 concurrent leases; each owns local heads 0..254, head 255 is the
; sole NANNET node/port (not a learning head).
;
; Authority: only a server-authenticated and NMP-admitted caller may call
; nhl_open / nhl_renew / nhl_release. owner_tag is an opaque comparison
; token, NOT an authenticator or a substitute for a signed identity.
;
; Each slot is eight NaN16 words:
; [generation_high_watermark, lease_id, universe_id, owner_tag,
;  nannet_node_point, deadline_tick, state, reserved]
; States: 0 virgin, 1 live, 2 revoked, 3 expired.
; Slot reuse requires strictly increasing generation. Active lease IDs,
; universe IDs, and NANNET node points must be distinct across slots.
; All times are monotonic tick values supplied by trusted router context.
; Prevent 32-bit wrapping by rolling over to a NEW registry generation
; before the high-water or tick counters approach 0xFFFFFFFF.
;
; nhl_open:
;   in  r1 slot 0..15, r2 generation>=1, r3 lease_id>=1,
;       r4 universe_id>=1, r5 owner_tag>=1, r6 node_point 1..255,
;       r7 deadline_tick, r8 now_tick
;   out r3 status
; nhl_resolve:
;   in  r1 slot, r2 lease_id, r3 universe_id, r4 generation,
;       r5 owner_tag, r6 head 0..254, r7 now_tick
;   out r1 universe_id, r2 local_head, r3 status, r4 node_point,
;       r5 generation; error clears output coordinates
; nhl_renew:
;   in  r1 slot, r2 generation, r3 lease_id, r4 owner_tag,
;       r5 new_deadline, r6 now_tick
;   out r3 status
; nhl_release:
;   in  r1 slot, r2 generation, r3 lease_id, r4 owner_tag
;   out r3 status
; Status: 0 OK, 1 INVALID, 2 STALE_GENERATION, 3 BUSY_OR_COLLISION,
;         4 EXPIRED, 5 WRONG_OWNER, 6 NOT_LIVE_OR_WRONG_ID,
;         7 NANNET_PORT_RESERVED.
;
; No JSON, Python, C or host-side lease arithmetic is used here.
;
nhl_open:
MOVI r9,16
CMP r1,r9
JMP[GE] @nhl_open_bad
CMP r2,r0
JMP[EQ] @nhl_open_bad
CMP r3,r0
JMP[EQ] @nhl_open_bad
CMP r4,r0
JMP[EQ] @nhl_open_bad
CMP r5,r0
JMP[EQ] @nhl_open_bad
CMP r6,r0
JMP[EQ] @nhl_open_bad
MOVI r9,255
CMP r6,r9
JMP[GT] @nhl_open_bad
CMP r7,r8
JMP[LE] @nhl_open_expired
; Offset is slot*8 (SHL, native NaNVM0; no host multiplication).
SHL r9,r1,3
MOVI r11,@nhl_table
ADD r11,r11,r9
LD r12,r11,0
CMP r2,r12
JMP[LE] @nhl_open_stale
LD r12,r11,6
MOVI r13,1
CMP r12,r13
JMP[NE] @nhl_open_scan_init
LD r12,r11,5
CMP r12,r8
JMP[GT] @nhl_open_busy
; Disjoint active lease identity checks over every other slot.
nhl_open_scan_init:
MOVI r9,@nhl_table
MOVI r10,0
nhl_open_scan:
MOVI r12,16
CMP r10,r12
JMP[GE] @nhl_open_commit
CMP r10,r1
JMP[EQ] @nhl_open_next
LD r12,r9,6
MOVI r13,1
CMP r12,r13
JMP[NE] @nhl_open_next
LD r12,r9,5
CMP r12,r8
JMP[LE] @nhl_open_next
LD r12,r9,1
CMP r12,r3
JMP[EQ] @nhl_open_busy
LD r12,r9,2
CMP r12,r4
JMP[EQ] @nhl_open_busy
LD r12,r9,4
CMP r12,r6
JMP[EQ] @nhl_open_busy
nhl_open_next:
MOVI r13,8
ADD r9,r9,r13
MOVI r13,1
ADD r10,r10,r13
JMP @nhl_open_scan
; Write all fields only after validation, preserving high-water on rejection.
nhl_open_commit:
ST r2,r11,0
ST r3,r11,1
ST r4,r11,2
ST r5,r11,3
ST r6,r11,4
ST r7,r11,5
MOVI r12,1
ST r12,r11,6
MOVI r3,0
RET
nhl_open_bad:
MOVI r3,1
RET
nhl_open_stale:
MOVI r3,2
RET
nhl_open_busy:
MOVI r3,3
RET
nhl_open_expired:
MOVI r3,4
RET

nhl_resolve:
MOVI r8,16
CMP r1,r8
JMP[GE] @nhl_resolve_bad
MOVI r8,255
CMP r6,r8
JMP[GE] @nhl_resolve_port
SHL r8,r1,3
MOVI r9,@nhl_table
ADD r9,r9,r8
LD r8,r9,0
CMP r8,r4
JMP[NE] @nhl_resolve_stale
LD r8,r9,6
MOVI r10,1
CMP r8,r10
JMP[NE] @nhl_resolve_missing
LD r8,r9,1
CMP r8,r2
JMP[NE] @nhl_resolve_missing
LD r8,r9,2
CMP r8,r3
JMP[NE] @nhl_resolve_missing
LD r8,r9,3
CMP r8,r5
JMP[NE] @nhl_resolve_owner
LD r8,r9,5
CMP r7,r8
JMP[GE] @nhl_resolve_expired
LD r4,r9,4
ADD r1,r3,r0
ADD r2,r6,r0
LD r5,r9,0
MOVI r3,0
RET
nhl_resolve_bad:
MOVI r3,1
JMP @nhl_resolve_fail
nhl_resolve_stale:
MOVI r3,2
JMP @nhl_resolve_fail
nhl_resolve_missing:
MOVI r3,6
JMP @nhl_resolve_fail
nhl_resolve_owner:
MOVI r3,5
JMP @nhl_resolve_fail
nhl_resolve_port:
MOVI r3,7
JMP @nhl_resolve_fail
nhl_resolve_expired:
MOVI r8,3
ST r8,r9,6
MOVI r3,4
nhl_resolve_fail:
MOVI r1,0
MOVI r2,0
MOVI r4,0
MOVI r5,0
RET

nhl_renew:
MOVI r7,16
CMP r1,r7
JMP[GE] @nhl_renew_bad
SHL r7,r1,3
MOVI r8,@nhl_table
ADD r8,r8,r7
LD r7,r8,0
CMP r7,r2
JMP[NE] @nhl_renew_stale
LD r7,r8,6
MOVI r9,1
CMP r7,r9
JMP[NE] @nhl_renew_missing
LD r7,r8,1
CMP r7,r3
JMP[NE] @nhl_renew_missing
LD r7,r8,3
CMP r7,r4
JMP[NE] @nhl_renew_owner
LD r7,r8,5
CMP r6,r7
JMP[GE] @nhl_renew_expired
CMP r5,r7
JMP[LE] @nhl_renew_bad
ST r5,r8,5
MOVI r3,0
RET
nhl_renew_bad:
MOVI r3,1
RET
nhl_renew_stale:
MOVI r3,2
RET
nhl_renew_missing:
MOVI r3,6
RET
nhl_renew_owner:
MOVI r3,5
RET
nhl_renew_expired:
MOVI r9,3
ST r9,r8,6
MOVI r3,4
RET

nhl_release:
MOVI r6,16
CMP r1,r6
JMP[GE] @nhl_release_bad
SHL r6,r1,3
MOVI r7,@nhl_table
ADD r7,r7,r6
LD r6,r7,0
CMP r6,r2
JMP[NE] @nhl_release_stale
LD r6,r7,6
MOVI r8,1
CMP r6,r8
JMP[NE] @nhl_release_missing
LD r6,r7,1
CMP r6,r3
JMP[NE] @nhl_release_missing
LD r6,r7,3
CMP r6,r4
JMP[NE] @nhl_release_owner
MOVI r6,2
ST r6,r7,6
MOVI r3,0
RET
nhl_release_bad:
MOVI r3,1
RET
nhl_release_stale:
MOVI r3,2
RET
nhl_release_missing:
MOVI r3,6
RET
nhl_release_owner:
MOVI r3,5
RET

; Each lease slot has independently addressable heads, with one network port.
; Registry lives inside guest memory; persistence/restore needs an authenticated
; snapshot bound to the external NANNET lease ledger before reset.
nhl_table:
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0
