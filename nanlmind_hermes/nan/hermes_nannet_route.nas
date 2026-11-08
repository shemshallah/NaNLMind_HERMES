; Native NaN16DE NANNET per-lease route handoff gate.
; Expected network point MUST be supplied by the authenticated NANNET/NaNROUTER
; server-side address binding, never copied from an untrusted client claim.
; This is a native second-layer consistency check, not token verification.
;
; ABI in: r1=lease slot, r2=lease id, r3=universe id, r4=generation,
;         r5=owner tag, r6=local head (0..254), r7=trusted now tick,
;         r8=server-bound node point (1..256).
; out OK: r1=universe id, r2=local head, r3=0, r4=NANNET node point,
;         r5=generation
; out failure: r1=r2=r4=r5=0, r3=error
; Status additionally 8=authenticated node mismatch.
; Lease record: nanlmind_hermes/nan/hermes_lease.nas -> nhl_table, 16 slots x 8 words.
nhl_nannet_route:
MOVI r9,16
CMP r1,r9
JMP[GE] @nhl_nannet_bad
MOVI r9,255
CMP r6,r9
JMP[GE] @nhl_nannet_port
CMP r8,r0
JMP[EQ] @nhl_nannet_bad
MOVI r9,256
CMP r8,r9
JMP[GT] @nhl_nannet_bad
SHL r9,r1,3
MOVI r10,@nhl_table
ADD r10,r10,r9
LD r9,r10,0
CMP r9,r4
JMP[NE] @nhl_nannet_stale
LD r9,r10,6
MOVI r11,1
CMP r9,r11
JMP[NE] @nhl_nannet_missing
LD r9,r10,1
CMP r9,r2
JMP[NE] @nhl_nannet_missing
LD r9,r10,2
CMP r9,r3
JMP[NE] @nhl_nannet_missing
LD r9,r10,3
CMP r9,r5
JMP[NE] @nhl_nannet_owner
LD r9,r10,5
CMP r9,r7
JMP[LE] @nhl_nannet_expired
LD r9,r10,4
CMP r9,r8
JMP[NE] @nhl_nannet_node_mismatch
ADD r1,r3,r0
ADD r2,r6,r0
LD r4,r10,4
LD r5,r10,0
MOVI r3,0
RET
nhl_nannet_bad:
MOVI r3,1
JMP @nhl_nannet_fail
nhl_nannet_stale:
MOVI r3,2
JMP @nhl_nannet_fail
nhl_nannet_missing:
MOVI r3,6
JMP @nhl_nannet_fail
nhl_nannet_owner:
MOVI r3,5
JMP @nhl_nannet_fail
nhl_nannet_port:
MOVI r3,7
JMP @nhl_nannet_fail
nhl_nannet_node_mismatch:
MOVI r3,8
JMP @nhl_nannet_fail
nhl_nannet_expired:
MOVI r9,3
ST r9,r10,6
MOVI r3,4
nhl_nannet_fail:
MOVI r1,0
MOVI r2,0
MOVI r4,0
MOVI r5,0
RET
