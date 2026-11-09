; HERMES NANNET authoritative route binding, native NaN16DE.
; Authenticated portal endpoint HW2 maps to point 3 inside BS0, but the
; point is verified from a trusted registry, not parsed from an HTTP claim.
main:
MOVI r1,0
MOVI r2,7
MOVI r3,111
MOVI r4,901
MOVI r5,77
MOVI r6,3
MOVI r7,300
MOVI r8,100
CALL @nhl_open
CMP r3,r0
JMP[NE] @fail
; Correct authenticated point, owner, generation and head.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,77
MOVI r6,24
MOVI r7,101
MOVI r8,3
CALL @nhl_nannet_route
CMP r3,r0
JMP[NE] @fail
MOVI r9,901
CMP r1,r9
JMP[NE] @fail
MOVI r9,24
CMP r2,r9
JMP[NE] @fail
MOVI r9,3
CMP r4,r9
JMP[NE] @fail
MOVI r9,7
CMP r5,r9
JMP[NE] @fail
; NANNET point 4 belongs elsewhere. Must not return an address.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,77
MOVI r6,24
MOVI r7,101
MOVI r8,4
CALL @nhl_nannet_route
MOVI r9,8
CMP r3,r9
JMP[NE] @fail
CMP r1,r0
JMP[NE] @fail
CMP r2,r0
JMP[NE] @fail
; Wrong owner is denied even if node point and lease ID agree.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,78
MOVI r6,24
MOVI r7,101
MOVI r8,3
CALL @nhl_nannet_route
MOVI r9,5
CMP r3,r9
JMP[NE] @fail
; Network port at head 255 cannot be used for learning.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,77
MOVI r6,255
MOVI r7,101
MOVI r8,3
CALL @nhl_nannet_route
MOVI r9,7
CMP r3,r9
JMP[NE] @fail
; Generation replay blocked.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,6
MOVI r5,77
MOVI r6,24
MOVI r7,101
MOVI r8,3
CALL @nhl_nannet_route
MOVI r9,2
CMP r3,r9
JMP[NE] @fail
; Deadline boundary expires lease.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,77
MOVI r6,24
MOVI r7,300
MOVI r8,3
CALL @nhl_nannet_route
MOVI r9,4
CMP r3,r9
JMP[NE] @fail
; Cannot revive an expired generation even if the point is now right.
MOVI r1,0
MOVI r2,111
MOVI r3,901
MOVI r4,7
MOVI r5,77
MOVI r6,24
MOVI r7,200
MOVI r8,3
CALL @nhl_nannet_route
MOVI r9,6
CMP r3,r9
JMP[NE] @fail
MOVI r1,@message_pass
JMP @say
fail:
MOVI r1,@message_fail
say:
MOVI r11,255
say_loop:
LD r2,r1,0
MOVI r3,4
say_byte:
BOOL.8 r4,r2,r11
CMP r4,r0
JMP[EQ] @say_done
PUTC r4
SHR r2,r2,8
MOVI r12,1
SUB r3,r3,r12
JMP[NE] @say_byte
ADDI r1,r1,1
JMP @say_loop
say_done:
HALT
.include "../nan16/hermes_lease.nas"
.include "../nan16/hermes_nannet_route.nas"
message_pass:
.str "HERMES NANNET ROUTE PASS\n"
message_fail:
.str "HERMES NANNET ROUTE FAIL\n"
