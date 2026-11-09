; Persistent HERMES guest loaded by real canonical NaN0S C runtime.
; This image is assembled as NaN16 words with LABELS emitted by nmk.awk.
; The stable load entry and n16_server_exchange are inert compatibility stubs:
; N16T protocol runs in the existing separate N16T guest VM/pool, not here.
main:
HALT
n16_server_exchange:
HALT
.include "hermes_lease.nas"
.include "hermes_nannet_route.nas"
