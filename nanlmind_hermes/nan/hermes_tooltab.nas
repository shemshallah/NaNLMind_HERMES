; HERMES connector verified MTH allowlist, native NaN16 table.
; Entry: FNV-1a/32 of canonical kernel ID, max workspace words,
; verification bit (1=real native NaN0S CI), step budget hint.
; HERMES cannot claim a native kernel by merely supplying its own hash.
; Only checked u32 kernels are currently wired to signed leased execution.
hc_tool_count:
.word 2
hc_tooltab:
.word 0x39CA1CBF,0,1,1024
.word 0x69B91132,0,1,1024
