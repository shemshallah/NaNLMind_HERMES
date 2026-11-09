; MTH double-endian pair codec, native NaN16.
; Adapted from NaN0S src/lib/n16de.nas (blob 47d64c3446d57669160317228258c35d2184ad17).
; This is a redundant data carrier, not a cryptographic primitive.
;
; Each 16-bit chunk is encoded as two words:
;   member 0: class 10, tag 0 in payload[31:16], chunk in payload[15:0]
;   member 1: class 10, tag 1 in payload[31:16], bswap16(chunk) in payload[15:0]
; Decode requires both class/tag values and agreement after byte swapping.
;
; REENTRANT ABI (r13=1):
;   mth_de_encode: r1=chunk, r2=destination word address, r12=3-word caller frame.
;     frame[0]=saved return address, frame[1]=chunk, frame[2]=destination.
;   mth_de_decode: r1=pair address, r12=1-word caller frame.
;     frame[0]=saved return address.
;   encode returns no status; decode returns r1=chunk,r2=1 on success,
;   and r1=0,r2=0 on invalid/mismatched pair.
; The caller must provide a writable frame that is not shared by overlapping calls.
; No routine-local static return or operand slots are used.

.equ MTH_DE_CLASS 10
.equ MTH_DE_TAG_BE 0
.equ MTH_DE_TAG_LE 1
.equ MTH_DE_TAG_DELIM 2
.equ MTH_DE_TAG_RESYNC 3
.equ MTH_DE_TAG_CTRL 4

mth_de_bswap16:
  MOVI r5,255
  BOOL.8 r2,r1,r5
  SHL r2,r2,8
  SHR r3,r1,8
  BOOL.8 r3,r3,r5
  BOOL.14 r1,r2,r3
  RET

; r1=chunk, r2=destination, r12=caller frame[3]
mth_de_encode:
  STW r15,r12,r0
  ADD r6,r12,r13
  STW r1,r6,r0
  ADD r6,r6,r13
  STW r2,r6,r0

  MOVI r4,#MTH_DE_CLASS
  STW r1,r2,r4
  LD r1,r12,1
  CALL @mth_de_bswap16
  MOVI r4,#MTH_DE_TAG_LE
  SHL r4,r4,16
  BOOL.14 r1,r1,r4
  LD r2,r12,2
  ADD r2,r2,r13
  MOVI r4,#MTH_DE_CLASS
  STW r1,r2,r4

  LD r15,r12,0
  RET

; r1=pair address, r12=caller frame[1]
mth_de_decode:
  STW r15,r12,r0
  LDN r3,r1,0
  MOVI r4,#MTH_DE_CLASS
  CMP r3,r4
  JMP[NE] @mth_de_decode_bad
  LDN r3,r1,1
  CMP r3,r4
  JMP[NE] @mth_de_decode_bad
  LD r4,r1,0
  LD r5,r1,1
  SHR r6,r4,16
  CMP r6,r0
  JMP[NE] @mth_de_decode_bad
  SHR r6,r5,16
  MOVI r7,#MTH_DE_TAG_LE
  CMP r6,r7
  JMP[NE] @mth_de_decode_bad
  MOVI r6,0xFFFF
  BOOL.8 r4,r4,r6
  BOOL.8 r5,r5,r6
  ADD r1,r5,r0
  CALL @mth_de_bswap16
  CMP r1,r4
  JMP[NE] @mth_de_decode_bad
  ADD r1,r4,r0
  MOVI r2,1
  LD r15,r12,0
  RET

mth_de_decode_bad:
  MOVI r1,0
  MOVI r2,0
  LD r15,r12,0
  RET

mth_de_delim:
  MOVI r2,#MTH_DE_TAG_DELIM
  SHL r2,r2,16
  MOVI r3,#MTH_DE_CLASS
  STW r2,r1,r3
  RET

mth_de_resync:
  MOVI r2,#MTH_DE_TAG_RESYNC
  SHL r2,r2,16
  MOVI r3,#MTH_DE_CLASS
  STW r2,r1,r3
  RET
