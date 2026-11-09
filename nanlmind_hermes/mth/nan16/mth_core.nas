; MTH NaN16 core — first native kernel
; This file contains mathematical code only in NaN16.
; ABI: r13 = 1. mth_rot2_class: r1=n -> r1=rot2(n).
; mth_bswapm64: r1=address -> BSWAPM the word at that address in place.
; BSWAPM is the existing NaN16 full eight-byte reversal: it swaps both
; NaN faces and reverses the 32-bit payload byte order. It is distinct from
; the redundant double-endian pair-voting codec in n16de.nas.

mth_rot2_class:
  MOVI r2,3
  BOOL.8 r2,r1,r2
  SHL r2,r2,2
  SHR r3,r1,2
  MOVI r4,3
  BOOL.8 r3,r3,r4
  BOOL.14 r1,r2,r3
  RET

; Explicit wrapper around the frozen NaN16 face-swap operation.
; Caller supplies a writable word address in r1.
mth_bswapm64:
  BSWAPM r1,0
  RET

; Split the ordered class nibble into its two 2-bit face classes.
; mth_face_far: r1=n -> r1=c_far. Clobbers r2.
mth_face_far:
  MOVI r2,3
  BOOL.8 r1,r1,r2
  RET

; mth_face_near: r1=n -> r1=c_near. Clobbers r2.
mth_face_near:
  SHR r1,r1,2
  MOVI r2,3
  BOOL.8 r1,r1,r2
  RET

; A face class c = 2*quiet + sign.
; mth_face_sign: r1=c -> r1=sign bit.
mth_face_sign:
  MOVI r2,1
  BOOL.8 r1,r1,r2
  RET

; mth_face_quiet: r1=c -> r1=quiet bit.
mth_face_quiet:
  SHR r1,r1,1
  MOVI r2,1
  BOOL.8 r1,r1,r2
  RET

; mth_pack_faces: r1=c_near, r2=c_far -> r1=n, r2=1.
; Invalid face classes (>3) return r1=0, r2=0.
mth_pack_faces:
  MOVI r3,3
  CMP r1,r3
  JMP[GT] @mth_pack_faces_bad
  CMP r2,r3
  JMP[GT] @mth_pack_faces_bad
  SHL r1,r1,2
  BOOL.14 r1,r1,r2
  MOVI r2,1
  RET
mth_pack_faces_bad:
  MOVI r1,0
  MOVI r2,0
  RET
