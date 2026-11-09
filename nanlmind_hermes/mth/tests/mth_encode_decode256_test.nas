; MTH full 256x256 encode/decode fixture — NaN16 float32 only.
;
; Exercises both full square dense linear kernels in one image:
;   mth_encode256_full_f32   (256 -> 256)
;   mth_decode256_full_f32   (256 -> 256)
;
; The 65,536-word W buffer is reused for both cases.
;
; Encode fixture:
;   W = 2I (W[i,i]=2.0 for all 256), x=1, b=1
;   -> y[i] = 1 + 2*1 = 3.0 for all 256.
;
; Decode fixture (second bias case):
;   W = 2I, z=1, b=2
;   -> y[i] = 2 + 2*1 = 4.0 for all 256.
;
; Both result sweeps are exhaustive over 0..255. The output slot is cleared
; before decode so a decode that fails to write cannot inherit the encode
; result. Total guest working set is 66,304 words plus code and strings,
; above the legacy 512 KiB / 65,536-word ceiling and within the 1 MiB /
; 131,072-word target.

.org 0
JMP @mth_encode_decode256_test_main
.include "../nan16/mth_encode256_full.nas"
.include "../nan16/mth_decode256_full.nas"

mth_encode_decode256_test_main:
  MOVI r13,1

  ; ---- zero W (65,536 words) ----
  LA r4,@mth_encode_decode256_w
  MOVI r6,0
  MOVHI r6,1
mth_encode_decode256_zero_w:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_zero_w

  ; W[i,i] = 2.0 for i=0..255; row stride 256, diagonal stride 257.
  LA r5,@mth_encode_decode256_w
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
  MOVI r8,0
mth_encode_decode256_diag:
  ADD r9,r5,r8
  ST r7,r9,0
  MOVI r9,257
  ADD r8,r8,r9
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_diag

  ; x = 1.0 for all 256.
  LA r2,@mth_encode_decode256_v
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode256_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_fill_x

  ; b = 1.0 for all 256.
  LA r3,@mth_encode_decode256_b
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode256_fill_b_encode:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_fill_b_encode

  ; ---- encode: W=2I, x=1, b=1 -> y=3.0 ----
  LA r1,@mth_encode_decode256_w
  LA r2,@mth_encode_decode256_v
  LA r3,@mth_encode_decode256_b
  LA r4,@mth_encode_decode256_y
  CALL @mth_encode256_full_f32
  CMP r1,r13
  JMP[NE] @mth_encode_decode256_fail

  LA r4,@mth_encode_decode256_y
  MOVI r6,0
  MOVI r7,256
  MOVI r10,0
  MOVHI r10,0x4040
mth_encode_decode256_check_encode:
  ADD r8,r4,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_encode_decode256_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_encode_decode256_check_encode

  ; ---- decode: clear y, keep W=2I and z=1, set b=2 -> y=4.0 ----
  LA r4,@mth_encode_decode256_y
  MOVI r6,256
mth_encode_decode256_clear_y:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_clear_y

  LA r3,@mth_encode_decode256_b
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x4000
mth_encode_decode256_fill_b_decode:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode256_fill_b_decode

  LA r1,@mth_encode_decode256_w
  LA r2,@mth_encode_decode256_v
  LA r3,@mth_encode_decode256_b
  LA r4,@mth_encode_decode256_y
  CALL @mth_decode256_full_f32
  CMP r1,r13
  JMP[NE] @mth_encode_decode256_fail

  LA r4,@mth_encode_decode256_y
  MOVI r6,0
  MOVI r7,256
  MOVI r10,0
  MOVHI r10,0x4080
mth_encode_decode256_check_decode:
  ADD r8,r4,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_encode_decode256_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_encode_decode256_check_decode

  LA r1,@mth_encode_decode256_pass
  CALL @mth_encode_decode256_puts
  HALT

mth_encode_decode256_fail:
  LA r1,@mth_encode_decode256_bad
  CALL @mth_encode_decode256_puts
  HALT

mth_encode_decode256_puts:
  MOVI r4,255
mth_encode_decode256_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_encode_decode256_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_encode_decode256_puts_byte
  ADD r1,r1,r13
  JMP @mth_encode_decode256_puts_word

; W: 65,536 words at 512..66047, reused by both cases.
.org 512
mth_encode_decode256_w:
.word 0
.org 66048
mth_encode_decode256_v:
.word 0
.org 66304
mth_encode_decode256_b:
.word 0
.org 66560
mth_encode_decode256_y:
.word 0
.org 66816
mth_encode_decode256_pass:
.str "MTH ENCODE DECODE 256 PASS\n"
mth_encode_decode256_bad:
.str "MTH ENCODE DECODE 256 FAIL\n"
