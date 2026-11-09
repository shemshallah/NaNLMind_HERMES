; MTH encode/decode fixture — NaN16 float32 only.
;
; Exercises both R9 adapter linear kernels in one image:
;   mth_encode256_128_f32   (256 -> 128)
;   mth_decode128_256_f32   (128 -> 256)
;
; The 32,768-word W buffer is reused for both cases.
;
; Encode fixture:
;   W is a 128x256 rectangular identity (W[i,i]=1 for i<128, else 0),
;   b=1, x=1 -> y[i]=2 for all 128.
;
; Decode fixture:
;   W is a 256x128 rectangular identity (W[i,i]=1 for i<128, else 0),
;   b=1, z=1 -> y[0..127]=2 and y[128..255]=1.
;
; Both result sweeps are exhaustive, so they include the sample and endpoint
; positions (0, 64, 127, 128, 255) called out by the ABI contract.

.org 0
JMP @mth_encode_decode_test_main
.include "../nan16/mth_encode256_128.nas"
.include "../nan16/mth_decode128_256.nas"

mth_encode_decode_test_main:
  MOVI r13,1

  ; ---- encode fixture: zero W (32,768 words) ----
  MOVI r4,@mth_encode_decode_w
  MOVI r6,32768
mth_encode_decode_zero_w_encode:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_zero_w_encode

  ; W[i,i] = 1.0 for i=0..127; encode row stride 256, diagonal stride 257.
  MOVI r5,@mth_encode_decode_w
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_encode_diag:
  ST r7,r5,0
  MOVI r8,257
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_encode_diag

  ; b = 1.0 for all 128.
  MOVI r3,@mth_encode_decode_b
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_fill_b_encode:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_fill_b_encode

  ; x = 1.0 for all 256.
  MOVI r2,@mth_encode_decode_v
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_fill_x:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_fill_x

  MOVI r1,@mth_encode_decode_w
  MOVI r2,@mth_encode_decode_v
  MOVI r3,@mth_encode_decode_b
  MOVI r4,@mth_encode_decode_y
  CALL @mth_encode256_128_f32
  CMP r1,r13
  JMP[NE] @mth_encode_decode_fail

  ; y[i] = 2.0 for all 128 (includes i=0, 64, 127).
  MOVI r4,@mth_encode_decode_y
  MOVI r6,0
  MOVI r7,128
  MOVI r10,0
  MOVHI r10,0x4000
mth_encode_decode_check_encode:
  ADD r8,r4,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_encode_decode_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_encode_decode_check_encode

  ; ---- decode fixture: zero W again ----
  MOVI r4,@mth_encode_decode_w
  MOVI r6,32768
mth_encode_decode_zero_w_decode:
  ST r0,r4,0
  ADD r4,r4,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_zero_w_decode

  ; W[i,i] = 1.0 for i=0..127; decode row stride 128, diagonal stride 129.
  MOVI r5,@mth_encode_decode_w
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_decode_diag:
  ST r7,r5,0
  MOVI r8,129
  ADD r5,r5,r8
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_decode_diag

  ; b = 1.0 for all 256.
  MOVI r3,@mth_encode_decode_b
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_fill_b_decode:
  ST r7,r3,0
  ADD r3,r3,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_fill_b_decode

  ; z = 1.0 for all 128.
  MOVI r2,@mth_encode_decode_v
  MOVI r6,128
  MOVI r7,0
  MOVHI r7,0x3F80
mth_encode_decode_fill_z:
  ST r7,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_encode_decode_fill_z

  MOVI r1,@mth_encode_decode_w
  MOVI r2,@mth_encode_decode_v
  MOVI r3,@mth_encode_decode_b
  MOVI r4,@mth_encode_decode_y
  CALL @mth_decode128_256_f32
  CMP r1,r13
  JMP[NE] @mth_encode_decode_fail

  ; y[i] = 2.0 for i=0..127 (row i has the single 1.0 on the diagonal).
  MOVI r4,@mth_encode_decode_y
  MOVI r6,0
  MOVI r7,128
  MOVI r10,0
  MOVHI r10,0x4000
mth_encode_decode_check_decode_hi:
  ADD r8,r4,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_encode_decode_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_encode_decode_check_decode_hi

  ; y[i] = 1.0 for i=128..255 (zero rows, bias only; includes i=255).
  MOVI r6,128
  MOVI r7,256
  MOVI r10,0
  MOVHI r10,0x3F80
mth_encode_decode_check_decode_lo:
  ADD r8,r4,r6
  LD r9,r8,0
  CMP r9,r10
  JMP[NE] @mth_encode_decode_fail
  ADD r6,r6,r13
  CMP r6,r7
  JMP[LT] @mth_encode_decode_check_decode_lo

  MOVI r1,@mth_encode_decode_pass
  CALL @mth_encode_decode_puts
  HALT

mth_encode_decode_fail:
  MOVI r1,@mth_encode_decode_bad
  CALL @mth_encode_decode_puts
  HALT

mth_encode_decode_puts:
  MOVI r4,255
mth_encode_decode_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_encode_decode_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_encode_decode_puts_byte
  ADD r1,r1,r13
  JMP @mth_encode_decode_puts_word

; W: 32,768 words, reused by both cases.
.org 512
mth_encode_decode_w:
.word 0
; v: 256-word vector slot (x for encode, z for decode).
.org 33280
mth_encode_decode_v:
.word 0
; b: 256-word bias slot (128 used by encode, 256 by decode).
.org 33536
mth_encode_decode_b:
.word 0
; y: 256-word output slot (128 used by encode, 256 by decode).
.org 33792
mth_encode_decode_y:
.word 0
.org 34048
mth_encode_decode_pass:
.str "MTH ENCODE DECODE PASS\n"
mth_encode_decode_bad:
.str "MTH ENCODE DECODE FAIL\n"
