; MTH native top-k 256 fixture — NaN16 float32 only.
;
; Dataset: scores[i] = (float)(i - 128), so the array contains distinct
; positive and negative values. Eight entries are then overwritten:
;
;   scores[0]   = 100.0     (duplicate with 50, 228, 255)
;   scores[50]  = 100.0
;   scores[255] = 100.0
;   scores[10]  = 127.0
;   scores[20]  = -128.0
;   scores[30]  = -77.0     (duplicate with 51)
;   scores[40]  = 0.0       (duplicate with 128)
;   scores[60]  = -100.0    (duplicate with 28)
;
; The expected index table is the exact descending-value order with ties
; broken by the smaller original index first. The fixture checks:
;
;   - k=1 returns 1, index 10, value 127.0;
;   - k=32 returns 32 and the exact 32-entry order, which crosses the
;     four-way 100.0 tie and therefore checks the tie-break;
;   - k=256 returns 256 and the exact 256-entry order;
;   - each selected value equals scores[selected_index] bit-exactly;
;   - k=0, k=257, and k=-1 return 0 and leave both full output spans
;     untouched;
;   - a NaN score returns 0 and leaves both full output spans untouched;
;   - restoring the score recovers the k=1 result.
;
; Expected order table and values are fixture data, not runtime math.

.org 0
JMP @mth_topk_256_test_main
.include "../nan16/mth_topk_256.nas"

mth_topk_256_test_main:
  MOVI r13,1

  ; scores[i] = (float)(i - 128), increasing then overwritten below.
  LA r1,@mth_topk_256_test_scores
  MOVI r5,0
mth_topk_256_test_build:
  MOVI r6,128
  SUB r7,r5,r6
  CVTIF r7,r7
  ADD r8,r1,r5
  ST r7,r8,0
  ADD r5,r5,r13
  MOVI r6,256
  CMP r5,r6
  JMP[LT] @mth_topk_256_test_build

  ; Overwrite the eight designed entries.
  LA r5,@mth_topk_256_test_scores

  ; 100.0 at 0, 50, 255 (228 already holds baseline 100.0).
  MOVI r6,0
  MOVHI r6,0x42C8
  ST r6,r5,0
  ST r6,r5,50
  ST r6,r5,255

  ; 127.0 at 10.
  MOVI r6,0
  MOVHI r6,0x42FE
  ST r6,r5,10

  ; -128.0 at 20.
  MOVI r6,0
  MOVHI r6,0xC300
  ST r6,r5,20

  ; -77.0 at 30.
  MOVI r6,0
  MOVHI r6,0xC29A
  ST r6,r5,30

  ; +0.0 at 40.
  ST r0,r5,40

  ; -100.0 at 60.
  MOVI r6,0
  MOVHI r6,0xC2C8
  ST r6,r5,60

  ; Case k=1: index 10, value 127.0.
  LA r1,@mth_topk_256_test_scores
  MOVI r2,1
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r13
  JMP[NE] @mth_topk_256_test_fail

  LA r5,@mth_topk_256_test_indices
  LD r6,r5,0
  MOVI r7,10
  CMP r6,r7
  JMP[NE] @mth_topk_256_test_fail

  LA r5,@mth_topk_256_test_values
  LD r6,r5,0
  MOVI r7,0
  MOVHI r7,0x42FE
  CMP r6,r7
  JMP[NE] @mth_topk_256_test_fail

  ; Case k=32: exact order across the four-way 100.0 tie.
  LA r1,@mth_topk_256_test_scores
  MOVI r2,32
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  MOVI r7,32
  CMP r1,r7
  JMP[NE] @mth_topk_256_test_fail
  MOVI r1,32
  CALL @mth_topk_256_test_check_prefix
  CMP r1,r13
  JMP[NE] @mth_topk_256_test_fail

  ; Case k=256: exact full order.
  LA r1,@mth_topk_256_test_scores
  MOVI r2,256
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  MOVI r7,256
  CMP r1,r7
  JMP[NE] @mth_topk_256_test_fail
  MOVI r1,256
  CALL @mth_topk_256_test_check_prefix
  CMP r1,r13
  JMP[NE] @mth_topk_256_test_fail

  ; k=0 rejection: return 0, both full outputs untouched.
  CALL @mth_topk_256_test_fill_sentinels
  LA r1,@mth_topk_256_test_scores
  MOVI r2,0
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r0
  JMP[NE] @mth_topk_256_test_fail
  CALL @mth_topk_256_test_check_sentinels

  ; k=257 rejection: return 0, both full outputs untouched.
  CALL @mth_topk_256_test_fill_sentinels
  LA r1,@mth_topk_256_test_scores
  MOVI r2,257
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r0
  JMP[NE] @mth_topk_256_test_fail
  CALL @mth_topk_256_test_check_sentinels

  ; k=-1 (0xFFFFFFFF) rejection: unsigned upper bound, outputs untouched.
  CALL @mth_topk_256_test_fill_sentinels
  LA r1,@mth_topk_256_test_scores
  MOVI r2,0xFFFF
  MOVHI r2,0xFFFF
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r0
  JMP[NE] @mth_topk_256_test_fail
  CALL @mth_topk_256_test_check_sentinels

  ; NaN score rejection: return 0, both full outputs untouched.
  CALL @mth_topk_256_test_fill_sentinels

  LA r5,@mth_topk_256_test_scores
  MOVI r6,0
  MOVHI r6,0x7FC0
  ST r6,r5,10
  LA r1,@mth_topk_256_test_scores
  MOVI r2,1
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r0
  JMP[NE] @mth_topk_256_test_fail
  CALL @mth_topk_256_test_check_sentinels

  ; Restore scores[10]=127.0 and confirm the selector recovers.
  LA r5,@mth_topk_256_test_scores
  MOVI r6,0
  MOVHI r6,0x42FE
  ST r6,r5,10
  LA r1,@mth_topk_256_test_scores
  MOVI r2,1
  LA r3,@mth_topk_256_test_indices
  LA r4,@mth_topk_256_test_values
  CALL @mth_topk_256
  CMP r1,r13
  JMP[NE] @mth_topk_256_test_fail
  LA r5,@mth_topk_256_test_indices
  LD r6,r5,0
  MOVI r7,10
  CMP r6,r7
  JMP[NE] @mth_topk_256_test_fail

  LA r1,@mth_topk_256_test_pass
  CALL @mth_topk_256_test_puts
  HALT

mth_topk_256_test_fail:
  LA r1,@mth_topk_256_test_bad
  CALL @mth_topk_256_test_puts
  HALT

; Check that indices_out[0..count-1] and values_out[0..count-1] match the
; fixture's expected index table and the original score payloads.
; r1 = count; returns r1=1 on success, r1=0 on mismatch. Clobbers r2-r12.
mth_topk_256_test_check_prefix:
  MOVI r5,0
  LA r6,@mth_topk_256_test_expected
  LA r7,@mth_topk_256_test_indices
  LA r8,@mth_topk_256_test_scores
  LA r9,@mth_topk_256_test_values
mth_topk_256_test_check_loop:
  ADD r10,r6,r5
  LD r10,r10,0
  ADD r11,r7,r5
  LD r11,r11,0
  CMP r11,r10
  JMP[NE] @mth_topk_256_test_check_bad
  ADD r11,r8,r10
  LD r11,r11,0
  ADD r12,r9,r5
  LD r12,r12,0
  CMP r12,r11
  JMP[NE] @mth_topk_256_test_check_bad
  ADD r5,r5,r13
  CMP r5,r1
  JMP[LT] @mth_topk_256_test_check_loop
  MOVI r1,1
  RET
mth_topk_256_test_check_bad:
  MOVI r1,0
  RET

; Fill both full output spans with the sentinel 0xDEADBEEF.
mth_topk_256_test_fill_sentinels:
  MOVI r7,0xDEAD
  MOVHI r7,0xDEAD
  LA r5,@mth_topk_256_test_indices
  LA r6,@mth_topk_256_test_values
  MOVI r8,0
mth_topk_256_test_fill_loop:
  ADD r9,r5,r8
  ST r7,r9,0
  ADD r9,r6,r8
  ST r7,r9,0
  ADD r8,r8,r13
  MOVI r9,256
  CMP r8,r9
  JMP[LT] @mth_topk_256_test_fill_loop
  RET

; Require both full output spans to remain sentinel.
mth_topk_256_test_check_sentinels:
  MOVI r7,0xDEAD
  MOVHI r7,0xDEAD
  LA r5,@mth_topk_256_test_indices
  LA r6,@mth_topk_256_test_values
  MOVI r8,0
mth_topk_256_test_sentinel_loop:
  ADD r9,r5,r8
  LD r10,r9,0
  CMP r10,r7
  JMP[NE] @mth_topk_256_test_fail
  ADD r9,r6,r8
  LD r10,r9,0
  CMP r10,r7
  JMP[NE] @mth_topk_256_test_fail
  ADD r8,r8,r13
  MOVI r9,256
  CMP r8,r9
  JMP[LT] @mth_topk_256_test_sentinel_loop
  RET

mth_topk_256_test_puts:
  MOVI r4,255
mth_topk_256_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_topk_256_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_topk_256_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_topk_256_test_puts_word

.org 1024
mth_topk_256_test_scores:
.word 0
.org 1280
mth_topk_256_test_indices:
.word 0
.org 1536
mth_topk_256_test_values:
.word 0
.org 1792
; Exact expected order: descending value, ties by smaller original index.
mth_topk_256_test_expected:
.word 10, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240
.word 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 0, 50, 228, 255, 227
.word 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211
.word 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195
.word 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179
.word 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163
.word 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147
.word 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131
.word 130, 129, 40, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116
.word 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100
.word 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84
.word 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68
.word 67, 66, 65, 64, 63, 62, 61, 59, 58, 57, 56, 55, 54, 53, 52, 30
.word 51, 49, 48, 47, 46, 45, 44, 43, 42, 41, 39, 38, 37, 36, 35, 34
.word 33, 32, 31, 29, 28, 60, 27, 26, 25, 24, 23, 22, 21, 19, 18, 17
.word 16, 15, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20

mth_topk_256_test_pass:
.str "MTH TOPK 256 PASS\n"
mth_topk_256_test_bad:
.str "MTH TOPK 256 FAIL\n"
