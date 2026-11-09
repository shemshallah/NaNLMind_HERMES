; MTH 256-element colored noise fixture — NaN16 only.
;
; Checks:
;   1. same seed + alpha twice -> bit-identical 256-word outputs;
;   2. alpha=0 -> documented white sequence U(LCG^i(seed)) at sampled indices;
;   3. every output finite and in [-1,1];
;   4. alpha sweep 0, 0.5, 0.9, 0.99, 0.999 on one seed:
;      sign agreement nondecreasing, first-difference energy strictly
;      decreasing, normalized lag-1 correlation strictly increasing;
;   5. alpha=0.999 -> 255/255 same-sign adjacent pairs and correlation > 0.9999.
;
; The fixture computes all metrics in NaN16 float32 in increasing index order.
; It stores the sweep metrics at mth_cn_metric_sign/diff/corr for --peek.

.org 0
JMP @mth_cn_test_main
.include "../nan16/mth_colored_noise_256.nas"

mth_cn_test_main:
  MOVI r13,1

  ; --- 1. repeatability: seed 0x12345678, alpha 0.5, two buffers ----------
  MOVI r1,0x5678
  MOVHI r1,0x1234
  MOVI r2,@mth_cn_out_a
  MOVI r3,@mth_cn_alpha_05
  CALL @mth_colored_noise_256
  CMP r1,r13
  JMP[NE] @mth_cn_fail

  MOVI r1,0x5678
  MOVHI r1,0x1234
  MOVI r2,@mth_cn_out_b
  MOVI r3,@mth_cn_alpha_05
  CALL @mth_colored_noise_256
  CMP r1,r13
  JMP[NE] @mth_cn_fail

  MOVI r4,@mth_cn_out_a
  MOVI r5,@mth_cn_out_b
  MOVI r6,256
mth_cn_compare_loop:
  LD r7,r4,0
  LD r8,r5,0
  CMP r7,r8
  JMP[NE] @mth_cn_fail
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cn_compare_loop

  ; --- 2. alpha=0 is white: sampled values are fixed LCG draws ------------
  MOVI r1,0x5678
  MOVHI r1,0x1234
  MOVI r2,@mth_cn_out_a
  MOVI r3,@mth_cn_alpha_00
  CALL @mth_colored_noise_256
  CMP r1,r13
  JMP[NE] @mth_cn_fail

  MOVI r4,@mth_cn_out_a
  MOVI r5,@mth_cn_white_expected
  MOVI r6,6
mth_cn_white_loop:
  LD r7,r4,0
  LD r8,r5,0
  CMP r7,r8
  JMP[NE] @mth_cn_fail
  ADD r4,r4,r13
  ADD r5,r5,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_cn_white_loop

  MOVI r4,@mth_cn_out_a
  MOVI r6,255
  ADD r4,r4,r6
  LD r7,r4,0
  MOVI r8,@mth_cn_white_last
  LD r8,r8,0
  CMP r7,r8
  JMP[NE] @mth_cn_fail

  ; --- 3. range check the white output ------------------------------------
  MOVI r1,@mth_cn_out_a
  CALL @mth_cn_check_range

  ; --- 4. alpha sweep on the fixed seed -----------------------------------
  MOVI r12,0
mth_cn_alpha_loop:
  MOVI r3,@mth_cn_alphas
  ADD r3,r3,r12
  MOVI r1,0x5678
  MOVHI r1,0x1234
  MOVI r2,@mth_cn_out_a
  CALL @mth_colored_noise_256
  CMP r1,r13
  JMP[NE] @mth_cn_fail

  MOVI r1,@mth_cn_out_a
  CALL @mth_cn_check_range

  MOVI r1,@mth_cn_out_a
  CALL @mth_cn_metrics
  ; r2 = same-sign adjacent count, r3 = first-difference energy (bits),
  ; r4 = normalized lag-1 correlation (bits)
  MOVI r5,@mth_cn_metric_sign
  ADD r5,r5,r12
  ST r2,r5,0
  MOVI r5,@mth_cn_metric_diff
  ADD r5,r5,r12
  ST r3,r5,0
  MOVI r5,@mth_cn_metric_corr
  ADD r5,r5,r12
  ST r4,r5,0

  CMP r12,r0
  JMP[EQ] @mth_cn_metrics_advance

  ; sign agreement must not decrease
  MOVI r5,@mth_cn_metric_sign
  ADD r6,r5,r12
  SUB r6,r6,r13
  LD r7,r6,0
  CMP r2,r7
  JMP[LT] @mth_cn_fail

  ; first-difference energy must decrease
  MOVI r5,@mth_cn_metric_diff
  ADD r6,r5,r12
  SUB r6,r6,r13
  LD r7,r6,0
  FCMP r3,r7
  JMP[GE] @mth_cn_fail

  ; normalized lag-1 correlation must increase
  MOVI r5,@mth_cn_metric_corr
  ADD r6,r5,r12
  SUB r6,r6,r13
  LD r7,r6,0
  FCMP r4,r7
  JMP[LE] @mth_cn_fail

mth_cn_metrics_advance:
  ADD r12,r12,r13
  MOVI r6,5
  CMP r12,r6
  JMP[LT] @mth_cn_alpha_loop

  ; --- 5. alpha=0.999 is strongly correlated ------------------------------
  MOVI r5,@mth_cn_metric_sign
  MOVI r6,4
  ADD r5,r5,r6
  LD r7,r5,0
  MOVI r8,255
  CMP r7,r8
  JMP[NE] @mth_cn_fail

  MOVI r5,@mth_cn_metric_corr
  ADD r5,r5,r6
  LD r7,r5,0
  MOVI r8,@mth_cn_corr_threshold
  LD r8,r8,0
  FCMP r7,r8
  JMP[LE] @mth_cn_fail

  MOVI r1,@mth_cn_pass
  CALL @mth_cn_puts
  HALT

; --- helpers --------------------------------------------------------------
; r1 = 256 float32 payloads. Fail unless every |x| <= 1.0 and finite.
mth_cn_check_range:
  MOVI r4,256
  MOVI r7,0xFFFF
  MOVHI r7,0x7FFF           ; 0x7FFFFFFF sign mask
  MOVI r8,0
  MOVHI r8,0x3F80           ; 1.0 bits
  MOVI r9,0
  MOVHI r9,0x7F80           ; +inf bits
mth_cn_range_loop:
  LD r5,r1,0
  BOOL.8 r6,r5,r7
  CMP r6,r9
  JMP[GE] @mth_cn_fail      ; inf or NaN
  CMP r6,r8
  JMP[GT] @mth_cn_fail      ; magnitude above 1.0
  ADD r1,r1,r13
  SUB r4,r4,r13
  CMP r4,r0
  JMP[GT] @mth_cn_range_loop
  RET

; r1 = 256 float32 payloads.
; returns r2 = count(out[i-1],out[i] same sign), i=1..255
;         r3 = sum (out[i]-out[i-1])^2
;         r4 = sum(out[i]*out[i-1]) / sqrt(sum(out[i-1]^2)*sum(out[i]^2))
; Preserves r12-r14. All float sums use increasing i.
mth_cn_metrics:
  ADD r7,r1,r0
  MOVI r8,255
  LD r9,r7,0
  ADD r7,r7,r13
  MOVI r2,0
  MOVI r3,0
  MOVI r4,0
  MOVI r5,0
  MOVI r6,0
mth_cn_metrics_loop:
  LD r10,r7,0
  SHR r1,r9,31
  SHR r11,r10,31
  CMP r1,r11
  JMP[NE] @mth_cn_metrics_no_sign
  ADD r2,r2,r13
mth_cn_metrics_no_sign:
  FSUB r11,r10,r9
  FMUL r11,r11,r11
  FADD r3,r3,r11
  FMUL r11,r10,r9
  FADD r4,r4,r11
  FMUL r11,r9,r9
  FADD r5,r5,r11
  FMUL r11,r10,r10
  FADD r6,r6,r11
  ADD r9,r10,r0
  ADD r7,r7,r13
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_cn_metrics_loop
  FMUL r11,r5,r6
  FSQRT r11,r11
  FDIV r4,r4,r11
  RET

mth_cn_fail:
  MOVI r1,@mth_cn_bad
  CALL @mth_cn_puts
  HALT

mth_cn_puts:
  MOVI r4,255
mth_cn_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_cn_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_cn_puts_byte
  ADD r1,r1,r13
  JMP @mth_cn_puts_word

; --- data ----------------------------------------------------------------
.org 4096
mth_cn_out_a:
.word 0

.org 4352
mth_cn_out_b:
.word 0

.org 4608
mth_cn_alphas:
mth_cn_alpha_00:
.word 0x00000000          ; 0.0
mth_cn_alpha_05:
.word 0x3F000000          ; 0.5
.word 0x3F666666          ; 0.9
.word 0x3F7D70A4          ; 0.99
.word 0x3F7FBE77          ; 0.999

; alpha=0, seed 0x12345678: U(seed), U(LCG(seed)), ... and U(LCG^255(seed)).
mth_cn_white_expected:
.word 0xBF5B9754
.word 0xBDABCD90
.word 0x3F1A60BC
.word 0xBF34480C
.word 0xBED28DAC
.word 0x3D1BC1C0
mth_cn_white_last:
.word 0x3DD3A330

.org 4704
mth_cn_metric_sign:
.word 0
.word 0
.word 0
.word 0
.word 0
mth_cn_metric_diff:
.word 0
.word 0
.word 0
.word 0
.word 0
mth_cn_metric_corr:
.word 0
.word 0
.word 0
.word 0
.word 0
mth_cn_corr_threshold:
.word 0x3F7FF972          ; 0.9999

.org 4800
mth_cn_pass:
.str "MTH COLORED NOISE 256 PASS\n"
mth_cn_bad:
.str "MTH COLORED NOISE 256 FAIL\n"
