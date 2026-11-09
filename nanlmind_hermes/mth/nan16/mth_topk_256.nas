; MTH native top-k selector for 256 float32 scores — NaN16 float32 only.
;
; ABI:
;   r1  = scores, 256 float32 payloads
;   r2  = k, integer 1..256
;   r3  = indices_out, k words (original indices, descending value order)
;   r4  = values_out, k float32 payloads (selected score values)
;   r13 = 1
;   returns r1 = k on success
;   returns r1 = 0 when k is outside 1..256 or any score is NaN
;   clobbers r1-r11 and the flags; preserves r12, r14, and r15
;
; Selection is a stable partial selection sort over the original array.
; For each output slot j = 0..k-1 the kernel scans the 256 scores in
; increasing original index and picks the greatest value not selected
; yet. Equal values therefore keep the smaller original index first.
; The scan comparison is:
;
;   candidate > best                       -> take candidate
;   candidate == best and i < best_index   -> take candidate
;   otherwise                              -> keep best
;
; Because the scan is increasing, the explicit equal-value index test is
; normally false; it pins the tie rule independently of the scan order.
;
; All validation completes before the first output write:
;   - k == 0 or k > 256 is rejected;
;   - every score is checked for NaN with FCMP x,x. An ordered value is
;     either EQ or NE, while unordered (NaN) matches no ordinary
;     predicate and falls through to the invalid return.
; Infinities are ordered and accepted. -0.0 equals +0.0 for ordering, so
; the smaller index wins that tie. On rejection indices_out and
; values_out are untouched.
;
; No host math is used. The kernel is a leaf: it contains no CALL and
; does not write r15. Its only private storage is the 256-word selected
; flag span at mth_topk_256_selected, so the kernel is non-reentrant and
; non-concurrent. scores is read-only.

mth_topk_256:
  ; Bounded k: 1..256. CMP is unsigned, so a negative word also fails
  ; the upper bound.
  CMP r2,r0
  JMP[EQ] @mth_topk_256_invalid
  MOVI r5,256
  CMP r2,r5
  JMP[GT] @mth_topk_256_invalid

  ; Preflight every score for NaN. FCMP x,x is EQ for every ordered value;
  ; an ordered non-equal value cannot occur here, but the NE branch keeps
  ; the test explicit. Unordered NaN matches neither and reaches invalid.
  MOVI r6,0
mth_topk_256_preflight:
  ADD r7,r1,r6
  LD r7,r7,0
  FCMP r7,r7
  JMP[EQ] @mth_topk_256_preflight_next
  JMP[NE] @mth_topk_256_preflight_next
  JMP @mth_topk_256_invalid
mth_topk_256_preflight_next:
  ADD r6,r6,r13
  MOVI r7,256
  CMP r6,r7
  JMP[LT] @mth_topk_256_preflight

  ; Clear the selected flags. Still no output has been written.
  LA r5,@mth_topk_256_selected
  MOVI r6,0
mth_topk_256_clear:
  ST r0,r5,0
  ADD r5,r5,r13
  ADD r6,r6,r13
  MOVI r7,256
  CMP r6,r7
  JMP[LT] @mth_topk_256_clear

  ; Output slot j.
  MOVI r5,0
mth_topk_256_slot:
  ; best_index = 256 sentinel, best_value = 0 placeholder.
  MOVI r7,256
  MOVI r8,0
  ; Scan i = 0..255 in increasing order.
  MOVI r6,0
mth_topk_256_scan:
  ; Skip an already selected index.
  LA r9,@mth_topk_256_selected
  ADD r10,r9,r6
  LD r10,r10,0
  CMP r10,r0
  JMP[NE] @mth_topk_256_scan_next

  ; candidate = scores[i]
  ADD r10,r1,r6
  LD r10,r10,0

  ; First unselected index becomes the current best.
  MOVI r11,256
  CMP r7,r11
  JMP[EQ] @mth_topk_256_take

  ; Ordered value comparison.
  FCMP r10,r8
  JMP[GT] @mth_topk_256_take
  JMP[EQ] @mth_topk_256_tie
  ; Remaining ordered case is candidate < best. Unordered cannot occur
  ; after the preflight, and this fall-through also skips it.
  JMP @mth_topk_256_scan_next

mth_topk_256_tie:
  ; Equal values: the smaller original index wins.
  CMP r6,r7
  JMP[GE] @mth_topk_256_scan_next
  JMP @mth_topk_256_take

mth_topk_256_take:
  ADD r7,r6,r0
  ADD r8,r10,r0

mth_topk_256_scan_next:
  ADD r6,r6,r13
  MOVI r11,256
  CMP r6,r11
  JMP[LT] @mth_topk_256_scan

  ; Mark the winner selected.
  LA r9,@mth_topk_256_selected
  ADD r9,r9,r7
  ST r13,r9,0

  ; indices_out[j] = best_index, values_out[j] = best_value.
  ADD r9,r3,r5
  ST r7,r9,0
  ADD r9,r4,r5
  ST r8,r9,0

  ADD r5,r5,r13
  CMP r5,r2
  JMP[LT] @mth_topk_256_slot

  ; Return the actual selected count (= the validated k).
  ADD r1,r2,r0
  RET

mth_topk_256_invalid:
  MOVI r1,0
  RET

; Private selected-index flags. 0 = still available, 1 = already emitted.
; Static storage makes the kernel non-reentrant and non-concurrent.
mth_topk_256_selected:
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
.word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
