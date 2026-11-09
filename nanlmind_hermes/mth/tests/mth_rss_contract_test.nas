; MTH RSS typed descriptor/contract checker native fixture — NaN16 only.
;
; Cases:
;   1. allow CSR(256,E=6)->vector(256) QUERY, Q16.16, exact bounds;
;   2. allow vector(128)->vector(128) FILTER, integer scale 1;
;   3. deny null descriptor;
;   4. deny descriptor version mismatch;
;   5. deny operation-id mismatch;
;   6. deny kind mismatch;
;   7. deny domain mismatch;
;   8. deny dimension above contract maximum;
;   9. deny CSR shape mismatch;
;  10. deny undersized input payload;
;  11. deny undersized output payload;
;  12. deny unsupported scale policy;
;  13. deny missing required policy flag;
;  14. deny forbidden policy flag;
;  15. deny numeric bound above contract maximum;
;  16. deny malformed contract (max_dim0=0).
;
; The checker must not execute the operation or mutate either descriptor.

.org 0
JMP @mth_rss_test_main
.include "../nan16/mth_rss_contract.nas"

mth_rss_test_main:
  MOVI r13,1

  ; ---- Case 1: allow CSR -> vector --------------------------------------
  MOVI r1,@mth_rss_msg_allow
  CALL @mth_rss_puts
  CALL @mth_rss_setup_csr
  CALL @mth_rss_call
  CMP r1,r13
  JMP[NE] @mth_rss_test_fail
  CMP r2,r0
  JMP[NE] @mth_rss_test_fail
  MOVI r1,@mth_rss_msg_allow_ok
  CALL @mth_rss_puts

  ; ---- Case 2: allow vector -> vector, integer scale --------------------
  MOVI r1,@mth_rss_msg_vec
  CALL @mth_rss_puts
  CALL @mth_rss_setup_vector
  CALL @mth_rss_call
  CMP r1,r13
  JMP[NE] @mth_rss_test_fail
  CMP r2,r0
  JMP[NE] @mth_rss_test_fail
  MOVI r1,@mth_rss_msg_vec_ok
  CALL @mth_rss_puts

  ; ---- Case 3: null descriptor ------------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,0
  MOVI r2,@mth_rss_contract_work
  CALL @mth_rss_contract_check
  MOVI r3,1
  CALL @mth_rss_expect_deny

  ; ---- Case 4: descriptor version mismatch ------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,2
  ST r2,r1,0
  CALL @mth_rss_call
  MOVI r3,2
  CALL @mth_rss_expect_deny

  ; ---- Case 5: operation-id mismatch ------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,3
  ST r2,r1,1
  CALL @mth_rss_call
  MOVI r3,3
  CALL @mth_rss_expect_deny

  ; ---- Case 6: kind mismatch --------------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,2
  ST r2,r1,2
  CALL @mth_rss_call
  MOVI r3,4
  CALL @mth_rss_expect_deny

  ; ---- Case 7: domain mismatch ------------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,3
  ST r2,r1,4
  CALL @mth_rss_call
  MOVI r3,5
  CALL @mth_rss_expect_deny

  ; ---- Case 8: dimension above maximum ----------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,257
  ST r2,r1,5
  CALL @mth_rss_call
  MOVI r3,6
  CALL @mth_rss_expect_deny

  ; ---- Case 9: CSR shape mismatch ---------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,255
  ST r2,r1,7
  CALL @mth_rss_call
  MOVI r3,7
  CALL @mth_rss_expect_deny

  ; ---- Case 10: undersized input payload --------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,268
  ST r2,r1,9
  CALL @mth_rss_call
  MOVI r3,8
  CALL @mth_rss_expect_deny

  ; ---- Case 11: undersized output payload -------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,255
  ST r2,r1,10
  CALL @mth_rss_call
  MOVI r3,8
  CALL @mth_rss_expect_deny

  ; ---- Case 12: unsupported scale policy --------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,3
  ST r2,r1,11
  CALL @mth_rss_call
  MOVI r3,9
  CALL @mth_rss_expect_deny

  ; ---- Case 13: missing required policy flag ----------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  ST r0,r1,12
  CALL @mth_rss_call
  MOVI r3,10
  CALL @mth_rss_expect_deny

  ; ---- Case 14: forbidden policy flag -----------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,5
  ST r2,r1,12
  CALL @mth_rss_call
  MOVI r3,10
  CALL @mth_rss_expect_deny

  ; ---- Case 15: bound above contract maximum ----------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_desc_work
  MOVI r2,1
  MOVHI r2,0x0040                ; 0x00400001
  ST r2,r1,13
  CALL @mth_rss_call
  MOVI r3,11
  CALL @mth_rss_expect_deny

  ; ---- Case 16: malformed contract --------------------------------------
  CALL @mth_rss_setup_csr
  MOVI r1,@mth_rss_contract_work
  ST r0,r1,5                    ; max_dim0 = 0
  CALL @mth_rss_call
  MOVI r3,12
  CALL @mth_rss_expect_deny

  MOVI r1,@mth_rss_msg_pass
  CALL @mth_rss_puts
  HALT

mth_rss_test_fail:
  MOVI r1,@mth_rss_msg_fail
  CALL @mth_rss_puts
  HALT


; r1/r2 hold the checker result; expected deny reason is in r3.
mth_rss_expect_deny:
  CMP r1,r0
  JMP[NE] @mth_rss_test_fail
  CMP r2,r3
  JMP[NE] @mth_rss_test_fail
  RET

; r1=dst, r2=src; copies 16 words.
mth_rss_copy16:
  MOVI r3,16
mth_rss_copy16_loop:
  LD r4,r2,0
  ST r4,r1,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_rss_copy16_loop
  RET

mth_rss_setup_csr:
  ADD r14,r15,r0
  MOVI r1,@mth_rss_contract_work
  MOVI r2,@mth_rss_contract_template
  CALL @mth_rss_copy16
  MOVI r1,@mth_rss_desc_work
  MOVI r2,@mth_rss_desc_template
  CALL @mth_rss_copy16
  ADD r15,r14,r0
  RET

mth_rss_setup_vector:
  ADD r14,r15,r0
  MOVI r1,@mth_rss_contract_work
  MOVI r2,@mth_rss_contract_vector
  CALL @mth_rss_copy16
  MOVI r1,@mth_rss_desc_work
  MOVI r2,@mth_rss_desc_vector
  CALL @mth_rss_copy16
  ADD r15,r14,r0
  RET

; Call the checker on the work copies.  The kernel preserves r15, but this
; wrapper must not keep its own return link in a kernel-clobbered register.
mth_rss_call:
  ST r15,r0,@mth_rss_link_save
  MOVI r1,@mth_rss_desc_work
  MOVI r2,@mth_rss_contract_work
  CALL @mth_rss_contract_check
  LD r15,r0,@mth_rss_link_save
  RET

mth_rss_puts:
  MOVI r4,255
mth_rss_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_rss_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_rss_puts_byte
  ADD r1,r1,r13
  JMP @mth_rss_puts_word


.org 2048
mth_rss_desc_work:
.org 2304
mth_rss_contract_work:
.org 2560
mth_rss_link_save:
.word 0
mth_rss_contract_template:
.word 1
.word 2
.word 3
.word 1
.word 2
.word 256
.word 65280
.word 131072
.word 256
.word 3
.word 1
.word 4
.word 4194304
.word 4
.word 0
.word 0
mth_rss_desc_template:
.word 1
.word 2
.word 3
.word 1
.word 2
.word 256
.word 6
.word 256
.word 0
.word 269
.word 256
.word 0
.word 1
.word 4194304
.word 4194304
.word 0
mth_rss_contract_vector:
.word 1
.word 4
.word 1
.word 1
.word 3
.word 256
.word 0
.word 256
.word 256
.word 3
.word 1
.word 4
.word 1000000
.word 1
.word 0
.word 0
mth_rss_desc_vector:
.word 1
.word 4
.word 1
.word 1
.word 3
.word 128
.word 0
.word 128
.word 0
.word 128
.word 128
.word 1
.word 1
.word 1000
.word 1000
.word 0

.org 3000
mth_rss_msg_allow:
.str "RSS ALLOW "
mth_rss_msg_allow_ok:
.str "ALLOW OK\n"
mth_rss_msg_vec:
.str "RSS VEC "
mth_rss_msg_vec_ok:
.str "VEC OK\n"
mth_rss_msg_pass:
.str "MTH RSS CONTRACT PASS\n"
mth_rss_msg_fail:
.str "MTH RSS CONTRACT FAIL\n"
