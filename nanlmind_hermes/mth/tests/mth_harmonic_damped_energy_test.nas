.org 0
JMP @mth_harmonic_wave_test_main
.include "../nan16/mth_harmonic_accel_256.nas"
.include "../nan16/mth_graph_wave.nas"
.include "../nan16/mth_graph_step_energy_256.nas"

mth_harmonic_wave_test_main:
  MOVI r13,1

  ; u=1.0, v=0.0.
  MOVI r1,@mth_harmonic_u
  MOVI r6,256
  MOVI r7,0
  MOVHI r7,0x3F80
mth_harmonic_fill_u:
  ST r7,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_harmonic_fill_u

  MOVI r2,@mth_harmonic_v
  MOVI r6,256
mth_harmonic_fill_v:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_harmonic_fill_v

  ; Four velocity-Verlet steps with a = -u.
  MOVI r6,4
  MOVI r8,@mth_harmonic_steps
  ST r6,r8,0
mth_harmonic_step_loop:
  ; a = -k*u.
  MOVI r1,@mth_harmonic_u
  MOVI r2,@mth_harmonic_k
  MOVI r3,@mth_harmonic_a
  CALL @mth_harmonic_accel_256_f32
  CMP r1,r13
  JMP[NE] @mth_harmonic_fail

  ; One wave step.
  MOVI r1,@mth_harmonic_u
  MOVI r2,@mth_harmonic_v
  MOVI r3,@mth_harmonic_a
  MOVI r4,@mth_harmonic_u_next
  MOVI r5,@mth_harmonic_v_next
  MOVI r6,@mth_harmonic_dt
  LD r6,r6,0
  MOVI r7,@mth_harmonic_gamma
  LD r7,r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_harmonic_fail

  ; Copy u_next/v_next back to u/v.
  MOVI r1,@mth_harmonic_u_next
  MOVI r2,@mth_harmonic_u
  CALL @mth_harmonic_copy256
  MOVI r1,@mth_harmonic_v_next
  MOVI r2,@mth_harmonic_v
  CALL @mth_harmonic_copy256

  MOVI r6,@mth_harmonic_steps
  LD r7,r6,0
  SUB r7,r7,r13
  ST r7,r6,0
  CMP r7,r0
  JMP[GT] @mth_harmonic_step_loop

  ; Final energy with a=-u. Total should remain near 128.0.
  MOVI r1,@mth_harmonic_u
  MOVI r2,@mth_harmonic_k
  MOVI r3,@mth_harmonic_a
  CALL @mth_harmonic_accel_256_f32
  CMP r1,r13
  JMP[NE] @mth_harmonic_fail

  MOVI r1,@mth_harmonic_u
  MOVI r2,@mth_harmonic_v
  MOVI r3,@mth_harmonic_a
  MOVI r4,@mth_harmonic_energy
  CALL @mth_graph_step_energy_256_f32
  CMP r1,r13
  JMP[NE] @mth_harmonic_fail

  MOVI r4,@mth_harmonic_energy
  MOVI r6,2
  ADD r4,r4,r6
  LD r5,r4,0
  ; Damped total must be strictly below 128.0. For positive float32,
  ; unsigned bit order matches numeric order, so use integer CMP.
  MOVI r6,0
  MOVHI r6,0x4300
  CMP r5,r6
  JMP[LT] @mth_harmonic_ok
  JMP @mth_harmonic_fail

mth_harmonic_ok:
  MOVI r1,@mth_harmonic_pass
  CALL @mth_harmonic_puts
  HALT

mth_harmonic_fail:
  MOVI r1,@mth_harmonic_bad
  CALL @mth_harmonic_puts
  HALT

; r1=source, r2=dest; 256 float32 words.
mth_harmonic_copy256:
  MOVI r3,256
mth_harmonic_copy_loop:
  LD r4,r1,0
  ST r4,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_harmonic_copy_loop
  RET

mth_harmonic_puts:
  MOVI r4,255
mth_harmonic_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_harmonic_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_harmonic_puts_byte
  ADD r1,r1,r13
  JMP @mth_harmonic_puts_word

.org 1024
mth_harmonic_u:
.word 0
.org 1536
mth_harmonic_v:
.word 0
.org 2048
mth_harmonic_a:
.word 0
.org 2560
mth_harmonic_u_next:
.word 0
.org 3072
mth_harmonic_v_next:
.word 0
.org 3584
mth_harmonic_energy:
.word 0
.org 3648
mth_harmonic_k:
.word 0x3F800000
.org 3712
mth_harmonic_dt:
.word 0x3DCCCCCD
.org 3776
mth_harmonic_gamma:
.word 0x3DCCCCCD
.org 3840
mth_harmonic_steps:
.word 0
.org 3904
mth_harmonic_pass:
.str "MTH HARMONIC DAMPED ENERGY PASS\n"
mth_harmonic_bad:
.str "MTH HARMONIC DAMPED ENERGY FAIL\n"
