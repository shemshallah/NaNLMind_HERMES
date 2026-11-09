; MTH graph-H wave energy fixture — native NaN16 float32 only.
;
; H is the exact 256-position ring Laplacian:
;   (L u)[i] = 2*u[i] - u[(i-1) mod 256] - u[(i+1) mod 256].
; Initial state: u = impulse [1,0,...,0], v = 0, dt = 0.1, gamma = 0.
; The initial total energy is 0.5*u^T L u = 1.0.
;
; Four velocity-Verlet steps are composed natively:
;   Lu = L u; a = -Lu; wave step; copy u_next/v_next back to u/v.
; After the fourth step Lu, a=-Lu and the step energy are recomputed.
; The final total must lie in [0.9, 1.1], i.e. within 0.1 of 1.0.
;
; Predicate note for this VM/toolchain: after FCMP A,B, JMP[GT] executes when
; A > B (verified with a tiny integer and float probe on the canonical
; nan_cli). Therefore "reject above" is FCMP total,upper; GT and
; "reject below" is FCMP lower,total; GT. No host math is used.
;
; Step budget for the runner: 20,000,000.

.org 0
JMP @mth_ring_wave_main
.include "../nan16/mth_ring256_laplacian.nas"
.include "../nan16/mth_negate256_f32.nas"
.include "../nan16/mth_graph_wave.nas"
.include "../nan16/mth_graph_step_energy_256.nas"

mth_ring_wave_main:
  MOVI r13,1

  ; u = 0, v = 0.
  MOVI r1,@mth_ring_wave_u
  MOVI r6,256
mth_ring_wave_zero_u:
  ST r0,r1,0
  ADD r1,r1,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_ring_wave_zero_u

  MOVI r2,@mth_ring_wave_v
  MOVI r6,256
mth_ring_wave_zero_v:
  ST r0,r2,0
  ADD r2,r2,r13
  SUB r6,r6,r13
  CMP r6,r0
  JMP[GT] @mth_ring_wave_zero_v

  ; u[0] = 1.0: unit impulse at ring position 0.
  MOVI r1,@mth_ring_wave_u
  MOVI r6,0
  MOVHI r6,0x3F80
  ST r6,r1,0

  ; Four steps.
  MOVI r6,@mth_ring_wave_steps
  MOVI r7,4
  ST r7,r6,0

  ; Initial total energy check: 0.5*u^T L u must be exactly 1.0.
  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_lu
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  MOVI r1,@mth_ring_wave_lu
  MOVI r2,@mth_ring_wave_a
  CALL @mth_negate256_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_v
  MOVI r3,@mth_ring_wave_a
  MOVI r4,@mth_ring_wave_energy
  CALL @mth_graph_step_energy_256_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  MOVI r4,@mth_ring_wave_energy
  LD r5,r4,2
  MOVI r6,0
  MOVHI r6,0x3F80
  CMP r5,r6
  JMP[NE] @mth_ring_wave_fail

mth_ring_wave_step_loop:
  ; Lu = L u.
  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_lu
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  ; a = -Lu.
  MOVI r1,@mth_ring_wave_lu
  MOVI r2,@mth_ring_wave_a
  CALL @mth_negate256_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  ; One velocity-Verlet step with the supplied a = -H u.
  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_v
  MOVI r3,@mth_ring_wave_a
  MOVI r4,@mth_ring_wave_u_next
  MOVI r5,@mth_ring_wave_v_next
  MOVI r6,@mth_ring_wave_dt
  LD r6,r6,0
  MOVI r7,@mth_ring_wave_gamma
  LD r7,r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  ; Copy u_next/v_next back to u/v.
  MOVI r1,@mth_ring_wave_u_next
  MOVI r2,@mth_ring_wave_u
  CALL @mth_ring_wave_copy256
  MOVI r1,@mth_ring_wave_v_next
  MOVI r2,@mth_ring_wave_v
  CALL @mth_ring_wave_copy256

  MOVI r6,@mth_ring_wave_steps
  LD r7,r6,0
  SUB r7,r7,r13
  ST r7,r6,0
  CMP r7,r0
  JMP[GT] @mth_ring_wave_step_loop

  ; Final Lu, a=-Lu, and energy.
  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_lu
  CALL @mth_ring256_laplacian_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  MOVI r1,@mth_ring_wave_lu
  MOVI r2,@mth_ring_wave_a
  CALL @mth_negate256_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  MOVI r1,@mth_ring_wave_u
  MOVI r2,@mth_ring_wave_v
  MOVI r3,@mth_ring_wave_a
  MOVI r4,@mth_ring_wave_energy
  CALL @mth_graph_step_energy_256_f32
  CMP r1,r13
  JMP[NE] @mth_ring_wave_fail

  ; total = energy[2].
  MOVI r4,@mth_ring_wave_energy
  LD r5,r4,2

  ; Reject total > 1.1 (upper = 0x3F8CCCCD).
  MOVI r6,0xCCCD
  MOVHI r6,0x3F8C
  FCMP r5,r6
  JMP[GT] @mth_ring_wave_fail

  ; Reject total < 0.9 (lower = 0x3F666666).
  MOVI r6,0x6666
  MOVHI r6,0x3F66
  FCMP r6,r5
  JMP[GT] @mth_ring_wave_fail

  MOVI r1,@mth_ring_wave_pass
  CALL @mth_ring_wave_puts
  HALT

mth_ring_wave_fail:
  MOVI r1,@mth_ring_wave_bad
  CALL @mth_ring_wave_puts
  HALT

; r1=source, r2=dest; 256 float32 words. Clobbers r3-r4.
mth_ring_wave_copy256:
  MOVI r3,256
mth_ring_wave_copy_loop:
  LD r4,r1,0
  ST r4,r2,0
  ADD r1,r1,r13
  ADD r2,r2,r13
  SUB r3,r3,r13
  CMP r3,r0
  JMP[GT] @mth_ring_wave_copy_loop
  RET

mth_ring_wave_puts:
  MOVI r4,255
mth_ring_wave_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_ring_wave_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_ring_wave_puts_byte
  ADD r1,r1,r13
  JMP @mth_ring_wave_puts_word

.org 1024
mth_ring_wave_u:
.word 0
.org 2048
mth_ring_wave_v:
.word 0
.org 3072
mth_ring_wave_lu:
.word 0
.org 4096
mth_ring_wave_a:
.word 0
.org 5120
mth_ring_wave_u_next:
.word 0
.org 6144
mth_ring_wave_v_next:
.word 0
.org 7168
mth_ring_wave_energy:
.word 0
.org 7232
mth_ring_wave_dt:
.word 0x3DCCCCCD
.org 7233
mth_ring_wave_gamma:
.word 0
.org 7234
mth_ring_wave_steps:
.word 0
.org 7296
mth_ring_wave_pass:
.str "MTH RING WAVE ENERGY PASS\n"
mth_ring_wave_bad:
.str "MTH RING WAVE ENERGY FAIL\n"
