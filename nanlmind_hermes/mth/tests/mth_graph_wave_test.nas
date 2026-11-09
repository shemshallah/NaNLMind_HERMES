; MTH graph wave velocity-Verlet fixture -- native NaN16 float32 only.
;
; Covers:
;   A. u=1, v=0, a=1, dt=1, gamma=0  -> u_next=1.5, v_next=1.0
;      (the requested 1.5/1.0 pair is exact for u=1; see the note below)
;   B. same with gamma=0.5            -> v_next=0.5
;   C. u=0, v=0, a=1, dt=1, gamma=0  -> u_next=0.5, v_next=1.0
;   J. non-uniform index-200 spot check -> u_next=4.0, v_next=3.0
;   K. dt=0.5, gamma=0.25, non-unit state -> u_next=2.75, v_next=1.75
;   L. dt=1, gamma=1 -> damping factor exactly 0 is accepted
;   D. dt=0 rejection with output atomicity
;   E. gamma=-0.5 rejection with output atomicity
;   F. damping factor 1-gamma*dt < 0 rejection with output atomicity
;   G. non-finite dt and gamma rejection with output atomicity
;   H. input/output overlap rejection with output/input atomicity
;   I. out-of-range output span rejection
;
; Note on the requested fixture numbers: with the specified update
;   v_half = v + 0.5*dt*a
;   u_next = u + dt*v_half
; the value u=0, v=0, a=1, dt=1 gives u_next=0.5, not 1.5. The exact
; 1.5/1.0 pair corresponds to u=1. Case A uses u=1 for that pair, and
; case C separately pins the u=0 result at the mathematically consistent
; 0.5. This was recorded in docs/MTH_GRAPH_WAVE_ABI.md rather than
; changing the kernel to match an inconsistent oracle.

.org 0
JMP @mth_wave_test_main
.include "../nan16/mth_graph_wave.nas"

mth_wave_test_main:
  MOVI r13,1

  ; ---------------- Case A: u=1, v=0, a=1, dt=1, gamma=0 ----------------
  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0                  ; gamma = 0.0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_un
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  CALL @mth_wave_test_expect

  ; ---------------- Case B: dt=1, gamma=0.5 -> v_next=0.5 ---------------
  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  MOVHI r7,0x3F00            ; gamma = 0.5
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F00            ; 0.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3F00            ; 0.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0x3F00            ; 0.5
  CALL @mth_wave_test_expect

  ; u_next is unaffected by the velocity damping factor.
  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect

  ; ---------------- Case C: u=0 -> u_next=0.5, v_next=1.0 ---------------
  MOVI r1,@mth_wave_u
  MOVI r2,256
  MOVI r3,0                  ; 0.0
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F00            ; 0.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3F00            ; 0.5
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  CALL @mth_wave_test_expect

  ; Restore u=1.0 for the next cases.
  MOVI r1,@mth_wave_u
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_fill

  ; ---------------- Case J: non-uniform spot value at index 200 ---------
  ; u[200]=2, v[200]=1, a[200]=2, dt=1, gamma=0 ->
  ; v_half = 1 + 0.5*2 = 2, u_next = 2 + 2 = 4, v_next = 3.
  ; All other indices keep u=1, v=0, a=1 -> u_next=1.5, v_next=1.0.
  MOVI r1,@mth_wave_u
  MOVI r2,200
  ADD r1,r1,r2
  MOVI r3,0
  MOVHI r3,0x4000            ; 2.0
  ST r3,r1,0

  MOVI r1,@mth_wave_v
  MOVI r2,200
  ADD r1,r1,r2
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  ST r3,r1,0

  MOVI r1,@mth_wave_a
  MOVI r2,200
  ADD r1,r1,r2
  MOVI r3,0
  MOVHI r3,0x4000            ; 2.0
  ST r3,r1,0

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x4080            ; 4.0
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x4040            ; 3.0
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_un
  MOVI r2,255
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect

  ; Restore uniform inputs before case K.
  MOVI r1,@mth_wave_u
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_v
  MOVI r2,256
  MOVI r3,0
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_a
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_fill

  ; ---------------- Case K: dt=0.5, gamma=0.25 --------------------------
  ; Index 0: u=2, v=1, a=2.
  ;   half_dt=0.25, t=0.5, v_half=1.5, u_next=2+0.5*1.5=2.75,
  ;   d=1-0.25*0.5=0.875, v_next=(1.5+0.5)*0.875=1.75.
  ; Index 200 stays uniform u=1, v=0, a=1:
  ;   u_next=1+0.5*0.25=1.125, v_next=(0.25+0.25)*0.875=0.4375.
  MOVI r1,@mth_wave_u
  MOVI r3,0
  MOVHI r3,0x4000            ; 2.0
  ST r3,r1,0
  MOVI r1,@mth_wave_v
  MOVI r3,0
  MOVHI r3,0x3F80            ; 1.0
  ST r3,r1,0
  MOVI r1,@mth_wave_a
  MOVI r3,0
  MOVHI r3,0x4000            ; 2.0
  ST r3,r1,0

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F00            ; dt = 0.5
  MOVI r7,0
  MOVHI r7,0x3E80            ; gamma = 0.25
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x4030            ; 2.75
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3FE0            ; 1.75
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3F90            ; 1.125
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,0
  MOVHI r3,0x3EE0            ; 0.4375
  CALL @mth_wave_test_expect

  ; Restore uniform u=1, v=0, a=1 for the rejection cases.
  MOVI r1,@mth_wave_u
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_v
  MOVI r2,256
  MOVI r3,0
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_a
  MOVI r2,256
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_fill

  ; ---------------- Case L: damping factor exactly 0 --------------------
  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  MOVHI r7,0x3F80            ; gamma = 1.0 -> d = 0
  CALL @mth_graph_wave_step_f32
  CMP r1,r13
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3FC0            ; 1.5
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,0                  ; 0.0
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,0                  ; 0.0
  CALL @mth_wave_test_expect

  ; ---------------- Case D: dt=0 rejection, outputs preserved -----------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,77
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,77
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0                  ; dt = 0.0
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r0
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,77
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,77
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_un
  MOVI r2,255
  MOVI r3,77
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,77
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,77
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,255
  MOVI r3,77
  CALL @mth_wave_test_expect

  ; ---------------- Case E: gamma<0 rejection ---------------------------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,88
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,88
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  MOVHI r7,0xBF00            ; gamma = -0.5
  CALL @mth_graph_wave_step_f32
  CMP r1,r0
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,88
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,255
  MOVI r3,88
  CALL @mth_wave_test_expect

  ; ---------------- Case F: damping factor <0 rejection ----------------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,99
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,99
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  MOVHI r7,0x4000            ; gamma = 2.0 -> d = -1.0
  CALL @mth_graph_wave_step_f32
  CMP r1,r0
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,200
  MOVI r3,99
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,200
  MOVI r3,99
  CALL @mth_wave_test_expect

  ; ---------------- Case G1: non-finite dt rejection -------------------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,111
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,111
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x7F80            ; dt = +inf
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  CMP r1,r0
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,111
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,111
  CALL @mth_wave_test_expect

  ; ---------------- Case G2: non-finite gamma rejection ----------------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,112
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,112
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_un
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  MOVHI r7,0x7F80            ; gamma = +inf
  CALL @mth_graph_wave_step_f32
  CMP r1,r0
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,112
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,112
  CALL @mth_wave_test_expect

  ; ---------------- Case H: input/output overlap rejection -------------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,123
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,123
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,@mth_wave_u         ; u_next overlaps u on purpose
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  MOVI r2,2
  CMP r1,r2
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_un
  MOVI r2,0
  MOVI r3,123
  CALL @mth_wave_test_expect
  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,123
  CALL @mth_wave_test_expect

  ; The input u must also be untouched (u[0] was restored to 1.0).
  MOVI r1,@mth_wave_u
  MOVI r2,0
  MOVI r3,0
  MOVHI r3,0x3F80
  CALL @mth_wave_test_expect

  ; ---------------- Case I: out-of-range output span rejection ---------
  MOVI r1,@mth_wave_un
  MOVI r2,256
  MOVI r3,124
  CALL @mth_wave_test_fill
  MOVI r1,@mth_wave_vn
  MOVI r2,256
  MOVI r3,124
  CALL @mth_wave_test_fill

  MOVI r1,@mth_wave_u
  MOVI r2,@mth_wave_v
  MOVI r3,@mth_wave_a
  MOVI r4,0xFFFF
  MOVHI r4,0x3F              ; r4 = 0x003FFFFF, cannot hold 256 words
  MOVI r5,@mth_wave_vn
  MOVI r6,0
  MOVHI r6,0x3F80            ; dt = 1.0
  MOVI r7,0
  CALL @mth_graph_wave_step_f32
  MOVI r2,2
  CMP r1,r2
  JMP[NE] @mth_wave_test_fail

  MOVI r1,@mth_wave_vn
  MOVI r2,0
  MOVI r3,124
  CALL @mth_wave_test_expect

  MOVI r1,@mth_wave_test_pass
  CALL @mth_wave_test_puts
  HALT

mth_wave_test_fail:
  MOVI r1,@mth_wave_test_bad
  CALL @mth_wave_test_puts
  HALT

; r1 = base, r2 = index, r3 = expected 32-bit payload; jumps to fail on mismatch.
mth_wave_test_expect:
  ADD r4,r1,r2
  LD r4,r4,0
  CMP r4,r3
  JMP[NE] @mth_wave_test_fail
  RET

; r1 = base, r2 = count, r3 = 32-bit payload; fills count words.
mth_wave_test_fill:
  CMP r2,r0
  RET[EQ]
  ST r3,r1,0
  ADD r1,r1,r13
  SUB r2,r2,r13
  JMP @mth_wave_test_fill

mth_wave_test_puts:
  MOVI r4,255
mth_wave_test_puts_word:
  LD r2,r1,0
  MOVI r3,4
mth_wave_test_puts_byte:
  BOOL.8 r5,r2,r4
  CMP r5,r0
  RET[EQ]
  PUTC r5
  SHR r2,r2,8
  SUB r3,r3,r13
  JMP[NE] @mth_wave_test_puts_byte
  ADD r1,r1,r13
  JMP @mth_wave_test_puts_word

; Five fixed, pairwise-disjoint 256-word float32 fixture regions.
; u and a are all 1.0; v is all 0.0; u_next/v_next start as 77 sentinels
; and are overwritten by the successful cases. Runtime fills reset the
; spans for the rejection cases.
.org 2048
mth_wave_u:
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000

.org 2304
mth_wave_v:
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
.word 0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000

.org 2560
mth_wave_a:
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000
.word 0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000,0x3F800000

.org 2816
mth_wave_un:
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77

.org 3072
mth_wave_vn:
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77
.word 77,77,77,77,77,77,77,77,77,77,77,77,77,77,77,77

.org 3328
mth_wave_test_pass:
.str "MTH GRAPH WAVE STEP PASS\n"
mth_wave_test_bad:
.str "MTH GRAPH WAVE STEP FAIL\n"
