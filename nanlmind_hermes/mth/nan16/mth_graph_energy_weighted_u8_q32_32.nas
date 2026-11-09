; MTH exact weighted graph energy for 256-vertex symmetric CSR — NaN16 only.
; E(x) = sum over undirected edges {i,j} of w_ij * (x_i-x_j)^2.
;
; ABI: r1=row_offsets, r2=neighbors, r3=weights, r4=x,
;      r5=output (U64 Q32.32 low word then high word),
;      r6=edge_count, r7=caller-owned 16-word U32 scratch, r13=1.
; State is signed Q16.16 in [-1,+1]; weights are U8 integer [1,255].
; Returns r1=1 success, r1=0 malformed descriptor/state/weight.
; Uses no CALLs and no static scratch; all validation precedes output writes.
;
; Maximum raw difference=2^17; per-edge weighted square <=255*2^34.
; At most C(256,2)=32640 undirected edges:
; E_raw <= 142,991,487,192,268,800 < 2^57 < 2^64.
; Output is exact unsigned U64 Q32.32, no rounding.

mth_graph_energy_weighted_u8_q32_32:
  ADD r11,r7,r0
  ST r1,r11,0
  ST r2,r11,1
  ST r3,r11,2
  ST r4,r11,3
  ST r5,r11,4
  ST r6,r11,5

  CMP r6,r0
  JMP[LT] @mth_energy_invalid
  MOVI r8,65280
  CMP r6,r8
  JMP[GT] @mth_energy_invalid

  ; State preflight: signed Q16.16 in [-1,+1].
  ADD r8,r4,r0
  MOVI r9,256
mth_energy_validate_x:
  LD r10,r8,0
  SHR r12,r10,17
  CMP r12,r0
  JMP[EQ] @mth_energy_x_positive
  MOVI r1,32767
  CMP r12,r1
  JMP[EQ] @mth_energy_x_valid
  JMP @mth_energy_invalid
mth_energy_x_positive:
  MOVI r1,0
  MOVHI r1,1
  CMP r10,r1
  JMP[GT] @mth_energy_invalid
mth_energy_x_valid:
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_energy_validate_x

  ; Offset preflight, including offset[0]=0 and final edge_count.
  LD r8,r11,0
  LD r12,r8,0
  CMP r12,r0
  JMP[NE] @mth_energy_invalid
  MOVI r9,257
  MOVI r10,0
mth_energy_validate_offsets:
  LD r12,r8,0
  CMP r12,r10
  JMP[LT] @mth_energy_invalid
  LD r1,r11,5
  CMP r12,r1
  JMP[GT] @mth_energy_invalid
  SUB r1,r12,r10
  MOVI r2,255
  CMP r1,r2
  JMP[GT] @mth_energy_invalid
  ADD r10,r12,r0
  ADD r8,r8,r13
  SUB r9,r9,r13
  CMP r9,r0
  JMP[GT] @mth_energy_validate_offsets
  LD r1,r11,5
  CMP r10,r1
  JMP[NE] @mth_energy_invalid

  ; Validate sorted unique non-self endpoints and canonical nonzero U8 weights.
  MOVI r6,0
mth_energy_validate_row:
  LD r1,r11,0
  ADD r1,r1,r6
  LD r7,r1,0
  ADD r1,r1,r13
  LD r8,r1,0
  ADD r9,r7,r0
  MOVI r10,0
mth_energy_validate_edge:
  CMP r9,r8
  JMP[EQ] @mth_energy_next_validate_row
  LD r1,r11,1
  ADD r1,r1,r9
  LD r12,r1,0
  SHR r1,r12,8
  CMP r1,r0
  JMP[NE] @mth_energy_invalid
  MOVI r1,256
  CMP r12,r1
  JMP[LT] @mth_energy_endpoint_ok
  JMP @mth_energy_invalid
mth_energy_endpoint_ok:
  CMP r12,r6
  JMP[EQ] @mth_energy_invalid
  CMP r9,r7
  JMP[EQ] @mth_energy_first_neighbor
  CMP r12,r10
  JMP[GT] @mth_energy_sorted_neighbor
  JMP @mth_energy_invalid
mth_energy_sorted_neighbor:
mth_energy_first_neighbor:
  ADD r10,r12,r0
  LD r1,r11,2
  ADD r1,r1,r9
  LD r1,r1,0
  SHR r12,r1,8
  CMP r12,r0
  JMP[NE] @mth_energy_invalid
  CMP r1,r0
  JMP[EQ] @mth_energy_invalid
  ADD r9,r9,r13
  JMP @mth_energy_validate_edge
mth_energy_next_validate_row:
  ADD r6,r6,r13
  MOVI r1,256
  CMP r6,r1
  JMP[LT] @mth_energy_validate_row

  ; Zero the U64 energy accumulator and begin exact edge reduction.
  MOVI r8,0
  ST r8,r11,10
  ST r8,r11,11
  MOVI r8,0
  ST r8,r11,6
mth_energy_apply_row:
  LD r8,r11,6
  LD r1,r11,0
  ADD r1,r1,r8
  LD r9,r1,0
  ADD r1,r1,r13
  LD r12,r1,0
  ST r9,r11,7
  ST r12,r11,8
  LD r1,r11,3
  ADD r1,r1,r8
  LD r10,r1,0
  ST r10,r11,9

mth_energy_apply_edge:
  LD r8,r11,7
  LD r9,r11,8
  CMP r8,r9
  JMP[EQ] @mth_energy_next_apply_row
  LD r1,r11,1
  ADD r1,r1,r8
  LD r10,r1,0
  LD r6,r11,6
  CMP r10,r6
  JMP[GT] @mth_energy_process_edge
  JMP @mth_energy_next_edge

mth_energy_process_edge:
  LD r1,r11,3
  ADD r1,r1,r10
  LD r3,r1,0
  LD r4,r11,9
  SUB r1,r4,r3
  SHR r2,r1,31
  CMP r2,r0
  JMP[EQ] @mth_energy_abs_ready
  SUB r1,r0,r1
mth_energy_abs_ready:
  ; Exact square via unsigned 32x32 -> 64 shift/add.
  MOVI r3,0
  MOVI r4,0
  MOVI r5,0
  ADD r6,r1,r0
  ADD r7,r1,r0
  MOVI r8,32
mth_energy_square_loop:
  MOVI r10,1
  BOOL.8 r9,r7,r10
  CMP r9,r0
  JMP[EQ] @mth_energy_square_shift
  ADD r10,r4,r0
  ADD r4,r4,r6
  CMP r4,r10
  JMP[LT] @mth_energy_square_carry
  ADD r3,r3,r5
  JMP @mth_energy_square_shift
mth_energy_square_carry:
  ADD r3,r3,r5
  ADD r3,r3,r13
mth_energy_square_shift:
  SHR r9,r6,31
  SHL r6,r6,1
  SHL r5,r5,1
  BOOL.14 r5,r5,r9
  SHR r7,r7,1
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_energy_square_loop

  ; Multiply the 64-bit square by the U8 edge weight (8 shift/add rounds).
  LD r1,r11,2
  LD r2,r11,7
  ADD r1,r1,r2
  LD r7,r1,0
  MOVI r5,0
  MOVI r6,0
  MOVI r8,8
mth_energy_weight_loop:
  MOVI r10,1
  BOOL.8 r9,r7,r10
  CMP r9,r0
  JMP[EQ] @mth_energy_weight_shift
  ADD r10,r5,r0
  ADD r5,r5,r4
  CMP r5,r10
  JMP[LT] @mth_energy_weight_carry
  ADD r6,r6,r3
  JMP @mth_energy_weight_shift
mth_energy_weight_carry:
  ADD r6,r6,r3
  ADD r6,r6,r13
mth_energy_weight_shift:
  SHR r9,r4,31
  SHL r4,r4,1
  SHL r3,r3,1
  BOOL.14 r3,r3,r9
  SHR r7,r7,1
  SUB r8,r8,r13
  CMP r8,r0
  JMP[GT] @mth_energy_weight_loop

  ; Add weighted square into the U64 total.
  LD r9,r11,10
  LD r10,r11,11
  ADD r1,r9,r0
  ADD r9,r9,r5
  CMP r9,r1
  JMP[LT] @mth_energy_acc_carry
  ADD r10,r10,r6
  JMP @mth_energy_acc_store
mth_energy_acc_carry:
  ADD r10,r10,r13
  ADD r10,r10,r6
mth_energy_acc_store:
  ST r9,r11,10
  ST r10,r11,11

mth_energy_next_edge:
  LD r8,r11,7
  ADD r8,r8,r13
  ST r8,r11,7
  JMP @mth_energy_apply_edge

mth_energy_next_apply_row:
  LD r8,r11,6
  ADD r8,r8,r13
  ST r8,r11,6
  MOVI r9,256
  CMP r8,r9
  JMP[LT] @mth_energy_apply_row

  ; Commit the scalar only after the complete exact reduction.
  LD r1,r11,4
  LD r8,r11,10
  ST r8,r1,0
  LD r9,r11,11
  ST r9,r1,1
  MOVI r1,1
  RET

mth_energy_invalid:
  MOVI r1,0
  RET
