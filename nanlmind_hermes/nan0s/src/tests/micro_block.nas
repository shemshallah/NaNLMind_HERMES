; micro_block.nas - 0xNaN16 raw micro-transformer forward primitive.
; Two tokens, identity Q/K/V, 2-way softmax via FEXP, context mix, tanh MLP.
; No host math: every step is a NaNVM1 op in the one machine.
MOVI r13,1
; x0 = 1.0, x1 = 0.5
MOVI r1,0
MOVHI r1,16256
MOVI r2,0
MOVHI r2,16128
; identity wq = wk = wv = 1.0
MOVI r3,0
MOVHI r3,16256
FMUL r4,r1,r3      ; q0 = x0*wq
FMUL r5,r2,r3      ; q1 = x1*wq
FMUL r6,r4,r4      ; score0 = q0*k0
FMUL r7,r5,r5      ; score1 = q1*k1
FEXP r8,r6
FEXP r9,r7
FADD r10,r8,r9
FDIV r11,r8,r10    ; attention alpha0
FDIV r12,r9,r10    ; attention alpha1
FMUL r4,r11,r1     ; context0
FMUL r5,r12,r2     ; context1
FADD r4,r4,r5      ; context readout
FTANH r4,r4        ; MLP activation
CVTFI r4,r4
MOVI r5,48
ADD r5,r5,r4
PUTC r5
MOVI r5,10
PUTC r5
HALT
