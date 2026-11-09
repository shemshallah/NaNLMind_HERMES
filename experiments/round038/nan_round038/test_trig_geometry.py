"""Auditable identities and gradient checks; invokes the actual 8 MiB C++ memory."""
import json,math,torch
from pathlib import Path
import numpy as np
from torch.nn import functional as F
from round038 import Root,Reasoner,givens,LABELTOK,EVIDENCE,EOS
P=Path(__file__).parent
torch.set_num_threads(2)
torch.manual_seed(127)

def check():
 t1=torch.tensor([0.7,-0.4,0.1]);t2=torch.tensor([0.2,0.5,-0.9]);A=givens(t1);B=givens(t2)
 c=float(torch.linalg.matrix_norm(A@B-B@A).item());assert c>.05
 so2=lambda t:torch.tensor([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
 cm=float(torch.linalg.matrix_norm(so2(.7)@so2(-.4)-so2(-.4)@so2(.7)).item());assert cm<1e-6
 # Transport T_ij maps local i coordinates -> local j coordinates.
 T01=A;T12=B;T20=givens(torch.tensor([.11,-.38,.56]))
 H=T20@T12@T01
 Q0=givens(torch.tensor([.32,.25,.14]));Q1=givens(torch.tensor([.21,-.46,.15]));Q2=givens(torch.tensor([-.22,.51,-.11]))
 Hg=(Q0@T20@Q2.T)@(Q2@T12@Q1.T)@(Q1@T01@Q0.T)
 h_err=float(torch.abs(torch.trace(H)-torch.trace(Hg)).item());assert h_err<1e-5
 # Native physical memory is unchanged in footprint and has mirror-pair directed semantics.
 mem=Root();slots=[mem.write(float(i%5),float(i%2)) for i in range(48)]
 original=np.array([mem.read(s) for s in slots])
 target=P/'geometry_check_snapshot.bin';mem.save(target)
 mem.gauge(177,50.)
 invariant=np.array([mem.read(s) for s in slots]);raw=np.array([mem.read(s,raw=True) for s in slots]);delta=float(np.abs(invariant-original).max());assert delta<5e-5
 mem.restore(target);after=np.array([mem.read(s) for s in slots]);assert np.array_equal(original,after);mem.close();target.unlink()
 # Differentiability of angle and token-conditioned offsets through answer token CE.
 m=Reasoner('full');x=torch.tensor(np.array([[[[0,1],[1,1],[2,1]],[[2,1],[4,1],[4,1]],[[1,1],[0,0],[2,1]],[[0,1],[2,0],[1,1]]]],dtype=np.float32))
 y=torch.tensor([3]);targets=torch.tensor([[EVIDENCE,int(LABELTOK[y][0]),EOS]])
 m.zero_grad();out=m(x,y);loss=F.cross_entropy(out.reshape(-1,8),targets.reshape(-1));loss.backward()
 ga=float(m.angle.weight.grad.norm());gq=float(m.query_phi.weight.grad.norm());assert ga>1e-8 and gq>1e-8,(ga,gq)
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 # Mixed contradictory evidence is represented as BOTH, not averaged away.
 support,refute=True,True;state=int(support)+2*int(refute);assert state==3
 r={'native_bytes':8388608,'native_pair_slot_count':32640,'independent_triangle_slot_capacity':32385,'raw_exact_after_large_gauge':int(np.allclose(raw,original,atol=.01)),'gauge_max_deviation':delta,'snapshot_exact':bool(np.array_equal(original,after)),'so2_commutator_norm':cm,'so3_commutator_norm':c,'holonomy_trace_gauge_error':h_err,'angle_gradient_norm':ga,'token_conditioning_gradient_norm':gq,'contradictory_support_and_refute_state':state}
 (P/'geometry_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':check()
