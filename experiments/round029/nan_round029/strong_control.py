"""Retrained strong baseline: learn invariance from the three raw edge observations under gauge augmentation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,log_loss
from experiment import embeddings,encode,decode
BASE=Path(__file__).resolve().parent
torch.set_num_threads(2)

def train(x,y,v,yv,t,yt,seed,epochs=160,hidden=32):
 torch.manual_seed(seed)
 m=nn.Sequential(nn.Linear(x.shape[1],hidden),nn.GELU(),nn.Linear(hidden,int(y.max()+1)))
 opt=torch.optim.AdamW(m.parameters(),lr=.008,weight_decay=.02)
 x=torch.tensor(x,dtype=torch.float32);v=torch.tensor(v,dtype=torch.float32);t=torch.tensor(t,dtype=torch.float32)
 y=torch.tensor(y,dtype=torch.long);yv=torch.tensor(yv,dtype=torch.long)
 best=1e5;beststate=None;bestep=0
 for ep in range(epochs):
  opt.zero_grad();loss=nn.functional.cross_entropy(m(x),y);loss.backward();opt.step()
  with torch.no_grad():vl=float(nn.functional.cross_entropy(m(v),yv))
  if vl<best:best=vl;beststate={k:p.clone() for k,p in m.state_dict().items()};bestep=ep+1
 m.load_state_dict(beststate)
 with torch.no_grad():p=m(t).softmax(-1).numpy()
 return {'accuracy':float(accuracy_score(yt,p.argmax(1))),'ce':float(log_loss(yt,p,labels=np.arange(int(y.max()+1)))),'best_step':bestep,'params':sum(p.numel() for p in m.parameters())}

out={}
for seed in [17,29,41]:
 print('seed',seed,flush=True)
 cats,parts,_=embeddings(seed)
 (z,y),(zv,yv),(zt,yt)=parts
 r=np.random.default_rng(seed+123)
 tr=encode(z,3.,r);v=encode(zv,3.,r);te=encode(zt,3.,r)
 out[str(seed)]={'learned_raw_all_edges':train(tr,y,v,yv,te,yt,seed),
                 'cycle_same_gauge':train(decode(tr,'cycle'),y,decode(v,'cycle'),yv,decode(te,'cycle'),yt,seed,hidden=64)}
 print({k:(round(v['accuracy'],3),round(v['ce'],3)) for k,v in out[str(seed)].items()},flush=True)
 (BASE/'strong_control_results.json').write_text(json.dumps(out,indent=2))
