import os, json, time, ctypes, math, argparse, hashlib, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
N=16
class Native:
 def __init__(self):
  self.lib=ctypes.CDLL(str(BASE/'libnative_cycle.so'))
  L=self.lib
  L.nan_new.argtypes=[ctypes.c_int];L.nan_new.restype=ctypes.c_void_p
  L.nan_free.argtypes=[ctypes.c_void_p]
  L.nan_get.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float)]
  L.nan_apply.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_float]
  L.nan_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
  L.nan_audit.argtypes=[ctypes.c_void_p];L.nan_audit.restype=ctypes.c_int
  L.nan_snapshot.argtypes=[ctypes.c_void_p,ctypes.c_char_p];L.nan_snapshot.restype=ctypes.c_int
  L.nan_restore.argtypes=[ctypes.c_void_p,ctypes.c_char_p];L.nan_restore.restype=ctypes.c_int
  self.ptr=L.nan_new(0);assert self.ptr
 def get(self):
  a=np.zeros((N,N),dtype='float32');self.lib.nan_get(self.ptr,a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
  return torch.from_numpy(a.T.copy())
 def apply(self,grad,lr):
  a=np.ascontiguousarray(grad.detach().cpu().numpy(),dtype='float32')
  self.lib.nan_apply(self.ptr,a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),ctypes.c_float(lr))
 def gauge(self,seed,sigma=2):self.lib.nan_gauge(self.ptr,seed,sigma)
 def audit(self):return self.lib.nan_audit(self.ptr)==1
 def save(self,p):assert self.lib.nan_snapshot(self.ptr,str(p).encode())
 def restore(self,p):assert self.lib.nan_restore(self.ptr,str(p).encode())
 def close(self):self.lib.nan_free(self.ptr)

class LM(nn.Module):
 def __init__(self,hidden=48):
  super().__init__();self.emb=nn.Embedding(256,hidden);self.gru=nn.GRUCell(hidden,hidden)
  self.key=nn.Linear(hidden,N);self.val=nn.Linear(N,hidden,bias=False)
  self.head=nn.Linear(hidden,256);self.pair=nn.Parameter(torch.zeros(N,N))
 def forward(self,x,W):
  h=self.emb.weight.new_zeros((x.size(0),self.emb.embedding_dim));out=[]
  for t in range(x.size(1)):
   h=self.gru(self.emb(x[:,t]),h)
   rel=torch.tanh(self.val(torch.tanh(self.key(h)@W)))
   out.append(self.head(h+rel))
  return torch.stack(out,1)

def load_data(limit=1800):
 groups={k:[] for k in ['train','val','test']}
 with open('/mnt/data/nan_round025/corpus.jsonl') as f:
  for ln in f:
   r=json.loads(ln)
   if r.get('pos')!='noun':continue
   s=r.get('split')
   if s in groups and len(groups[s])<limit:
    text=(r.get('gloss') or '').strip()
    if len(text)>=18:groups[s].append(np.frombuffer(text.encode('utf-8'),dtype=np.uint8).astype(np.int64))
   if all(len(v)>=limit for v in groups.values()):break
 return groups

def batch(data,rng,b=8,t=48):
 x=np.zeros((b,t),dtype=np.int64);y=x.copy()
 for i in range(b):
  arr=data[rng.integers(len(data))];start=int(rng.integers(max(1,len(arr)-1)))
  z=np.resize(arr[start:] if len(arr[start:])>=2 else arr,t+1)
  x[i]=z[:-1];y[i]=z[1:]
 return torch.from_numpy(x),torch.from_numpy(y)
@torch.no_grad()
def ev(model,data,W,cap=240):
 model.eval();n=0;loss=0.;hits=0
 for arr in data[:cap]:
  if len(arr)<2:continue
  # constant-length each gloss; reset at doc boundaries
  x=torch.tensor(arr[:-1],dtype=torch.long)[None,:];y=torch.tensor(arr[1:],dtype=torch.long)
  # process in chunks for runtime, but each independent doc only, so focus <=96 tokens
  x=x[:,:96];y=y[:96]
  logits=model(x,W)
  loss+=F.cross_entropy(logits.reshape(-1,256),y,reduction='sum').item()
  hits+=int((logits.argmax(-1).flatten()==y).sum());n+=len(y)
 return {'n_bytes':n,'bits_per_byte':loss/n/math.log(2),'accuracy':hits/n}

def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=100);p.add_argument('--seeds',type=int,nargs='+',default=[17,29]);a=p.parse_args()
 torch.set_num_threads(1);data=load_data();summary={'counts':{k:len(v) for k,v in data.items()},'runs':{}}
 for seed in a.seeds:
  for kind in ('plain','native'):
   torch.manual_seed(seed);rng=np.random.default_rng(seed)
   model=LM();root=Native() if kind=='native' else None
   # optimizer on plain pair only, exclude it from native optimizer
   params=[p for name,p in model.named_parameters() if kind=='plain' or name!='pair']
   optim=torch.optim.AdamW(params,lr=.002)
   best=1e9;beststep=0;history=[];bestmodel=None;rootbest=None
   for step in range(1,a.steps+1):
    x,y=batch(data['train'],rng);model.train();optim.zero_grad()
    if root:
     W=root.get().detach().clone().requires_grad_(True)
    else:W=model.pair
    logits=model(x,W);loss=F.cross_entropy(logits.reshape(-1,256),y.flatten());loss.backward()
    if root: root.apply(W.grad,.003)
    nn.utils.clip_grad_norm_(params,1);optim.step()
    if step==1 or step%25==0 or step==a.steps:
     w=root.get() if root else model.pair.detach()
     v=ev(model,data['val'],w,100)
     history.append({'step':step,**v});print(seed,kind,step,round(v['bits_per_byte'],4),flush=True)
     if v['bits_per_byte']<best:
      best=v['bits_per_byte'];beststep=step
      bestmodel={k:z.detach().clone() for k,z in model.state_dict().items()}
      if root:rootbest=BASE/f'root_{seed}.bin';root.save(rootbest)
   model.load_state_dict(bestmodel)
   if root:root.restore(rootbest)
   w=root.get() if root else model.pair.detach()
   test=ev(model,data['test'],w,200)
   result={'best_step':beststep,'validation':history,'test':test,'params':sum(p.numel() for p in params)}
   if root:
    before=ev(model,data['test'],w,100);root.gauge(seed+500,2.0);after=ev(model,data['test'],root.get(),100)
    result['gauge_test']={'before':before,'after':after,'max_weight_diff':float((root.get()-w).abs().max()),'audit':root.audit()}
    root.restore(rootbest)
    # zero ablation
    abl=ev(model,data['test'],torch.zeros_like(w),100)
    result['memory_disabled']=abl
    root.save(BASE/f'root_{seed}.bin');root.close()
   torch.save({'state':model.state_dict(),'seed':seed,'kind':kind,'beststep':beststep},BASE/f'{kind}_{seed}.pt')
   summary['runs'][f'{kind}_{seed}']=result
 (BASE/'results.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps({k:v['test'] for k,v in summary['runs'].items()}))
if __name__=='__main__':main()
