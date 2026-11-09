#!/usr/bin/env python3
"""Real free-running byte LM with geometric resident-memory adapter and strict controls.
Uses WordNet synset-disjoint train/validation/test glosses and held-out Alice paragraphs.
"""
import os,json,math,random,copy,hashlib,ctypes,argparse,time,re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BASE=Path(__file__).resolve().parent
class Native:
 def __init__(self):
  L=ctypes.CDLL(str(BASE/'libnative_cycle.so'));self.L=L
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
  a=np.empty((16,16),dtype='float32');self.L.nan_get(self.ptr,a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
  return torch.from_numpy(a.T.copy())
 def apply(self,g,lr):
  a=np.ascontiguousarray(g.detach().cpu().numpy(),dtype='float32');self.L.nan_apply(self.ptr,a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),ctypes.c_float(lr))
 def gauge(self,s,scale=1):self.L.nan_gauge(self.ptr,s,ctypes.c_float(scale))
 def audit(self):return self.L.nan_audit(self.ptr)==1
 def save(self,p):assert self.L.nan_snapshot(self.ptr,str(p).encode())
 def load(self,p):assert self.L.nan_restore(self.ptr,str(p).encode())
 def close(self):self.L.nan_free(self.ptr)
class ByteGenerator(nn.Module):
 def __init__(self,h=96,variant='plain'):
  super().__init__();self.variant=variant;self.h=h
  self.emb=nn.Embedding(256,h);self.gru=nn.GRU(h,h,batch_first=True)
  self.norm=nn.LayerNorm(h)
  self.query=nn.Linear(h,16);self.value=nn.Linear(16,h,bias=False)
  self.gate=nn.Linear(h,h)
  self.head=nn.Linear(h,256)
  self.pair=nn.Parameter(torch.zeros(16,16))
 def forward(self,x,W=None,h=None):
  e=self.emb(x);z,h=self.gru(e,h);z=self.norm(z)
  if W is None:W=self.pair
  retrieval=torch.tanh(self.value(torch.tanh(self.query(z)@W)))
  out=self.head(z+torch.sigmoid(self.gate(z))*retrieval)
  return out,h
 def params_no_pair(self):return [p for n,p in self.named_parameters() if n!='pair']

def dataset(path,alice,limit=8000):
 data={k:[] for k in ('train','val','test')}
 with open(path) as f:
  for ln in f:
   r=json.loads(ln);s=r.get('split')
   if s not in data or len(data[s])>=limit:continue
   gloss=(r.get('gloss') or '').strip()
   lemma=(r.get('lemmas') or [''])[0]
   if 18<=len(gloss)<=350:
    # At inference no structural answer rule; the network learns next-byte text on natural glosses.
    data[s].append((lemma+': '+gloss+'\n').encode('utf8'))
   if all(len(x)>=limit for x in data.values()):break
 txt=Path(alice).read_text(errors='replace');blocks=[p.strip() for p in re.split(r'\n\s*\n',txt) if len(p.strip())>50]
 # split by paragraph hashes rather than substring to prevent duplicate paragraphs in different sets
 alice_d={k:[] for k in data};seen=set()
 for b in blocks:
  key=hashlib.sha256(b.encode()).digest()
  if key in seen:continue
  seen.add(key);sp=['train','train','train','train','val','test'][int.from_bytes(key[:4],'big')%6]
  alice_d[sp].append((b+'\n').encode())
 return data,alice_d

def batch(data,rng,b=24,t=96):
 x=np.empty((b,t),dtype=np.int64);y=np.empty_like(x)
 for i in range(b):
  v=data[int(rng.integers(len(data)))];st=int(rng.integers(max(1,len(v)-1)))
  while len(v[st:])<2:st=0
  z=np.frombuffer(v[st:st+t+1],dtype=np.uint8).astype(np.int64)
  if len(z)<t+1:z=np.resize(z,t+1)
  x[i]=z[:-1];y[i]=z[1:]
 return torch.from_numpy(x),torch.from_numpy(y)
@torch.no_grad()
def evaluate(model,data,W,cap=150):
 model.eval();n=0;ll=0.;hits=0
 for v in data[:cap]:
  if len(v)<2:continue
  x=torch.tensor(list(v[:-1][:192]),dtype=torch.long)[None,:];y=torch.tensor(list(v[1:][:192]),dtype=torch.long)
  logits,_=model(x,W)
  ll+=F.cross_entropy(logits.flatten(0,1),y,reduction='sum').item();hits+=(logits.argmax(-1).flatten()==y).sum().item();n+=len(y)
 return {'bpb':ll/n/math.log(2),'accuracy':hits/n,'bytes':n}
@torch.no_grad()
def generate(model,W,prompt,n=240,temp=.72,topk=30,seed=0):
 torch.manual_seed(seed);model.eval();h=None
 p=prompt.encode('utf-8')
 for t in range(0,len(p)-1,192):_,h=model(torch.tensor(list(p[t:min(t+192,len(p)-1)]),dtype=torch.long)[None,:],W,h)
 tok=torch.tensor([[p[-1] if p else 32]],dtype=torch.long);out=[]
 for i in range(n):
  logits,h=model(tok,W,h);vec=logits[0,-1]/temp;v,idx=vec.topk(topk);prob=F.softmax(v,dim=0)
  tok=idx[torch.multinomial(prob,1)].view(1,1);out.append(int(tok.item()))
 return (p+bytes(out)).decode('utf-8',errors='replace')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variants',nargs='+',choices=['plain','native'],default=['plain','native']);ap.add_argument('--eval-cap',type=int,default=40);ap.add_argument('--test-cap',type=int,default=120);ap.add_argument('--steps',type=int,default=400);ap.add_argument('--seeds',type=int,nargs='+',default=[17,29]);ap.add_argument('--hidden',type=int,default=96);ap.add_argument('--batch',type=int,default=24);ap.add_argument('--context',type=int,default=96);ap.add_argument('--limit',type=int,default=7000);ap.add_argument('--eval-every',type=int,default=100);ap.add_argument('--corpus',default='/mnt/data/nan_round025/corpus.jsonl');ap.add_argument('--alice',default='/mnt/data/nan_cycle_round022/alice_excerpt.txt');a=ap.parse_args()
 torch.set_num_threads(2);data,alice=dataset(a.corpus,a.alice,a.limit)
 report={'config':vars(a),'corpus':{k:len(v) for k,v in data.items()},'alice':{k:len(v) for k,v in alice.items()},'models':{}}
 if (BASE/'results.json').exists():
  report['models'].update(json.loads((BASE/'results.json').read_text()).get('models',{}))
 for seed in a.seeds:
  for variant in a.variants:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ByteGenerator(a.hidden,variant)
   root=Native() if variant=='native' else None
   optimizer=torch.optim.AdamW(model.parameters() if root is None else model.params_no_pair(),lr=.002)
   best=1e8;beststate=None;bestroot=None;hist=[]
   t0=time.monotonic()
   for step in range(1,a.steps+1):
    model.train();x,y=batch(data['train'],rng,a.batch,a.context)
    optimizer.zero_grad();W=root.get().detach().clone().requires_grad_(True) if root else model.pair
    logits,_=model(x,W);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward()
    if root:root.apply(W.grad,.01)
    nn.utils.clip_grad_norm_(model.params_no_pair(),1);optimizer.step()
    if step==1 or step%a.eval_every==0 or step==a.steps:
     wt=root.get() if root else model.pair.detach();v=evaluate(model,data['val'],wt,a.eval_cap)
     hist.append({'step':step,**v});print('VAL',seed,variant,step,round(v['bpb'],4),flush=True)
     if v['bpb']<best:
      best=v['bpb'];beststate=copy.deepcopy(model.state_dict());beststep=step
      if root:
       bestroot=BASE/f'root_{seed}.bin';root.save(bestroot)
   model.load_state_dict(beststate)
   if root:root.load(bestroot)
   W=root.get() if root else model.pair.detach()
   test=evaluate(model,data['test'],W,a.test_cap)
   alice_test=evaluate(model,alice['test'],W,30) if alice['test'] else None
   gens={p:generate(model,W,p,n=260,seed=seed+98) for p in ['The little','A person who','The world was']}
   record={'history':hist,'best_step':beststep,'test':test,'alice_test':alice_test,'samples':gens,'seconds':time.monotonic()-t0,'params':sum(p.numel() for p in model.parameters())}
   if root:
    pre=W.detach().clone();root.gauge(seed+999,1.0);new=root.get();record['gauge_max_abs']=(new-pre).abs().max().item();record['native_audit']=root.audit();record['gauge_test']=evaluate(model,data['test'],new,a.test_cap)
    root.save(BASE/f'root_gauged_{seed}.bin');root.close()
   else:
    record['gauge_test']=None
   torch.save({'model':model.state_dict(),'hidden':a.hidden,'variant':variant,'seed':seed,'step':beststep},BASE/f'{variant}_{seed}.pt')
   report['models'][f'{variant}_{seed}']=record;print('TEST',seed,variant,json.dumps(test),'native drift',record.get('gauge_max_abs'),flush=True)
   (BASE/'results.json').write_text(json.dumps(report,indent=2))
 (BASE/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
