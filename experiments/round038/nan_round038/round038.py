"""Round 038: six learned relational fields + controls on WordNet-backed fact episodes.

Task: classify four-valued evidence from candidate 3-hop paths and explicit negative facts.
The native source is the actual R037 8 MiB canonical pair geometry. Both paths and
factual surface patterns are controlled; this is NOT natural-language pretraining.
"""
import argparse,ctypes,json,math,random,subprocess,time,hashlib,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import Writer,encode,wordnet,paths,TEMPLATES,normalize
BASE=Path(__file__).resolve().parent
CATS=['unknown','supports','refutes','both']
NREL=5
PAD=4
P=4;E=3
V=7 # PAD=0, BOS=1, evidence=2, unknown=3,supports=4,refutes=5,both=6 (EOS=7 actually)
V=8
BOS=1;EVIDENCE=2;EOS=7
LABELTOK=torch.tensor([3,4,5,6],dtype=torch.long)

class Root:
 def __init__(self):
  lib=BASE/'libtrig.so'
  subprocess.run(['g++','-std=c++20','-O3','-fPIC','-shared',str(BASE/'native_trig.cpp'),'-o',str(lib)],check=True,capture_output=True)
  self.l=ctypes.CDLL(str(lib));l=self.l;l.mem_new.restype=ctypes.c_void_p
  l.mem_free.argtypes=[ctypes.c_void_p]
  l.mem_write_lane.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float]
  l.mem_read_lane.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];l.mem_read_lane.restype=ctypes.c_float
  l.mem_raw_lane.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];l.mem_raw_lane.restype=ctypes.c_float
  l.mem_gauge_lane.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float,ctypes.c_int]
  l.mem_snapshot.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.mem_snapshot.restype=ctypes.c_int
  l.mem_restore.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.mem_restore.restype=ctypes.c_int
  self.p=l.mem_new();assert self.p;self.count=0
 def write(self,kind,link):
  s=self.count;assert s<32385;self.count+=1
  self.l.mem_write_lane(self.p,s,0,float(kind))
  self.l.mem_write_lane(self.p,s,1,float(link))
  return s
 def read(self,s,raw=False):
  method=self.l.mem_raw_lane if raw else self.l.mem_read_lane
  return [float(method(self.p,s,i)) for i in range(2)]
 def gauge(self,seed,sigma=25.):
  for i in range(2):self.l.mem_gauge_lane(self.p,seed+7919*i,float(sigma),i)
 def save(self,path):assert self.l.mem_snapshot(self.p,str(path).encode())==1
 def restore(self,path):assert self.l.mem_restore(self.p,str(path).encode())==1
 def close(self):self.l.mem_free(self.p)

def get_writer_preds(seed=29):
 ck=torch.load(BASE/f'writer_{seed}.pt',map_location='cpu',weights_only=False)
 w=Writer();w.load_state_dict(ck['state_dict']);w.eval()
 with torch.no_grad():pred=w(encode([normalize(t) for t,c in TEMPLATES])).argmax(-1).tolist()
 assert [pred[i] for i in [0,1,2,3,4,5,8,9,10,11]]==[c for i,(t,c) in enumerate(TEMPLATES) if i in [0,1,2,3,4,5,8,9,10,11]],pred
 # Class 0 and class 1 are same semantic directed evidence after swapping the sentence's entities.
 return pred

def make_episodes(chains,seed=29,writer_predictions=None):
 rng=random.Random(seed);examples=[];descriptors=[]
 for ch in chains:
  if len(ch)<4:continue
  support=rng.choice([0,1]);refute=rng.choice([0,1])
  candidates=[];truepos=rng.randrange(4);negpos=rng.choice([j for j in range(4) if j!=truepos])
  for k in range(4):
   # Relation labels: 0 valid known directed subtype, 2 explicit negation,
   # 3 uncertain, 4 unknown/absent. A single internal gap breaks a chain.
   r=[0,0,0];link=[1,1,1]
   # Each distractor is near-miss, often same multiset in a different order.
   if not (support and k==truepos):
    fault=rng.randrange(3)
    if fault==0:r[rng.randrange(3)]=rng.choice([2,3,4])
    elif fault==1:link[rng.choice([0,1])]=0
    else:
     r[rng.randrange(3)]=rng.choice([2,3]);link[rng.choice([0,1])]=0
   if refute and k==negpos:
    r=[2,PAD,PAD];link=[1,1,1]
   candidates.append((r,link))
  # Round037 trained writer is the actual predicate-to-code source, with
  # supplied entity spans; template-class outputs are cached because the
  # writer receives exactly the same normalized sentence regardless of names.
  if writer_predictions is not None:
   rebuilt=[]
   for rs,ls in candidates:
    mapped=[]
    for rel in rs:
     if rel==0:
      cls=writer_predictions[rng.choice([0,1,2,3,4,5,6,7])]
      mapped.append(0 if cls in (0,1) else cls) # resolve reversed sentence arguments
     elif rel==2:mapped.append(writer_predictions[rng.choice([8,9])])
     elif rel==3:mapped.append(writer_predictions[rng.choice([10,11])])
     else:mapped.append(PAD)
    rebuilt.append((mapped,ls))
   candidates=rebuilt
  # Avoid false support/refute from accidental distractor patterns.
  has_support=any(all(t==0 for t in rs) and ls[0]*ls[1]==1 for rs,ls in candidates)
  has_refute=any(rs==[2,PAD,PAD] for rs,ls in candidates)
  if has_support!=bool(support) or has_refute!=bool(refute):continue
  truth=(1 if support else 0)+(2 if refute else 0)
  examples.append((candidates,truth))
  descriptors.append({'source':ch[0],'target':ch[3],'truth':CATS[truth]})
 return examples,descriptors

def encode_native(examples,snapshot_name="root_038_wordnet.bin"):
 root=Root();refs=[]
 for paths,truth in examples:
  refs.append([ [root.write(rel,link) for rel,link in zip(rs,ls)] for rs,ls in paths])
 def extract(raw=False):
  arr=np.zeros((len(refs),P,E,2),dtype=np.float32)
  for i,pp in enumerate(refs):
   for j,rr in enumerate(pp):
    for k,s in enumerate(rr):arr[i,j,k]=root.read(s,raw=raw)
  return arr
 clean=extract();assert len(refs)*P*E==root.count
 snapshot=BASE/snapshot_name;root.save(snapshot)
 root.gauge(321,20.)
 gauged=extract();raw=extract(raw=True)
 err=float(np.max(np.abs(gauged-clean)))
 root.restore(snapshot)
 restored=extract();assert np.array_equal(clean,restored)
 root.close()
 return torch.from_numpy(clean),torch.from_numpy(gauged),{'edges':len(examples)*P*E,'gauge_max_abs':err,'raw_exact_feature_matches':int(np.sum(np.all(np.isclose(clean,raw,rtol=0,atol=.01),axis=(1,2,3)))),'snapshot_exact':bool(np.array_equal(clean,restored)),'size_bytes':snapshot.stat().st_size}

# Differentiable SO(3) rotations; products are order sensitive. All SO(2)/SO(3)
# transforms are norm-preserving (within floating point roundoff).
def givens(theta):
 # angle shape (...,3). Explicit matrix composition R_x Rx R_y Ry R_z Rz.
 a,b,c=theta.unbind(-1);za=torch.zeros_like(a);one=torch.ones_like(a)
 ca,sa=a.cos(),a.sin();cb,sb=b.cos(),b.sin();cc,sc=c.cos(),c.sin()
 def stack(*rows):return torch.stack([torch.stack(row,-1) for row in rows],-2)
 rx=stack((one,za,za),(za,ca,-sa),(za,sa,ca))
 ry=stack((cb,za,sb),(za,one,za),(-sb,za,cb))
 rz=stack((cc,-sc,za),(sc,cc,za),(za,za,one))
 return rz@ry@rx

class Reasoner(nn.Module):
 def __init__(self,mode,hidden=24):
  super().__init__();self.mode=mode;self.h=hidden
  self.kind=nn.Embedding(NREL,hidden)
  self.angle=nn.Embedding(NREL,3)
  nn.init.uniform_(self.angle.weight,-1.,1.)
  self.edge_encoder=nn.GRU(hidden+1,hidden,batch_first=True) if mode=='conventional' else None
  self.set_encoder=nn.Sequential(nn.Linear(E*(NREL+1),hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh()) if mode=='deepsets' else None
  if mode in ('frozen_phase','frozen_rotation'):self.angle.weight.requires_grad_(False)
  self.query_phi=nn.Linear(hidden,3,bias=False) if mode in ('dynamic','full') else None
  # Two-dimensional phase and three-dimensional noncommutative coordinates.
  dim=hidden if mode in ('conventional','deepsets') else (2 if mode in ('phase','frozen_phase') else 3 if mode in ('rotation','frozen_rotation','dynamic','holonomy','full') else hidden)
  extra=2 # boundary discontinuity + evidence multiplicity
  if mode in ('holonomy','full'):extra+=2 # trace and path reverse-cycle invariant
  self.path_key=nn.Sequential(nn.Linear(dim+extra,hidden),nn.Tanh(),nn.Linear(hidden,1))
  self.path_val=nn.Sequential(nn.Linear(dim+extra,hidden),nn.Tanh())
  self.context=nn.Linear(hidden,hidden)
  self.embed=nn.Embedding(V,hidden)
  self.rnn=nn.GRUCell(hidden*2,hidden)
  self.out=nn.Linear(hidden,V)
  self.h0=nn.Parameter(torch.zeros(hidden))
 def features(self,rel,valid,h):
  b=rel.size(0);m=rel.size(1)
  # Tensor rel[B,P,E], valid[B,P,E] retains native lane provenance.
  if self.mode=='conventional':
   v=torch.cat([self.kind(rel),valid.unsqueeze(-1)],-1).reshape(b*m,E,self.h+1)
   _,last=self.edge_encoder(v);p=last[-1].reshape(b,m,self.h)
  elif self.mode=='deepsets':
   f=torch.cat([F.one_hot(rel,NREL).float(),valid.unsqueeze(-1)],dim=-1)
   p=self.set_encoder(f.reshape(b,m,E*(NREL+1)))
  elif self.mode=='scalar':
   p=self.kind(rel).mean(dim=2)
  elif self.mode in ('phase','frozen_phase'):
   angles=self.angle(rel)[...,0].sum(dim=2)
   p=torch.stack([angles.cos(),angles.sin()],-1)
  else:
   theta=self.angle(rel)
   if self.query_phi is not None:
    delta=self.query_phi(h).reshape(b,1,1,3)
    theta=theta+delta # token-conditioned temporary transport; persistent memory unchanged
   mats=givens(theta)
   product=torch.eye(3,device=rel.device,dtype=theta.dtype).reshape(1,1,3,3).expand(b,m,3,3)
   for j in range(E):product=mats[:,:,j]@product
   v0=torch.tensor([1.,0.,0.],device=rel.device,dtype=theta.dtype)
   p=(product@v0).squeeze(-1) if False else (product@v0)
  gaps=1-valid[:,:,:2] # explicit adjacency side information shared by controls
  bad=gaps.sum(-1).unsqueeze(-1)
  multiplicity=(rel!=PAD).float().sum(-1,keepdim=True)/3.
  feats=[p,bad,multiplicity]
  if self.mode in ('holonomy','full'):
   # Gauge-invariant conjugacy observables for the cycle with a fixed closure.
   tr=product.diagonal(dim1=-2,dim2=-1).sum(-1,keepdim=True)/3.
   closure=givens(self.angle.weight[0].reshape(1,1,3)).transpose(-1,-2)
   ho=closure@product
   holo=ho.diagonal(dim1=-2,dim2=-1).sum(-1,keepdim=True)/3.
   feats.extend([tr,holo])
  return torch.cat(feats,dim=-1)
 def decode(self,x,teacher=True,labels=None):
  rel=torch.clamp(x[...,0].round().long(),0,NREL-1)
  valid=x[...,1].clamp(0.,1.)
  b=rel.size(0);h=self.h0.unsqueeze(0).expand(b,-1)
  prev=torch.full((b,),BOS,dtype=torch.long,device=rel.device)
  all_logits=[];attention=[]
  targets=torch.stack([torch.full((b,),EVIDENCE,device=rel.device),LABELTOK.to(rel.device)[labels] if labels is not None else torch.zeros(b,device=rel.device,dtype=torch.long),torch.full((b,),EOS,device=rel.device)],dim=1)
  for t in range(3):
   feat=self.features(rel,valid,h)
   logits_a=self.path_key(feat).squeeze(-1)
   w=F.softmax(logits_a,-1)
   val=self.path_val(feat)
   context=(w.unsqueeze(-1)*val).sum(1)
   if self.mode=='deepsets':context=context+val.max(1).values
   h=self.rnn(torch.cat([self.embed(prev),self.context(context)],-1),h)
   logits=self.out(h);all_logits.append(logits);attention.append(w)
   prev=logits.argmax(-1) if not teacher else targets[:,t]
  return torch.stack(all_logits,1),torch.stack(attention,1)
 def forward(self,x,labels):return self.decode(x,teacher=True,labels=labels)[0]

def train_one(mode,x,y,splits,seed=17,steps=220,lr=.003,task="wordnet"):
 torch.manual_seed(seed);model=Reasoner(mode)
 opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
 tr,va,te=splits
 targets=LABELTOK[y]
 # Decoding output tokens: [evidence, categorical label, EOS].
 def measure(idx,xdata):
  model.eval()
  with torch.no_grad():
   logits,_=model.decode(xdata[idx],teacher=False)
   class_logits=logits[:,1,3:7]
   pred=class_logits.argmax(-1)
   acc=(pred==y[idx]).float().mean().item()
   targets_=torch.stack([torch.full_like(y[idx],EVIDENCE),LABELTOK[y[idx]],torch.full_like(y[idx],EOS)],dim=-1)
   # teacher-forced CE measures next-token probability; pass separately.
   forced=model(xdata[idx],y[idx])
   ce=F.cross_entropy(forced.flatten(0,1),targets_.flatten()).item()
   return round(acc,4),round(ce,4),pred.cpu().tolist()
 best=-1;best_state=None;trace=[];batch_size=96
 for step in range(steps):
  model.train();sel=tr[torch.randint(len(tr),(batch_size,))]
  target=torch.stack([torch.full_like(y[sel],EVIDENCE),LABELTOK[y[sel]],torch.full_like(y[sel],EOS)],dim=-1)
  opt.zero_grad();out=model(x[sel],y[sel]);loss=F.cross_entropy(out.flatten(0,1),target.flatten());loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
  if step%40==0 or step==steps-1:
   a,ce,_=measure(va,x);trace.append([step,round(float(loss.item()),4),a,ce])
   if a>best or (a==best and (best_state is None or ce<best_ce)):
    best=a;best_ce=ce;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(best_state)
 p_test,ce_test,preds=measure(te,x)
 p_gauge,ce_gauge,gpreds=measure(te,GAUGE_X)
 model.eval()
 with torch.no_grad():
  _,att=model.decode(x[te],teacher=False)
 torch.save({'model':best_state,'mode':mode,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'architecture':'joint candidate reasoning and autoregressive 3-token decoder'},BASE/f'trig_{task}_{mode}_{seed}.pt')
 return {'mode':mode,'parameters':sum(p.numel() for p in model.parameters()),'validation_accuracy':round(best,4),'test_accuracy':p_test,'test_token_cross_entropy':ce_test,'gauge_accuracy':p_gauge,'gauge_token_cross_entropy':ce_gauge,'gauge_output_unchanged':round(float(np.mean(np.array(preds)==np.array(gpreds))),4),'attention_mean_entropy':round(float((-(att[:,-1]*(att[:,-1]+1e-9).log()).sum(-1)).mean().item()),4),'trace':trace}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--wordnet',default='/mnt/data/files/wn3.1.dict.zip');ap.add_argument('--steps',type=int,default=220);ap.add_argument('--seed',type=int,default=29);ap.add_argument('--limit',type=int,default=900);ap.add_argument('--modes',default='conventional,deepsets,scalar,phase,frozen_phase,rotation,frozen_rotation,holonomy,dynamic,full');a=ap.parse_args()
 torch.set_num_threads(2);start=time.time();preds=get_writer_preds(a.seed)
 rows=wordnet(Path(a.wordnet));chain=paths(rows,seed=a.seed+1000,limit=a.limit)
 examples,desc=make_episodes(chain,seed=a.seed+137,writer_predictions=preds);assert len(examples)>300,len(examples)
 split1=int(len(examples)*.60);split2=int(len(examples)*.80)
 ix=torch.randperm(len(examples),generator=torch.Generator().manual_seed(a.seed))
 splits=(ix[:split1],ix[split1:split2],ix[split2:]);y=torch.tensor([e[1] for e in examples])
 x,gauge,audit=encode_native(examples,snapshot_name=f"root_038_wordnet_{a.seed}.bin");GAUGE_X=gauge
 counts=Counter([CATS[z] for z in y.tolist()]);modes=a.modes.split(',')
 out={'source':'Round 037 native 8MiB C++ mirror-pair root + trained relation writer checkpoint; WordNet 3.1 concept-chain identities','limitation':'Controlled simulated candidate-path evidence from true WordNet chains; negations, ambiguity and distractors are synthetic; entity spans supplied, no free-form prose encoder or semantic open-world model','seed':a.seed,'wordnet_concepts':len(rows),'episodes':len(examples),'splits':list(map(len,splits)),'class_counts':counts,'writer_template_predictions':preds,'memory_audit':audit,'baseline_oracle_accuracy':1.0,'models':[],'elapsed_seconds':None}
 for m in modes:
  print('TRAIN',m,flush=True)
  try:
   r=train_one(m,x,y,splits,seed=a.seed,steps=a.steps,task="wordnet")
   print(m,'test',r['test_accuracy'],'gauge',r['gauge_accuracy'],'CE',r['test_token_cross_entropy'],flush=True)
   out['models'].append(r)
  except Exception as e:
   import traceback;traceback.print_exc();out['models'].append({'mode':m,'error':str(e)})
  (BASE/f'round038_seed{a.seed}_results.json').write_text(json.dumps(out,indent=2))
 out['elapsed_seconds']=round(time.time()-start,1)
 (BASE/f'round038_seed{a.seed}_results.json').write_text(json.dumps(out,indent=2))
 print('COMPLETE',out['elapsed_seconds'],flush=True)
