#!/usr/bin/env python3
"""Learned four-valued response generator driven by native independent support/refutation cycle lanes.
The refutation slot is explicitly written as evidence; linguistic realization is trained autoregressively.
No claim of unrestricted open-domain fluency or neural extraction of negation from arbitrary prose.
"""
import copy,json,random,torch,ctypes
from pathlib import Path
from torch import nn
from torch.nn import functional as F
import dynamic_joint as dj
P=Path(__file__).resolve().parent

torch.set_num_threads(1)
PATTERNS={
 'supports':[
  'the relationships support that <S> belongs to <D> .',
  'yes , the evidence indicates that <S> is a kind of <D> .',
  'the recorded facts establish that <S> is a type of <D> .'],
 'refutes':[
  'the explicit evidence refutes that <S> is a kind of <D> .',
  'no , the stored statement denies that <S> belongs to <D> .',
  'a negative claim explicitly rejects the relationship between <S> and <D> .'],
 'both':[
  'the evidence is conflicting : one path supports <S> as a kind of <D> , but another statement refutes it .',
  'there are both supporting and opposing claims about <S> belonging to <D> .',
  'the stored facts disagree : <S> is supported and refuted as a type of <D> .'],
 'unknown':[
  'i do not have sufficient evidence to classify <S> as <D> .',
  'the available information does not establish a relationship between <S> and <D> .',
  'there is not enough evidence to say that <S> belongs to <D> .']
}
W=['<pad>','<bos>','<eos>']+sorted({w for phrases in PATTERNS.values() for t in phrases for w in t.split()})
V={w:i for i,w in enumerate(W)}
KINDS=list(PATTERNS)

class Speaker(nn.Module):
 def __init__(self,hidden=80):
  super().__init__();self.mem=nn.Sequential(nn.Linear(3,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
  self.emb=nn.Embedding(len(W),hidden,padding_idx=0)
  self.gru=nn.GRU(hidden,hidden,batch_first=True)
  self.head=nn.Linear(hidden,len(W))
 def forward(self,features,teachers):
  h=self.mem(features);seq=self.emb(teachers[:,:-1])+h[:,None,:]
  z,_=self.gru(seq,h[None,:,:].contiguous())
  return self.head(z)
 @torch.no_grad()
 def generate(self,features,eps,temperature=0,maxwords=42):
  h=self.mem(features);hidden=h[None,:,:].contiguous();prev=torch.full((len(eps),1),V['<bos>'])
  results=[[] for _ in eps]
  for _ in range(maxwords):
   z,hidden=self.gru(self.emb(prev)+h[:,None,:],hidden);scores=self.head(z[:,0])
   tok=(scores.argmax(-1,keepdim=True) if temperature==0 else torch.multinomial(F.softmax(scores/temperature,-1),1))
   for i,v in enumerate(tok.flatten().tolist()):results[i].append(v)
   prev=tok
  lines=[]
  for i,e in enumerate(eps):
   line=[]
   for t in results[i]:
    w=W[t]
    if w=='<eos>':break
    if w=='<S>':w=e.nodes[e.src]
    if w=='<D>':w=e.nodes[e.dst]
    if w not in ('<pad>','<bos>'):line.append(w)
   lines.append(' '.join(line).replace(' ,',',').replace(' .','.').replace(' :',':'))
  return lines

class FourMemory:
 def __init__(self,base,root):
  self.base=base;self.root=root
  root.L.ep_set_layer.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_float]
  root.L.ep_get_layer.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
  root.L.ep_get_layer.restype=ctypes.c_float
 def features(self,eps,kinds):
  # Read the support path through the neural writer and the actual physical cycles.
  with torch.no_grad():support,_,_=self.base.graph(eps,native=True)
  negatives=[]
  for i,(e,kind) in enumerate(zip(eps,kinds)):
   a,b=e.src,e.dst;aa=i*8+min(a,b)+1;bb=i*8+max(a,b)+1
   sign=1 if a<b else -1
   self.root.L.ep_set_layer(self.root.ptr,1,aa,bb,sign if kind in ('both','refutes') else 0.)
   raw=float(self.root.L.ep_get_layer(self.root.ptr,1,aa,bb))
   negatives.append(sign*raw)
  return torch.stack([support[:,0],torch.tensor(negatives)*9-4.5,support[:,1]],1)

def build(paths,rng,n):
 eps=[];kind=[];target=[]
 for i in range(n):
  which=KINDS[i%4]
  e=dj.episode(paths,rng,2,4,truth=int(which in ('supports','both')))
  eps.append(e);kind.append(which)
  target.append([V['<bos>']]+[V[w] for w in rng.choice(PATTERNS[which]).split()]+[V['<eos>']])
 return eps,kind,target

def batch_targets(arr):
 o=torch.zeros(len(arr),max(map(len,arr)),dtype=torch.long)
 for i,r in enumerate(arr):o[i,:len(r)]=torch.tensor(r)
 return o

def semantic_label(s):
 s=s.lower()
 if any(z in s for z in ('conflicting','both supporting','disagree')):return 'both'
 if any(z in s for z in ('explicit evidence refutes','stored statement denies','claim explicitly rejects')):return 'refutes'
 if any(z in s for z in ('not have sufficient','not establish a relationship','not enough evidence')):return 'unknown'
 if any(z in s for z in ('relationships support','evidence indicates','recorded facts establish')):return 'supports'
 return 'invalid'

def run(seed=17,steps=600):
 torch.manual_seed(seed+58);rng=random.Random(seed)
 ck=torch.load(P/f'model_{seed}.pt',weights_only=False,map_location='cpu')
 base=dj.Joint(ck['dim']);base.load_state_dict(ck['model']);base.eval()
 root=dj.NativeRoot();dj._active_native=root;mem=FourMemory(base,root)
 speaker=Speaker();opt=torch.optim.AdamW(speaker.parameters(),lr=.003)
 paths=dj.load_paths(P/'corpus.jsonl');history=[];best=None;bestloss=1e9
 for step in range(1,steps+1):
  eps,kinds,targets=build(paths['train'],rng,24)
  f=mem.features(eps,kinds);y=batch_targets(targets)
  speaker.train();opt.zero_grad();log=speaker(f,y)
  loss=F.cross_entropy(log.flatten(0,1),y[:,1:].flatten(),ignore_index=0)
  loss.backward();nn.utils.clip_grad_norm_(speaker.parameters(),1);opt.step()
  if step==1 or step%120==0:
   valrng=random.Random(seed+1000);ve,vk,vt=build(paths['val'],valrng,24)
   with torch.no_grad():vl=F.cross_entropy(speaker(mem.features(ve,vk),batch_targets(vt)).flatten(0,1),batch_targets(vt)[:,1:].flatten(),ignore_index=0)
   history.append({'step':step,'train_ce':float(loss.detach()),'val_ce':float(vl)})
   print('FOUR_STATE',seed,step,round(float(loss.detach()),4),round(float(vl),4),flush=True)
   if vl<bestloss:bestloss=float(vl);best=copy.deepcopy(speaker.state_dict())
 speaker.load_state_dict(best);speaker.eval()
 test_rng=random.Random(seed+2000);eps,kinds,_=build(paths['test'],test_rng,160)
 outputs=[];gauge_outputs=[]
 with torch.no_grad():
  for i in range(0,len(eps),24):
   subset=eps[i:i+24];state=kinds[i:i+24];outputs+=speaker.generate(mem.features(subset,state),subset)
  root.gauge(seed+777,4.)
  for i in range(0,len(eps),24):
   subset=eps[i:i+24];state=kinds[i:i+24];gauge_outputs+=speaker.generate(mem.features(subset,state),subset)
 guesses=[semantic_label(x) for x in outputs]
 result={'seed':seed,'correct':sum(a==b for a,b in zip(kinds,guesses)),'total':len(kinds),'gauge_output_stability':sum(x==y for x,y in zip(outputs,gauge_outputs)),'gauge_total':len(outputs),'categories':{c:{'correct':sum(a==b==c for a,b in zip(kinds,guesses)),'total':kinds.count(c)} for c in KINDS},'history':history,'examples':[{'state':kind,'generated':out,'correct':kind==guess} for kind,out,guess in zip(kinds[:12],outputs[:12],guesses[:12])],'native_audit':root.audit(),'parameter_count':sum(x.numel() for x in speaker.parameters())}
 torch.save({'state':best,'words':W,'seed':seed},P/f'four_speaker_{seed}.pt')
 (P/f'four_speaker_{seed}.json').write_text(json.dumps(result,indent=2))
 root.close();print('FOUR_RESULT',seed,result['correct'],result['total'],result['gauge_output_stability'],flush=True)
 return result
if __name__=='__main__':run()
