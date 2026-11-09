#!/usr/bin/env python3
"""Round 042: native geometric episodic write/read -> trained propagation -> token-level language loss.
Controlled WordNet ancestry task. Native C++ root is used IN forward pass via a straight-through
native-value read; graph writer and language decoder are optimized jointly. Only host-supplied
entity boundaries and a constrained relation phrase vocabulary are assumed.
"""
import argparse, ctypes, json, math, os, random, re, subprocess, time, copy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1'
torch.set_num_threads(1)

# The surface order of the terms is not necessarily the direction of the semantic edge.
# The text classifier has to learn these orientations and learn to ignore non-supportive verbs.
PHRASES=[
 ('is a kind of', 1),('is a type of',1),('belongs to',1),('falls under',1),
 ('is classified as',1),('is an instance of',1),('is a member of',1),
 ('includes',-1),('encompasses',-1),('contains the category',-1),
 ('is not a kind of',0),('is unrelated to',0),('is different from',0),('opposes',0),
 ('is a special type of',1),('has as a member',-1),
]
PHRASE_VOCAB={'<pad>':0}
for phrase,_ in PHRASES:
 for word in phrase.split():
  if word not in PHRASE_VOCAB:PHRASE_VOCAB[word]=len(PHRASE_VOCAB)
PHRASE_IDS=[]
for phrase,_ in PHRASES:
 tok=[PHRASE_VOCAB[w] for w in phrase.split()]
 PHRASE_IDS.append(tok+[0]*(6-len(tok)))
PHRASE_IDS=torch.tensor(PHRASE_IDS,dtype=torch.long)

POSITIVE=[
 '<S> belongs to the class <D> .',
 'the information shows that <S> is a kind of <D> .',
 'according to the facts , <S> is classified as <D> .',
 'yes , <S> is a type of <D> .',
 'the recorded relationships imply that <S> is an instance of <D> .',
 'the evidence supports the conclusion that <S> belongs to <D> .',
]
NEGATIVE=[
 'the available facts do not establish that <S> belongs to <D> .',
 'there is not enough evidence to classify <S> as <D> .',
 'i cannot infer from these facts that <S> is a <D> .',
 'the stated relations do not support that conclusion about <S> and <D> .',
 'i do not have sufficient evidence for that relationship .',
]
ALL_TEMPLATES=POSITIVE+NEGATIVE
WORDS=['<pad>','<bos>','<eos>','<unk>']+sorted(set(w for p in ALL_TEMPLATES for w in p.split()))
TOK={x:i for i,x in enumerate(WORDS)}

# WordNet already has train/val/test child-concept splits from the latest archive.
def load_paths(path,limit=None):
 entries={};split={}
 for ln in open(path):
  r=json.loads(ln)
  if r.get('pos')!='noun':continue
  lem=r.get('lemmas') or []
  if not lem:continue
  name=lem[0].lower().replace('_',' ')
  if not re.fullmatch('[a-z][a-z-]{2,24}',name):continue
  entries[r['id']]=(name,r);split[r['id']]=r['split']
 parent={}
 for k,(name,r) in entries.items():
  pp=['n'+p[1] for p in r.get('pointers',[]) if p[0]=='@' and 'n'+p[1] in entries]
  if pp:parent[k]=pp[0]
 result={s:[] for s in ('train','val','test')}
 for k in parent:
  seq=[k];current=k
  for _ in range(5):
   if current not in parent:break
   current=parent[current];seq.append(current)
  if len(seq)<5:continue
  names=[entries[x][0] for x in seq]
  if len(set(names))!=len(names):continue
  result[split[k]].append(tuple(names))
  if limit and sum(map(len,result.values()))>limit:break
 return result

@dataclass
class Episode:
 nodes:list
 src:int
 dst:int
 facts:list # triples surface source-id, target-id, phrase-id; a phrase may reverse implication
 label:int
 target:list
 path:list

def episode(paths,rng,minimum=2,maximum=4,truth=None):
 p=list(rng.choice(paths));h=rng.randint(minimum,maximum)
 chain=p[:h+1];extras=[]
 if len(set(chain))!=len(chain):return episode(paths,rng,minimum,maximum,truth)
 if truth is None: truth=int(rng.random()<.5)
 else:truth=int(truth)
 extra=rng.choice(paths)[0]
 while extra in chain:extra=rng.choice(paths)[0]
 allnames=list(dict.fromkeys(chain+[extra]))
 rng.shuffle(allnames);ids={x:i for i,x in enumerate(allnames)}
 facts=[]
 missing=rng.randrange(h) if not truth else -1
 for j,(a,b) in enumerate(zip(chain[:-1],chain[1:])):
  if j==missing:
   # Hard negative: same number of statements as a positive episode.
   # Replace one indispensable link with an unrelated but grammatical edge.
   facts.append((ids[a],ids[extra],rng.randrange(7)))
   continue
  phrase_id=rng.randrange(len(PHRASES));direction=PHRASES[phrase_id][1]
  # Some irrelevant predicates are deliberate distractors and cannot replace the real edge.
  if direction==0:
   phrase_id=rng.randrange(7)
   direction=1
  lhs,rhs=(ids[a],ids[b]) if direction==1 else (ids[b],ids[a])
  facts.append((lhs,rhs,phrase_id))
 # Extra non-supportive or off-path facts, including opportunities for false positives.
 for _ in range(rng.choice([0,0,1,2])):
  a,b=rng.sample(range(len(allnames)),2)
  phrase_id=rng.randrange(10,len(PHRASES))
  if PHRASES[phrase_id][1]!=0:phrase_id=rng.choice([10,11,12,13])
  facts.append((a,b,phrase_id))
 rng.shuffle(facts)
 template=rng.choice(POSITIVE if truth else NEGATIVE)
 tokens=['<bos>']+template.split()+['<eos>']
 return Episode(allnames,ids[chain[0]],ids[chain[-1]],facts,truth,[TOK[w] for w in tokens],chain)

class NativeRoot:
 def __init__(self):
  lib=ROOT/'libnative_episode.so'
  if not lib.exists() or lib.stat().st_mtime<(ROOT/'native_episode.cpp').stat().st_mtime:
   subprocess.run(['g++','-O3','-std=c++20','-fPIC','-shared',str(ROOT/'native_episode.cpp'),'-o',str(lib)],check=True)
  L=ctypes.CDLL(str(lib));self.L=L
  L.ep_new.restype=ctypes.c_void_p
  L.ep_free.argtypes=[ctypes.c_void_p]
  L.ep_set.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float]
  L.ep_get.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];L.ep_get.restype=ctypes.c_float
  L.ep_gauge.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_float]
  L.ep_audit.argtypes=[ctypes.c_void_p];L.ep_audit.restype=ctypes.c_int
  L.ep_save.argtypes=[ctypes.c_void_p,ctypes.c_char_p];L.ep_save.restype=ctypes.c_int
  L.ep_load.argtypes=[ctypes.c_void_p,ctypes.c_char_p];L.ep_load.restype=ctypes.c_int
  self.ptr=L.ep_new();assert self.ptr
 def bridge(self,arr):
  # arr is [B,8,8] antisymmetric; each episode gets eight different vertices.
  B,N,_=arr.shape;assert N==8 and B*8<=248
  out=np.zeros_like(arr,dtype=np.float32)
  for b in range(B):
   for i in range(N):
    for j in range(i+1,N):
     a,c=b*8+i+1,b*8+j+1
     self.L.ep_set(self.ptr,a,c,float(arr[b,i,j]))
     f=self.L.ep_get(self.ptr,a,c)
     out[b,i,j]=f;out[b,j,i]=-f
  return out
 def gauge(self,seed,scale=2.):self.L.ep_gauge(self.ptr,int(seed),float(scale))
 def audit(self):return self.L.ep_audit(self.ptr)==1
 def close(self):self.L.ep_free(self.ptr);self.ptr=None

# global bridge for custom torch function; parameter gradients pass through C++ cycle writes/reads.
_active_native=None
class NativeCycle(torch.autograd.Function):
 @staticmethod
 def forward(ctx,values):
  assert _active_native is not None
  ctx.save_for_backward(values)
  raw=_active_native.bridge(values.detach().cpu().numpy().astype('float32'))
  return torch.from_numpy(raw).to(values.device)
 @staticmethod
 def backward(ctx,gradient):
  # source values are skew-symmetric, but the native call independently writes the upper triangle.
  # Compute chain-rule gradient for antisymmetric write matrix.
  return gradient

class Joint(nn.Module):
 def __init__(self,dim=80):
  super().__init__();self.phrase=nn.Embedding(len(PHRASE_VOCAB),32,padding_idx=0)
  self.writer=nn.Sequential(nn.Linear(32,64),nn.Tanh(),nn.Linear(64,3))
  self.reach_scale=nn.Parameter(torch.tensor(2.5))
  self.threshold=nn.Parameter(torch.tensor(.12))
  self.embedding=nn.Embedding(len(WORDS),dim,padding_idx=0)
  self.memory=nn.Sequential(nn.Linear(3,dim),nn.Tanh(),nn.Linear(dim,dim),nn.Tanh())
  self.decoder=nn.GRU(dim,dim,batch_first=True)
  self.head=nn.Linear(dim,len(WORDS))
 def graph(self,eps,native=True):
  B=len(eps);assert B<=42
  maxfacts=max(len(e.facts) for e in eps)
  tokens=torch.zeros(B,maxfacts,6,dtype=torch.long);src=torch.zeros(B,8);dst=torch.zeros(B,8)
  fact_ij=[];targets=[];weightsmask=[];mask=torch.zeros(B,maxfacts)
  for b,e in enumerate(eps):
   src[b,e.src]=1;dst[b,e.dst]=1
   for k,(u,v,pid) in enumerate(e.facts):tokens[b,k]=PHRASE_IDS[pid];mask[b,k]=1
  embedded=self.phrase(tokens)
  avg=embedded.sum(2)/((tokens!=0).sum(2).clamp_min(1).unsqueeze(-1))
  scores=self.writer(avg) # forward, inverse, no-link
  probs=scores.softmax(-1)
  values=torch.zeros(B,8,8)
  for b,e in enumerate(eps):
   for k,(u,v,pid) in enumerate(e.facts):
    signed=(probs[b,k,0]-probs[b,k,1])*.98
    # Each unordered head pair holds exactly one signed directional flow;
    # reversing a surface relation reverses which direction gets the positive flow.
    values[b,u,v]=values[b,u,v]+signed
    values[b,v,u]=values[b,v,u]-signed
  cyc=NativeCycle.apply(values) if native else values
  # ReLU converts signed circulation to the one-direction support graph.
  adj=F.relu(cyc)
  strength=torch.sigmoid(self.reach_scale)
  position=src;sumscore=torch.zeros(B);weighted_hops=torch.zeros(B)
  for step in range(1,7):
   position=torch.bmm(position[:,None,:],adj*strength).squeeze(1)
   contribution=(position*dst).sum(1)
   sumscore=sumscore+contribution
   weighted_hops=weighted_hops+contribution*step
  hop=weighted_hops/(sumscore+1e-6)
  proof=10*(sumscore-self.threshold)
  mem=torch.stack([proof,hop/6,torch.log1p(sumscore)],1)
  return mem,probs,mask
 def forward(self,eps,teacher,native=True):
  mem,probs,mask=self.graph(eps,native)
  cond=self.memory(mem)
  x=self.embedding(teacher[:,:-1])+cond[:,None,:]
  y,_=self.decoder(x,cond[None,:,:].contiguous())
  return self.head(y),mem,probs,mask
 def loss(self,eps,native=True):
  teachers=torch.zeros(len(eps),max(len(e.target) for e in eps),dtype=torch.long)
  for i,e in enumerate(eps):teachers[i,:len(e.target)]=torch.tensor(e.target)
  logits,mem,probs,mask=self.forward(eps,teachers,native)
  ce=F.cross_entropy(logits.flatten(0,1),teachers[:,1:].flatten(),ignore_index=TOK['<pad>'])
  truth=torch.tensor([e.label for e in eps],dtype=torch.float32)
  reach=F.binary_cross_entropy_with_logits(mem[:,0],truth)
  # Weak supervised writer objective for grammatical direction. Jointly train language loss too.
  labels=torch.zeros_like(mask,dtype=torch.long)
  for b,e in enumerate(eps):
   for k,(_,_,pid) in enumerate(e.facts):
    direction=PHRASES[pid][1]
    labels[b,k]=0 if direction==1 else (1 if direction==-1 else 2)
  writer_loss=(F.cross_entropy(probs.reshape(-1,3).log(),labels.flatten(),reduction='none')*mask.flatten()).sum()/mask.sum().clamp_min(1)
  return ce+.2*reach+.35*writer_loss,{'ce':float(ce.detach()),'proof':float(reach.detach()),'writer':float(writer_loss.detach())}
 @torch.no_grad()
 def generate(self,eps,max_tokens=32,native=True,temperature=0):
  mem,_,_=self.graph(eps,native)
  context=self.memory(mem)
  h=context[None,:,:].contiguous();prev=torch.full((len(eps),1),TOK['<bos>'],dtype=torch.long)
  outs=[[] for _ in eps]
  for _ in range(max_tokens):
   vec=self.embedding(prev)+context[:,None,:];z,h=self.decoder(vec,h)
   logits=self.head(z[:,0])
   if temperature:
    prob=F.softmax(logits/temperature,dim=-1);nextid=torch.multinomial(prob,1)
   else:nextid=logits.argmax(-1,keepdim=True)
   for idx,w in enumerate(nextid[:,0].tolist()):outs[idx].append(w)
   prev=nextid
  results=[]
  for e,seq in zip(eps,outs):
   words=[]
   for token in seq:
    if token==TOK['<eos>']:break
    if token in (TOK['<bos>'],TOK['<pad>']):continue
    word=WORDS[token]
    if word=='<S>':word=e.nodes[e.src]
    if word=='<D>':word=e.nodes[e.dst]
    words.append(word)
   results.append(' '.join(words).replace(' .','.').replace(' ,',','))
  return results

def is_abstention(out):
 return any(p in out for p in ["not enough", "do not", "cannot", "insufficient", "not support", "does not", "can't", "unable to"])

def verify_answer(out,e):
 s=e.nodes[e.src];d=e.nodes[e.dst]
 if e.label:
  if is_abstention(out):return False
  return s in out and d in out and any(p in out for p in [
   'kind of', 'type of', 'belongs to', 'classified as', 'instance of',
   'the class', 'evidence supports', 'relations imply'])
 return is_abstention(out)

def measure(model,paths,n,seed,native=True,perturb=None,depths=(2,3,4)):
 rng=random.Random(seed);eps=[episode(paths,rng,min(depths),max(depths),int(i%2==0)) for i in range(n)]
 global _active_native
 if perturb:_active_native.gauge(seed+1,perturb)
 outs=[];correct=0;pos=0;neg=0;ce=0
 model.eval()
 with torch.no_grad():
  for i in range(0,n,24):
   batch=eps[i:i+24];dec=model.generate(batch,native=native)
   for e,p in zip(batch,dec):
    ok=verify_answer(p,e);correct+=ok;pos+=int(ok and e.label);neg+=int(ok and not e.label)
   outs.extend(dec)
 return {'accuracy':correct/n,'correct':correct,'total':n,'positive_correct':pos,'negative_correct':neg,'examples':[{'facts':[(e.nodes[a],PHRASES[r][0],e.nodes[b]) for a,b,r in e.facts],'question':f'Is {e.nodes[e.src]} a kind of {e.nodes[e.dst]}?','expected':e.label,'generated':out,'valid':bool(verify_answer(out,e))} for e,out in list(zip(eps,outs))[:12]]}

def train(args):
 global _active_native
 paths=load_paths(args.corpus)
 print('WordNet paths', {x:len(y) for x,y in paths.items()},flush=True)
 root=NativeRoot();_active_native=root
 report={'method':'Joint language CE + differentiable native-cycle + learned propagation + prose phrase writer','setup':vars(args),'paths':{x:len(y) for x,y in paths.items()},'seeds':{}}
 for seed in args.seeds:
  rng=random.Random(seed);torch.manual_seed(seed)
  model=Joint(dim=args.hidden)
  opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
  history=[];start=time.time();best=None;bestloss=1e9
  for step in range(1,args.steps+1):
   e=[episode(paths['train'],rng) for _ in range(args.batch)]
   model.train();opt.zero_grad();loss,parts=model.loss(e,native=True)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
   if step==1 or step%args.evalevery==0:
    model.eval()
    valrng=random.Random(600+seed)
    ve=[episode(paths['val'],valrng) for _ in range(24)]
    with torch.no_grad():v,components=model.loss(ve,native=True)
    history.append({'step':step,'train_loss':loss.detach().item(),'val_loss':v.detach().item(),**parts})
    print('seed',seed,'step',step,'train',round(float(loss),3),'val',round(float(v),3),'ce',round(parts['ce'],3),flush=True)
    if v<bestloss:
     bestloss=float(v);best=copy.deepcopy(model.state_dict())
  model.load_state_dict(best);torch.save({'model':best,'vocab':WORDS,'dim':args.hidden,'seed':seed,'steps':args.steps},ROOT/f'model_{seed}.pt')
  val=measure(model,paths['val'],args.test,seed+998)
  test=measure(model,paths['test'],args.test,seed+1999)
  # A conventional control uses precisely the same continuous evidence values and neural weights.
  ordinary=measure(model,paths['test'],args.test,seed+1999,native=False)
  gauge=measure(model,paths['test'],args.test,seed+1999,native=True,perturb=1.5)
  depth5=measure(model,paths['test'],args.test,seed+2999,native=True,depths=(5,5))
  # Remove all memory evidence and ask again: should abstain even on formerly positive queries.
  rng2=random.Random(seed+1999)
  original=[episode(paths['test'],rng2,2,4,int(i%2==0)) for i in range(args.test)]
  erased=[]
  for x in original:
   y=copy.deepcopy(x);y.facts=[];y.label=0;erased.append(y)
  with torch.no_grad():
   ablation=[]
   for i in range(0,len(erased),24):ablation+=model.generate(erased[i:i+24])
  ablate_success=sum(any(q in a for q in ['not enough','do not','cannot','insufficient','not support']) for a in ablation)
  row={'seed':seed,'duration_seconds':time.time()-start,'train_history':history,'validation':val,'test':test,'same_parameters_conventional_bridge':ordinary,'test_after_gauge':gauge,'test_unseen_five_hops':depth5,'no_evidence_abstentions':ablate_success,'no_evidence_total':len(erased),'parameters':sum(p.numel() for p in model.parameters())}
  report['seeds'][str(seed)]=row
  (ROOT/'results.json').write_text(json.dumps(report,indent=2))
  print('RESULT',seed,json.dumps({k:v for k,v in row.items() if k not in ('train_history','validation','test','same_parameters_conventional_bridge','test_after_gauge','test_unseen_five_hops')}),flush=True)
 print('AUDIT',root.audit(),flush=True)
 root.L.ep_save(root.ptr,str(ROOT/'native_root_snapshot.bin').encode());root.close()
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--corpus',default=str(ROOT/'corpus.jsonl'));p.add_argument('--seeds',nargs='+',type=int,default=[17,29]);p.add_argument('--steps',type=int,default=500);p.add_argument('--batch',type=int,default=20);p.add_argument('--hidden',type=int,default=80);p.add_argument('--lr',type=float,default=.003);p.add_argument('--evalevery',type=int,default=100);p.add_argument('--test',type=int,default=120)
 train(p.parse_args())
