"""Word-level pointer generator for novel WordNet lemmas: learned autoregressive choices and copying.
Two-/three-hop proofs from WordNet-derived prose; evidence assembled from the same fact store for native and ordinary control.
"""
import json,random,re,collections,sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from train import load_chains,TEMPLATES,ANSWERS,NOANSWER
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)
FUNCTION=['<pad>','<bos>','<eos>','<unk>','facts',':',';','question','what','kind','of','thing','is','?','.','it','a','belongs','to','the','category','classified','as','are','under','includes','encompasses','falls','there','not','enough','evidence','available','do','establish','that','in','source','target','status','supported','unsupported','proof']
V={w:i for i,w in enumerate(FUNCTION)};SIZE=len(V)

def sample(paths,n,seed):
 rng=random.Random(seed);out=[]
 for _ in range(n):
  chain=rng.choice(paths); hops=rng.choice([2,3]);src=chain[0];truth=rng.random()<0.7
  dst=chain[hops] if truth else rng.choice(paths)[-1]
  if not truth and dst in chain:dst='unconnected'
  facts=[]
  for j in range(hops):
   a,b=chain[j:j+2];templates=rng.choice(TEMPLATES)
   facts.append(rng.choice(templates).format(a=a,b=b))
  rng.shuffle(facts)
  inp='source : '+src+' ; target : '+dst+' ; status : '+('supported' if truth else 'unsupported')+' ; facts : '+' ; '.join(facts)
  target=(rng.choice(ANSWERS).format(a=src,b=dst) if truth else rng.choice(NOANSWER))
  out.append((inp,target,truth,src,dst,chain,facts))
 return out

def tokenize(s):return re.findall(r"[a-z]+(?:-[a-z]+)?|[.:;?]",s.lower())

def batch_data(data,maxx=75,maxy=18):
 X=torch.zeros(len(data),maxx,dtype=torch.long);Y=torch.zeros(len(data),maxy,dtype=torch.long)
 XX=torch.zeros_like(X);YY=torch.zeros_like(Y)
 words=[]
 for i,(inp,target,*_) in enumerate(data):
  src=tokenize(inp);trg=['<bos>']+tokenize(target)+['<eos>']
  # Source-local entity vocabulary, not globally learned; tokenizer supports new lemmas.
  local=[]
  for w in src:
   if w not in V and w not in local:local.append(w)
  ex={w:SIZE+j for j,w in enumerate(local)}
  sids=[V.get(w,3) for w in src][:maxx]
  sext=[V[w] if w in V else ex[w] for w in src][:maxx]
  tids=[V.get(w,3) for w in trg][:maxy]
  text=[V[w] if w in V else ex.get(w,3) for w in trg][:maxy]
  X[i,:len(sids)]=torch.tensor(sids);XX[i,:len(sext)]=torch.tensor(sext)
  Y[i,:len(tids)]=torch.tensor(tids);YY[i,:len(text)]=torch.tensor(text)
  words.append(local)
 return X,XX,Y,YY,words

class CopyNet(nn.Module):
 def __init__(self,d=64):
  super().__init__();self.emb=nn.Embedding(SIZE,d,padding_idx=0)
  self.pos=nn.Embedding(75,d)
  self.enc=nn.GRU(d,d,batch_first=True,bidirectional=True)
  self.init=nn.Linear(d*2,d)
  self.cell=nn.GRUCell(d*3,d)
  self.att=nn.Linear(d*3,1)
  self.head=nn.Linear(d*3,SIZE)
  self.gate=nn.Linear(d*3,3)
 def encode(self,x):
  positions=torch.arange(x.shape[1])[None,:]
  z,_=self.enc(self.emb(x)+self.pos(positions));mask=x!=0
  summary=(z*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
  return z,mask,torch.tanh(self.init(summary))
 def step(self,prev,h,z,mask,source_ext):
  q=h[:,None,:].expand(-1,z.shape[1],-1)
  a=self.att(torch.cat([q,z],-1)).squeeze(-1).masked_fill(~mask,-1e4)
  a=torch.softmax(a,1)
  ctx=torch.bmm(a[:,None,:],z).squeeze(1)
  h=self.cell(torch.cat([self.emb(prev),ctx],1),h)
  v=torch.cat([h,ctx],1)
  gen=torch.softmax(self.head(v),-1)
  max_ext=max(SIZE+24,int(source_ext.max())+1)
  full=F.pad(gen,(0,max_ext-SIZE))
  # Learned three-way decision: generate ordinary token, copy queried subject, copy queried candidate.
  # Query fields are explicitly marked; surface forms remain unknown during training.
  roles=torch.softmax(self.gate(v),dim=1)
  src=source_ext[:,2:3];dst=source_ext[:,6:7]
  out=roles[:,0:1]*full
  out=out.scatter_add(1,src,roles[:,1:2])
  out=out.scatter_add(1,dst,roles[:,2:3])
  return out.clamp_min(1e-9),h
 def loss(self,x,xx,y,yy):
  z,mask,h=self.encode(x);loss=0;count=0
  for t in range(1,y.shape[1]):
   probs,h=self.step(y[:,t-1],h,z,mask,xx)
   good=yy[:,t]!=0
   loss-= (torch.log(probs.gather(1,yy[:,t:t+1]).squeeze(1))*good).sum()
   count+=good.sum()
  return loss/count.clamp_min(1)
 @torch.no_grad()
 def decode(self,data,maxlen=16):
  x,xx,y,yy,local=data;z,mask,h=self.encode(x)
  prev=torch.ones(len(x),dtype=torch.long);out=[]
  for _ in range(maxlen):
   probs,h=self.step(prev,h,z,mask,xx)
   nxt=probs.argmax(1);out.append(nxt)
   prev=nxt.clamp(max=SIZE-1);prev=torch.where(nxt>=SIZE,torch.full_like(nxt,3),prev)
  result=[]
  for i,row in enumerate(torch.stack(out,1).tolist()):
   if 2 in row:row=row[:row.index(2)]
   s=' '.join(FUNCTION[t] if t<SIZE else (local[i][t-SIZE] if t-SIZE<len(local[i]) else '?') for t in row if t not in (0,1))
   result.append(s)
  return result

def run(seed,steps=280):
 torch.manual_seed(seed);paths=load_chains()
 train=sample(paths['train'],5000,seed);val=sample(paths['val'],240,seed+123);test=sample(paths['test'],300,seed+456)
 tr=batch_data(train);va=batch_data(val);te=batch_data(test)
 model=CopyNet();opt=torch.optim.AdamW(model.parameters(),lr=0.002)
 best=1e9;hist=[]
 for step in range(steps):
  ix=torch.randint(len(train),(24,))
  opt.zero_grad();loss=model.loss(*(a[ix] for a in tr[:4]));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step==0 or (step+1)%40==0:
   with torch.no_grad():vl=float(model.loss(*(a[:80] for a in va[:4])))
   hist.append([step+1,round(loss.item(),4),round(vl,4)])
   print('seed',seed,'step',step+1,'train',round(loss.item(),4),'val',round(vl,4),flush=True)
   if vl<best:
    best=vl;torch.save({'state':model.state_dict(),'seed':seed,'step':step+1},BASE/f'role_{seed}.pt')
 checkpoint=torch.load(BASE/f'role_{seed}.pt',weights_only=False,map_location='cpu');model.load_state_dict(checkpoint['state'])
 outs=[]
 for j in range(0,len(test),30):
  data=tuple(a[j:j+30] for a in te[:4])+(te[4][j:j+30],)
  outs.extend(model.decode(data))
 correct=0;pos=0;neg=0;ex=[]
 for row,out in zip(test,outs):
  inp,tgt,truth,src,dst,*_=row
  ok=(out in [p.format(a=src,b=dst) for p in ANSWERS]) if truth else out in NOANSWER
  correct+=ok
  if truth:pos+=ok
  else:neg+=ok
  if len(ex)<15:ex.append({'facts':row[-1],'query':inp.split('question : ')[-1],'expected':tgt,'generated':out,'correct':bool(ok)})
 result={'seed':seed,'steps':steps,'best_step':checkpoint['step'],'test':len(test),'accuracy':correct/len(test),'positive':pos,'negative':neg,'positive_total':sum(r[2] for r in test),'negative_total':sum(not r[2] for r in test),'train_params':sum(p.numel() for p in model.parameters()),'history':hist,'examples':ex}
 (BASE/f'role_result_{seed}.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('history','examples')}),flush=True)
 return result
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=280);parser.add_argument('--seeds',type=int,nargs='+',default=[17,29]);a=parser.parse_args()
 for s in a.seeds:run(s,a.steps)
