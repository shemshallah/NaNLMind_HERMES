"""Round 036: learned autoregressive (word-token) seq2seq on two/three-hop WordNet chains.
Real held-out WordNet child synsets. Entities normalized to slot tokens; exact names
are copied back from source at presentation, never predicted by the network.
This is a controlled compositional language benchmark, not an open domain LLM.
"""
import json, random, sys, time, zipfile, hashlib, math, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).parent; torch.set_num_threads(2)
WN=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/wn3.1.dict.zip')

def parse_wn(path):
 with zipfile.ZipFile(path) as z:s=z.read('wn3.1.dict/dict/data.noun').decode('utf-8')
 r={}
 for ln in s.splitlines():
  if not ln or not ln[:8].isdigit():continue
  l,_,gloss=ln.partition('|');p=l.split()
  try:
   num=int(p[3],16);name=p[4].replace('_',' ').lower();i=4+2*num;npnt=int(p[i]);i+=1;par=[]
   for _ in range(npnt):
    symbol,ref,pos,_=p[i:i+4];i+=4
    if symbol in ('@','@i') and pos=='n':par.append(ref)
   if 2<=len(name)<=26 and name.isascii():r[p[0]]=(name,par,gloss.strip())
  except (ValueError,IndexError):continue
 return r

def cases_build(rows,count=6000):
 rng=random.Random(20261108);paths=[]; seen=set()
 for a,(nm,pars,_) in rows.items():
  if a in seen:continue
  for b in pars:
   if b not in rows:continue
   for c in rows[b][1]:
    if c not in rows:continue
    if len({a,b,c})<3:continue
    d=None
    for q in rows[c][1]:
     if q in rows and len({a,b,c,q})==4:d=q;break
    path=[a,b,c]+([d] if d is not None else [])
    if len({rows[k][0] for k in path})!=len(path):continue
    seen.add(a);paths.append(path);break
   if a in seen:break
 rng.shuffle(paths)
 templates=[('e0 is a type of e1 .','e1 is a type of e2 .'),('e0 belongs to e1 .','e1 belongs to e2 .'),('e0 is classified as e1 .','e1 is classified as e2 .')]
 cases=[]
 for path in paths[:count]:
  for negative in [False,True]:
   for hops in [2,3] if len(path)>3 else [2]:
    # source-disjoint by child ID, tests never see corresponding synsets during training
    t=rng.randrange(3);verb=['is a type of','belongs to','is classified as'][t]
    facts=[f'e{i} {verb} e{i+1} .' for i in range(hops)]
    if negative:facts[-1]=f'e{hops+1} {verb} e{hops} .'
    rng.shuffle(facts)
    prompt=' premise : ' + ' next : '.join(facts) + f' question : what is e0 relative to e{hops} ?'
    if negative:target='the evidence is insufficient .'
    else:
     target=[f'e0 is a type of e{hops} .',f'e0 belongs to e{hops} .',f'e0 is classified as e{hops} .'][rng.randrange(3)]
    cases.append({'child':path[0],'path':path,'hops':hops,'negative':negative,'premises':facts,'prompt':prompt,'answer':target,'entities':[rows[k][0] for k in path]})
 # group by child, fixed and reproducible
 child_ids=list(dict.fromkeys(c['child'] for c in cases));rng.shuffle(child_ids);n=len(child_ids)
 bins={k:[] for k in ('train','validation','test')};m={cid:('train' if i<int(.75*n) else 'validation' if i<int(.875*n) else 'test') for i,cid in enumerate(child_ids)}
 for c in cases:bins[m[c['child']]].append(c)
 return bins

class Seq2Seq(nn.Module):
 def __init__(self,vocab,dim=64):
  super().__init__();self.emb=nn.Embedding(vocab,dim,padding_idx=0);self.enc=nn.GRU(dim,dim,batch_first=True,bidirectional=True);self.init=nn.Linear(dim*2,dim);self.query=nn.Linear(dim,2*dim);self.dec=nn.GRU(dim+dim*2,dim,batch_first=True);self.out=nn.Sequential(nn.Linear(dim+2*dim,dim*2),nn.Tanh(),nn.Linear(dim*2,vocab))
 def forward(self,x,y=None,max_len=14):
  mask=x.ne(0);z,_=self.enc(self.emb(x));pooled=(z*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1);h=torch.tanh(self.init(pooled)).unsqueeze(0)
  if y is not None:
   context=pooled.unsqueeze(1).expand(-1,y.shape[1],-1)
   o,_=self.dec(torch.cat([self.emb(y),context],dim=-1),h)
   score=torch.bmm(self.query(o),z.transpose(1,2));score=score.masked_fill(~mask.unsqueeze(1),-1e5)
   ctx=torch.bmm(F.softmax(score,dim=-1),z)
   return self.out(torch.cat([o,ctx],dim=-1))
  t=x.new_full((x.size(0),),2);generated=[]
  for k in range(max_len):
   o,h=self.dec(torch.cat([self.emb(t),pooled],dim=-1).unsqueeze(1),h)
   score=torch.bmm(z,self.query(o.squeeze(1)).unsqueeze(-1)).squeeze(-1).masked_fill(~mask,-1e5)
   ctx=torch.bmm(F.softmax(score,dim=-1).unsqueeze(1),z).squeeze(1)
   t=self.out(torch.cat([o.squeeze(1),ctx],dim=-1)).argmax(-1);generated.append(t)
  return torch.stack(generated,dim=1)

def vocab_make(data):
 words=set()
 for v in data.values():
  for c in v:words.update(c['prompt'].split());words.update(c['answer'].split())
 vocab=['<pad>','<unk>','<bos>','<eos>']+sorted(words);return {w:i for i,w in enumerate(vocab)},vocab

def encode(data,stoi,maxx=58,maxy=14):
 x=[];y=[]
 for c in data:
  xx=[stoi.get(t,1) for t in c['prompt'].split()][:maxx];yy=[2]+[stoi.get(t,1) for t in c['answer'].split()]+[3]
  x.append(xx+[0]*(maxx-len(xx)));y.append(yy[:maxy]+[0]*max(0,maxy-len(yy)))
 return torch.tensor(x),torch.tensor(y)

def eval_model(model,enc,stoi,itos,batch=128):
 model.eval();correct=0;exact=0;tot=0;loss=0;tokens=0;outputs=[]
 with torch.no_grad():
  for start in range(0,len(enc),batch):
   block=enc[start:start+batch];x,y=encode(block,stoi);lg=model(x,y[:,:-1]);l=F.cross_entropy(lg.flatten(0,1),y[:,1:].flatten(),ignore_index=0,reduction='sum');tokens+=(y[:,1:]!=0).sum().item();loss+=l.item()
   pred=model(x,max_len=13)
   for p,truth,c in zip(pred,y,block):
    got=[]
    for t in p.tolist():
     if t==3:break
     if t>3:got.append(itos[t])
    answer=' '.join(got);correct+=answer==c['answer'];tot+=1
    if len(outputs)<5:outputs.append({'input':c['prompt'],'target':c['answer'],'generated':answer,'entities':c['entities']})
 return {'exact':correct/tot,'correct':correct,'n':tot,'nll_nats_per_token':loss/max(1,tokens),'examples':outputs}

def main():
 st=time.monotonic();torch.manual_seed(23);rows=parse_wn(WN);data=cases_build(rows);stoi,itos=vocab_make(data)
 print('WordNet',len(rows),'cases',{k:len(v) for k,v in data.items()},'vocab',len(itos),flush=True)
 results={'wordnet_synsets':len(rows),'splits':{k:len(v) for k,v in data.items()},'vocab':itos,'seeds':[]}
 for seed in [17,29]:
  torch.manual_seed(seed);model=Seq2Seq(len(itos),dim=32);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
  best=1e9;best_sd=None;hist=[];rng=np.random.default_rng(seed)
  train=data['train'];val=data['validation']
  # Full train with shuffled minibatches, validation model selection
  for step in range(145):
   idx=rng.integers(0,len(train),size=32);block=[train[i] for i in idx];x,y=encode(block,stoi)
   opt.zero_grad();lg=model(x,y[:,:-1]);loss=F.cross_entropy(lg.flatten(0,1),y[:,1:].flatten(),ignore_index=0);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
   if step%48==0 or step==144:
    probe=[val[i] for i in rng.choice(len(val),min(128,len(val)),replace=False)];vx,vy=encode(probe,stoi)
    with torch.no_grad():vl=float(F.cross_entropy(model(vx,vy[:,:-1]).flatten(0,1),vy[:,1:].flatten(),ignore_index=0))
    hist.append({'step':step+1,'train_loss':float(loss),'validation_loss':vl});print(seed,step+1,round(float(loss),4),round(vl,4),flush=True)
    if vl<best:best=vl;best_sd=copy.deepcopy(model.state_dict())
  model.load_state_dict(best_sd)
  ev=eval_model(model,data['test'],stoi,itos)
  ev['seed']=seed;ev['validation_trace']=hist;ev['parameters']=sum(p.numel() for p in model.parameters());results['seeds'].append(ev)
  torch.save({'model':best_sd,'vocab':itos,'seed':seed,'params':ev['parameters']},ROOT/f'seq2seq_{seed}.pt')
  print('TEST',seed,ev['exact'],ev['nll_nats_per_token'],flush=True)
 results['elapsed_s']=time.monotonic()-st
 (ROOT/'results.json').write_text(json.dumps(results,indent=2));(ROOT/'test_examples.json').write_text(json.dumps(data['test'][:100],indent=2))

if __name__=='__main__':main()
