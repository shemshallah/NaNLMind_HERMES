#!/usr/bin/env python3
"""Subword autoregressive free generator conditioned on C(256,2) native cycle readout.
BPE learned from train split only; no hardcoded answer templates or retrieval during generation.
"""
import argparse,copy,ctypes,json,math,random,time
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import sentencepiece as spm
from train_generate import Native
BASE=Path(__file__).resolve().parent
class LM(nn.Module):
 def __init__(self,vocab,hidden=128):
  super().__init__();self.emb=nn.Embedding(vocab,hidden);self.gru=nn.GRU(hidden,hidden,batch_first=True);self.norm=nn.LayerNorm(hidden)
  self.key=nn.Linear(hidden,16);self.value=nn.Linear(16,hidden,bias=False);self.gate=nn.Linear(hidden,hidden)
  self.head=nn.Linear(hidden,vocab);self.pair=nn.Parameter(torch.zeros(16,16))
 def forward(self,x,W=None,h=None):
  z,h=self.gru(self.emb(x),h);z=self.norm(z)
  if W is None:W=self.pair
  r=torch.tanh(self.value(torch.tanh(self.key(z)@W)))
  return self.head(z+torch.sigmoid(self.gate(z))*r),h
 def nonpair(self):return [v for k,v in self.named_parameters() if k!='pair']
def load_docs(path,maxtrain=25000,maxval=1500,maxtest=1500):
 groups={k:[] for k in ('train','val','test')};caps={'train':maxtrain,'val':maxval,'test':maxtest}
 for l in open(path):
  r=json.loads(l);sp=r['split'];s=(r.get('gloss') or '').strip()
  if len(s)>=25 and len(groups[sp])<caps[sp]:groups[sp].append(s)
  if all(len(groups[k])==caps[k] for k in caps):break
 return groups

def prep(d,vocab=2048):
 inp=BASE/'spm_corpus.txt'
 inp.write_text('\n'.join(d['train']))
 spm.SentencePieceTrainer.train(input=str(inp),model_prefix=str(BASE/'wordnet_bpe'),vocab_size=vocab,model_type='bpe',character_coverage=1.0,hard_vocab_limit=False,pad_id=0,unk_id=1,bos_id=2,eos_id=3)
 tok=spm.SentencePieceProcessor(model_file=str(BASE/'wordnet_bpe.model'))
 return tok,{k:[np.array([tok.bos_id()]+tok.encode(s,out_type=int)+[tok.eos_id()],dtype=np.int64) for s in v] for k,v in d.items()}
def batch(data,rng,b=24,t=48):
 x=np.empty((b,t),np.int64);y=np.empty_like(x)
 for i in range(b):
  arr=data[int(rng.integers(len(data)))];st=int(rng.integers(max(1,len(arr)-1)));z=arr[st:st+t+1]
  if len(z)<t+1:z=np.resize(z,t+1)
  x[i]=z[:-1];y[i]=z[1:]
 return torch.from_numpy(x),torch.from_numpy(y)
@torch.no_grad()
def ev(model,data,W,cap=120):
 model.eval();n=0;ll=0;hit=0
 for arr in data[:cap]:
  x=torch.from_numpy(arr[:-1]).long()[None,:];y=torch.from_numpy(arr[1:]).long()
  log,_=model(x,W);ll+=F.cross_entropy(log.reshape(-1,log.size(-1)),y,reduction='sum').item();n+=len(y);hit+=(log.argmax(-1).reshape(-1)==y).sum().item()
 return {'n_tokens':n,'loss_nats':ll/n,'perplexity':math.exp(ll/n),'token_accuracy':hit/n}
@torch.no_grad()
def gen(model,tok,W,prompt,steps=130,temp=.8,topk=35,seed=8):
 torch.manual_seed(seed);model.eval();h=None;context=[tok.bos_id()]+tok.encode(prompt,out_type=int)
 if len(context)>1:_,h=model(torch.tensor(context[:-1])[None,:],W)
 v=torch.tensor([[context[-1]]]);out=[]
 for t in range(steps):
  logits,h=model(v,W,h);scores=logits[0,-1]/temp;z,ix=scores.topk(topk);pr=F.softmax(z,0);v=ix[torch.multinomial(pr,1)].view(1,1);id=int(v.item())
  if id==tok.eos_id():break
  if id not in (tok.pad_id(),tok.bos_id()):out.append(id)
 return prompt+tok.decode(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',default=str(BASE/'corpus.jsonl') if (BASE/'corpus.jsonl').exists() else '/mnt/data/nan_round025/corpus.jsonl');p.add_argument('--steps',type=int,default=500);p.add_argument('--seed',type=int,default=17);p.add_argument('--variants',nargs='+',default=['plain','native']);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--hidden',type=int,default=128);p.add_argument('--vocab',type=int,default=2048);p.add_argument('--batch',type=int,default=24);p.add_argument('--context',type=int,default=48);a=p.parse_args();torch.set_num_threads(2)
 d=load_docs(a.corpus);tok,data=prep(d,a.vocab)
 summary={'params':vars(a),'documents':{k:len(v) for k,v in d.items()},'models':{}}
 for variant in a.variants:
  torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);model=LM(tok.vocab_size(),a.hidden)
  root=Native() if variant=='native' else None
  opt=torch.optim.AdamW(model.nonpair() if root else model.parameters(),lr=.002)
  best=1e6;beststate=None;hist=[];start=time.monotonic()
  for step in range(1,a.steps+1):
   model.train();x,y=batch(data['train'],rng,a.batch,a.context);opt.zero_grad();W=root.get().clone().detach().requires_grad_(True) if root else model.pair
   logits,_=model(x,W);loss=F.cross_entropy(logits.reshape(-1,tok.vocab_size()),y.flatten());loss.backward()
   if root:root.apply(W.grad,.02)
   nn.utils.clip_grad_norm_(model.nonpair(),1);opt.step()
   if step==1 or step%a.eval_every==0 or step==a.steps:
    q=ev(model,data['val'],root.get() if root else model.pair,20);hist.append({'step':step,**q});print(variant,step,round(q['loss_nats'],4),flush=True)
    if q['loss_nats']<best:
     best=q['loss_nats'];beststep=step;beststate=copy.deepcopy(model.state_dict())
     if root:root.save(BASE/'subword_root.bin')
  model.load_state_dict(beststate)
  if root:root.load(BASE/'subword_root.bin')
  W=root.get() if root else model.pair.detach()
  test=ev(model,data['test'],W,120)
  samples={p:gen(model,tok,W,p,120,seed=a.seed+2) for p in ['A person who','a place where','the act of']}
  info={'history':hist,'best_step':beststep,'test':test,'samples':samples,'seconds':time.monotonic()-start,'parameter_count':sum(z.numel() for z in model.parameters())}
  if root:
   before=W.clone();root.gauge(3123,1.5);after=root.get();info['gauge_readout_max_abs']=(before-after).abs().max().item();info['gauge_test']=ev(model,data['test'],after,60);info['native_audit']=root.audit();root.close()
  torch.save({'model':model.state_dict(),'hidden':a.hidden,'vocab':tok.vocab_size(),'variant':variant,'seed':a.seed},BASE/f'subword_{variant}.pt')
  summary['models'][variant]=info;(BASE/'subword_results.json').write_text(json.dumps(summary,indent=2));print('TEST',variant,json.dumps(test),flush=True)
if __name__=='__main__':main()
