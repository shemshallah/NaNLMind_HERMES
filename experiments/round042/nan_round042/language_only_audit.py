#!/usr/bin/env python3
"""Strong ablation: no supervised writer classes and no proof-status labels.
Train native relational writer + differentiable propagator + decoder from token loss alone.
Assesses whether fluently reciting answers can bootstrap genuine graph-reasoning gradients.
"""
import json,random,time,torch
from torch.nn import functional as F
from pathlib import Path
import dynamic_joint as dj
B=Path(__file__).resolve().parent

def fit(seed=17,steps=1000):
 paths=dj.load_paths(B/'corpus.jsonl')
 root=dj.NativeRoot();dj._active_native=root
 rng=random.Random(seed);torch.manual_seed(seed)
 model=dj.Joint();opt=torch.optim.AdamW(model.parameters(),lr=.003)
 hist=[]
 for step in range(1,steps+1):
  e=[dj.episode(paths['train'],rng,2,2 if step<250 else 4) for _ in range(24)]
  target=torch.zeros(24,max(len(x.target) for x in e),dtype=torch.long)
  for i,x in enumerate(e):target[i,:len(x.target)]=torch.tensor(x.target)
  model.train();opt.zero_grad()
  logits,_,_,_=model(e,target,native=True)
  loss=F.cross_entropy(logits.flatten(0,1),target[:,1:].flatten(),ignore_index=0)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  if step in (1,250,500,1000,steps):
   row=dj.measure(model,paths['val'],80,seed+3001)
   hist.append({'step':step,'train_ce':float(loss.detach()),'validation_accuracy':row['accuracy']})
   print('LANGUAGE_ONLY',seed,step,round(float(loss.detach()),4),row['accuracy'],flush=True)
 report=dj.measure(model,paths['test'],180,seed+4001)
 result={'seed':seed,'steps':steps,'test':{k:v for k,v in report.items() if k!='examples'},'history':hist,'examples':report['examples'][:6],'root_audit':root.audit()}
 (B/f'language_only_{seed}.json').write_text(json.dumps(result,indent=2))
 torch.save({'state':model.state_dict(),'seed':seed,'steps':steps},B/f'language_only_model_{seed}.pt')
 root.close();return result
if __name__=='__main__':fit()
