"""Second training track: expose the model to false same-slot relations.
Evaluates whether it learns semantic relation words rather than just matching slots.
"""
import json,copy,random,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from train import parse_wn,cases_build,vocab_make,encode,Seq2Seq,eval_model
def generated(m,stoi,itos,cases):
 x,_=encode(cases,stoi)
 with torch.no_grad():pred=m(x,max_len=13)
 outs=[]
 for line in pred.tolist():
  toks=[]
  for i in line:
   if i==3:break
   if i>3:toks.append(itos[i])
  outs.append(' '.join(toks))
 return outs

def semantic_ok(a,c):
 if c['negative']:return a=='the evidence is insufficient .'
 h=c['hops'];return a in [f'e0 is a type of e{h} .',f'e0 belongs to e{h} .',f'e0 is classified as e{h} .']
T=Path(__file__).parent;torch.set_num_threads(2)
rows=parse_wn(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/wn3.1.dict.zip'));data=cases_build(rows)
for name,cases in data.items():
 additions=[]
 for c in cases:
  if c['negative']:continue
  v=copy.deepcopy(c);v['negative']=True;v['kind']='relation_conflict';v['answer']='the evidence is insufficient .'
  # corrupt exactly one premise, preserving e0/e1/e2 connectivity
  before=v['premises'][0]
  for phrase in ('belongs to','is a type of','is classified as'):before=before.replace(phrase,'opposes')
  v['premises'][0]=before;v['prompt']=' premise : '+' next : '.join(v['premises'])+f' question : what is e0 relative to e{v["hops"]} ?'
  additions.append(v)
 cases.extend(additions)
 random.Random(31).shuffle(cases)
stoi,itos=vocab_make(data)
print('data',{k:len(v) for k,v in data.items()},'vocab',len(itos),flush=True)
seed=29;torch.manual_seed(seed);m=Seq2Seq(len(itos),32);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.0001)
best=1e9;best_sd=None;import copy as cp
rng=random.Random(29);t=time.monotonic()
for step in range(270):
 block=rng.choices(data['train'],k=40);x,y=encode(block,stoi)
 opt.zero_grad();lg=m(x,y[:,:-1]);loss=F.cross_entropy(lg.flatten(0,1),y[:,1:].flatten(),ignore_index=0);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%45==0 or step==269:
  val=rng.sample(data['validation'],128);x,y=encode(val,stoi)
  with torch.no_grad():v=float(F.cross_entropy(m(x,y[:,:-1]).flatten(0,1),y[:,1:].flatten(),ignore_index=0))
  print(step+1,'train',round(float(loss.detach()),4),'val',round(v,4),flush=True)
  if v<best:best=v;best_sd=cp.deepcopy(m.state_dict())
m.load_state_dict(best_sd);m.eval()
metrics=[]
for kind in ('connected','disconnected','relation_conflict'):
 cases=[c for c in data['test'] if ('connected' if not c['negative'] else c.get('kind','disconnected'))==kind]
 ans=[]
 for k in range(0,len(cases),256):ans+=generated(m,stoi,itos,cases[k:k+256])
 rate=sum(semantic_ok(a,c) for a,c in zip(ans,cases))/len(cases)
 metrics.append({'kind':kind,'n':len(cases),'semantic_accuracy':rate,'samples':[{'prompt':c['prompt'],'generated':a,'target':c['answer']} for c,a in zip(cases,ans)][:3]})
 print(kind,len(cases),rate,flush=True)
torch.save({'model':best_sd,'vocab':itos,'seed':seed},T/'robust_29.pt')
(T/'robust_results.json').write_text(json.dumps({'metrics':metrics,'seconds':time.monotonic()-t},indent=2))
