"""Differentiable multi-hop propagation over native/conventional evidence, no supplied proof status.
Scalar transition strength and readout threshold are optimized by BCE. Relation evidence stays dynamic.
"""
import json,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from role_train import load_chains,sample
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)

def encode(rows,nodes=8):
 As=torch.zeros(len(rows),nodes,nodes);S=torch.zeros(len(rows),nodes);D=torch.zeros(len(rows),nodes)
 Y=torch.tensor([int(r[2]) for r in rows],dtype=torch.float32)
 for i,row in enumerate(rows):
  _,_,truth,src,dst,chain,*_=row
  names=list(dict.fromkeys(list(chain[:4])+[dst]))
  # randomized node order; no absolute identity is trained
  random.Random(i*17+233).shuffle(names)
  if len(names)>nodes:raise ValueError('too many nodes')
  index={word:j for j,word in enumerate(names)}
  for a,b in zip(chain[:-1],chain[1:]):As[i,index[a],index[b]]=1.
  S[i,index[src]]=1.;D[i,index[dst]]=1.
 return As,S,D,Y

class DiffCycleReasoner(nn.Module):
 def __init__(self):
  super().__init__();self.strength=nn.Parameter(torch.tensor(0.0));self.scale=nn.Parameter(torch.tensor(1.0));self.threshold=nn.Parameter(torch.tensor(0.4))
 def forward(self,A,src,dst,steps=3):
  p=src
  weight=torch.sigmoid(self.strength)
  for _ in range(steps):
   p=p+weight*torch.bmm(p[:,None,:],A).squeeze(1)
  score=(p*dst).sum(-1)
  return (F.softplus(self.scale)+1.)*(score-self.threshold)

def run(seed,steps=180):
 torch.manual_seed(seed);paths=load_chains()
 tr=encode(sample(paths['train'],2000,seed));test=encode(sample(paths['test'],400,seed+456))
 model=DiffCycleReasoner();opt=torch.optim.Adam(model.parameters(),lr=.03);hist=[]
 for step in range(steps):
  ix=torch.randint(2000,(128,));opt.zero_grad();logit=model(*(x[ix] for x in tr[:3]));loss=F.binary_cross_entropy_with_logits(logit,tr[3][ix]);loss.backward();opt.step()
  if step in [0,39,79,119,179]:hist.append([step+1,round(loss.item(),5)])
 with torch.no_grad():
  logits=model(*test[:3]);pred=logits>0;correct=(pred==test[3].bool());acc=correct.float().mean().item()
 result={'seed':seed,'accuracy':acc,'correct':int(correct.sum()),'total':len(correct),'learned_strength':float(torch.sigmoid(model.strength)),'learned_threshold':float(model.threshold),'history':hist}
 torch.save(model.state_dict(),BASE/f'reasoner_{seed}.pt')
 (BASE/f'reasoner_{seed}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
 return result
if __name__=='__main__':
 data=[run(s) for s in [17,29,43]]
 (BASE/'reasoner_summary.json').write_text(json.dumps(data,indent=2))
