#!/usr/bin/env python3
"""Retrained matched-capacity conventional signed-adjacency control.
Same parameterization, batches, seed, optimizer and loss; skips native pair IO.
"""
import copy,json,random,torch
from pathlib import Path
import dynamic_joint as dj
B=Path(__file__).resolve().parent

def run(seeds=(17,29,43),steps=650):
 paths=dj.load_paths(B/'corpus.jsonl');results={}
 for seed in seeds:
  rng=random.Random(seed);torch.manual_seed(seed);m=dj.Joint(dim=80);opt=torch.optim.AdamW(m.parameters(),lr=.003)
  best=float('inf');state=None;chosen=0
  for step in range(1,steps+1):
   samples=[dj.episode(paths['train'],rng) for _ in range(20)]
   m.train();opt.zero_grad();loss,_=m.loss(samples,native=False);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
   if step==1 or step%130==0:
    m.eval();valrng=random.Random(600+seed)
    ex=[dj.episode(paths['val'],valrng) for _ in range(24)]
    with torch.no_grad():vl,_=m.loss(ex,native=False)
    if vl.item()<best:best=vl.item();state=copy.deepcopy(m.state_dict());chosen=step
  m.load_state_dict(state);m.eval()
  t=dj.measure(m,paths['test'],180,seed+1999,native=False)
  d=dj.measure(m,paths['test'],180,seed+2999,native=False,depths=(5,5))
  torch.save({'model':state,'dim':80,'seed':seed,'best_step':chosen},B/f'dense_{seed}.pt')
  ck=torch.load(B/f'model_{seed}.pt',map_location='cpu',weights_only=False)
  comparison=max(float((state[k]-ck['model'][k]).abs().max()) for k in state)
  row={'test_correct':t['correct'],'test_total':t['total'],'five_hop_correct':d['correct'],'five_hop_total':d['total'],'best_step':chosen,'native_model_max_parameter_difference':comparison}
  results[str(seed)]=row
  print('MATCHED_DENSE',seed,row,flush=True)
 (B/'matched_dense_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':run()
