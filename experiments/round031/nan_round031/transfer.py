import json,math,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train import Native,LM,load_data,batch,ev,BASE
from importlib.machinery import SourceFileLoader
import sys
sys.path.insert(0,'/mnt/data/nan_cycle_round022')
from byte_cycle_lm import load_dataset

def main():
 torch.set_num_threads(1)
 alice=load_dataset('/mnt/data/nan_cycle_round022/alice_excerpt.txt','/mnt/data/nan_cycle_round022/split_manifest.json')
 corp={k:[np.frombuffer(v,dtype=np.uint8).astype(np.int64) for v in val if len(v)>2] for k,val in alice.items()}
 word=load_data(1200);results={}
 for seed in (17,29):
  for kind in ('plain','native'):
   saved=torch.load(BASE/f'{kind}_{seed}.pt',map_location='cpu',weights_only=False)
   model=LM();model.load_state_dict(saved['state'])
   root=Native() if kind=='native' else None
   if root:root.restore(BASE/f'root_{seed}.bin')
   W=lambda:root.get() if root else model.pair.detach()
   before={'wordnet':ev(model,word['val'],W(),100),'alice':ev(model,corp['val'],W(),10)}
   rng=np.random.default_rng(seed+800)
   params=[p for name,p in model.named_parameters() if kind=='plain' or name!='pair']
   opt=torch.optim.AdamW(params,lr=.001)
   for step in range(30):
    model.train();x,y=batch(corp['train'],rng,8,48);opt.zero_grad()
    w=root.get().detach().clone().requires_grad_(True) if root else model.pair
    loss=F.cross_entropy(model(x,w).reshape(-1,256),y.flatten());loss.backward()
    if root:root.apply(w.grad,.002)
    torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
   after={'wordnet':ev(model,word['val'],W(),100),'alice':ev(model,corp['val'],W(),10)}
   results[f'{kind}_{seed}']={'before':before,'after':after}
   if root:root.save(BASE/f'transfer_root_{seed}.bin');root.close()
   torch.save(model.state_dict(),BASE/f'transfer_{kind}_{seed}.pt')
   print(kind,seed,'alice',round(before['alice']['bits_per_byte'],3),'->',round(after['alice']['bits_per_byte'],3),'wordnet',round(before['wordnet']['bits_per_byte'],3),'->',round(after['wordnet']['bits_per_byte'],3),flush=True)
 (BASE/'transfer_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
