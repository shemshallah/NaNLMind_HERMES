import json,sys
import numpy as np,torch
from train import Native,LM,load_data,ev,BASE
sys.path.insert(0,'/mnt/data/nan_cycle_round022')
from byte_cycle_lm import load_dataset
alice=load_dataset('/mnt/data/nan_cycle_round022/alice_excerpt.txt','/mnt/data/nan_cycle_round022/split_manifest.json')
a=[np.frombuffer(z,dtype=np.uint8).astype(np.int64) for z in alice['test']]
res=json.loads((BASE/'transfer_results.json').read_text())
for seed in (17,29):
 for kind in ('plain','native'):
  model=LM();model.load_state_dict(torch.load(BASE/f'transfer_{kind}_{seed}.pt',map_location='cpu',weights_only=True))
  root=Native() if kind=='native' else None
  if root:root.restore(BASE/f'transfer_root_{seed}.bin')
  w=root.get() if root else model.pair.detach()
  res[f'{kind}_{seed}']['alice_test']=ev(model,a,w,len(a))
  if root:
   res[f'{kind}_{seed}']['alice_test_no_memory']=ev(model,a,torch.zeros_like(w),len(a))
   root.close()
(BASE/'transfer_results.json').write_text(json.dumps(res,indent=2))
for key in res: print(key,res[key]['alice_test'],res[key].get('alice_test_no_memory'))
