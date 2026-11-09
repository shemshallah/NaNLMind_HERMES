"""Round 038 order stress: explicit compositional grammar, NOT WordNet truth.
Every candidate path has the same set {a,b,c}, same edge count, identical links.
The labels depend only on the order of composition, with two target patterns:
  012 -> support; 210 -> refute; both if both; neither otherwise.
An SO(2) relation-phase sum is invariant under permutations by construction.
"""
import argparse,itertools,random,math,json,time
from pathlib import Path
import numpy as np
import torch
from collections import Counter
from round038 import encode_native,train_one,CATS
import round038
BASE=Path(__file__).parent
ALL=list(itertools.permutations((0,1,2)))
def build(n,seed):
 rng=random.Random(seed);out=[]
 for i in range(n):
  label=i%4;s=bool(label&1);r=bool(label&2)
  patterns=[]
  if s:patterns.append((0,1,2))
  if r:patterns.append((2,1,0))
  # Exclude both signature patterns for the other paths.
  distractors=[p for p in ALL if p!=(0,1,2) and p!=(2,1,0)]
  while len(patterns)<4:patterns.append(rng.choice(distractors))
  rng.shuffle(patterns)
  out.append(([(list(p),[1,1,1]) for p in patterns],label))
 rng.shuffle(out)
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=500);ap.add_argument('--count',type=int,default=900);ap.add_argument('--seed',type=int,default=41);ap.add_argument('--modes',default='conventional,deepsets,scalar,phase,frozen_phase,rotation,frozen_rotation,holonomy,dynamic,full');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 torch.set_num_threads(2)
 start=time.time();ex=build(a.count,a.seed);x,GAUGE,audit=encode_native(ex,snapshot_name=f"root_038_order_{a.seed}.bin");round038.GAUGE_X=GAUGE
 y=torch.tensor([z for _,z in ex],dtype=torch.long)
 idx=torch.randperm(len(ex),generator=torch.Generator().manual_seed(a.seed+221));splits=(idx[:int(.6*len(ex))],idx[int(.6*len(ex)):int(.8*len(ex))],idx[int(.8*len(ex)):])
 out={'task':'Ordered relation composition; synthetic permutation probe, NOT natural-language learning','all_paths_identical_multisets':True,'classes':dict(Counter([CATS[i] for i in y.tolist()])), 'episodes':len(ex),'splits':list(map(len,splits)),'seed':a.seed,'memory_audit':audit,'models':[]}
 if a.resume and (BASE/f'order_seed{a.seed}_results.json').exists():
  old=json.loads((BASE/f'order_seed{a.seed}_results.json').read_text())
  if old.get('seed')==a.seed and old.get('episodes')==len(ex):out['models']=old.get('models',[])
 for mode in a.modes.split(','):
  if any(z.get('mode')==mode and 'error' not in z for z in out['models']):continue
  print('TRAIN',mode,flush=True)
  try:
   z=train_one(mode,x,y,splits,seed=a.seed,steps=a.steps,task="order");out['models'].append(z)
   print(mode,z['test_accuracy'],z['test_token_cross_entropy'],flush=True)
  except Exception as e:
   import traceback;traceback.print_exc();out['models'].append({'mode':mode,'error':str(e)})
  (BASE/f'order_seed{a.seed}_results.json').write_text(json.dumps(out,indent=2))
 out['elapsed_seconds']=round(time.time()-start,2);(BASE/f'order_seed{a.seed}_results.json').write_text(json.dumps(out,indent=2))
 print('COMPLETE',out['elapsed_seconds'],flush=True)
