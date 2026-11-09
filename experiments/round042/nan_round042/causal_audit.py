#!/usr/bin/env python3
"""Causal interventions, hard negatives, node relabeling, gauge drift and token-loss gradients."""
import copy,json,random,hashlib
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import dynamic_joint as dj
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)

def load(seed):
 ck=torch.load(BASE/f'model_{seed}.pt',map_location='cpu',weights_only=False)
 m=dj.Joint(ck['dim']);m.load_state_dict(ck['model']);m.eval();return m

def describe(scores):return {k:round(float(v),8) if isinstance(v,(float,np.floating)) else v for k,v in scores.items()}

def gen(model,episodes,native=True):
 out=[]
 with torch.no_grad():
  for start in range(0,len(episodes),20):out+=model.generate(episodes[start:start+20],native=native)
 return out

def permute(e,seed):
 e=copy.deepcopy(e);order=list(range(len(e.nodes)));random.Random(seed).shuffle(order)
 rename={old:new for new,old in enumerate(order)}
 e.nodes=[e.nodes[j] for j in order]
 e.src=rename[e.src];e.dst=rename[e.dst]
 e.facts=[(rename[a],rename[b],p) for a,b,p in e.facts]
 return e

def run():
 root=dj.NativeRoot();dj._active_native=root
 paths=dj.load_paths(BASE/'corpus.jsonl');result={}
 for seed in (17,29,43):
  model=load(seed);rng=random.Random(seed+901)
  positive=[dj.episode(paths['test'],rng,2,4,truth=1) for _ in range(80)]
  negative=[dj.episode(paths['test'],rng,2,4,truth=0) for _ in range(80)]
  full=positive+negative
  base=gen(model,full)
  b_correct=sum(dj.verify_answer(o,e) for o,e in zip(base,full))
  permuted=[permute(e,i*49+seed) for i,e in enumerate(full)]
  renames=gen(model,permuted)
  unchanged_permute=sum(a==b for a,b in zip(base,renames))
  standard=gen(model,full,native=False)
  same_conventional=sum(a==b for a,b in zip(base,standard))
  root.gauge(seed+799,4.0)
  gauged=gen(model,full)
  same_gauge=sum(a==b for a,b in zip(base,gauged))
  # Actual physical intervention: the native function writes the source model's
  # signed edge, then the audit zeros precisely one indispensable middle pair
  # within the 8 MiB object *before* computing the triangle read.
  original_bridge=root.bridge;current=[]
  def delete_link(arr):
   out=original_bridge(arr)
   for i,e in enumerate(current):
    a=e.nodes.index(e.path[1]);b=e.nodes.index(e.path[2]);aa=i*8+min(a,b)+1;bb=i*8+max(a,b)+1
    root.L.ep_set(root.ptr,aa,bb,0.)
    out[i,a,b]=0;out[i,b,a]=0
   return out
  root.bridge=delete_link
  erased=[]
  with torch.no_grad():
   for i in range(0,len(positive),20):
    current=positive[i:i+20]
    erased+=model.generate(current,native=True)
  root.bridge=original_bridge
  removed_abstentions=sum(any(q in s for q in ['not enough','do not','cannot','insufficient','not support']) for s in erased)
  restored=gen(model,positive)
  same_restored=sum(a==b for a,b in zip(base[:len(positive)],restored))
  # Direct gradient from *next-token cross entropy only* to neural prose writer.
  model.train();ep=[dj.episode(paths['train'],random.Random(seed*31+i),2,4) for i in range(12)]
  T=torch.zeros(len(ep),max(len(e.target) for e in ep),dtype=torch.long)
  for i,e in enumerate(ep):T[i,:len(e.target)]=torch.tensor(e.target)
  output,_,_,_=model.forward(ep,T,native=True)
  ce=F.cross_entropy(output.flatten(0,1),T[:,1:].flatten(),ignore_index=0)
  grads=torch.autograd.grad(ce, [model.writer[-1].weight, model.reach_scale],allow_unused=False)
  raw_grads={'decoder_ce':float(ce.detach()),'writer_gradient_norm':float(grads[0].norm()),'propagator_gradient_magnitude':float(grads[1].abs())}
  row={'baseline_correct':b_correct,'total':len(full),'node_relabel_output_unchanged':unchanged_permute,'conventional_output_unchanged':same_conventional,'gauge_output_unchanged':same_gauge,'native_indispensable_edge_deletion_abstentions':removed_abstentions,'native_deletion_total':len(positive),'native_edge_restore_output_unchanged':same_restored,**raw_grads,'examples':[{'query':f'Is {e.nodes[e.src]} a kind of {e.nodes[e.dst]}?','before':base[i],'edge_deleted':erased[i],'restored':restored[i]} for i,e in enumerate(positive[:5])]}
  result[str(seed)]=row
  print('CAUSAL',seed,describe({k:v for k,v in row.items() if k!='examples'}),flush=True)
 root.L.ep_save(root.ptr,str(BASE/'causal_native_snapshot.bin').encode());print('ROOT_AUDIT',root.audit(),flush=True);root.close()
 (BASE/'causal_results.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':run()
