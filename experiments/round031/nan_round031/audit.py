import json,random,time,ctypes, numpy as np
import torch
from pathlib import Path
from train import Native
BASE=Path(__file__).resolve().parent
# direct native memory benchmarks plus geometry invariance, full root save / restore
root=Native();rng=np.random.default_rng(7)
a=rng.normal(size=(16,16)).astype('float32');root.apply(torch.from_numpy(a),.2)
start=time.perf_counter();count=10000
for _ in range(count):z=root.get()
read_ms=1000*(time.perf_counter()-start)/count
x=root.get();root.gauge(77,2.);y=root.get();err=float((x-y).abs().max());audit=root.audit()
p=BASE/'audit_root.bin';root.save(p);root.apply(torch.from_numpy(np.ones((16,16),dtype='float32')),.002);root.restore(p)
restored=float((root.get()-y).abs().max());root.close()
# full WordNet typed relation audit: literal synset references and two-hop ancestry
records={};parents={}
with open('/mnt/data/nan_round025/corpus.jsonl') as f:
 for line in f:
  o=json.loads(line)
  key=o['id']
  records[key]=o
# WordNet id is n000... parse pointer offsets
for sid,o in records.items():
 ups=[]
 for ptr in o.get('pointers',[]):
  if len(ptr)>=3 and ptr[0] in ('@','@i'):
   target=ptr[2]+str(ptr[1]).zfill(8)
   if target in records:ups.append(target)
 parents[sid]=ups
positive=[]
for child,ps in parents.items():
 if child[0]!='n':continue
 for mid in ps:
  for grand in parents.get(mid,[]):
   if grand!=child and grand not in ps:
    positive.append((child,mid,grand))
random.Random(13).shuffle(positive)
pos=positive[:1500]
# these exact chains are truths by path traversal, direct-only not able to prove; no neural generalization claim.
results={'native':{'reads':count,'mean_read_ms':read_ms,'gauge_max_diff':err,'mirror_audit':audit,'snapshot_max_diff':restored},'wordnet':{'synsets':len(records),'two_hop_candidates':len(positive),'two_hop_proofs_checked':len(pos),'all_chain_edges_verified':all(b in parents[a] and c in parents[b] for a,b,c in pos),'direct_edge_only_hits':sum(c in parents[a] for a,b,c in pos)}}
(BASE/'audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
def dummy():pass
