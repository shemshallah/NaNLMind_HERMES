#!/usr/bin/env python3
"""Neural writer-adaptation retention and held-out novel-paraphrase generation."""
import copy,json,random,torch
from pathlib import Path
import dynamic_joint as dj
from converse import Universe
B=Path(__file__).resolve().parent
u=Universe(B/'adapted_model_17.pt',B/'retention_temp.bin')
dj._active_native=u.root
paths=dj.load_paths(B/'corpus.jsonl')
rng=random.Random(9023)
eps=[dj.episode(paths['test'],rng,2,4,int(i%2==0)) for i in range(180)]

def changed(e):
 e=copy.deepcopy(e)
 rel=[]
 for a,b,p in e.facts:
  d=dj.PHRASES[p][1]
  if d==0:
   rel.append((a,b,18)) # new irrelevant phrase
  elif d==1:
   rel.append((a,b,16)) # "is subsumed by" ; forward
  else:
   rel.append((a,b,17)) # "comprises" ; inverted
 e.facts=rel
 return e

def run(eps):
 outs=[]
 with torch.no_grad():
  for i in range(0,len(eps),20):outs.extend(u.model.generate(eps[i:i+20],native=True))
 return {'correct':sum(dj.verify_answer(o,e) for o,e in zip(outs,eps)),'total':len(eps),'examples':[{'question':e.nodes[e.src]+' -> '+e.nodes[e.dst],'generated':s,'correct':dj.verify_answer(s,e)} for s,e in list(zip(outs,eps))[:8]]}
original=run(eps);novel=run([changed(e) for e in eps]);u.root.gauge(4031,2.5);novel_gauge=run([changed(e) for e in eps]);
result={'old_phrases':original,'new_phrases':novel,'new_phrases_after_gauge':novel_gauge,'native_integrity':u.root.audit()}
(B/'retention_results.json').write_text(json.dumps(result,indent=2));print({k:{'correct':v['correct'],'total':v['total']} for k,v in result.items() if isinstance(v,dict) and 'correct' in v})
u.close()
if __name__=='__main__':pass
