"""Round 040: native gauge invariant episodic knowledge, matched ordinary index, learned prose relation reader.
This is NOT an open-domain neural language model. Output is rule-rendered from composed evidence.
"""
import ctypes, json, hashlib, random, re, time, os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
BASE=Path(__file__).resolve().parent

# The entity names are arbitrary nonce concepts not seen during classifier training.
TRAIN_PATTERNS=[
 ('{a} is a type of {b}.',1,0),('{a} belongs to the class {b}.',1,0),
 ('{a} is classified under {b}.',1,0),('{a} is one kind of {b}.',1,0),
 ('{b} encompasses {a}.',1,1),('{b} includes the category {a}.',1,1),
 ('{a} is not a kind of {b}.',-1,0),('{a} does not belong to {b}.',-1,0),
 ('{a} opposes {b}.',0,0),('{a} resembles {b}.',0,0)]
TEST_PATTERNS=[
 ('{a} is a type of {b}.',1,0),('{a} belongs to the class {b}.',1,0),
 ('{b} encompasses {a}.',1,1),('{a} is not a kind of {b}.',-1,0),
 ('{a} opposes {b}.',0,0)]
# reversed-source flag intentionally remaps roles to same relation direction.

def train_reader(seed):
 rng=random.Random(seed); xs=[];ys=[]
 for k in range(1200):
  a=f'concept{k:05d}';b=f'notion{k:05d}';tmpl,lab,rev=rng.choice(TRAIN_PATTERNS)
  sentence=tmpl.format(a=a,b=b)
  # Mask entity tokens before classification, preserving relation vocabulary and order.
  sentence=sentence.replace(a,'ENTITY_A').replace(b,'ENTITY_B')
  xs.append(sentence);ys.append((lab,rev))
 # Classify direction + polarity jointly.
 code={(1,0):0,(1,1):1,(-1,0):2,(0,0):3}
 y=[code[q] for q in ys]
 clf=make_pipeline(TfidfVectorizer(ngram_range=(1,3),lowercase=True),LogisticRegression(max_iter=250,random_state=seed))
 clf.fit(xs,y)
 return clf

class Native:
 def __init__(self):
  self.lib=ctypes.CDLL(str(BASE/'libepisode.so'))
  self.lib.ep_new.restype=ctypes.c_void_p
  self.lib.ep_free.argtypes=[ctypes.c_void_p]
  self.lib.ep_set.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float]
  self.lib.ep_get.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];self.lib.ep_get.restype=ctypes.c_float
  self.lib.ep_raw.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];self.lib.ep_raw.restype=ctypes.c_float
  self.lib.ep_gauge.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_float]
  self.lib.ep_audit.argtypes=[ctypes.c_void_p];self.lib.ep_audit.restype=ctypes.c_int
  self.lib.ep_save.argtypes=[ctypes.c_void_p,ctypes.c_char_p];self.lib.ep_save.restype=ctypes.c_int
  self.lib.ep_load.argtypes=[ctypes.c_void_p,ctypes.c_char_p];self.lib.ep_load.restype=ctypes.c_int
  self.ptr=self.lib.ep_new();assert self.ptr
  self.slot=0;self.records=[]
 def close(self):self.lib.ep_free(self.ptr)
 def append(self,a,b,polarity,source):
  self.slot+=1;idx=self.slot-1
  # All 32,385 rooted independent triangles available using 1<=a<b<=255.
  aa=1; t=idx
  while t>=255-aa:
   t-=255-aa;aa+=1
  bb=aa+1+t
  assert 1<=aa<bb<=255
  self.lib.ep_set(self.ptr,aa,bb,float(polarity))
  self.records.append((aa,bb,a,b,source))
 def edges(self):
  return [(a,b,round(float(self.lib.ep_get(self.ptr,i,j)))) for i,j,a,b,_ in self.records if abs(self.lib.ep_get(self.ptr,i,j))>.5]
 def raw_edges(self):
  return [(a,b,round(float(self.lib.ep_raw(self.ptr,i,j)))) for i,j,a,b,_ in self.records if abs(self.lib.ep_raw(self.ptr,i,j))>.5]
 def gauge(self,seed,scale):self.lib.ep_gauge(self.ptr,seed,scale)

def parse_sentence(s,clf):
 # Entity discovery is supplied by an explicit documented protocol, not learned NLP.
 m=re.fullmatch(r'([A-Za-z][a-z0-9]+) (.*?) ([A-Za-z][a-z0-9]+)\.',s)
 # Avoid greedy assumptions by using the known noun boundaries in our controlled grammar.
 if not m:return None
 a,rel,b=m.groups()
 masked=f'ENTITY_A {rel} ENTITY_B.'
 label=int(clf.predict([masked])[0])
 if label==0:return (a,b,1)
 if label==1:return (b,a,1)
 if label==2:return (a,b,-1)
 return (a,b,0)

def answer(src,dst,edges,maxhops=4):
 graph={};negative=set()
 for a,b,s in edges:
  if s>0:graph.setdefault(a,set()).add(b)
  elif s<0:negative.add((a,b))
 paths=[(src,[src])]
 for depth in range(maxhops):
  nxt=[]
  for a,path in paths:
   for b in sorted(graph.get(a,())):
    if (a,b) in negative or b in path:continue
    np=path+[b]
    if b==dst:return True,np
    nxt.append((b,np))
  paths=nxt
 return False,None

def run(seed,episodes=200,added=80,drift=50.):
 rng=random.Random(seed);clf=train_reader(seed)
 geo=Native();ctrl=[]; all_old=[]; all_new=[]
 def ingest(start,n,dest):
  for q in range(start,start+n):
   names=[f'z{seed}k{q}x{j}' for j in range(5)]
   # n=5 distinct names, 3 hops, question is 0 -> 3; final distractor disconnected.
   sentences=[]
   for j in range(3):
    tmpl,lab,rev=rng.choice([x for x in TEST_PATTERNS if x[1]==1])
    sentences.append(tmpl.format(a=names[j],b=names[j+1]))
   # Contradictory distractor on separate endpoint, never traversed.
   sentences.append(f'{names[4]} opposes {names[0]}.')
   for sentence in sentences:
    item=parse_sentence(sentence,clf)
    if item:
     a,b,s=item
     geo.append(a,b,s,sentence);ctrl.append((a,b,s))
   dest.append((names[0],names[3],True))
   dest.append((names[0],names[4],False))
   # negative sent separately: explicit unsupported relation query.
 def measure(qs,edges):
  return sum(answer(a,b,edges)[0]==target for a,b,target in qs),len(qs)
 ingest(0,episodes,all_old)
 baseline_before=measure(all_old,ctrl);g_before=measure(all_old,geo.edges());old_edges=geo.edges()
 geo.gauge(seed+10000,drift)
 g_after=measure(all_old,geo.edges());raw_after=measure(all_old,geo.raw_edges())
 # Adaptation consists of new facts being written after prior training; writer weights are frozen.
 ingest(episodes,added,all_new)
 geo.gauge(seed+20000,drift/2)
 old_ret=measure(all_old,geo.edges());new_ret=measure(all_new,geo.edges())
 control_old=measure(all_old,ctrl);control_new=measure(all_new,ctrl)
 # verify snapshots against the same root layout, across an extra shift.
 snap=BASE/f'root_{seed}.bin';assert geo.lib.ep_save(geo.ptr,str(snap).encode())
 before=np.asarray([geo.lib.ep_get(geo.ptr,aa,bb) for aa,bb,*_ in geo.records],dtype=np.float32)
 geo.gauge(seed+30000,drift)
 assert geo.lib.ep_load(geo.ptr,str(snap).encode())
 after=np.asarray([geo.lib.ep_get(geo.ptr,aa,bb) for aa,bb,*_ in geo.records],dtype=np.float32)
 assert np.array_equal(before,after)
 audit=geo.lib.ep_audit(geo.ptr)
 maxdiff=float(np.max(np.abs(before-after)))
 result={'seed':seed,'train_pattern_count':len(TRAIN_PATTERNS),'source_train_sentences':1200,'old_questions':len(all_old),'new_questions':len(all_new),'native_written_records':len(geo.records),'baseline_before':baseline_before,'native_before':g_before,'native_after_gauge':g_after,'raw_after_gauge':raw_after,'native_old_after_adaptation':old_ret,'native_new_after_adaptation':new_ret,'control_old_after_adaptation':control_old,'control_new_after_adaptation':control_new,'native_pair_audit':audit,'max_snapshot_readout_error':float(maxdiff),'positive_native_records':int(sum(round(float(geo.lib.ep_get(geo.ptr,aa,bb)))>0 for aa,bb,*_ in geo.records)),'snapshot_exact':bool(np.array_equal(before,after)), 'example':{'premises':[r[-1] for r in geo.records[:4]],'query':all_old[0][:2], 'answer_path':answer(*all_old[0][:2],geo.edges())[1]}}
 geo.close();return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--episodes',type=int,default=200);p.add_argument('--added',type=int,default=80);args=p.parse_args()
 results=[run(s,args.episodes,args.added) for s in args.seeds]
 (BASE/'results.json').write_text(json.dumps(results,indent=2))
 for x in results:print(json.dumps({k:v for k,v in x.items() if k!='example'}))
