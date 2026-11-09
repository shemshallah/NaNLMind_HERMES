"""Native root gauge audit and honest semantic-generation metrics."""
import sys, json, subprocess, ctypes, hashlib, random, math
from pathlib import Path
import torch
from train import parse_wn,cases_build,Seq2Seq,encode
T=Path(__file__).parent;torch.set_num_threads(2)
rows=parse_wn(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/wn3.1.dict.zip'));split=cases_build(rows)

def load(seed):
 ck=torch.load(T/f'seq2seq_{seed}.pt',map_location='cpu',weights_only=False);itos=ck['vocab'];stoi={t:i for i,t in enumerate(itos)};m=Seq2Seq(len(itos),32);m.load_state_dict(ck['model']);m.eval();return m,stoi,itos

def generated(m,stoi,itos,cases):
 x,_=encode(cases,stoi)
 with torch.no_grad():pred=m(x,max_len=13)
 ans=[]
 for p in pred.tolist():
  toks=[]
  for i in p:
   if i==3:break
   if i>3:toks.append(itos[i])
  ans.append(' '.join(toks))
 return ans

def semantic_ok(answer,c):
 if c['negative']:return answer=='the evidence is insufficient .'
 return answer in [f'e0 is a type of e{c["hops"]} .',f'e0 belongs to e{c["hops"]} .',f'e0 is classified as e{c["hops"]} .']

out={'semantic':[],'native':{}}
for seed in (17,29):
 m,stoi,itos=load(seed);cases=split['test'];answers=[]
 for i in range(0,len(cases),256):answers+=generated(m,stoi,itos,cases[i:i+256])
 npos=sum(not c['negative'] for c in cases);npos_ok=sum(semantic_ok(a,c) for a,c in zip(answers,cases) if not c['negative']);nneg_ok=sum(semantic_ok(a,c) for a,c in zip(answers,cases) if c['negative'])
 checks={'seed':seed,'n':len(cases),'valid_positive_accuracy':npos_ok/npos,'negative_abstention_accuracy':nneg_ok/(len(cases)-npos),'semantic_accuracy':(npos_ok+nneg_ok)/len(cases),'exact_surface_match':sum(a==c['answer'] for a,c in zip(answers,cases))/len(cases),'examples':[{'facts':c['premises'],'generated':a,'reference':c['answer'],'semantic_correct':semantic_ok(a,c)} for a,c in zip(answers,cases)][:12]}
 out['semantic'].append(checks);print('MODEL',seed,checks['semantic_accuracy'],checks['valid_positive_accuracy'],checks['negative_abstention_accuracy'],flush=True)

# Native full-root check with 90 selected real held-out sentences. Each byte of
# prompt stored as real-valued invariant circulation on a unique anchored triangle.
cpp=T/'native_episodic.cpp';subprocess.run(['g++','-std=c++20','-O3','-shared','-fPIC',str(cpp),'-o',str(T/'libepisodic.so')],check=True)
l=ctypes.CDLL(str(T/'libepisodic.so'));l.memory_new.restype=ctypes.c_void_p
l.memory_store.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_int)
l.memory_read.argtypes=(ctypes.c_void_p,ctypes.c_int);l.memory_read.restype=ctypes.c_int
l.memory_raw.argtypes=(ctypes.c_void_p,ctypes.c_int);l.memory_raw.restype=ctypes.c_int
l.memory_gauge.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_float)
l.memory_free.argtypes=(ctypes.c_void_p,)
cases=split['test'][:90];maxlen=170;assert len(cases)*maxlen<32385
p=l.memory_new();assert p
try:
 for i,c in enumerate(cases):
  data=c['prompt'].encode('ascii');assert len(data)<maxlen
  for j,b in enumerate(data):l.memory_store(p,i*maxlen+j,int(b))
 pristine=hashlib.sha256(ctypes.string_at(p,8388608)).hexdigest()
 before=[bytes(l.memory_read(p,i*maxlen+j) for j in range(len(c['prompt']))).decode('ascii') for i,c in enumerate(cases)]
 assert all(t==c['prompt'] for t,c in zip(before,cases))
 l.memory_gauge(p,2091,80.0)
 cycled=[];raw=[]
 for i,c in enumerate(cases):
  n=len(c['prompt']);cycled.append(bytes(l.memory_read(p,i*maxlen+j) for j in range(n)).decode('ascii',errors='replace'))
  raw.append(bytes(l.memory_raw(p,i*maxlen+j) for j in range(n)).decode('ascii',errors='replace'))
 assert cycled==before
 m,stoi,itos=load(29)
 restored=[dict(c,prompt=txt) for c,txt in zip(cases,cycled)];reference=generated(m,stoi,itos,cases);cycle_gen=generated(m,stoi,itos,restored)
 # raw edge decoded bytes; unknown tokens map to UNK rather than accidently using correct prompt
 raw_cases=[dict(c,prompt=txt) for c,txt in zip(cases,raw)];raw_gen=generated(m,stoi,itos,raw_cases)
 out['native']={'root_size_bytes':8388608,'test_cases':len(cases),'pristine_root_sha256':pristine,'cycle_exact_prompt_recovery':sum(a==b for a,b in zip(cycled,before))/len(cases),'raw_exact_prompt_recovery':sum(a==b for a,b in zip(raw,before))/len(cases),'cycle_generation_unchanged':sum(a==b for a,b in zip(cycle_gen,reference))/len(cases),'raw_generation_unchanged':sum(a==b for a,b in zip(raw_gen,reference))/len(cases),'cycle_semantic_accuracy':sum(semantic_ok(a,c) for a,c in zip(cycle_gen,cases))/len(cases),'raw_semantic_accuracy':sum(semantic_ok(a,c) for a,c in zip(raw_gen,cases))/len(cases),'conventional_memory_prompt_recovery':1.0,'conventional_generation_unchanged':1.0}
 print('NATIVE',out['native'],flush=True)
finally:l.memory_free(p)
(T/'audit_results.json').write_text(json.dumps(out,indent=2))
