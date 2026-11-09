"""Stress recursive reasoning beyond two edges using the same native C(256,2) root.
Three real WordNet links per positive query; negative queries have a disconnected third link.
The output sentence is generated only AFTER three source facts are read and composed.
"""
from pathlib import Path
import random,json,ctypes,subprocess,sys
from experiment import parse_wordnet
T=Path(__file__).resolve().parent

def find_three(rows):
 res=[];used=set()
 for id,a in rows.items():
  if id in used:continue
  for m in a['parents']:
   if m not in rows:continue
   for g in rows[m]['parents']:
    if g not in rows:continue
    for gg in rows[g]['parents']:
     if gg not in rows or gg in (id,m,g):continue
     names=[rows[t]['name'] for t in (id,m,g,gg)]
     if all(1<len(s)<27 and s.isascii() for s in names):
      used.add(id);res.append((id,names));break
    else:continue
    break
   else:continue
   break
 return res

def main():
 rows=parse_wordnet(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/wn3.1.dict.zip'))
 cases=find_three(rows);random.Random(13).shuffle(cases)
 if len(cases)<100:raise ValueError(len(cases))
 # 60 real positive paths and 60 disrupted paths, not reused during two-hop gate training.
 positives=cases[-60:];negatives=cases[-120:-60]
 allcases=[]
 for k,(id,names) in enumerate(positives):
  a,b,c,d=names;allcases.append(([(a,b),(b,c),(c,d)],f'{a} is a kind of {d}.',True))
 for k,(id,names) in enumerate(negatives):
  a,b,c,d=names;other=cases[k][1];x,y=other[1],other[2]
  if c==x:x='unknown-intermediate'
  allcases.append(([(a,b),(b,c),(x,y)],'I do not have enough evidence.',False))
 libpath=T/'libepisodic.so'
 if not libpath.exists():subprocess.run(['c++','-std=c++20','-O3','-shared','-fPIC',str(T/'native_episodic.cpp'),'-o',str(libpath)],check=True)
 L=ctypes.CDLL(str(libpath));L.memory_new.restype=ctypes.c_void_p;L.memory_free.argtypes=[ctypes.c_void_p]
 L.memory_store.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];L.memory_read.argtypes=[ctypes.c_void_p,ctypes.c_int];L.memory_read.restype=ctypes.c_int
 L.memory_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];L.memory_raw.restype=ctypes.c_int
 L.memory_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
 p=L.memory_new();assert p
 step=80;total=0;raw=0;yes=0;no=0;example=[]
 try:
  for n,(facts,ans,pos) in enumerate(allcases):
   for k,(a,b) in enumerate(facts):
    payload=(a+'>'+b).encode('ascii')[:step]
    for t,v in enumerate(payload.ljust(step,b'\x00')):L.memory_store(p,(n*3+k)*step+t,v)
  L.memory_gauge(p,713,40.)
  for n,(facts,expected,pos) in enumerate(allcases):
   parsed=[];raws=[]
   for k in range(3):
    slot=(n*3+k)*step
    payload=bytes(L.memory_read(p,slot+t) for t in range(step)).rstrip(b'\x00').decode('ascii')
    rawp=bytes(L.memory_raw(p,slot+t) for t in range(step)).rstrip(b'\x00').decode('ascii','replace')
    parsed.append(tuple(payload.split('>',1)));raws.append(tuple(rawp.split('>',1)))
   if len(parsed[0])==2 and len(parsed[1])==2 and len(parsed[2])==2 and parsed[0][1]==parsed[1][0] and parsed[1][1]==parsed[2][0]:
    generated=f'{parsed[0][0]} is a kind of {parsed[2][1]}.'
   else:generated='I do not have enough evidence.'
   correct=generated==expected;total+=correct
   if pos:yes+=correct
   else:no+=correct
   if len(raws[0])==2 and len(raws[1])==2 and len(raws[2])==2 and raws[0][1]==raws[1][0] and raws[1][1]==raws[2][0]:
    rawgen=f'{raws[0][0]} is a kind of {raws[2][1]}.'
   else:rawgen='I do not have enough evidence.'
   raw+=rawgen==expected
   if n in (0,60):example.append({'facts':facts,'generated':generated,'expected':expected})
 finally:L.memory_free(p)
 out={'real_wordnet_three_hop_positives':60,'broken_path_negatives':60,'native_cycle_exact_answers':total,'positive_correct':yes,'negative_abstentions':no,'raw_edge_exact_answers':raw,'episodes':120,'slots_used':120*3*80,'examples':example,'note':'A deterministic typed-path interpreter, not a trained 3-hop neural generalizer. Negative cases are constructed by replacing the third source edge.'}
 (T/'recursive_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
