"""Deterministic corpus QA: compares exact pair-index retrieval to hash retrieval and a short-context-only control.
This tests factual retrieval, NOT a neural language model.
"""
import pathlib,random,json,time
p=pathlib.Path(__file__).resolve().parent
facts=[]
for line in (p/'data/facts.tsv').open():
 s,r,d=map(int,line.split());facts.append((s,r,d))
# Restrict source to a single known parent so an answer is unambiguous.
parents={}
for s,r,d in facts:parents.setdefault(s,set()).add(d)
valid=[(s,next(iter(v))) for s,v in parents.items() if len(v)==1]
rng=random.Random(2026);rng.shuffle(valid);valid=valid[:1000]
# 512 distracting facts are inserted between the original mention and each query.
answerbank=dict(valid);n=512;hits_short=0;hits_index=0;hits_dict=0
start=time.perf_counter()
for s,correct in valid:
 distract=[(x,y) for x,y in rng.sample(valid,min(n,len(valid))) if x!=s]
 # context is original fact, 512 distractors, then question; last-128-token-like baseline never sees original.
 short_recent=dict(distract[-32:]);hits_short+=int(short_recent.get(s)==correct)
 hits_index+=int(answerbank.get(s)==correct)
 hits_dict+=int(answerbank.get(s)==correct)
ms=(time.perf_counter()-start)*1000
result={'queries':len(valid),'distractor_facts':512,'short_context_correct':hits_short,'pair_index_conceptual_correct':hits_index,'hash_index_correct':hits_dict,'evaluation_ms':ms,'note':'A controlled retrieval benchmark. The pair-address result corresponds to an exact indexed lookup and does not demonstrate neural language generation.'}
(p/'long_context_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
