"""Holdout language and depth shift controls: known limitation tests."""
import json
from pathlib import Path
from train import parse_wn,cases_build,Seq2Seq,encode
import torch
def load(seed):
 ck=torch.load(T/f'seq2seq_{seed}.pt',weights_only=False);itos=ck['vocab'];stoi={t:i for i,t in enumerate(itos)};m=Seq2Seq(len(itos),32);m.load_state_dict(ck['model']);m.eval();return m,stoi,itos
def generated(m,stoi,itos,cases):
 x,_=encode(cases,stoi)
 with torch.no_grad():y=m(x,max_len=13)
 outs=[]
 for row in y.tolist():
  words=[]
  for i in row:
   if i==3:break
   if i>3:words.append(itos[i])
  outs.append(' '.join(words))
 return outs
def semantic_ok(a,c):
 if c['negative']:return a=='the evidence is insufficient .'
 h=c['hops'];return a in [f'e0 is a type of e{h} .',f'e0 belongs to e{h} .',f'e0 is classified as e{h} .']
split=cases_build(parse_wn(Path('/mnt/data/wn3.1.dict.zip')))
T=Path(__file__).parent
m,stoi,itos=load(29)
positives=[c for c in split['test'] if not c['negative']][:300]

def score(cases):
 answers=[]
 for i in range(0,len(cases),128):answers+=generated(m,stoi,itos,cases[i:i+128])
 return {'n':len(cases),'valid_answer_rate':sum(semantic_ok(a,c) for a,c in zip(answers,cases))/len(cases),'first_samples':[{'prompt':c['prompt'],'expected':c['answer'],'generated':a} for c,a in zip(cases,answers)][:3]}
# Construct contradictory relation with same entity structure, requiring lexical semantics, not positional identity
contradict=[]
for c in positives:
 v=dict(c);v['negative']=True;v['answer']='the evidence is insufficient .';v['prompt']=c['prompt'].replace('belongs to','opposes').replace('is a type of','opposes').replace('is classified as','opposes');contradict.append(v)
# Paraphrase using unseen relation wording preserving the same positive graph
paraphrase=[]
for c in positives:
 v=dict(c);v['prompt']=c['prompt'].replace('belongs to','descends from').replace('is a type of','is a subtype of').replace('is classified as','is a subclass of');paraphrase.append(v)
# 4-hop chains are outside max training depth; source is compositional but no training examples for e4 conclusion
four=[]
for c in positives:
 if c['hops']!=3:continue
 v=dict(c);v['hops']=4;v['premises']=c['premises']+['e3 belongs to e4 .'];v['prompt']=c['prompt'].replace(' question :',' next : e3 belongs to e4 . question :').replace('relative to e3 ?','relative to e4 ?');v['answer']='e0 belongs to e4 .';four.append(v)
r={'in_distribution':score(positives),'contradiction_wording':score(contradict),'unseen_paraphrase':score(paraphrase),'four_hop_extrapolation':score(four[:300])}
(T/'stress_results.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:{'n':v['n'],'valid_answer_rate':v['valid_answer_rate']} for k,v in r.items()},indent=2))
