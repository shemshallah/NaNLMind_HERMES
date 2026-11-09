"""Round 037: verify actual trained autoregressive seq2seq output from native
cycle-read multi-source sentences, against conventional and damaged raw reads.
Neural checkpoint was trained independently in Round 036; no new generative training.
"""
import json, random, ctypes
from pathlib import Path
import torch
from experiment import Native,wordnet,paths
from seq2seq_reference import Seq2Seq, encode
P=Path(__file__).parent
torch.set_num_threads(2)
ck=torch.load(P/'robust_29.pt',map_location='cpu',weights_only=False)
itos=ck['vocab'];stoi={t:i for i,t in enumerate(itos)}
m=Seq2Seq(len(itos),32);m.load_state_dict(ck['model']);m.eval()
rows=wordnet(Path('/mnt/data/wn3.1.dict.zip'));chains=paths(rows,limit=1500)[-110:]
rng=random.Random(137);n=Native();slot=0;episodes=[]
try:
 for i,ch in enumerate(chains[:90]):
  hop=2 if i%2==0 else min(3,len(ch)-1)
  negative=i%3==0
  facts=[f'e{j} is a type of e{j+1} .' for j in range(hop)]
  if negative:facts[-1]=f'e{hop+1} is a type of e{hop} .'
  rng.shuffle(facts)
  prompt=' premise : '+' next : '.join(facts)+f' question : what is e0 relative to e{hop} ?'
  # Physical root stores all prompt bytes, each in one independent triangle
  start=slot
  for byte in prompt.encode('ascii'):
   n.write(byte);slot+=1
  episodes.append({'start':start,'prompt':prompt,'hops':hop,'negative':negative,'concepts':ch})
 assert slot<32385
 original=[e['prompt'] for e in episodes]
 def decode(read):
  return [''.join(chr(max(0,min(255,round(read(e['start']+j))))) for j in range(len(e['prompt']))) for e in episodes]
 def generate(inputs):
  generated=[]
  for off in range(0,len(inputs),30):
   chunk=[{'prompt':s,'answer':'the evidence is insufficient .'} for s in inputs[off:off+30]]
   x,_=encode(chunk,stoi)
   with torch.no_grad():ys=m(x,max_len=13).tolist()
   for line in ys:
    words=[]
    for z in line:
     if z==3:break
     if z>3:words.append(itos[z])
    generated.append(' '.join(words))
  return generated
 before=generate(original)
 n.gauge(1701,65)
 cycle=decode(n.read);raw=decode(n.raw)
 output_cycle=generate(cycle);output_raw=generate(raw)
 def semantic(a,e):
  if e['negative']:return a=='the evidence is insufficient .'
  h=e['hops'];return a in [f'e0 is a type of e{h} .',f'e0 belongs to e{h} .',f'e0 is classified as e{h} .']
 out={'episodes':len(episodes),'input_bytes_stored':slot,'neural_checkpoint':'Round036 robust_29.pt','generative_model_parameters':sum(p.numel() for p in m.parameters()),'native_cycle_source_recovery':sum(a==b for a,b in zip(cycle,original)),'raw_source_recovery':sum(a==b for a,b in zip(raw,original)),'cycle_neural_outputs_unchanged':sum(a==b for a,b in zip(output_cycle,before)),'raw_neural_outputs_unchanged':sum(a==b for a,b in zip(output_raw,before)),'native_semantic_correct':sum(semantic(a,e) for a,e in zip(output_cycle,episodes)),'conventional_index_semantic_correct':sum(semantic(a,e) for a,e in zip(before,episodes)),'examples':[{'premises':e['prompt'],'native_generated_slots':a,'actual_names':e['concepts'][:e['hops']+1]} for e,a in list(zip(episodes,output_cycle))[:8]]}
 (P/'generation_results.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k!='examples'},indent=2))
finally:n.close()
