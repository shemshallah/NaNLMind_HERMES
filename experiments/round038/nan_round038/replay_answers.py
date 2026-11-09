"""Reproduce a small held-out answer exhibit from the trained Round038 hybrid."""
import json,torch
from pathlib import Path
from round038 import Reasoner,get_writer_preds,wordnet,paths,make_episodes,encode_native,CATS
BASE=Path(__file__).parent
if __name__=='__main__':
 torch.set_num_threads(2);seed=29
 preds=get_writer_preds(seed);data=paths(wordnet(Path('/mnt/data/files/wn3.1.dict.zip')),seed=seed+1000,limit=900)
 examples,descriptors=make_episodes(data,seed=seed+137,writer_predictions=preds)
 x,gauge,audit=encode_native(examples,snapshot_name='root_038_replay_temporary.bin')
 model=Reasoner('full');ck=torch.load(BASE/f'trig_wordnet_full_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
 ix=torch.randperm(len(examples),generator=torch.Generator().manual_seed(seed));test=ix[int(.8*len(examples)):]
 with torch.no_grad():logits,_=model.decode(x[test],teacher=False);tokens=logits.argmax(-1).tolist()
 classes=[int((row[1]-3)) if len(row)>1 and row[1] in (3,4,5,6) else -1 for row in tokens]
 gold=[examples[int(k)][1] for k in test]
 assert all(a==b for a,b in zip(classes,gold)),(sum(a==b for a,b in zip(classes,gold)),len(classes))
 exhibit=[]
 for target_label in range(4):
  for j,idx in enumerate(test):
   if gold[j]!=target_label:continue
   d=descriptors[int(idx)]
   exhibit.append({'question':f"Does the supplied evidence establish that '{d['source']}' is a kind of '{d['target']}'?",'generated_tokens':['evidence',CATS[classes[j]],'<eos>'],'evidence_state':CATS[gold[j]],'note':'Synthetic additional negation/uncertainty/distractor facts are part of the episode, not necessarily WordNet ontology assertions.'})
   if sum(z['evidence_state']==CATS[target_label] for z in exhibit)>=3:break
 out={'checkpoint':'trig_wordnet_full_29.pt','heldout_episodes':len(gold),'exact_autoregressive_label_predictions':sum(a==b for a,b in zip(classes,gold)),'generated_examples':exhibit}
 (BASE/'generated_answers_038.json').write_text(json.dumps(out,indent=2))
 (BASE/'root_038_replay_temporary.bin').unlink()
 print(json.dumps({'heldout_episodes':len(gold),'exact':out['exact_autoregressive_label_predictions'],'examples':len(exhibit)},indent=2))
