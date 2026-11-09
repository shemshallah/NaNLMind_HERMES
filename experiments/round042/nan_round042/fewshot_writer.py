#!/usr/bin/env python3
"""Continual learning of unseen relation wording with replay of existing grammatical distinctions.
Teacher provides THREE labeled new relation phrases. Replay protects the old 16 phrases.
Ablation studies relation-expression adaptation, not unrestricted grammar acquisition.
"""
import json,torch
from pathlib import Path
import torch.nn.functional as F
import dynamic_joint as dj
B=Path(__file__).resolve().parent
NEW=[('is subsumed by',1),('comprises',-1),('conflicts with',0)]

def run(seed=17,steps=160):
 torch.manual_seed(seed+3)
 ck=torch.load(B/f'model_{seed}.pt',map_location='cpu',weights_only=False)
 m=dj.Joint(ck['dim']);m.load_state_dict(ck['model']);m.eval()
 oldp=list(dj.PHRASES);newp=oldp+NEW
 oldv=dict(dj.PHRASE_VOCAB);v=dict(oldv)
 for phrase,_ in NEW:
  for word in phrase.split():
   if word not in v:v[word]=len(v)
 ids=[]
 for p,_ in newp:
  words=[v[w] for w in p.split()]
  ids.append(words+[0]*(6-len(words)))
 indices=torch.tensor(ids)
 oldw=m.phrase.weight.detach().clone()
 newemb=torch.nn.Embedding(len(v),oldw.shape[1],padding_idx=0)
 with torch.no_grad():newemb.weight[:len(oldv)].copy_(oldw)
 m.phrase=newemb
 def predict():
  with torch.no_grad():
   e=m.phrase(indices)
   z=e.sum(1)/(indices!=0).sum(1).clamp_min(1).unsqueeze(-1)
   scores=m.writer(z).softmax(-1)
  return scores
 before=predict();labels=torch.tensor([0 if direction==1 else (1 if direction==-1 else 2) for _,direction in newp])
 # Jointly retain the original phrase repertoire via replay at every update.
 opt=torch.optim.AdamW(list(m.phrase.parameters())+list(m.writer.parameters()),lr=.01)
 for step in range(steps):
  opt.zero_grad();e=m.phrase(indices);z=e.sum(1)/(indices!=0).sum(1).clamp_min(1).unsqueeze(-1)
  score=m.writer(z)
  loss=F.cross_entropy(score,labels)+0.15*F.mse_loss(m.phrase.weight[:len(oldv)],oldw)
  loss.backward();torch.nn.utils.clip_grad_norm_(list(m.phrase.parameters())+list(m.writer.parameters()),1);opt.step()
 after=predict()
 b=before.argmax(-1);a=after.argmax(-1)
 old_before=int((b[:len(oldp)]==labels[:len(oldp)]).sum());old_after=int((a[:len(oldp)]==labels[:len(oldp)]).sum())
 new_before=int((b[len(oldp):]==labels[len(oldp):]).sum());new_after=int((a[len(oldp):]==labels[len(oldp):]).sum())
 row={'seed':seed,'steps':steps,'old_phrase_correct_before':old_before,'old_phrase_correct_after':old_after,'old_phrase_total':len(oldp),'new_phrase_correct_before':new_before,'new_phrase_correct_after':new_after,'new_phrase_total':len(NEW),'before_new_probs':before[len(oldp):].tolist(),'after_new_probs':after[len(oldp):].tolist(),'new_phrases':NEW}
 torch.save({'model':m.state_dict(),'dim':ck['dim'],'seed':seed,'phrases':newp,'phrase_vocab':v,'source_checkpoint':f'model_{seed}.pt'},B/f'adapted_model_{seed}.pt')
 (B/f'fewshot_writer_{seed}.json').write_text(json.dumps(row,indent=2))
 print('FEW_SHOT',row,flush=True)
 return row
if __name__=='__main__':run(17)
