"""Frozen Round-039 unrestricted decoder as independent contrast to Round-040 evidence QA.
This decoder is NOT conditioned on the new episode native memory. Its weak output
is deliberately reported, not presented as successful end-to-end generation.
"""
from pathlib import Path
import json, torch, sentencepiece as spm, numpy as np
from torch import nn
from torch.nn import functional as F
BASE=Path(__file__).resolve().parent
class LM(nn.Module):
 def __init__(self,vocab,hidden=96):
  super().__init__();self.emb=nn.Embedding(vocab,hidden);self.gru=nn.GRU(hidden,hidden,batch_first=True);self.norm=nn.LayerNorm(hidden)
  self.key=nn.Linear(hidden,16);self.value=nn.Linear(16,hidden,bias=False);self.gate=nn.Linear(hidden,hidden)
  self.head=nn.Linear(hidden,vocab);self.pair=nn.Parameter(torch.zeros(16,16))
 def forward(self,x,W=None,h=None):
  z,h=self.gru(self.emb(x),h);z=self.norm(z)
  if W is None:W=self.pair
  r=torch.tanh(self.value(torch.tanh(self.key(z)@W)))
  return self.head(z+torch.sigmoid(self.gate(z))*r),h
@torch.no_grad()
def generate(model,tok,W,prompt,steps=100,seed=8,temp=.82,topk=32):
 torch.manual_seed(seed);model.eval();h=None
 context=[tok.bos_id()]+tok.encode(prompt,out_type=int)
 if len(context)>1:_,h=model(torch.tensor(context[:-1])[None,:],W)
 v=torch.tensor([[context[-1]]]);out=[]
 for _ in range(steps):
  logits,h=model(v,W,h);scores=logits[0,-1]/temp;z,ix=scores.topk(topk);pr=F.softmax(z,0)
  v=ix[torch.multinomial(pr,1)].view(1,1);id=int(v.item())
  if id==tok.eos_id():break
  if id not in (tok.pad_id(),tok.bos_id()):out.append(id)
 return prompt+tok.decode(out)
if __name__=='__main__':
 torch.set_num_threads(2)
 ckpt=torch.load(BASE/'prior_native_lm.pt',map_location='cpu',weights_only=True)
 tok=spm.SentencePieceProcessor(model_file=str(BASE/'wordnet_bpe.model'))
 model=LM(ckpt['vocab'],ckpt['hidden']);model.load_state_dict(ckpt['model']);model.eval()
 # The current trained coefficients are encoded in 8MiB root, not the zeroed .pair placeholder.
 root=(BASE/'prior_lm_root.bin').read_bytes();assert len(root)==8388616 and root[:8]==b'NANCYC30'
 W=np.zeros((16,16),dtype=np.float32)
 # Round-039 native layout: 32768-byte head area and 256 bytes per unordered pair.
 def pair_idx(i,j):
  if i>j:i,j=j,i
  return i*(511-i)//2+j-i-1
 def edge(i,j,c):
  import struct
  off=8+32768+pair_idx(i,j)*256+(0 if i<j else 128)+4*c
  return struct.unpack_from('<f',root,off)[0]
 for d in range(16):
  a=1+2*d;b=a+1
  for c in range(16):W[d,c]=edge(0,a,c)+edge(a,b,c)-edge(0,b,c)
 W=torch.from_numpy(W)
 prompts=['An unfamiliar organism is a kind of animal. Therefore','A small animal','An instrument used for']
 outputs={s:generate(model,tok,W,s,seed=77+idx) for idx,s in enumerate(prompts)}
 (BASE/'free_samples.json').write_text(json.dumps(outputs,indent=2))
 print(json.dumps(outputs,indent=2))
