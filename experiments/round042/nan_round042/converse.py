#!/usr/bin/env python3
"""Interactive, persistent single-universe concept chat for the Round 042 trained model.
The 8 MiB C++ mirror root stores named, directional, learned-strength relationships.
Free-vocabulary conversation is not supported; this is a grounded conceptual dialogue.
"""
import argparse, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dynamic_joint import Joint,NativeRoot,PHRASES,PHRASE_IDS,WORDS,Episode,ROOT

def norm(s):return re.sub(r'\s+',' ',s.strip().strip('.?! ').lower())

def parse_fact(s):
 s=norm(s).replace('belongs to the category ', 'belongs to ').replace('falls under the category ', 'falls under ')
 for idx,(phrase,_) in sorted(enumerate(PHRASES),key=lambda x:len(x[1][0]),reverse=True):
  m=re.match(r'^(.*?)\s+'+re.escape(phrase)+r'\s+(.*?)$',s)
  if m and norm(m.group(1)) and norm(m.group(2)):
   return norm(m.group(1)),idx,norm(m.group(2))
 raise ValueError('Unrecognized relationship wording. Use phrases such as "is a kind of", "belongs to", "includes", or "opposes".')

def parse_question(s):
 s=norm(s)
 for pat in [r'^is (.+?) (?:a kind of|a type of|an instance of|a member of) (.+)$',r'^does (.+?) belong to (.+)$',r'^(.+?)\s*->\s*(.+)$']:
  m=re.fullmatch(pat,s)
  if m:return norm(m.group(1)),norm(m.group(2))
 raise ValueError('Ask e.g. "Is a penguin a kind of animal?", "Does a penguin belong to animal?" or "penguin -> animal".')

class Universe:
 def __init__(self,model_path,root_path=None):
  ck=torch.load(model_path,map_location='cpu',weights_only=False)
  if 'phrases' in ck:
   import dynamic_joint as dj
   # The new phrase ids are saved with the adapted embedding checkpoint.
   PHRASES.clear();PHRASES.extend([tuple(p) for p in ck['phrases']])
   dj.PHRASE_VOCAB.clear();dj.PHRASE_VOCAB.update(ck['phrase_vocab'])
   new_ids=[]
   for phrase,_ in PHRASES:
    t=[dj.PHRASE_VOCAB[word] for word in phrase.split()]
    new_ids.append(t+[0]*(6-len(t)))
   new_tensor=torch.tensor(new_ids,dtype=torch.long)
   dj.PHRASE_IDS=new_tensor
   global PHRASE_IDS
   PHRASE_IDS=new_tensor
  self.model=Joint(ck['dim']);self.model.load_state_dict(ck['model']);self.model.eval()
  self.speaker=None
  sp=ROOT/f"four_speaker_{ck.get('seed',17)}.pt"
  if sp.exists():
   from four_state_language import Speaker
   voice=torch.load(sp,map_location='cpu',weights_only=False)
   self.speaker=Speaker();self.speaker.load_state_dict(voice['state']);self.speaker.eval()
  self.root=NativeRoot()
  self.root.L.ep_set_layer.argtypes=[__import__('ctypes').c_void_p,__import__('ctypes').c_int,__import__('ctypes').c_int,__import__('ctypes').c_int,__import__('ctypes').c_float]
  self.root.L.ep_get_layer.argtypes=[__import__('ctypes').c_void_p,__import__('ctypes').c_int,__import__('ctypes').c_int,__import__('ctypes').c_int]
  self.root.L.ep_get_layer.restype=__import__('ctypes').c_float
  self.root_path=Path(root_path) if root_path else ROOT/'my_universe.bin'
  self.meta_path=self.root_path.with_suffix('.json')
  self.names={};self.facts=[]
  if self.meta_path.exists():
   data=json.loads(self.meta_path.read_text());self.names=data['names'];self.facts=data['facts']
   assert self.root.L.ep_load(self.root.ptr,str(self.root_path).encode()),'Failed to restore native root'
   actual=hashlib.sha256(self.root_path.read_bytes()).hexdigest()
   if actual!=data['sha256']:raise RuntimeError('Snapshot checksum mismatch')
 def _id(self,name,create=True):
  if name not in self.names:
   if not create:return None
   if len(self.names)>=255:raise ValueError('Universe full: 255 named heads (head 0 reserved for geometric anchor).')
   self.names[name]=len(self.names)+1
  return self.names[name]
 def teach(self,statement,persist=True):
  a,pid,b=parse_fact(statement);ai=self._id(a);bi=self._id(b)
  if ai==bi:raise ValueError('Self-edge is not a mirror pair.')
  with torch.no_grad():
   t=PHRASE_IDS[pid].unsqueeze(0);embed=self.model.phrase(t)
   h=embed.sum(1)/(t.ne(0).sum(1).clamp_min(1).unsqueeze(-1))
   scores=self.model.writer(h).softmax(-1)[0]
   v=float((scores[0]-scores[1])*.98)
  lo,hi=sorted((ai,bi));value=v if ai<bi else -v
  if pid==10: # an explicit negative ISA assertion; independent mirror lane
   self.root.L.ep_set_layer(self.root.ptr,1,lo,hi,(1 if ai<bi else -1)*float(scores[2]))
   typ='refutes'
  elif PHRASES[pid][1]==0:
   typ='other' # not logically equivalent to ISA refutation
  else:
   self.root.L.ep_set_layer(self.root.ptr,0,lo,hi,value)
   typ='supports'
  self.facts=[f for f in self.facts if not (f['a']==a and f['b']==b and f.get('type')==typ)]
  self.facts.append({'a':a,'b':b,'phrase':PHRASES[pid][0],'strength':float(scores[2]) if typ=='refutes' else v,'type':typ,'statement':statement})
  if persist:self.save()
  return f'Learned ({typ}): {a} {PHRASES[pid][0]} {b}' 
 def save(self):
  p=self.root_path;tmp=p.with_suffix('.tmp');tmpmeta=self.meta_path.with_suffix('.json.tmp')
  assert self.root.L.ep_save(self.root.ptr,str(tmp).encode())
  digest=hashlib.sha256(tmp.read_bytes()).hexdigest()
  tmpmeta.write_text(json.dumps({'names':self.names,'facts':self.facts,'sha256':digest},indent=2))
  tmp.replace(p);tmpmeta.replace(self.meta_path)
 def _graph(self):
  n=len(self.names);A=torch.zeros(1,n,n)
  for a in range(1,n+1):
   for b in range(a+1,n+1):
    v=float(self.root.L.ep_get(self.root.ptr,a,b));A[0,a-1,b-1]=max(0,v);A[0,b-1,a-1]=max(0,-v)
  return A
 def _mem(self,src,dst):
  A=self._graph();n=A.size(1)
  s=torch.zeros(1,n);d=torch.zeros(1,n);s[0,src-1]=1;d[0,dst-1]=1
  weight=torch.sigmoid(self.model.reach_scale)
  position=s;sumscore=torch.zeros(1);weighted_hops=torch.zeros(1)
  for step in range(1,7):
   position=torch.bmm(position[:,None,:],A*weight).squeeze(1)
   contribution=(position*d).sum(1)
   sumscore=sumscore+contribution;weighted_hops=weighted_hops+contribution*step
  hops=weighted_hops/(sumscore+1e-6)
  return torch.stack([10*(sumscore-self.model.threshold),hops/6,torch.log1p(sumscore)],dim=1)
 def ask(self,s,temperature=0,seed=17):
  a,b=parse_question(s);src=self._id(a,False);dst=self._id(b,False)
  if src is None or dst is None:return 'The available facts do not establish that relationship.'
  mem=self._mem(src,dst)
  # Four-valued evidence: native refutation is an independent second cycle lane.
  neg=float(self.root.L.ep_get_layer(self.root.ptr,1,min(src,dst),max(src,dst)))
  if src>dst:neg=-neg
  has_neg=neg>0.50
  has_support=float(mem[0,0].detach())>0
  if self.speaker is not None:
   from dynamic_joint import Episode
   nodes=['']*len(self.names)
   for name,k in self.names.items():nodes[k-1]=name
   eps=[Episode(nodes,src-1,dst-1,[],0,[],[])]
   features=torch.stack([mem[0,0],torch.tensor(neg)*9-4.5,mem[0,1]])[None,:]
   with torch.no_grad():return self.speaker.generate(features,eps,temperature=temperature)[0]
  if has_neg and has_support:
   return f'Conflicting evidence: the stored relations support {a} being a kind of {b}, but an explicit assertion refutes it.'
  if has_neg:
   return f'The stored evidence explicitly refutes that {a} is a kind of {b}.'
  # Autoregressive generation from learned memory conditioning, no proof-status token.
  with torch.no_grad():
   cond=self.model.memory(mem)
   h=cond.unsqueeze(0).contiguous()
   prev=torch.tensor([[1]])
   words=[];torch.manual_seed(seed)
   for _ in range(38):
    z,h=self.model.decoder(self.model.embedding(prev)+cond[:,None,:],h)
    logits=self.model.head(z[:,0]);token=torch.multinomial(F.softmax(logits/temperature,-1),1) if temperature>0 else logits.argmax(-1,keepdim=True)
    w=WORDS[int(token[0,0])]
    if w=='<eos>':break
    if w=='<S>':w=a
    if w=='<D>':w=b
    if w not in ('<bos>','<pad>'):words.append(w)
    prev=token
  return ' '.join(words).replace(' .','.').replace(' ,',',')
 def describe(self,name):
  name=norm(name).removeprefix('a ').removeprefix('an ').removeprefix('the ')
  ai=self._id(name,False)
  if ai is None:return 'I have no stored information about '+name+'.'
  candidates=[]
  for other,bi in self.names.items():
   if bi==ai:continue
   low,high=sorted((ai,bi))
   val=float(self.root.L.ep_get_layer(self.root.ptr,0,low,high))
   if ai>bi:val=-val
   if val>.50:candidates.append((val,other))
  if not candidates:return 'I do not have a supported classification for '+name+'.'
  _,parent=max(candidates)
  return self.ask(f'is {name} a kind of {parent}?')
 def evidence_path(self,a,b):
  from collections import deque
  a,b=norm(a),norm(b)
  src=self._id(a,False);dst=self._id(b,False)
  if src is None or dst is None:return []
  backward={v:k for k,v in self.names.items()};q=deque([(src,[src])]);seen={src}
  while q:
   at,path=q.popleft()
   if at==dst:return [backward[v] for v in path]
   if len(path)>6:continue
   for v in range(1,len(self.names)+1):
    if v==at or v in seen:continue
    low,high=sorted((at,v))
    val=float(self.root.L.ep_get_layer(self.root.ptr,0,low,high))
    if at>v:val=-val
    if val>0.5:seen.add(v);q.append((v,path+[v]))
  return []
 def chat(self,utterance):
  s=utterance.strip()
  if not s:return ''
  q=norm(s)
  if q.startswith('why '):
   request=q[4:]
   try:
    a,b=parse_question(request)
    answer=self.ask(request)
    path=self.evidence_path(a,b)
    return answer+('\nEvidence: '+' → '.join(path) if len(path)>1 else '')
   except ValueError:pass
  m=re.fullmatch(r'what is (?:a |an |the )?(.+)',q)
  if m:return self.describe(m.group(1))
  if q.startswith('is ') or q.startswith('does ') or '->' in q:
   try:return self.ask(q)
   except ValueError:pass
  try:return self.teach(s)
  except ValueError:return 'I cannot yet interpret that sentence. Teach me a supported relation or ask about two concepts.'
 def gauge(self,seed=123,scale=4.0):self.root.gauge(seed,scale);self.save()
 def forget(self,a,b):
  a,b=norm(a),norm(b)
  ia=self._id(a,False);ib=self._id(b,False)
  if ia is None or ib is None:return 'No matching fact exists.'
  self.root.L.ep_set_layer(self.root.ptr,0,min(ia,ib),max(ia,ib),0.)
  self.root.L.ep_set_layer(self.root.ptr,1,min(ia,ib),max(ia,ib),0.)
  self.facts=[f for f in self.facts if {f['a'],f['b']}!={a,b}]
  self.save();return 'Cleared the pair relationship.'
 def close(self):self.root.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',default=str(ROOT/'model_17.pt'));ap.add_argument('--root',default=str(ROOT/'my_universe.bin'));ap.add_argument('--demo',action='store_true');a=ap.parse_args()
 u=Universe(a.model,a.root)
 if a.demo:
  for t in ['penguin is a kind of bird','bird is a kind of animal','animal belongs to the category organism']:
   print('teach>',t,'=>',u.teach(t))
  for q in ['Is penguin a kind of animal?','Is penguin a kind of organism?','Is penguin a kind of machine?']:
   print('ask>',q,'=>',u.ask(q))
  before=u.ask('Is penguin a kind of organism?');u.gauge(33,4);after=u.ask('Is penguin a kind of organism?')
  print('Gauge stable:',before==after,'response:',after)
  u.forget('bird','animal');print('after deleting bird -> animal:',u.ask('Is penguin a kind of organism?'))
  u.teach('bird is a kind of animal');print('after restoring:',u.ask('Is penguin a kind of organism?'))
 else:
  print('Single native geometric universe. Commands: teach <sentence>, ask <question>, forget <a> -> <b>, gauge, save, list, quit')
  while True:
   try:line=input('nan> ').strip()
   except (EOFError,KeyboardInterrupt):break
   if not line:continue
   if line in ('q','quit','exit'):break
   try:
    if line.startswith('teach '):print(u.teach(line[6:]))
    elif line.startswith('ask '):print(u.ask(line[4:]))
    elif line.startswith('forget '):
     aa,bb=parse_question(line[7:]);print(u.forget(aa,bb))
    elif line=='gauge':u.gauge();print('Gauge shifted, cycle values preserved.')
    elif line=='save':u.save();print('Snapshot saved.')
    elif line=='list':
     for item in u.facts:print(item['statement'])
    else:print(u.chat(line))
   except Exception as ex:print(type(ex).__name__,str(ex))
 u.close()
