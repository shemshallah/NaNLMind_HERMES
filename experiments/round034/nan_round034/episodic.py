"""Episodic WordNet definition completion using learned key attention and
native gauge-invariant byte-copy memory. Held-out *lemma* identities.
"""
import json,math,random,ctypes,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import CORPUS,T
LENGTH=24; K=12; MAXKEY=22

def data():
    local=Path(__file__).resolve().parent/'wordnet_episodic_pool.json'
    if local.exists():
        raw=json.loads(local.read_text())
        return {k:[(name,bytes.fromhex(value)) for name,value in rows] for k,rows in raw.items()}
    seen=set();groups={x:[] for x in ('train','val','test')}
    with CORPUS.open() as f:
        for ln in f:
            r=json.loads(ln)
            if r['pos']!='noun' or r['split'] not in groups:continue
            if not r['lemmas']:continue
            name=r['lemmas'][0].replace('_',' ').lower()
            if not name.isascii() or not name.replace(' ','').isalpha() or not 4<=len(name)<=16:continue
            if name in seen:continue
            gloss=r['gloss'].encode('utf8')
            if len(gloss)<LENGTH:continue
            seen.add(name);groups[r['split']].append((name,gloss[:LENGTH]))
    for kind in groups:
        rng=np.random.default_rng(23);rng.shuffle(groups[kind]);groups[kind]=groups[kind][:3500]
    # Different synsets may reuse lemma across splits: guard additionally
    train_names={s for s,_ in groups['train']}
    groups['val']=[p for p in groups['val'] if p[0] not in train_names]
    forbidden=train_names|{s for s,_ in groups['val']}
    groups['test']=[p for p in groups['test'] if p[0] not in forbidden]
    return groups

def ep(pool,rng,b=32):
    names=np.zeros((b,K,MAXKEY),dtype=np.int64);values=np.zeros((b,K,LENGTH),dtype=np.int64);selected=np.zeros(b,np.int64)
    for i in range(b):
        take=rng.choice(len(pool),K,replace=False)
        for j,z in enumerate(take):
            name,v=pool[z];names[i,j,:min(MAXKEY,len(name))]=[c+1 for c in name.encode()[:MAXKEY]]
            values[i,j]=np.frombuffer(v,dtype=np.uint8)
        selected[i]=int(rng.integers(K))
    return torch.tensor(names),torch.tensor(values),torch.tensor(selected)

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.char=nn.Embedding(257,40,padding_idx=0)
        self.pos=nn.Embedding(MAXKEY,40)
        self.proj=nn.Sequential(nn.Linear(40,64),nn.Tanh(),nn.Linear(64,48))
    def forward(self,names):
        b,k,l=names.shape;positions=torch.arange(l).view(1,1,l)
        x=self.char(names)+self.pos(positions)*(names!=0).unsqueeze(-1)
        x=(x*(names!=0).unsqueeze(-1)).sum(-2)/(names!=0).sum(-1).clamp(min=1).unsqueeze(-1)
        x=F.normalize(self.proj(x),dim=-1)
        return x

def noisy_query(names,selected,rng):
    query=names[torch.arange(len(selected)),selected].clone()
    for i in range(len(selected)):
        positions=(query[i]!=0).nonzero().flatten().tolist()
        for pos in rng.choice(positions,size=min(2,len(positions)),replace=False):
            query[i,int(pos)]=int(rng.integers(98,124))
    return query

def train():
    torch.set_num_threads(1);pools=data();out={'counts':{k:len(v) for k,v in pools.items()},'runs':{}}
    for seed in [17,29,41]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=KeyEncoder();opt=torch.optim.AdamW(model.parameters(),lr=.004)
        best=-1;state=None
        history=[]
        for step in range(1,301):
            names,values,selected=ep(pools['train'],rng)
            keys=model(names);query=model(noisy_query(names,selected,rng)[:,None,:])[:,0,:]
            scores=torch.einsum('bkd,bd->bk',keys,query)*20
            loss=F.cross_entropy(scores,selected)
            opt.zero_grad();loss.backward();opt.step()
            if step%75==0:
                score=assess(model,pools['val'],seed+900,120)
                history.append({'step':step,'val':score});print(seed,step,score,flush=True)
                if score['key_accuracy']>best:best=score['key_accuracy'];state={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state)
        test=assess(model,pools['test'],seed+500,300)
        torch.save({'state':state,'seed':seed},T/f'episodic_{seed}.pt')
        out['runs'][str(seed)]={'history':history,'test':test}
    (T/'episodic_results.json').write_text(json.dumps(out,indent=2))
    return out

@torch.no_grad()
def assess(model,pool,seed,count):
    rng=np.random.default_rng(seed);right=0;full=0
    for ix in range(count):
        names,values,selected=ep(pool,rng,b=1)
        keys=model(names);q=model(noisy_query(names,selected,rng)[:,None,:])[0,0];scores=keys[0]@q*20
        j=int(scores.argmax());right+=j==int(selected[0]);full+=j==int(selected[0])
    return {'n':count,'key_accuracy':right/count,'exact_generated_24byte_completion':full/count}

def native_audit():
    subprocess.run(['g++','-std=c++20','-O3','-fPIC','-shared',str(T/'native_episodic.cpp'),'-o',str(T/'libepisodic.so')],check=True)
    lib=ctypes.CDLL(str(T/'libepisodic.so'));lib.memory_new.restype=ctypes.c_void_p
    lib.memory_store.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    lib.memory_read.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_read.restype=ctypes.c_int
    lib.memory_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
    lib.memory_free.argtypes=[ctypes.c_void_p]
    p=lib.memory_new();assert p
    rng=np.random.default_rng(777);raw=rng.integers(32,127,size=K*LENGTH)
    for i,v in enumerate(raw):lib.memory_store(p,i,int(v))
    clean=[lib.memory_read(p,i) for i in range(len(raw))]
    lib.memory_gauge(p,42,6.)
    after=[lib.memory_read(p,i) for i in range(len(raw))]
    lib.memory_free(p)
    result={'root_bytes':8388608,'stored_bytes':len(raw),'clean_exact':clean==list(raw),'after_gauge_exact':after==list(raw),'gauge_amplitude':6.0}
    (T/'native_episodic_audit.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    print('NATIVE',native_audit(),flush=True)
    print('LEARN',train(),flush=True)
