"""Round 033: genuine held-out WordNet byte generation plus episodic geometry.
No quantum claim. Supports reproducible local execution.
"""
import os, json, math, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
T=Path(__file__).resolve().parent
CORPUS=Path(os.environ.get('NAN_CORPUS_JSONL','/mnt/data/nan_round025/corpus.jsonl'))

def load_corpus(n=1800):
    rng=np.random.default_rng(123)
    pools={k:[] for k in ('train','val','test')}
    with CORPUS.open() as f:
        for line in f:
            x=json.loads(line)
            if x.get('pos')!='noun' or x.get('split') not in pools:continue
            s=x['gloss'].encode('utf8');
            if len(s)>=32 and len(s)<=256:pools[x['split']].append(s)
    for k in pools:
        ix=rng.permutation(len(pools[k]))[:n]
        pools[k]=[pools[k][i] for i in ix]
    return pools

class Model(nn.Module):
    def __init__(self,mode='plain',width=48,m=24):
        super().__init__();self.mode=mode;self.m=m
        self.emb=nn.Embedding(256,width)
        self.gru=nn.GRU(width,width,batch_first=True)
        self.write=nn.Linear(width,m)
        self.read=nn.Linear(m,width,bias=False)
        self.gate=nn.Linear(width,width)
        self.head=nn.Linear(width,256)
        self.memory=nn.Parameter(torch.zeros(m))
    def forward(self,x,shuffle=False,gauge_sigma=0):
        emb=self.emb(x);h,_=self.gru(emb)
        if self.mode=='plain':
            # same m trainable slots, no geometric cycle reference
            mem=self.memory.expand(x.shape[0],self.m)
        else:
            # each m-channel is the oriented circulation of triangle (0, 2d+1, 2d+2)
            # Three edges express the state with a shared gauge at each vertex.
            base=self.memory
            g=torch.randn(x.shape[0],2*self.m+1,device=x.device)*gauge_sigma
            e0a= g[:,0,None]-g[:,1::2]
            eab=base[None,:]+g[:,1::2]-g[:,2::2]
            e0b= g[:,0,None]-g[:,2::2]
            if shuffle:e0b=torch.roll(e0b,1,dims=-1)
            mem=e0a+eab-e0b
        # query-dependent read/write: a geometry-derived vector shifts the GRU prediction.
        query=torch.tanh(self.write(h));mread=query*mem[:,None,:]
        out=h+torch.sigmoid(self.gate(h))*torch.tanh(self.read(mread))
        return self.head(out)

def make_batch(data,rng,batch=12,context=72):
    x=np.zeros((batch,context),np.int64);y=x.copy()
    for j in range(batch):
        d=data[int(rng.integers(len(data)))];start=int(rng.integers(max(1,len(d)-context-1)))
        seg=d[start:start+context+1]
        if len(seg)<context+1:seg=seg+bytes([32])*(context+1-len(seg))
        z=np.frombuffer(seg,dtype=np.uint8)
        x[j],y[j]=z[:-1],z[1:]
    return torch.from_numpy(x),torch.from_numpy(y)
@torch.no_grad()
def evaluate(model,data,gauge=0,shuffle=False,max_docs=160):
    model.eval();loss=0;n=0;hits=0
    for d in data[:max_docs]:
        s=np.frombuffer(d[:129],dtype=np.uint8).astype('int64')
        if len(s)<2:continue
        x=torch.from_numpy(s[:-1]).unsqueeze(0);y=torch.from_numpy(s[1:])
        logits=model(x,shuffle=shuffle,gauge_sigma=gauge)[0]
        loss+=float(F.cross_entropy(logits,y,reduction='sum'))
        hits+=int((logits.argmax(-1)==y).sum());n+=len(y)
    return {'bytes':n,'bpb':loss/n/math.log(2),'accuracy':hits/n}
@torch.no_grad()
def generate(model,prompt='a living creature',n=120,temperature=.8,seed=99):
    model.eval();torch.manual_seed(seed)
    seq=list(prompt.encode('utf-8'))
    for i in range(n):
        x=torch.tensor(seq[-100:],dtype=torch.long)[None,:]
        logits=model(x)[0,-1]/temperature
        p=torch.softmax(logits,dim=0)
        nxt=int(torch.multinomial(p,1));seq.append(nxt)
    return bytes(seq).decode('utf-8',errors='replace')

def train():
    torch.set_num_threads(1)
    data=load_corpus();result={'corpus_counts':{k:len(v) for k,v in data.items()},'runs':{}}
    for seed in [17,29]:
        for mode in ['plain','cycle']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=Model(mode);opt=torch.optim.AdamW(model.parameters(),lr=.002)
            best=1e5;beststate=None;beststep=0;history=[]
            for step in range(1,181):
                model.train();x,y=make_batch(data['train'],rng)
                opt.zero_grad();z=model(x,gauge_sigma=1.2 if mode=='cycle' else 0)
                loss=F.cross_entropy(z.reshape(-1,256),y.reshape(-1));loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                if step%45==0 or step==1:
                    ev=evaluate(model,data['val'],max_docs=90)
                    history.append({'step':step,**ev})
                    print(seed,mode,step,round(ev['bpb'],4),flush=True)
                    if ev['bpb']<best:
                        best=ev['bpb'];beststep=step;beststate={k:v.cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(beststate)
            test=evaluate(model,data['test'],max_docs=160)
            perturb=evaluate(model,data['test'],gauge=5,max_docs=60)
            scrambled=evaluate(model,data['test'],gauge=5,shuffle=True,max_docs=60) if mode=='cycle' else None
            disabled=model.memory.detach().clone();model.memory.data.zero_()
            ablated=evaluate(model,data['test'],max_docs=60)
            model.memory.data.copy_(disabled)
            sample=generate(model,n=100)
            result['runs'][f'{mode}_{seed}']={'best_step':beststep,'history':history,'test':test,'gauge':perturb,'wrong_cycle':scrambled,'zero_memory':ablated,'sample':sample,'parameters':sum(x.numel() for x in model.parameters())}
            torch.save({'state':model.state_dict(),'mode':mode,'seed':seed},T/f'{mode}_{seed}.pt')
    (T/'lm_results.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':train()
