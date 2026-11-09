"""Round 034: learned open-set retrieval router & fair deterministic controls.
Previously held out lemma identities from WordNet. Training reroutes retrieval/copy,
not a claim of fluent generation or NaN-specific advantage.
"""
import json, ctypes, hashlib, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from episodic import data, ep, noisy_query, KeyEncoder, LENGTH, K, T

# Use experiment corpus via imported episodic module; cwd-independent but external
# corpus needs --corpus or NAN_CORPUS_JSONL environment variable at first import.
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parent

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,24),nn.Tanh(),nn.Linear(24,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).flatten()

def char_distance(a,b):
    a=[int(z) for z in a if z];b=[int(z) for z in b if z]
    # bounded Levenshtein distance, no large allocations
    prev=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        cur=[i]+[0]*len(b)
        for j,d in enumerate(b,1):cur[j]=min(cur[j-1]+1,prev[j]+1,prev[j-1]+int(c!=d))
        prev=cur
    return prev[-1]/max(len(a),len(b),1)

@torch.no_grad()
def example(encoder,pool,rng,positive=True):
    names,values,target=ep(pool,rng,b=1)
    query=noisy_query(names,target,rng)
    if not positive:
        # The absent query comes from an independent member not in this episode.
        present={tuple(n.tolist()) for n in names[0]}
        for _ in range(20):
            n=pool[int(rng.integers(len(pool)))][0]
            arr=np.zeros(names.shape[-1],np.int64)
            arr[:min(len(n),len(arr))]=[x+1 for x in n.encode()[:len(arr)]]
            if tuple(arr.tolist()) not in present:
                query=torch.tensor(arr[None,:]);break
        else:raise RuntimeError('Failed to construct absent query')
        # Corrupt absent key as well, so corruption is not a presence cue.
        for j in rng.choice(np.where(query[0].numpy()!=0)[0],size=2,replace=False):query[0,int(j)]=int(rng.integers(98,124))
    keys=encoder(names)[0];q=encoder(query[:,None,:])[0,0]
    score=(keys@q).cpu().numpy();top=np.argsort(score)[::-1];pick=int(top[0]);margin=float(score[top[0]]-score[top[1]])
    ds=np.array([char_distance(query[0].tolist(),names[0,j].tolist()) for j in range(K)])
    ranked=np.sort(ds)
    feats=np.array([float(score[top[0]]),margin,float(ds[pick]),float(ranked[0]),float(ranked[1]-ranked[0])],np.float32)
    success=bool(positive and pick==int(target[0]))
    # Return all information needed to do generation and controls.
    return feats, int(positive), int(success), pick, int(target[0]),names[0].numpy(),values[0].numpy(),query[0].numpy()

def collect(encoder,pool,seed,n):
    rng=np.random.default_rng(seed);samples=[]
    for i in range(n):samples.append(example(encoder,pool,rng,positive=(i%2==0)))
    return samples

def f1(pred,y):
    tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
    return 2*tp/max(2*tp+fp+fn,1)

def score_router(probs,samples,threshold):
    y=np.array([x[1] for x in samples]);p=(probs>=threshold).astype(int)
    selected=np.array([x[2] for x in samples]);
    return dict(accept_accuracy=float((p==y).mean()),positive_accept=float(p[y==1].mean()),negative_reject=float((1-p[y==0]).mean()),exact_grounded_completion=float((p*selected).sum()/len(y)),false_answer_rate=float((p*(1-y)).sum()/len(y)),f1=f1(p,y),n=len(y))

def thresholds(probs,samples):
    y=np.array([x[1] for x in samples]);best=None
    for t in np.linspace(0,1,201):
        pred=(probs>=t).astype(int)
        # reward exact copy, penalize unsupported answers
        utility=float((pred*np.array([s[2] for s in samples])).mean()-2*(pred*(1-y)).mean())
        if best is None or utility>best[0]:best=(utility,float(t))
    return best[1]

def native_test(encoder,router,pool,threshold,seed=819,n=100):
    lib=ctypes.CDLL(str(OUT/'libepisodic.so'));lib.memory_new.restype=ctypes.c_void_p
    lib.memory_store.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    lib.memory_read.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_read.restype=ctypes.c_int
    lib.memory_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_raw.restype=ctypes.c_int
    lib.memory_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
    lib.memory_free.argtypes=[ctypes.c_void_p]
    samples=collect(encoder,pool,seed,n)
    with torch.no_grad():prob=torch.sigmoid(router(torch.tensor(np.stack([s[0] for s in samples])))).numpy()
    result=dict(episodes=n,accepted=0,positive=0,native_exact=0,plain_index_exact=0,raw_exact=0,false_accept=0,root_bytes=8388608)
    for i,(s,p) in enumerate(zip(samples,prob)):
        feats,positive,successful,pick,target,names,vals,query=s
        if not positive:continue
        result['positive']+=1
        if p<threshold:continue
        result['accepted']+=1
        ptr=lib.memory_new();assert ptr
        try:
            for j in range(K):
                for t in range(LENGTH):lib.memory_store(ptr,j*LENGTH+t,int(vals[j,t]))
            lib.memory_gauge(ptr,seed+i,8.)
            reconstructed=[lib.memory_read(ptr,pick*LENGTH+t) for t in range(LENGTH)]
            raw=[lib.memory_raw(ptr,pick*LENGTH+t) for t in range(LENGTH)]
            expected=vals[target].tolist()
            result['native_exact']+=reconstructed==expected
            result['plain_index_exact']+=vals[pick].tolist()==expected
            result['raw_exact']+=raw==expected
        finally:lib.memory_free(ptr)
    return result

def main():
    pools=data();all_runs={}
    for seed in [17,29,41]:
        torch.manual_seed(seed)
        enc=KeyEncoder();enc.load_state_dict(torch.load(OUT/f'episodic_{seed}.pt',weights_only=True,map_location='cpu')['state']);enc.eval()
        train=collect(enc,pools['train'],seed+1,1200);val=collect(enc,pools['val'],seed+2,400);test=collect(enc,pools['test'],seed+3,600)
        model=Router();opt=torch.optim.AdamW(model.parameters(),lr=.008)
        X=torch.tensor(np.stack([s[0] for s in train]));Y=torch.tensor([s[1] for s in train],dtype=torch.float32)
        XV=torch.tensor(np.stack([s[0] for s in val]));XT=torch.tensor(np.stack([s[0] for s in test]))
        best=-1e9;best_state=None
        for step in range(160):
            ids=torch.randperm(len(X))[:128]
            loss=F.binary_cross_entropy_with_logits(model(X[ids]),Y[ids]);opt.zero_grad();loss.backward();opt.step()
            if (step+1)%20==0:
                with torch.no_grad():pv=torch.sigmoid(model(XV)).numpy()
                thr=thresholds(pv,val)
                met=score_router(pv,val,thr)
                utility=met['exact_grounded_completion']-2*met['false_answer_rate']
                if utility>best:best=utility;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);model.eval()
        with torch.no_grad():pv=torch.sigmoid(model(XV)).numpy();pt=torch.sigmoid(model(XT)).numpy()
        thr=thresholds(pv,val)
        # Strong string edit-distance control: threshold tuned on validation.
        ed_val=np.array([1-s[0][3] for s in val]);ed_test=np.array([1-s[0][3] for s in test]);et=thresholds(ed_val,val)
        # Learned retrieval-only control: max cosine confidence with tuned threshold.
        co_val=np.array([s[0][0] for s in val]);co_test=np.array([s[0][0] for s in test]);ct=thresholds(co_val,val)
        report={'learned_router':score_router(pt,test,thr),'edit_distance':score_router(ed_test,test,et),'cosine_confidence':score_router(co_test,test,ct),'thresholds':{'learned':thr,'edit':et,'cosine':ct},'forced_copy':{'exact_grounded_completion':sum(s[2] for s in test)/len(test),'false_answer_rate':.5},'native':native_test(enc,model,pools['test'],thr,seed=seed+700,n=60)}
        torch.save({'state':model.state_dict(),'seed':seed,'threshold':thr},OUT/f'router_{seed}.pt')
        all_runs[str(seed)]=report
        print('seed',seed,report['learned_router'],report['edit_distance'],flush=True)
    (OUT/'results.json').write_text(json.dumps(all_runs,indent=2));return all_runs

if __name__=='__main__':main()
