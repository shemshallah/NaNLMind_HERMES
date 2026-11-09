"""Out-of-distribution edit noise & no-memory rejection stress-test.
Training episodes used exactly two corrupted query characters. These tests vary that.
"""
import json, numpy as np, torch
from pathlib import Path
from episodic import data,ep,KeyEncoder,K
from route_train import Router,char_distance,score_router
OUT=Path(__file__).resolve().parent
torch.set_num_threads(1)

def make(encoder,pool,rng,positive,changes):
    names,vals,target=ep(pool,rng,b=1)
    if positive:
        query=names[0,int(target[0])].clone()
    else:
        present={tuple(n.tolist()) for n in names[0]}
        for z in range(30):
            s=pool[int(rng.integers(len(pool)))][0]
            a=np.zeros(names.shape[-1],np.int64)
            a[:min(len(s),len(a))]=[c+1 for c in s.encode()[:len(a)]]
            if tuple(a.tolist()) not in present:query=torch.tensor(a);break
        else:raise RuntimeError('absent key failed')
    pos=np.where(query.numpy()!=0)[0]
    for j in rng.choice(pos,size=min(changes,len(pos)),replace=False):
        query[int(j)]=int(rng.integers(98,124))
    with torch.no_grad():
        keys=encoder(names)[0];q=encoder(query[None,None,:])[0,0];scores=(keys@q).numpy()
    top=np.argsort(scores)[::-1];pick=int(top[0]);
    ds=np.array([char_distance(query.tolist(),names[0,j].tolist()) for j in range(K)])
    sorted_ds=np.sort(ds)
    feats=np.array([scores[top[0]],scores[top[0]]-scores[top[1]],ds[pick],sorted_ds[0],sorted_ds[1]-sorted_ds[0]],np.float32)
    return feats,int(positive),int(bool(positive and pick==int(target[0]))),pick

def main():
    pools=data();results={}
    for seed in [17,29,41]:
        enc=KeyEncoder();enc.load_state_dict(torch.load(OUT/f'episodic_{seed}.pt',weights_only=True,map_location='cpu')['state']);enc.eval()
        ck=torch.load(OUT/f'router_{seed}.pt',weights_only=True,map_location='cpu');gate=Router();gate.load_state_dict(ck['state']);gate.eval()
        results[str(seed)]={}
        for noise in [0,1,2,3,4,6]:
            rng=np.random.default_rng(seed+1000+noise)
            samples=[make(enc,pools['test'],rng,i%2==0,noise) for i in range(300)]
            x=torch.tensor(np.stack([z[0] for z in samples]));
            with torch.no_grad():p=torch.sigmoid(gate(x)).numpy()
            y=np.array([s[1] for s in samples]);success=np.array([s[2] for s in samples]);a=(p>=ck['threshold']).astype(int)
            results[str(seed)][str(noise)]={'accept_accuracy':float(np.mean(a==y)),'exact_complete_all':float(np.mean(a*success)),'false_answer_rate':float(np.mean(a*(1-y))),'positive_accept_rate':float(np.mean(a[y==1])),'negative_reject_rate':float(np.mean(1-a[y==0])),'n':len(y)}
        print(seed,results[str(seed)],flush=True)
    (OUT/'stress_results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
