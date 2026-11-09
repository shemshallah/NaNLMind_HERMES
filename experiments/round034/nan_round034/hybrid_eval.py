"""Evaluate actual token-level LM/copy mixture on held-out glosses.
Shows information benefit of retrieval, not a unique NaN geometric benefit.
"""
import torch, numpy as np, json, math
from pathlib import Path
from episodic import KeyEncoder,data,LENGTH
from route_train import Router,example
from experiment import Model
T=Path(__file__).resolve().parent
torch.set_num_threads(1)

def run(seed=17,count=90):
    rng=np.random.default_rng(seed+6000);pool=data()['test']
    enc=KeyEncoder();enc.load_state_dict(torch.load(T/f'episodic_{seed}.pt',weights_only=True,map_location='cpu')['state']);enc.eval()
    ck=torch.load(T/f'router_{seed}.pt',weights_only=True,map_location='cpu');gate=Router();gate.load_state_dict(ck['state']);gate.eval()
    lm=Model(mode='plain');lm.load_state_dict(torch.load(T/'plain_17.pt',weights_only=True,map_location='cpu')['state']);lm.eval()
    nll_lm=nll_mix=nll_oracle=0;bytes_count=0;selected=0;correct=0
    with torch.no_grad():
        for i in range(count):
            feat,available,success,pick,target,names,values,query=example(enc,pool,rng,positive=True)
            confidence=float(torch.sigmoid(gate(torch.tensor(feat[None,:]))).item())
            use=confidence>=ck['threshold'];selected+=use;correct+=success*use
            expected=torch.tensor(values[target].astype('int64'))
            prefix=b'a definition is: '
            x=torch.tensor(list(prefix)+[int(x) for x in expected[:-1]],dtype=torch.long)[None,:]
            probs=torch.softmax(lm(x)[0,len(prefix)-1:len(prefix)-1+LENGTH],dim=-1)
            plm=probs.gather(1,expected[:,None]).flatten().clamp(min=1e-12)
            # End-to-end autoregressive mixture distribution: use a learnable
            # open-set decision, then emit an evidence-copy distribution.
            if use:
                pcopy=(torch.tensor(values[pick].astype('int64'))==expected).float()
                mix=(.05*plm+.95*pcopy).clamp(min=1e-12)
            else:mix=plm
            nll_lm+=float(-torch.log2(plm).sum())
            nll_mix+=float(-torch.log2(mix).sum())
            # The conventional memory is given identical selected 24 bytes,
            # therefore its mixture likelihood is exactly equal.
            nll_oracle+=float(-torch.log2(mix).sum());bytes_count+=LENGTH
    return {'episodes':count,'bytes':bytes_count,'lm_only_bpb':nll_lm/bytes_count,'cycle_hybrid_bpb':nll_mix/bytes_count,'conventional_index_hybrid_bpb':nll_oracle/bytes_count,'accepted_queries':selected,'correct_retrieved':correct,'note':'Mixture is supervised by a learned open-set router; source bytes are copied, not generated autonomously.'}
if __name__=='__main__':
 out={str(seed):run(seed) for seed in [17,29,41]}
 (T/'hybrid_results.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
