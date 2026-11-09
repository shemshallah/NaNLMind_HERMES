#!/usr/bin/env python3
"""Controlled autoregressive WordNet generator: no short-gloss repetition.
Matched seeds and training windows; native cycle weights vs conventional dense 16x16 weights.
"""
import argparse,copy,json,math,time,os,sys
from pathlib import Path
import numpy as np,torch,sentencepiece as spm
from torch import nn
from torch.nn import functional as F
from train_generate import Native
from train_subword import LM,load_docs,ev,gen
P=Path(__file__).resolve().parent

def stream_from_docs(d,tok):
    # Document-separated stream, does not tile/repeat individual glosses to fill windows.
    chunks=[]
    for gloss in d:
        chunks.extend([tok.bos_id(),*tok.encode(gloss,out_type=int),tok.eos_id()])
    return np.asarray(chunks,dtype=np.int64)

def batch(stream,starts,bs,ctx):
    st=starts.integers(0,len(stream)-ctx-2,size=bs)
    inds=st[:,None]+np.arange(ctx+1)[None,:]
    windows=stream[inds]
    return torch.from_numpy(windows[:,:-1].copy()),torch.from_numpy(windows[:,1:].copy())

@torch.no_grad()
def eval_full(m,docs,w,cap=600):
    m.eval();ll=0;hits=0;n=0
    for arr in docs[:cap]:
        x=torch.tensor(arr[:-1],dtype=torch.long)[None,:];y=torch.tensor(arr[1:],dtype=torch.long)
        logits,_=m(x,w);ll+=F.cross_entropy(logits[0],y,reduction='sum').item()
        hits+=(logits[0].argmax(-1)==y).sum().item();n+=len(y)
    return {'tokens':n,'nats_per_token':ll/n,'perplexity':math.exp(ll/n),'next_token_acc':hits/n}

def run(args):
    torch.set_num_threads(2)
    d=load_docs(args.corpus,25000,1500,1500)
    tok=spm.SentencePieceProcessor(model_file=str(P/'wordnet_bpe.model'))
    docids={k:[np.array([tok.bos_id(),*tok.encode(s,out_type=int),tok.eos_id()],dtype=np.int64) for s in v] for k,v in d.items()}
    stream=stream_from_docs(d['train'],tok)
    report={'setup':vars(args),'train_stream_tokens':len(stream),'documents':{k:len(v) for k,v in d.items()},'variants':{}}
    for seed in args.seeds:
      for variant in ('plain','native'):
        rng=np.random.default_rng(seed);torch.manual_seed(seed)
        m=LM(tok.vocab_size(),args.hidden)
        root=Native() if variant=='native' else None
        opt=torch.optim.AdamW(m.nonpair() if root else m.parameters(),lr=.002)
        best=1e10;best_state=None;beststep=0;history=[];t0=time.time()
        for step in range(1,args.steps+1):
            x,y=batch(stream,rng,args.batch,args.context)
            m.train();opt.zero_grad()
            w=root.get().detach().clone().requires_grad_(True) if root else m.pair
            logits,_=m(x,w);loss=F.cross_entropy(logits.reshape(-1,tok.vocab_size()),y.flatten());loss.backward()
            if root:root.apply(w.grad,.02)
            nn.utils.clip_grad_norm_(m.nonpair(),1.);opt.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                w=root.get() if root else m.pair.detach()
                v=eval_full(m,docids['val'],w,100)
                history.append({'step':step,**v})
                print(seed,variant,step,'val ppl',round(v['perplexity'],2),flush=True)
                if v['nats_per_token']<best:
                    best=v['nats_per_token'];beststep=step;best_state=copy.deepcopy(m.state_dict())
                    if root:root.save(P/f'best_root_{seed}.bin')
        m.load_state_dict(best_state)
        if root:root.load(P/f'best_root_{seed}.bin')
        w=root.get() if root else m.pair.detach()
        test=eval_full(m,docids['test'],w,args.testdocs)
        prompts=['a person who','a place where','a small animal','an instrument used for','a person who studies']
        samples={s:gen(m,tok,w,s,steps=100,seed=seed+19) for s in prompts}
        ablate=eval_full(m,docids['test'],torch.zeros_like(w),min(150,args.testdocs))
        original=eval_full(m,docids['test'],w,min(150,args.testdocs))
        key=f'{variant}_{seed}'
        row={'test':test,'test_150':original,'zero_memory_150':ablate,'best_step':beststep,'history':history,'samples':samples,'parameters':sum(z.numel() for z in m.parameters()),'duration_sec':time.time()-t0}
        if root:
            before=w.clone();root.gauge(seed+987,2.0);after=root.get()
            row['gauge_max_cycle_delta']=float((before-after).abs().max())
            row['gauge_test_150']=eval_full(m,docids['test'],after,min(150,args.testdocs))
            row['native_audit']=root.audit()
            root.save(P/f'gauged_root_{seed}.bin');root.close()
        torch.save({'model':m.state_dict(),'seed':seed,'variant':variant,'hidden':args.hidden,'vocab':tok.vocab_size(),'best_step':beststep},P/f'{key}.pt')
        report['variants'][key]=row
        (P/'results.json').write_text(json.dumps(report,indent=2))
        print('FINAL',key,json.dumps(test),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--corpus',default=str(P/'corpus.jsonl'));p.add_argument('--seeds',nargs='+',type=int,default=[17,29]);p.add_argument('--steps',type=int,default=850);p.add_argument('--hidden',type=int,default=96);p.add_argument('--batch',type=int,default=12);p.add_argument('--context',type=int,default=48);p.add_argument('--eval-every',type=int,default=170);p.add_argument('--testdocs',type=int,default=500);run(p.parse_args())
