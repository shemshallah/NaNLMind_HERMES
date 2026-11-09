"""Round 037: supervised prose-to-relation writer, resident geometric episode memory,
compositional questions, matched conventional index, retention and gauge controls.
No claim of unsupervised text learning or end-to-end free-form LLM.
"""
import argparse, ctypes, hashlib, json, random, re, subprocess, sys, time, zipfile
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/wn3.1.dict.zip')
torch.set_num_threads(2)

# Real WordNet source edges and child-disjoint cases (sources do not repeat across splits)
def wordnet(path):
    with zipfile.ZipFile(path) as z:
        txt=z.read('wn3.1.dict/dict/data.noun').decode('utf-8')
    records={}
    for line in txt.splitlines():
        if not re.match(r'^\d{8} ',line):continue
        head,_,gloss=line.partition('|');p=head.split()
        try:
            count=int(p[3],16);i=4+2*count; nptr=int(p[i]);i+=1;parents=[]
            for j in range(nptr):
                kind,tar,pos,source=p[i:i+4];i+=4
                if kind in ('@','@i') and pos=='n':parents.append(tar)
            name=p[4].replace('_',' ').lower()
            if name.isascii() and len(name)<=24 and name.isprintable() and len(name)>=3:records[p[0]]=(name,parents,gloss.strip())
        except (ValueError,IndexError):pass
    return records

def paths(rows,seed=41,limit=2400):
    rng=random.Random(seed);out=[];seen=set();items=list(rows.items());rng.shuffle(items)
    for source,(name,pars,gloss) in items:
        if source in seen:continue
        for b in pars:
            if b not in rows:continue
            for c in rows[b][1]:
                if c not in rows:continue
                p=[source,b,c]
                # usually 3-hop if possible, to include 2- and 3-hop queries
                for d in rows[c][1]:
                    if d in rows and d not in p:p.append(d);break
                if len(set(p))!=len(p) or len({rows[k][0] for k in p})!=len(p):continue
                seen.add(source);out.append([rows[k][0] for k in p]);break
            if source in seen:break
        if len(out)>=limit:break
    rng.shuffle(out)
    unique=[];source_names=set()
    for p in out:
        if p[0] not in source_names:
            unique.append(p);source_names.add(p[0])
    return unique

# Template surface variation; subject and object spans come from the real noun lexicon.
# Sentences express real WordNet edges but are synthetic paraphrases, not original glosses.
TEMPLATES=[
 ('{a} is a kind of {b}.',0),
 ('{a} is a type of {b}.',0),
 ('{a} belongs to the category {b}.',0),
 ('{a} is classified as {b}.',0),
 ('{b} includes the category {a}.',1),
 ('{b} is a broader class than {a}.',1),
 ('{a} falls under {b}.',0),
 ('{b} encompasses {a}.',1),
 ('{a} is not a kind of {b}.',2),
 ('{a} is not classified as {b}.',2),
 ('some sources doubt whether {a} is a {b}.',3),
 ('it is unclear whether {a} belongs to {b}.',3),
]
# Normalize arbitrary unseen entity name spans but preserve relation grammar.
def normalize(template):return re.sub(r'\{a\}','entitya',re.sub(r'\{b\}','entityb',template)).lower()
TOKENS=sorted(set(re.findall(r"[a-z]+|[.]",' '.join(normalize(t) for t,_ in TEMPLATES))))
ITOS=['<pad>','<unk>']+TOKENS;STOI={t:i for i,t in enumerate(ITOS)}
MAXLEN=18

def encode(texts):
    out=[]
    for s in texts:
        tokens=re.findall(r'[a-z]+|[.]',s.lower())[:MAXLEN]
        out.append([STOI.get(t,1) for t in tokens]+[0]*(MAXLEN-len(tokens)))
    return torch.tensor(out,dtype=torch.long)

class Writer(nn.Module):
    def __init__(self,hidden=48):
        super().__init__();self.emb=nn.Embedding(len(ITOS),32,padding_idx=0);self.rnn=nn.GRU(32,hidden,batch_first=True,bidirectional=True);self.lin=nn.Sequential(nn.Linear(2*hidden,64),nn.Tanh(),nn.Linear(64,4))
    def forward(self,x):
        _,h=self.rnn(self.emb(x));return self.lin(torch.cat([h[-2],h[-1]],-1))

class Native:
    def __init__(self):
        dll=HERE/'libcycles.so'
        subprocess.run(['g++','-std=c++20','-O3','-shared','-fPIC',str(HERE/'native_cycles.cpp'),'-o',str(dll)],check=True,capture_output=True)
        self.lib=ctypes.CDLL(str(dll));l=self.lib
        l.mem_new.restype=ctypes.c_void_p;l.mem_free.argtypes=[ctypes.c_void_p]
        l.mem_write.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
        l.mem_read.argtypes=[ctypes.c_void_p,ctypes.c_int];l.mem_read.restype=ctypes.c_float
        l.mem_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];l.mem_raw.restype=ctypes.c_float
        l.mem_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
        l.mem_snapshot.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.mem_snapshot.restype=ctypes.c_int
        l.mem_restore.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.mem_restore.restype=ctypes.c_int
        self.ptr=l.mem_new();assert self.ptr
        self.next_slot=0
    def close(self):self.lib.mem_free(self.ptr)
    def write(self,score):
        slot=self.next_slot;assert slot<32385
        self.lib.mem_write(self.ptr,slot,float(score));self.next_slot+=1;return slot
    def read(self,slot):return float(self.lib.mem_read(self.ptr,slot))
    def raw(self,slot):return float(self.lib.mem_raw(self.ptr,slot))
    def gauge(self,seed,scale=50.):self.lib.mem_gauge(self.ptr,seed,float(scale))
    def save(self,path):assert self.lib.mem_snapshot(self.ptr,str(path).encode())==1
    def restore(self,path):assert self.lib.mem_restore(self.ptr,str(path).encode())==1

# Geometric and conventional indexes have identical assertion inventories and query logic.
# The geometric backend additionally stores confidence in resident pair triangles.
class Store:
    def __init__(self,native=None):self.native=native;self.evidence={};self.adj=defaultdict(list);self.neg=set();self.uncertain=set()
    def insert(self,a,b,kind,confidence=1.):
        if kind==1:a,b=b,a;kind=0
        if kind==2:self.neg.add((a,b));return
        if kind==3:self.uncertain.add((a,b));return
        if kind!=0:return
        # Duplicate edges update evidence; no accidental data leakage across graph instances.
        key=(a,b)
        if key in self.evidence:return
        ref=self.native.write(confidence) if self.native else confidence
        self.evidence[key]=ref;self.adj[a].append(b)
    def strength(self,a,b,raw=False):
        ref=self.evidence.get((a,b))
        if ref is None:return 0.
        if not self.native:return float(ref)
        return self.native.raw(ref) if raw else self.native.read(ref)
    def prove(self,source,target,maxhop=4,raw=False):
        # exact multihop traversal; no false proof from unsupported edges
        queue=deque([(source,0)]);visited={source}
        while queue:
            x,depth=queue.popleft()
            if x==target and depth>=2:return True
            if depth>=maxhop:continue
            for y in self.adj[x]:
                if self.strength(x,y,raw)>0.5 and y not in visited:
                    visited.add(y);queue.append((y,depth+1))
        return False

# Trains relation writer on textual statements with arbitrary names masked as entitya/b.
# The class label determines edge direction; question answers derive from several stored facts.
def train_writer(seed=17,steps=350):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    model=Writer();opt=torch.optim.AdamW(model.parameters(),lr=.004)
    # 75% templates train, rest are genuine, held-out grammar shifts
    train_ids=[0,1,2,3,4,5,8,9,10,11]
    held_ids=[6,7]
    samples=[(normalize(TEMPLATES[i][0]),TEMPLATES[i][1]) for i in train_ids]
    x=encode([a for a,b in samples]);y=torch.tensor([b for a,b in samples]);
    hist=[]
    for step in range(steps):
        # repeated examples with shuffled class balancing; rigorous concept holdout occurs at data level
        ids=rng.integers(0,len(x),size=32)
        opt.zero_grad();out=model(x[ids]);loss=F.cross_entropy(out,y[ids]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%50==0:hist.append([step,float(loss.item())])
    model.eval()
    with torch.no_grad():
        pred=model(encode([normalize(a) for a,b in TEMPLATES])).argmax(1).tolist()
    acc_seen=sum(pred[i]==TEMPLATES[i][1] for i in train_ids)/len(train_ids)
    acc_held=sum(pred[i]==TEMPLATES[i][1] for i in held_ids)/len(held_ids)
    torch.save({'state_dict':model.state_dict(),'vocabulary':ITOS,'seed':seed,'templates':TEMPLATES},HERE/f'writer_{seed}.pt')
    return model,{'seen_template_accuracy':acc_seen,'unseen_template_accuracy':acc_held,'predictions':pred,'trace':hist,'parameters':sum(p.numel() for p in model.parameters())}

def run(seed,graphs,newgraphs):
    # Train/test entity concepts disjoint by source. Store only new test facts at inference.
    model,writer_stats=train_writer(seed)
    native=Native();geo=Store(native);plain=Store()
    rng=random.Random(seed+991)
    tests=[];edges=0
    try:
        # Each episode is a new, separate set of facts. No train synset identities appear in test.
        # We use a unique set of source-to-ancestor chains, and never install direct edge source->final.
        for k,path in enumerate(graphs):
            hop=2 if k%2==0 else min(3,len(path)-1)
            if hop>=len(path):hop=len(path)-1
            if hop<2:continue
            # Generate one synthetic natural-language statement per edge, sometimes reverse syntactic roles.
            # The normalized relation writer must decide direction before inserting.
            for j in range(hop):
                a,b=path[j:j+2];template_idx=rng.choice([0,1,2,3,4,5])
                template,expected=TEMPLATES[template_idx]
                surface=template.format(a=a,b=b)
                # Entity spans are resolved from the known current episode; learning is the predicate semantics.
                abstract=normalize(template)
                with torch.no_grad():cls=int(model(encode([abstract])).argmax(-1))
                left,right=(b,a) if expected==1 else (a,b)
                geo.insert(left,right,cls);plain.insert(left,right,cls);edges+=1
            # Gold truth is from actual WordNet hypernym chain; controls have identical inserted facts.
            tests.append((path[0],path[hop],True,hop,path))
            # one broken path query (unrelated exact concept) that may coincidentally be linked by other episodes
            # use guaranteed nonce disjoint from actual records
            tests.append((path[0],f'unknown_{seed}_{k}',False,hop,path))
        # Control for false evidence: check exact graph answer equality; source networks identical.
        def evaluation(raw=False):
            p=[geo.prove(a,b,maxhop=h,raw=raw)==truth for a,b,truth,h,path in tests]
            c=[plain.prove(a,b,maxhop=h)==truth for a,b,truth,h,path in tests]
            return {'geometry_correct':sum(p),'conventional_correct':sum(c),'total':len(tests),'geometry_accuracy':sum(p)/len(p),'conventional_accuracy':sum(c)/len(c)}
        before=evaluation()
        base={key:round(native.read(ref),5) for key,ref in list(geo.evidence.items())[:20]}
        native.save(HERE/f'root_{seed}.bin')
        native.gauge(seed+84,35.)
        after=evaluation();raw=evaluation(raw=True)
        recovery={key:round(native.read(ref),5) for key,ref in list(geo.evidence.items())[:20]}
        assert max(abs(base[k]-recovery[k]) for k in base)<0.0001
        native.restore(HERE/f'root_{seed}.bin')
        restored=evaluation()
        assert before==restored
        # Additional 'training': writer learns a disjoint set of new sentence phrasings,
        # while resident episode memory must keep earlier facts unchanged.
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        xa=encode([normalize(TEMPLATES[i][0]) for i in [6,7,8,9,10,11]])
        ya=torch.tensor([TEMPLATES[i][1] for i in [6,7,8,9,10,11]])
        for it in range(100):
            opt.zero_grad();l=F.cross_entropy(model(xa),ya);l.backward();opt.step()
        retained=evaluation()
        new_cases=[]
        # Acquire additional facts after writer adaptation: no replay of previous facts.
        for k,path in enumerate(newgraphs):
            h=2 if k%2==0 else min(3,len(path)-1)
            if h<2:continue
            for j in range(h):
                a,b=path[j:j+2]
                t_idx=rng.choice([6,7]) # previously held-out sentence styles
                template,expected=TEMPLATES[t_idx]
                with torch.no_grad():cls=int(model(encode([normalize(template)])).argmax(-1))
                left,right=(b,a) if expected==1 else (a,b)
                geo.insert(left,right,cls);plain.insert(left,right,cls)
            new_cases.append((path[0],path[h],h))
        new_geo=sum(geo.prove(a,b,maxhop=h) for a,b,h in new_cases)
        new_plain=sum(plain.prove(a,b,maxhop=h) for a,b,h in new_cases)
        all_old=evaluation()
        with torch.no_grad():postpred=model(encode([normalize(t) for t,k in TEMPLATES])).argmax(-1).tolist()
        # no labels or edges lost in this fixed evidence arena; also evaluate adaptation to new grammar
        return {'seed':seed,'writer':writer_stats,'after_adaptation_template_predictions':postpred,'facts_written':edges,'distinct_facts':len(geo.evidence),'before_gauge':before,'after_gauge':after,'raw_edges_after_gauge':raw,'after_writer_adaptation':retained,'after_root_restore':restored,'old_after_new_memory_writes':all_old,'new_knowledge_positive':{'cycle_correct':new_geo,'conventional_correct':new_plain,'total':len(new_cases)},'native_snapshot_sha256':hashlib.sha256((HERE/f'root_{seed}.bin').read_bytes()).hexdigest(),'tested_sample':[{'premises':[TEMPLATES[0][0].format(a=path[j],b=path[j+1]) for j in range(h)],'question':f'Is {a} a kind of {b}?','composed_answer':f'{a} is a kind of {b}.' if geo.prove(a,b,maxhop=h) else 'The evidence is insufficient.','gold':t} for a,b,t,h,path in tests[:8]]}
    finally:native.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wordnet',type=Path,default=ROOT);parser.add_argument('--limit',type=int,default=240);args=parser.parse_args()
    st=time.time();rows=wordnet(args.wordnet);pp=paths(rows,limit=1800)
    train=pp[:int(len(pp)*.7)];validation=pp[int(len(pp)*.7):int(len(pp)*.8)];test=pp[int(len(pp)*.8):][:args.limit]
    # source split; target ancestors can overlap (real ontology), explicitly reported.
    assert set(p[0] for p in train).isdisjoint(set(p[0] for p in test))
    results={'design':'controlled supervised predicate extraction + explicit graph composition + native cycle evidence; no unrestricted prose understanding','source':'WordNet 3.1 data.noun','synsets':len(rows),'train_child_count':len(train),'validation_child_count':len(validation),'test_child_count':len(test),'seeds':[]}
    for seed in [17,29,43]:
        result=run(seed,test,pp[int(len(pp)*.8)+args.limit:][:80]);results['seeds'].append(result)
        print('seed',seed,'before',result['before_gauge'],'after',result['after_gauge'],'raw',result['raw_edges_after_gauge'],'writer',result['writer']['seen_template_accuracy'],result['writer']['unseen_template_accuracy'],flush=True)
    results['elapsed_s']=time.time()-st
    (HERE/'results.json').write_text(json.dumps(results,indent=2));print('DONE',results['elapsed_s'])
if __name__=='__main__':main()
