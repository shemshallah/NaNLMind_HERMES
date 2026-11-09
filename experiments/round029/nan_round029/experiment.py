"""Round 029: gauge-invariant learned gloss classification in K256 mirror-pair geometry.
Train-only vectorizer, held-out synsets, root-backed audit and matched decoder controls.
Requires sklearn, numpy, torch. No quantum-performance claims.
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, log_loss
import torch
from torch import nn
BASE=Path(__file__).resolve().parent
SRC=Path('/mnt/data/nan_round025/corpus.jsonl')
torch.set_num_threads(2)

def wordnet(seed=0, n_per_class=480):
    rng=random.Random(seed); groups={}
    with open(SRC) as f:
      for line in f:
        a=json.loads(line)
        if a['pos']=='noun' and a.get('gloss'):
          groups.setdefault(int(a['lexfile']),[]).append((a['id'],a['gloss']))
    # use well supported classes, sufficiently many test/train examples
    cats=sorted(k for k,v in groups.items() if len(v)>=n_per_class)
    cats=cats[:16]
    items=[]
    for k in cats:
      v=groups[k][:];rng.shuffle(v)
      for i,g in v[:n_per_class]:items.append((i,g,cats.index(k)))
    rng.shuffle(items)
    # per-class split to retain label balance; no duplicate synsets
    train=[];val=[];test=[]
    for c in range(len(cats)):
      a=[v for v in items if v[2]==c]
      train.extend(a[:int(.7*len(a))]);val.extend(a[int(.7*len(a)):int(.85*len(a))]);test.extend(a[int(.85*len(a)):])
    return cats,train,val,test

def embeddings(seed):
    cats,train,val,test=wordnet(seed)
    tf=TfidfVectorizer(max_features=8000,ngram_range=(1,2),min_df=2,stop_words='english',sublinear_tf=True)
    tf.fit([x[1] for x in train])
    svd=TruncatedSVD(n_components=48,n_iter=5,random_state=seed)
    tr=svd.fit_transform(tf.transform([x[1] for x in train])).astype('float32')
    va=svd.transform(tf.transform([x[1] for x in val])).astype('float32')
    te=svd.transform(tf.transform([x[1] for x in test])).astype('float32')
    # Train-only scale; preserve numerically meaningful amplitude
    scale=max(np.std(tr),1e-5)
    return cats, [(tr/scale,np.array([x[2] for x in train])),(va/scale,np.array([x[2] for x in val])),(te/scale,np.array([x[2] for x in test]))],{'train':len(train),'val':len(val),'test':len(test),'feature_scale':float(scale),'train_synset_ids':[x[0] for x in train[:4]],'test_synset_ids':[x[0] for x in test[:4]]}

# Disjoint anchored triangles (0,a,b) in the true K256 addressing.
D=48
A=np.arange(1,2*D,2);B=A+1

def encode(z, noise, rng):
    # for each semantic vector, 3 oriented edge measurements: (0,a), (a,b), (0,b)
    n=z.shape[0];g=rng.normal(0,noise,size=(n,256)).astype('float32')
    oa=g[:,[0]]-g[:,A];ob=g[:,[0]]-g[:,B]
    ab=z+g[:,A]-g[:,B]
    return np.concatenate([oa,ab,ob],axis=1)

def decode(x,kind='cycle'):
    oa,ab,ob=np.split(x,3,axis=1)
    if kind=='cycle':return oa+ab-ob
    if kind=='raw':return ab
    if kind=='wrong_triangle':return oa+ab-ob[:,np.roll(np.arange(D),1)]
    if kind=='correct_conventional':return oa+ab-ob
    raise ValueError(kind)

def learn(xtr,ytr,xval,yval,xt,yt,seed,epochs=130):
    torch.manual_seed(seed)
    # identical 48-input dimensional MLP, identical parameter counts for all readouts
    model=nn.Sequential(nn.Linear(48,64),nn.GELU(),nn.Linear(64,int(max(ytr)+1)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.008,weight_decay=.02)
    tr=torch.tensor(xtr,dtype=torch.float32);val=torch.tensor(xval,dtype=torch.float32);test=torch.tensor(xt,dtype=torch.float32)
    y=torch.tensor(ytr,dtype=torch.long);yv=torch.tensor(yval,dtype=torch.long);yt_=torch.tensor(yt,dtype=torch.long)
    best=1e10;state=None;best_step=None
    for epoch in range(epochs):
      model.train();optimizer.zero_grad();logits=model(tr);loss=nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step()
      model.eval()
      with torch.no_grad():v=float(nn.functional.cross_entropy(model(val),yv))
      if v<best:best=v;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_step=epoch+1
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
      out=model(test);p=out.softmax(-1).numpy()
    acc=float(accuracy_score(yt,p.argmax(1)));ce=float(log_loss(yt,p,labels=np.arange(max(ytr)+1)))
    return {'accuracy':acc,'ce':ce,'best_step':best_step,'parameters':sum(p.numel() for p in model.parameters())}

def run(seed):
  cats,parts,metadata=embeddings(seed)
  (tr,yt),(va,yv),(te,ye)=parts
  # OOD random vertex gauge: train noise tiny, test gauge huge, geometry controls cancel exactly.
  rng=np.random.default_rng(seed+100)
  train=encode(tr,.02,rng);val=encode(va,.02,rng);test=encode(te,3.0,rng)
  modes=['cycle','raw','wrong_triangle','correct_conventional']
  results={}
  for mode in modes:
    t0=time.monotonic();r=learn(decode(train,mode),yt,decode(val,mode),yv,decode(test,mode),ye,seed)
    r['train_seconds']=round(time.monotonic()-t0,2)
    results[mode]=r
  # clean test: raw address should work when there is no gauge nuisance
  rng=np.random.default_rng(seed+900)
  clean_train=encode(tr,0.,rng);clean_val=encode(va,0.,rng);clean_test=encode(te,0.,rng)
  results['raw_clean']=learn(decode(clean_train,'raw'),yt,decode(clean_val,'raw'),yv,decode(clean_test,'raw'),ye,seed)
  # Exact gauge invariance under arbitrary fresh vertex shifts; error in float32.
  z=te[:64];r=np.random.default_rng(seed+300)
  s1=encode(z,0.,r);s2=encode(z,5.,r)
  results['max_gauge_error']=float(np.max(np.abs(decode(s1,'cycle')-decode(s2,'cycle'))))
  results['max_cycle_reconstruction_error']=float(np.max(np.abs(z-decode(s2,'cycle'))))
  results['metadata']=metadata;results['classes']=cats
  return results

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[17,29,41]);args=p.parse_args()
  output={}
  for seed in args.seeds:
    print('seed',seed,flush=True);output[str(seed)]=run(seed)
    print({k:(round(v['accuracy'],4),round(v['ce'],4)) for k,v in output[str(seed)].items() if isinstance(v,dict) and 'accuracy' in v},flush=True)
    (BASE/'results.json').write_text(json.dumps(output,indent=2))
  print('saved',BASE/'results.json')
