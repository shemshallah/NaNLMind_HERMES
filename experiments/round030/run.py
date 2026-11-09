"""Round 030: WordNet synset-disjoint semantic prediction with learned weights
physically stored in 8MiB mirror pair lanes. Gauge tests are synthetic OOD controls.
"""
import ctypes as C, json, random, time, hashlib, argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
BASE=Path(__file__).resolve().parent
DATA=Path('/mnt/data/nan_round025/corpus.jsonl')
D=24;K=16
f32=np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
i32=np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
class Native:
 def __init__(self):
  self.lib=C.CDLL(str(BASE/'libnative_cycle.so'));l=self.lib
  l.nan_new.argtypes=[C.c_int];l.nan_new.restype=C.c_void_p
  l.nan_free.argtypes=[C.c_void_p];l.nan_get.argtypes=[C.c_void_p,f32]
  l.nan_predict.argtypes=[C.c_void_p,f32,f32];l.nan_raw_predict.argtypes=[C.c_void_p,f32,f32]
  l.nan_epoch.argtypes=[C.c_void_p,f32,i32,C.c_int,C.c_float,C.c_uint];l.nan_epoch.restype=C.c_double
  l.nan_gauge.argtypes=[C.c_void_p,C.c_int,C.c_float]
  l.nan_snapshot.argtypes=[C.c_void_p,C.c_char_p];l.nan_snapshot.restype=C.c_int
  l.nan_restore.argtypes=[C.c_void_p,C.c_char_p];l.nan_restore.restype=C.c_int
  l.nan_audit.argtypes=[C.c_void_p];l.nan_audit.restype=C.c_int
  self.ptr=l.nan_new(0)
  if not self.ptr:raise MemoryError('root allocation')
 def close(self):self.lib.nan_free(self.ptr)
 def weights(self):
  out=np.empty((K,D),np.float32);self.lib.nan_get(self.ptr,out);return out
 def train(self,x,y,epochs=12,lr=.045,seed=0):
  return [self.lib.nan_epoch(self.ptr,x,y,len(y),lr,seed+e) for e in range(epochs)]
 def shift(self,seed,sigma):self.lib.nan_gauge(self.ptr,seed,sigma)
 def snapshot(self,path):return self.lib.nan_snapshot(self.ptr,str(path).encode())
 def restore(self,path):return self.lib.nan_restore(self.ptr,str(path).encode())
 def audit(self):return bool(self.lib.nan_audit(self.ptr))
 def predict(self,x,raw=False):
  out=np.empty((len(x),K),np.float32)
  method=self.lib.nan_raw_predict if raw else self.lib.nan_predict
  for a,b in zip(x,out):method(self.ptr,a,b)
  return out

def corpus(seed,per_class=220):
 rng=random.Random(seed);groups={}
 with DATA.open() as f:
  for line in f:
   a=json.loads(line)
   if a.get('pos')=='noun' and a.get('gloss'):
    groups.setdefault(int(a['lexfile']),[]).append((a['id'],a['gloss']))
 cats=sorted(k for k,v in groups.items() if len(v)>=per_class)[:K]
 if len(cats)!=K:raise RuntimeError('insufficient WordNet categories')
 train=[];valid=[];test=[]
 for ci,c in enumerate(cats):
  row=groups[c][:];rng.shuffle(row);row=row[:per_class]
  for p,segment in [(0,train),(1,valid),(2,test)]:
   lo,hi=[(0,150),(150,185),(185,220)][p]
   segment.extend([(uid,gloss,ci) for uid,gloss in row[lo:hi]])
 for s in (train,valid,test):rng.shuffle(s)
 tf=TfidfVectorizer(max_features=5500,ngram_range=(1,2),min_df=2,sublinear_tf=True,stop_words='english')
 tr_txt=[r[1] for r in train];tf.fit(tr_txt)
 svd=TruncatedSVD(n_components=D,n_iter=4,random_state=seed)
 xtr=svd.fit_transform(tf.transform(tr_txt)).astype(np.float32)
 xval=svd.transform(tf.transform([r[1] for r in valid])).astype(np.float32)
 xte=svd.transform(tf.transform([r[1] for r in test])).astype(np.float32)
 scale=float(np.std(xtr))
 xs=[np.ascontiguousarray(x/scale) for x in [xtr,xval,xte]]
 ys=[np.array([r[2] for r in s],np.int32) for s in (train,valid,test)]
 return xs,ys,{'categories':cats,'train':len(train),'val':len(valid),'test':len(test),'train_ids_sha256':hashlib.sha256('\n'.join(r[0] for r in train).encode()).hexdigest(),'test_ids_sha256':hashlib.sha256('\n'.join(r[0] for r in test).encode()).hexdigest()}

def evaluate(y,logits):
 p=np.exp((logits-logits.max(1,keepdims=True)).astype(np.float64));p/=p.sum(1,keepdims=True)
 return {'accuracy':float(accuracy_score(y,p.argmax(1))),'cross_entropy':float(log_loss(y,p,labels=list(range(K))))}

def run(seed):
 xs,ys,meta=corpus(seed);xtr,xv,xt=xs;ytr,yv,yt=ys
 native=Native();start=time.perf_counter()
 losses=native.train(xtr,ytr,epochs=18,lr=.02,seed=seed)
 train_sec=time.perf_counter()-start
 cycle_before=native.predict(xt);raw_before=native.predict(xt,raw=True)
 wbefore=native.weights().copy()
 path=BASE/f'root_{seed}.bin';assert native.snapshot(path)
 native.shift(seed+1000,3.0)
 cycle_after=native.predict(xt);raw_after=native.predict(xt,raw=True)
 max_drift=float(np.max(np.abs(wbefore-native.weights())))
 audit=native.audit();assert audit
 assert native.restore(path)
 restore_err=float(np.max(np.abs(native.predict(xt)-cycle_before)))
 native.close()
 # A truly capacity-matched conventional linear model fitted on the same 24D input.
 # Identical 16*24 semantic weights, plus optional learned intercept not used to match native.
 conventional=LogisticRegression(fit_intercept=False,C=1.0,max_iter=200,random_state=seed)
 conventional.fit(xtr,ytr)
 logits_plain=conventional.decision_function(xt)
 # Ablation: standard real-valued equation using the same reconstructed 24D weight matrix.
 reconstructed=xt@wbefore.T
 assert np.max(abs(cycle_before-reconstructed))<.002
 result={'seed':seed,'samples':meta,'native_training_seconds':train_sec,'train_loss_start':losses[0],'train_loss_end':losses[-1],
  'cycle_before':evaluate(yt,cycle_before),'cycle_after_large_gauge':evaluate(yt,cycle_after),
  'raw_before':evaluate(yt,raw_before),'raw_after_large_gauge':evaluate(yt,raw_after),
  'conventional_same_features':evaluate(yt,logits_plain),
  'native_vs_real_linear_max_logit_error':float(np.max(abs(cycle_before-reconstructed))),
  'cycle_max_drift_from_gauge':max_drift,'max_cycle_logits_change':float(np.max(abs(cycle_after-cycle_before))),
  'restore_max_logit_error':restore_err,'mirror_audit':audit,'root_bytes':8388608}
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='*',type=int,default=[17,29,41]);a=parser.parse_args()
 t=time.time();out={'runs':[run(seed) for seed in a.seeds]}
 out['elapsed_seconds']=time.time()-t
 (BASE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
