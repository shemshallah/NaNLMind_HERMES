"""Actual shared 8MiB root used to emit held-out WordNet gloss-prefix bytes.
This compares gauge cycle decoding with naïve single-edge and ordinary indexed copy.
"""
import ctypes,json
from pathlib import Path
import numpy as np
import torch
from episodic import T,K,LENGTH,KeyEncoder,data,ep,noisy_query

def run(count=120):
 torch.set_num_threads(1)
 lib=ctypes.CDLL(str(T/'libepisodic.so'))
 lib.memory_new.restype=ctypes.c_void_p
 lib.memory_store.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
 lib.memory_read.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_read.restype=ctypes.c_int
 lib.memory_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
 lib.memory_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_raw.restype=ctypes.c_int
 lib.memory_free.argtypes=[ctypes.c_void_p]
 model=KeyEncoder();model.load_state_dict(torch.load(T/'episodic_17.pt',map_location='cpu',weights_only=True)['state']);model.eval()
 pool=data()['test'];rng=np.random.default_rng(818)
 res={'episodes':count,'key_found':0,'native_exact_clean':0,'native_exact_after_gauge':0,'conventional_copy_exact':0,'raw_bytes_correct_after_gauge':0,'total_bytes':count*LENGTH,'shifts':8.0}
 for k in range(count):
  names,vals,target=ep(pool,rng,b=1)
  with torch.no_grad():
   keys=model(names);q=model(noisy_query(names,target,rng)[:,None,:])[0,0]
   pred=int((keys[0]@q).argmax())
  res['key_found']+=pred==int(target[0]); expected=vals[0,int(target[0])].tolist()
  # One episode instantiated into the actual contiguous C++ root.
  p=lib.memory_new();assert p
  for j in range(K):
   for t in range(LENGTH):lib.memory_store(p,j*LENGTH+t,int(vals[0,j,t]))
  before=[lib.memory_read(p,pred*LENGTH+t) for t in range(LENGTH)]
  res['native_exact_clean']+=before==expected
  # The plain indexed-memory control can use the same retrieved values.
  res['conventional_copy_exact']+=vals[0,pred].tolist()==expected
  lib.memory_gauge(p,900+k,8.)
  after=[lib.memory_read(p,pred*LENGTH+t) for t in range(LENGTH)]
  raw_after=[lib.memory_raw(p,pred*LENGTH+t) for t in range(LENGTH)]
  res['raw_bytes_correct_after_gauge']+=sum(int(x==y) for x,y in zip(raw_after,expected))
  res['native_exact_after_gauge']+=after==expected
  lib.memory_free(p)
 for key in ('key_found','native_exact_clean','native_exact_after_gauge','conventional_copy_exact'):
  res[key+'_fraction']=res[key]/count
 res['raw_byte_accuracy_after_gauge']=res['raw_bytes_correct_after_gauge']/(count*LENGTH)
 (T/'integration_results.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':run()
