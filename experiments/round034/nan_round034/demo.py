"""Demonstrate generated grounded completion from an unseen held-out WordNet name.
The 24-byte completion is copied from native cycle memory, not synthesized prose.
"""
import argparse,ctypes,json,torch,numpy as np
from episodic import data,ep,KeyEncoder,K,LENGTH
from route_train import Router,example
from pathlib import Path
T=Path(__file__).resolve().parent

def run(seed=17,episode=2,positive=True):
 pool=data()['test'];rng=np.random.default_rng(seed+1000+episode)
 key=KeyEncoder();key.load_state_dict(torch.load(T/f'episodic_{seed}.pt',map_location='cpu',weights_only=True)['state']);key.eval()
 ck=torch.load(T/f'router_{seed}.pt',map_location='cpu',weights_only=True)
 gate=Router();gate.load_state_dict(ck['state']);gate.eval()
 with torch.no_grad():
  features,available,correct,pick,target,names,values,query=example(key,pool,rng,positive=positive)
  score=float(torch.sigmoid(gate(torch.tensor(features[None,:]))).item())
 accepted=score>=ck['threshold'];pred='I do not have a grounded definition for that query.'
 if accepted:
  lib=ctypes.CDLL(str(T/'libepisodic.so'));lib.memory_new.restype=ctypes.c_void_p
  lib.memory_store.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
  lib.memory_read.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.memory_read.restype=ctypes.c_int
  lib.memory_gauge.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float];lib.memory_free.argtypes=[ctypes.c_void_p]
  p=lib.memory_new();assert p
  try:
   for j in range(K):
    for t in range(LENGTH):lib.memory_store(p,j*LENGTH+t,int(values[j,t]))
   lib.memory_gauge(p,991,8.)
   pred=bytes(lib.memory_read(p,pick*LENGTH+t) for t in range(LENGTH)).decode('utf-8','replace')
  finally:lib.memory_free(p)
 out={'query':bytes([x-1 for x in query if x]).decode('ascii','replace'),'available':positive,'copy_probability':score,'accepted':accepted,'completion':pred,'correct':bool(accepted and correct) if positive else not accepted}
 print(json.dumps(out,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,default=17,choices=[17,29,41]);a.add_argument('--episode',type=int,default=2);a.add_argument('--absent',action='store_true');args=a.parse_args();run(args.seed,args.episode,not args.absent)
