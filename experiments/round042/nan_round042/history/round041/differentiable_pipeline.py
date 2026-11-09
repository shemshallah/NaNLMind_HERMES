"""Learned writer -> 8MiB native triangle memory -> differentiable 3-step reasoning -> autoregressive pointer generation.
Controls: same evidence as conventional index, gauge, raw-edge, retention. Episode-scoped facts avoid cross-episode interference.
"""
import json,torch,collections
from pathlib import Path
from integrated_audit import Root,fit_writer,parse_neural,gen_with_graph,score_outputs,rewrite_status
from role_train import load_chains,sample,batch_data,CopyNet
from diff_reasoner import DiffCycleReasoner
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)

def make_local(graph,source,target):
 # Only names in this recorded episode. This selection is metadata-indexed, not a learned retrieval policy.
 names=list(dict.fromkeys([source,target]+[a for a,b in graph]+[b for a,b in graph]))
 assert len(names)<=8
 nameidx={w:i for i,w in enumerate(names)}
 A=torch.zeros((1,8,8));s=torch.zeros((1,8));d=torch.zeros((1,8))
 for a,b in graph:A[0,nameidx[a],nameidx[b]]=1
 s[0,nameidx[source]]=1;d[0,nameidx[target]]=1
 return A,s,d

def run(seed):
 old=sample(load_chains()['test'],220,seed+456)
 new=sample(load_chains()['test'],80,seed+10000)
 writer=fit_writer(seed)
 reason=DiffCycleReasoner();reason.load_state_dict(torch.load(BASE/f'reasoner_{seed}.pt',weights_only=True,map_location='cpu'));reason.eval()
 model=CopyNet();model.load_state_dict(torch.load(BASE/f'role_{seed}.pt',weights_only=True,map_location='cpu')['state']);model.eval()
 root=Root();episodes=[];standard=[];writer_errors=0
 def append(rows):
  nonlocal writer_errors
  for row in rows:
   edges=[]
   for sent in row[-1]:
    a,b,_=parse_neural(sent,writer)
    root.add(a,b);edges.append((a,b))
   episodes.append(list(range(len(root.metadata)-len(edges),len(root.metadata))))
   standard.append(edges)
 def rooted_graph(idx,raw=False):
  edges=[]
  import ctypes
  from integrated_audit import lib
  for j in episodes[idx]:
   a,b,x,y=root.metadata[j]
   val=lib.ep_raw(root.ptr,a,b) if raw else lib.ep_get(root.ptr,a,b)
   if val>0.5:edges.append((x,y))
  return edges
 def read(rows,mode='native',offset=0):
  adjusted=[]
  for i,row in enumerate(rows):
   edges=standard[i+offset] if mode=='conventional' else rooted_graph(i+offset,raw=mode=='raw')
   with torch.no_grad():logit=reason(*make_local(edges,row[3],row[4]))
   status=bool(logit.item()>0)
   adjusted.append((rewrite_status(row[0],status),*row[1:]))
  d=batch_data(adjusted);out=[]
  for j in range(0,len(rows),32):
   b=tuple(t[j:j+32] for t in d[:4])+(d[4][j:j+32],)
   out+=model.decode(b)
  return score_outputs(out,rows),out
 append(old)
 clean,gen1=read(old)
 control,_=read(old,'conventional')
 root.gauge(seed+9000,20.)
 gauge,gen2=read(old)
 raw,_=read(old,'raw')
 append(new)
 root.gauge(seed+11000,10.)
 old_after,_=read(old)
 new_after,_=read(new,offset=len(old))
 control_old,_=read(old,'conventional')
 control_new,_=read(new,'conventional',offset=len(old))
 p=BASE/f'diff_snapshot_{seed}.bin';root.save(p)
 g0=root.graph();root.gauge(seed+15000,12.);root.load(p)
 snap=root.graph()==g0
 res={'seed':seed,'native_old':clean,'control_old':control,'native_after_gauge':gauge,'raw_after_gauge':raw,'retention_native':old_after,'retention_control':control_old,'new_native':new_after,'new_control':control_new,'same_output_after_gauge':sum(a==b for a,b in zip(gen1,gen2)),'snapshot_restore':snap,'sample':[{'facts':r[-1],'query':r[0].split(' ; status :')[0],'generated':g} for r,g in list(zip(old,gen2))[:8]]}
 (BASE/f'diff_pipeline_{seed}.json').write_text(json.dumps(res,indent=2))
 print(json.dumps({k:v for k,v in res.items() if k!='sample'}),flush=True)
 root.close();return res
if __name__=='__main__':
 x=[run(seed) for seed in [17,29,43]]
 (BASE/'diff_pipeline_summary.json').write_text(json.dumps(x,indent=2))
