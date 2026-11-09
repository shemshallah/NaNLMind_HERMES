"""Native 4-/5-hop held-out WordNet chains, learned propagator trained on 3 hops.
Independent directed real-hypernym path; nonpath distractors and missing-edge intervention.
"""
import json,random,re,torch
from pathlib import Path
from integrated_audit import Root,lib,fit_writer,parse_neural,rewrite_status
from diff_reasoner import DiffCycleReasoner
from differentiable_pipeline import make_local
from role_train import CopyNet,batch_data
from train import TEMPLATES,ANSWERS,NOANSWER
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)

def find_paths():
    records={};parent={}
    for ln in (BASE/'corpus.jsonl').open():
        r=json.loads(ln)
        if r.get('pos')!='noun' or not r.get('lemmas'):continue
        name=r['lemmas'][0].lower().replace('_',' ')
        if not re.fullmatch('[a-z][a-z-]{2,19}',name):continue
        records[r['id']]=(name,r['split'],r['pointers'])
    for k,(_,_,ptrs) in records.items():
        options=['n'+p[1] for p in ptrs if p[0]=='@' and 'n'+p[1] in records]
        if options:parent[k]=options[0]
    paths=[]
    for k,(_,split,_) in records.items():
        if split!='test':continue
        cur=k;walk=[cur]
        for _ in range(5):
            if cur not in parent:break
            cur=parent[cur];walk.append(cur)
        if len(walk)==6:
            names=[records[z][0] for z in walk]
            if len(set(names))==6:paths.append(names)
    return paths

def outputs(model,reason,rows,graphs,steps):
    with torch.no_grad():
        present=[bool(reason(*make_local(e,a,b),steps=steps).item()>0) for (a,b),e in zip(rows,graphs)]
        fake=[('source : '+a+' ; target : '+b+' ; status : '+('supported' if s else 'unsupported')+' ; facts :', '', s,a,b,[],[]) for (a,b),s in zip(rows,present)]
        d=batch_data(fake);out=[]
        for j in range(0,len(fake),24):
            x=tuple(t[j:j+24] for t in d[:4])+(d[4][j:j+24],)
            out.extend(model.decode(x))
    correct=sum(any(t.format(a=a,b=b)==z for t in ANSWERS) if s else z in NOANSWER for (a,b),s,z in zip(rows,present,out))
    return present,out,correct

def run(seed,n=55):
    rng=random.Random(seed+50000)
    paths=find_paths();assert len(paths)>n,(len(paths),n)
    chosen=rng.sample(paths,n)
    writer=fit_writer(seed)
    model=CopyNet();model.load_state_dict(torch.load(BASE/f'role_{seed}.pt',weights_only=True,map_location='cpu')['state']);model.eval()
    reason=DiffCycleReasoner();reason.load_state_dict(torch.load(BASE/f'reasoner_{seed}.pt',weights_only=True,map_location='cpu'));reason.eval()
    root=Root();episodes=[];control=[];queries=[];depths=[]
    for k,p in enumerate(chosen):
        d=4 if k%2==0 else 5
        slots=[];edges=[]
        for a,b in zip(p[:d],p[1:d+1]):
            templ=rng.choice(rng.choice(TEMPLATES));sent=templ.format(a=a,b=b)
            x,y,_=parse_neural(sent,writer)
            slots.append(len(root.metadata));root.add(x,y);edges.append((x,y))
        episodes.append(slots);control.append(edges);queries.append((p[0],p[d]));depths.append(d)
    def native():
        return [[(root.metadata[j][2],root.metadata[j][3]) for j in es if lib.ep_get(root.ptr,*root.metadata[j][:2])>.5] for es in episodes]
    def batch(graphs):
        all_s=[];all_o=[];total=0
        for step in (4,5):
            ids=[i for i,d in enumerate(depths) if d==step]
            s,o,c=outputs(model,reason,[queries[i] for i in ids],[graphs[i] for i in ids],steps=step)
            all_s.extend(s);all_o.extend(o);total+=c
        return sum(all_s),total,len(all_s),all_o
    before=batch(native());ctrl=batch(control)
    # Deliberately use 5 steps for both 4-/5-hop test; strict 4-hop generalization checked separately.
    def all_steps(graphs):
        s,o,c=outputs(model,reason,queries,graphs,steps=5)
        return sum(s),c,o
    full=all_steps(native());ordinary=all_steps(control)
    root.gauge(seed+12345,17.);gauged=all_steps(native())
    # Remove middle edges: proof should not be re-created from unrelated vertices.
    removed=[]
    for slots in episodes:
        j=slots[len(slots)//2];removed.append(j);a,b,_,_=root.metadata[j];lib.ep_set(root.ptr,a,b,0.)
    after=all_steps(native());ordinary_missing=all_steps([[e for j,e in zip(slots,edges) if j != rem] for slots,edges,rem in zip(episodes,control,removed)])
    root.gauge(seed+56789,10.);after_gauge=all_steps(native())
    for j in removed:
        a,b,_,_=root.metadata[j];lib.ep_set(root.ptr,a,b,1.)
    restored=all_steps(native())
    result={'seed':seed,'tested':n,'four_hop':sum(x==4 for x in depths),'five_hop':sum(x==5 for x in depths),
            'native_complete_supported':full[0],'native_complete_correct_generation':full[1],
            'conventional_complete_supported':ordinary[0],'conventional_complete_correct_generation':ordinary[1],
            'native_after_gauge_supported':gauged[0],'native_after_gauge_generation_unchanged':sum(x==y for x,y in zip(full[2],gauged[2])),
            'native_missing_edge_unsupported':n-after[0],'native_missing_edge_correct_generation':after[1],
            'control_missing_edge_unsupported':n-ordinary_missing[0],'control_missing_edge_correct_generation':ordinary_missing[1],
            'missing_edge_after_gauge_unchanged':sum(x==y for x,y in zip(after[2],after_gauge[2])),
            'restored_supported':restored[0],'restored_correct_generation':restored[1]}
    root.close();print(result,flush=True)
    (BASE/f'deep_native_{seed}.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    a=[run(s) for s in (17,29,43)]
    (BASE/'deep_native_summary.json').write_text(json.dumps(a,indent=2))
