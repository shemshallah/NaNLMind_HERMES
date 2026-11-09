"""Round 041 causal audit. Change native triangle evidence, recompute learned proof,
and autoregressively regenerate. Contrasts with an equally informed ordinary edge table.
"""
import json, random, torch
from pathlib import Path
from integrated_audit import Root,lib,fit_writer,parse_neural,rewrite_status
from diff_reasoner import DiffCycleReasoner
from differentiable_pipeline import make_local
from role_train import load_chains, CopyNet, batch_data
BASE=Path(__file__).resolve().parent
torch.set_num_threads(1)

def evaluate(model,reason,base_rows,graphs):
    changed=[];statuses=[]
    with torch.no_grad():
        for row,edges in zip(base_rows,graphs):
            status=bool(reason(*make_local(edges,row[3],row[4]),steps=3).item()>0)
            statuses.append(status)
            changed.append((rewrite_status(row[0],status),*row[1:]))
        data=batch_data(changed)
        results=[]
        for start in range(0,len(changed),32):
            sliced=tuple(v[start:start+32] for v in data[:4])+(data[4][start:start+32],)
            results.extend(model.decode(sliced))
    return statuses,results

def valid_answer(answer,row):
    from train import ANSWERS,NOANSWER
    if row[2]:return answer in [x.format(a=row[3],b=row[4]) for x in ANSWERS]
    return answer in NOANSWER

def run(seed,n=80):
    from role_train import sample
    rng=random.Random(seed+800)
    candidates=[r for r in sample(load_chains()['test'],n*9,seed+20700) if r[2] and len(r[-1])==3]
    rows=candidates[:n]
    assert len(rows)==n
    writer=fit_writer(seed)
    model=CopyNet();model.load_state_dict(torch.load(BASE/f'role_{seed}.pt',weights_only=True,map_location='cpu')['state']);model.eval()
    reason=DiffCycleReasoner();reason.load_state_dict(torch.load(BASE/f'reasoner_{seed}.pt',weights_only=True,map_location='cpu'));reason.eval()
    root=Root();episodes=[];ordinary=[]
    for row in rows:
        slots=[];edges=[]
        for text in row[-1]:
            a,b,_=parse_neural(text,writer)
            root.add(a,b)
            slots.append(len(root.metadata)-1)
            edges.append((a,b))
        episodes.append(slots);ordinary.append(edges)
    def read_native():
        return [[(root.metadata[j][2],root.metadata[j][3]) for j in slots if lib.ep_get(root.ptr,*root.metadata[j][:2])>0.5] for slots in episodes]
    def read_raw():
        return [[(root.metadata[j][2],root.metadata[j][3]) for j in slots if lib.ep_raw(root.ptr,*root.metadata[j][:2])>0.5] for slots in episodes]
    base_native=read_native()
    s0,g0=evaluate(model,reason,rows,base_native)
    cs0,cg0=evaluate(model,reason,rows,ordinary)
    # Suppress exactly one indispensable edge from every episode, preserving all other episodes.
    removed=[]
    for slots in episodes:
        j=slots[rng.randrange(len(slots))];removed.append(j)
        a,b,_,_=root.metadata[j]
        lib.ep_set(root.ptr,a,b,0.)
    after_native=read_native()
    s1,g1=evaluate(model,reason,rows,after_native)
    ordinary_removed=[[p for p in edges if p != (root.metadata[j][2],root.metadata[j][3])] for edges,j in zip(ordinary,removed)]
    cs1,cg1=evaluate(model,reason,rows,ordinary_removed)
    # Gauge interference should not change the intervention consequence.
    root.gauge(seed+43000,18.)
    sg,gg=evaluate(model,reason,rows,read_native())
    # Restore removed links by cycle writing, despite gauge shifts.
    for j in removed:
        a,b,_,_=root.metadata[j];lib.ep_set(root.ptr,a,b,1.)
    s2,g2=evaluate(model,reason,rows,read_native())
    result={
        'seed':seed,'total':n,
        'initial_supported':sum(s0),'initial_generated_correct':sum(valid_answer(g,r) for g,r in zip(g0,rows)),
        'conventional_initial_supported':sum(cs0),'conventional_initial_generated_correct':sum(valid_answer(g,r) for g,r in zip(cg0,rows)),
        'after_edge_removed_unsupported':sum(not s for s in s1),
        'after_edge_removed_generated_abstentions':sum(g in __import__('train').NOANSWER for g in g1),
        'conventional_after_edge_removed_unsupported':sum(not s for s in cs1),
        'conventional_after_edge_removed_generated_abstentions':sum(g in __import__('train').NOANSWER for g in cg1),
        'gauge_same_status':sum(x==y for x,y in zip(s1,sg)),
        'gauge_same_generation':sum(x==y for x,y in zip(g1,gg)),
        'restored_supported':sum(s2),'restored_generated_correct':sum(valid_answer(g,r) for g,r in zip(g2,rows)),
        'raw_after_gauge_false_supported':sum(bool(reason(*make_local(edges,row[3],row[4]),steps=3).item()>0) for row,edges in zip(rows,read_raw())),
        'samples':[{'premises':rows[i][-1],'query':f'{rows[i][3]} -> {rows[i][4]}','before':g0[i],'after_removal':g1[i],'after_restore':g2[i]} for i in range(min(5,n))]
    }
    root.close()
    (BASE/f'counterfactual_{seed}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    return result

if __name__=='__main__':
    out=[run(s) for s in (17,29,43)]
    (BASE/'counterfactual_summary.json').write_text(json.dumps(out,indent=2))
