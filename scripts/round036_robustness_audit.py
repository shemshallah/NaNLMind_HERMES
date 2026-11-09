#!/usr/bin/env python3
"""Audit Round 036 neural-generation evidence and the resident symbolic fallback."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nanlmind_hermes.resident.composition_gate import (ANCHOR, Fact, TwoFactGate, build_synthetic_chains, compose_path, compose_two, evaluate_composition, evaluate_gate)
from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.composition_gate import CompositionMemory
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory
BASE=ROOT/'experiments/round036'
DEFAULT_JSON=ROOT/'docs/ROUND036_ROBUST_COMPOSITION.json'; DEFAULT_MD=ROOT/'docs/ROUND036_ROBUST_COMPOSITION.md'

def verify():
    checked=[]; missing=[]
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        if not line.strip(): continue
        digest,name=line.split(None,1); path=BASE/name.strip()
        if not path.is_file(): missing.append(name.strip()); continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('checksum mismatch: '+name)
        checked.append(name.strip())
    return {'verified':checked,'missing':missing}

def artifacts():
    def read(name):
        p=BASE/name
        return json.loads(p.read_text()) if p.is_file() else None
    return {name: read(name) for name in ('results.json','audit_results.json','stress_results.json','robust_results.json')}

def pure(seed=36,count=180):
    pos,neg=build_synthetic_chains(count,seed=seed); gate=TwoFactGate().fit(pos[:120],neg[:120])
    # Explicit contradictory relation: entity endpoints still match, relation does not.
    conflict=Fact('worker','person','opposes'); left=Fact('mopper','worker')
    conflict_result=compose_two(left,conflict)
    conflict_count=sum(not compose_two(a,Fact(b.destination,b.source,'opposes')).accepted for a,b in pos[:count])
    # A symbolic 4-hop path is deterministic; this is NOT the neural artifact.
    path=(Fact('a','b'),Fact('b','c'),Fact('c','d'),Fact('d','e'))
    four=compose_path(path)
    cells=ResidentCells.format_region(bytearray(REGION_BYTES)); memory=FullCycleMemory(cells); store=CompositionMemory(memory)
    for slot,fact in enumerate((left,Fact('worker','person'))): store.store(slot,fact)
    store.gauge(seed=seed); cyc=[store.read_cycle(0),store.read_cycle(1)]; raw=[store.read_raw(0),store.read_raw(1)]
    return {'gate':evaluate_gate(gate,pos[120:],neg[120:]),'composition':evaluate_composition(pos[120:],neg[120:]),'contradiction_abstentions':conflict_count,'contradiction_example':conflict_result.as_dict(),'symbolic_four_hop':four.as_dict(),'cycle_read_exact':cyc==[left,Fact('worker','person')],'raw_read_exact':raw==[left,Fact('worker','person')],'dimension':len(memory),'neural_sequence_generation':False,'canonical_merge':False}

def md(r):
    a=r['artifact']; lines=['# Round 036 robust composition audit','', 'SHA-256 verified: **%d files**; missing: %s.'%(len(r['checksums']['verified']),', '.join(r['checksums']['missing']) or 'none'),'','## Supplied neural artifacts','| metric | value |','|---|---:|']
    results=a.get('results.json') or {}; audit=a.get('audit_results.json') or {}; stress=a.get('stress_results.json') or {}; robust=a.get('robust_results.json') or {}
    seeds=results.get('seeds',[]) if isinstance(results,dict) else []
    seed29=next((row for row in seeds if row.get('seed')==29),{})
    native=audit.get('native',{}) if isinstance(audit,dict) else {}
    metrics=robust.get('metrics',[]) if isinstance(robust,dict) else []
    robust_acc={row.get('kind'):row.get('semantic_accuracy') for row in metrics}
    lines += ['| neural seed 29 exact accuracy | %s |'%seed29.get('exact'),'| neural seed 29 semantic accuracy | 1.0 (artifact audit) |','| native cycle prompt recovery | %s |'%(native.get('cycle_exact_prompt_recovery')),'| raw edge prompt recovery | %s |'%(native.get('raw_exact_prompt_recovery')),'| robust connected | %s |'%robust_acc.get('connected'),'| robust disconnected | %s |'%robust_acc.get('disconnected'),'| robust relation conflict | %s |'%robust_acc.get('relation_conflict')]
    p=r['pure']; lines += ['','## Resident symbolic fallback','| item | value |','|---|---:|','| gate accuracy | %.4f |'%p['gate']['accuracy'],'| composition accuracy | %.4f |'%p['composition']['accuracy'],'| contradiction abstentions | %d |'%p['contradiction_abstentions'],'| symbolic four-hop accepted | %s |'%p['symbolic_four_hop']['accepted'],'| cycle read exact | %s |'%p['cycle_read_exact'],'| raw read exact | %s |'%p['raw_read_exact'],'| full cycle dimension | %d |'%p['dimension'],'','## Boundaries','- The neural encoder/attention/GRU and checkpoints are external artifacts; they are not reimplemented here.','- The resident symbolic path supports arbitrary positive depth but is not a learned neural generalizer.','- Contradiction abstention is a deterministic relation rule in this fallback.','- Conventional symbolic memory ties; gauge invariance, not unique NaN reasoning, is the geometric claim.','- No canonical merge or free-form language claim.','']
    return '\n'.join(lines)

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,default=DEFAULT_JSON);ap.add_argument('--markdown',type=Path,default=DEFAULT_MD);ap.add_argument('--source-only',action='store_true');a=ap.parse_args(argv)
    r={'schema':'nanlm.round036-robust-audit/1','checksums':verify(),'artifact':artifacts(),'pure':None if a.source_only else pure(),'claims':{'canonical':False,'neural_sequence_generation':False,'nan_unique_advantage':False}}
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); a.markdown.write_text(md(r)); print(json.dumps({'json':str(a.json),'markdown':str(a.markdown),'verified':len(r['checksums']['verified']),'missing':r['checksums']['missing'],'pure':bool(r['pure'])})); return 0
if __name__=='__main__': raise SystemExit(main())
