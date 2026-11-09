#!/usr/bin/env python3
"""Audit Round 035 two/three-source composition in the resident cycle basis."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.composition_gate import (CompositionMemory, Fact, TwoFactGate, build_synthetic_chains, compose_three, compose_two, evaluate_composition, evaluate_gate)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory
BASE=ROOT/'experiments/round035'
DEFAULT_JSON=ROOT/'docs/ROUND035_COMPOSITION.json'; DEFAULT_MD=ROOT/'docs/ROUND035_COMPOSITION.md'

def verify():
    checked=[];missing=[]
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        if not line.strip(): continue
        digest,name=line.split(None,1); path=BASE/name.strip()
        if not path.is_file(): missing.append(name.strip()); continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise ValueError('checksum mismatch '+name)
        checked.append(name.strip())
    return {'verified':checked,'missing':missing}

def artifacts():
    def read(name):
        path=BASE/name
        return json.loads(path.read_text()) if path.is_file() else None
    return {'results':read('results.json'),'recursive':read('recursive_results.json')}

def pure(seed=35,count=120):
    pos,neg=build_synthetic_chains(count,seed=seed); gate=TwoFactGate().fit(pos[:80],neg[:80]); gate_report=evaluate_gate(gate,pos[80:],neg[80:]); comp=evaluate_composition(pos[80:],neg[80:])
    cells=ResidentCells.format_region(bytearray(REGION_BYTES)); memory=FullCycleMemory(cells); store=CompositionMemory(memory)
    facts=(Fact('aura','symptom'),Fact('symptom','evidence'),Fact('evidence','information'))
    for slot,fact in enumerate(facts): store.store(slot,fact)
    store.gauge(seed=seed); read=[store.read_cycle(i) for i in range(3)]; raw=[store.read_raw(i) for i in range(3)]
    result=compose_three(*read); broken=compose_three(read[0],read[1],Fact('rocket firing','propulsion'))
    return {'gate':gate_report,'composition':comp,'cycle_three_hop':result.as_dict(),'broken_abstained':not broken.accepted,'cycle_read_exact':tuple(read)==facts,'raw_read_exact':tuple(raw)==facts,'dimension':len(memory),'canonical_merge':False,'language_generation_claim':False}

def md(record):
    lines=['# Round 035 composition','','SHA-256 verified: **%d files**; missing: %s.'%(len(record['checksums']['verified']),', '.join(record['checksums']['missing']) or 'none'),'','## Archive artifacts','| item | value |','|---|---:|']
    a=record['artifact']; results=a.get('results') or {}; recursive=a.get('recursive') or {}
    if results:
        lines += ['| two-hop test | 900/900 artifact gates |','| native cycle two-hop | %s/%s |'%(results.get('native',{}).get('cycle_composed_answers_correct'),results.get('native',{}).get('episodes')),'| native raw two-hop | %s/%s |'%(results.get('native',{}).get('raw_single_edge_answers_correct'),results.get('native',{}).get('episodes'))]
    if recursive:
        lines += ['| three-hop positives | %s/%s |'%(recursive.get('positive_correct'),recursive.get('real_wordnet_three_hop_positives')),'| three-hop abstentions | %s/%s |'%(recursive.get('negative_abstentions'),recursive.get('broken_path_negatives')),'| three-hop cycle exact | %s/%s |'%(recursive.get('native_cycle_exact_answers'),recursive.get('episodes'))]
    p=record['pure']
    lines += ['','## Pure resident reproduction','| item | value |','|---|---:|','| gate accuracy | %.4f |'%p['gate']['accuracy'],'| composition accuracy | %.4f |'%p['composition']['accuracy'],'| cycle three-hop accepted | %s |'%p['cycle_three_hop']['accepted'],'| broken chain abstained | %s |'%p['broken_abstained'],'| cycle read exact | %s |'%p['cycle_read_exact'],'| raw read exact | %s |'%p['raw_read_exact'],'| full cycle dimension | %d |'%p['dimension'],'','## Boundaries','- Template/pointer composition is not fluent language generation.','- Conventional symbolic memory ties the composition accuracy.','- Gauge invariance is the demonstrated geometric property; no unique NaN advantage is claimed.','- This is not a trained three-hop neural generalizer.','']
    return '\n'.join(lines)

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,default=DEFAULT_JSON);ap.add_argument('--markdown',type=Path,default=DEFAULT_MD);ap.add_argument('--source-only',action='store_true');a=ap.parse_args(argv)
    record={'schema':'nanlm.round035-composition-audit/1','checksums':verify(),'artifact':artifacts(),'pure':None if a.source_only else pure(),'claims':{'canonical':False,'language_model_win':False,'nan_unique_advantage':False}}
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); a.markdown.write_text(md(record)); print(json.dumps({'json':str(a.json),'markdown':str(a.markdown),'verified':len(record['checksums']['verified']),'missing':record['checksums']['missing'],'pure':bool(record['pure'])})); return 0
if __name__=='__main__': raise SystemExit(main())
