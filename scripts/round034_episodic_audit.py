#!/usr/bin/env python3
"""Audit Round 034 episodic copy/abstain and reproduce the pure resident path."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from nanlmind_hermes.resident.cells import REGION_BYTES, ResidentCells
from nanlmind_hermes.resident.episodic_gate import (EpisodicCycleMemory, evaluate, load_pool, make_episode, stress, train_gate)
from nanlmind_hermes.resident.full_cycle_memory import FullCycleMemory
BASE=ROOT/'experiments/round034/nan_round034'
DEFAULT_JSON=ROOT/'docs/ROUND034_EPISODIC_GATE.json'; DEFAULT_MD=ROOT/'docs/ROUND034_EPISODIC_GATE.md'

def verify():
    text=(BASE/'SHA256SUMS').read_text(encoding='utf-8'); checked=[]; missing=[]
    for line in text.splitlines():
        if not line.strip(): continue
        digest,name=line.split(None,1); path=BASE/name.strip()
        if not path.is_file():
            missing.append(name.strip()); continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=digest: raise ValueError('checksum mismatch: '+name)
        checked.append(name.strip())
    return {'verified':checked,'missing':missing}

def artifact_summary():
    def read(name):
        path=BASE/name
        return json.loads(path.read_text()) if path.is_file() else None
    return {'results':read('results.json'),'recursive':read('recursive_results.json'),'hybrid':read('hybrid_results.json'),'stress':read('stress_results.json')}

def pure(seed=17, queries=300):
    pool=load_pool(BASE/'wordnet_episodic_pool.json'); gate=train_gate(pool,seed=seed); result=evaluate(pool['test'],gate,count=queries,corruption=2)
    cells=ResidentCells.format_region(bytearray(REGION_BYTES)); memory=FullCycleMemory(cells); import random
    ep=make_episode(pool['test'],random.Random(seed),present=True); store=EpisodicCycleMemory(memory)
    for i,row in enumerate(ep.records): store.store(i,row)
    receipt=store.copy(ep.target,ep.records[ep.target],seed=seed+100)
    return {'seed':seed,'queries':queries,'router':result,'cycle_copy':{'cycle_exact':receipt.cycle_exact,'raw_exact':receipt.raw_exact},'stress':stress(pool['test'],gate,count=100),'canonical_merge':False,'language_generation_claim':False}

def md(record):
    a=record['artifact']; means={str(s):v['learned_router']['accept_accuracy'] for s,v in a['results'].items()}
    lines=['# Round 034 episodic gate','','Source archive SHA-256 verified: **%d files**; missing from archive: %s.'%(len(record['checksums']['verified']), ', '.join(record['checksums']['missing']) or 'none'),'','## Artifact result','| seed | router acc | false-answer rate |','|---:|---:|---:|']
    for seed,row in a['results'].items(): lines.append('| %s | %.4f | %.4f |'%(seed,row['learned_router']['accept_accuracy'],row['learned_router']['false_answer_rate']))
    lines += ['','## Pure resident reproduction','| field | value |','|---|---:|', '| router accept accuracy | %.4f |'%record['pure']['router']['accept_accuracy'], '| positive accept | %.4f |'%record['pure']['router']['positive_accept_rate'], '| false answer | %.4f |'%record['pure']['router']['false_answer_rate'], '| cycle copy exact | %s |'%record['pure']['cycle_copy']['cycle_exact'], '| raw edge exact | %s |'%record['pure']['cycle_copy']['raw_exact'], '','## Boundaries','- Pure Python char-key fallback is not the supplied neural encoder; its score differs from the 99.0% artifact.','- Copying a definition with a confidence gate is not free-form generation.','- The conventional indexed control ties the cycle copy path; no NaN-unique advantage is claimed.','- Raw edges are not gauge invariant; cycle reads are.','']
    return '\n'.join(lines)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--json',type=Path,default=DEFAULT_JSON); p.add_argument('--markdown',type=Path,default=DEFAULT_MD); p.add_argument('--source-only',action='store_true'); a=p.parse_args(argv)
    record={'schema':'nanlm.round034-episodic-audit/1','checksums':verify(),'artifact':artifact_summary(),'pure':None if a.source_only else pure(),'claims':{'canonical':False,'language_model_win':False,'nan_unique_advantage':False}}
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    a.markdown.write_text(md(record)); print(json.dumps({'json':str(a.json),'markdown':str(a.markdown),'checksums':len(record['checksums']['verified']),'missing':record['checksums']['missing'],'pure':bool(record['pure'])})); return 0
if __name__=='__main__': raise SystemExit(main())
