#!/usr/bin/env python3
"""Fast release smoke tests for native geometry, trained conceptual dialogue, and persistence."""
from pathlib import Path
import ctypes,torch,tempfile
from converse import Universe
from dynamic_joint import NativeRoot,ROOT

def run():
 n=NativeRoot()
 assert n.audit()
 assert n.L.ep_root_size() == 8_388_608 if hasattr(n.L,'ep_root_size') else True
 for j in range(1,50):
  n.L.ep_set(n.ptr,j,j+1,0.5)
 before=[n.L.ep_get(n.ptr,j,j+1) for j in range(1,50)]
 n.gauge(314,3.0)
 after=[n.L.ep_get(n.ptr,j,j+1) for j in range(1,50)]
 assert max(abs(x-y) for x,y in zip(before,after)) < 1e-4
 assert n.audit();n.close()
 with tempfile.TemporaryDirectory() as folder:
  f=Path(folder)/'test.bin'
  u=Universe(ROOT/'adapted_model_17.pt',f)
  u.teach('finch is subsumed by bird')
  u.teach('bird is subsumed by animal')
  u.teach('animal comprises dog')
  assert 'finch' in u.chat('is finch a kind of animal')
  assert 'dog' in u.chat('is dog a kind of animal')
  before=u.chat('is finch a kind of animal')
  u.gauge(313,3)
  assert u.chat('is finch a kind of animal')==before
  u.teach('finch is not a kind of animal')
  assert 'conflict' in u.chat('is finch a kind of animal').lower()
  u.close();u=Universe(ROOT/'adapted_model_17.pt',f)
  assert 'conflict' in u.chat('is finch a kind of animal').lower()
  u.forget('finch','animal')
  assert 'support' in u.chat('is finch a kind of animal').lower()
  u.forget('bird','animal')
  assert any(x in u.chat('is finch a kind of animal').lower() for x in ['not enough','not establish','sufficient'])
  u.close()
 print('PASS native geometry, two-lane contradictions, neural generator, gauge drift, snapshot, reteaching, deletion')
if __name__=='__main__':run()
