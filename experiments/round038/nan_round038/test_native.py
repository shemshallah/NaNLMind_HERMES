"""Minimal independent native precision, snapshot and gauge tests."""
import ctypes, hashlib, tempfile, random
from pathlib import Path
from experiment import Native
n=Native();rng=random.Random(70)
try:
 vals=[rng.uniform(-2,2) for _ in range(600)]
 for x in vals:n.write(x)
 baseline=[n.read(i) for i in range(len(vals))]
 assert max(abs(a-b) for a,b in zip(vals,baseline))<3e-7
 with tempfile.TemporaryDirectory() as d:
  snap=Path(d)/'root.bin';n.save(snap)
  assert snap.stat().st_size==8388608
  n.gauge(101,45.0)
  shifted=[n.read(i) for i in range(len(vals))]
  err=max(abs(a-b) for a,b in zip(baseline,shifted))
  assert err<0.00005,err
  n.restore(snap)
  reloaded=[n.read(i) for i in range(len(vals))]
  assert baseline==reloaded
  print({'geometry_max_absolute_error_after_gauge':err,'snapshot_bytes':snap.stat().st_size,'restoration_exact':True,'cells_checked':len(vals)})
finally:n.close()
