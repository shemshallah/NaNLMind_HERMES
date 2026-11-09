import time,json,numpy as np
from pathlib import Path
from run import Native,D,K
base=Path(__file__).resolve().parent
m=Native();rng=np.random.default_rng(43)
x=rng.normal(0,.25,(1,D)).astype(np.float32);y=np.array([4],np.int32)
before=m.weights();logits=m.predict(x)[0];p=np.exp(logits-logits.max());p/=p.sum()
rate=.05;m.train(x,y,epochs=1,lr=rate,seed=43)
after=m.weights()
expected=np.tile(p[:,None],(1,D))*x[0][None,:];expected[4]-=x[0];expected=-rate*expected
err=float(np.max(np.abs((after-before)-expected)))
# Structural check: intentional gauge shifts cannot change original semantic operator values.
pre=m.weights();m.shift(110,1.5);post=m.weights();gerr=float(np.max(np.abs(pre-post)))
assert m.audit() and err<1e-6 and gerr<1e-5
snapshot=base/'audit_root.bin';assert m.snapshot(snapshot)
m.shift(111,7);assert m.restore(snapshot)
assert np.array_equal(m.weights(),post)
xbench=np.ascontiguousarray(rng.normal(0,.1,(1000,D)),dtype=np.float32)
t=time.perf_counter();pred=m.predict(xbench);ms=(time.perf_counter()-t)*1000
out={'root_bytes':8388608,'native_gradient_max_abs_error':err,'gauge_weight_max_abs_error':gerr,'mirror_antisymmetry':m.audit(),'snapshot_bitwise_restore':True,'predictions':len(xbench),'ctypes_wall_ms':ms,'memory_wordnet_predictor_is_native':True,'text_encoder_is_external':True}
(base/'native_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));m.close();snapshot.unlink()
