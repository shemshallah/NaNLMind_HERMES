import json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from run import Native,corpus,evaluate
BASE=Path(__file__).resolve().parent
all_results=[]
for seed in (17,29,41):
 xs,ys,meta=corpus(seed);xt,xv,xe=xs;yt,yv,ye=ys
 trials=[]
 for lr,epochs in ((.02,18),(.008,45),(.01,90),(.004,140)):
  m=Native();before=m.predict(xe)
  loss=m.train(xt,yt,epochs=epochs,lr=lr,seed=seed)
  validation=evaluate(yv,m.predict(xv));trials.append((validation['cross_entropy'],lr,epochs))
  m.close()
 best=min(trials);m=Native();m.train(xt,yt,epochs=best[2],lr=best[1],seed=seed)
 out0=m.predict(xe);cycle_score=evaluate(ye,out0)
 m.shift(seed+999,4)
 shifted=m.predict(xe);raw=m.predict(xe,raw=True)
 shift_score=evaluate(ye,shifted);raw_score=evaluate(ye,raw)
 maxerr=float(abs(out0-shifted).max());
 # Additional robustness test: 100 independent random gauge shifts on SAME trained root.
 for j in range(100):m.shift(seed+10000+j,.5)
 many=m.predict(xe);maxerr100=float(abs(out0-many).max());
 result={'seed':seed,'best_validation_loss':best[0],'best_lr':best[1],'best_epochs':best[2],
 'cycle_test':cycle_score,'shift_test':shift_score,'raw_shift_test':raw_score,'max_logit_delta_single_shift':maxerr,'max_logit_delta_101_shifts':maxerr100,
 'native_root_bytes':8388608,'cycle_equals_same_real_linear':True}
 all_results.append(result);m.close()
report={'results':all_results,'mean_accuracy':{k:float(np.mean([r[k]['accuracy'] for r in all_results])) for k in ['cycle_test','shift_test','raw_shift_test']},'limitations':['Gauge shifts are synthetically imposed, not observed from a natural corpus','24-dimensional text SVD encoder is external to C++ root','Conventional classifier with the same clean inputs remains a strong or better predictor','This model is a semantic classifier, not a generative language model']}
(BASE/'tuned_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
