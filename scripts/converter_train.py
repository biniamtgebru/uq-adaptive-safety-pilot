from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle, time
from pilot.converter import *
from pilot.common import Ensemble, conformal_q, quantile_level
t=time.time()
X,Im,V,Yn = run_open(60000, lambda k:1.0, seed=1000)
D,_ = residual_targets(Im,V,Yn)
print('sim time',time.time()-t, 'resid std',D.std(0), 'max',np.abs(D).max(0))
print('approx. noise-floor std (measurement noise only)', np.sqrt(2)*sig_meas, 'norm of sat resid mean', np.linalg.norm(D,axis=1).mean())
t=time.time()
ens=Ensemble(M=5,seed=0); ens.fit(X,D)
print('train time',time.time()-t)
# calibration set (fresh seed)
Xc,Imc,Vc,Ync = run_open(30000, lambda k:1.0, seed=2000)
mc,_=ens.predict_batch(Xc)
p0=np.array([f0(Imc[k],Vc[k]) for k in range(len(Imc))])
sc=np.linalg.norm(Ync-(p0+mc),axis=1)
s_base=np.linalg.norm(Ync-p0,axis=1)
print('cal score mean learned',sc.mean(),'q95',np.quantile(sc,.95),' | physics-only mean',s_base.mean(),'q95',np.quantile(s_base,.95))
pickle.dump(dict(ens=ens,cal=sc),open(ARTIFACTS/'converter_model.pkl','wb'))
