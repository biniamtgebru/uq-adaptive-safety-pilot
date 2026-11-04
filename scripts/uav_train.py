from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle, time
from pilot.uav import *
from pilot.common import Ensemble, conformal_q, quantile_level
from pilot.common import Ensemble, conformal_q
t=time.time()
X,Vm,A,Y = run(60000, lambda k:1.0, 1000)
D = targets(Vm,A,Y)
print('sim',time.time()-t,'resid std',D.std(0),'|v| mean',np.linalg.norm(Vm,axis=1).mean(),'|a| mean',np.linalg.norm(A,axis=1).mean())
ens=Ensemble(M=5,hidden=(32,32),seed=0); ens.fit(X,D); print('train',time.time()-t)
pickle.dump(ens,open(ARTIFACTS/'uav_ensemble.pkl','wb'))
Xc,Vc,Ac,Yc = run(30000, lambda k:1.0, 2000)
for norm in [False,True]:
    sc,_=scores(ens,Xc,Vc,Ac,Yc,norm)
    print('normalized' if norm else 'plain','cal mean',sc.mean(),'q95',np.quantile(sc,.95))
s0=np.linalg.norm(targets(Vc,Ac,Yc),axis=1); print('physics-only mean',s0.mean(),'q95',np.quantile(s0,.95))
# shifted regime error with unchanged model
for kap in [0.79,0.7]:
    Xs,Vs,As,Ys=run(20000,lambda k:kap,3000)
    ss,_=scores(ens,Xs,Vs,As,Ys,False); print('kappa',kap,'score mean',ss.mean(),'q95',np.quantile(ss,.95))
