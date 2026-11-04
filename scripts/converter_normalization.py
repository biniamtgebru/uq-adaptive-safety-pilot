from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle
from pilot.converter import *
from pilot.common import Ensemble, conformal_q, quantile_level
m=pickle.load(open(ARTIFACTS/'converter_model.pkl','rb')); ens=m['ens']
alpha=0.05
def scores(seed,n=30000):
    X,Im,V,Yn=run_open(n,lambda k:1.0,seed)
    mu,sd=ens.predict_batch(X)
    p0=np.array([f0(Im[k],V[k]) for k in range(n)])
    s=np.linalg.norm(Yn-(p0+mu),axis=1)
    dv=np.linalg.norm(V-E-R*Im,axis=1)
    return s,sd,dv
sc,sdc,dvc=scores(2000)
variants={
 'plain':lambda s,sd,dv: s,
 'ens-std':lambda s,sd,dv: s/(sd+0.01),
 'phys (dv)':lambda s,sd,dv: s/(1+dv/20.0),
}
data=[scores(7000+i,20000) for i in range(20)]
res={k:[] for k in variants}; wid={k:[] for k in variants}
for name,f in variants.items():
    q=conformal_q(f(sc,sdc,dvc),alpha)
    for (s,sd,dv) in data:
        z=f(s,sd,dv)
        cov_all=(z<=q).mean(); tr=dv>20
        res[name].append([cov_all,(z[tr]<=q).mean(),(z[~tr]<=q).mean()])
        # mean radius in amps: back-transform
        if name=='plain': rad=np.full(len(s),q)
        elif name=='ens-std': rad=q*(sd+0.01)
        else: rad=q*(1+dv/20.0)
        wid[name].append([rad.mean(), rad[tr].mean(), rad[~tr].mean()])
for k in variants:
    a=np.array(res[k]); w=np.array(wid[k])
    print(k,'cov all %.3f  trans %.3f±%.3f  steady %.3f | radius all %.3f trans %.3f steady %.3f'%(a[:,0].mean(),a[:,1].mean(),a[:,1].std(),a[:,2].mean(),w[:,0].mean(),w[:,1].mean(),w[:,2].mean()))

np.savez(RESULTS / "converter_normalization.npz", names=list(variants),
         cov=np.array([np.array(res[k]).mean(0) for k in variants]),   # columns: overall, transient, steady
         cov_sd=np.array([np.array(res[k]).std(0) for k in variants]),
         rad=np.array([np.array(wid[k]).mean(0) for k in variants]))

# save raw per-seed results so the plot script can draw them
np.savez(RESULTS / "converter_normalization.npz", names=list(variants),
         cov=np.array([res[k] for k in variants]), radius=np.array([wid[k] for k in variants]))
