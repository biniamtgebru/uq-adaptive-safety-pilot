from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle, sys, time
from pilot.converter import *
from pilot.common import Ensemble, conformal_q, quantile_level
m=pickle.load(open(ARTIFACTS/'converter_model.pkl','rb')); ens=m['ens']; cal=m['cal']
alpha=0.05
def nscores(Xs,Im,V,Yn):
    mu,_=ens.predict_batch(Xs); p0=np.array([f0(Im[k],V[k]) for k in range(len(Im))])
    dv=np.linalg.norm(V-E-R*Im,axis=1)
    return np.linalg.norm(Yn-(p0+mu),axis=1)/(1+dv/DV0)
Xc,Imc,Vc,Ync=run_open(30000,lambda k:1.0,seed=2000)
cal=nscores(Xc,Imc,Vc,Ync); q_static=conformal_q(cal,alpha)
kappa=float(sys.argv[2]); nseeds=int(sys.argv[1]); tag=sys.argv[3]
# oracle fixed margin: calibrated on shifted-regime scores (knowledge of the shift)
X,Im,V,Yn=run_open(20000,lambda k:kappa,seed=3000)
s_sh=nscores(X,Im,V,Yn); q_or=conformal_q(s_sh,alpha)
print('q_static',q_static,'q_oracle',q_or)
import pilot.converter as _s
_s.IREF=17.6   # stress test: references closer to the limit (slightly beyond training range)
OFF=int(sys.argv[4]) if len(sys.argv)>4 else 0
methods=['none','zero','fixed','aci','oracle']
Npre,Npost=15000,15000
out={mth:[] for mth in methods}
t=time.time()
for sd in range(nseeds):
    for mth in methods:
        r=closed_loop_filter(ens,mth,q_static,cal,q_or,seed=9000+OFF+sd,N_pre=Npre,N_post=Npost,kappa_shift=kappa,alpha=alpha)
        k0=r['k0']
        out[mth].append(dict(vpre=r['viol'][:k0].mean(),vpost=r['viol'][k0:].mean(),
            maxpost=r['inorm'][k0:].max(),trk_pre=np.sqrt((r['trk'][:k0]**2).mean()),trk_post=np.sqrt((r['trk'][k0:]**2).mean()),
            act_pre=r['act'][:k0].mean(),act_post=r['act'][k0:].mean(),q_pre=r['q'][:k0].mean(),q_post=r['q'][k0+1000:].mean()))
        if sd==0 and mth=='none': print('run time',time.time()-t)
    print('seed',sd,'done',time.time()-t,flush=True)
pickle.dump(out,open(RESULTS/f'converter_safety_{tag}.pkl','wb'))
print('kappa',kappa)
for mth in methods:
    d=out[mth]; g=lambda k:(np.mean([x[k] for x in d]),np.std([x[k] for x in d]))
    print('%-7s viol_pre %.4f viol_post %.4f max|i|_post %.2f | rms_trk pre %.3f post %.3f | act pre %.3f post %.3f | q pre %.3f post %.3f'%(
        mth,g('vpre')[0],g('vpost')[0],g('maxpost')[0],g('trk_pre')[0],g('trk_post')[0],g('act_pre')[0],g('act_post')[0],g('q_pre')[0],g('q_post')[0]))
