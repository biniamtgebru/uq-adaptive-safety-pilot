from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle, sys, time
from pilot.uav import *
from pilot.common import Ensemble, conformal_q, quantile_level
ens=pickle.load(open(ARTIFACTS/'uav_ensemble.pkl','rb'))
kappa=float(sys.argv[1]); normalized=sys.argv[2]=='1'; nseeds=int(sys.argv[3]); tag=sys.argv[4]
alpha=0.05; H=40
Xc,Vc,Ac,Yc=run(30000,lambda k:1.0,2000); cal,_=scores(ens,Xc,Vc,Ac,Yc,normalized); q_static=conformal_q(cal,alpha)
# monitor thresholds from a long nominal trajectory: at most 6 alarm events in 60k steps (about 1 per 10^4 steps)
Xn,Vn,An,Yn=run(60000,lambda k:1.0,2500); sn,_=scores(ens,Xn,Vn,An,Yn,normalized)
P=5
qa,ea,aa=aci_series(sn,cal,alpha)
st_f=np.convolve((sn>q_static).astype(float),np.ones(H)/H,mode='full')[:60000]
st_n=np.convolve(ea.astype(float),np.ones(H)/H,mode='full')[:60000]
st_a=alpha-aa
thr=dict(P=P,frozen=calibrate_threshold(st_f,P,6,H),naive=calibrate_threshold(st_n,P,6,H),alpha=calibrate_threshold(st_a,P,6,H))
print('q_static %.4f'%q_static,'thresholds',{k:round(float(v),3) for k,v in thr.items()})
R_=[]; t=time.time()
for sd in range(nseeds):
    R_.append(calib_exp(ens,q_static,cal,5000+sd,6000,6000,kappa,normalized,alpha=alpha,thr=thr))
    if sd==0: print('one run',time.time()-t)
def ms(key,j=None):
    a=np.array([r[key] if j is None else r[key][j] for r in R_],float); return np.nanmean(a),np.nanstd(a)
print('kappa',kappa,'normalized',normalized,'seeds',nseeds)
for j,nm in enumerate(['static','rolling','ACI']):
    print('%-7s pre %.3f±%.3f  pre_aggr %.3f±%.3f  first200 %.3f±%.3f  later %.3f±%.3f  width pre %.4f post %.4f'%(nm,*ms('pre',j),*ms('pre_aggr',j),*ms('first',j),*ms('later',j),ms('w_pre',j)[0],ms('w_post',j)[0]))
print('block-cov std %.3f vs iid %.3f'%(ms('blk_std')[0],R_[0]['blk_iid']))
for nm in ['frozen','naive','alpha']:
    d=np.array([r['d_'+nm] for r in R_]); fa=0
    print('monitor %-7s detect %d/%d  delay median %.2f s (IQR %.2f-%.2f)  false-alarm events/run %.2f  (runs with any: %d/%d)'%(nm,np.isfinite(d).sum(),nseeds,np.nanmedian(d),np.nanpercentile(d,25) if np.isfinite(d).any() else np.nan,np.nanpercentile(d,75) if np.isfinite(d).any() else np.nan,np.mean([x['fa_'+nm] for x in R_]),sum(1 for x in R_ if x['fa_'+nm]>0),nseeds))
np.savez(RESULTS/f'uav_trace_{tag}.npz',d_frozen=[x['d_frozen'] for x in R_],d_naive=[x['d_naive'] for x in R_],d_alpha=[x['d_alpha'] for x in R_],static=np.mean([r['e_static'] for r in R_],0),aci=np.mean([r['e_aci'] for r in R_],0),roll=np.mean([r['e_roll'] for r in R_],0),qaci=np.mean([r['q_aci'] for r in R_],0),qstatic=q_static)
