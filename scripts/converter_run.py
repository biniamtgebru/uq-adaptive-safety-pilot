from _paths import ARTIFACTS, RESULTS, FIGURES
import numpy as np, pickle, sys, time
from pilot.converter import *
from pilot.common import Ensemble, conformal_q, quantile_level
m=pickle.load(open(ARTIFACTS/'converter_model.pkl','rb')); ens=m['ens']; cal=m['cal']
alpha=0.05
q_static=conformal_q(cal,alpha)
print('q_static',q_static)
nseeds=int(sys.argv[1]); kappa=float(sys.argv[2]); tag=sys.argv[3]
R_=[]
t=time.time()
for sd in range(nseeds):
    r=calibration_experiment(ens,q_static,cal,seed=5000+sd,N_pre=20000,N_post=20000,kappa_shift=kappa,alpha=alpha)
    R_.append(r)
    if sd==0: print('one run time',time.time()-t)
def ms(key,idx=None):
    a=np.array([r[key] if idx is None else r[key][idx] for r in R_],float)
    return np.nanmean(a),np.nanstd(a)
names=['static','rolling','ACI']
print('kappa',kappa,'seeds',nseeds)
for j,nm in enumerate(names):
    print(nm, 'pre_trans %.3f±%.3f'%ms('pre_trans',j), 'pre %.3f±%.3f'%ms('pre',j), 'first1000 %.3f±%.3f'%ms('first',j), 'later %.3f±%.3f'%ms('later',j),
          'width_pre %.3f'%ms('width_pre',j)[0],'width_post %.3f'%ms('width_post',j)[0])
print('delay ms %.1f±%.1f'%ms('delay'),'nan',np.isnan([r['delay'] for r in R_]).sum(),'false alarm runs',sum(r['false_alarm'] for r in R_),'ACI sat frac %.3f'%ms('sat_frac')[0])
pickle.dump({k:[r[k] for r in R_] for k in ['pre','pre_trans','n_trans','first','later','width_pre','width_post','delay','false_alarm','sat_frac']},open(RESULTS/f'converter_cal_{tag}.pkl','wb'))
# save mean running coverage traces for figure
n=40000;k0=20000
def trace(key):
    return np.mean([np.array(r[key],float) for r in R_],axis=0)
np.savez(RESULTS/f'converter_trace_{tag}.npz',static=trace('cov_err_static'),roll=trace('cov_err_roll'),aci=trace('cov_err_aci'),qaci=trace('q_aci'),qroll=trace('q_roll'),
         s=np.mean([r['s'] for r in R_],axis=0), q_static=q_static)
