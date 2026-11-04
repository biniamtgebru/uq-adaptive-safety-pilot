"""Fast end-to-end sanity check (about 1 minute). Uses small data and few seeds, so numbers are rough;
it only verifies that the pipeline runs and shows the expected qualitative behaviour."""
from _paths import ARTIFACTS, RESULTS
import numpy as np
from pilot.common import Ensemble, conformal_q
import pilot.uav as U
import pilot.converter as C

# ---- UAV: small ensemble, short runs
X, Vm, A, Y = U.run(15000, lambda k: 1.0, 1000)
ens = Ensemble(M=2, hidden=(16, 16), seed=0); ens.fit(X, U.targets(Vm, A, Y))
Xc, Vc, Ac, Yc = U.run(8000, lambda k: 1.0, 2000)
cal, _ = U.scores(ens, Xc, Vc, Ac, Yc, False)
q = conformal_q(cal, 0.05)
thr = dict(P=5, frozen=0.3, naive=0.2, alpha=0.05)
res = [U.calib_exp(ens, q, cal, 5000 + s, 3000, 3000, 0.79, False, thr=thr) for s in range(3)]
pre = np.mean([r["pre"][0] for r in res]); st = np.mean([r["later"][0] for r in res]); aci = np.mean([r["later"][2] for r in res])
print(f"UAV  pre-shift static coverage {pre:.3f} | post-shift static {st:.3f} vs ACI {aci:.3f}")
assert 0.88 < pre < 1.0, "pre-shift coverage should be near 0.95"
assert aci > st + 0.1, "ACI should recover coverage that the static envelope loses"

# ---- Converter: tiny closed-loop run
Xk, Imk, Vk, Ynk = C.run_open(4000, lambda k: 1.0, 1)
D, _ = C.residual_targets(Imk, Vk, Ynk)
assert np.isfinite(D).all() and np.abs(Imk).max() < 40, "converter loop should stay bounded"
print("converter closed loop OK; max |i| = %.1f A" % np.abs(Imk).max())
print("SMOKE TEST PASSED")
