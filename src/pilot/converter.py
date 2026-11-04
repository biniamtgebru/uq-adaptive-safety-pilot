"""
Pilot study (toy model, NOT the host's converter model).
dq-frame converter-side current loop with saturating inductance.
Learned residual = ensemble of small MLPs trained on nominal data.
Shift = effective-inductance change at step k0.
Part 1: online calibration (static split conformal / rolling / ACI) + validity monitor.
Part 2: discrete-time barrier (current-limit) filter with different margins.
"""
import numpy as np
from pilot.common import Ensemble, conformal_q, quantile_level

# ---------------- plant parameters ----------------
Ts = 1e-4
L0 = 3e-3
R = 0.1
w0 = 2 * np.pi * 50
E = np.array([325.0, 0.0])
J = np.array([[0.0, 1.0], [-1.0, 0.0]])
beta, Is = 1.0, 20.0          # inductance saturation
Vmax = 400.0
sig_meas, sig_v = 0.02, 0.5   # A, V
Kp = L0 * 2 * np.pi * 600
Ki = Kp * R / L0
IREF = 16.0                   # max reference magnitude (A)
ILIM = 18.0                   # current limit (A)
DV0 = 20.0                    # voltage-step scale for normalized scores
HOLD = 150                    # reference hold steps


def Leff(i, kappa):
    return kappa * L0 / (1.0 + beta * (i @ i) / Is ** 2)


def f0(i, v):
    return i + Ts * ((v - E - R * i) / L0 + w0 * (J @ i))


def plant_step(i, v, kappa, rng):
    di = (v - E - R * i) / Leff(i, kappa) + w0 * (J @ i)
    return i + Ts * di + rng.normal(0, sig_v * Ts / L0, 2)


def new_ref(rng):
    mag = rng.uniform(0, IREF)
    ang = rng.uniform(0, 2 * np.pi)
    return mag * np.array([np.cos(ang), np.sin(ang)])


class PI:
    def __init__(self):
        self.integ = np.zeros(2)

    def __call__(self, im, ref):
        err = ref - im
        self.integ = np.clip(self.integ + err * Ts, -200 / Ki, 200 / Ki)
        u = Kp * err + Ki * self.integ
        v = E + R * im - w0 * L0 * (J @ im) + u
        n = np.linalg.norm(v)
        if n > Vmax:
            v = v * Vmax / n
        return v


def feat(im, v):
    return np.concatenate([im, (v - E) / 100.0])


def run_open(n, kappa_fn, seed, ens=None):
    """Closed loop with PI only; returns features X, next-measured current, prior measured etc."""
    rng = np.random.default_rng(seed)
    i = np.zeros(2)
    ctl = PI()
    ref = new_ref(rng)
    X = np.zeros((n, 4)); Yn = np.zeros((n, 2)); Im = np.zeros((n, 2)); It = np.zeros((n, 2)); V = np.zeros((n, 2))
    for k in range(n):
        if k % HOLD == 0:
            ref = new_ref(rng)
        im = i + rng.normal(0, sig_meas, 2)
        v = ctl(im, ref)
        inext = plant_step(i, v, kappa_fn(k), rng)
        X[k] = feat(im, v); Im[k] = im; V[k] = v; It[k] = i
        Yn[k] = inext + rng.normal(0, sig_meas, 2)   # measured next current (independent meas. noise draw is fine)
        i = inext
    return X, Im, V, Yn


def run_trace(n, kappa_fn, seed):
    """Same trajectory as run_open(seed) but also returns true currents, references and voltage commands (for plotting)."""
    rng = np.random.default_rng(seed)
    i = np.zeros(2); ctl = PI(); ref = new_ref(rng)
    I = np.zeros((n, 2)); REF = np.zeros((n, 2)); V = np.zeros((n, 2))
    for k in range(n):
        if k % HOLD == 0:
            ref = new_ref(rng)
        im = i + rng.normal(0, sig_meas, 2)
        v = ctl(im, ref)
        inext = plant_step(i, v, kappa_fn(k), rng)
        rng.normal(0, sig_meas, 2)            # keep RNG stream aligned with run_open
        I[k] = i; REF[k] = ref; V[k] = v
        i = inext
    return I, REF, V


def residual_targets(Im, V, Yn):
    pred0 = np.array([f0(Im[k], V[k]) for k in range(len(Im))])
    return Yn - pred0, pred0


# ---------------- Part 1 ----------------
def calibration_experiment(ens, q_static, cal_scores, seed, N_pre, N_post, kappa_shift, alpha=0.05,
                           W=1000, gamma=0.01, mon_N=200, mon_thr=0.25):
    k0 = N_pre
    n = N_pre + N_post
    X, Im, V, Yn = run_open(n, lambda k: 1.0 if k < k0 else kappa_shift, seed)
    rres_pred, _ = ens.predict_batch(X)
    pred0 = np.array([f0(Im[k], V[k]) for k in range(n)])
    s = np.linalg.norm(Yn - (pred0 + rres_pred), axis=1)
    # frozen-weights-prediction (no residual learning) for reference
    out = {}
    hist = np.concatenate([cal_scores[-W:], s])
    err_static = s > q_static
    q_roll = np.zeros(n); q_aci = np.zeros(n); err_roll = np.zeros(n, bool); err_aci = np.zeros(n, bool)
    a_k = alpha; sat = 0
    for k in range(n):
        win = hist[k:k + W]
        q_roll[k] = np.quantile(win, quantile_level(W, alpha), method='higher')
        err_roll[k] = s[k] > q_roll[k]
        lev = 1 - a_k
        if lev >= 1.0:
            q_aci[k] = win.max(); sat += 1
        elif lev <= 0:
            q_aci[k] = 0.0
        else:
            q_aci[k] = np.quantile(win, lev, method='higher')
        err_aci[k] = s[k] > q_aci[k]
        a_k = a_k + gamma * (alpha - err_aci[k])
    # monitor on frozen reference envelope
    roll = np.convolve(err_static.astype(float), np.ones(mon_N) / mon_N, mode='full')[:n]
    alarm = roll > mon_thr
    alarm[:mon_N] = False
    pre_alarms = int(alarm[:k0].sum() > 0)
    post = np.where(alarm[k0:])[0]
    delay = post[0] * Ts * 1000 if len(post) else np.nan
    cov = lambda e, a, b: 1 - e[a:b].mean()
    dv = np.linalg.norm(V - E - R * Im, axis=1)
    tr = np.zeros(n, bool); tr[:k0] = dv[:k0] > 20.0     # pre-shift transient steps
    trc = lambda e: 1 - e[tr].mean() if tr.sum() > 20 else np.nan
    res = dict(
        pre_trans=[trc(err_static), trc(err_roll), trc(err_aci)], n_trans=int(tr.sum()),
        pre=[cov(err_static, k0 - 10000, k0), cov(err_roll, k0 - 10000, k0), cov(err_aci, k0 - 10000, k0)],
        first=[cov(err_static, k0, k0 + 1000), cov(err_roll, k0, k0 + 1000), cov(err_aci, k0, k0 + 1000)],
        later=[cov(err_static, k0 + 1000, n), cov(err_roll, k0 + 1000, n), cov(err_aci, k0 + 1000, n)],
        width_pre=[q_static, q_roll[k0 - 10000:k0].mean(), q_aci[k0 - 10000:k0].mean()],
        width_post=[q_static, q_roll[k0 + 1000:].mean(), q_aci[k0 + 1000:].mean()],
        delay=delay, false_alarm=pre_alarms, sat_frac=sat / n,
        cov_err_static=err_static, cov_err_roll=err_roll, cov_err_aci=err_aci,
        q_aci=q_aci, q_roll=q_roll, s=s)
    return res


# ---------------- Part 2: barrier filter ----------------
def closed_loop_filter(ens, method, q_static, cal_scores, q_oracle, seed, N_pre, N_post, kappa_shift,
                       alpha=0.05, W=1000, gamma=0.01):
    k0 = N_pre; n = N_pre + N_post
    rng = np.random.default_rng(seed)
    i = np.zeros(2); ctl = PI(); ref = new_ref(rng)
    hist = list(cal_scores[-W:])
    a_k = alpha
    viol = np.zeros(n, bool); trk = np.zeros(n); inorm = np.zeros(n); qs = np.zeros(n); act = np.zeros(n, bool)
    for k in range(n):
        if k % HOLD == 0:
            ref = new_ref(rng)
        kap = 1.0 if k < k0 else kappa_shift
        im = i + rng.normal(0, sig_meas, 2)
        v = ctl(im, ref)
        # margin
        if method == 'none':
            q = None
        elif method == 'zero':
            q = 0.0
        elif method == 'fixed':
            q = q_static
        elif method == 'oracle':
            q = q_oracle
        else:  # aci
            win = np.asarray(hist[-W:])
            lev = 1 - a_k
            if lev >= 1.0:
                q = win.max()
            elif lev <= 0:
                q = 0.0
            else:
                q = np.quantile(win, lev, method='higher')
        scale0 = 1.0 + np.linalg.norm(v - E - R * im) / DV0
        q = None if q is None else q * scale0
        qs[k] = 0 if q is None else q
        # prediction at nominal command
        rhat = ens.predict_all(feat(im, v)).mean(0)
        if q is not None:
            a = f0(im, v) + rhat
            rho = max(ILIM - q, 0.0)
            na = np.linalg.norm(a)
            if na > rho:
                act[k] = True
                v = v - (L0 / Ts) * (a - a * rho / na)
                nv = np.linalg.norm(v)
                if nv > Vmax:
                    v = v * Vmax / nv
        # predicted next current at applied command (for score)
        rhat2 = ens.predict_all(feat(im, v)).mean(0)
        pred = f0(im, v) + rhat2
        inext = plant_step(i, v, kap, rng)
        ymeas = inext + rng.normal(0, sig_meas, 2)
        scale1 = 1.0 + np.linalg.norm(v - E - R * im) / DV0
        s = np.linalg.norm(ymeas - pred) / scale1
        if method == 'aci':
            e = s > (q / scale0)
            a_k = a_k + gamma * (alpha - e)
        hist.append(s)
        if len(hist) > 5 * W:
            hist = hist[-W:]
        i = inext
        viol[k] = np.linalg.norm(i) > ILIM
        trk[k] = np.linalg.norm(i - ref)
        inorm[k] = np.linalg.norm(i)
    return dict(viol=viol, trk=trk, inorm=inorm, q=qs, act=act, k0=k0)
