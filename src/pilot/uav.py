"""
UAV pilot (toy model). Horizontal translational loop of a multirotor at 20 Hz.
v_{k+1} = v_k + Ts( kappa * a_k - cd*|v|*v + w_k ),  w = Ornstein-Uhlenbeck gust (tau = 1 s)
Nominal model: v_{k+1} = v_k + Ts a_k  (no drag, no gust, kappa = 1).
Shift: thrust effectiveness kappa drops (air density / altitude, rotor wear).
Learned residual: ensemble of MLPs on (v, a) trained at kappa = 1.
"""
import numpy as np
from pilot.common import Ensemble, conformal_q, quantile_level

Ts = 0.05
cd = 0.2
tau, sig_w = 1.0, 0.25
sig_m = 0.01
Kp, Kd = 1.0, 1.8
AMAX = 6.0
HOLD = 120                    # waypoint hold: 6 s
V0, A0 = 3.0, 3.0             # normalization scales


def run(n, kappa_fn, seed):
    rng = np.random.default_rng(seed)
    p = np.zeros(2); v = np.zeros(2); w = rng.normal(0, sig_w, 2)
    ref = rng.uniform(-20, 20, 2)
    X = np.zeros((n, 4)); Y = np.zeros((n, 2)); Vm = np.zeros((n, 2)); A = np.zeros((n, 2))
    a_ou = np.sqrt(2 * Ts / tau)
    for k in range(n):
        if k % HOLD == 0:
            ref = rng.uniform(-20, 20, 2)
        vm = v + rng.normal(0, sig_m, 2)
        a = Kp * (ref - p) - Kd * vm
        na = np.linalg.norm(a)
        if na > AMAX:
            a = a * AMAX / na
        vn = v + Ts * (kappa_fn(k) * a - cd * np.linalg.norm(v) * v + w)
        p = p + Ts * v
        w = w * (1 - Ts / tau) + sig_w * a_ou * rng.normal(0, 1, 2)
        v = vn
        X[k] = np.concatenate([vm, a]); A[k] = a; Vm[k] = vm
        Y[k] = vn + rng.normal(0, sig_m, 2)
    return X, Vm, A, Y


def targets(Vm, A, Y):
    return Y - (Vm + Ts * A)


def scale(Vm, A):
    return 1.0 + np.linalg.norm(Vm, axis=1) / V0 + np.linalg.norm(A, axis=1) / A0


def scores(ens, X, Vm, A, Y, normalized):
    mu, sd = ens.predict_batch(X)
    s = np.linalg.norm(targets(Vm, A, Y) - mu, axis=1)
    return (s / scale(Vm, A) if normalized else s), sd


def calib_exp(ens, q_static, cal, seed, N_pre, N_post, kappa, normalized, alpha=0.05, W=400, gamma=0.01,
              H=40, thr=None):
    k0 = N_pre; n = N_pre + N_post
    X, Vm, A, Y = run(n, lambda k: 1.0 if k < k0 else kappa, seed)
    s, _ = scores(ens, X, Vm, A, Y, normalized)
    hist = np.concatenate([cal[-W:], s])
    e_static = s > q_static
    q_roll = np.zeros(n); q_aci = np.zeros(n); e_roll = np.zeros(n, bool); e_aci = np.zeros(n, bool); a_hist = np.zeros(n)
    ak = alpha
    for k in range(n):
        win = hist[k:k + W]
        q_roll[k] = np.quantile(win, quantile_level(W, alpha), method='higher'); e_roll[k] = s[k] > q_roll[k]
        lev = 1 - ak
        q_aci[k] = win.max() if lev >= 1 else (0.0 if lev <= 0 else np.quantile(win, lev, method='higher'))
        e_aci[k] = s[k] > q_aci[k]; a_hist[k] = ak
        ak = ak + gamma * (alpha - e_aci[k])
    cov = lambda e, a, b: 1 - e[a:b].mean()
    # strata: aggressive-maneuver steps (pre-shift only)
    aggr = np.zeros(n, bool); aggr[:k0] = (np.linalg.norm(A, axis=1)[:k0] > 3.0) | (np.linalg.norm(Vm, axis=1)[:k0] > 5.0)
    trc = lambda e: 1 - e[aggr].mean() if aggr.sum() > 20 else np.nan
    # block variability of static coverage pre-shift (40-step = 2 s blocks)
    nb = (k0 - 2000) // H
    blk = np.array([1 - e_static[2000 + j * H:2000 + (j + 1) * H].mean() for j in range(nb)])
    # monitors (thresholds supplied in thr dict, calibrated on a long nominal trajectory)
    roll_static = np.convolve(e_static.astype(float), np.ones(H) / H, mode='full')[:n]
    roll_aci = np.convolve(e_aci.astype(float), np.ones(H) / H, mode='full')[:n]
    stats = dict(frozen=roll_static, naive=roll_aci, alpha=(alpha - a_hist))
    mon = {}
    for nm, st in stats.items():
        idx, hit = first_alarm(st, thr[nm], thr['P'], start=H)
        pre_ev = n_events(hit[:k0])
        post = np.where(hit[k0:])[0]
        mon[nm] = (post[0] * Ts if len(post) else np.nan, pre_ev)
    return dict(
        pre=[cov(e_static, 1000, k0), cov(e_roll, 1000, k0), cov(e_aci, 1000, k0)],
        pre_aggr=[trc(e_static), trc(e_roll), trc(e_aci)], n_aggr=int(aggr.sum()),
        first=[cov(e_static, k0, k0 + 200), cov(e_roll, k0, k0 + 200), cov(e_aci, k0, k0 + 200)],
        later=[cov(e_static, k0 + 200, n), cov(e_roll, k0 + 200, n), cov(e_aci, k0 + 200, n)],
        w_pre=[q_static, q_roll[1000:k0].mean(), q_aci[1000:k0].mean()],
        w_post=[q_static, q_roll[k0 + 200:].mean(), q_aci[k0 + 200:].mean()],
        blk_std=blk.std(), blk_iid=np.sqrt(alpha * (1 - alpha) / H),
        d_frozen=mon['frozen'][0], d_naive=mon['naive'][0], d_alpha=mon['alpha'][0],
        fa_frozen=mon['frozen'][1], fa_naive=mon['naive'][1], fa_alpha=mon['alpha'][1],
        e_static=e_static, e_aci=e_aci, e_roll=e_roll, q_aci=q_aci, s=s)


def aci_series(s, cal, alpha=0.05, W=400, gamma=0.01):
    n = len(s); hist = np.concatenate([cal[-W:], s]); ak = alpha
    q = np.zeros(n); e = np.zeros(n, bool); a_hist = np.zeros(n)
    for k in range(n):
        win = hist[k:k + W]; lev = 1 - ak
        q[k] = win.max() if lev >= 1 else (0.0 if lev <= 0 else np.quantile(win, lev, method='higher'))
        e[k] = s[k] > q[k]; a_hist[k] = ak
        ak = ak + gamma * (alpha - e[k])
    return q, e, a_hist


def first_alarm(stat, thr, P, start=0):
    above = stat > thr
    run_ = np.convolve(above.astype(int), np.ones(P, int), mode='full')[:len(stat)]
    hit = run_ >= P
    hit[:start] = False
    idx = np.where(hit)[0]
    return (idx[0] if len(idx) else None), hit


def n_events(hit):
    return int(((hit.astype(int)[1:] - hit.astype(int)[:-1]) == 1).sum() + (1 if hit[0] else 0))


def calibrate_threshold(stat, P, max_events, start=0):
    cands = np.unique(np.quantile(stat[start:], np.linspace(0.5, 1.0, 400)))
    for c in cands:
        _, hit = first_alarm(stat, c, P, start)
        if n_events(hit) <= max_events:
            return c
    return cands[-1]
