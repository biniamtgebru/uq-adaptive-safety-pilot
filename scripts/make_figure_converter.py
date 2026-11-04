"""Figure for the dq inverter current-loop pilot (toy model).
Needs: artifacts/converter_model.pkl (run converter_train.py), results/converter_trace_derate.npz
(run converter_run.py 30 0.6 derate) and results/converter_normalization.npz (run converter_normalization.py)."""
from _paths import ARTIFACTS, RESULTS, FIGURES
import pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pilot.converter import run_trace, calibration_experiment, Ts
from pilot.common import conformal_q

KAPPA, N_PRE, N_POST, SEED, ALPHA = 0.6, 20000, 20000, 5000, 0.05
k0 = N_PRE
m = pickle.load(open(ARTIFACTS / "converter_model.pkl", "rb")); ens, cal = m["ens"], m["cal"]
q_static = conformal_q(cal, ALPHA)

# single-seed run (the same trajectory used by the calibration experiment for this seed)
I, REF, V = run_trace(N_PRE + N_POST, lambda k: 1.0 if k < k0 else KAPPA, SEED)
r = calibration_experiment(ens, q_static, cal, seed=SEED, N_pre=N_PRE, N_post=N_POST, kappa_shift=KAPPA, alpha=ALPHA)
t = (np.arange(N_PRE + N_POST) - k0) * Ts * 1000.0       # ms from shift onset

tr = np.load(RESULTS / "converter_trace_derate.npz")
nm = np.load(RESULTS / "converter_normalization.npz", allow_pickle=True)

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.2))

# (a) dq currents around the shift
w = (t > -120) & (t < 120)
ax[0, 0].plot(t[w], I[w, 0], color="tab:blue", lw=0.9, label="$i_d$")
ax[0, 0].plot(t[w], I[w, 1], color="tab:red", lw=0.9, label="$i_q$")
ax[0, 0].plot(t[w], REF[w, 0], color="tab:blue", lw=0.7, ls="--", label="$i_d^*$")
ax[0, 0].plot(t[w], REF[w, 1], color="tab:red", lw=0.7, ls="--", label="$i_q^*$")
ax[0, 0].axvline(0, color="gray", lw=0.8)
ax[0, 0].set_xlabel("time from shift onset (ms)"); ax[0, 0].set_ylabel("dq current (A)")
ax[0, 0].set_title("(a) dq currents, effective inductance $\\times$0.6 at t = 0", fontsize=7)
ax[0, 0].set_ylim(-14, 18)
ax[0, 0].legend(ncol=4, fontsize=6, frameon=False, loc="upper center")

# (b) one-step prediction-error score and the envelopes
w2 = (t > -120) & (t < 400)
ax[0, 1].semilogy(t[w2], r["s"][w2], color="0.6", lw=0.5, label="score $s_k$")
ax[0, 1].axhline(q_static, color="tab:red", lw=1.0, label="static envelope")
ax[0, 1].semilogy(t[w2], np.maximum(r["q_aci"][w2], 1e-4), color="tab:blue", lw=1.0, label="ACI envelope")
ax[0, 1].axvline(0, color="gray", lw=0.8)
ax[0, 1].set_xlabel("time from shift onset (ms)"); ax[0, 1].set_ylabel("one-step error norm (A)")
ax[0, 1].set_title("(b) prediction error and conformal radii (one seed)", fontsize=7)
ax[0, 1].legend(fontsize=6, frameon=False, loc="lower right")

# (c) coverage over time, mean over seeds (window 500 steps = 50 ms)
W = 500
roll = lambda e: 1 - np.convolve(e, np.ones(W) / W, mode="full")[:len(e)]
w3 = (t > -300) & (t < 1500)
for key, lab, c in [("static", "static split", "tab:red"), ("roll", "rolling window", "tab:orange"), ("aci", "ACI", "tab:blue")]:
    ax[1, 0].plot(t[w3], roll(tr[key])[w3], color=c, lw=1.0, label=lab)
ax[1, 0].axhline(0.95, color="k", ls=":", lw=0.6); ax[1, 0].axvline(0, color="gray", lw=0.8)
ax[1, 0].set_ylim(0, 1.03)
ax[1, 0].set_xlabel("time from shift onset (ms)"); ax[1, 0].set_ylabel("coverage (50 ms window)")
ax[1, 0].set_title("(c) coverage, mean of 30 seeds", fontsize=7)
ax[1, 0].legend(fontsize=6, frameon=False, loc="lower right")

# (d) marginal vs transient coverage for score normalizations
names = [str(x) for x in nm["names"]]
lab = {"plain": "plain", "ens-std": "ensemble-std\nnormalized", "phys (dv)": "voltage-step\nnormalized"}
cov, sd = nm["cov"], nm["cov_sd"]
x = np.arange(len(names)); b = 0.36
ax[1, 1].bar(x - b / 2, cov[:, 0], b, label="all steps", color="tab:gray")
ax[1, 1].bar(x + b / 2, cov[:, 1], b, yerr=sd[:, 1], capsize=2, label="voltage-transient steps", color="tab:purple", error_kw={"lw": 0.6})
ax[1, 1].axhline(0.95, color="k", ls=":", lw=0.6)
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([lab.get(n, n) for n in names], fontsize=6.5)
ax[1, 1].set_ylim(0.5, 1.12); ax[1, 1].set_ylabel("coverage before shift")
ax[1, 1].set_title("(d) marginal vs conditional coverage", fontsize=7)
ax[1, 1].legend(fontsize=6, frameon=False, loc="upper left", ncol=2)

for a in ax.ravel(): a.tick_params(width=0.6, length=2.5)
plt.tight_layout(pad=0.6)
plt.savefig(FIGURES / "fig_converter.pdf"); plt.savefig(FIGURES / "fig_converter.png", dpi=200)
print("wrote", FIGURES / "fig_converter.png")
