"""Build the pilot figure (coverage around the shift; monitor detection delay) from results/uav_trace_plain79.npz."""
from _paths import RESULTS, FIGURES
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(RESULTS / "uav_trace_plain79.npz")
Ts, k0, H = 0.05, 6000, 40
roll = lambda x: 1 - np.convolve(x, np.ones(H) / H, mode="full")[:len(x)]
t = (np.arange(12000) - k0) * Ts
plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5})
fig, ax = plt.subplots(1, 2, figsize=(7.1, 1.3), gridspec_kw={"width_ratios": [1.6, 1]})
m = (t > -20) & (t < 45)
for key, lab, c in [("static", "static split", "tab:red"), ("roll", "rolling window", "tab:orange"), ("aci", "ACI", "tab:blue")]:
    ax[0].plot(t[m], roll(d[key])[m], label=lab, color=c, lw=0.9)
ax[0].axhline(0.95, color="k", ls=":", lw=0.6); ax[0].axvline(0, color="gray", lw=0.6)
ax[0].set_xlabel("time from shift onset (s)", labelpad=1); ax[0].set_ylabel("coverage (2 s window)"); ax[0].set_ylim(0.2, 1.02)
ax[0].legend(loc="lower left", frameon=False, fontsize=5.8, labelspacing=0.2)
ax[0].set_title("(a) UAV pilot: coverage after thrust effectiveness 1.0 $\\to$ 0.79 (mean of 30 seeds)", fontsize=6.4, pad=2)
names = ["frozen-envelope\nexceedance", "ACI $\\alpha_k$\ndrop", "ACI own\ncoverage"]
dd = [d["d_frozen"], d["d_alpha"], d["d_naive"]]
med = [np.nanmedian(x) for x in dd]; lo = [np.nanpercentile(x, 25) for x in dd]; hi = [np.nanpercentile(x, 75) for x in dd]
ax[1].bar(range(3), med, color=["tab:green", "tab:green", "tab:gray"],
          yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], capsize=2, error_kw={"lw": 0.6})
ax[1].set_yscale("log"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, fontsize=5.4); ax[1].set_ylabel("detection delay (s)")
ax[1].set_title("(b) monitor statistic (median, IQR)", fontsize=6.4, pad=2)
for a in ax: a.tick_params(width=0.5, length=2, pad=1.5)
plt.tight_layout(pad=0.25)
plt.savefig(FIGURES / "fig_pilot.pdf"); plt.savefig(FIGURES / "fig_pilot.png", dpi=200)
print("wrote", FIGURES / "fig_pilot.pdf")
