"""The structure of the allocation, drawn."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply()

d = json.load(open(R / "results/tile_table.json"))
gap = json.load(open(R / "results/hull_gap.json"))
D = np.array(d["D"])[:, d["j"]:]
cost = np.array(d["cost"])[d["j"]:]
ref = d["ref_mse"]
T = D.shape[0]
db = lambda m: 10 * np.log10(m / ref)

lams = np.concatenate([[0.0], np.geomspace(1e-9, 1e-2, 3000)])
C_, Dd = [], []
for lam in lams:
    k = np.argmin(D + lam * cost[None, :], axis=1)
    C_.append(cost[k].mean()); Dd.append(D[np.arange(T), k].mean())
C_, Dd = np.array(C_), np.array(Dd)
sv = 100 * (1 - C_)
dbv = db(Dd)

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.3))

# a: monotonicity in the multiplier
ax[0].semilogx(np.clip(lams, 1e-9, None), sv, color=ns.BLUE, lw=1.3)
a2 = ax[0].twinx()
a2.semilogx(np.clip(lams, 1e-9, None), dbv, color=ns.VERM, lw=1.3)
a2.set_ylabel("dB", color=ns.VERM); a2.tick_params(axis="y", colors=ns.VERM)
a2.grid(False)
ax[0].set_xlabel("λ"); ax[0].set_ylabel("MACs saved (%)", color=ns.BLUE)
ax[0].tick_params(axis="y", colors=ns.BLUE)
ns.panel(ax[0], "a")

# b: the frontier with both ends marked
ax[1].plot(dbv, sv, "-", color=ns.INK, lw=1.4)
ax[1].plot([dbv[0]], [sv[0]], "o", ms=6, color=ns.GREEN)
ax[1].plot([dbv[-1]], [sv[-1]], "o", ms=6, color=ns.ORANGE)
ax[1].axvspan(dbv.min() - 0.02, dbv[0], color=ns.VERM, alpha=0.10, lw=0)
ax[1].axvspan(dbv[-1], dbv.max() + 0.05, color="#bbbbbb", alpha=0.22, lw=0)
ax[1].set_xlabel("dB"); ax[1].set_ylabel("MACs saved (%)")
ax[1].set_xlim(dbv[0] - 0.02, dbv[-1] + 0.04)
ns.panel(ax[1], "b", dx=-0.24)

# c: what convexity costs, against the exact Pareto set
rows = gap["rows"]
b_ = [r["budget_db"] for r in rows]
ax[2].plot(b_, [r["pareto_saving"] for r in rows], "-o", ms=3.5, lw=1.1,
           color=ns.PURPLE, label="exact Pareto")
ax[2].plot(b_, [r["hull_saving"] for r in rows], "--s", ms=3.5, lw=1.1,
           color=ns.BLUE, label="reachable by λ")
ax[2].set_xlabel("budget (dB)"); ax[2].set_ylabel("MACs saved (%)")
ax[2].legend(fontsize=5.5, loc="lower right")
ns.panel(ax[2], "c", dx=-0.24)

fig.tight_layout()
fig.savefig(R / "docs/figures/theory.png", dpi=300)
print("  -> docs/figures/theory.png")
print(f"    floor {dbv[0]:.4f} dB at {sv[0]:.2f}%   "
      f"saturation {dbv[-1]:.4f} dB at {sv[-1]:.2f}%")
