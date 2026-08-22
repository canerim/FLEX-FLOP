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
ns.apply(ns.for_column())
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

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.6))

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
# Two shaded bands and two dots, and nothing on the panel said which was which:
# the caption read "shaded regions are infeasible or wasted" and left the reader
# to guess the order. Label them where they are.
_lo, _hi = ax[1].get_ylim() if ax[1].get_ylim()[1] > 0 else (0, sv.max())
_ytxt = sv.max() * 0.52
ax[1].text(dbv[0] - 0.008, _ytxt, "infeasible", rotation=90, ha="center",
           va="center", fontsize=ns.fs(6), color=ns.VERM)
ax[1].text(dbv[-1] + 0.018, _ytxt, "wasted", rotation=90, ha="center",
           va="center", fontsize=ns.fs(6), color=ns.INK2)
ax[1].annotate("floor", (dbv[0], sv[0]), textcoords="offset points",
               xytext=(9, 4), fontsize=ns.fs(6), color=ns.GREEN)
ax[1].annotate("saturation", (dbv[-1], sv[-1]), textcoords="offset points",
               xytext=(-44, 7), fontsize=ns.fs(6), color=ns.ORANGE)
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
ax[2].legend(fontsize=ns.fs(6), loc="lower right")
ns.panel(ax[2], "c", dx=-0.24)

fig.tight_layout(w_pad=2.2, h_pad=1.2)
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "theory.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
print("  -> docs/figures/theory.png, paper/figures/theory.png")
print(f"    floor {dbv[0]:.4f} dB at {sv[0]:.2f}%   "
      f"saturation {dbv[-1]:.4f} dB at {sv[-1]:.2f}%")
