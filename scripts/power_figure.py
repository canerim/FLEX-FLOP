"""Three units for one saving: arithmetic, time, energy.

The point of the figure is the gap between the bars, not their height. A
multiply-accumulate count is what the method optimises; seconds are what a
deployment feels; joules are what it pays for. They do not agree, and the
ordering is always the same.

    python scripts/power_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply(ns.for_column())
P = json.loads((ROOT / "results/supp_power.json").read_text())
F = json.loads((ROOT / "results/supp_footprint.json").read_text())

rows = [r for r in P["rows"] if r["size"] == "2048x1280"]
qps = [r["qp"] for r in rows]
mac = [r["mac_saving_pct"] for r in rows]
tim = [r["time_saving_pct"] for r in rows]
eng = [r["energy_saving_pct"] for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 1.62),
                       gridspec_kw={"width_ratios": [1.25, 1.0, 1.0],
                                    "wspace": 0.34})

# (a) the three units side by side
x = np.arange(len(qps)); w = 0.26
for k, (v, c, lab) in enumerate([(mac, ns.BLUE, "MACs"),
                                 (tim, ns.ORANGE, "wall clock"),
                                 (eng, ns.GREEN, "energy")]):
    ax[0].bar(x + (k - 1) * w, v, w, color=c, label=lab, lw=0)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"q{q}" for q in qps])
ax[0].set_ylabel("saved (%)")
ax[0].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.1, borderpad=0,
             loc="upper right")
ax[0].set_ylim(0, 40)
ns.panel(ax[0], "a")
ax[0].set_title("1080p, 0.1 dB budget", fontsize=ns.fs(6), color=ns.INK2, loc="left")

# (b) why: the board never gets cooler
pf = [r["full"]["watts_mean"] for r in rows]
pr = [r["routed"]["watts_mean"] for r in rows]
ax[1].plot(x, pf, "o-", color=ns.INK2, ms=2.4, lw=0.9, label="full decode")
ax[1].plot(x, pr, "s--", color=ns.VERM, ms=2.4, lw=0.9, label="routed")
ax[1].axhline(P["power_limit_w"], color=ns.GRID, lw=0.6, ls=":")
ax[1].text(x[0], P["power_limit_w"] - 0.6, "board limit", fontsize=ns.fs(6),
           color=ns.INK2, ha="left", va="top")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"q{q}" for q in qps])
ax[1].set_ylabel("board power (W)")
ax[1].set_ylim(282, 304)
ax[1].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.4, borderpad=0,
             loc="lower right")
ns.panel(ax[1], "b")

# (c) what routing costs in memory, at every resolution that fitted
fr = [r for r in F["rows"] if not r.get("oom")]
lbl = [r["size"].split("x")[0] + "p" if False else r["size"] for r in fr]
xf = np.arange(len(fr))
ax[2].bar(xf - 0.19, [r["peak_mb_full"] for r in fr], 0.38, color=ns.INK2,
          lw=0, label="full")
ax[2].bar(xf + 0.19, [r["peak_mb_routed"] for r in fr], 0.38, color=ns.VERM,
          lw=0, label="routed")
for i, r in enumerate(fr):
    ax[2].text(i, r["peak_mb_routed"] + 30, f"+{r['peak_delta_pct']:.1f}%",
               ha="center", fontsize=ns.fs(6), color=ns.VERM)
ax[2].set_xticks(xf)
ax[2].set_xticklabels([s.replace("x", "×") for s in lbl], fontsize=ns.fs(6))
ax[2].set_ylabel("peak memory (MB)")
ax[2].set_ylim(0, 1250)
ax[2].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.1, borderpad=0,
             loc="upper left")
ns.panel(ax[2], "c")

for A in ax:
    A.spines["top"].set_visible(False); A.spines["right"].set_visible(False)
    A.tick_params(labelsize=ns.fs(6), length=2, width=0.5)
    A.yaxis.label.set_size(ns.fs(5.6))

for _d in (ROOT / "docs/figures", ROOT / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "power.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
print("  wrote docs/figures/power.png")
