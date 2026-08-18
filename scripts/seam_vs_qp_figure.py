"""Seam penalty against rate, over the whole QP range, and what it costs us.

Panel a is the measurement: the pure tiling penalty at every rate, under the
stock padding rule and under the one we ship, both on the UNTRAINED warm start
so nothing but geometry is in the number -- plus the trained run's actual floor,
same frames, same reference, which is what the routed results are built on.

Panel b is why it matters. The routed system is quoted at 0.1 dB below the
release, and the floor is charged inside that 0.1 dB: it is spent before a
single tile has exited early. What is left is what buys compute.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns

d = json.load(open(R / "results/seam_vs_qp.json"))
rows = d["rows"]
qp = np.array([r["qp"] for r in rows], float)
zer = np.array([r["zeros"] for r in rows])
rep = np.array([r["replicate"] for r in rows])
tra = np.array([r["trained"] for r in rows])
BUDGET = 0.1

fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))

# ---- a: the penalty curves -------------------------------------------------
ax[0].axhspan(0, BUDGET, color=ns.BLUE, alpha=0.07, lw=0)
ax[0].axhline(BUDGET, color=ns.BLUE, lw=0.8, ls=(0, (4, 2)))
ax[0].text(1, BUDGET * 1.08, "0.1 dB budget", fontsize=5, color=ns.BLUE)
ax[0].plot(qp, zer, marker="o", ms=3.5, lw=1.2, color=ns.VERM,
           label="zeros padding — stock behaviour")
ax[0].plot(qp, rep, marker="s", ms=3.5, lw=1.2, color=ns.ORANGE,
           label="replicate padding — what we ship")
ax[0].plot(qp, tra, marker="^", ms=3.5, lw=1.2, color=ns.GREEN,
           label="…and after training (the actual floor)")
ax[0].set_yscale("log")
ax[0].set_xlabel("qp   (0 = lowest rate  →  63 = highest)")
ax[0].set_ylabel("dB below the release  (log)")
ax[0].legend(loc="upper left", fontsize=5)
ax[0].set_title("Pure tiling penalty: every tile at full depth, "
                "no early exit anywhere", fontsize=6, color=ns.INK2, loc="left")
for x, y in ((qp[0], zer[0]), (qp[-1], zer[-1]), (qp[-1], rep[-1]),
             (qp[-1], tra[-1])):
    ax[0].annotate(f"{y:.3f}", (x, y), fontsize=5, color=ns.INK2,
                   textcoords="offset points", xytext=(0, 5), ha="center")
ns.panel(ax[0], "a")

# ---- b: what the budget is spent on ---------------------------------------
# The floor is charged inside the 0.1 dB, so the headroom that early exit can
# actually spend is 0.1 minus the floor, and it shrinks with rate.
head = BUDGET - tra
ax[1].bar(qp, tra, width=4.5, color=ns.GREEN, label="floor: the seam, unavoidable")
ax[1].bar(qp, head, width=4.5, bottom=tra, color=ns.SKY,
          label="headroom early exit can spend")
ax[1].axhline(BUDGET, color=ns.INK, lw=0.8)
for x, f in zip(qp, tra):
    ax[1].annotate(f"{100*f/BUDGET:.0f}%", (x, f), fontsize=5, color=ns.INK,
                   textcoords="offset points", xytext=(0, 1.5), ha="center")
ax[1].set_xlabel("qp")
ax[1].set_ylabel("dB of the 0.1 dB budget")
ax[1].set_ylim(0, BUDGET * 1.25)
ax[1].legend(loc="upper left", fontsize=5)
ax[1].set_title("Labels: share of the budget gone before any tile exits early",
                fontsize=6, color=ns.INK2, loc="left")
ns.panel(ax[1], "b", dx=-0.18)

fig.tight_layout()
out = R / "docs/figures/seam_vs_qp.png"
fig.savefig(out, dpi=300)
print(f"  -> {out}")

# The two dB conventions, printed rather than assumed equal: the seam tables use
# the 6:1:1 YUV PSNR DCVC-UF reports, the routed curve uses an RGB MSE ratio.
print(f"\n  {'qp':>4}{'YUV 6:1:1':>12}{'RGB ratio':>12}{'diff':>9}")
for r in rows:
    print(f"  {r['qp']:>4}{r['trained']:>12.4f}{r['trained_rgb']:>12.4f}"
          f"{r['trained_rgb']-r['trained']:>+9.4f}")
