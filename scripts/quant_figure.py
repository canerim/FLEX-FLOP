"""What quantisation does to the ladder, not to the average.

The obvious plot is "quality against bit-width", and it answers the wrong
question. Early exit does not care how good the decoder is; it cares how
DIFFERENT the exits are from each other, because that difference is the only
thing a router has to work with. A decoder that is uniformly worse still routes
fine. A decoder whose exits have converged does not route at all.

So the left panel is per-exit quality, and the right panel is the quantity that
decides whether the two levers compose: the spread between the shallowest and
deepest exit, against arithmetic cost in BOPs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import naturestyle as ns  # noqa: E402
ns.apply()

d = json.loads((R / "results/quant_BEST.json").read_text())
rows = d["rows"]
bits = sorted({r["bits"] for r in rows}, reverse=True)
qps = sorted({r["qp"] for r in rows})
get = {(r["bits"], r["qp"]): r for r in rows}

fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))

# a: per-exit quality at the highest rate, where the effect is largest
QP = qps[-1]
K = len(get[(bits[0], QP)]["db_per_exit"])
for c, b in zip(ns.SERIES, bits):
    y = get[(b, QP)]["db_per_exit"]
    ax[0].plot(range(K), y, marker="o", ms=4, lw=1.2, color=c,
               label=("fp32" if b == 32 else f"{b} bit"))
ax[0].set_xticks(range(K)); ax[0].set_xlabel("exit")
ax[0].set_ylabel(f"dB above the release, qp {QP}")
ax[0].set_yscale("log")
ax[0].legend(loc="lower left", fontsize=5, frameon=False, ncol=2)
ax[0].set_title("Every exit degrades — the deepest most", fontsize=6,
                color=ns.INK2, loc="left")
ns.panel(ax[0], "a")

# b: the spread, which is what routing actually consumes
for c, q in zip(ns.SERIES, qps):
    sp = [get[(b, q)]["db_per_exit"][0] - get[(b, q)]["db_per_exit"][-1]
          for b in bits]
    bo = [get[(b, q)]["bops_vs_fp32"] for b in bits]
    ax[1].plot(bo, sp, marker="o", ms=4.5, lw=1.3, color=c, label=f"qp {q}")
    for x, y, b in zip(bo, sp, bits):
        if q == qps[-1]:
            ax[1].annotate(("fp32" if b == 32 else f"{b} b"), (x, y), fontsize=4.6,
                           color=ns.INK2, textcoords="offset points",
                           xytext=(3, -7))
ax[1].set_xscale("log")
ax[1].set_xlabel("arithmetic cost (BOPs vs fp32)")
ax[1].set_ylabel("spread, exit 0 − deepest (dB)")
ax[1].legend(loc="center right", fontsize=5, frameon=False)
# Not "quantisation flattens the ladder" -- it does that at HIGH rate, where the
# spread was large to begin with. At qp0 the spread is flat to slightly rising.
# The general claim would have been wrong in a way the figure itself disproves.
ax[1].set_title("The ladder collapses at high rate, not at low", fontsize=6,
                color=ns.INK2, loc="left")
ns.panel(ax[1], "b", dx=-0.18)

fig.tight_layout(w_pad=2.0)
for out in (R / "docs/figures/quant.png", R / "results/quant.png"):
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("  wrote docs/figures/quant.png and results/quant.png")
for q in qps:
    print(f"  qp{q} spread: " + "  ".join(
        f"{('fp32' if b==32 else str(b)+'b')}={get[(b,q)]['db_per_exit'][0]-get[(b,q)]['db_per_exit'][-1]:.3f}"
        for b in bits))
