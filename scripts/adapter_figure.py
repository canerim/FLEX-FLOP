"""What the exit adapters are worth, per exit and per rate."""
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
d = json.load(open(R / "results/adapter_ablation.json"))
qps = [r["qp"] for r in d["rows"]]
exits = sorted(int(e) for e in d["rows"][0]["trained"])
COL = {0: ns.BLUE, 32: ns.ORANGE, 63: ns.VERM}

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.6))

# a: with and without, one panel
w = 0.36
for i, e in enumerate(exits):
    for k, q in enumerate(qps):
        r = d["rows"][k]
        t, o = r["trained"][str(e)], r["all_off"][str(e)]
        ax[0].plot([i - w / 2 + k * w / len(qps)] * 2, [t, o], "-",
                   color=COL[q], lw=1.4, alpha=0.85)
        ax[0].plot(i - w / 2 + k * w / len(qps), o, "o", ms=3.6, color=COL[q])
        ax[0].plot(i - w / 2 + k * w / len(qps), t, "_", ms=6, color=COL[q])
ax[0].set_yscale("log")
ax[0].set_xticks(range(len(exits))); ax[0].set_xticklabels(exits)
ax[0].set_xlabel("exit"); ax[0].set_ylabel("dB below the release (log)")
for q in qps:
    ax[0].plot([], [], "-", color=COL[q], lw=1.4, label=f"q{q}")
ax[0].plot([], [], "o", color="#666666", ms=3.6, label="without")
ax[0].plot([], [], "_", color="#666666", ms=6, label="with")
ax[0].legend(fontsize=ns.fs(6), loc="lower left", ncol=2, columnspacing=0.7)
ns.panel(ax[0], "a")

# b: the gain
for q, r in zip(qps, d["rows"]):
    g = [r["all_off"][str(e)] - r["trained"][str(e)] for e in exits]
    ax[1].plot(exits, g, "-o", ms=4, lw=1.2, color=COL[q], label=f"q{q}")
ax[1].axhline(0, color=ns.INK, lw=0.7)
ax[1].set_xticks(exits); ax[1].set_xlabel("exit")
ax[1].set_ylabel("dB recovered by the adapter")
ax[1].legend(fontsize=ns.fs(6), loc="upper right")
ns.panel(ax[1], "b", dx=-0.24)

# c: gain against the number of blocks the exit skips
K = max(exits) + 1
b_per = 2
for q, r in zip(qps, d["rows"]):
    sk = [(K - 1 - e) * b_per for e in exits]
    g = [r["all_off"][str(e)] - r["trained"][str(e)] for e in exits]
    ax[2].plot(sk, g, "o", ms=5, color=COL[q], label=f"q{q}")
ax[2].set_xlabel("blocks the exit skips")
ax[2].set_ylabel("dB recovered")
ax[2].set_xticks(sorted(set(sk)))
ns.panel(ax[2], "c", dx=-0.24)

fig.tight_layout(w_pad=2.2, h_pad=1.2)
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "adapter_gain.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
print("  -> docs/figures/adapter_gain.png")
