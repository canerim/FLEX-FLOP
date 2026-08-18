"""Does the exit map have to be recomputed per frame and per rate?"""
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

d = json.load(open(R / "results/map_transfer.json"))
B = d["budget_db"]
time_rows = [r for r in d["rows"] if r["kind"] == "time"]
rate_rows = [r for r in d["rows"] if r["kind"] == "rate"]
qps = sorted({r["qp"] for r in time_rows})
COL = {0: ns.BLUE, 32: ns.ORANGE, 63: ns.VERM}

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.3))

# a: quality when the map is reused across frames
for q in qps:
    rs = sorted([r for r in time_rows if r["qp"] == q], key=lambda r: r["offset"])
    ax[0].plot([r["offset"] for r in rs], [r["transfer_db"] for r in rs],
               "-o", ms=3.5, lw=1.1, color=COL[q], label=f"q{q}")
    ax[0].plot([r["offset"] for r in rs], [r["in_place_db"] for r in rs],
               "--", lw=0.8, color=COL[q], alpha=0.6)
ax[0].axhline(B, color=ns.INK, lw=0.8, ls=(0, (4, 2)))
ax[0].set_xlabel("frames since the map was computed")
ax[0].set_ylabel("delivered dB")
ax[0].legend(fontsize=5.5, loc="upper left")
ax[0].set_title("Reuse across time", fontsize=6, color=ns.INK2, loc="left")
ns.panel(ax[0], "a")

# b: saving when reused across frames
for q in qps:
    rs = sorted([r for r in time_rows if r["qp"] == q], key=lambda r: r["offset"])
    ax[1].plot([r["offset"] for r in rs], [r["transfer_saving"] for r in rs],
               "-o", ms=3.5, lw=1.1, color=COL[q])
    ax[1].plot([r["offset"] for r in rs], [r["in_place_saving"] for r in rs],
               "--", lw=0.8, color=COL[q], alpha=0.6)
ax[1].set_xlabel("frames since the map was computed")
ax[1].set_ylabel("MACs saved (%)")
ax[1].set_title("Reuse across time", fontsize=6, color=ns.INK2, loc="left")
ns.panel(ax[1], "b", dx=-0.24)

# c: the rate transfer matrix, as delivered dB
src = sorted({r["from"] for r in rate_rows})
dst = sorted({r["to"] for r in rate_rows})
Mx = np.full((len(src), len(dst)), np.nan)
for r in rate_rows:
    Mx[src.index(r["from"]), dst.index(r["to"])] = r["transfer_db"]
im = ax[2].imshow(Mx, cmap="magma_r", vmin=B * 0.8, vmax=B * 2.1)
for i in range(len(src)):
    for j2 in range(len(dst)):
        ax[2].text(j2, i, f"{Mx[i, j2]:.3f}", ha="center", va="center",
                   fontsize=5.5,
                   color="white" if Mx[i, j2] > B * 1.5 else ns.INK)
ax[2].set_xticks(range(len(dst))); ax[2].set_xticklabels([f"q{q}" for q in dst])
ax[2].set_yticks(range(len(src))); ax[2].set_yticklabels([f"q{q}" for q in src])
ax[2].set_xlabel("applied at"); ax[2].set_ylabel("map computed at")
ax[2].grid(False)
cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=5); cb.set_label("delivered dB", fontsize=5.5)
ax[2].set_title("Reuse across rate", fontsize=6, color=ns.INK2, loc="left")
ns.panel(ax[2], "c", dx=-0.28)

fig.tight_layout()
fig.savefig(R / "docs/figures/map_transfer.png", dpi=300)
print("  -> docs/figures/map_transfer.png")
