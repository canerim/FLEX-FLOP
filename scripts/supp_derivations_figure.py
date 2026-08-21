"""The two pictures the derivations section needs, drawn from one file.

Both panels come from results/tile_table.json alone, so the fixed-proportion
set and the per-tile set are the same tiles, the same exits and the same cost
vector, and the gap between them is not an artefact of pooling two
measurements. The main paper's Figure 9 already draws monotonicity, the
frontier and the Pareto gap; neither panel here repeats it.

    ./.venv/bin/python scripts/supp_derivations_figure.py

writes docs/figures/supp_achievable.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns   # noqa: E402
ns.apply()

d = json.load(open(R / "results/tile_table.json"))
j = d["j"]
D = np.array(d["D"])[:, j:]                 # tiles x usable exits
cost = np.array(d["cost"])[j:]
ref = d["ref_mse"]
bits = np.array(d["bits_per_tile"])
T, K = D.shape
db = lambda m: 10 * np.log10(m / ref)

lams = np.concatenate([[0.0], np.geomspace(1e-9, 1e-2, 4000)])
choice = np.stack([np.argmin(D + lam * cost[None, :], axis=1) for lam in lams])
C = cost[choice].mean(axis=1)
Dm = np.take_along_axis(D, choice.T, axis=1).T.mean(axis=1)
sv, dbv = 100 * (1 - C), db(Dm)

# fixed proportions: the K vertices and their lower hull, same tiles
vx = 100 * (1 - cost)
vy = db(D.mean(axis=0))

fig, ax = plt.subplots(1, 2, figsize=(3.42, 1.62))

# a -- the achievable set: one multiplier per tile against one for the frame
ax[0].plot(sv, dbv, "-", color=ns.BLUE, lw=1.3, label="per tile")
ax[0].plot(vx, vy, "--s", color=ns.ORANGE, lw=1.0, ms=3.0,
           label="fixed proportions")
for x, y, kk in zip(vx, vy, range(j, j + K)):
    ax[0].annotate(f"{kk}", (x, y), textcoords="offset points",
                   xytext=(3.5, 2.5), fontsize=6, color=ns.ORANGE)
# the vertical gap at the middle vertex, where both sets are interior
xg = vx[1]
yg = np.interp(xg, sv[::-1], dbv[::-1])
ax[0].annotate("", xy=(xg, yg), xytext=(xg, vy[1]),
               arrowprops=dict(arrowstyle="<->", lw=0.6, color=ns.INK2))
# Right of the arrow, not left of it: on the left it sat on the vertex label
# for exit 4, which is three data units away at this scale.
ax[0].text(xg + 1.0, 0.5 * (yg + vy[1]), f"{vy[1] - yg:.3f} dB",
           fontsize=6, va="center", ha="left", color=ns.INK2)
ax[0].set_xlabel("MACs saved (%)")
ax[0].set_ylabel("dB below released")
ax[0].legend(fontsize=6, loc="upper left")
ns.panel(ax[0], "a", dx=-0.30)

# b -- what the price does to each tile, tiles ordered by coded bits
order = np.argsort(bits)
img = choice[:, order].T + j
cmap = ListedColormap([ns.SKY, ns.BLUE, ns.PURPLE, ns.VERM][:K])
norm = BoundaryNorm(np.arange(j - 0.5, j + K + 0.5), cmap.N)
ax[1].imshow(img, aspect="auto", origin="lower", cmap=cmap, norm=norm,
             extent=[np.log10(max(lams[1], 1e-9)), np.log10(lams[-1]), 0, T])
ax[1].set_xlabel("log$_{10}$ λ")
ax[1].set_ylabel("tile, by coded bits")
ax[1].set_xlim(-7.5, -3.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[1],
                  ticks=np.arange(j, j + K), pad=0.03, fraction=0.09)
cb.set_label("exit", fontsize=6)
cb.ax.tick_params(labelsize=6, width=0.4, length=1.6)
cb.outline.set_linewidth(0.4)
ns.panel(ax[1], "b", dx=-0.34)

fig.tight_layout(pad=0.35, w_pad=1.0)
out = R / "docs/figures/supp_achievable.png"
fig.savefig(out, dpi=400)
print(f"  -> {out}")
print(f"    fixed-proportion vertices at savings {np.round(vx, 2).tolist()}")
print(f"    gap at the middle vertex ({xg:.2f}%): {vy[1] - yg:.4f} dB")
print(f"    per-tile frontier: {len(set(np.round(sv, 6)))} distinct savings")
