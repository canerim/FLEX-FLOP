"""The router, drawn in the teaser's language.

Figure 1 of the paper shows where the exit map is applied and says nothing
about where it comes from. This is that missing half: StemRouterHeadV2, the
144,024-parameter head that reads three things the decode has already
produced and emits one exit per tile.

Every number is read off the module rather than remembered. proj_stem is
Conv2d(384, 48, 1); proj_lat is Conv2d(512, 32, 1) over the latent
concatenated with the entropy model's scales; each is pooled to a (mean, std)
pair per tile, giving 96 and 64; qp arrives as one number per FRAME, giving
the 161st; and the MLP is LayerNorm -> 161 -> 256 -> 256 -> 6.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teaser_style import *   # noqa

setup()
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

fig, ax = canvas(11.0, 4.7, (0, 28), (0, 12))

ax.text(0.3, 11.6, "FLEX router", fontsize=14, fontweight="bold", va="top")
ax.text(0.3, 10.85, "one exit per tile, from tensors the decode already holds",
        fontsize=8.2, color=GREY, va="top")

# ---------------------------------------------------------------- inputs
group(ax, 0.30, 3.70, 7.55, 6.10, "Inputs — already computed by the decode",
      ("#fafafa", "#c4c4c4"), fs=7.6, label_dy=-0.30)

stack(ax, 0.85, 7.85, 1.60, 1.30, 3, BLUEG)
ax.text(1.65, 8.50, "stem\nfeature", ha="center", va="center", fontsize=7.2,
        zorder=9)
tensor(ax, 1.65, 7.70, "[B, 384, H/8, W/8]")

stack(ax, 0.85, 5.75, 1.60, 1.30, 3, PURPLE)
ax.text(1.65, 6.40, "latent\n+ scales", ha="center", va="center", fontsize=7.2,
        zorder=9)
tensor(ax, 1.65, 5.60, "[B, 2×256, H/16, W/16]")

box(ax, 1.00, 4.25, 1.30, 0.80, "qp", GREY_B, fs=8.2)
tensor(ax, 1.65, 4.10, "one per frame")

box(ax, 3.55, 7.85, 2.40, 1.30, "Conv 1×1\n384 → 48", BLUE, fs=7.2, tc="white")
box(ax, 3.55, 5.75, 2.40, 1.30, "Conv 1×1\n512 → 32", BLUE, fs=7.2, tc="white")
arrow(ax, (2.62, 8.50), (3.55, 8.50))
arrow(ax, (2.62, 6.40), (3.55, 6.40))
tensor(ax, 4.75, 7.70, "0.133% of a decode")
tensor(ax, 4.75, 5.60, "0.030%")

box(ax, 6.35, 7.85, 1.05, 1.30, "pool\nper tile", GREEN, fs=6.6)
box(ax, 6.35, 5.75, 1.05, 1.30, "pool\nper tile", GREEN, fs=6.6)
arrow(ax, (6.05, 8.50), (6.35, 8.50), ms=7)
arrow(ax, (6.05, 6.40), (6.35, 6.40), ms=7)

# --------------------------------------------------------------- concat bus
BUS = 8.55
for y0, n in ((8.50, "96"), (6.40, "64")):
    ax.plot([7.45, BUS], [y0, y0], color=INK, lw=1.15, zorder=4)
    ax.text(7.70, y0 + 0.18, n, fontsize=6.8, color=GREY)
ax.plot([2.32, BUS], [4.65, 4.65], color=INK, lw=1.15, zorder=4)
ax.text(7.70, 4.83, "1", fontsize=6.8, color=GREY)
ax.plot([BUS, BUS], [4.65, 8.50], color=INK, lw=1.15, zorder=4)
arrow(ax, (BUS, 6.90), (9.55, 6.90))
ax.text(8.68, 8.30, "d = 161", fontsize=7.8, fontweight="bold")

# ----------------------------------------------------------------- MLP
group(ax, 9.55, 5.15, 8.05, 3.60,
      "MLP — one vector per tile, so its width is free",
      ("#f4f1fa", "#9186c2"), fs=7.6, label_dy=4.15)
xs, x = [], 9.85
for lab, w in (("LayerNorm", 1.30), ("161 → 256", 1.25), ("SiLU", 0.85),
               ("256 → 256", 1.25), ("SiLU", 0.85), ("256 → 6", 1.20)):
    fill = GREEN if lab == "SiLU" else PURPLE
    box(ax, x, 6.15, w, 1.60, lab, fill, fs=6.8)
    if xs:
        arrow(ax, (xs[-1], 6.95), (x, 6.95), lw=0.9, ms=7)
    x += w; xs.append(x); x += 0.22
ax.text(13.55, 5.75, "144,024 parameters", fontsize=7.2, color=GREY,
        ha="center")

# --------------------------------------------------------------- logits
arrow(ax, (17.35, 6.95), (18.10, 6.95))
box(ax, 18.10, 5.95, 1.55, 2.00, "", WHITE, z=2)
for k in range(6):
    ax.add_patch(plt.Rectangle((18.30, 6.15 + k * 0.28), 1.15, 0.22,
                               facecolor=EXITS[k], edgecolor="#8a8a8a",
                               linewidth=0.4, zorder=4))
ax.text(18.87, 8.15, "logits", ha="center", fontsize=7.4)
tensor(ax, 18.87, 5.85, "6 exits")

# ------------------------------------------------------- argmin, exit map
arrow(ax, (19.75, 6.95), (20.45, 6.95))
box(ax, 20.45, 6.00, 2.95, 1.90,
    "argmin$_k$\n$[\\,\\hat m_k + \\lambda\\,c_k\\,]$", PINKG, fs=7.8)
tensor(ax, 21.92, 5.90, "clamped to $k \\geq j$")
arrow(ax, (23.50, 6.95), (24.20, 6.95))

E = EXITS
gridglyph(ax, 24.20, 5.55, 2.80, 2.80, 5, 5,
          colors=[E[2], E[2], E[3], E[2], E[4], E[2], E[3], E[2], E[2], E[5],
                  E[3], E[2], E[2], E[4], E[2], E[2], E[2], E[3], E[2], E[2],
                  E[4], E[2], E[2], E[2], E[3]])
ax.text(25.60, 8.65, "exit map", ha="center", fontsize=8.0, fontweight="bold")
tensor(ax, 25.60, 5.40, "one exit per 256 px tile")

for k in range(2, 6):
    x = 20.45 + (k - 2) * 1.55
    ax.add_patch(plt.Rectangle((x, 3.95), 0.55, 0.42, facecolor=E[k],
                               edgecolor="#8a8a8a", linewidth=0.4, zorder=4))
    ax.text(x + 0.72, 4.16, f"$e_{k}$", fontsize=7.2, va="center")
ax.text(20.45, 3.35,
        "$e_0$ and $e_1$ are unreachable: forward() clamps to $j = 2$",
        fontsize=6.8, color=GREY)

# --------------------------------------------------------------- footnotes
ax.text(0.30, 2.35,
        "Trained by regret-weighted cross-entropy to the oracle's own choice: "
        "the weight is what the mistake costs, so capacity is spent where it "
        "matters.", fontsize=7.4)
ax.text(0.30, 1.65,
        "Agreement with the oracle 78.7%, against 37.9% for the best constant "
        "choice; 76.2% held out. Total cost 0.163% of one decode.",
        fontsize=7.4, color=GREY)
ax.text(0.30, 0.80,
        "The entropy model's scales are what v1 threw away, and the two "
        "signals most correlated with the oracle's choice live there.",
        fontsize=7.4, color=ACCENT)

fig.savefig(OUT / "fig_router_teaser.pdf")
fig.savefig(OUT / "fig_router_teaser.png")
print("  fig_router_teaser yazildi")
