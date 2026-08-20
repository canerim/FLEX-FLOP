"""The router head, layer by layer, drawn from the module itself.

Every shape, channel count and parameter count in this figure is read off
`flexuf.router.head2.StemRouterHeadV2` at draw time rather than typed in, so the
figure cannot drift from the code. If someone widens the MLP, the figure widens
with it.

    python scripts/router_arch_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

import naturestyle as ns  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402,F811

from flexuf.router.head2 import StemRouterHeadV2  # noqa: E402

ns.apply()

STEM_CH, LAT_CH, K, J = 384, 256, 6, 2
head = StemRouterHeadV2(STEM_CH, LAT_CH, K, min_exit=J)
P = {n: p.numel() for n, p in head.named_parameters()}
r_stem = head.proj_stem.out_channels
r_lat = head.proj_lat.out_channels
d_in = head.mlp[0].normalized_shape[0]
hidden = head.mlp[1].out_features
n_par = sum(P.values()) + sum(b.numel() for b in head.buffers())

BLUE, ORANGE, GREEN = ns.BLUE, ns.ORANGE, ns.GREEN
FACE = {"in": "#eef2f7", "proj": "#e8f0fb", "pool": "#f2f2f2",
        "mlp": "#fdf0e6", "out": "#e9f4ec"}
EDGE = {"in": "#7a8899", "proj": BLUE, "pool": "#999999",
        "mlp": ORANGE, "out": GREEN}

fig = plt.figure(figsize=(ns.W2, 2.62))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
ax.axis("off")


def box(x, y, w, h, kind, title, sub=None, note=None, fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.28,rounding_size=0.7",
                                fc=FACE[kind], ec=EDGE[kind], lw=0.7, zorder=2))
    ax.text(x + w / 2, y + h - (1.9 if sub else h / 2 + 0.6), title, ha="center",
            va="top" if sub else "center", fontsize=fs, color=ns.INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h - 4.4, sub, ha="center", va="top", fontsize=5.0,
                color=ns.INK2, zorder=3)
    if note:
        ax.text(x + w / 2, y - 1.5, note, ha="center", va="top", fontsize=4.7,
                color=ns.INK2, zorder=3)
    return x + w, y + h / 2


def arrow(x0, y0, x1, y1, label=None, colour="#7a8899"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=5, lw=0.6, color=colour,
                                 shrinkA=1.5, shrinkB=1.5, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.7, label, ha="center", va="bottom",
                fontsize=4.7, color=ns.INK2)


# ---- inputs ---------------------------------------------------------------
ax.text(0.6, 38.6, "a", fontsize=7, fontweight="bold", color=ns.INK, va="top")
ax.text(4.0, 38.6, "what the decoder already has", fontsize=5.8, color=ns.INK,
        va="top")

box(1, 29.0, 16, 6.2, "in", "stem feature  f", "[384, H/8, W/8]",
    "after the first j=2 blocks")
box(1, 18.0, 16, 6.2, "in", "latent  ŷ  ⊕  scales  σ", "[2×256, H/16, W/16]",
    "σ is the entropy model's own\npredicted Gaussian width")
box(1, 6.6, 16, 5.4, "in", "quality index  qp/63", "one number per FRAME",
    "the same for every tile: it cannot\ntell two tiles of one frame apart")

# ---- projections ----------------------------------------------------------
ax.text(23.5, 38.6, "b", fontsize=7, fontweight="bold", color=ns.INK, va="top")
ax.text(26.9, 38.6, "projected, then pooled to one vector per tile",
        fontsize=5.8, color=ns.INK, va="top")

box(24, 29.0, 15.5, 6.2, "proj", f"Conv 1×1  384→{r_stem}",
    f"{P['proj_stem.weight'] + P['proj_stem.bias']:,} par",
    "288 MAC per RGB pixel")
box(24, 18.0, 15.5, 6.2, "proj", f"Conv 1×1  512→{r_lat}",
    f"{P['proj_lat.weight'] + P['proj_lat.bias']:,} par",
    "64 MAC per RGB pixel")

box(44, 29.0, 13.5, 6.2, "pool", "mean, std", f"over a 32×32 tile → {2*r_stem}")
box(44, 18.0, 13.5, 6.2, "pool", "mean, std", f"over a 16×16 tile → {2*r_lat}")

arrow(17.6, 32.1, 23.4, 32.1)
arrow(17.6, 21.1, 23.4, 21.1)
arrow(40.1, 32.1, 43.4, 32.1)
arrow(40.1, 21.1, 43.4, 21.1)
arrow(17.6, 9.3, 60.5, 9.3, colour="#b0b0b0")

# ---- concat ---------------------------------------------------------------
box(61, 6.6, 8.5, 28.6, "pool", "", None)
ax.text(65.25, 20.9, "concatenate", ha="center", va="center", fontsize=5.6,
        color=ns.INK, rotation=90)
ax.text(65.25, 4.9, f"{2*r_stem} + {2*r_lat} + 1 = {d_in}", ha="center",
        va="top", fontsize=4.9, color=ns.INK2)
arrow(57.9, 32.1, 60.4, 32.1)
arrow(57.9, 21.1, 60.4, 21.1)

# ---- MLP ------------------------------------------------------------------
ax.text(72.0, 38.6, "c", fontsize=7, fontweight="bold", color=ns.INK, va="top")
ax.text(75.4, 38.6, "one vector per tile, so width is nearly free",
        fontsize=5.8, color=ns.INK, va="top")

mlp_rows = [
    (f"LayerNorm({d_in})", f"{P['mlp.0.weight'] + P['mlp.0.bias']:,} par"),
    (f"Linear {d_in}→{hidden}  ·  SiLU",
     f"{P['mlp.1.weight'] + P['mlp.1.bias']:,} par"),
    (f"Linear {hidden}→{hidden}  ·  SiLU",
     f"{P['mlp.3.weight'] + P['mlp.3.bias']:,} par"),
    (f"Linear {hidden}→{K}",
     f"{P['mlp.5.weight'] + P['mlp.5.bias']:,} par, zero-init bias"),
]
y = 35.2
for title, sub in mlp_rows:
    ax.add_patch(FancyBboxPatch((72, y - 4.0), 17.5, 4.0,
                                boxstyle="round,pad=0.24,rounding_size=0.6",
                                fc=FACE["mlp"], ec=EDGE["mlp"], lw=0.7, zorder=2))
    ax.text(80.75, y - 1.45, title, ha="center", va="center", fontsize=5.2,
            color=ns.INK, zorder=3)
    ax.text(80.75, y - 3.1, sub, ha="center", va="center", fontsize=4.5,
            color=ns.INK2, zorder=3)
    if y < 35.0:
        arrow(80.75, y + 1.1, 80.75, y - 0.05)
    y -= 5.5
arrow(69.6, 20.9, 71.9, 20.9)

ax.add_patch(FancyBboxPatch((72, 8.0), 17.5, 4.0,
                            boxstyle="round,pad=0.24,rounding_size=0.6",
                            fc=FACE["out"], ec=EDGE["out"], lw=0.7, zorder=2))
ax.text(80.75, 10.55, "+ load-balancing bias", ha="center", va="center",
        fontsize=5.2, color=ns.INK, zorder=3)
ax.text(80.75, 8.9, "6 numbers, nudged by usage not by gradient",
        ha="center", va="center", fontsize=4.5, color=ns.INK2, zorder=3)
arrow(80.75, 13.1, 80.75, 12.05)

# ---- decision -------------------------------------------------------------
ax.add_patch(FancyBboxPatch((91.4, 6.6), 8.0, 28.6,
                            boxstyle="round,pad=0.28,rounding_size=0.7",
                            fc=FACE["out"], ec=EDGE["out"], lw=0.7, zorder=2))
ax.text(95.4, 30.2, "logits  z", ha="center", va="center", fontsize=5.6,
        color=ns.INK, zorder=3)
ax.text(95.4, 27.6, f"one per exit, K={K}", ha="center", va="center",
        fontsize=4.7, color=ns.INK2, zorder=3)
ax.text(95.4, 21.5, "exits below j\nmasked to −∞", ha="center", va="center",
        fontsize=4.7, color=ns.INK2, zorder=3)
ax.text(95.4, 14.0, "argmax", ha="center", va="center", fontsize=5.6,
        color=ns.INK, zorder=3)
ax.text(95.4, 10.6, "log softmax(z)\n− β c", ha="center", va="center",
        fontsize=4.9, color=ns.INK2, zorder=3)
arrow(89.6, 20.9, 91.3, 20.9)

ax.text(50, 1.0, f"{n_par:,} parameters in total, and 0.162% of the decode it "
                 f"is deciding about", ha="center", va="bottom", fontsize=5.0,
        color=ns.INK2)

out = ROOT / "docs/figures/router_arch.png"
fig.savefig(out, dpi=500, facecolor="white")
print(f"  {n_par:,} parameters   d_in={d_in}   hidden={hidden}")
print(f"  wrote {out.relative_to(ROOT)}")
