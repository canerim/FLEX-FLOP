"""What the boxes in Figure 1 actually contain.

Three zooms, all read off the source rather than described:

  a  patchify -- the reshape that makes tiles independent, and where the seam
     is born. `flexuf/backbone/decoder.py:patchify`
  b  one exit group = 2 DepthConvBlocks. `~/DCVC/src/layers/layers.py:128`
  c  the two adapters. `flexuf/backbone/decoder.py:Conv1x1Adapter, FFNAdapter`

Labels only. The argument is in the caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()

C = 384
BLUE, ORANGE, GREEN, VERM = ns.BLUE, ns.ORANGE, ns.GREEN, ns.VERM
INK, INK2 = ns.INK, ns.INK2


def blank(a):
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 1)
    for s in a.spines.values():
        s.set_visible(False)
    a.grid(False)


def chip(a, x, y, w, h, txt, ec, fc="#ffffff", fs=5.4, bold=False, sub=None):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                               facecolor=fc, edgecolor=ec, lw=0.7,
                               transform=a.transAxes, zorder=3))
    a.text(x + w / 2, y + h * (0.67 if sub else 0.5), txt, ha="center",
           va="center", fontsize=fs, color=INK,
           weight="bold" if bold else "normal", transform=a.transAxes, zorder=4)
    if sub:
        a.text(x + w / 2, y + h * 0.23, sub, ha="center", va="center",
               fontsize=4.1, color=INK2, transform=a.transAxes, zorder=4)


def _res(a, x0, x1, ytop, hgt):
    """A residual bypass drawn as a flat bracket, entirely inside the panel."""
    xm = (x0 + x1) / 2
    a.plot([x0, x0, x1, x1], [ytop, ytop + hgt, ytop + hgt, ytop],
           color=INK2, lw=0.6, transform=a.transAxes, clip_on=False,
           solid_joinstyle="miter")
    a.annotate("", (x1, ytop), (x1, ytop + hgt * 0.45), xycoords=a.transAxes,
               textcoords=a.transAxes,
               arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK2,
                               mutation_scale=5))
    del xm


def arrow(a, x0, y0, x1, y1, c=INK2, style="-|>"):
    a.annotate("", (x1, y1), (x0, y0), xycoords=a.transAxes,
               textcoords=a.transAxes,
               arrowprops=dict(arrowstyle=style, lw=0.7, color=c,
                               mutation_scale=6, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- a: patchify
def panel_patchify(a):
    blank(a)
    a.text(0.0, 0.99, "patchify", fontsize=6.5, weight="bold", color=ORANGE,
           transform=a.transAxes, va="top")

    # the canvas, with the tile lattice drawn on it
    x0, y0, w, h = 0.02, 0.30, 0.26, 0.40
    a.add_patch(Rectangle((x0, y0), w, h, facecolor="#f4f8fb",
                          edgecolor=BLUE, lw=0.8, transform=a.transAxes))
    for i in range(1, 8):
        a.plot([x0 + i * w / 8] * 2, [y0, y0 + h], color=BLUE, lw=0.35,
               alpha=0.55, transform=a.transAxes)
    for i in range(1, 5):
        a.plot([x0, x0 + w], [y0 + i * h / 5] * 2, color=BLUE, lw=0.35,
               alpha=0.55, transform=a.transAxes)
    a.add_patch(Rectangle((x0 + 2 * w / 8, y0 + 3 * h / 5), w / 8, h / 5,
                          facecolor=ORANGE, alpha=0.55, edgecolor=ORANGE,
                          lw=0.9, transform=a.transAxes))
    a.text(x0 + w / 2, y0 + h + 0.08, "[1, 384, 160, 256]", ha="center",
           fontsize=5, color=INK2, transform=a.transAxes)
    a.text(x0 + w / 2, y0 - 0.11, "5 × 8 tiles of 32 × 32", ha="center",
           fontsize=5, color=INK2, transform=a.transAxes)

    # the surgery
    xs = 0.345
    for k, t in enumerate((".view(1, 384, 5, 32, 8, 32)",
                           ".permute(0, 2, 4, 1, 3, 5)",
                           ".reshape(40, 384, 32, 32)")):
        a.text(xs, 0.62 - 0.13 * k, t, fontsize=5.2, family="monospace",
               color=INK, transform=a.transAxes)
    a.text(xs, 0.23, "0 FLOP", fontsize=5.4, color=GREEN, weight="bold",
           transform=a.transAxes)
    arrow(a, 0.30, 0.50, 0.335, 0.50)

    # tiles as batch elements
    bx, by, bw, bh = 0.645, 0.36, 0.075, 0.28
    for i in range(4, -1, -1):
        off = i * 0.013
        a.add_patch(Rectangle((bx + off, by - off), bw, bh,
                              facecolor=ORANGE if i == 0 else "#fdf1e0",
                              alpha=0.55 if i == 0 else 1.0,
                              edgecolor=ORANGE, lw=0.7, transform=a.transAxes,
                              zorder=3 + i))
    a.add_patch(Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=VERM,
                          lw=1.0, transform=a.transAxes, zorder=9))
    arrow(a, 0.605, 0.50, 0.638, 0.50)
    a.text(bx + bw / 2 + 0.02, by + bh + 0.10,
           "40 batch elements,\neach 384 × 32 × 32", ha="center", fontsize=5,
           color=INK2, transform=a.transAxes, linespacing=1.5)
    a.text(0.80, 0.50,
           "every later 3×3 meets padding\nhere, not a neighbour — this\n"
           "reshape is where the seam is born",
           fontsize=4.9, color=VERM, va="center", ha="left",
           transform=a.transAxes, linespacing=1.7)


# ------------------------------------------------------- b: one DepthConvBlock
def panel_block(a):
    blank(a)
    a.text(0.0, 0.99, "one exit group  =  2 × DepthConvBlock", fontsize=6.5,
           weight="bold", color=BLUE, transform=a.transAxes, va="top")

    H = 0.17
    y = 0.53                                   # dc row
    y2 = 0.09                                  # ffn row
    XIN, XOUT = 0.06, 0.845                    # in / out chips
    JOIN = 0.775                               # the two residual adds

    chip(a, XIN, y, 0.075, H, "in", BLUE)
    a.text(0.0, y + H / 2, "dc", fontsize=5.4, color=INK2, style="italic",
           va="center", transform=a.transAxes)
    a.text(0.0, y2 + H / 2, "ffn", fontsize=5.4, color=INK2, style="italic",
           va="center", transform=a.transAxes)

    dc = [(0.17, 0.135, "PW 1×1", "384 → 384", BLUE, "#ffffff", False),
          (0.325, 0.085, "WSiLU", None, BLUE, "#ffffff", False),
          (0.43, 0.145, "DW 3×3", "groups = 384", VERM, "#fdf0ec", True),
          (0.595, 0.135, "PW 1×1", "384 → 384", BLUE, "#ffffff", False)]
    for x, w, t, sb, ec, fc, bd in dc:
        chip(a, x, y, w, H, t, ec, fc=fc, sub=sb, bold=bd)
    arrow(a, XIN + 0.075, y + H / 2, 0.17, y + H / 2)
    for k in range(len(dc) - 1):
        arrow(a, dc[k][0] + dc[k][1], y + H / 2, dc[k + 1][0], y + H / 2)
    arrow(a, 0.73, y + H / 2, JOIN - 0.018, y + H / 2)
    a.text(JOIN, y + H / 2, "⊕", fontsize=9, color=INK2, ha="center",
           va="center", transform=a.transAxes)
    _res(a, XIN + 0.037, JOIN, y + H + 0.015, 0.075)

    ffn = [(0.17, 0.185, "PW 1×1", "384 → 1536"),
           (0.375, 0.205, "WSiLUChunkAdd", "4:1  →  384"),
           (0.60, 0.13, "PW 1×1", "384 → 384")]
    for x, w, t, sb in ffn:
        chip(a, x, y2, w, H, t, GREEN, fc="#eef8f4", sub=sb)
    arrow(a, 0.115, y2 + H / 2, 0.17, y2 + H / 2)
    for k in range(len(ffn) - 1):
        arrow(a, ffn[k][0] + ffn[k][1], y2 + H / 2, ffn[k + 1][0], y2 + H / 2)
    arrow(a, 0.73, y2 + H / 2, JOIN - 0.018, y2 + H / 2)
    a.text(JOIN, y2 + H / 2, "⊕", fontsize=9, color=INK2, ha="center",
           va="center", transform=a.transAxes)
    _res(a, 0.115, JOIN, y2 + H + 0.015, 0.075)

    # dc output feeds the ffn row, down the left margin so it crosses nothing
    a.plot([0.115, 0.115], [y + H / 2 - 0.005, y2 + H / 2], color=INK2, lw=0.6,
           transform=a.transAxes)
    a.plot([0.115, JOIN], [y + H / 2 - 0.005] * 2, color=INK2, lw=0.6,
           transform=a.transAxes, zorder=1)
    chip(a, XOUT, y2, 0.075, H, "out", BLUE)
    arrow(a, JOIN + 0.018, y2 + H / 2, XOUT, y2 + H / 2)

    a.text(0.43, y - 0.085,
           "the only operator with spatial extent — 0.29% of the block, "
           "and the whole cause of the seam",
           fontsize=4.8, color=VERM, ha="center", transform=a.transAxes)
    a.text(0.43, y2 - 0.085, "74.8% of the block, and entirely pointwise",
           fontsize=4.8, color=GREEN, ha="center", transform=a.transAxes)


# ------------------------------------------------------------- c: the adapters
def panel_adapters(a):
    blank(a)
    a.text(0.0, 0.99, "the two exit adapters", fontsize=6.5, weight="bold",
           color=ORANGE, transform=a.transAxes, va="top")

    y = 0.55
    a.text(0.0, y + 0.26, "conv1x1  ·  exits skipping < 4 blocks",
           fontsize=5.2, color=INK2, transform=a.transAxes)
    chip(a, 0.02, y, 0.10, 0.15, "f", ORANGE, fc="#fdf6ec")
    chip(a, 0.17, y, 0.17, 0.15, "PW 1×1", ORANGE, fc="#fdf6ec",
         sub="384 → 384,  W = 0")
    a.text(0.40, y + 0.075, "⊕", fontsize=9, color=INK2, ha="center",
           va="center", transform=a.transAxes)
    for x0, x1 in ((0.125, 0.165), (0.343, 0.385)):
        arrow(a, x0, y + 0.075, x1, y + 0.075)
    _res(a, 0.07, 0.40, y + 0.155, 0.085)
    a.text(0.46, y + 0.075, "1 C²", fontsize=5.6, color=INK, va="center",
           weight="bold", transform=a.transAxes)

    y2 = 0.12
    a.text(0.0, y2 + 0.26, "ffn  ·  exits skipping ≥ 4 blocks",
           fontsize=5.2, color=INK2, transform=a.transAxes)
    chip(a, 0.02, y2, 0.08, 0.15, "f", ORANGE, fc="#fdf6ec")
    chip(a, 0.13, y2, 0.15, 0.15, "PW 1×1", ORANGE, fc="#fdf6ec",
         sub="384 → 1536")
    chip(a, 0.29, y2, 0.19, 0.15, "WSiLUChunkAdd", ORANGE, fc="#fdf6ec",
         sub="4:1  →  384")
    chip(a, 0.49, y2, 0.15, 0.15, "PW 1×1", ORANGE, fc="#fdf6ec",
         sub="384 → 384,  W = 0")
    a.text(0.68, y2 + 0.075, "⊕", fontsize=9, color=INK2, ha="center",
           va="center", transform=a.transAxes)
    for x0, x1 in ((0.105, 0.125), (0.283, 0.285), (0.483, 0.485),
                   (0.643, 0.665)):
        arrow(a, x0, y2 + 0.075, x1, y2 + 0.075)
    _res(a, 0.06, 0.68, y2 + 0.155, 0.085)
    a.text(0.74, y2 + 0.075, "5 C²", fontsize=5.6, color=VERM, va="center",
           weight="bold", transform=a.transAxes)
    a.text(0.74, y2 - 0.005, "not 2", fontsize=4.6, color=VERM, va="center",
           transform=a.transAxes)
    a.text(0.02, 0.02, "both residual and zero-initialised, so at step 0 the "
           "ladder is bit-exact DCVC-UF", fontsize=4.8, color=INK2,
           transform=a.transAxes)


def main(out="docs/figures/pipeline_detail.png"):
    fig = plt.figure(figsize=(ns.W2, 4.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.0], hspace=0.50)
    for i, (fn, lab) in enumerate(((panel_patchify, "a"),
                                   (panel_block, "b"),
                                   (panel_adapters, "c"))):
        ax = fig.add_subplot(gs[i])
        fn(ax)
        ns.panel(ax, lab, dx=-0.03, dy=1.02)
    fig.savefig(R / out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
