"""The decoder we modify, drawn as its own paper draws it.

Redrawn after Figure 2 of DCVC-UF (Li, Li and Lu, ACM MM 2024) rather than
reproduced, so the parts this work touches can be marked and everything else
greyed back. The point of the figure is the reader's orientation: the frozen
analysis path on the left, the frame-specific decoders on the right, and the one
block FLEX-UF replaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()

GREY, GREYE = "#f2f2f2", "#b4b4b4"
BLUE, BLUEE = "#eaf2fa", "#4a86c5"
ORNG, ORNGE = "#fdf0e3", "#e08a2e"
PINK, PINKE = "#fbeef5", "#c46ba3"
INK, INK2 = ns.INK, ns.INK2


def box(a, x, y, w, h, txt, fc, ec, fs=5.0, sub=None, bold=False, alpha=1.0):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                               facecolor=fc, edgecolor=ec, lw=0.8, zorder=3,
                               alpha=alpha, transform=a.transAxes))
    a.text(x + w / 2, y + h * (0.64 if sub else 0.5), txt, ha="center",
           va="center", fontsize=fs, color=INK, zorder=4,
           weight="bold" if bold else "normal", transform=a.transAxes)
    if sub:
        a.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
               fontsize=4.2, color=INK2, zorder=4, transform=a.transAxes)


def arr(a, x0, y0, x1, y1, c=INK2, lw=0.7):
    a.annotate("", (x1, y1), (x0, y0), xycoords=a.transAxes,
               textcoords=a.transAxes, zorder=2,
               arrowprops=dict(arrowstyle="-|>", lw=lw, color=c,
                               mutation_scale=6, shrinkA=0, shrinkB=0))


def main(out="docs/figures/baseline.png"):
    fig, a = plt.subplots(figsize=(ns.W2, 1.95))
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 1)
    for s in a.spines.values():
        s.set_visible(False)
    a.grid(False)

    Y, H = 0.30, 0.26

    # the frozen analysis path
    a.add_patch(Rectangle((0.015, 0.16), 0.475, 0.48, facecolor="#fafafa",
                          edgecolor=GREYE, lw=0.7, ls=(0, (3, 2)), zorder=1,
                          transform=a.transAxes))
    a.text(0.253, 0.665, "frozen: encoder, entropy model, bitstream",
           ha="center", fontsize=5.2, color=INK2, transform=a.transAxes)

    box(a, 0.03, Y, 0.075, H, "patchify", GREY, GREYE, sub="$\\downarrow 8$")
    box(a, 0.125, Y, 0.105, H, "chunk encoder", GREY, GREYE, sub="DC blocks")
    box(a, 0.25, Y, 0.06, H, "Q", GREY, GREYE)
    box(a, 0.33, Y, 0.145, H, "entropy model", GREY, GREYE,
        sub="$\\mu$, $\\sigma$, hyperprior")
    for x0, x1 in ((0.105, 0.125), (0.23, 0.25), (0.31, 0.33)):
        arr(a, x0, Y + H / 2, x1, Y + H / 2)
    a.text(0.022, Y + H + 0.10, "$X_i$", fontsize=6, color=INK,
           transform=a.transAxes)
    arr(a, 0.022, Y + H + 0.075, 0.05, Y + H + 0.02)

    # the synthesis path
    box(a, 0.515, Y, 0.10, H, "chunk decoder", BLUE, BLUEE, sub="DC blocks")
    box(a, 0.635, Y, 0.145, H, "frame-specific\ndecoders", ORNG, ORNGE,
        bold=True)
    box(a, 0.80, Y, 0.075, H, "head", BLUE, BLUEE, sub="$\\uparrow 8$")
    for x0, x1 in ((0.475, 0.515), (0.615, 0.635), (0.78, 0.80)):
        arr(a, x0, Y + H / 2, x1, Y + H / 2)
    a.text(0.895, Y + H / 2, "$\\hat{X}_i$", fontsize=6, color=INK,
           va="center", transform=a.transAxes)
    arr(a, 0.875, Y + H / 2, 0.89, Y + H / 2)

    # cross-chunk context, drawn as the feedback it is
    box(a, 0.30, 0.79, 0.19, 0.155, "cross-chunk context", PINK, PINKE)
    a.plot([0.6875, 0.6875, 0.49], [Y + H, 0.8675, 0.8675], color=PINKE,
           lw=0.7, transform=a.transAxes, zorder=2)
    arr(a, 0.51, 0.8675, 0.495, 0.8675, c=PINKE)
    a.plot([0.30, 0.4025, 0.4025], [0.8675, 0.8675, Y + H + 0.005],
           color=PINKE, lw=0.7, transform=a.transAxes, zorder=2)
    arr(a, 0.4025, Y + H + 0.08, 0.4025, Y + H + 0.005, c=PINKE)

    # what this paper changes
    a.add_patch(Rectangle((0.632, Y - 0.012), 0.151, H + 0.024,
                          facecolor="none", edgecolor=ns.VERM, lw=1.2,
                          zorder=5, transform=a.transAxes))
    a.annotate("", (0.7075, Y - 0.02), (0.7075, 0.10), xycoords=a.transAxes,
               textcoords=a.transAxes,
               arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ns.VERM,
                               mutation_scale=6))
    a.text(0.7075, 0.055, "FLEX-UF replaces this with a per-tile exit ladder",
           ha="center", fontsize=5.4, color=ns.VERM, transform=a.transAxes)

    fig.savefig(R / out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
