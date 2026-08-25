"""Saving over the two knobs that set it: the rate, and the budget.

Every table in this project fixes one of these and sweeps the other, which
hides the shape. The saving is a surface over (rate, budget), and the surface
has a visible edge: each rate saturates at its own budget, beyond which the
ceiling is flat and no further tolerance buys anything. Reading that edge off
a stack of tables is work; reading it off the surface is not.

A surface is used rather than 3D bars because the domain is genuinely
two-dimensional and the quantity is continuous in both. The right-hand panel
carries the same data as three ordinary curves, so nothing has to be read out
of the perspective.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flexplus"))
import style as S
from budget_search import Sweep, QPS, release_rows

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"
MARK = [0.10, 0.16, 0.20]


PERFECT = 0.121317


def main():
    s = Sweep(str(RES / "cells_ctc64_e8.npz"), 2)
    ceil = s.ceiling_saving(0)
    sat = {q: s.ceiling_db(q) for q in QPS}
    budgets = np.round(np.linspace(0.05, 0.30, 51), 5)
    cache = RES / "surface_saving_dense.npy"
    if cache.exists():
        Z = np.load(cache)
    else:
        Z = np.zeros((len(QPS), len(budgets)))
        for i, q in enumerate(QPS):
            for jx, b in enumerate(budgets):
                Z[i, jx] = s.operate(q, float(b))[1]
        np.save(cache, Z)

    # A rate axis fine enough that the surface reads as a surface. The
    # measurement is on five rates; interpolating BETWEEN them is a drawing
    # choice, not a claim, so the five measured ridges stay drawn on top.
    yi = np.linspace(0, len(QPS) - 1, 61)
    Zi = np.empty((len(yi), len(budgets)))
    for jx in range(len(budgets)):
        Zi[:, jx] = np.interp(yi, np.arange(len(QPS)), Z[:, jx])

    fig = plt.figure(figsize=(6.9, 2.85))
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    X, Y = np.meshgrid(budgets, yi)
    norm = plt.Normalize(Z.min(), ceil)
    ax.plot_surface(X, Y, Zi, cmap=cm.viridis, norm=norm, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, shade=True,
                    zorder=1)
    # the ceiling, as a plane rather than a wire cage
    ax.plot_surface(X, Y, np.full_like(Zi, ceil), color="#9a9a9a", alpha=0.10,
                    linewidth=0, antialiased=True, shade=False, zorder=2)
    # where the surface meets it: one curve, not five markers
    ys = np.linspace(0, len(QPS) - 1, 200)
    xs = np.interp(ys, np.arange(len(QPS)), [sat[q] for q in QPS])
    ax.plot(xs, ys, [ceil] * len(ys), color=S.VERM, lw=1.6, zorder=6)
    # the budget at which exactly one rate is on the ceiling
    zs = np.interp(ys, np.arange(len(QPS)),
                   [s.operate(q, PERFECT)[1] for q in QPS])
    ax.plot([PERFECT] * len(ys), ys, zs, color="white", lw=1.7, zorder=7)

    ax.set_xlabel("budget (dB)", labelpad=-4)
    ax.set_ylabel("rate", labelpad=-4)
    # set_zlabel is swallowed by the tight layout at this box aspect; the
    # label goes on the figure instead, beside the tick column it belongs to.
    fig.text(0.012, 0.60, "saving (%)", fontsize=7.0, rotation=90,
             va="center", ha="left", color=S.INK)
    ax.set_yticks(range(len(QPS)))
    ax.set_yticklabels([f"q{q}" for q in QPS])
    ax.set_xticks([0.05, 0.15, 0.25])
    ax.set_zticks([25, 30, 35, 40])
    ax.tick_params(pad=-2.5, length=0)
    ax.view_init(elev=22, azim=-126)
    ax.set_box_aspect((1.3, 1.0, 0.70))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("none")
        pane.line.set_color(S.MUTED)
        pane.line.set_linewidth(0.5)
    ax.grid(False)
    ax.text(sat[63] + 0.010, len(QPS) - 1, ceil + 0.9, "saturation locus",
            color=S.VERM, fontsize=6.0, zorder=8)
    ax.text(PERFECT + 0.004, 0.0, float(zs[0]) + 0.6, "0.121 dB",
            color=S.INK, fontsize=6.0, zorder=8, ha="left",
            fontweight="bold")
    fig.text(0.045, 0.955, "a", fontsize=8.0, fontweight="bold", va="top",
             color=S.INK)
    fig.text(0.085, 0.955, "saving over rate and budget", fontsize=7.0,
             va="top", color=S.INK)

    ax2 = fig.add_subplot(1, 2, 2)
    marks = [(0.10, S.BLUE, "0.10 dB"), (PERFECT, S.VERM, "0.121 dB"),
             (0.20, S.GREEN, "0.20 dB")]
    for b, col, lab in marks:
        y = [s.operate(q, b)[1] for q in QPS]
        ax2.plot(range(len(QPS)), y, "o-", color=col, clip_on=False, zorder=3)
        ax2.annotate(lab, (len(QPS) - 1, y[-1]), xytext=(5, 0),
                     textcoords="offset points", color=col, fontsize=6.4,
                     va="center", fontweight="bold", annotation_clip=False)
    ax2.axhline(ceil, color=S.INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax2.annotate(f"ceiling {ceil:.2f}%", (0.02, ceil), xytext=(0, 3),
                 textcoords="offset points", fontsize=6.2, color=S.INK)
    ax2.set_xticks(range(len(QPS)))
    ax2.set_xticklabels([f"q{q}" for q in QPS])
    ax2.set_xlim(-0.15, len(QPS) - 1 + 0.78); ax2.set_ylim(28, 41.6)
    ax2.set_xlabel("rate"); ax2.set_ylabel("saving (%)")
    ax2.set_title("at 0.121 dB exactly one rate is on the ceiling",
                  fontsize=7.0, pad=4, loc="left")
    S.panel(ax2, "b", dx=-0.19)
    S.ygrid(ax2); S.despine(ax2)
    fig.subplots_adjust(wspace=0.26, left=0.02, right=0.965, top=0.90,
                        bottom=0.14)
    fig.savefig(OUT / "fig13_surface.pdf")
    fig.savefig(OUT / "fig13_surface.png")
    plt.close(fig)
    print(f"  fig13 yazildi   tavan {ceil:.4f}%   kusursuz butce {PERFECT:.6f} dB")


if __name__ == "__main__":
    main()
