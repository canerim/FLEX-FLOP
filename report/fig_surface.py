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


def main():
    s = Sweep(str(RES / "cells_ctc64_e8.npz"), 2)
    ceil = s.ceiling_saving(0)
    sat = {q: s.ceiling_db(q) for q in QPS}
    budgets = np.round(np.linspace(0.05, 0.30, 26), 4)
    cache = RES / "surface_saving.npy"
    if cache.exists():
        Z = np.load(cache)
    else:
        Z = np.zeros((len(QPS), len(budgets)))
        for i, q in enumerate(QPS):
            for jx, b in enumerate(budgets):
                Z[i, jx] = s.operate(q, float(b))[1]
        np.save(cache, Z)

    fig = plt.figure(figsize=(6.9, 2.75))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    X, Y = np.meshgrid(budgets, np.arange(len(QPS)))
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True,
                    rstride=1, cstride=1, alpha=0.94, vmin=Z.min(), vmax=ceil)
    # the ceiling, and where each rate meets it
    ax.plot_wireframe(X, Y, np.full_like(Z, ceil), color=S.MUTED, lw=0.35,
                      rstride=1, cstride=6, alpha=0.55)
    for i, q in enumerate(QPS):
        ax.plot([sat[q]], [i], [ceil], "o", color=S.VERM, ms=3.4, zorder=10)
    for b in MARK:
        ax.plot([b] * len(QPS), np.arange(len(QPS)),
                [s.operate(q, b)[1] for q in QPS], color="white", lw=1.4,
                zorder=9)
    ax.set_xlabel("budget (dB)", labelpad=-4)
    ax.set_ylabel("rate", labelpad=-6)
    ax.set_zlabel("saving (%)", labelpad=-6)
    ax.set_yticks(range(len(QPS)))
    ax.set_yticklabels([f"q{q}" for q in QPS])
    ax.set_xticks([0.05, 0.15, 0.25])
    ax.tick_params(pad=-2)
    ax.view_init(elev=24, azim=-128)
    ax.set_box_aspect((1.25, 1.0, 0.72))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_edgecolor(S.GRID)
    ax.grid(False)
    ax.set_title("saving over rate and budget", fontsize=7.0, pad=-2,
                 loc="left")
    # Axes3D.text takes (x, y, z, s); the 2-D panel helper cannot be reused, so
    # the letter is placed in figure coordinates instead.
    fig.text(0.045, 0.95, "a", fontsize=8.0, fontweight="bold", va="top",
             color=S.INK)

    ax2 = fig.add_subplot(1, 2, 2)
    for i, b in enumerate(MARK):
        y = [s.operate(q, b)[1] for q in QPS]
        ax2.plot(range(len(QPS)), y, "o-", color=S.CAT[i], clip_on=False,
                 zorder=3)
        ax2.annotate(f"{b:.2f} dB", (len(QPS) - 1, y[-1]), xytext=(5, 0),
                     textcoords="offset points", color=S.CAT[i], fontsize=6.4,
                     va="center", fontweight="bold", annotation_clip=False)
    ax2.axhline(ceil, color=S.VERM, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax2.annotate(f"ceiling {ceil:.2f}%", (0, ceil), xytext=(2, 3),
                 textcoords="offset points", fontsize=6.2, color=S.VERM)
    ax2.set_xticks(range(len(QPS)))
    ax2.set_xticklabels([f"q{q}" for q in QPS])
    ax2.set_xlim(-0.15, len(QPS) - 1 + 0.75)
    ax2.set_xlabel("rate"); ax2.set_ylabel("saving (%)")
    ax2.set_title("three budgets, and where each rate saturates",
                  fontsize=7.0, pad=4, loc="left")
    S.panel(ax2, "b", dx=-0.19)
    S.ygrid(ax2); S.despine(ax2)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(OUT / "fig13_surface.pdf")
    fig.savefig(OUT / "fig13_surface.png")
    plt.close(fig)
    print(f"  fig13 yazildi   tavan {ceil:.2f}%   doyma "
          + " ".join(f"q{q}:{sat[q]:.4f}" for q in QPS))


if __name__ == "__main__":
    main()
