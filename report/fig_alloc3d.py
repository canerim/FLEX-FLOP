"""What the allocation looks like at three budgets, one plot each.

The saving is a scalar; the allocation that produces it is a distribution
over rates and exits, and that is where saturation is visible as a shape
rather than as a number. At 0.121317 dB the lowest rate has collapsed onto
the shallowest exit it may take -- one bar and a remainder -- which is what
"on the ceiling" means and what no table shows.

Bars, not a surface: the domain is two discrete axes, five rates by six
exits, and interpolating between exits would draw depths the decoder cannot
run. Every (rate, exit) pair gets a footprint on the floor whether or not it
is used, so an empty cell reads as measured absence rather than as missing
data. Colour encodes the exit sequentially; height carries the share alone.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flexplus"))
import style as S
from budget_search import Sweep, QPS

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"
# The perfect budget is q0's exact first-saturation decibel, not a rounded
# printout of it: at the rounded value the ceiling test fails by 1e-7 and the
# bisection returns 40.07 instead of the 40.08 that is the whole point.
BUDGETS = [
    (0.100000, "0.10 dB", "the budget the paper reports",
     "no rate has reached its ceiling; every rate still has depth to give up"),
    (None, "0.121317 dB", "the perfect budget",
     "exactly one rate is on the ceiling, and it is the lowest one"),
    (0.200000, "0.20 dB", "past the point of return",
     "three rates saturated; the extra tolerance buys them nothing"),
]


def solve(s, q, b):
    """The allocation at this budget: saturated, or bisected on the true dB."""
    if s.ceiling_db(q) <= b:
        return s.at(q, 1e9), True
    lo, hi = 0.0, 1e-8
    while s.at(q, hi)[0] <= b and hi < 1e9:
        hi *= 4
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if s.at(q, mid)[0] <= b:
            lo = mid
        else:
            hi = mid
    return s.at(q, lo), False


def draw(label, kicker, note, H, sv, satd, ceil, K, j):
    nq = len(QPS)
    fig = plt.figure(figsize=(3.42, 3.22))
    ax = fig.add_axes([-0.055, 0.130, 1.05, 0.775], projection="3d")
    cmap = cm.viridis
    dx = dy = 0.62

    # Floor footprints first, so an unused (rate, exit) pair is visibly empty.
    # Exits below the split are not merely unchosen -- forward() clamps to j,
    # so no allocation can reach them -- and they are hatched, not shaded, so
    # the plot does not read as if the router declined them.
    for i in range(nq):
        for k in range(K):
            avail = k >= j
            r = Rectangle((k - dx / 2, i - dy / 2), dx, dy,
                          facecolor=(0.90, 0.90, 0.91) if avail else "none",
                          edgecolor="none" if avail else (0.78, 0.78, 0.80),
                          hatch=None if avail else "////", linewidth=0.0,
                          alpha=0.55, zorder=0)
            ax.add_patch(r)
            art3d.pathpatch_2d_to_3d(r, z=0, zdir="z")

    # Bars back to front so the painter's algorithm has nothing to decide.
    for i in range(nq - 1, -1, -1):
        for k in range(K - 1, -1, -1):
            h = H[i, k]
            if h < 0.04:
                continue
            ax.bar3d(k - dx / 2, i - dy / 2, 0, dx, dy, h,
                     color=cmap(0.08 + 0.82 * k / (K - 1)),
                     edgecolor="white", linewidth=0.4, shade=True,
                     alpha=1.0, zsort="max")

    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$e_{k}$" for k in range(K)])
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in QPS])
    ax.set_zlim(0, 100)
    ax.set_zticks([0, 25, 50, 75, 100])
    ax.set_zticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlim(-0.6, K - 0.4); ax.set_ylim(-0.6, nq - 0.4)
    ax.set_xlabel("exit taken", labelpad=-4)
    ax.set_ylabel("rate point", labelpad=-3)
    ax.tick_params(labelsize=6.0, pad=-2.0, length=0)
    ax.tick_params(axis="z", pad=-0.5, labelsize=5.8)
    ax.view_init(elev=23, azim=-61)
    ax.set_box_aspect((1.22, 1.0, 0.66))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color(S.MUTED); axis.line.set_linewidth(0.5)
    ax.zaxis._axinfo["grid"].update(color=(0.86, 0.86, 0.87), linewidth=0.4)
    ax.grid(False)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0
    ax.yaxis._axinfo["grid"]["linewidth"] = 0

    fig.text(0.035, 0.980, label, fontsize=9.6, fontweight="bold", va="top",
             color=S.INK)
    fig.text(0.035, 0.916, kicker, fontsize=6.6, va="top", color=S.VERM,
             fontweight="bold")
    fig.text(0.035, 0.874, note, fontsize=6.2, va="top", color=S.INK2)
    fig.text(0.965, 0.980, "share of 64 px cells", fontsize=6.0, va="top",
             ha="right", color=S.MUTED)
    fig.text(0.965, 0.941, "hatched: below the split, unreachable",
             fontsize=5.6, va="top", ha="right", color=S.MUTED)

    # The saving per rate, on its own line, so nothing sits inside the box.
    y = 0.088
    fig.text(0.035, y, "MAC saving", fontsize=6.0, color=S.MUTED, va="center")
    x = 0.325
    for q, v, t in zip(QPS, sv, satd):
        fig.text(x, y, f"q{q}", fontsize=5.6, color=S.MUTED,
                 ha="center", va="center")
        fig.text(x, y - 0.046, f"{v:.1f}" + ("●" if t else ""),
                 fontsize=6.8, ha="center", va="center",
                 color=S.VERM if t else S.INK,
                 fontweight="bold" if t else "normal")
        x += 0.132
    fig.text(0.035, y - 0.046, f"ceiling {ceil:.2f}%", fontsize=6.0,
             color=S.MUTED, va="center")
    return fig


def main():
    s = Sweep(str(RES / "cells_ctc64_e8.npz"), 2)
    ceil = s.ceiling_saving(0)
    perfect = s.ceiling_db(0)
    print(f"  tam butce {perfect:.9f} dB, tavan {ceil:.4f}%", flush=True)
    names = ["fig14a_alloc_010", "fig14b_alloc_perfect", "fig14c_alloc_020"]
    dump = {}
    for (b, label, kicker, note), name in zip(BUDGETS, names):
        b = perfect if b is None else b
        H = np.zeros((len(QPS), s.K)); sv = []; satd = []
        for i, q in enumerate(QPS):
            (db, saving, k), sat = solve(s, q, b)
            H[i] = np.bincount(k, minlength=s.K) / len(k) * 100
            sv.append(saving); satd.append(sat)
        fig = draw(label, kicker, note, H, sv, satd, ceil, s.K, s.j)
        fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png")
        plt.close(fig)
        dump[f"{b:.6f}"] = {"hist_pct": H.tolist(), "saving_pct": sv,
                            "saturated": [bool(x) for x in satd]}
        print(f"  {name}: " + "  ".join(
            f"q{q}:{v:.2f}%{'*' if t else ''}"
            for q, v, t in zip(QPS, sv, satd)), flush=True)
    (RES / "alloc3d.json").write_text(json.dumps(
        {"ckpt_epoch": 8, "cell_px": 64, "split_depth": 2,
         "ceiling_pct": ceil, "perfect_db": perfect,
         "budgets": dump}, indent=2))
    print("  yazildi results/alloc3d.json")


if __name__ == "__main__":
    main()
