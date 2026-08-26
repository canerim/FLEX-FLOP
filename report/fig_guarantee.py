"""A budget the set keeps on average is not a budget the frame keeps.

The headline bisects one lambda per rate so that the MEAN per-frame decibel
lands on 0.1. What each frame then receives is a distribution, and the
distribution is what a deployment cares about: 138 of 265 frame-rate pairs are
served worse than the number on the abstract, and the worst is two and a half
times it.

Panel a is that distribution, as an empirical CDF so no binning choice is
hiding in it, for four ways of setting lambda. Panel b prices the tail against
what it costs: every mode is one point, saving on one axis and worst-case
decibel on the other, so a mode that buys a shorter tail with compute is
visibly paying and a mode that does not is visibly not.

Measured on the dumped per-cell tables, which carry no tiling penalty, so the
absolute decibels sit below a tiled decode's. What is being read here is the
SHAPE of the tail and which rule shortens it, and that is a property of the
allocation rather than of the seam.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"
BUDGET = 0.1


def ecdf(v):
    v = np.sort(np.asarray(v))
    return v, np.arange(1, v.size + 1) / v.size


def main():
    G = json.loads((RES / "per_frame_guarantee.json").read_text())
    Sf = json.loads((RES / "safe_routing.json").read_text())
    D = json.loads((RES / "per_frame_dists.json").read_text())

    modes = [
        ("oracle map, one $\\lambda$ per rate", D["oracle_global"], S.MUTED, "-"),
        ("oracle map, one $\\lambda$ per frame", D["oracle_perframe"], S.VERM, "-"),
        ("predicted map, one $\\lambda$ per rate", D["pred_global"], S.SKY, "-"),
        ("predicted map, $\\lambda$ signalled per frame", D["pred_siglam"],
         S.BLUE, "-"),
    ]

    fig = plt.figure(figsize=(7.0, 2.55))
    ax = fig.add_axes([0.065, 0.185, 0.40, 0.70])
    for lab, v, c, ls in modes:
        x, y = ecdf(v)
        ax.step(x, 100 * y, where="post", color=c, lw=1.25, ls=ls, zorder=3)
    ax.axvline(BUDGET, color=S.INK, lw=0.7, ls=(0, (2.6, 2.0)), zorder=2)
    ax.text(BUDGET + 0.004, 6, "the budget", fontsize=6.0, color=S.INK,
            rotation=90, va="bottom")
    ax.set_xlim(0.0, 0.50); ax.set_ylim(0, 101)
    ax.set_xlabel("$\\Delta$PSNR delivered to a frame (dB)")
    ax.set_ylabel("frames at or below (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    S.ygrid(ax)
    S.despine(ax) if hasattr(S, "despine") else None
    S.panel(ax, "a", dx=-0.135)

    # direct labels, no legend box
    lab_y = {0: 30, 1: 92, 2: 16, 3: 76}
    lab_x = {0: 0.185, 1: 0.128, 2: 0.245, 3: 0.128}
    for i, (lab, v, c, ls) in enumerate(modes):
        ax.text(lab_x[i], lab_y[i], lab, fontsize=5.7, color=c, ha="left",
                va="center", fontweight="bold")

    ax2 = fig.add_axes([0.605, 0.185, 0.365, 0.70])
    pts = [("oracle, per rate", D["oracle_global"], G["modes"]["global"]["saving_pct"], S.MUTED, 1),
           ("oracle, per frame", D["oracle_perframe"], G["modes"]["perframe"]["saving_pct"], S.VERM, 1),
           ("predicted, per rate", D["pred_global"], Sf["rows"][0]["saving_pct"], S.SKY, 1),
           ("predicted, $\\lambda$ signalled", D["pred_siglam"],
            Sf["signalled_lambda_perframe"]["saving_pct"], S.BLUE, 1)]
    for lab, v, sv, c, _ in pts:
        ax2.scatter([sv], [max(v)], s=26, color=c, zorder=4,
                    edgecolor="white", linewidth=0.6)
    for r in Sf["decoder_perframe"]:
        ax2.scatter([r["saving_pct"]], [r["max"]], s=9, color=S.GREEN,
                    zorder=3, alpha=0.85)
    ax2.plot([r["saving_pct"] for r in Sf["decoder_perframe"]],
             [r["max"] for r in Sf["decoder_perframe"]], color=S.GREEN,
             lw=0.8, alpha=0.7, zorder=2)
    ax2.axhline(BUDGET, color=S.INK, lw=0.7, ls=(0, (2.6, 2.0)), zorder=1)
    ax2.text(6.5, BUDGET + 0.006, "the budget", fontsize=6.0, color=S.INK)
    lbl = [("oracle map, one $\\lambda$ per rate", 31.84, 0.240, S.MUTED,
            "right", -0.9, 0.010),
           ("oracle map,\none $\\lambda$ per frame", 32.14, 0.114, S.VERM,
            "center", 0.0, 0.016),
           ("predicted map,\none $\\lambda$ per rate", 24.17, 0.475, S.SKY,
            "left", 0.9, -0.028),
           ("predicted map,\n$\\lambda$ signalled per frame", 25.06, 0.114,
            S.BLUE, "center", 0.0, -0.058)]
    for lab, x, y, c, ha, dx, dy in lbl:
        ax2.text(x + dx, y + dy, lab, fontsize=5.6, color=c, ha=ha,
                 va="bottom", fontweight="bold", linespacing=1.25)
    ax2.text(9.0, 0.355, "predicted map throttling itself\non its own forecast",
             fontsize=5.6, color=S.GREEN, ha="left", linespacing=1.3)
    ax2.set_xlim(3, 38); ax2.set_ylim(0.05, 0.545)
    ax2.set_xlabel("mean MAC saving (%)")
    ax2.set_ylabel("worst frame $\\Delta$PSNR (dB)")
    S.ygrid(ax2)
    S.panel(ax2, "b", dx=-0.145)

    fig.savefig(OUT / "fig17_guarantee.pdf")
    fig.savefig(OUT / "fig17_guarantee.png")
    print("  fig17_guarantee yazildi")


if __name__ == "__main__":
    main()
