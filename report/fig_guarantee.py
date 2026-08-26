"""A budget the set keeps on average is not a budget the frame keeps.

The headline bisects one multiplier per rate so that the MEAN per-frame
decibel lands on 0.1. What each frame then receives is a distribution, and the
distribution is what a deployment cares about.

Panel a is that distribution on the DEPLOYED tiled path, both modes measured
by the same script on the same checkpoint so they differ only in where the
multiplier is chosen. Panel b prices the tail against what it costs, and adds
the decoder-side modes, which are measured on the dumped per-cell tables
because they exist to be compared with each other rather than quoted: filled
markers are the tiled path, hollow ones the tables, and the two are never
added together.
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
    T = json.loads((RES / "guarantee_tiled_e9.json").read_text())

    fig = plt.figure(figsize=(7.0, 2.55))

    # ---- a: the deployed path, the two modes the paper can claim ----------
    ax = fig.add_axes([0.065, 0.185, 0.40, 0.70])
    for lab, key, c in (("one $\\lambda$ per rate", "tiled_global", S.MUTED),
                        ("one $\\lambda$ per frame", "tiled_perframe", S.VERM)):
        x, y = ecdf(D[key])
        ax.step(x, 100 * y, where="post", color=c, lw=1.4, zorder=3)
    ax.axvline(BUDGET, color=S.INK, lw=0.7, ls=(0, (2.6, 2.0)), zorder=2)
    ax.text(BUDGET + 0.004, 5, "the budget", fontsize=6.0, color=S.INK,
            rotation=90, va="bottom")
    ax.set_xlim(0.0, 0.28); ax.set_ylim(0, 101)
    ax.set_xlabel("$\\Delta$PSNR delivered to a frame (dB)")
    ax.set_ylabel("frames at or below (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    S.ygrid(ax)
    S.panel(ax, "a", dx=-0.135)
    ax.text(0.106, 82, "one $\\lambda$ per frame", fontsize=6.2, color=S.VERM,
            fontweight="bold", va="center")
    ax.text(0.150, 42, "one $\\lambda$ per rate", fontsize=6.2, color=S.MUTED,
            fontweight="bold", va="center")
    ax.text(0.118, 18, f"{T['global']['over']} of {T['global']['n']} frames\n"
            "above the budget", fontsize=5.7, color=S.MUTED, va="center",
            linespacing=1.3)

    # ---- b: the tail against its price ------------------------------------
    ax2 = fig.add_axes([0.605, 0.185, 0.365, 0.70])
    ax2.scatter([T["global"]["saving"]], [T["global"]["max"]], s=30,
                color=S.MUTED, zorder=5, edgecolor="white", linewidth=0.6)
    ax2.scatter([T["perframe"]["saving"]], [T["perframe"]["max"]], s=30,
                color=S.VERM, zorder=5, edgecolor="white", linewidth=0.6)
    ax2.scatter([Sf["rows"][0]["saving_pct"]], [Sf["rows"][0]["max"]], s=26,
                facecolor="none", edgecolor=S.SKY, linewidth=1.0, zorder=4)
    sg = Sf["signalled_lambda_perframe"]
    ax2.scatter([sg["saving_pct"]], [sg["max"]], s=26, facecolor="none",
                edgecolor=S.BLUE, linewidth=1.0, zorder=4)
    xs = [r["saving_pct"] for r in Sf["decoder_perframe"]]
    ys = [r["max"] for r in Sf["decoder_perframe"]]
    ax2.plot(xs, ys, color=S.GREEN, lw=0.8, alpha=0.7, zorder=2)
    ax2.scatter(xs, ys, s=8, facecolor="none", edgecolor=S.GREEN,
                linewidth=0.7, zorder=3)
    ax2.axhline(BUDGET, color=S.INK, lw=0.7, ls=(0, (2.6, 2.0)), zorder=1)
    ax2.text(4.2, BUDGET + 0.009, "the budget", fontsize=6.0, color=S.INK)
    for lab, x, y, c, ha, dx, dy in [
            ("oracle map,\none $\\lambda$ per rate", T["global"]["saving"],
             T["global"]["max"], S.MUTED, "right", -1.0, 0.012),
            ("oracle map,\none $\\lambda$ per frame", T["perframe"]["saving"],
             T["perframe"]["max"], S.VERM, "left", 1.1, 0.004),
            ("predicted map,\none $\\lambda$ per rate",
             Sf["rows"][0]["saving_pct"], Sf["rows"][0]["max"], S.SKY,
             "left", 1.0, -0.030),
            ("predicted map,\n$\\lambda$ signalled per frame",
             sg["saving_pct"], sg["max"], S.BLUE, "right", -1.1, 0.004)]:
        ax2.text(x + dx, y + dy, lab, fontsize=5.6, color=c, ha=ha,
                 va="bottom", fontweight="bold", linespacing=1.25)
    ax2.text(8.0, 0.360, "predicted map throttling\nitself on its own forecast",
             fontsize=5.6, color=S.GREEN, ha="left", linespacing=1.3)
    ax2.text(0.50, -0.245, "filled: deployed tiled path       "
             "hollow: per-cell tables", transform=ax2.transAxes,
             fontsize=5.5, color=S.MUTED, ha="center")
    ax2.set_xlim(2, 41); ax2.set_ylim(0.05, 0.545)
    ax2.set_xlabel("mean MAC saving (%)")
    ax2.set_ylabel("worst frame $\\Delta$PSNR (dB)")
    S.ygrid(ax2)
    S.panel(ax2, "b", dx=-0.145)

    fig.savefig(OUT / "fig17_guarantee.pdf")
    fig.savefig(OUT / "fig17_guarantee.png")
    print("  fig17_guarantee yazildi (gercek karolu yol)")


if __name__ == "__main__":
    main()
