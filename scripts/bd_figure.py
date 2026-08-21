"""What the saving costs in the currency a compression reviewer reads.

Everything else in this paper prices the trade-off in dB. A reader from video
coding prices it in BD-rate: the average bitrate difference at matched quality,
integrated over the rate range, against an anchor. Here the anchor is the
released DCVC-UF decoder itself -- the same bitstream, decoded in full -- so the
BD-rate is not a comparison between two codecs but the price of decoding the
same file with fewer operations.

  a  the trade-off itself: BD-rate against decoder MACs saved, one line per
     configuration, each point labelled with the budget that produced it
  b  why 0.3 dB is where the paper stops: the saving saturates, the price
     does not

Panel b is the finding worth stating out loud. Between 0.1 and 0.3 dB the
saving nearly doubles for under two points of BD-rate. Between 0.3 and 0.5 it
moves by under a point while the BD-rate keeps climbing, so the last stretch of
budget buys almost no compute and is paid for in full.

    python scripts/bd_figure.py [--out docs/figures/bdrate.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import naturestyle as ns          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

ns.apply()

# The two configurations the paper reports, in the order it introduces them,
# under the names the prose uses rather than the keys in the file.
CONFIGS = [
    ("A signalled", "A, signalled map", ns.BLUE, "-o"),
    ("B router", "B, predicted (no side information)", ns.ORANGE, "--s"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/bdrate.json")
    ap.add_argument("--out", default="docs/figures/bdrate.png")
    # The main paper is out of pages, so it gets the trade-off alone at one
    # column; the supplement, which has room, gets the saturation panel with
    # it. Same numbers, same script, one flag.
    ap.add_argument("--layout", choices=("wide", "column"), default="wide")
    a = ap.parse_args()
    wide = a.layout == "wide"

    d = json.loads((ROOT / a.src).read_text())
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], []).append(r)
    for k in by:
        by[k].sort(key=lambda r: r["budget_db"])

    if wide:
        fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
    else:
        fig, _ax = plt.subplots(1, 1, figsize=(ns.W1, 2.3))
        ax = [_ax, None]

    # ---------------------------------------------------------------- a
    # The origin is the released decoder: no saving, no rate cost. Drawing it
    # as a point rather than describing it in the caption is what makes the
    # panel read as a trade-off and not as two arbitrary curves.
    ax[0].plot([0], [0], marker="*", ms=8, color=ns.BLACK, ls="none",
               label="DCVC-UF (released)", zorder=4)
    for i, (key, label, col, style) in enumerate(CONFIGS):
        rs = by.get(key)
        if not rs:
            continue
        x = [r["saving_pct_vs_release"] for r in rs]
        y = [r["bd_rate_pct"] for r in rs]
        ax[0].plot(x, y, style, color=col, lw=1.1, ms=4, label=label, zorder=3)
        # The budget is what the operator sets, so it belongs on the point
        # rather than in a third legend to cross-reference. The two
        # configurations converge at 0.5 dB -- both are within a point of the
        # ceiling there -- so their labels go on opposite sides of the curve.
        off = (5, 4) if i == 0 else (-5, -11)
        ha = "left" if i == 0 else "right"
        for r, xi, yi in zip(rs, x, y):
            ax[0].annotate(f"{r['budget_db']:g} dB", (xi, yi),
                           textcoords="offset points", xytext=off,
                           fontsize=6, color=col, ha=ha)
    ax[0].set_xlabel("decoder MACs saved (%)")
    ax[0].set_ylabel("BD-rate vs the released decoder (%)")
    ax[0].set_xlim(-3, None)
    ax[0].legend(loc="upper left", handlelength=1.8, labelspacing=0.35)
    ns.panel(ax[0], "a")

    # ---------------------------------------------------------------- b
    # Both quantities against the budget that produced them. Two axes because
    # they are in different units.
    if not wide:
        fig.tight_layout()
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        plt.close(fig)
        print(f"  -> {a.out}")
        return 0
    rs = by.get("A signalled", [])
    b = [r["budget_db"] for r in rs]
    sav = [r["saving_pct_vs_release"] for r in rs]
    bd = [r["bd_rate_pct"] for r in rs]
    ax[1].plot(b, sav, "-o", color=ns.GREEN, lw=1.2, ms=4, label="MACs saved")
    ax[1].set_xlabel("budget (dB)")
    ax[1].set_ylabel("decoder MACs saved (%)", color=ns.GREEN)
    ax[1].tick_params(axis="y", colors=ns.GREEN)
    ax[1].set_ylim(0, 46)

    ax2 = ax[1].twinx()
    ax2.plot(b, bd, "--s", color=ns.VERM, lw=1.2, ms=4, label="BD-rate")
    ax2.set_ylabel("BD-rate (%)", color=ns.VERM)
    ax2.tick_params(axis="y", colors=ns.VERM)
    ax2.set_ylim(0, 4)
    ax2.grid(False)

    # Where the saving stops moving, computed rather than drawn at a chosen x:
    # the operating point should be visible in the data, not asserted by the
    # figure.
    if len(b) >= 3:
        # The knee is the point the smallest gain starts from, not the one it
        # ends at: argmin over the increments indexes the segment, and the
        # segment's left end is the budget past which there is nothing left.
        knee = int(np.argmin(np.diff(sav)))
        ax[1].axvline(b[knee], color=ns.INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax[1].annotate(f"{sav[-1] - sav[knee]:.1f} points left,\n"
                       f"{bd[-1] - bd[knee]:.1f}% BD-rate to take them",
                       (b[knee], 4), textcoords="offset points",
                       xytext=(5, 0), fontsize=6, color=ns.INK2, va="bottom")

    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper left", handlelength=1.8,
                 labelspacing=0.35)
    ns.panel(ax[1], "b", dx=-0.20)

    fig.tight_layout(w_pad=2.4)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"  -> {a.out}")
    for key, _, _, _ in CONFIGS:
        for r in by.get(key, []):
            print(f"     {key:<12} {r['budget_db']:g} dB   "
                  f"{r['saving_pct_vs_release']:5.2f}% saved   "
                  f"{r['bd_rate_pct']:.3f}% BD-rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
