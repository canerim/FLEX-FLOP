"""Saving against how many tiles a frame has.

The method's limit is not the picture, it is the number of choices. A 416x240
frame is two tiles at 256 px, and two tiles cannot be allocated: either both go
deep and nothing is saved, or one goes shallow and the budget is spent. The
sequences where FLEX-UF saves nothing are exactly those, and this figure says so
as a law rather than as a caveat.

Tile count is read off the sequence name, since it is fixed by the resolution
and the tile side. Savings are the per-sequence values already measured for the
supplement, at the 0.1 dB budget, all five rates.

    python scripts/granularity_figure.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns                                    # noqa: E402

ns.apply()
OUT = (ROOT / "docs" / "figures", ROOT / "paper" / "figures")
TILE = 256
RATE_COLS = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]


def tiles_of(name):
    m = re.search(r"(\d{3,4})x(\d{3,4})", name)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return math.ceil(w / TILE) * math.ceil(h / TILE)


def main():
    src = ROOT / "results/supp_per_sequence_PAPER_b010.json"
    d = json.load(open(src))
    fig, ax = plt.subplots(figsize=(ns.W1, 1.95))

    allx, ally = [], []
    for col, row in zip(RATE_COLS, d["rows"]):
        xs, ys = [], []
        for e in row["per_sequence"]:
            t = tiles_of(e["seq"])
            if t:
                xs.append(t)
                ys.append(e["saving_pct_vs_release"])
        ax.scatter(xs, ys, s=7, c=col, alpha=0.75, linewidths=0,
                   label=f"q{row['qp']}", zorder=3)
        allx += xs
        ally += ys

    allx, ally = np.array(allx, float), np.array(ally, float)
    # A saturating curve in the number of choices: nothing at one tile, most of
    # the ceiling by forty. Fitted with one free scale on log tiles, and drawn
    # only over the range the data covers.
    lx = np.log2(allx)
    A = np.vstack([lx, np.ones_like(lx)]).T
    m, c = np.linalg.lstsq(A, ally, rcond=None)[0]
    gx = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(2 ** gx, m * gx + c, color=ns.INK2, lw=1.0, zorder=2,
            dashes=(4, 2))
    r = np.corrcoef(lx, ally)[0, 1]

    ax.axhline(0, color=ns.VERM, lw=0.7, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 15, 40])
    ax.set_xticklabels(["2", "4", "8", "15", "40"])
    ax.set_xlabel(f"tiles per frame at {TILE} px")
    ax.set_ylabel("MACs saved (%)")
    ax.text(0.02, 0.94, f"r = {r:.2f} against log tiles", transform=ax.transAxes,
            fontsize=6, color=ns.INK2, va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=6, ncol=5,
              handletextpad=0.2, columnspacing=0.7, borderpad=0.1)
    ns.tidy(ax) if hasattr(ns, "tidy") else None
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    for dd in OUT:
        dd.mkdir(parents=True, exist_ok=True)
        fig.savefig(dd / "granularity.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")

    two = ally[allx <= 2]
    forty = ally[allx >= 40]
    stats = {
        "source": str(src.relative_to(ROOT)),
        "tile_px": TILE,
        "n_points": int(len(ally)),
        "slope_per_doubling": float(m),
        "r_log_tiles": float(r),
        "mean_at_two_tiles": float(two.mean()) if len(two) else None,
        "mean_at_forty_tiles": float(forty.mean()) if len(forty) else None,
        "n_nonpositive": int((ally <= 0).sum()),
        "n_nonpositive_at_two": int((two <= 0).sum()) if len(two) else 0,
    }
    json.dump(stats, open(ROOT / "results/granularity.json", "w"), indent=2)
    print(f"  {len(ally)} sequence-rate points, r = {r:.3f}")
    print(f"  slope {m:.2f} points per doubling of tile count")
    print(f"  2 tiles: mean {stats['mean_at_two_tiles']:.1f}%   "
          f"40 tiles: mean {stats['mean_at_forty_tiles']:.1f}%")
    print(f"  {stats['n_nonpositive']} of {len(ally)} points at or below zero, "
          f"{stats['n_nonpositive_at_two']} of them at two tiles")
    print("  -> docs/figures/granularity.png, paper/figures/granularity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
