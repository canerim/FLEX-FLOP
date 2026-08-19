"""The trade-off normalised onto each rate's own usable band.

`make_docs_figs.py` already draws the raw curve (docs/figures/tradeoff.png) --
saving against budget, and its inverse. This adds the question that one cannot
answer: once each rate's floor and saturation point are divided out, is what
remains the same curve? If it is, the five rates differ only in where their band
sits, and the operating structure is the whole story.

  a  saving against the budget, one line per rate, floor and saturation marked
  b  the same, with the budget axis rescaled to each rate's own band
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()


def main(src="results/signalled_RECIPE512_grid.json",
         sat="results/saturation_RECIPE512_ctc53.json",
         out="docs/figures/budget_band.png"):
    d = json.load(open(R / src))
    rows = [r for r in d["rows"] if r.get("budget_reachable")]
    qps = sorted({r["qp"] for r in rows})
    cols = [ns.SERIES[i % len(ns.SERIES)] for i in range(len(qps))]

    band = {}
    p = R / sat
    if p.exists():
        for r in json.load(open(p))["rows"]:
            band[r["qp"]] = (r["floor_db"], r["saturation_db"])

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
    for c, q in zip(cols, qps):
        rs = sorted([r for r in rows if r["qp"] == q],
                    key=lambda r: r["budget_db"])
        x = [r["budget_db"] for r in rs]
        y = [r["saving_pct_vs_release"] for r in rs]
        ax[0].plot(x, y, marker="o", ms=2.6, color=c, label=f"qp {q}")
        if q in band:
            f, s = band[q]
            ax[0].plot([f], [0], marker="|", ms=6, color=c)
            ax[0].plot([s], [max(y)], marker="|", ms=6, color=c)
            u = [(v - f) / (s - f) for v in x]
            ax[1].plot(u, y, marker="o", ms=2.6, color=c)
    ax[0].set_xlabel("quality budget (dB below the release)")
    ax[0].set_ylabel("compute saved (%)")
    ax[0].legend(fontsize=5.4, loc="lower right")
    ax[1].set_xlabel("position in the usable band")
    ax[1].set_ylabel("compute saved (%)")
    ax[1].set_xlim(0, 1)
    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(R / out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
