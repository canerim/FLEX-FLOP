"""Configuration C: what a partial signalling budget buys.

Two panels, no captions inside them.
  a  saving against bits/frame, one line per qp, A and B as the endpoints
  b  fraction of the A-B gap recovered against fraction of the map sent
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()


def main(src=None, out="docs/figures/hybrid.png", lor=None):
    def _pick(*names):
        for n in names:
            if (R / n).exists():
                return R / n
        return R / names[-1]
    src = src or _pick("results/hybrid_RECIPE512_b01_fixed.json",
                       "results/hybrid_RECIPE512_b01.json")
    lor = lor or _pick("results/hybrid_lorenz_b01_fixed.json",
                       "results/hybrid_lorenz_b01.json")
    d = json.load(open(src))
    L = {}
    if Path(lor).exists():
        for r in json.load(open(lor))["rows"]:
            if r.get("lorenz_at_rho") is not None:
                L[(r["qp"], round(r["rho"], 6))] = 100 * r["lorenz_at_rho"]
    rows = [r for r in d["rows"] if r.get("budget_reachable")]
    qps = sorted({r["qp"] for r in rows})
    cols = [ns.SERIES[i % len(ns.SERIES)] for i in range(len(qps))]

    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.5))
    for c, q in zip(cols, qps):
        rs = sorted([r for r in rows if r["qp"] == q], key=lambda r: r["rho"])
        b = [r["map_bits"] for r in rs]
        s = [r["saving_pct_vs_release"] for r in rs]
        ax[0].plot(b, s, marker="o", ms=3, color=c, label=f"qp {q}")
        lo, hi = s[0], s[-1]
        if hi - lo > 1e-9:
            ax[1].plot([r["rho"] for r in rs],
                       [100 * (v - lo) / (hi - lo) for v in s],
                       marker="o", ms=3, color=c, label=f"qp {q}")
        if L:
            xy = [(L[(q, round(r["rho"], 6))],
                   100 * (r["saving_pct_vs_release"] - lo) / (hi - lo))
                  for r in rs if (q, round(r["rho"], 6)) in L and hi - lo > 1e-9]
            if xy:
                ax[2].plot(*zip(*sorted(xy)), marker="o", ms=3, color=c)
    ax[1].plot([0, 1], [0, 100], color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    ax[2].plot([0, 100], [0, 100], color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    ax[2].set_xlabel("Lorenz bound (%)")
    ax[2].set_ylabel("measured recovery (%)")
    ax[0].set_xlabel("signalled bits per frame")
    ax[0].set_ylabel(f"saved at {d['budget_db']:g} dB (%)")
    ax[0].legend(fontsize=5, frameon=False, loc="lower right")
    ax[1].set_xlabel("fraction of tiles signalled")
    ax[1].set_ylabel("gap to A recovered (%)")
    for i, l in enumerate("abc"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    p = R / out
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
