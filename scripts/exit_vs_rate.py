"""Where the ladder is used, as the rate rises.

The headline saving falls from the lowest rate to the highest and the paper says
why in words: high-rate reconstructions carry detail the shallow exits cannot
reproduce. This is that sentence as a picture. The same 0.1 dB budget, the same
ladder, five rates, and the mass of the allocation walking down the ladder.

Panel b puts the three budgets beside each other at one rate, so the two ways of
buying depth -- a looser budget and a lower rate -- can be read off one figure.

    python scripts/exit_vs_rate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns                                    # noqa: E402

ns.apply(ns.for_column())
OUT = (ROOT / "docs" / "figures", ROOT / "paper" / "figures")
RATE_COLS = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]


def hist_for(row):
    """Tiles per exit over the whole set, weighted as the set actually is."""
    h = None
    for c in row["per_class"].values():
        v = np.array(c["hist"], float)
        h = v if h is None else h + v
    return h / h.sum()


def main():
    D = json.load(open(ROOT / "results/supp_per_class_budgets.json"))
    rows = D["rows"]
    qps = sorted({r["qp"] for r in rows})
    K = len(next(iter(rows[0]["per_class"].values()))["hist"])

    # Exits below the split depth do not exist, so their bars are always zero
    # and were taking a third of the axis. Plot the live rungs and say so in
    # the caption.
    live = [i for i in range(K)
            if any(hist_for(r)[i] > 0 for r in rows)]
    lo_i = min(live)
    fig, ax = plt.subplots(1, 2, figsize=(ns.W2 * 0.78, 1.85))

    # a: the 0.1 dB budget, five rates
    at01 = {r["qp"]: r for r in rows if abs(r["budget_db"] - 0.1) < 1e-9}
    w = 0.16
    for n, q in enumerate(qps):
        h = hist_for(at01[q]) * 100
        ax[0].bar(np.arange(len(live)) + (n - 2) * w, h[live], width=w,
                  color=RATE_COLS[n], lw=0, label=f"q{q}")
    ax[0].set_xticks(range(len(live)))
    ax[0].set_xticklabels([f"e{i}" for i in live])
    ax[0].set_xlabel("exit taken")
    ax[0].set_ylabel("tiles (%)")
    ax[0].set_ylim(0, 78)
    ax[0].legend(frameon=False, fontsize=ns.fs(6), ncol=5, handletextpad=0.25,
                 columnspacing=0.6, borderpad=0.05, loc="upper center", bbox_to_anchor=(0.55, 1.14))
    ns.panel(ax[0], "a")

    # b: one rate, the three budgets
    q_mid = qps[len(qps) // 2]
    buds = sorted({r["budget_db"] for r in rows})
    cols = [ns.BLUE, ns.ORANGE, ns.GREEN if hasattr(ns, "GREEN") else "#2a9d5c"]
    w = 0.24
    for n, b in enumerate(buds):
        r = next(r for r in rows
                 if r["qp"] == q_mid and abs(r["budget_db"] - b) < 1e-9)
        h = hist_for(r) * 100
        ax[1].bar(np.arange(len(live)) + (n - 1) * w, h[live], width=w,
                  color=cols[n], lw=0, label=f"{b:g} dB")
    ax[1].set_xticks(range(len(live)))
    ax[1].set_xticklabels([f"e{i}" for i in live])
    ax[1].set_xlabel(f"exit taken, q{q_mid}")
    ax[1].legend(frameon=False, fontsize=ns.fs(6), ncol=3, handletextpad=0.25,
                 columnspacing=0.6, borderpad=0.05, loc="upper center", bbox_to_anchor=(0.6, 1.14))
    ax[1].set_ylim(0, 112)
    ns.panel(ax[1], "b")

    for a_ in ax:
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)

    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "exit_vs_rate.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")

    lo, hi = hist_for(at01[qps[0]]), hist_for(at01[qps[-1]])
    mean_lo = float((np.arange(K) * lo).sum())
    mean_hi = float((np.arange(K) * hi).sum())
    st = {"budget_db": 0.1, "K": K,
          "mean_exit_low_rate": mean_lo, "mean_exit_high_rate": mean_hi,
          "share_two_low": float(100 * lo[2]), "share_two_high": float(100 * hi[2]),
          "share_deepest_low": float(100 * lo[-1]),
          "share_deepest_high": float(100 * hi[-1]),
          "qp_low": qps[0], "qp_high": qps[-1]}
    json.dump(st, open(ROOT / "results/exit_vs_rate.json", "w"), indent=2)
    print(f"  mean exit {mean_lo:.2f} at q{qps[0]} -> {mean_hi:.2f} at "
          f"q{qps[-1]}")
    print(f"  share at the shallowest live exit: {100*lo[2]:.1f}% -> "
          f"{100*hi[2]:.1f}%")
    print("  -> docs/figures/exit_vs_rate.png, paper/figures/exit_vs_rate.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
