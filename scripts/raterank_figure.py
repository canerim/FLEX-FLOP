"""The free baseline against the trained head.

  a  saving at the budget: oracle, trained head, parameter-free rule
  b  what the bit count is actually correlated with
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


def main(out="docs/figures/raterank.png"):
    rr = json.load(open(R / "results/raterank_RECIPE512_b01.json"))
    p = R / "results/router_RECIPE512_b01_fixed.json"
    if not p.exists():
        p = R / "results/router_RECIPE512_b01.json"
    b1 = json.load(open(p))
    B = {r["qp"]: r["saving_pct_vs_release"] for r in b1["rows"]
         if r.get("budget_reachable")}
    rs = [r for r in rr["rows"] if r.get("budget_reachable")]
    q = [r["qp"] for r in rs]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    ax[0].plot(q, [r["oracle_saving_pct_vs_release"] for r in rs], marker="o",
               color=ns.PURPLE, label="A  signalled")
    ax[0].plot(q, [B.get(x) for x in q], marker="s", color=ns.BLUE,
               label="B  144 K head")
    ax[0].plot(q, [r["saving_pct_vs_release"] for r in rs], marker="^",
               color=ns.GREEN, label="bits, 0 params")
    ax[0].fill_between(q, [B.get(x) for x in q],
                       [r["saving_pct_vs_release"] for r in rs],
                       where=[r["saving_pct_vs_release"] > B.get(r["qp"], 0)
                              for r in rs],
                       color=ns.GREEN, alpha=0.13, lw=0, interpolate=True)
    ax[0].set_xlabel("qp"); ax[0].set_ylabel("saved at 0.1 dB (%)")
    ax[0].legend(fontsize=5.4, loc="lower left")

    ax[1].plot(q, [-r["spearman_bits_vs_exit"] for r in rs], marker="o",
               color=ns.ORANGE, label="depth chosen")
    ax[1].plot(q, [r["spearman_bits_vs_spread"] for r in rs], marker="s",
               color=ns.VERM, label="gain from depth")
    ax[1].plot(q, [r["agreement"] for r in rs], marker="^", color=ns.INK2,
               lw=0.8, ls=(0, (3, 2)), label="agreement")
    ax[1].axhline(0.718, color=ns.BLUE, lw=0.7, ls=(0, (1, 2)))
    ax[1].set_xlabel("qp"); ax[1].set_ylabel("Spearman $\\rho$")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=5.4, loc="lower left")
    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(R / out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
