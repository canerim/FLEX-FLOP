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
import naturestyle as ns
from savings import sv, pick  # noqa: E402

ns.apply()


def main(out="docs/figures/raterank.png"):
    rr = json.load(open(R / "results/raterank_RECIPE512_b01.json"))
    # The same choice make_paper_tables makes, by provenance rather than by a
    # hand-written order. This script preferred router_..._b01_fixed.json and
    # drew configuration B at 27.2% at q0 while the table printed 23.6 -- a
    # different head, measured before the checkpoint was pinned.
    b1, _src = pick("router_RECIPE512_b01_PAPER.json",
                    "router_RECIPE512_b01_fixed.json",
                    "router_RECIPE512_b01.json")
    print(f"  configuration B from {_src}")
    # The canonical definition, the hook count where the file has one. Drawn
    # from saving_pct_vs_release these curves sat three points above the
    # tables they are read beside.
    B = {r["qp"]: sv(r) for r in b1["rows"]
         if r.get("budget_reachable")}
    rs = [r for r in rr["rows"] if r.get("budget_reachable")]
    q = [r["qp"] for r in rs]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    ax[0].plot(q, [sv(r, "oracle_saving_pct") for r in rs], marker="o",
               color=ns.PURPLE, label="A  signalled")
    ax[0].plot(q, [B.get(x) for x in q], marker="s", color=ns.BLUE,
               label="B  144 K head")
    ax[0].plot(q, [sv(r) for r in rs], marker="^",
               color=ns.GREEN, label="bits, 0 params")
    ax[0].fill_between(q, [B.get(x) for x in q],
                       [sv(r) for r in rs],
                       where=[sv(r) > B.get(r["qp"], 0)
                              for r in rs],
                       color=ns.GREEN, alpha=0.13, lw=0, interpolate=True)
    ax[0].set_xlabel("qp"); ax[0].set_ylabel("saved at 0.1 dB (%)")
    ax[0].legend(fontsize=6, loc="lower left")

    ax[1].plot(q, [-r["spearman_bits_vs_exit"] for r in rs], marker="o",
               color=ns.ORANGE, label="depth chosen")
    ax[1].plot(q, [r["spearman_bits_vs_spread"] for r in rs], marker="s",
               color=ns.VERM, label="gain from depth")
    ax[1].plot(q, [r["agreement"] for r in rs], marker="^", color=ns.INK2,
               lw=0.8, ls=(0, (3, 2)), label="agreement")
    ax[1].axhline(0.718, color=ns.BLUE, lw=0.7, ls=(0, (1, 2)),
                 label="head's held-out agreement")
    ax[1].set_xlabel("qp"); ax[1].set_ylabel("Spearman $\\rho$")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=6, loc="lower left")
    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    for _d in (R / "docs/figures", R / "paper/figures"):
        _d.mkdir(parents=True, exist_ok=True)
        fig.savefig(_d / "raterank.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
