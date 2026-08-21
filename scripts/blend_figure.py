"""Blending the two decoder-side signals.

One panel: saving against the blend weight, one line per rate. The two ends are
the parameter-free rule and the head's own ordering.
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


def main(src="results/combined_RECIPE512_b01.json",
         out="docs/figures/blend.png"):
    d = json.load(open(R / src))
    rows = [r for r in d["rows"] if r.get("budget_reachable")]
    qps = sorted({r["qp"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    for i, q in enumerate(qps):
        rs = sorted([r for r in rows if r["qp"] == q], key=lambda r: r["gamma"])
        c = ns.SERIES[i % len(ns.SERIES)]
        x = [r["gamma"] for r in rs]
        y = [r["saving_pct_vs_release"] for r in rs]
        ax[0].plot(x, y, marker="o", ms=3, color=c, label=f"qp {q}")
        ax[1].plot(x, [v - y[0] for v in y], marker="o", ms=3, color=c)
    ax[1].axhline(0, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    for a_ in ax:
        a_.set_xlabel("blend weight $w$   (0 = bits, 1 = head)")
    ax[0].set_ylabel("saved at 0.1 dB (%)")
    ax[1].set_ylabel("change from $w{=}0$ (points)")
    ax[0].legend(fontsize=6, loc="lower left")
    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(R / out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
