"""How much of the collapsed trade-off curve survives a change of sweep.

Section E of the supplement rescales the budget axis onto each rate's own
window and fits one exponent to what is left. The exponent is quoted twice in
the main paper, once per training run, and the difference between the two is
read as a property of training. This figure prices that reading against every
other choice the fit rests on.

  a  the rescaled points from two sweeps of the same pinned checkpoint, with
     the power law fitted to each; they are not the same curve
  b  the exponent under one change at a time, against the exponent the change
     of checkpoint produces

Reads results/signalled_RECIPE512_grid.json, results/supp_paper_curve_PAPER.json
and results/saturation_RECIPE512_ctc53.json on runs/RECIPE512/ckpt_PAPER.pth.tar,
and results/signalled_BEST_grid.json with results/saturation_BEST_ctc53.json on
runs/BEST/ckpt_eval.pth.tar. The fit is imported from the section module, so the
figure and the table beside it cannot disagree. Writes
docs/figures/band_sensitivity_PAPER.png. No GPU.
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
from supp.e_budget import _curves, _fit, _win, QPS, WLO, WHI  # noqa: E402

ns.apply()


def _J(name):
    return json.load(open(R / "results" / name))


def main(out="docs/figures/band_sensitivity_PAPER.png"):
    sat = _J("saturation_RECIPE512_ctc53.json")
    S = {r["qp"]: r for r in sat["rows"]}
    C = sat["ceiling_pct"]
    cK = _J("router_RECIPE512_b01.json")["deepest_exit_cost"]
    grid = _J("signalled_RECIPE512_grid.json")
    front = _J("supp_paper_curve_PAPER.json")
    gB, sB = _J("signalled_BEST_grid.json"), _J("saturation_BEST_ctc53.json")

    cur = _curves(grid, sat)
    cur3 = {}
    for q in QPS:
        pts = []
        for o in front["op_points"]:
            if (o["qp"] != q or o.get("saturated")
                    or o.get("budget_reachable") is False):
                continue
            f, s = S[q]["floor_db"], S[q]["saturation_db"]
            pts.append(((o["db_vs_uf_per_frame"] - f) / (s - f),
                        100 * (1 - cK * (1 - o["saving_pct"] / 100))))
        cur3[q] = pts

    def rowcurve(key):
        c = {}
        for q in QPS:
            f, s = S[q]["floor_db"], S[q]["saturation_db"]
            c[q] = [((r[key] - f) / (s - f), r["saving_pct_vs_release"])
                    for r in front["rows"] if r["qp"] == q]
        return c

    curF = {}
    for q in QPS:
        rs = sorted((r for r in grid["rows"]
                     if r["qp"] == q and r.get("budget_reachable")),
                    key=lambda r: r["budget_db"])
        f, s = rs[0]["floor_db"], S[q]["saturation_db"]
        curF[q] = [(u, r["saving_pct_vs_release"]) for r, u in
                   ((r, (r["budget_db"] - f) / (s - f)) for r in rs)
                   if 0.0 <= u <= 1.0]

    b_grid = _fit(_win(cur), C)[0]
    b_front = _fit(_win(cur3), C)[0]
    bars = [
        ("second checkpoint", _fit(_win(_curves(gB, sB)), sB["ceiling_pct"])[0]),
        ("floor from the grid", _fit(_win(curF), C)[0]),
        ("ceiling C = %.1f" % sB["ceiling_pct"], _fit(_win(cur), sB["ceiling_pct"])[0]),
        ("pooled budget axis", _fit(_win(rowcurve("db_vs_uf")), C)[0]),
        ("per-frame budget axis", _fit(_win(rowcurve("db_vs_uf_per_frame")), C)[0]),
        ("second sweep, 53 frames", b_front),
    ]

    cols = [ns.SERIES[i % len(ns.SERIES)] for i in range(len(QPS))]
    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 1.62))

    for q, c in zip(QPS, cols):
        x = [p[0] for p in cur[q]]
        y = [p[1] for p in cur[q]]
        ax[0].plot(x, y, ls="none", marker="o", ms=2.4, color=c,
                   label=f"q{q}")
        x3 = [p[0] for p in cur3[q]]
        y3 = [p[1] for p in cur3[q]]
        ax[0].plot(x3, y3, ls="none", marker="^", ms=2.6, mfc="none",
                   mew=0.6, color=c)
    u = np.linspace(0.05, 1.0, 100)
    ax[0].plot(u, C * u ** b_grid, color=ns.INK, lw=0.8,
               label="β = %.2f" % b_grid)
    ax[0].plot(u, C * u ** b_front, color=ns.INK, lw=0.8, ls=(0, (3, 2)),
               label="β = %.2f" % b_front)
    ax[0].axvspan(WLO, WHI, color=ns.GRID, alpha=0.45, lw=0, zorder=0)
    ax[0].set_xlabel("position u in the window")
    ax[0].set_ylabel("compute saved (%)")
    ax[0].legend(ncol=2, handlelength=1.0, handletextpad=0.4,
                 columnspacing=0.6, borderpad=0.1, labelspacing=0.15,
                 loc="lower right")
    ns.panel(ax[0], "a")

    ypos = np.arange(len(bars))
    ax[1].barh(ypos, [b for _, b in bars], height=0.62, color=ns.SKY,
               edgecolor=ns.BLUE, lw=0.4)
    ax[1].axvline(b_grid, color=ns.INK, lw=0.8)
    for y, (_, b) in zip(ypos, bars):
        ax[1].text(b + 0.008, y, "%.2f" % b, va="center", fontsize=6,
                   color=ns.INK)
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels([n for n, _ in bars], fontsize=6)
    ax[1].set_xlim(0, 0.52)
    ax[1].set_xlabel("fitted exponent β")
    ax[1].grid(axis="y", visible=False)
    ns.panel(ax[1], "b", dx=-0.62)

    fig.tight_layout(pad=0.25, w_pad=1.2)
    (R / out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(R / out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"  -> {out}  baseline beta {b_grid:.3f}, second sweep {b_front:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
