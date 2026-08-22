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
from savings import sv
import naturestyle as ns  # noqa: E402

ns.apply(ns.for_column())
def main(src="results/signalled_RECIPE512_grid.json",
         sat="results/saturation_RECIPE512_ctc53.json",
         out="docs/figures/budget_band.png",
         stats_out="results/band_collapse.json"):
    d = json.load(open(R / src))
    rows = [r for r in d["rows"] if r.get("budget_reachable")]
    qps = sorted({r["qp"] for r in rows})
    cols = [ns.SERIES[i % len(ns.SERIES)] for i in range(len(qps))]

    band, CEIL = {}, None
    p = R / sat
    if p.exists():
        sd = json.load(open(p))
        # The saturation file carries the arithmetic ceiling. The curves here
        # are hook-counted, so their asymptote is the measured one; fitting a
        # measured curve to a modelled ceiling put the fit 0.8 points above
        # every point it was fitted to.
        CEIL = sd.get("ceiling_pct")
        _cm = R / "results/ceiling_measured.json"
        if _cm.exists():
            CEIL = json.load(open(_cm))["ceiling_measured_pct"]
        for r in sd["rows"]:
            band[r["qp"]] = (r["floor_db"], r["saturation_db"])

    # How much of the rate dependence is the band, and how much is left over?
    # At a matched decibel the five rates are far apart; at a matched POSITION
    # in their own band they should coincide if the floor and the saturation
    # point are the whole story.
    import numpy as _np
    cur = {}
    for q in qps:
        rs = sorted([r for r in rows if r["qp"] == q],
                    key=lambda r: r["budget_db"])
        if q not in band:
            continue
        f, sa = band[q]
        u = _np.array([(r["budget_db"] - f) / (sa - f) for r in rs])
        yv = _np.array([sv(r) for r in rs])
        m = (u >= 0) & (u <= 1.0001)
        cur[q] = (u[m], yv[m])
    stats = {}
    if len(cur) > 1:
        g = _np.linspace(0.05, 0.95, 19)
        Y = _np.array([_np.interp(g, *cur[q]) for q in cur])
        sp = Y.max(0) - Y.min(0)
        raw = [sv(r) for r in rows
               if abs(r["budget_db"] - 0.1) < 1e-9]
        # And a one-parameter description of the collapsed curve. The ceiling
        # C is architectural and in closed form; u is the position in the band.
        # Fitted in log space, so it is a straight line fit and not an
        # optimiser's opinion.
        Uc = _np.concatenate([cur[q][0] for q in cur])
        Yc = _np.concatenate([cur[q][1] for q in cur])
        keep = (Uc > 1e-6) & (Yc > 0)
        Uc, Yc = _np.minimum(Uc[keep], 1.0), Yc[keep]
        C = CEIL
        lu, ly = _np.log(Uc), _np.log(Yc / C)
        pw = float((lu @ ly) / (lu @ lu))
        pred = C * Uc ** pw
        r2 = float(1 - ((Yc - pred) ** 2).sum()
                   / ((Yc - Yc.mean()) ** 2).sum())
        stats = {"power_exponent": pw, "power_r2": r2,
                 "power_max_err": float(_np.abs(pred - Yc).max()),
                 "power_mean_err": float(_np.abs(pred - Yc).mean()),
                 "power_n": int(Uc.size), "ceiling_pct": C,
                 "band_spread_max": float(sp.max()),
                 "band_spread_mean": float(sp.mean()),
                 "band_spread_max_excl_edge": float(sp[1:].max()),
                 "raw_spread_at_tenth_db": float(max(raw) - min(raw)),
                 "n_rates": len(cur)}
        json.dump(stats, open(R / stats_out, "w"), indent=2)
        print(f"  collapsed curve: saving = {C:.2f} * u^{pw:.3f}, "
              f"R2 = {r2:.4f}, max err {stats['power_max_err']:.2f} points")
        print(f"  at a matched dB the rates spread by "
              f"{stats['raw_spread_at_tenth_db']:.1f} points; at a matched "
              f"position in their own band, {stats['band_spread_mean']:.1f} on "
              f"average and {stats['band_spread_max_excl_edge']:.1f} at worst")
        print(f"  -> {stats_out}")

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
    for c, q in zip(cols, qps):
        rs = sorted([r for r in rows if r["qp"] == q],
                    key=lambda r: r["budget_db"])
        x = [r["budget_db"] for r in rs]
        y = [sv(r) for r in rs]
        ax[0].plot(x, y, marker="o", ms=2.6, color=c, label=f"qp {q}")
        if q in band:
            f, s = band[q]
            ax[0].plot([f], [0], marker="|", ms=6, color=c)
            ax[0].plot([s], [max(y)], marker="|", ms=6, color=c)
            u = [(v - f) / (s - f) for v in x]
            ax[1].plot(u, y, marker="o", ms=2.6, color=c)
    ax[0].set_xlabel("quality budget (dB below the release)")
    ax[0].set_ylabel("compute saved (%)")
    # The tick marks had no key: one at y=0 for each rate's floor, one at the
    # top for its saturation point. A reader should not have to infer that.
    # Bottom left, clear of the rate labels that sit along the curves.
    ax[0].text(0.02, 0.06, "ticks: floor and saturation, one pair per rate",
               transform=ax[0].transAxes, fontsize=ns.fs(6),
               color=ns.INK2, va="top", linespacing=1.3)
    ax[0].legend(fontsize=ns.fs(6), loc="lower right")
    ax[1].set_xlabel("position in the usable band")
    ax[1].set_ylabel("compute saved (%)")
    ax[1].set_xlim(0, 1)
    if stats:
        gg = _np.linspace(0.02, 1.0, 100)
        ax[1].plot(gg, stats["ceiling_pct"] * gg ** stats["power_exponent"],
                   color=ns.INK2, lw=0.8, ls=(0, (3, 2)),
                   label="fitted power law")
        # After the line is drawn, not before it: called earlier the legend had
        # nothing to list and matplotlib drew an empty box.
        ax[1].legend(fontsize=ns.fs(6), loc="lower right", frameon=False)
    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    # The name comes from the argument. It used to be the literal
    # "budget_band.png" in a loop over both figure directories, so the run
    # that draws the second training run's collapse overwrote the main
    # paper's figure with it and reported the name it had not written. The
    # paper carried the wrong picture under a caption that said otherwise.
    # naturestyle's savefig mirrors into the twin directory and records the
    # overlap audit under the real name, so one call is enough.
    _out = R / out
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out, dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    print(f"  -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
