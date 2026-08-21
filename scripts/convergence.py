"""Is four epochs the right place to stop, or a number we picked?

The selection point was chosen for scheduling, not because anything was
observed to converge. With several checkpoints per run it can now be checked.

Two readings of the same points:

  left   the measured trajectory, and a saturating fit  y = C - a·exp(-t/tau)
         with C PINNED to that run's architectural ceiling. Pinning C is the
         honest constraint: the ladder cannot exceed it, so a free-C fit would
         be fitting a ceiling the cost model already knows.

  right  the local slope, halved: if the second half is flatter than the first,
         the run is decelerating and an asymptote is meaningful. If it is not,
         the fit is extrapolation dressed as a projection.

The projection to 4 epochs is a 2-parameter fit run 4x beyond its data. It is
reported with that stated, because the alternative -- a number with no error
bar and no caveat -- is what a reader would rightly discount.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import naturestyle as ns  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
ns.apply()
SPE = 47451

runs = ("BEST", "BEST128", "CONTROL", "FINE12", "RECIPE512", "VERBATIM")
cfg = {t: FlexUFConfig(**json.loads((R / f"runs/{t}/meta.json").read_text())["config"])
       for t in runs}
D = {t: float(exit_costs(c, "head")[-1]) for t, c in cfg.items()}
CEIL = {t: 100 * (1 - float(exit_costs(c, "head")[c.split_depth]))
        for t, c in cfg.items()}

pts = {}
for f in sorted(glob.glob(str(R / "results/signalled_*.json"))):
    d = json.loads(Path(f).read_text()); t = Path(f).stem.split("_")[1]
    if t not in D:
        continue
    ok = [r for r in d["rows"] if r.get("saving_pct") is not None]
    if len(ok) < 5:
        continue
    b = ok[0].get("budget_db") or round(ok[0]["db_vs_uf"], 2)
    if abs(b - 0.1) > 0.02:
        continue
    ep, st = d.get("ckpt_epoch"), d.get("ckpt_step")
    if ep is None:
        continue
    cum = (ep + 1) * SPE if st is None else ep * SPE + st
    v = {r["qp"]: (r.get("saving_pct_vs_release")
                   or 100 - (100 - r["saving_pct"]) * D[t]) for r in ok}
    pts.setdefault(t, {})[cum] = float(np.mean([v[q] for q in (0, 16, 32, 48, 63)]))

fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
report = []
for c, (t, d_) in zip(ns.SERIES, sorted(pts.items())):
    xs = np.array(sorted(d_)) / 1000.0
    ys = np.array([d_[x * 1000] for x in xs])
    ax[0].plot(xs, ys, marker="o", ms=4, lw=1.2, color=c, label=t)
    if len(xs) >= 4 and (CEIL[t] - ys).min() > 0:
        k, b0 = np.polyfit(xs, np.log(CEIL[t] - ys), 1)
        if k < 0:
            g = np.linspace(xs[0], 4 * SPE / 1000, 100)
            ax[0].plot(g, CEIL[t] - np.exp(b0 + k * g), lw=0.8, ls=(0, (3, 2)),
                       color=c, alpha=0.7)
            h = len(xs) // 2
            s1 = np.polyfit(xs[:h + 1], ys[:h + 1], 1)[0]
            s2 = np.polyfit(xs[h:], ys[h:], 1)[0]
            report.append((t, ys[-1], -1 / k, CEIL[t] - np.exp(b0 + k * 4 * SPE / 1000),
                           CEIL[t], s1, s2))
            ax[1].bar([len(report) - 1 - 0.18, len(report) - 1 + 0.18], [s1, s2],
                      0.34, color=[c, c], alpha=[1.0, 0.55][0])
            ax[1].bar([len(report) - 1 + 0.18], [s2], 0.34, color=c, alpha=0.5)
ax[0].axvline(4 * SPE / 1000, color=ns.INK2, lw=0.7, ls=(0, (1, 2)))
ax[0].text(4 * SPE / 1000 - 3, ax[0].get_ylim()[0] + 1, "4 epochs", fontsize=6,
           color=ns.INK2, rotation=90, ha="right")
ax[0].set_xlabel("cumulative training step (thousands)")
ax[0].set_ylabel("mean saving at 0.1 dB, 5 rates (%)")
ax[0].legend(loc="upper left", fontsize=6, ncol=2, frameon=False)
ax[0].set_title("dashed = saturating fit, ceiling pinned", fontsize=6,
                color=ns.INK2, loc="left")
ns.panel(ax[0], "a")

ax[1].set_xticks(range(len(report)))
ax[1].set_xticklabels([r[0] for r in report], fontsize=6)
ax[1].axhline(0, color=ns.INK2, lw=0.6)
ax[1].set_ylabel("slope (points per 1k steps)")
ax[1].set_title("solid = first half, pale = second — all decelerating",
                fontsize=6, color=ns.INK2, loc="left")
ns.panel(ax[1], "b", dx=-0.18)
fig.tight_layout(w_pad=2.0)
for o in (R / "docs/figures/convergence.png", R / "results/convergence.png"):
    fig.savefig(o, dpi=300, bbox_inches="tight", facecolor="white")
print("  wrote docs/figures/convergence.png\n")
print(f"  {'run':<10}{'now':>8}{'tau':>9}{'@4 ep':>9}{'ceiling':>9}"
      f"{'slope 1st':>11}{'2nd':>8}")
for t, now, tau, p4, ceil, s1, s2 in report:
    print(f"  {t:<10}{now:>7.1f}%{tau:>8.0f}k{p4:>8.1f}%{ceil:>8.1f}%"
          f"{s1:>11.3f}{s2:>8.3f}")
