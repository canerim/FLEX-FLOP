"""Two figures for what a set mean hides.

Both replace tables that report a distribution by its mean. A mean over 53
sequences of six different resolutions is close to the least informative number
those measurements can produce, and the paper quotes it repeatedly.

  spread.png   every sequence as a point, per rate
  exituse.png  how deep each test class actually has to go

    python scripts/spread_figs.py
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
import naturestyle as ns  # noqa: E402

ns.apply()
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
RATE_COLS = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=5.4, length=2, width=0.5)
    ax.xaxis.label.set_size(5.8)
    ax.yaxis.label.set_size(5.8)
    return ax


def short(name):
    return name.split("_")[0]


def spread():
    S = json.loads((RES / "per_sequence.json").read_text())
    rows = sorted(S["rows"], key=lambda r: r["qp"])
    fig, ax = plt.subplots(figsize=(ns.W1, 1.72))
    tidy(ax)
    rng = np.random.default_rng(0)
    for n, r in enumerate(rows):
        v = np.array([p["saving_pct"] for p in r["per_sequence"]])
        x = n + rng.uniform(-0.17, 0.17, size=v.size)
        ax.scatter(x, v, s=5, color=RATE_COLS[n % len(RATE_COLS)], lw=0,
                   alpha=0.85, zorder=3)
        ax.plot([n - 0.30, n + 0.30], [r["median"]] * 2, color=ns.INK, lw=1.0,
                zorder=4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"q{r['qp']}" for r in rows])
    ax.set_ylabel("decoder MACs saved (%)")
    ax.set_xlabel("quality index")
    lo, hi = rows[0], rows[-1]
    ax.annotate(short(hi["worst_seq"]), (len(rows) - 1, hi["min"]),
                textcoords="offset points", xytext=(8, -2), fontsize=4.6,
                color=ns.INK2)
    ax.annotate(short(lo["best_seq"]), (0, lo["max"]),
                textcoords="offset points", xytext=(8, -2), fontsize=4.6,
                color=ns.INK2)
    ax.text(0.02, 0.05, "bar is the median", transform=ax.transAxes,
            fontsize=4.8, color=ns.INK2)
    fig.savefig(FIG / "spread.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    print(f"  wrote spread.png   q0 spans {rows[0]['min']:.1f} to "
          f"{rows[0]['max']:.1f}%, q63 spans {rows[-1]['min']:.1f} to "
          f"{rows[-1]['max']:.1f}%")


def exituse():
    P = json.loads((RES / "supp_per_class_budgets.json").read_text())
    rows = [r for r in P["rows"] if abs(r["budget_db"] - 0.1) < 1e-9]
    qs = sorted({r["qp"] for r in rows})
    classes = list(rows[0]["per_class"].keys())
    K = len(rows[0]["per_class"][classes[0]]["hist"])
    M = np.full((len(classes), len(qs)), np.nan)
    for ci, c in enumerate(classes):
        for qi, q in enumerate(qs):
            r = next((x for x in rows if x["qp"] == q), None)
            if not r or c not in r["per_class"]:
                continue
            h = np.array(r["per_class"][c]["hist"], dtype=float)
            if h.sum():
                M[ci, qi] = (h * np.arange(K)).sum() / h.sum()
    order = np.argsort(np.nanmean(M, axis=1))
    res = {c: rows[0]["per_class"][c]["res"] for c in classes
           if c in rows[0]["per_class"]}
    classes = [classes[i] for i in order]
    M = M[order]

    fig, ax = plt.subplots(figsize=(ns.W1, 1.5))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=2, vmax=5)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"q{q}" for q in qs], fontsize=5.4)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c}  {res.get(c, '')}" for c in classes], fontsize=4.8)
    ax.tick_params(length=0)
    for ci in range(len(classes)):
        for qi in range(len(qs)):
            if np.isnan(M[ci, qi]):
                continue
            ax.text(qi, ci, f"{M[ci, qi]:.1f}", ha="center", va="center",
                    fontsize=4.6,
                    color="white" if M[ci, qi] > 3.6 else ns.INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.tick_params(labelsize=4.6, length=1.5, width=0.4)
    cb.set_label("mean exit taken", fontsize=5.2)
    ax.set_xlabel("quality index", fontsize=5.8)
    fig.savefig(FIG / "exituse.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    print(f"  wrote exituse.png   mean exit {np.nanmin(M):.2f} to "
          f"{np.nanmax(M):.2f} across {len(classes)} classes")


if __name__ == "__main__":
    spread()
    exituse()
