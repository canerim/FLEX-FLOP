"""The rate-quality plane, and the spread behind the mean.

Two things a codec reader asks for that the paper did not have. First, the
operating points on the axes the field is read on -- PSNR against bitrate, with
the released decoder as the reference curve. Second, the spread: every headline
here is a mean over 53 sequences, and a mean without a distribution is an
assertion.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
from savings import sv
ns.apply(ns.for_column())
D = 1.0095
QPS = [0, 16, 32, 48, 63]


def J(*names):
    for n in names:
        p = R / "results" / n
        if p.exists():
            return json.load(open(p))
    return None


anc = J("anchor_RECIPE512_ctc53.json")
why = J("why_qp.json")
sig = J("signalled_RECIPE512_ctc53.json")
pc = J("curve_RECIPE512_ctc53.json")
# One file for both axes, measured together on the same frames.
#
# This plot used to take PSNR from results/anchor_RECIPE512_ctc53.json -- 53
# frames, one per sequence, RECIPE512 -- and the bitrate from
# results/why_qp.json, which is 40 sequences at TWO frames each on a BEST
# checkpoint. Rate from one test set against quality from another, on the plane
# a compression reviewer reads first. The bitrates were 4% low at q0 and 23%
# low at q63, which moves every point of the released curve left and flatters
# the whole picture. results/rd_absolute_PAPER.json measures bpp and PSNR in
# one pass over the paper's own \NumSeq frames and asserts on the weights that
# the two encoders are bit-identical before it reports a shared rate.
_rda = J("rd_absolute_PAPER.json")
if _rda:
    rel = {r["qp"]: r["psnr_release"] for r in _rda["rows"]}
    bpp = {r["qp"]: r["bpp"] for r in _rda["rows"]}
else:
    rel = {r["qp"]: r["stock_psnr"] for r in anc["rows"]}
    bpp = {r["qp"]: r["bpp"] for r in why["rows"]}
qs = [q for q in QPS if q in rel and q in bpp]

# Two panels, not three. The third was a per-sequence violin, which is what
# spread.png already is, at column width and with the sequences labelled. Two
# pictures of one distribution is one too many.
fig, ax = plt.subplots(1, 2, figsize=(ns.W2 * 0.72, 2.2))

# ---- a: the rate-quality plane ---------------------------------------------
ax[0].plot([bpp[q] for q in qs], [rel[q] for q in qs], "-o", ms=4, lw=1.3,
           color=ns.INK, label="released")
for b, c in ((0.1, ns.BLUE), (0.3, ns.ORANGE)):
    rows = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
    ys = [rel[q] - rows[q]["db_vs_uf"] for q in qs if q in rows]
    xs = [bpp[q] for q in qs if q in rows]
    # No saving printed here. A number inside a PNG cannot be checked against
    # the tables, and this legend read "24% saved" beside a table saying 21.5:
    # a different definition averaged over a different set of rates. The
    # caption carries it from the same macro the table does.
    ax[0].plot(xs, ys, "-s", ms=3.4, lw=1.0, color=c, label=f"{b:g} dB budget")
ax[0].set_xlabel("bitrate (bpp)"); ax[0].set_ylabel("PSNR (dB)")
ax[0].legend(fontsize=ns.fs(6), loc="lower right")
ns.panel(ax[0], "a")

# ---- b: the same, quality axis expanded ------------------------------------
for b, c in ((0.1, ns.BLUE), (0.3, ns.ORANGE)):
    rows = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
    xs = [bpp[q] for q in qs if q in rows]
    ax[1].plot(xs, [-rows[q]["db_vs_uf"] for q in qs if q in rows], "-s",
               ms=3.4, lw=1.0, color=c)
    # Alternating above and below the point. Two neighbouring rates sit
    # close enough on this axis that both labels above put 34 on top of 30.
    for _i, q in enumerate(qs):
        if q in rows:
            _up = _i % 2 == 0
            ax[1].annotate(f"{sv(rows[q]):.0f}",
                           (bpp[q], -rows[q]["db_vs_uf"]), fontsize=ns.fs(6),
                           color=c, ha="center", textcoords="offset points",
                           va="bottom" if _up else "top",
                           xytext=(0, 3 if _up else -3))
ax[1].axhline(0, color=ns.INK, lw=1.0)
ax[1].set_xlabel("bitrate (bpp)"); ax[1].set_ylabel("dB vs the release")
ax[1].set_ylim(-0.35, 0.06)
for _a in ax:
    _a.yaxis.set_major_locator(MaxNLocator(nbins=5))
ns.panel(ax[1], "b", dx=-0.24)

# ---- c: the spread across sequences ----------------------------------------
# Panel c removed; the per-sequence statistics it computed are still dumped
# below because the caption of spread.png and the sentence beside it are held to
# them by check_paper.
_ps = json.load(open(R / "results/supp_per_sequence_PAPER_b010.json"))
by = {r["qp"]: [e["saving_pct_vs_release"] for e in r["per_sequence"]]
      for r in _ps["rows"]}
ks = sorted(by)

fig.tight_layout()
for _d in (R / "docs/figures", R / "paper/figures"):
    _d.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d / "rd_spread.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
# The prose beside this panel quoted a median and an IQR that were typed in by
# hand from an older run of this script, under the older saving definition, and
# drifted by half a point when the definition was fixed. Dumped so make_paper_
# tables can turn them into macros and check_paper can hold the sentence to them.
_st = {}
for q in ks:
    v = np.array(by[q])
    _st[str(q)] = {"median": float(np.median(v)),
                   "p25": float(np.percentile(v, 25)),
                   "p75": float(np.percentile(v, 75)),
                   "min": float(v.min()), "max": float(v.max()),
                   "n": int(len(v))}
json.dump({"target_db": 0.1, "rows": _st},
          open(R / "results/rd_spread_stats.json", "w"), indent=2)
print("  -> docs/figures/rd_spread.png, paper/figures/rd_spread.png")
for q in ks:
    v = np.array(by[q])
    print(f"    q{q:<3} n={len(v):>2}  median {np.median(v):5.1f}  "
          f"IQR {np.percentile(v,25):5.1f}-{np.percentile(v,75):5.1f}  "
          f"min {v.min():5.1f}  max {v.max():5.1f}")
