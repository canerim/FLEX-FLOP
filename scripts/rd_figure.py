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

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply()

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
rel = {r["qp"]: r["stock_psnr"] for r in anc["rows"]}
bpp = {r["qp"]: r["bpp"] for r in why["rows"]}
qs = [q for q in QPS if q in rel and q in bpp]

fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.4))

# ---- a: the rate-quality plane ---------------------------------------------
ax[0].plot([bpp[q] for q in qs], [rel[q] for q in qs], "-o", ms=4, lw=1.3,
           color=ns.INK, label="released")
for b, c in ((0.1, ns.BLUE), (0.3, ns.ORANGE)):
    rows = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
    ys = [rel[q] - rows[q]["db_vs_uf"] for q in qs if q in rows]
    xs = [bpp[q] for q in qs if q in rows]
    sv = [rows[q]["saving_pct_vs_release"] for q in qs if q in rows]
    ax[0].plot(xs, ys, "-s", ms=3.4, lw=1.0, color=c,
               label=f"{b:g} dB, {np.mean(sv):.0f}% saved")
ax[0].set_xlabel("bitrate (bpp)"); ax[0].set_ylabel("PSNR (dB)")
ax[0].legend(fontsize=5.5, loc="lower right")
ns.panel(ax[0], "a")

# ---- b: the same, quality axis expanded ------------------------------------
for b, c in ((0.1, ns.BLUE), (0.3, ns.ORANGE)):
    rows = {r["qp"]: r for r in sig["rows"]
            if abs(r["budget_db"] - b) < 1e-9 and r.get("budget_reachable")}
    xs = [bpp[q] for q in qs if q in rows]
    ax[1].plot(xs, [-rows[q]["db_vs_uf"] for q in qs if q in rows], "-s",
               ms=3.4, lw=1.0, color=c)
    for q in qs:
        if q in rows:
            ax[1].annotate(f"{rows[q]['saving_pct_vs_release']:.0f}",
                           (bpp[q], -rows[q]["db_vs_uf"]), fontsize=5,
                           color=c, ha="center", textcoords="offset points",
                           xytext=(0, 3))
ax[1].axhline(0, color=ns.INK, lw=1.0)
ax[1].set_xlabel("bitrate (bpp)"); ax[1].set_ylabel("dB vs the release")
ax[1].set_ylim(-0.35, 0.06)
ns.panel(ax[1], "b", dx=-0.24)

# ---- c: the spread across sequences ----------------------------------------
by = {}
for q in QPS:
    ops = [o for o in pc["op_points"] if o["qp"] == q and o.get("per_sequence")]
    if not ops:
        continue
    o = min(ops, key=lambda o: abs(o.get("target_db", 9) - 0.1))
    by[q] = [100 - (100 - s["saving_pct"]) * D for s in o["per_sequence"]]
ks = sorted(by)
parts = ax[2].violinplot([by[q] for q in ks], positions=range(len(ks)),
                         widths=0.75, showextrema=False, showmedians=True)
for b_ in parts["bodies"]:
    b_.set_facecolor(ns.SKY); b_.set_alpha(0.55); b_.set_edgecolor("none")
parts["cmedians"].set_color(ns.INK); parts["cmedians"].set_linewidth(1.1)
for i, q in enumerate(ks):
    v = np.array(by[q])
    ax[2].plot(np.full(len(v), i) + np.random.default_rng(q).uniform(-.16, .16,
               len(v)), v, ".", ms=1.8, color=ns.INK2, alpha=0.55)
ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels([f"q{q}" for q in ks])
ax[2].set_ylabel("MACs saved (%)")
ns.panel(ax[2], "c", dx=-0.24)

fig.tight_layout()
fig.savefig(R / "docs/figures/rd_spread.png", dpi=300)
print("  -> docs/figures/rd_spread.png")
for q in ks:
    v = np.array(by[q])
    print(f"    q{q:<3} n={len(v):>2}  median {np.median(v):5.1f}  "
          f"IQR {np.percentile(v,25):5.1f}-{np.percentile(v,75):5.1f}  "
          f"min {v.min():5.1f}  max {v.max():5.1f}")
