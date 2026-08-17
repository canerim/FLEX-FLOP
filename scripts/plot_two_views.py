"""The two views the result should be read through, and the law behind them.

Left  — fix the SAVING, watch the dB cost rise with QP. This is the view that
        answers "what does 20% cost me?", and the answer depends strongly on
        where on the rate curve you stand.
Right — fix the dB budget, watch the saving fall with QP. Same surface, cut the
        other way; this is the view a deployment would use.

Both are measured against RELEASED DCVC-UF on CTC, same bitstream, same latent.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SUR, INK, INK2, INK3, GC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e2"
def sty(a, t, sub=None):
    a.set_facecolor(SUR); a.set_title(t, color=INK, fontsize=11.5, loc="left", pad=17 if sub else 8)
    if sub: a.text(0, 1.012, sub, transform=a.transAxes, color=INK3, fontsize=8.2, va="bottom")
    a.grid(True, color=GC, lw=0.7, zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=8.5, length=0)
    for k, sp in a.spines.items(): sp.set_visible(k == "bottom"); sp.set_color(GC)

rows = json.load(open("results/paper_curve_grid128.json"))["rows"]
qps = sorted({r["qp"] for r in rows})

def db_at_saving(qp, s):
    """Least dB that reaches at least `s`% saving -- the frontier read vertically."""
    c = [r for r in rows if r["qp"] == qp and r["saving_pct"] >= s]
    return min(c, key=lambda r: r["db_vs_uf"])["db_vs_uf"] if c else np.nan

def saving_at_db(qp, d):
    c = [r for r in rows if r["qp"] == qp and r["db_vs_uf"] <= d]
    return max(c, key=lambda r: r["saving_pct"])["saving_pct"] if c else np.nan

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 5.9), facecolor=SUR)
fig.suptitle("FLEX-UF — ayni yuzey, iki kesit", fontsize=15.5, color=INK, x=0.04, ha="left", y=0.97)
fig.text(0.04, 0.9,
         "Yayinlanmis DCVC-UF referansli, CTC, ayni bitstream ve latent (encoder donuk, max|diff| = 0). "
         "ORACLE: kusursuz secim varsayimi.", color=INK2, fontsize=9, ha="left")

for c, s in zip([S[0], S[2], S[1]], (15, 20, 30)):
    ys = [db_at_saving(q, s) for q in qps]
    ax.plot(qps, ys, color=c, lw=2.2, marker="o", ms=7, label=f"%{s} tasarruf",
            markeredgecolor=SUR, markeredgewidth=1.5, zorder=3)
    for q, y in zip(qps, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.3f}", (q, y), textcoords="offset points", xytext=(0, 8),
                        color=c, fontsize=7.5, ha="center")
ax.axhline(0.1, color=INK3, lw=1.3, ls=(0, (4, 3)), zorder=2)
ax.text(63, 0.108, "0.1 dB", color=INK3, fontsize=8.5, ha="right")
ax.set_xticks(qps); ax.set_xlabel("QP (dusuk = dusuk bit hizi)", color=INK2, fontsize=9.5)
ax.set_ylabel("yayinlanmis DCVC-UF'in ALTINDA, dB", color=INK2, fontsize=9.5)
ax.legend(fontsize=9, frameon=False, loc="upper left")
sty(ax, "1 · Tasarruf sabit -> dB maliyeti", "Ayni tasarruf, yuksek QP'de birkac kat pahali.")

for c, d in zip([S[3], S[4], S[0]], (0.1, 0.2, 0.3)):
    ys = [saving_at_db(q, d) for q in qps]
    bx.plot(qps, ys, color=c, lw=2.2, marker="s", ms=7, label=f"{d} dB butce",
            markeredgecolor=SUR, markeredgewidth=1.5, zorder=3)
    for q, y in zip(qps, ys):
        if not np.isnan(y):
            bx.annotate(f"{y:.0f}%", (q, y), textcoords="offset points", xytext=(0, 8),
                        color=c, fontsize=7.5, ha="center")
bx.set_xticks(qps); bx.set_xlabel("QP", color=INK2, fontsize=9.5)
bx.set_ylabel("decoder hesaplama tasarrufu (%)", color=INK2, fontsize=9.5)
bx.set_ylim(0, 46); bx.legend(fontsize=9, frameon=False, loc="upper right")
sty(bx, "2 · dB sabit -> tasarruf", "Ayni butce, yuksek QP'de cok daha az kazandiriyor.")

fig.tight_layout(rect=[0, 0.02, 1, 0.875])
fig.savefig("results/two_views.png", dpi=135, facecolor=SUR)
print("wrote results/two_views.png")
