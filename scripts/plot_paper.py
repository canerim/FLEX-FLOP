"""The paper figure: saving against RELEASED DCVC-UF, per QP.

Every earlier plot measured "dB lost" against our own deepest exit, which has
drifted from the released decoder. This one uses the released decoder as the
reference throughout, so a point at (0.08 dB, 26%) means exactly what a reader
would assume: 26% less decoder compute, 0.08 dB below published DCVC-UF, same
bitstream.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, INK3, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e2"
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/paper_curve_grid128.json"))
rows = d["rows"]
qps = sorted({r["qp"] for r in rows})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle("FLEX-UF — yayinlanmis DCVC-UF'e gore tasarruf", fontsize=15,
             color=INK, x=0.045, ha="left", y=0.97)
fig.text(0.045, 0.905,
         "Ayni bitstream, ayni latent (encoder donuk, max|diff| = 0). Fark yalniz sentez. "
         "ORACLE: kusursuz router varsayimi — ust sinir, basari degil.",
         color=INK2, fontsize=9, ha="left")

for a in (ax, bx):
    a.set_facecolor(SURFACE); a.grid(True, color=GRIDC, lw=0.7, zorder=0)
    a.set_axisbelow(True); a.tick_params(colors=INK2, labelsize=8.5, length=0)
    for k, sp in a.spines.items():
        sp.set_visible(k == "bottom"); sp.set_color(GRIDC)

for c, qp in zip(S, qps):
    pts = sorted([(r["db_vs_uf"], r["saving_pct"]) for r in rows if r["qp"] == qp])
    pts = [p for p in pts if -0.02 <= p[0] <= 0.42]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2,
            marker="o", ms=3.5, label=f"qp {qp}", zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
ax.axvline(0.1, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(0.104, 1, "0.1 dB", color=INK3, fontsize=8.5, rotation=90, va="bottom")
ax.axhspan(30, 40, color=S[3], alpha=0.10, zorder=1)
ax.text(0.30, 35, "hedef %30-40", color=INK3, fontsize=8.5, va="center")
ax.set_xlabel("yayinlanmis DCVC-UF'in ALTINDA, dB", color=INK2, fontsize=9.5)
ax.set_ylabel("decoder hesaplama tasarrufu (%)", color=INK2, fontsize=9.5)
ax.set_xlim(-0.01, 0.42); ax.set_ylim(0, 46)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_title("Frontier — her QP icin", color=INK, fontsize=11.5, loc="left", pad=8)

best = []
for qp in qps:
    ok = [r for r in rows if r["qp"] == qp and r["db_vs_uf"] <= 0.1]
    best.append(max(ok, key=lambda r: r["saving_pct"])["saving_pct"] if ok else 0.0)
bars = bx.bar(range(len(qps)), best, 0.55, color=S[0], zorder=3)
for i, v in enumerate(best):
    bx.text(i, v + 0.7, f"{v:.1f}%", ha="center", color=INK2, fontsize=9)
bx.axhspan(30, 40, color=S[3], alpha=0.10, zorder=1)
bx.text(len(qps) - 0.5, 35, "hedef", color=INK3, fontsize=8.5, va="center", ha="right")
bx.set_xticks(range(len(qps))); bx.set_xticklabels([f"qp {q}" for q in qps])
bx.set_ylabel("tasarruf (%)", color=INK2, fontsize=9.5); bx.set_ylim(0, 46)
bx.set_title("0.1 dB butcesinde", color=INK, fontsize=11.5, loc="left", pad=8)
bx.text(0, -0.16, f"kaynak: {Path(d['ckpt']).parent.name} epoch 0 — TEK epoch, 128px tile",
        transform=bx.transAxes, color=INK3, fontsize=8)

fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig("results/paper_curve.png", dpi=135, facecolor=SURFACE)
print("wrote results/paper_curve.png")
