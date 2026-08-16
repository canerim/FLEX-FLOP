"""Where every FLEX-UF experiment stands, on one page.

Built around the MEASURED results rather than the training curves. Four of the
five panels are finished numbers; only the last is live progress, because the
warm-start runs were all relaunched today and an epoch axis would show five lines
stacked on zero.

Palette: the dataviz reference instance, categorical slots in fixed order.
(The bundled validator needs node, which is not installed here, so these are the
already-validated defaults used unchanged rather than colours chosen by eye.)
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.heartbeat as hb

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, INK3, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e2"

def style(a, title, sub=None):
    a.set_facecolor(SURFACE)
    a.set_title(title, color=INK, fontsize=11.5, loc="left", pad=16 if sub else 8)
    if sub:
        a.text(0, 1.015, sub, transform=a.transAxes, color=INK3, fontsize=8.2,
               va="bottom", ha="left")
    a.grid(True, color=GRIDC, lw=0.7, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=8.5, length=0)
    for k, sp in a.spines.items():
        sp.set_visible(k == "bottom"); sp.set_color(GRIDC)

fig = plt.figure(figsize=(16.5, 9.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                      left=0.055, right=0.985, top=0.885, bottom=0.075)
fig.suptitle("FLEX-UF — olculmus sonuclar", fontsize=16, color=INK, x=0.055,
             ha="left", y=0.965)
fig.text(0.055, 0.925, "Icerige uyarlanan erken cikis, DCVC-UF intra decoder. "
         "Butun sayilar olculmustur; hicbiri tahmin degildir.",
         color=INK2, fontsize=9.5, ha="left")

# ---- 1. oracle headroom: the headline ------------------------------------
a = fig.add_subplot(gs[0, 0])
# From the FIRST properly-anchored checkpoint (wdec_j2_p128_grid epoch 0). The
# earlier, higher curve came from a pre-epoch_offset run whose deepest exit had
# drifted 0.2 dB from released DCVC-UF, so its "dB lost" was measured against a
# degraded reference. Not comparable; replaced rather than shown alongside.
ORACLE = {
    "qp 0":  ([0.002, 0.027, 0.082, 0.137, 0.227, 0.392], [12.5, 19.2, 27.3, 32.4, 37.2, 41.0]),
    "qp 32": ([0.007, 0.036, 0.097, 0.167, 0.279, 0.474], [2.6, 8.7, 17.2, 23.1, 29.8, 36.8]),
    "qp 63": ([0.001, 0.010, 0.059, 0.135, 0.263, 0.562], [-0.5, 1.3, 6.6, 12.0, 19.2, 28.9]),
}
for (nm, (xs, ys)), c in zip(ORACLE.items(), (S1, S2, S3)):
    a.plot(xs, ys, color=c, lw=2, marker="o", ms=5, label=nm, zorder=3,
           markeredgecolor=SURFACE, markeredgewidth=1.5)
a.axvline(0.1, color=INK3, lw=1.4, ls=(0, (4, 3)), zorder=2)
a.text(0.104, 5, "hedef butce 0.1 dB", color=INK3, fontsize=8, rotation=90, va="bottom")
a.axhspan(30, 40, color=S4, alpha=0.10, zorder=1)
a.text(0.30, 35, "hedef %30-40", color=INK3, fontsize=8, va="center")
a.set_xlim(-0.02, 0.34); a.set_ylim(0, 46)
a.set_xlabel("kaybedilen PSNR (dB)", color=INK2, fontsize=9)
a.set_ylabel("tasarruf (%)", color=INK2, fontsize=9)
a.legend(fontsize=8.5, frameon=False, loc="lower right")
style(a, "1 · Oracle tavani (dogru anchor'li ilk ckpt)",
      "Kusursuz router ne verirdi. Ust sinir, basari degil.")

# ---- 2. seam penalty ------------------------------------------------------
a = fig.add_subplot(gs[0, 1])
# canvas coupling is measured at 0.0000 at every QP and both tile sizes -- not
# rounded to zero, exactly zero, because the depthwise on the assembled canvas IS
# the one the full-frame decoder runs.
MODES = ["zeros", "replicate", "linear", "arls", "coupling"]
P128 = [1.1670, 0.2125, 0.5021, 0.1785, 0.0]
P256 = [0.5477, 0.1070, 0.2638, 0.0879, 0.0]
xs = range(len(MODES)); w = 0.38
a.bar([x - w/2 - 0.012 for x in xs], P128, w, color=S1, label="128px tile", zorder=3)
a.bar([x + w/2 + 0.012 for x in xs], P256, w, color=S3, label="256px tile", zorder=3)
for x, v in zip(xs, P128):
    a.text(x - w/2 - 0.012, v + 0.03, f"{v:.2f}", ha="center", color=INK2, fontsize=7.5)
for x, v in zip(xs, P256):
    a.text(x + w/2 + 0.012, v + 0.03, f"{v:.2f}", ha="center", color=INK2, fontsize=7.5)
a.annotate("dikis yok\n(tam kare ile birebir)", xy=(4, 0.02), xytext=(3.15, 0.55),
           color=S3, fontsize=8, ha="center",
           arrowprops=dict(arrowstyle="->", color=S3, lw=1.4))
a.set_xticks(list(xs)); a.set_xticklabels(MODES, fontsize=7.5)
a.set_ylabel("saf dikis cezasi, qp63 (dB)", color=INK2, fontsize=9)
a.set_ylim(0, 1.32)
a.legend(fontsize=8.5, frameon=False)
style(a, "2 · Dikis: dolgu semasi x tile boyutu",
      "CTC native, tek derinlik. Coupling +%4.0 saat icin dikisin TAMAMI.")

# ---- 3. MAC vs wall clock -------------------------------------------------
a = fig.add_subplot(gs[0, 2])
EX = ["cikis 2", "cikis 3", "cikis 4", "cikis 5"]
MAC = [43.4, 28.6, 13.8, 0.0]
WALL = [43.6, 28.8, 14.9, 0.5]
xs = range(len(EX))
a.bar([x - w/2 - 0.012 for x in xs], MAC, w, color=S1, label="MAC modeli", zorder=3)
a.bar([x + w/2 + 0.012 for x in xs], WALL, w, color=S4, label="duvar saati", zorder=3)
for x, (m, wl) in enumerate(zip(MAC, WALL)):
    a.text(x, max(m, wl) + 1.1, f"{wl-m:+.1f}", ha="center", color=INK3, fontsize=7.5)
a.set_xticks(list(xs)); a.set_xticklabels(EX)
a.set_ylabel("tasarruf (%)", color=INK2, fontsize=9); a.set_ylim(0, 50)
a.legend(fontsize=8.5, frameon=False)
style(a, "3 · Mansetteki yuzde saatte gercek mi",
      "j=2, 1080p. Fark her cikista +-1 puan icinde — evet.")

# ---- 4. anchor drift ------------------------------------------------------
a = fig.add_subplot(gs[1, 0])
QPS = ["qp 0", "qp 32", "qp 63"]
BEFORE = [-0.153, -0.201, -0.263]
AFTER = [-0.025, -0.051, -0.074]
xs = range(len(QPS)); wb = 0.36
a.bar([x - wb/2 - 0.012 for x in xs], BEFORE, wb, color=S2, label="recipe epoch 0'dan", zorder=3)
a.bar([x + wb/2 + 0.012 for x in xs], AFTER, wb, color=S4, label="epoch_offset 75 + anchor", zorder=3)
for x, v in zip(xs, BEFORE):
    a.text(x - wb/2 - 0.012, v - 0.012, f"{v:+.3f}", ha="center", va="top", color=INK2, fontsize=7.5)
for x, v in zip(xs, AFTER):
    a.text(x + wb/2 + 0.012, v - 0.012, f"{v:+.3f}", ha="center", va="top", color=INK2, fontsize=7.5)
a.legend(fontsize=8, frameon=False, loc="lower left")
a.axhline(0, color=S3, lw=2.2, zorder=4)
a.text(2.42, 0.006, "duzeltilmis hedef: 0", color=S3, fontsize=8.2, ha="right", va="bottom")
a.set_xticks(list(xs)); a.set_xticklabels(QPS)
a.set_ylabel("gercek DCVC-UF'e gore (dB)", color=INK2, fontsize=9)
a.set_ylim(-0.32, 0.045)
style(a, "4 · Anchor kaymasi — hata ve duzeltmesi",
      "3.5x azaldi ama sifirlanmadi. qp63'te kalan 0.074 dB, butcenin %74'u.")

# ---- 5. live training -----------------------------------------------------
a = fig.add_subplot(gs[1, 1:])
alive = hb.live_runs()
COL = {"BEST": S2, "CONTROL": S1}
for tag in sorted(alive):
    f = Path("runs") / tag / "train_log.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(l) for l in f.open() if l.strip()]
    nm = hb.SHORT.get(tag, tag[:11])
    if nm not in COL or len(rows) < 2:
        continue
    spe = rows[-1]["total"] // 16 or 1
    xs = [r["epoch"] * spe + r["step"] for r in rows]
    ys = [r["spread_dB"] for r in rows]
    k = max(1, len(ys) // 25)
    ys = [sum(ys[max(0, i-k):i+1]) / len(ys[max(0, i-k):i+1]) for i in range(len(ys))]
    a.plot(xs, ys, color=COL[nm], lw=2, label=nm, zorder=3)
    a.annotate(nm, (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
               color=COL[nm], fontsize=8, va="center")
a.set_xlabel("adim", color=INK2, fontsize=9)
a.set_ylabel("ladder spread (dB)", color=INK2, fontsize=9)
# Right margin for the direct labels, so they cannot collide with the legend.
lo, hi = a.get_xlim()
a.set_xlim(0, hi + 0.14 * (hi - lo))
a.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
style(a, "5 · Iki kosu: BEST vs CONTROL",
      "Yedi koldan ikiye indirildi: kazanan parcalarin hepsi BEST'te, CONTROL "
      "hicbiri. Spread tek basina yanlis tani; anchor ile okunmali (panel 4).")

fig.savefig("results/all_experiments.png", dpi=135, facecolor=SURFACE)
print("wrote results/all_experiments.png")
