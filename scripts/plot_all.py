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
ORACLE = {  # achieved dB -> saving %, from scripts/oracle_diagnostic.py
    "qp 0":  ([-0.006, 0.013, 0.044, 0.061, 0.101, 0.244], [30.8, 36.0, 40.3, 41.5, 42.3, 42.6]),
    "qp 32": ([0.018, 0.047, 0.091, 0.118, 0.152, 0.231], [21.7, 30.3, 37.0, 39.5, 41.4, 42.3]),
    "qp 63": ([0.011, 0.044, 0.102, 0.158, 0.224, 0.308], [12.1, 20.7, 29.3, 34.2, 38.2, 41.3]),
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
style(a, "1 · Oracle tavani", "Kusursuz router ne verirdi. Ust sinir, basari degil.")

# ---- 2. seam penalty ------------------------------------------------------
a = fig.add_subplot(gs[0, 1])
MODES = ["zeros", "replicate", "linear", "arls"]
P128 = [1.1670, 0.2125, 0.5021, 0.1785]
P256 = [0.5477, 0.1070, 0.2638, 0.0879]
xs = range(len(MODES)); w = 0.38
a.bar([x - w/2 - 0.012 for x in xs], P128, w, color=S1, label="128px tile", zorder=3)
a.bar([x + w/2 + 0.012 for x in xs], P256, w, color=S3, label="256px tile", zorder=3)
for x, v in zip(xs, P128):
    a.text(x - w/2 - 0.012, v + 0.03, f"{v:.2f}", ha="center", color=INK2, fontsize=7.5)
for x, v in zip(xs, P256):
    a.text(x + w/2 + 0.012, v + 0.03, f"{v:.2f}", ha="center", color=INK2, fontsize=7.5)
a.set_xticks(list(xs)); a.set_xticklabels(MODES)
a.set_ylabel("saf dikis cezasi, qp63 (dB)", color=INK2, fontsize=9)
a.set_ylim(0, 1.32)
a.legend(fontsize=8.5, frameon=False)
style(a, "2 · Dikis: dolgu semasi x tile boyutu",
      "CTC native. arls en iyi ama +%10.7 decode suresi — reddedildi.")

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
DRIFT = [-0.153, -0.201, -0.263]
xs = range(len(QPS))
a.bar(list(xs), DRIFT, 0.5, color=S2, zorder=3)
for x, v in zip(xs, DRIFT):
    a.text(x, v - 0.014, f"{v:+.3f}", ha="center", va="top", color=INK2, fontsize=8.5)
a.axhline(0, color=S3, lw=2.2, zorder=4)
a.text(2.42, 0.006, "duzeltilmis hedef: 0", color=S3, fontsize=8.2, ha="right", va="bottom")
a.set_xticks(list(xs)); a.set_xticklabels(QPS)
a.set_ylabel("gercek DCVC-UF'e gore (dB)", color=INK2, fontsize=9)
a.set_ylim(-0.32, 0.045)
style(a, "4 · Anchor kaymasi — bulunan hata",
      "Recipe epoch 0'dan okunuyordu. Tek epochta bu kadar dustu.")

# ---- 5. live training -----------------------------------------------------
a = fig.add_subplot(gs[1, 1:])
alive = hb.live_runs()
COL = {"WD-j2/128": S1, "GRID-j2/128": S3, "GRID-j2/256": S4,
       "DISTILL-j2/256": S2, "WD-j4/256": S5}
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
style(a, "5 · Canli kosular — ladder ayrisiyor mu",
      "Bugun hepsi anchor duzeltmesiyle yeniden basladi, o yuzden adim ekseni. "
      "Spread tek basina yanlis tani: anchor ile birlikte okunmali (panel 4).")

fig.savefig("results/all_experiments.png", dpi=135, facecolor=SURFACE)
print("wrote results/all_experiments.png")
