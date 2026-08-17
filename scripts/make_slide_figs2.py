"""One figure per slide. Each is a measurement, not an illustration."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TUM, OR, GR, YE, GY = "#0065BD", "#E37222", "#A2AD00", "#FFDC00", "#999999"
SUR, INK, INK2, INK3, GC = "#ffffff", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e2"
def sty(a, t):
    a.set_facecolor(SUR); a.set_title(t, color=INK, fontsize=11, loc="left", pad=8)
    a.grid(True, color=GC, lw=0.7, zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=9, length=0)
    for k, sp in a.spines.items(): sp.set_visible(k == "bottom"); sp.set_color(GC)

# --- 2. where the decoder's compute goes --------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=SUR)
lab = ["upsample\n8.2%", "12 trunk blocks\n89.4%", "head\n2.4%"]
a.barh([0], [8.16], color=TUM); a.barh([0], [89.44], left=8.16, color=OR)
a.barh([0], [2.40], left=97.60, color=GR)
for x, w, l in zip([0, 8.16, 97.6], [8.16, 89.44, 2.40], lab):
    if w > 5: a.text(x + w/2, 0, l, ha="center", va="center", fontsize=10, color="white")
a.set_xlim(0, 100); a.set_yticks([]); a.set_xlabel("decode'un yuzdesi", color=INK2, fontsize=9)
sty(a, "453.5 GMAC / 1080p kare"); a.set_title("453.5 GMAC / 1080p kare", loc="left", fontsize=11)
b.barh([1, 0], [99.66, 0.34], color=[TUM, OR])
b.set_yticks([0, 1]); b.set_yticklabels(["3x3 depthwise\n0.34%", "1x1 pointwise\n99.66%"], fontsize=9)
b.set_xlim(0, 100); b.set_xlabel("blok icindeki pay", color=INK2, fontsize=9)
sty(b, "Bir DepthConvBlock icinde")
b.text(50, 0, "dikisin TEK sebebi", ha="center", va="center", fontsize=9, color="white")
fig.tight_layout(); fig.savefig("results/fig_macs.png", dpi=140, facecolor=SUR)

# --- 5. Microsoft's schedule, and where each run reads it ---------------
sys.path.insert(0, ".")
from train_flexuf_image import get_training_strategy
st = get_training_strategy()
lr = [e[1] for e in st]; pw = [e[2] for e in st]
fig, a = plt.subplots(figsize=(11, 3.0), facecolor=SUR)
a.step(range(len(lr)), lr, where="post", color=TUM, lw=2.2)
a.set_yscale("log"); a.set_ylabel("lr", color=INK2, fontsize=9)
a.set_xlabel("Microsoft takviminde epoch", color=INK2, fontsize=9)
a.axvspan(90, 105, color=OR, alpha=.14)
a.text(97, 1.3e-4, "512x512 fazi", ha="center", fontsize=9, color=OR)
for x, nm, c in ((0, "e1-e4 (BURADAN OKUYORDU\nkayma 0.20 dB)", "#c0392b"),
                 (75, "BEST / CONTROL\noffset 75", TUM),
                 (90, "VERBATIM\noffset 90", GR),
                 (99, "RECIPE512\noffset 99", OR)):
    a.axvline(x, color=c, ls="--", lw=1.5)
    a.annotate(nm, (x, 2.4e-4), fontsize=8, color=c, ha="left" if x < 80 else "right",
               xytext=(4 if x < 80 else -4, 0), textcoords="offset points")
a.set_ylim(5e-7, 6e-4)
sty(a, "Ayni takvim, dort farkli okuma noktasi")
fig.tight_layout(); fig.savefig("results/fig_schedule.png", dpi=140, facecolor=SUR)

# --- 6. anchor drift -----------------------------------------------------
fig, a = plt.subplots(figsize=(9.5, 3.1), facecolor=SUR)
q = ["qp 0", "qp 32", "qp 63"]; xs = np.arange(3); w = .26
for off, v, c, l in ((-w-.01, [-.153, -.201, -.263], "#c0392b", "recipe epoch 0'dan (HATA)"),
                     (0, [-.025, -.051, -.074], OR, "offset 75 + anchor"),
                     (w+.01, [0, 0, 0], GR, "govde donuk (tavan cokuyor)")):
    a.bar(xs + off, v, w, color=c, label=l, zorder=3)
    for x, y in zip(xs + off, v):
        if y: a.text(x, y - .008, f"{y:+.3f}", ha="center", va="top", fontsize=8, color=INK2)
a.set_xticks(xs); a.set_xticklabels(q); a.set_ylim(-.30, .05)
a.set_ylabel("gercek DCVC-UF'e gore, dB", color=INK2, fontsize=9)
a.legend(fontsize=8.5, frameon=False, loc="lower left")
sty(a, "Anchor kaymasi: hata, duzeltmesi, ve duzeltmenin bedeli")
fig.tight_layout(); fig.savefig("results/fig_anchor.png", dpi=140, facecolor=SUR)

# --- 8. seam ------------------------------------------------------------
fig, a = plt.subplots(figsize=(9.5, 3.1), facecolor=SUR)
m = ["zeros", "replicate", "linear", "arls", "canvas\ncoupling"]
p128 = [1.1670, .2125, .5021, .1785, 0.0]; p256 = [.5477, .1070, .2638, .0879, 0.0]
xs = np.arange(5); w = .38
a.bar(xs - w/2 - .01, p128, w, color=TUM, label="128px tile", zorder=3)
a.bar(xs + w/2 + .01, p256, w, color=GR, label="256px tile", zorder=3)
for x, v in zip(xs - w/2 - .01, p128): a.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
for x, v in zip(xs + w/2 + .01, p256): a.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
a.annotate("dikis YOK\ntam kare ile birebir", xy=(4, .04), xytext=(3.0, .62), fontsize=9,
           color=GR, ha="center", arrowprops=dict(arrowstyle="->", color=GR, lw=1.5))
a.set_xticks(xs); a.set_xticklabels(m, fontsize=8.5); a.set_ylim(0, 1.35)
a.set_ylabel("saf dikis cezasi qp63, dB", color=INK2, fontsize=9)
a.legend(fontsize=8.5, frameon=False)
sty(a, "Dikis: dolgu semasi x tile boyutu, CTC")
fig.tight_layout(); fig.savefig("results/fig_seam.png", dpi=140, facecolor=SUR)

# --- 9. MAC vs wall clock -----------------------------------------------
fig, a = plt.subplots(figsize=(9.5, 3.1), facecolor=SUR)
ex = ["cikis 2", "cikis 3", "cikis 4", "cikis 5"]
mac = [43.4, 28.6, 13.8, 0.0]; wall = [43.6, 28.8, 14.9, 0.5]
xs = np.arange(4); w = .38
a.bar(xs - w/2 - .01, mac, w, color=TUM, label="MAC modeli", zorder=3)
a.bar(xs + w/2 + .01, wall, w, color=YE, label="duvar saati (1080p)", zorder=3)
for x, (u, v) in enumerate(zip(mac, wall)):
    a.text(x, max(u, v) + 1.2, f"{v-u:+.1f} puan", ha="center", fontsize=8.5, color=INK3)
a.set_xticks(xs); a.set_xticklabels(ex); a.set_ylim(0, 50)
a.set_ylabel("tasarruf, %", color=INK2, fontsize=9); a.legend(fontsize=8.5, frameon=False)
sty(a, "Manset yuzde saatte de gercek mi — evet, +-1 puan")
fig.tight_layout(); fig.savefig("results/fig_cost.png", dpi=140, facecolor=SUR)

# --- 12. theory: adaptivity gain vs rate --------------------------------
th = json.loads(Path("results/theory_check.json").read_text())
qps = [0, 16, 32, 48, 63]
fig, a = plt.subplots(figsize=(9.5, 3.1), facecolor=SUR)
for lam, c, mk in (("1e-05", GY, "s"), ("3e-05", TUM, "o"), ("1e-04", OR, "^")):
    y = [100 * th[str(q)]["delta"][lam]["delta"] / th[str(q)]["delta"][lam]["J_oracle"] for q in qps]
    a.plot(qps, y, color=c, lw=2.2, marker=mk, ms=7, label=f"$\\lambda$ = {lam}",
           markeredgecolor=SUR, markeredgewidth=1.4, zorder=3)
a.set_xticks(qps); a.set_xlabel("QP", color=INK2, fontsize=9)
a.set_ylabel(r"$\Delta / J$  (%)", color=INK2, fontsize=9)
a.legend(fontsize=8.5, frameon=False)
a.annotate("yonlendirmenin degeri\nbit hizi ile buyuyor", (48, 3.52), fontsize=9, color=TUM,
           xytext=(-90, 14), textcoords="offset points",
           arrowprops=dict(arrowstyle="->", color=TUM, lw=1.4))
sty(a, "Uyarlanma kazanci $\\Delta(\\lambda)$ — teoremin olculen buyuklugu")
fig.tight_layout(); fig.savefig("results/fig_theory.png", dpi=140, facecolor=SUR)

# --- 4. warm start: bit-exact --------------------------------------------
fig, a = plt.subplots(figsize=(9.5, 2.9), facecolor=SUR)
labels = ["en derin cikis\n(adim 0)", "cikis 4", "cikis 3", "cikis 2"]
vals = [0.0, -0.0634, -0.1516, -0.3447]
cols = [GR, GY, GY, GY]
a.barh(range(4), vals, .55, color=cols, zorder=3)
for i, v in enumerate(vals):
    a.text(v - .008 if v else .004, i, "max|diff| = 0.0" if v == 0 else f"{v:.4f} dB",
           va="center", ha="right" if v else "left", fontsize=9,
           color=GR if v == 0 else INK2, fontweight="bold" if v == 0 else "normal")
a.set_yticks(range(4)); a.set_yticklabels(labels, fontsize=9); a.invert_yaxis()
a.set_xlim(-.42, .12); a.set_xlabel("yayinlanmis DCVC-UF'e gore, dB (qp0, egitim YOK)",
                                    color=INK2, fontsize=9)
sty(a, "Adim 0: en derin cikis yayinlanmis modelin KENDISI")
fig.tight_layout(); fig.savefig("results/fig_warmstart.png", dpi=140, facecolor=SUR)
print("6 figur yazildi")
