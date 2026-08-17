"""Compact per-slide figures, sized for the right half of a 10x7.5in slide."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUM, OR, GR, GY, RD = "#0065BD", "#E37222", "#A2AD00", "#999999", "#C4071B"
SUR, INK, INK2, GC = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e2"
R = Path("results")

def base(w=5.0, h=3.4):
    f, a = plt.subplots(figsize=(w, h), facecolor=SUR)
    a.set_facecolor(SUR); a.grid(True, color=GC, lw=.7, zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=8.5, length=0)
    for k, s in a.spines.items(): s.set_visible(k == "bottom"); s.set_color(GC)
    return f, a

def save(f, n):
    f.tight_layout(); f.savefig(R / n, dpi=150, facecolor=SUR); plt.close(f)
    print("wrote", n)

# 2 · where the compute is
f, a = base()
lab = ["upsample\n8.2%", "12-block trunk\n89.4%", "head\n2.4%"]
v = [8.16, 89.44, 2.40]
a.barh([0], [v[0]], color=TUM, zorder=3)
a.barh([0], [v[1]], left=v[0], color=OR, zorder=3)
a.barh([0], [v[2]], left=v[0]+v[1], color=GY, zorder=3)
a.text(v[0]/2, 0, "upsample\n8.2%", ha="center", va="center", fontsize=8.5, color="white")
a.text(v[0]+v[1]/2, 0, "12-block trunk — 89.4%\nthe part an exit can skip",
       ha="center", va="center", fontsize=10, color="white", fontweight="bold")
a.text(v[0]+v[1]+v[2]/2, 0, "head", ha="center", va="center", fontsize=7, color="white")
a.set_yticks([]); a.set_xlim(0, 100); a.set_xlabel("decode MAC payi (%)", color=INK2, fontsize=9)
a.set_title("453.5 GMAC / 1080p kare — %99.7'si pointwise 1x1", color=INK, fontsize=10.5, loc="left")
save(f, "s_mac.png")

# 4 · bit-exact warm start
f, a = base(5.0, 3.2)
names = ["deepest exit\n== stock UF", "all exits ==\ntruncated UF", "j=K ==\nfull decode",
         "key remap\nbijection", "seam repair\nidentity"]
a.bar(range(5), [1]*5, color=GR, zorder=3)
for i in range(5):
    a.text(i, .5, "max|diff|\n= 0.0", ha="center", va="center", fontsize=9,
           color="white", fontweight="bold")
a.set_xticks(range(5)); a.set_xticklabels(names, fontsize=7.5)
a.set_yticks([]); a.set_ylim(0, 1.25)
a.set_title("Sifir toleransli kontroller — her commit'te", color=INK, fontsize=10.5, loc="left")
save(f, "s_exact.png")

# 5 · recipe compliance
f, a = base(5.0, 3.4)
items = ["takvim satirlari", "106 giris", "AdamW 1e-4", "clip 0.1 + skip",
         "ImageFolder + lambdas", "64 QP", "encoder 74 tensor", "255 paylasilan tensor"]
a.barh(range(len(items)), [1]*len(items), color=TUM, zorder=3)
for i in range(len(items)):
    a.text(.02, i, "  ok", va="center", fontsize=9, color="white", fontweight="bold")
a.set_yticks(range(len(items))); a.set_yticklabels(items, fontsize=8.5)
a.set_xticks([]); a.set_xlim(0, 1); a.invert_yaxis()
a.set_title("verify_recipe.py — uyusmazlikta hata verir", color=INK, fontsize=10.5, loc="left")
save(f, "s_recipe.png")

# 6 · anchor drift
f, a = base(5.0, 3.3)
q = ["qp 0", "qp 32", "qp 63"]; b4 = [-.153, -.201, -.263]; af = [-.025, -.051, -.074]
x = np.arange(3); w = .34
a.bar(x - w/2, b4, w, color=RD, label="recipe epoch 0'dan", zorder=3)
a.bar(x + w/2, af, w, color=OR, label="duzeltilmis", zorder=3)
a.axhline(0, color=GR, lw=2.2, zorder=4)
for i, v in enumerate(b4): a.text(i - w/2, v - .008, f"{v:+.3f}", ha="center", va="top", fontsize=8)
for i, v in enumerate(af): a.text(i + w/2, v - .008, f"{v:+.3f}", ha="center", va="top", fontsize=8)
a.text(2.45, .006, "govde donuk: 0.000", color=GR, fontsize=8.5, ha="right")
a.set_xticks(x); a.set_xticklabels(q); a.set_ylim(-.30, .05)
a.set_ylabel("gercek DCVC-UF'e gore (dB)", color=INK2, fontsize=9)
a.legend(fontsize=8, frameon=False, loc="lower left")
a.set_title("Anchor kaymasi ve duzeltmesi", color=INK, fontsize=10.5, loc="left")
save(f, "s_anchor.png")

# 8 · seam
f, a = base(5.0, 3.3)
m = ["zeros", "replicate", "arls", "coupling"]
p128 = [1.167, .213, .179, 0.0]; p256 = [.548, .107, .088, 0.0]
x = np.arange(4); w = .36
a.bar(x - w/2, p128, w, color=TUM, label="128px tile", zorder=3)
a.bar(x + w/2, p256, w, color=GR, label="256px tile", zorder=3)
for i, v in enumerate(p128): a.text(i - w/2, v + .03, f"{v:.2f}", ha="center", fontsize=7.5)
for i, v in enumerate(p256): a.text(i + w/2, v + .03, f"{v:.2f}", ha="center", fontsize=7.5)
a.annotate("dikis yok\n(tam kare ile birebir)", xy=(3, .04), xytext=(2.1, .62),
           color=GR, fontsize=8.5, ha="center",
           arrowprops=dict(arrowstyle="->", color=GR, lw=1.4))
a.set_xticks(x); a.set_xticklabels(m, fontsize=9); a.set_ylim(0, 1.32)
a.set_ylabel("saf dikis cezasi qp63 (dB)", color=INK2, fontsize=9)
a.legend(fontsize=8, frameon=False)
a.set_title("Dikis: dolgu semasi x tile boyutu", color=INK, fontsize=10.5, loc="left")
save(f, "s_seam.png")

# 11 · router vs signalling
f, a = base(5.0, 3.3)
lab = ["tahmin eden\nrouter", "SINYALLI\nsistem", "oracle\n(ust sinir)"]
sv = [24.8, 15.7, 16.5]; db = [0.367, 0.088, 0.078]
c = [RD, GR, GY]
a.bar(range(3), sv, .5, color=c, zorder=3)
for i, (s_, d_) in enumerate(zip(sv, db)):
    a.text(i, s_ + .6, f"{s_:.1f}%", ha="center", fontsize=10)
    a.text(i, s_/2, f"{d_:.3f} dB", ha="center", fontsize=10, color="white", fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(lab, fontsize=8.5)
a.set_ylim(0, 30); a.set_ylabel("tasarruf (%), qp32", color=INK2, fontsize=9)
a.set_title("Router 4 kat dB odiyor; sinyal butcenin icinde", color=INK, fontsize=10.5, loc="left")
save(f, "s_router.png")

# 12 · adaptivity gain
f, a = base(5.0, 3.3)
t = json.load(open(R / "theory_check.json"))
qs = [0, 16, 32, 48, 63]
g = [100 * t[str(q)]["delta"]["3e-05"]["delta"] / t[str(q)]["delta"]["3e-05"]["J_oracle"] for q in qs]
a.plot(qs, g, color=OR, lw=2.4, marker="o", ms=8, zorder=3, markeredgecolor=SUR, markeredgewidth=1.5)
for q, v in zip(qs, g):
    a.annotate(f"{v:.2f}%", (q, v), textcoords="offset points", xytext=(0, 9),
               ha="center", fontsize=8.5, color=OR)
a.set_xticks(qs); a.set_xlabel("QP", color=INK2, fontsize=9)
a.set_ylabel(r"$\Delta/J$  (uyarlanmanin degeri)", color=INK2, fontsize=9)
a.set_ylim(0, 5.4)
a.set_title("Teoremin oncegordugu buyume, olculmus", color=INK, fontsize=10.5, loc="left")
save(f, "s_gain.png")
