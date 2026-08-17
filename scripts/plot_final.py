"""The figure the result actually is: three ways to choose the exit, one axis.

Left  — the frontier, all three on the same axes so the comparison needs no
        caption to be fair.
Right — the operating point that matters: the most saving still inside 0.1 dB
        of released DCVC-UF.

ORACLE is an upper bound and is drawn dashed so it can never be mistaken for a
result. ROUTER is the deployable predictor and sits OUTSIDE the frontier, which
is the honest shape of that failure. SIGNALLED is the working system, and its
rate cost is written on the figure rather than left to the text.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1,S2,S3,S4 = "#2a78d6","#eb6834","#1baf7a","#eda100"
SUR,INK,INK2,INK3,GC = "#fcfcfb","#0b0b0b","#52514e","#8a8985","#e6e5e2"
def sty(a,t,sub=None):
    a.set_facecolor(SUR); a.set_title(t,color=INK,fontsize=11.5,loc="left",pad=17 if sub else 8)
    if sub: a.text(0,1.012,sub,transform=a.transAxes,color=INK3,fontsize=8.2,va="bottom")
    a.grid(True,color=GC,lw=0.7,zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2,labelsize=8.5,length=0)
    for k,sp in a.spines.items(): sp.set_visible(k=="bottom"); sp.set_color(GC)

orc = json.load(open("results/paper_curve_grid128.json"))["rows"]
sig = {r["qp"]: r for r in json.load(open("results/signalled_grid128.json"))["rows"]}
qps = sorted(sig)

fig,(ax,bx) = plt.subplots(1,2,figsize=(14,5.9),facecolor=SUR,
                           gridspec_kw={"width_ratios":[1.35,1]})
fig.suptitle("FLEX-UF — cikisi nasil sececegiz: uc yaklasim",fontsize=15.5,
             color=INK,x=0.045,ha="left",y=0.97)
fig.text(0.045,0.902,
  "Yayinlanmis DCVC-UF referansli, ayni bitstream, ayni latent (encoder donuk, max|diff| = 0). "
  "Sinyalli sistemin harita maliyeti bpp'ye DAHIL.",color=INK2,fontsize=9,ha="left")

for c,qp in zip([S1,S2,S3,S4,"#e87ba4"],qps):
    pts=sorted([(r["db_vs_uf"],r["saving_pct"]) for r in orc if r["qp"]==qp])
    pts=[p for p in pts if -.02<=p[0]<=.30]
    ax.plot([p[0] for p in pts],[p[1] for p in pts],color=c,lw=1.6,ls=(0,(5,2)),
            alpha=.75,zorder=2,label=f"qp {qp}" if qp==qps[0] else None)
    ax.plot(sig[qp]["db_vs_uf"],sig[qp]["saving_pct"],marker="o",ms=9,color=c,
            markeredgecolor=SUR,markeredgewidth=1.6,zorder=4)
    ax.annotate(f"qp{qp}",(sig[qp]["db_vs_uf"],sig[qp]["saving_pct"]),
                textcoords="offset points",xytext=(9,-3),color=c,fontsize=8.5)
# the predicting router, qp32, measured
ax.plot(0.367,24.8,marker="X",ms=13,color="#c0392b",markeredgecolor=SUR,
        markeredgewidth=1.6,zorder=5)
ax.annotate("tahmin eden router (qp32)\n%24.8 icin 0.367 dB — frontier DISI",
            (0.367,24.8),textcoords="offset points",xytext=(-14,16),
            color="#c0392b",fontsize=8.5,ha="right")
ax.axvline(.1,color=INK3,lw=1.4,ls=(0,(4,3)),zorder=1)
ax.text(.104,1,"0.1 dB butce",color=INK3,fontsize=8.5,rotation=90,va="bottom")
ax.set_xlim(-.01,.42); ax.set_ylim(0,34)
ax.set_xlabel("yayinlanmis DCVC-UF'in ALTINDA, dB",color=INK2,fontsize=9.5)
ax.set_ylabel("decoder hesaplama tasarrufu (%)",color=INK2,fontsize=9.5)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([],[],color=INK3,ls=(0,(5,2)),lw=1.6,label="ORACLE (ulasilamaz ust sinir)"),
    Line2D([],[],color=INK3,marker="o",ls="",ms=8,label="SINYALLI sistem (calisan)"),
    Line2D([],[],color="#c0392b",marker="X",ls="",ms=9,label="tahmin eden router"),
],fontsize=8.5,frameon=False,loc="upper left")
sty(ax,"Frontier","Kesikli = ust sinir, dolu daire = gercek sistem. Ayni eksende, ayni referans.")

xs=range(len(qps))
o=[max([r for r in orc if r["qp"]==q and r["db_vs_uf"]<=.1],
       key=lambda r:r["saving_pct"])["saving_pct"] for q in qps]
s=[sig[q]["saving_pct"] for q in qps]
w=.38
bx.bar([x-w/2-.012 for x in xs],o,w,color="#c8c8c4",label="ORACLE (ust sinir)",zorder=3)
bx.bar([x+w/2+.012 for x in xs],s,w,color=S3,label="SINYALLI (calisan)",zorder=3)
for x,v in zip(xs,s): bx.text(x+w/2+.012,v+.5,f"{v:.1f}%",ha="center",color=INK2,fontsize=8.5)
for x,v in zip(xs,o): bx.text(x-w/2-.012,v+.5,f"{v:.1f}",ha="center",color=INK3,fontsize=7.5)
bx.set_xticks(list(xs)); bx.set_xticklabels([f"qp {q}" for q in qps])
bx.set_ylim(0,32); bx.set_ylabel("tasarruf (%)",color=INK2,fontsize=9.5)
bx.legend(fontsize=8.5,frameon=False)
mb=sum(r["bpp_added"] for r in sig.values())/len(sig)
bx.text(0,-0.17,f"harita maliyeti bpp'ye dahil: ortalama {mb:.6f} bpp "
        f"(~200 bit/kare, bit hizinin ~onbinde 1.3'u)",
        transform=bx.transAxes,color=INK3,fontsize=8)
sty(bx,"0.1 dB butcesinde","Calisan sistem, ulasilamaz sinirin 1-2 puan icinde.")

fig.tight_layout(rect=[0,0.03,1,0.875])
fig.savefig("results/final_curve.png",dpi=135,facecolor=SUR)
print("wrote results/final_curve.png")
