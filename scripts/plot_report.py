"""Report figure for the main experiment: what BEST carries and why each piece is there.

Every panel is a measurement that decided something. Nothing here is a
projection: where a number is an oracle it says ORACLE, where it is one epoch it
says one epoch.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1,S2,S3,S4,S5 = "#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"
SUR,INK,INK2,INK3,GC = "#fcfcfb","#0b0b0b","#52514e","#8a8985","#e6e5e2"
def sty(a,t,sub=None):
    a.set_facecolor(SUR); a.set_title(t,color=INK,fontsize=11,loc="left",pad=17 if sub else 7)
    if sub: a.text(0,1.012,sub,transform=a.transAxes,color=INK3,fontsize=7.8,va="bottom")
    a.grid(True,color=GC,lw=0.7,zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2,labelsize=8,length=0)
    for k,sp in a.spines.items(): sp.set_visible(k=="bottom"); sp.set_color(GC)

fig = plt.figure(figsize=(16.5,10.4),facecolor=SUR)
gs = fig.add_gridspec(3,3,hspace=0.52,wspace=0.27,left=0.055,right=0.985,top=0.9,bottom=0.06)
fig.suptitle("FLEX-UF — ana deney (BEST) ve dayandigi olcumler",fontsize=16,color=INK,x=0.055,ha="left",y=0.968)
fig.text(0.055,0.932,"Icerige uyarlanan erken cikis, DCVC-UF intra decoder. "
         "Her panel bir karari veren olcum; hicbiri projeksiyon degil.",color=INK2,fontsize=9.5,ha="left")

# 1 tile size
a=fig.add_subplot(gs[0,0])
m=["zeros","replicate","arls","coupling"]; p128=[1.1670,0.2125,0.1785,0.0]; p256=[0.5477,0.1070,0.0879,0.0]
xs=range(len(m)); w=0.38
a.bar([x-w/2-.012 for x in xs],p128,w,color=S1,label="128px",zorder=3)
a.bar([x+w/2+.012 for x in xs],p256,w,color=S3,label="256px  <- BEST",zorder=3)
for x,v in zip(xs,p128): a.text(x-w/2-.012,v+.03,f"{v:.2f}",ha="center",color=INK2,fontsize=7)
for x,v in zip(xs,p256): a.text(x+w/2+.012,v+.03,f"{v:.2f}",ha="center",color=INK2,fontsize=7)
a.set_xticks(list(xs)); a.set_xticklabels(m,fontsize=7.5); a.set_ylim(0,1.32)
a.set_ylabel("saf dikis cezasi qp63 (dB)",color=INK2,fontsize=8.5); a.legend(fontsize=8,frameon=False)
sty(a,"1 · Tile boyutu — neden 256px","256px dikisi yariya indirir, tavan ayni kalir.")

# 2 exit costs
a=fig.add_subplot(gs[0,1])
ex=["cikis 2","cikis 3","cikis 4","cikis 5"]; mac=[43.4,28.6,13.8,0.0]; wall=[43.6,28.8,14.9,0.5]
xs=range(4)
a.bar([x-w/2-.012 for x in xs],mac,w,color=S1,label="MAC modeli",zorder=3)
a.bar([x+w/2+.012 for x in xs],wall,w,color=S4,label="duvar saati",zorder=3)
for x,(u,v) in enumerate(zip(mac,wall)): a.text(x,max(u,v)+1.1,f"{v-u:+.1f}",ha="center",color=INK3,fontsize=7)
a.set_xticks(list(xs)); a.set_xticklabels(ex,fontsize=8); a.set_ylim(0,50)
a.set_ylabel("tasarruf (%)",color=INK2,fontsize=8.5); a.legend(fontsize=8,frameon=False)
sty(a,"2 · Mansetteki yuzde saatte gercek","Her cikista +-1 puan icinde. Manset hissedilir.")

# 3 anchor
a=fig.add_subplot(gs[0,2])
q=["qp 0","qp 32","qp 63"]; bef=[-.153,-.201,-.263]; aft=[-.025,-.051,-.074]; frz=[0,0,0]
xs=range(3); w3=0.26
a.bar([x-w3-.01 for x in xs],bef,w3,color=S2,label="recipe epoch 0'dan",zorder=3)
a.bar(list(xs),aft,w3,color=S4,label="offset 75 + anchor  <- BEST",zorder=3)
a.bar([x+w3+.01 for x in xs],frz,w3,color=S3,label="govde donuk (tavan cokuyor)",zorder=3)
for x,v in zip(xs,bef): a.text(x-w3-.01,v-.012,f"{v:+.3f}",ha="center",va="top",color=INK2,fontsize=6.8)
for x,v in zip(xs,aft): a.text(x+0,v-.012,f"{v:+.3f}",ha="center",va="top",color=INK2,fontsize=6.8)
a.set_xticks(list(xs)); a.set_xticklabels(q); a.set_ylim(-.30,.045)
a.set_ylabel("gercek DCVC-UF'e gore (dB)",color=INK2,fontsize=8.5); a.legend(fontsize=7.2,frameon=False,loc="lower left")
sty(a,"3 · Anchor kaymasi ve bedeli","3.5x azaldi. Dondurmak sifirlar ama tavani yikar.")

# 4 paper curve
a=fig.add_subplot(gs[1,:2])
d=json.load(open("results/paper_curve_grid128.json"))["rows"]
for c,qp in zip([S1,S2,S3,S4,S5],sorted({r["qp"] for r in d})):
    pts=sorted([(r["db_vs_uf"],r["saving_pct"]) for r in d if r["qp"]==qp])
    pts=[p for p in pts if -.02<=p[0]<=.42]
    a.plot([p[0] for p in pts],[p[1] for p in pts],color=c,lw=2,marker="o",ms=3,
           label=f"qp {qp}",zorder=3,markeredgecolor=SUR,markeredgewidth=.8)
a.axvline(.1,color=INK3,lw=1.4,ls=(0,(4,3)),zorder=2)
a.text(.104,1,"0.1 dB",color=INK3,fontsize=8,rotation=90,va="bottom")
a.axhspan(30,40,color=S4,alpha=.10,zorder=1); a.text(.33,35,"hedef %30-40",color=INK3,fontsize=8,va="center")
a.set_xlim(-.01,.42); a.set_ylim(0,46)
a.set_xlabel("yayinlanmis DCVC-UF'in ALTINDA, dB",color=INK2,fontsize=9)
a.set_ylabel("tasarruf (%)",color=INK2,fontsize=9); a.legend(fontsize=8,frameon=False,loc="lower right")
sty(a,"4 · ORACLE frontier — gercek DCVC-UF referansli",
    "Kusursuz router varsayimi. TEK epoch, 128px. Ust sinir, basari degil.")

# 5 oracle vs real router
a=fig.add_subplot(gs[1,2])
lbl=["oracle\n@%95","ROUTER\nlam 1e-4"]
sv=[21.1,24.8]; db=[0.174,0.367]
a.bar([0,1],sv,0.5,color=[S3,S2],zorder=3)
for i,(s_,d_) in enumerate(zip(sv,db)):
    a.text(i,s_+0.7,f"{s_:.1f}%",ha="center",color=INK2,fontsize=9)
    a.text(i,s_/2,f"{d_:.3f} dB",ha="center",color="white",fontsize=9,fontweight="bold")
a.set_xticks([0,1]); a.set_xticklabels(lbl,fontsize=8); a.set_ylim(0,32)
a.set_ylabel("tasarruf (%), qp32",color=INK2,fontsize=8.5)
sty(a,"5 · Acik soru: oracle vs GERCEK router",
    "Router daha cok tasarruf edip 2x dB odiyor — frontier'in DISINDA.")

# 6 live
a=fig.add_subplot(gs[2,:])


for t,c in (("BEST",S2),("CONTROL",S1)):
    f=Path("runs")/t/"train_log.jsonl"
    if not f.exists(): continue
    r=[json.loads(l) for l in f.open() if l.strip()]
    if len(r)<2: continue
    xs=[x["step"] for x in r]; ys=[x["spread_dB"] for x in r]
    k=max(1,len(ys)//25)
    ys=[sum(ys[max(0,i-k):i+1])/len(ys[max(0,i-k):i+1]) for i in range(len(ys))]
    a.plot(xs,ys,color=c,lw=2,label=t,zorder=3)
    a.annotate(t,(xs[-1],ys[-1]),textcoords="offset points",xytext=(6,0),color=c,fontsize=9,va="center")
lo,hi=a.get_xlim(); a.set_xlim(0,hi+0.12*(hi-lo))
a.set_xlabel("adim",color=INK2,fontsize=9); a.set_ylabel("ladder spread (dB)",color=INK2,fontsize=9)
a.legend(fontsize=8.5,frameon=False,loc="upper right")
sty(a,"6 · Canli: BEST vs CONTROL — tek degiskenli",
    "BEST tum kazanan parcalari tasiyor, CONTROL hicbirini. Spread tek basina yanlis tani; "
    "anchor ile okunmali (panel 3): ikisinin de en derin cikisi gercek DCVC-UF'e bagli.")

fig.savefig("results/report_best.png",dpi=130,facecolor=SUR)
print("wrote results/report_best.png")
