"""Render the RegLIC introductory figure (schematic + pinned measured values)."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

OUT = Path(__file__).parent
MACROS = (OUT.parents[1] / "data/refresh20260927/macros.tex").read_text()
def measured(name):
    match = re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9.]+)\}", MACROS)
    if not match:
        raise ValueError(f"Missing pinned paper macro: {name}")
    return float(match.group(1))

route, dither = measured("MainRouterSaving"), measured("MainDitherSaving")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
ink, muted, teal, coral, paper = "#18313d", "#59707a", "#008e83", "#c9544a", "#f6f8f7"
fig = plt.figure(figsize=(3.42, 2.75), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(.045, .945, "WHERE SHOULD THE DECODER WORK?", color=ink, fontsize=9,
        fontweight="bold", va="top")
ax.plot([.045, .955], [.895, .895], color="#cfdbdc", lw=.8)
ax.text(.045, .855, "illustrative spatial value of extra depth", color=muted,
        fontsize=7.1, va="top")

# A deliberately schematic value field; the printed caption states its status.
v = np.array([
    [0,0,0,1,1,1,2,3,2,1],
    [0,0,1,2,2,2,3,3,2,1],
    [0,1,2,3,3,2,1,2,2,0],
    [0,1,3,3,2,1,0,1,2,0],
    [0,0,2,3,2,0,0,1,1,0],
    [0,0,1,2,1,0,0,0,0,0],
])
pal = ["#e9f1ef", "#9fd1c7", "#51afa5", "#087f7b"]
for r in range(v.shape[0]):
    for c in range(v.shape[1]):
        ax.add_patch(Rectangle((.052 + c*.056, .464 + (5-r)*.044),
                               .052, .040, facecolor=pal[v[r,c]], edgecolor="white",
                               linewidth=.32))
ax.add_patch(Rectangle((.048, .458), .566, .276, fill=False, edgecolor="#d3ddde", lw=.7))
ax.text(.642, .705, "low gain", color=muted, fontsize=7.0, va="center")
ax.text(.642, .622, "medium", color=muted, fontsize=7.0, va="center")
ax.text(.642, .539, "high gain", color=muted, fontsize=7.0, va="center")
for y, col in [(.705,pal[0]),(.622,pal[1]),(.539,pal[3])]:
    ax.add_patch(Rectangle((.908,y-.018), .032,.032, color=col, ec="white", lw=.3))

ax.plot([.045,.955],[.413,.413],color="#cfdbdc",lw=.8)
ax.text(.045,.377,"MEASURED ALLOCATION AT 0.1 dB", color=ink, fontsize=8.1,
        fontweight="bold", va="top")
ax.text(.045,.327,"Modelled synthesis MAC saving",color=muted,fontsize=7.0,va="top")
ax.plot([.28,.88],[.147,.147],color="#9cadb2",lw=.65)
for x,t in zip([.28,.48,.68,.88],["0","10","20","30%"]):
    ax.plot([x,x],[.139,.155],color="#9cadb2",lw=.65)
    ax.text(x,.119,t,color=muted,fontsize=6.6,ha="center",va="top")
scale = lambda z: .28 + .60*z/30
for y,name,value,color in [(.245,"Dither",dither,"#9dafad"),
                           (.190,"Route",route,teal)]:
    ax.text(.052,y,name,color=ink,fontsize=7.5,va="center")
    ax.plot([scale(0),scale(value)],[y,y],color=color,lw=5.0,
            solid_capstyle="round",zorder=2)
    ax.scatter([scale(value)],[y],s=24,color=color,edgecolor="white",lw=.7,zorder=3)
    ax.text(.94,y,f"{value:.2f}%",ha="right",va="center",fontsize=7.2,
            color=color,fontweight="bold")
fig.savefig(OUT/"reglic_intro.pdf",bbox_inches="tight",pad_inches=.01)
fig.savefig(OUT/"reglic_intro.svg",bbox_inches="tight",pad_inches=.01)
fig.savefig(OUT/"reglic_intro.png",dpi=300,bbox_inches="tight",pad_inches=.01)
svg = OUT/"reglic_intro.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
