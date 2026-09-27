"""Intro figures: compute-quality trade-off (data) and the paradigm comparison (schematic).
Data: docs/figures/eval2/uf_analysis.json (DCVC-UF, CTC 53 frames x 5 qp, real tiled decodes)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.expanduser("~/FLEX-PLUS"); OUT = f"{ROOT}/cvpr2027/figs"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.labelpad": 2})
COL = {"oracle": "#2a78d6", "router": "#eb6834", "dither": "#1baf7a", "uniform": "#eda100", "release": "#1a1a19"}
LAB = {"oracle": "FLEX, encoder-side map", "router": "FLEX, decoder-side router", "dither": "content-blind dither", "uniform": "one exit per frame"}
MK = dict(markeredgecolor="white", markeredgewidth=0.5)

def tradeoff():
    A = json.load(open(f"{ROOT}/docs/figures/eval2/uf_analysis.json"))
    bud = ["0.05", "0.1", "0.15", "0.2", "0.3"]
    fig, ax = plt.subplots(figsize=(3.3, 2.25))
    ax.plot([100], [0], "D", color=COL["release"], ms=5, **MK, zorder=5)
    ax.annotate("released DCVC-UF", (100, 0), (5, 5), textcoords="offset points", ha="left", fontsize=6)
    for r in ("uniform", "dither", "router", "oracle"):
        xs = [100 - A["means"][b][r]["saving"] for b in bud]; ys = [A["means"][b][r]["db_611"] for b in bud]
        ax.plot(xs, ys, "-o", color=COL[r], ms=3.6, lw=1.1, label=LAB[r], **MK, zorder=4 if r == "oracle" else 3)
    xo = 100 - A["means"]["0.1"]["oracle"]["saving"]; yo = A["means"]["0.1"]["oracle"]["db_611"]
    ax.annotate(f"−{A['means']['0.1']['oracle']['saving']:.1f}% MACs\nat {yo:.3f} dB", (xo, yo), (6, -34),
                textcoords="offset points", fontsize=6, color=COL["oracle"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=COL["oracle"]))
    ax.axvline(60.9, color="#b9b8b2", lw=0.6, ls=(0, (2, 2)))
    ax.text(60.6, 0.19, "shallowest exit", fontsize=5.5, color="#6f6f6a", rotation=90, va="top")
    ax.set_xlim(102, 58); ax.set_ylim(-0.008, 0.195)
    ax.set_xlabel("Decoder compute (% of released MACs)"); ax.set_ylabel("PSNR loss vs. released (dB)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    ax.spines["left"].set_position(("outward", 2)); ax.spines["bottom"].set_position(("outward", 2))
    fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/intro_tradeoff.pdf"); fig.savefig(f"{OUT}/intro_tradeoff.png", dpi=300)

# ---------------------------------------------------------------- schematic
DEP = ["#dbe8f7", "#9fc3ec", "#4f8fdc", "#1c4f92"]          # shallow -> deep
INK = "#2b2b2a"; GREY = "#8c8b86"

def tiles(ax, x0, y0, grid, w=0.34, h=0.22):
    nr, nc = len(grid), len(grid[0])
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x0 + j * w, y0 + (nr - 1 - i) * h), w, h, fc=DEP[grid[i][j]], ec="white", lw=0.8))
    ax.add_patch(Rectangle((x0, y0), nc * w, nr * h, fc="none", ec=INK, lw=0.6))
    return x0 + nc * w, y0 + nr * h / 2

def chain(ax, x0, y, n, colors=None, w=0.26, gap=0.07, h=0.36, width_scale=1.0):
    xs = []
    for k in range(n):
        c = colors[k] if colors else "#e9e8e3"
        ax.add_patch(FancyBboxPatch((x0 + k * (w + gap), y - h * width_scale / 2), w, h * width_scale,
                                    boxstyle="round,pad=0,rounding_size=0.04", fc=c, ec=INK, lw=0.5))
        xs.append(x0 + k * (w + gap) + w / 2)
        if k: ax.add_patch(FancyArrowPatch((x0 + k * (w + gap) - gap, y), (x0 + k * (w + gap), y), arrowstyle="-|>", mutation_scale=4, lw=0.5, color=INK))
    return xs, x0 + n * (w + gap) - gap

def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, lw=kw.pop("lw", 0.6), color=kw.pop("color", INK), **kw))

def paradigms():
    fig, axs = plt.subplots(2, 2, figsize=(3.35, 2.55))
    titles = ["(a) Static decoder", "(b) One exit per image", "(c) Separate networks per patch", "(d) FLEX: one ladder, exit per tile"]
    for ax, t in zip(axs.flat, titles):
        ax.set_xlim(0, 4.2); ax.set_ylim(0, 1.55); ax.axis("off"); ax.set_title(t, fontsize=6.5, loc="left", pad=1.5)
    ax = axs[0, 0]
    xr, ym = tiles(ax, 0.05, 0.35, [[3] * 4] * 3)
    xs, xe = chain(ax, 1.75, ym, 6, [DEP[3]] * 6); arrow(ax, (xr + 0.05, ym), (1.72, ym))
    ax.text(2.7, 0.18, "every tile, every block", ha="center", fontsize=5.5, color=GREY)
    ax = axs[0, 1]
    xr, ym = tiles(ax, 0.05, 0.35, [[1] * 4] * 3)
    xs, xe = chain(ax, 1.75, ym, 6, [DEP[1]] * 3 + ["#f4f3ef"] * 3); arrow(ax, (xr + 0.05, ym), (1.72, ym))
    arrow(ax, (xs[2], ym - 0.2), (xs[2], ym - 0.45), color=DEP[2]); ax.text(xs[2], 0.18, "exit", ha="center", fontsize=5.5, color=GREY)
    ax.text(3.4, 0.18, "whole image", ha="center", fontsize=5.5, color=GREY)
    ax = axs[1, 0]
    g = [[0, 1, 3, 1], [1, 3, 2, 0], [0, 0, 1, 2]]
    xr, ym = tiles(ax, 0.05, 0.35, g)
    for k, (yy, n, c) in enumerate([(ym + 0.42, 2, DEP[0]), (ym, 4, DEP[1]), (ym - 0.42, 6, DEP[3])]):
        chain(ax, 1.75, yy, n, [c] * n, h=0.22)
        arrow(ax, (xr + 0.05, ym), (1.72, yy), lw=0.5, connectionstyle="arc3,rad=0")
    ax = axs[1, 1]
    xr, ym = tiles(ax, 0.05, 0.35, g)
    xs, xe = chain(ax, 1.75, ym + 0.12, 6, ["#c9c8c2"] * 2 + ["#e9e8e3"] * 4)
    arrow(ax, (xr + 0.05, ym + 0.12), (1.72, ym + 0.12))
    for e, k in enumerate([2, 3, 4, 5]):
        arrow(ax, (xs[k], ym + 0.12 - 0.19), (xs[k], ym - 0.28), color=DEP[e], lw=0.9)
    ax.add_patch(FancyBboxPatch((1.75, ym - 0.52), xe - 1.75, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc="#e3f1ea", ec=INK, lw=0.5))
    ax.text((1.75 + xe) / 2, ym - 0.42, "seam repair + head", ha="center", va="center", fontsize=5.3)
    ax.text(0.05, 0.18, "map in bitstream", fontsize=5.3, color=GREY)
    ax.text((xs[0] + xs[1]) / 2, ym + 0.36, "full-frame stem", ha="center", fontsize=5.0, color=GREY)
    hs = [Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in DEP]
    fig.legend(hs, ["shallow", "", "", "deep"], loc="lower center", ncol=4, frameon=False, fontsize=5.5, handlelength=1.2,
               columnspacing=0.3, handletextpad=0.3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.05, hspace=0.25)
    fig.savefig(f"{OUT}/paradigms.pdf"); fig.savefig(f"{OUT}/paradigms.png", dpi=300)

if __name__ == "__main__":
    tradeoff(); paradigms()
