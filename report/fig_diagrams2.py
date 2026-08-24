"""Two more diagrams: where the router sits, and where the compute goes.

The first answers a question the tables cannot: the router is on the DECODER
side and adds no bits, so its cost is a property of the decode rather than of
the bitstream. The second is a schematic and a measurement at once -- the
compute of one decode, decomposed, with what each of the three levers removes
marked on it.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"


def box(ax, x, y, w, h, label, fc, ec, fs=6.0, tc=None, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.035",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc or S.INK, zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=None, lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=6, color=color or S.MUTED,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 zorder=1))


def fig10_pipeline():
    """The decode, with the router path marked as decoder-side and bit-free."""
    fig, ax = plt.subplots(figsize=(6.5, 2.05))
    ax.set_xlim(0, 13.0); ax.set_ylim(-0.25, 3.05); ax.axis("off")
    y = 1.62; h = 0.56

    box(ax, 0.05, y, 1.05, h, "bitstream", "#f4f4f4", S.MUTED)
    arrow(ax, 1.10, y + h / 2, 1.44, y + h / 2)
    box(ax, 1.44, y, 1.30, h, "entropy\ndecode", "#f4f4f4", S.MUTED)
    arrow(ax, 2.74, y + h / 2, 3.08, y + h / 2)
    box(ax, 3.08, y, 1.00, h, r"$\hat{y}$", "#f4f4f4", S.MUTED, fs=7.2)
    arrow(ax, 4.08, y + h / 2, 4.42, y + h / 2)
    box(ax, 4.42, y, 1.20, h, "upsample", "#f4f4f4", S.MUTED)
    arrow(ax, 5.62, y + h / 2, 5.96, y + h / 2)
    box(ax, 5.96, y, 2.05, h, "trunk, run where\n$D(x)\\geq b$", "#e7f1f9",
        S.BLUE, fs=6.0)
    arrow(ax, 8.01, y + h / 2, 8.35, y + h / 2)
    box(ax, 8.35, y, 1.35, h, "stitch at\neach exit", "#e7f1f9", S.BLUE)
    arrow(ax, 9.70, y + h / 2, 10.04, y + h / 2)
    box(ax, 10.04, y, 1.05, h, "head", "#f4f4f4", S.MUTED)
    arrow(ax, 11.09, y + h / 2, 11.43, y + h / 2)
    box(ax, 11.43, y, 1.05, h, "image", "#f4f4f4", S.MUTED)

    # the router, below, fed from the decoder's own state
    yr = 0.34
    box(ax, 3.08, yr, 2.54, h, "cell features\n(bits, latent, stem)", "#eaf6f0",
        S.GREEN, fs=5.9)
    arrow(ax, 4.35, y, 4.35, yr + h, S.GREEN, ls=(0, (2.2, 1.6)))
    arrow(ax, 5.62, yr + h / 2, 5.96, yr + h / 2, S.GREEN)
    box(ax, 5.96, yr, 1.55, h, "router\n$\\hat{m}_k$", "#eaf6f0", S.GREEN)
    arrow(ax, 7.51, yr + h / 2, 7.85, yr + h / 2, S.GREEN)
    box(ax, 7.85, yr, 2.15, h, "argmin$_k$ $\\,m_k+\\lambda c_k$", "#eaf6f0",
        S.GREEN, fs=5.9)
    arrow(ax, 8.92, yr + h, 7.00, y, S.GREEN)
    ax.text(8.32, y - 0.40, "depth map $d(x)$", fontsize=5.8, color=S.GREEN,
            ha="left")

    ax.text(6.55, 2.72, "no bits are added: every router input is already in "
            "the decoder", fontsize=6.3, color=S.GREEN, ha="center",
            fontweight="bold")
    fig.savefig(OUT / "fig10_pipeline.pdf")
    fig.savefig(OUT / "fig10_pipeline.png")
    plt.close(fig)


def fig11_budget():
    """Where one decode's compute goes, and what each lever removes."""
    C = json.loads((RES / "cost_constants.json").read_text())
    pb, K, bpe = C["per_block"], C["K"], C["blocks_per_exit"]
    up, hd, trunk = C["SHARE_UPSAMPLE"], C["SHARE_HEAD"], C["SHARE_TRUNK"]
    ad = C["adapter"][2]
    seam = 1.0095 - (up + trunk + hd)

    fig, ax = plt.subplots(figsize=(6.5, 2.05))
    rows = [
        ("released decode", [("upsample", up, S.MUTED),
                             ("trunk, 12 blocks", trunk, S.BLUE),
                             ("head", hd, "#bdbdbd")], None),
        ("tiled floor, $j$=2\n(every tile deepest)",
         [("upsample", up, S.MUTED), ("shared 4 blk", 4 * pb, S.SKY),
          ("tiled 2 blk", 2 * pb, S.BLUE), ("adapter", ad, S.PURPLE),
          ("head", hd, "#bdbdbd"), ("seam repair", seam, S.VERM)], None),
        ("per-position floor, $j$=2",
         [("upsample", up, S.MUTED), ("6 blocks", 6 * pb, S.BLUE),
          ("adapter", ad, S.PURPLE), ("head", hd, "#bdbdbd")], "seam repair gone"),
        ("per-position floor, $j$=0",
         [("upsample", up, S.MUTED), ("2 blocks", 2 * pb, S.BLUE),
          ("adapter", ad, S.PURPLE), ("head", hd, "#bdbdbd")],
         "four more blocks freed"),
    ]
    ys = np.arange(len(rows))[::-1]
    for yy, (lab, segs, note) in zip(ys, rows):
        x = 0.0
        for i, (nm, v, col) in enumerate(segs):
            ax.barh(yy, v, 0.46, left=x, color=col, zorder=3,
                    edgecolor="white", linewidth=0.8)
            if v > 0.11:
                ax.text(x + v / 2, yy, nm, ha="center", va="center",
                        fontsize=5.6, color="white", zorder=4)
            x += v
        ax.text(x + 0.012, yy, f"{x:.3f}", va="center", fontsize=6.2,
                color=S.INK, fontweight="bold")
        if note:
            ax.text(x + 0.105, yy, note, va="center", fontsize=5.8,
                    color=S.GREEN)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(0, 1.32); ax.set_xlabel("cost, in units of one released decode")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=S.GRID, linewidth=0.4); ax.set_axisbelow(True)
    S.despine(ax)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("the floor is what caps the ceiling", fontsize=7.0, pad=4,
                 loc="left")
    # A key for the segments too narrow to carry a label inside them.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=S.MUTED, label="upsample"),
                       Patch(facecolor=S.PURPLE, label="adapter"),
                       Patch(facecolor="#bdbdbd", label="head"),
                       Patch(facecolor=S.VERM, label="seam repair")],
              loc="lower right", ncol=4, fontsize=5.9,
              bbox_to_anchor=(1.005, -0.44), columnspacing=1.1)
    fig.savefig(OUT / "fig11_budget.pdf")
    fig.savefig(OUT / "fig11_budget.png")
    plt.close(fig)


if __name__ == "__main__":
    fig10_pipeline(); fig11_budget(); print("  fig10, fig11 yazildi")
