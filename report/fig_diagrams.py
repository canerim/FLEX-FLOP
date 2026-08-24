"""The two diagrams: what the main experiment trains, and what the decoder runs.

Both are drawn rather than plotted, because what matters is the structure --
which exits exist, which the deployed forward can select, and where the
gradient reaches -- and none of that is a quantity on an axis.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
OUT = Path(__file__).resolve().parent / "fig"


def block(ax, x, y, w, h, fc, ec, label=None, fs=6.2, tc=S.INK, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc)


def fig1_training():
    """The ladder the main experiment trains, and the clamp it deploys."""
    fig, ax = plt.subplots(figsize=(6.6, 2.95))
    ax.set_xlim(-0.05, 12.9); ax.set_ylim(-1.30, 3.35); ax.axis("off")

    x0, bw, gap, gsp = 1.55, 0.70, 0.09, 0.30
    ytr = 2.00
    block(ax, 0.10, ytr - 0.28, 0.86, 0.56, "#f2f2f2", S.MUTED, "upsample", 5.8)
    ax.text(0.53, ytr + 0.46, r"latent $\hat{y}$", ha="center", fontsize=6.6,
            color=S.INK2)
    prev_r = 0.96
    centres = []
    for g in range(6):
        gx = x0 + g * (2 * bw + gap + gsp)
        ax.annotate("", xy=(gx, ytr), xytext=(prev_r, ytr),
                    arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=0.7,
                                    mutation_scale=6))
        for b in range(2):
            block(ax, gx + b * (bw + gap), ytr - 0.28, bw, 0.56,
                  "#eef4fa" if g >= 2 else "#fdf1e7",
                  S.BLUE if g >= 2 else S.ORANGE, f"blk{2*g+b+1}", 5.2)
        prev_r = gx + 2 * bw + gap
        cx = gx + bw + gap / 2
        centres.append(cx)
        usable = g >= 2
        col = S.BLUE if usable else S.ORANGE
        ax.add_line(plt.Line2D([cx, cx], [ytr - 0.28, ytr - 0.66], color=col, lw=0.9))
        if g < 5:
            block(ax, cx - 0.38, ytr - 1.10, 0.76, 0.44,
                  "#eef4fa" if usable else "#fdf1e7", col, f"adapter {g}", 5.1)
            ax.add_line(plt.Line2D([cx, cx], [ytr - 1.10, ytr - 1.44],
                                   color=col, lw=0.9))
        else:
            ax.text(cx, ytr - 0.88, "no adapter\n(raw feature)", ha="center",
                    va="center", fontsize=5.0, color=S.INK2, style="italic")
            ax.add_line(plt.Line2D([cx, cx], [ytr - 1.06, ytr - 1.44],
                                   color=col, lw=0.9, linestyle=(0, (2, 1.5))))
        ax.text(cx, ytr - 1.62, f"$e_{g}$", ha="center", fontsize=7.2, color=col,
                fontweight="bold")
        ax.text(cx, ytr - 1.92, f"{2 * (g + 1)} blocks", ha="center", fontsize=5.4,
                color=S.INK2)

    # the shared head, fed by every exit
    hy = ytr - 2.34
    block(ax, centres[0] - 0.38, hy - 0.24, centres[-1] - centres[0] + 0.76, 0.44,
          "#f2f2f2", S.MUTED, "shared head  ->  RGB", 6.0)
    for cx in centres:
        ax.add_line(plt.Line2D([cx, cx], [ytr - 2.04, hy + 0.20],
                               color=S.MUTED, lw=0.6))

    # regimes
    ax.annotate("", xy=(x0 + 2 * bw + gap + gsp - 0.10, ytr + 0.72),
                xytext=(x0 - 0.10, ytr + 0.72),
                arrowprops=dict(arrowstyle="-", color=S.ORANGE, lw=2.6))
    ax.text(x0 + (2 * bw + gap + gsp) / 2 - 0.10, ytr + 0.92,
            "trained, not selectable", ha="center", fontsize=6.3,
            color=S.VERM, fontweight="bold")
    ax.annotate("", xy=(prev_r + 0.02, ytr + 0.72),
                xytext=(x0 + 2 * (2 * bw + gap + gsp) - 0.10, ytr + 0.72),
                arrowprops=dict(arrowstyle="-", color=S.BLUE, lw=2.6))
    ax.text((prev_r + x0 + 2 * (2 * bw + gap + gsp)) / 2, ytr + 0.92,
            "selectable  --  4 exits", ha="center", fontsize=6.3,
            color=S.BLUE, fontweight="bold")

    ax.text(x0 + (2 * bw + gap + gsp) / 2 - 0.10, ytr - 2.72,
            "no RD gradient\nfeature distillation only", ha="center",
            va="top", fontsize=5.7, color=S.VERM)
    ax.text((prev_r + x0 + 2 * (2 * bw + gap + gsp)) / 2, ytr - 2.72,
            "RD loss + distillation      forward(): exit_map.clamp(min=$j$=2)",
            ha="center", va="top", fontsize=5.7, color=S.INK2)
    fig.savefig(OUT / "fig1_training.pdf")
    fig.savefig(OUT / "fig1_training.png")
    plt.close(fig)


def fig2_mechanism():
    """Tiled decoding, per-position depth, and the band the second one pays.

    The band is computed, not drawn by hand: the same four-by-four depth map
    is expanded to a sub-grid, dilated by D(x) = max_y (d(y) - dist(x, y)) in
    blocks, and every sub-position that ends up running deeper than its own
    allocation is what panel (c) shades. Sketching it would have made the band
    look like a fixed-width halo, which is exactly what it is not.
    """
    import numpy as np
    depth = np.array([[2, 2, 4, 5], [2, 3, 4, 5], [2, 2, 3, 4], [2, 2, 2, 3]])
    n, sub, bpe = 4, 6, 2
    d_fine = np.repeat(np.repeat(depth, sub, 0), sub, 1)
    b = (d_fine + 1.0) * bpe
    D = b.copy()
    for _ in range(int(b.max()) + 1):
        pd = np.pad(D, 1, mode="edge")
        m = np.maximum.reduce([pd[a:a + D.shape[0], c:c + D.shape[1]]
                               for a in range(3) for c in range(3)]) - 1.0
        nxt = np.maximum(D, m)
        if np.array_equal(nxt, D):
            break
        D = nxt
    extra = D > b + 1e-6                       # positions the band pays for
    band_pct = 100.0 * (D.sum() - b.sum()) / b.sum()

    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.30))

    ax = axes[0]
    ax.set_title("tiled decoding", fontsize=7.0, pad=5, loc="left")
    S.panel(ax, "a", dx=-0.02, dy=1.16)
    for r in range(n):
        for c in range(n):
            block(ax, c, n - 1 - r, 0.94, 0.94,
                  cmap(0.16 + 0.13 * depth[r, c]), "#ffffff",
                  f"{depth[r, c]}", 6.6, lw=1.8)
    for k in range(1, n):
        ax.axvline(k - 0.03, color=S.VERM, lw=1.6, ymin=0.14, ymax=0.99)
        ax.axhline(k - 0.03, color=S.VERM, lw=1.6, xmin=0.01, xmax=0.985)
    ax.text(n / 2 - 0.5, -0.62, "every cut edge convolves\nagainst an invented value",
            ha="center", va="top", fontsize=6.0, color=S.VERM)

    ax = axes[1]
    ax.set_title("per-position depth", fontsize=7.0, pad=5, loc="left")
    S.panel(ax, "b", dx=-0.02, dy=1.16)
    for r in range(n):
        for c in range(n):
            block(ax, c, n - 1 - r, 1.0, 1.0,
                  cmap(0.16 + 0.13 * depth[r, c]), "none", f"{depth[r, c]}", 6.6,
                  lw=0.0)
    ax.text(n / 2 - 0.5, -0.62,
            "no cut; every position is\ndecoded to its own depth",
            ha="center", va="top", fontsize=6.0, color=S.BLUE)

    ax = axes[2]
    ax.set_title("what it pays: the band", fontsize=7.0, pad=5, loc="left")
    S.panel(ax, "c", dx=-0.02, dy=1.16)
    for r in range(n):
        for c in range(n):
            block(ax, c, n - 1 - r, 1.0, 1.0, "#f5f5f5", "none", "", 6, lw=0.0)
    ax.imshow(np.where(extra, 1.0, np.nan), extent=[0, n, 0, n],
              cmap=mpl_green(), vmin=0, vmax=1, interpolation="nearest",
              zorder=3, alpha=0.95)
    ax.text(n / 2 - 0.5, -0.62,
            f"positions still computing for a\ndeeper neighbour: +{band_pct:.0f}% trunk",
            ha="center", va="top", fontsize=6.0, color=S.GREEN)

    for ax in axes:
        ax.set_xlim(-0.06, n + 0.02); ax.set_ylim(-1.35, n + 0.06)
        ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(OUT / "fig2_mechanism.pdf")
    fig.savefig(OUT / "fig2_mechanism.png")
    plt.close(fig)


def mpl_green():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("g", [S.GREEN, S.GREEN])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1_training(); fig2_mechanism()
    print("  fig1, fig2 yazildi")
