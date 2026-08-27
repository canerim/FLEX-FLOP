"""The visual language of the paper's Figure 1, as code.

The teaser was drawn by hand in diagrams.net, so a second diagram beside it
either matches it or looks like it came from a different paper. These are the
pieces it is made of: rounded boxes with pastel fills and a thin darker edge,
dashed rounded containers that group them with the label underneath, tensor
shapes in square brackets, small grids standing for feature maps, and an exit
ramp that runs light pink to near-black magenta.

Two arrow kinds, and they mean different things there. Solid black is the
tensor moving. Dashed magenta is a decision reaching the place it applies.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

INK = "#1a1a1a"
GREY = "#5b5b5b"

SALMON = ("#fbdedb", "#d98b83")
PURPLE = ("#e0d9ee", "#9186c2")
BLUEG  = ("#e2ebf7", "#6f9fd4")
PINKG  = ("#fbe3ee", "#d municipal")  # placeholder, replaced below
PINKG  = ("#fbe3ee", "#c76ba0")
BLUE   = ("#6f9fd4", "#3f6f9f")
GREEN  = ("#cfe6cf", "#7aa77a")
ORANGE = ("#f2913f", "#c96c1c")
MAGENTA= ("#e0397a", "#a01a52")
GREY_B = ("#ededed", "#a8a8a8")
WHITE  = ("#ffffff", "#8a8a8a")

# exit ramp, light to dark
EXITS = ["#fbdbe9", "#f3a7cb", "#e8579f", "#a3175c", "#6d0f3e", "#3f0a25"]
ACCENT = "#d81b74"


def setup():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "font.family": "DejaVu Sans", "text.color": INK,
        "figure.facecolor": "white",
    })


def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, fill, fs=7.2, tc=None, lw=1.0, r=0.10, z=3,
        weight="normal"):
    fc, ec = fill
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=z))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc or INK, zorder=z + 2, linespacing=1.35,
                fontweight=weight)


def group(ax, x, y, w, h, label, fill, fs=7.4, z=1, dash=(0, (4.0, 3.0)),
          label_dy=-0.30):
    fc, ec = fill
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.16",
                                facecolor=fc, edgecolor=ec, linewidth=1.0,
                                linestyle=dash, zorder=z))
    if label:
        ax.text(x + w / 2, y + label_dy, label, ha="center", va="top",
                fontsize=fs, color=INK, zorder=z + 1, linespacing=1.3)


def arrow(ax, p, q, color=INK, lw=1.15, ls="-", z=4, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z,
                                 joinstyle="round"))


def elbow(ax, pts, color=INK, lw=1.15, ls="-", z=4, ms=9):
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                zorder=z, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color, lw=lw, ls=ls, z=z, ms=ms)


def tensor(ax, x, y, text, fs=6.3, ha="center"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=fs, color=GREY)


def gridglyph(ax, x, y, w, h, nx, ny, colors=None, ec="#9a9a9a", lw=0.5, z=3):
    """A little grid standing for a feature map or an exit map."""
    cw, ch = w / nx, h / ny
    for i in range(nx):
        for j in range(ny):
            c = "#ffffff" if colors is None else colors[(j * nx + i) % len(colors)]
            ax.add_patch(Rectangle((x + i * cw, y + j * ch), cw, ch,
                                   facecolor=c, edgecolor=ec, linewidth=lw,
                                   zorder=z))


def stack(ax, x, y, w, h, n=3, fill=BLUEG, dx=0.06, dy=0.06, z=3):
    """A shallow stack of planes, for a multi-channel tensor."""
    fc, ec = fill
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, facecolor=fc,
                               edgecolor=ec, linewidth=0.8, zorder=z + (n - i)))
