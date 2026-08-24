"""One place for the look, so nine figures cannot drift from each other.

Okabe-Ito is the categorical palette: it was designed for colour-vision
deficiency rather than checked for it afterwards, which is why it is the
default in scientific publishing. Hues are assigned in a FIXED order and
never cycled -- a figure that needs a ninth series gets small multiples
instead.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe & Ito (2008), in the order used throughout.
BLUE   = "#0072B2"
VERM   = "#D55E00"
GREEN  = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY    = "#56B4E9"
YELLOW = "#F0E442"
CAT = [BLUE, VERM, GREEN, PURPLE, ORANGE, SKY]

INK   = "#1a1a1a"
INK2  = "#555555"
MUTED = "#8a8a8a"
GRID  = "#e6e6e6"
SURF  = "#ffffff"


def setup():
    """Nature-ish: no box, no grid unless a reading needs one, generous white.

    The rules that matter here are the ones a copy editor would enforce --
    axis lines only where they are read against, ticks pointing out, units in
    the label, one weight of type, and direct labels in place of a legend box
    wherever four or fewer series make that legible.
    """
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        "font.family": "DejaVu Sans", "font.size": 6.8,
        "axes.titlesize": 7.0, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
        "legend.fontsize": 6.4, "legend.frameon": False,
        "legend.handlelength": 1.1, "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.32, "legend.borderpad": 0.0,
        "axes.edgecolor": INK, "axes.linewidth": 0.55,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.labelpad": 2.6, "axes.titlepad": 3.0,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.axisbelow": True, "lines.linewidth": 1.25,
        "lines.markersize": 3.0, "lines.markeredgewidth": 0.0,
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.spines.top": False, "axes.spines.right": False,
        "patch.linewidth": 0.6,
    })


def panel(ax, letter, dx=-0.155, dy=1.045):
    """The bold panel letter outside the axes, not inside the title."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.0,
            fontweight="bold", va="top", ha="left", color=INK)


def ygrid(ax, alpha=1.0):
    """A horizontal rule set, for panels read as magnitudes."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4, alpha=alpha)
    ax.xaxis.grid(False)


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def label_end(ax, x, y, text, color, dx=0.12, dy=0.0, **kw):
    """A direct label at the end of a series, so the legend can stay small."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=6.6, va="center", ha="left",
                fontweight="bold", **kw)
