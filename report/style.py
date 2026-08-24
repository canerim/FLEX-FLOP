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
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.titlesize": 7.8, "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8, "legend.frameon": False,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.6,
        "lines.markersize": 3.6, "figure.facecolor": SURF,
        "axes.facecolor": SURF, "legend.handlelength": 1.4,
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def label_end(ax, x, y, text, color, dx=0.12, dy=0.0, **kw):
    """A direct label at the end of a series, so the legend can stay small."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=6.6, va="center", ha="left",
                fontweight="bold", **kw)
