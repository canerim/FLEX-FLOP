"""Nature's figure conventions, applied once so every figure inherits them.

From Nature's author guidelines (nature.com/nature/for-authors/final-submission
and the final-artwork guide): sans-serif throughout, Helvetica or Arial, the same
face in every figure; panel labels 8 pt bold lower-case a, b, c; all other text
at most 7 pt and never below 5 pt; line weight at least 0.25 pt; column widths
89 mm single, 183 mm double, 120 mm for the 1.5-column case; light background,
minimal decoration; colour-blind-safe palettes, and red/green pairs avoided.

The palette below is Okabe-Ito, which is designed to stay separable under
deuteranopia and protanopia -- the standard choice when a journal asks for
colour-blind-safe and does not supply one.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM      # Nature column widths, inches

# Okabe-Ito, colour-blind safe. Deliberately no red/green pairing.
BLUE, ORANGE, SKY, GREEN = "#0072B2", "#E69F00", "#56B4E9", "#009E73"
YELLOW, VERM, PURPLE, BLACK = "#F0E442", "#D55E00", "#CC79A7", "#000000"
SERIES = [BLUE, ORANGE, GREEN, VERM, PURPLE, SKY]
INK, INK2, GRID = "#000000", "#4d4d4d", "#d9d9d9"

def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "grid.linewidth": 0.4,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "lines.linewidth": 1.0, "lines.markersize": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
        "axes.edgecolor": INK2, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def panel(ax, letter, dx=-0.16, dy=1.06):
    """Nature panel label: 8 pt bold, lower case, upright."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", color=INK)
