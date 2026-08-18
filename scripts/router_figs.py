"""Figures for the router explainer. Minimal text inside the axes.

Every label that is an explanation rather than a name belongs in the caption, not
in the picture. Titles here are at most a few words; units and quantities are on
the axes; nothing is annotated that the caption can say better.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply()
OUT = R / "docs" / "figures"


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)


def box(ax, x, y, w, h, t1, t2=None, fc="white", ec=ns.INK2, fs=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=0.012",
                 facecolor=fc, edgecolor=ec, linewidth=0.7))
    ax.text(x + w / 2, y + h / 2 + (0.018 if t2 else 0), t1, ha="center",
            va="center", fontsize=fs)
    if t2:
        ax.text(x + w / 2, y + h / 2 - 0.022, t2, ha="center", va="center",
                fontsize=fs - 1.2, color=ns.INK2)


def arrow(ax, x0, y0, x1, y1, c=ns.INK2, lw=0.8, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=7, linewidth=lw, color=c, linestyle=ls,
                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> docs/figures/{name}")


# ============================================================ 1. what each sees
def what_each_sees():
    fig, ax = plt.subplots(figsize=(ns.W2, 2.5)); blank(ax)

    ax.add_patch(Rectangle((0.0, 0.52), 1.0, 0.46, fc="#fdf4f9",
                           ec=ns.PURPLE, lw=0.8))
    ax.text(0.015, 0.935, "A  ·  encoder", fontsize=7.5, weight="bold",
            color=ns.PURPLE)
    box(ax, 0.10, 0.70, 0.13, 0.14, "source", "x", fc="#f6e6ef")
    box(ax, 0.28, 0.70, 0.13, 0.14, "encoder", fc="#e8e8e8")
    box(ax, 0.46, 0.70, 0.13, 0.14, "decoder", "×K", fc="#dceaf7")
    box(ax, 0.64, 0.70, 0.15, 0.14, "D(t,k)", fc="#f6e6ef", ec=ns.PURPLE)
    box(ax, 0.84, 0.70, 0.13, 0.14, "argmin", fc="#f6e6ef", ec=ns.PURPLE)
    for x in (0.23, 0.41, 0.59, 0.79):
        arrow(ax, x, 0.77, x + 0.05, 0.77, c=ns.PURPLE)
    arrow(ax, 0.165, 0.70, 0.165, 0.60, c=ns.PURPLE)
    arrow(ax, 0.165, 0.60, 0.70, 0.60, c=ns.PURPLE)
    arrow(ax, 0.70, 0.60, 0.70, 0.70, c=ns.PURPLE)

    ax.add_patch(Rectangle((0.0, 0.02), 1.0, 0.46, fc="#eef6fb",
                           ec=ns.BLUE, lw=0.8))
    ax.text(0.015, 0.435, "B  ·  decoder", fontsize=7.5, weight="bold",
            color=ns.BLUE)
    box(ax, 0.10, 0.20, 0.13, 0.14, "source", "x", fc="#f0f0f0", ec="#cccccc")
    ax.plot([0.105, 0.225], [0.205, 0.335], color=ns.VERM, lw=1.4)
    ax.plot([0.105, 0.225], [0.335, 0.205], color=ns.VERM, lw=1.4)
    box(ax, 0.28, 0.20, 0.13, 0.14, "bitstream", fc="#e8e8e8")
    box(ax, 0.46, 0.20, 0.13, 0.14, "stem, ŷ,\nscales, q", fc="#dceaf7", fs=5.4)
    box(ax, 0.64, 0.20, 0.15, 0.14, "z(t)", fc="#e3f0f9", ec=ns.BLUE)
    box(ax, 0.84, 0.20, 0.13, 0.14, "argmax", fc="#e3f0f9", ec=ns.BLUE)
    for x in (0.41, 0.59, 0.79):
        arrow(ax, x, 0.27, x + 0.05, 0.27, c=ns.BLUE)
    save(fig, "router_sees.png")


# ============================================================ 2. the A search
def a_search():
    d = json.load(open(R / "results/tile_table.json"))
    D = np.array(d["D"])[:, d["j"]:]          # reachable exits only
    cost = np.array(d["cost"])[d["j"]:]
    ks = list(range(d["j"], d["K"]))
    order = np.argsort(D[:, 0])               # hardest tile first, for legibility
    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.3))

    im = ax[0].imshow(D[order] * 1e4, aspect="auto", cmap="magma_r")
    ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels(ks)
    ax[0].set_xlabel("exit"); ax[0].set_ylabel("tile")
    ax[0].set_yticks([]); ax[0].grid(False)
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("MSE ×10⁻⁴", fontsize=5.5)
    ns.panel(ax[0], "a")

    ax[1].step(range(len(ks)), cost, where="mid", color=ns.INK, lw=1.3)
    ax[1].plot(range(len(ks)), cost, "o", color=ns.ORANGE, ms=5)
    ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks)
    ax[1].set_xlabel("exit"); ax[1].set_ylabel("cost $c_k$")
    ax[1].set_ylim(0.5, 1.08)
    ns.panel(ax[1], "b", dx=-0.26)

    # Every allocation in the sweep at once: rows are tiles, columns are
    # lambda, colour is the chosen exit. A line plot per lambda encodes the same
    # thing far worse -- tiles have no natural order to connect along.
    sw = d["sweep"]
    Mmap = np.array([np.array(r_["map"])[order] for r_ in sw]).T
    cmap = matplotlib.colors.ListedColormap(
        [ns.VERM, ns.ORANGE, ns.SKY, ns.BLUE])
    im2 = ax[2].imshow(Mmap, aspect="auto", cmap=cmap, vmin=d["j"] - 0.5,
                       vmax=d["K"] - 0.5, origin="upper",
                       extent=[np.log10(sw[0]["lam"]), np.log10(sw[-1]["lam"]),
                               len(order), 0])
    ax[2].set_xlabel("log₁₀ λ"); ax[2].set_ylabel("tile")
    ax[2].set_yticks([]); ax[2].grid(False)
    cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03,
                       ticks=ks)
    cb2.ax.tick_params(labelsize=5)
    cb2.set_label("exit", fontsize=5.5)
    ns.panel(ax[2], "c", dx=-0.26)
    fig.tight_layout()
    save(fig, "router_a_search.png")


def a_frontier():
    """The bisection target: what the lambda sweep traces out."""
    d = json.load(open(R / "results/tile_table.json"))
    sw = d["sweep"]
    db = np.array([x["db"] for x in sw]); sv = np.array([x["saving"] for x in sw])
    fig, ax = plt.subplots(1, 2, figsize=(ns.W15, 2.2))
    ax[0].plot([x["lam"] for x in sw], db, "-o", ms=2.6, color=ns.BLUE)
    ax[0].set_xscale("log"); ax[0].set_xlabel("λ"); ax[0].set_ylabel("dB")
    ax[0].axhline(0.1, color=ns.VERM, lw=0.9, ls=(0, (4, 2)))
    ns.panel(ax[0], "a")
    ax[1].plot(db, sv, "-o", ms=2.6, color=ns.INK)
    ax[1].axvline(0.1, color=ns.VERM, lw=0.9, ls=(0, (4, 2)))
    i = int(np.argmin(np.abs(db - 0.1)))
    ax[1].plot([db[i]], [sv[i]], "o", ms=6, color=ns.VERM)
    ax[1].set_xlabel("dB"); ax[1].set_ylabel("MACs saved (%)")
    ns.panel(ax[1], "b", dx=-0.2)
    fig.tight_layout()
    save(fig, "router_frontier.png")


# ============================================================ 3. the B head
def b_head():
    fig, ax = plt.subplots(figsize=(ns.W15, 2.3)); blank(ax)
    box(ax, 0.02, 0.72, 0.20, 0.16, "stem", "384 ch", fc="#dceaf7")
    box(ax, 0.02, 0.50, 0.20, 0.16, "ŷ ‖ scales", "512 ch", fc="#dceaf7")
    box(ax, 0.02, 0.28, 0.20, 0.16, "q", fc="#dceaf7")
    box(ax, 0.30, 0.72, 0.16, 0.16, "1×1", "→ 48", fc="#e3f0f9", ec=ns.BLUE)
    box(ax, 0.30, 0.50, 0.16, 0.16, "1×1", "→ 32", fc="#e3f0f9", ec=ns.BLUE)
    arrow(ax, 0.22, 0.80, 0.30, 0.80); arrow(ax, 0.22, 0.58, 0.30, 0.58)
    box(ax, 0.52, 0.50, 0.14, 0.38, "pool\nper tile", fc="#e3f0f9", ec=ns.BLUE,
        fs=5.6)
    arrow(ax, 0.46, 0.80, 0.52, 0.75); arrow(ax, 0.46, 0.58, 0.52, 0.62)
    arrow(ax, 0.22, 0.36, 0.59, 0.36); arrow(ax, 0.59, 0.36, 0.59, 0.50)
    box(ax, 0.72, 0.50, 0.11, 0.38, "MLP", "161→256→256→6", fc="#e3f0f9",
        ec=ns.BLUE, fs=5.6)
    arrow(ax, 0.66, 0.69, 0.72, 0.69)
    box(ax, 0.89, 0.60, 0.09, 0.18, "z(t)", fc="#e3f0f9", ec=ns.BLUE)
    arrow(ax, 0.83, 0.69, 0.89, 0.69)
    ax.text(0.5, 0.13, "144,024 parameters   ·   0.163% of one decode",
            fontsize=6, ha="center", color=ns.INK)
    save(fig, "router_b_head.png")


if __name__ == "__main__":
    what_each_sees(); a_search(); a_frontier(); b_head()
