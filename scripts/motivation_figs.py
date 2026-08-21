"""Four plots for the first pages: where the field sits, and why depth per tile.

Chosen from the shapes a Nature figure uses for this kind of argument, and kept
to the ones that carry an argument the text is already making rather than
decorating it:

  a  cost against rate saving, every published codec a point  (Section 2)
  b  what one frame's tiles actually cost to decode well      (Section 2)
  c  the exit ladder, modelled against hook-counted           (Section 3)
  d  where the Lagrangian oracle sends tiles, by rate         (Section 3)

Rejected, and why, so the choice is on the record: rate-distortion curves and
per-class heatmaps duplicate tables the paper already carries; the Lorenz curve
of per-tile regret and the agreement-against-saving scatter belong to Section 5
and would be read as results before the method is stated; the seam map, the
adapter bar chart and the three-unit energy comparison already exist as figures;
per-sequence beeswarms and Spearman panels are supplement material.

    python scripts/motivation_figs.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"


def _bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    ax.xaxis.label.set_size(5.8)
    ax.yaxis.label.set_size(5.8)
    return ax


# --------------------------------------------------------------- a. the field
def field():
    """Decoder cost alone, as a bar per method.

    The obvious plot here is cost against BD-Rate, which is what the codec
    literature draws. We cannot honestly draw it: our BD-Rate is against the
    released DCVC-UF decoder on CTC intra frames, and the image codecs quote
    theirs against VTM-22.0 on Kodak. Putting both on one vertical axis would
    invite exactly the cross-anchor comparison the tables are careful to
    forbid. What IS comparable is the arithmetic each decoder spends per pixel,
    so that is the whole figure, and the point needs no second axis: every
    published decoder is one bar, fixed, and ours is a bar with a piece taken
    out of it that depends on the picture.
    """
    L = json.loads((RES / "literature.json").read_text())
    items = sorted(L["image"], key=lambda d: d["kmac_px"])
    names = [m["name"] for m in items]
    vals = [m["kmac_px"] for m in items]

    rel, ours = 219.0, 170.0
    names = ["FLEX-UF (this work)"] + names
    vals = [rel] + vals

    fig, ax = plt.subplots(figsize=(ns.W1, 1.85))
    y = np.arange(len(names))[::-1]
    cols = [ns.VERM] + [ns.BLUE] * (len(names) - 1)
    ax.barh(y, vals, height=0.62, color=cols, lw=0)
    # The part of our bar a 0.1 dB budget removes, drawn as a notch rather than
    # a second bar: the cost is the same decoder, just less of it run.
    ax.barh(y[0], rel - ours, height=0.62, left=ours, color="white", lw=0)
    ax.barh(y[0], rel - ours, height=0.62, left=ours, color=ns.VERM, lw=0,
            alpha=0.22)
    ax.plot([ours, ours], [y[0] - 0.31, y[0] + 0.31], color=ns.VERM, lw=0.9)
    ax.text(rel + 40, y[0], "0.1 dB budget removes this", fontsize=6,
            color=ns.VERM, va="center")

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.get_yticklabels()[0].set_color(ns.VERM)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6, length=2, width=0.5)
    ax.set_xlabel("decoder arithmetic (kMAC per pixel)", fontsize=6)
    ax.set_xlim(0, 2650)
    for yy, v in zip(y[1:], vals[1:]):
        ax.text(v + 40, yy, f"{v:.0f}", fontsize=6, color=ns.INK2,
                va="center")
    fig.savefig(FIG / "field.png", dpi=500, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    print("  wrote field.png")


# ------------------------------------------------------- b. tiles are unequal
def tiles():
    """Two facts about one frame, which together are the whole premise.

    a. Tiles of one frame are not equally hard: stopping every one of them at
       the shallowest exit costs some almost nothing and others most of a
       decibel.
    b. The bits the entropy model already spent on a tile track how deep the
       Lagrangian oracle sends it. Spearman, not Pearson: what the router needs
       is the ORDER, and the exit index is ordinal.
    """
    T = json.loads((RES / "tile_table.json").read_text())
    D = np.array(T["D"])                       # [tiles, exits]
    j, K = T["j"], T["K"]
    loss = 10 * np.log10(D[:, j] / D[:, K - 1])
    bits = np.array(T["bits_per_tile"]) / np.mean(T["bits_per_tile"])

    # The oracle's own assignment at the operating point the paper reports.
    sweep = T["sweep"]
    near = min(sweep, key=lambda x: abs(x["db"] - 0.1))
    exits = np.array(near["map"])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    rho = spearman(bits, exits)

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 1.5),
                           gridspec_kw={"width_ratios": [1.0, 1.2],
                                        "wspace": 0.44})
    _bare(ax[0]); _bare(ax[1])
    ax[0].hist(loss, bins=14, color=ns.BLUE, lw=0)
    ax[0].set_xlabel("dB lost at the shallowest exit")
    ax[0].set_ylabel("tiles")
    ns.panel(ax[0], "a")
    ax[0].set_title(f"one 1080p frame, {len(loss)} tiles", fontsize=6,
                    color=ns.INK2, loc="left")

    # A little horizontal jitter, because the exit index is discrete and the
    # points would otherwise sit on top of one another.
    rng = np.random.default_rng(0)
    ax[1].scatter(bits, exits + rng.uniform(-0.13, 0.13, size=len(exits)),
                  s=8, color=ns.ORANGE, lw=0, zorder=3)
    ax[1].set_xlabel("tile bits / mean tile bits")
    ax[1].set_ylabel("exit the oracle assigns")
    ax[1].set_yticks(sorted(set(exits.tolist())))
    ax[1].text(0.96, 0.08, f"Spearman {rho:+.2f}", transform=ax[1].transAxes,
               ha="right", fontsize=6, color=ns.INK2)
    ns.panel(ax[1], "b")
    ax[1].set_title("at the 0.1 dB budget", fontsize=6, color=ns.INK2,
                    loc="left")
    fig.savefig(FIG / "tiles_unequal.png", dpi=500, bbox_inches="tight",
                pad_inches=0.01, facecolor="white")
    print(f"  wrote tiles_unequal.png   loss {loss.min():.3f} to "
          f"{loss.max():.3f} dB, Spearman(bits, exit) = {rho:+.2f}")


# ------------------------------------------------------------- c. the ladder
def ladder():
    C = json.loads((RES / "ceiling_measured.json").read_text())
    ex = C["exits"]
    k = [e["exit"] for e in ex]
    mo = [100 * e["modelled"] for e in ex]
    me = [100 * e["measured"] for e in ex]
    j = C["split_depth"]
    fig, ax = plt.subplots(figsize=(ns.W1, 1.12))
    _bare(ax)
    ax.step(k, me, where="mid", color=ns.BLUE, lw=1.1, label="counted with hooks")
    ax.step(k, mo, where="mid", color=ns.INK2, lw=0.8, ls="--",
            label="arithmetic model")
    ax.axhline(100, color=ns.GRID, lw=0.6)
    ax.text(k[-1], 101, "released decoder", fontsize=6, color=ns.INK2,
            ha="right")
    ax.axvspan(-0.5, j - 0.5, color="#f2f2f2", zorder=0)
    ax.text((j - 1) / 2, 62, "below the split:\nno tile may stop here",
            fontsize=6, color=ns.INK2, ha="center")
    ax.annotate("", xy=(j, me[j]), xytext=(j, 100),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=ns.VERM))
    ax.text(j + 0.12, (me[j] + 100) / 2, f"ceiling\n{C['ceiling_measured_pct']:.1f}%",
            fontsize=6, color=ns.VERM, va="center")
    ax.set_xlabel("exit")
    ax.set_ylabel("cost, released decode = 100")
    ax.set_ylim(55, 108)
    ax.set_xticks(k)
    ax.legend(frameon=False, fontsize=6, handlelength=1.5, loc="lower right",
              borderpad=0)
    fig.savefig(FIG / "ladder.png", dpi=500, bbox_inches="tight",
                pad_inches=0.01, facecolor="white")
    print("  wrote ladder.png")


# ------------------------------------------------------ d. where tiles go
def allocation():
    # supp_per_class_budgets.json carries all five rates; the older file has
    # three. Either way take the 0.1 dB rows only: at 0.3 and 0.5 the budget
    # saturates the ladder and every tile sits on the cheapest rung, which is a
    # true fact about saturation and a useless picture of allocation.
    _f = ("supp_per_class_budgets.json"
          if (RES / "supp_per_class_budgets.json").exists()
          else "per_class_RECIPE512.json")
    P = json.loads((RES / _f).read_text())
    rows = {r["qp"]: r for r in P["rows"] if abs(r["budget_db"] - 0.1) < 1e-9}
    qs = sorted(rows)
    K = len(next(iter(rows.values()))["per_class"].values().__iter__().__next__()["hist"]) \
        if rows else 6
    share = np.zeros((len(qs), K))
    for i, q in enumerate(qs):
        h = np.zeros(K)
        for cl in rows[q]["per_class"].values():
            h += np.array(cl["hist"], dtype=float)
        share[i] = 100 * h / h.sum()
    fig, ax = plt.subplots(figsize=(ns.W1, 1.12))
    _bare(ax)
    bottom = np.zeros(len(qs))
    cols = ["#dfe7f3", "#c2d3ea", "#9dbadd", "#6f99cc", "#3f74b5", "#1f4e96"]
    for e in range(K):
        ax.bar([str(q) for q in qs], share[:, e], 0.62, bottom=bottom,
               color=cols[e % len(cols)], lw=0, label=f"exit {e}")
        for i in range(len(qs)):
            if share[i, e] > 7:
                ax.text(i, bottom[i] + share[i, e] / 2, f"{share[i, e]:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if e >= 3 else ns.INK)
        bottom += share[:, e]
    ax.set_xlabel("quality index")
    ax.set_ylabel("share of tiles (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6, ncol=3, handlelength=1.0,
              columnspacing=0.8, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              borderpad=0)
    fig.savefig(FIG / "allocation.png", dpi=500, bbox_inches="tight",
                pad_inches=0.01, facecolor="white")
    print("  wrote allocation.png")


if __name__ == "__main__":
    field(); tiles(); ladder(); allocation()
