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
    """Spines off, and nothing smaller than the house style allows.

    This used to set ticks at 6 pt and axis labels at 5.8, which is below the
    floor naturestyle applies to every other figure and below Nature's own.
    These four figures are the first four in the paper, so the first thing a
    reader met was the smallest type in it.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"],
                   length=2, width=0.5)
    ax.xaxis.label.set_size(plt.rcParams["axes.labelsize"])
    ax.yaxis.label.set_size(plt.rcParams["axes.labelsize"])
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
    # Taller than it was. At 1.12 inches the y-axis label did not fit the
    # figure and bbox_inches="tight" cropped its first letter off, so the
    # first thing a reader of Section 3 met was an axis labelled "ost,
    # released decode"; the released-decoder line ran under its own label and
    # the ceiling arrow sat on top of the step it measures.
    fig, ax = plt.subplots(figsize=(ns.W1, 1.55))
    _bare(ax)
    ax.step(k, me, where="mid", color=ns.BLUE, lw=1.3, label="counted with hooks",
            zorder=3)
    ax.step(k, mo, where="mid", color=ns.INK2, lw=0.9, ls="--",
            label="arithmetic model", zorder=3)
    ax.axhline(100, color=ns.INK2, lw=0.6, ls=(0, (1, 2)), zorder=1)
    # Above the line, at the left, where no series goes: the step reaches 100
    # only at the last exit, so the right-hand end is exactly where this label
    # used to collide with it.
    ax.text(k[-1] + 0.45, 100, "released\ndecoder", fontsize=6,
            color=ns.INK2, ha="right", va="bottom")
    ax.axvspan(-0.5, j - 0.5, color="#f2f2f2", zorder=0)
    # In the empty upper half of the grey band. At the bottom it lay across
    # the step it is describing, which the text audit cannot see because it
    # compares text against text and not against a line.
    ax.text((j - 1) / 2, 94, "below the split:\nno tile\nmay stop here",
            fontsize=6, color=ns.INK2, ha="center", va="top")
    # The ceiling arrow between the shallowest usable exit and the release,
    # drawn just right of the step so it measures the gap without covering it.
    xa = j + 0.42
    ax.annotate("", xy=(xa, me[j]), xytext=(xa, 100),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=ns.VERM,
                                shrinkA=0, shrinkB=0))
    ax.text(xa - 0.14, (me[j] + 100) / 2,
            f"ceiling\n{C['ceiling_measured_pct']:.1f}%",
            fontsize=6, color=ns.VERM, va="center", ha="right")
    ax.set_xlabel("exit")
    # Short enough to fit the axis. It used to read "cost, released decode =
    # 100", which needed more height than the figure has and was cropped to
    # "ost, released decode" in the paper; the released decoder's own line is
    # labelled on the plot, so the axis does not have to say it twice.
    # Eight characters. Anything longer is taller than a 1.55 inch axis and
    # gets cropped by the tight bounding box; the caption carries the unit.
    ax.set_ylabel("cost (%)")
    ax.set_ylim(52, 110)
    ax.set_xlim(-0.55, k[-1] + 0.55)
    ax.set_xticks(k)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.legend(frameon=False, fontsize=6, handlelength=1.6, loc="lower right",
              borderpad=0, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ladder.png", dpi=500, bbox_inches="tight",
                pad_inches=0.03, facecolor="white")
    print("  wrote ladder.png")


# ------------------------------------------------------ d. where tiles go
def allocation():
    """How the exit distribution moves with rate.

    This was a stacked bar and it had three faults. Its legend advertised six
    exits when the split depth makes the first two unreachable, so two of the
    six keys named categories that are structurally zero and two of the six
    shades were spent on them. Stacking put every band except the bottom one
    on a moving baseline, which is exactly the comparison the figure is for:
    the claim is that the mass moves deeper as the rate rises, and a reader
    could not follow any band but the first. And the share was encoded three
    times over -- bar height, gridded axis, and a number printed inside each
    segment.

    One line per reachable exit, labelled where it ends, is the same data with
    the trends legible and nothing drawn that cannot happen.
    """
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
    K = len(next(iter(rows[qs[0]]["per_class"].values()))["hist"])
    share = np.zeros((len(qs), K))
    for i, q in enumerate(qs):
        h = np.zeros(K)
        for cl in rows[q]["per_class"].values():
            h += np.array(cl["hist"], dtype=float)
        share[i] = 100 * h / h.sum()

    # Only the exits a tile can actually take. The clamp at the split depth
    # makes the shallower ones structurally empty, and drawing an empty
    # category invites the reader to wonder where it went.
    used = [e for e in range(K) if share[:, e].max() > 0]
    ramp = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b", "#041f4a"]

    fig, ax = plt.subplots(figsize=(ns.W1, 1.25))
    _bare(ax)
    x = np.arange(len(qs))
    for n, e in enumerate(used):
        ax.plot(x, share[:, e], "-o", color=ramp[n % len(ramp)], lw=1.2,
                ms=3, zorder=3, label=f"exit {e}")
    # Direct end-labels were the first attempt and the audit refused them: the
    # four lines converge into 5 points of each other at the highest rate, so
    # the labels sat on top of one another. The key goes in the upper right,
    # which no series enters -- the deepest exit reaches 28% and the axis runs
    # to 70.
    ax.legend(frameon=False, fontsize=6, ncol=2, handlelength=1.2,
              columnspacing=0.9, labelspacing=0.25, loc="upper right",
              borderpad=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlim(-0.15, len(qs) - 1 + 0.12)
    ax.set_ylim(0, max(70, share.max() * 1.08))
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("quality index")
    ax.set_ylabel("share of tiles (%)")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "allocation.png", dpi=500, bbox_inches="tight",
                pad_inches=0.01, facecolor="white")
    print("  wrote allocation.png")
    for n, e in enumerate(used):
        print(f"     exit {e}: " + " ".join(f"{v:5.1f}" for v in share[:, e]))


if __name__ == "__main__":
    field(); tiles(); ladder(); allocation()
