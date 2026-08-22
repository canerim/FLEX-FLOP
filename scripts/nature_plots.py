"""Three plots and one diagram, built to Nature's figure rules.

The rules this file is written against, from Nature's own figure guide:
panels in alphabetical order and laid out with little white space; panel size
follows content; text 5 to 7 pt at final size and consistent across panels;
panel letters lowercase, bold; no red-green pairs and no rainbow scales; keys
inside the figure rather than colour descriptions in the caption; text instead
of decorative icons.

    python scripts/nature_plots.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402

# These are hand-laid-out schematics: every box and label sits at a
# coordinate chosen against the others, so scaling the type moves text into
# text. They stay at the drawn size until they are redrawn at column width,
# which is a layout job and not a style switch.
ns.apply()
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
OUT = (FIG, ROOT / "paper" / "figures")


def _save(fig, name):
    """build_pdf reads paper/figures; writing only to docs/figures is how
    Figure 18 became a placeholder once (DECISIONS 97)."""
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=500, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")


# One sequential ramp, dark to light, used for every rate everywhere. Not a
# rainbow, and it survives greyscale.
RATE_COLS = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=ns.fs(6), length=2, width=0.5)
    ax.xaxis.label.set_size(ns.fs(5.8))
    ax.yaxis.label.set_size(ns.fs(5.8))
    return ax


def sv(r, key="saving_pct"):
    m = r.get(key + "_measured")
    return m if m is not None else r.get(key + "_vs_release")


# ------------------------------------------------------- 1. operating window
def window():
    """Saving against delivered quality, with the two limits that bound it."""
    G = json.loads((RES / "supp_budget_grid.json").read_text())
    S = {r["qp"]: r for r in json.loads(
        (RES / "saturation_RECIPE512_ctc53.json").read_text())["rows"]}
    qs = sorted({r["qp"] for r in G["rows"]})

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 1.72),
                           gridspec_kw={"width_ratios": [1.25, 1.0],
                                        "wspace": 0.3})
    tidy(ax[0]); tidy(ax[1])

    for n, q in enumerate(qs):
        rs = sorted([r for r in G["rows"] if r["qp"] == q],
                    key=lambda r: r["achieved_db"])
        ax[0].plot([r["achieved_db"] for r in rs],
                   [r["saving_pct"] for r in rs], "o-", ms=2.0, lw=0.9,
                   color=RATE_COLS[n % len(RATE_COLS)], label=f"q{q}")
        if q in S:
            ax[0].plot([S[q]["floor_db"]], [0], "v", ms=3.0,
                       color=RATE_COLS[n % len(RATE_COLS)], clip_on=False)
    ax[0].set_xlabel("distortion delivered, dB below the released decoder")
    ax[0].set_ylabel("decoder MACs saved (%)")
    ax[0].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.2, ncol=2,
                 borderpad=0, loc="lower right")
    # Down beside the floor markers, not across the top. At 0.94 of the axes it
    # ran straight through the q0 and q16 curves, which is where they are
    # steepest and where a reader is looking.
    ax[0].text(0.09, 0.10, "▽ floor: below this\nno allocation is feasible",
               transform=ax[0].transAxes, fontsize=ns.fs(6), color=ns.INK2,
               va="bottom", linespacing=1.25)
    ns.panel(ax[0], "a")

    # b. the two limits against rate, which is what makes the window finite
    fl = [S[q]["floor_db"] for q in qs if q in S]
    sa = [S[q]["saturation_db"] for q in qs if q in S]
    qq = [q for q in qs if q in S]
    ax[1].fill_between(qq, fl, sa, color="#dbe7f5", lw=0)
    ax[1].plot(qq, fl, "o-", ms=2.2, lw=0.9, color=ns.INK2, label="floor")
    ax[1].plot(qq, sa, "s-", ms=2.2, lw=0.9, color=ns.VERM, label="saturation")
    ax[1].set_xlabel("quality index")
    ax[1].set_ylabel("budget (dB)")
    ax[1].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.4, borderpad=0,
                 loc="upper left")
    ax[1].text(qq[len(qq) // 2], (fl[len(fl) // 2] + sa[len(sa) // 2]) / 2,
               "the window a budget\ncan do anything in", fontsize=ns.fs(6),
               color=ns.INK2, ha="center", va="center")
    ns.panel(ax[1], "b")
    _save(fig, "window.png")
    print("  wrote window.png")


# --------------------------------------------------- 2. regret concentration
def concentration():
    """How much of the loss sits in how few tiles."""
    H = json.loads((RES / "hybrid_lorenz_b01.json").read_text())
    rows = [r for r in H["rows"] if r.get("budget_reachable")]
    qs = sorted({r["qp"] for r in rows})

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 1.72),
                           gridspec_kw={"width_ratios": [1.0, 1.15],
                                        "wspace": 0.32})
    tidy(ax[0]); tidy(ax[1])

    ax[0].plot([0, 1], [0, 1], ls=":", lw=0.7, color=ns.GRID)
    for n, q in enumerate(qs):
        rs = sorted([r for r in rows if r["qp"] == q and
                     r.get("lorenz_at_rho") is not None],
                    key=lambda r: r["rho"])
        if not rs:
            continue
        x = [0] + [r["rho"] for r in rs]
        y = [0] + [r["lorenz_at_rho"] for r in rs]
        ax[0].plot(x, y, "-", lw=1.0, color=RATE_COLS[n % len(RATE_COLS)],
                   label=f"q{q}")
    ax[0].set_xlabel("fraction of tiles, worst first")
    ax[0].set_ylabel("fraction of the total regret")
    # The sweep covers rho up to 0.5, so the axis stops there. At xlim(0, 1)
    # the curves ended in the middle of an empty panel and looked truncated,
    # which is worse than an axis that says how far the measurement went.
    _xmax = max(r["rho"] for r in rows if r.get("lorenz_at_rho") is not None)
    ax[0].set_xlim(0, _xmax * 1.04); ax[0].set_ylim(0, 1.02)
    ax[0].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.2, ncol=2,
                 borderpad=0, loc="lower right")
    ax[0].text(0.30, 0.24, "equal shares", transform=ax[0].transAxes,
               fontsize=ns.fs(6), color=ns.INK2, rotation=27)
    ns.panel(ax[0], "a")

    gin = []
    for q in qs:
        g = [r["gini_regret"] for r in rows
             if r["qp"] == q and r.get("gini_regret") is not None]
        gin.append(float(np.mean(g)) if g else np.nan)
    ax[1].bar([str(q) for q in qs], gin, 0.6, color=RATE_COLS, lw=0)
    for i, g in enumerate(gin):
        ax[1].text(i, g + 0.012, f"{g:.2f}", ha="center", fontsize=ns.fs(6),
                   color=ns.INK2)
    ax[1].set_xlabel("quality index")
    ax[1].set_ylabel("Gini of per-tile regret")
    ax[1].set_ylim(0, 1.0)
    ax[1].axhline(0, color=ns.INK2, lw=0.5)
    ns.panel(ax[1], "b")
    _save(fig, "concentration.png")
    print(f"  wrote concentration.png   Gini {min(gin):.2f} to {max(gin):.2f}")


# --------------------------------------------------------- 3. who decides
def deciders():
    """Three ways to choose an exit, and the paradox in the third."""
    rr = json.loads((RES / "raterank_RECIPE512_b01.json").read_text())
    b1 = json.loads((RES / "router_RECIPE512_b01_PAPER.json").read_text())
    sg = json.loads((RES / "signalled_RECIPE512_ctc53.json").read_text())
    R = {r["qp"]: sv(r) for r in rr["rows"]}
    AG = {r["qp"]: r.get("agreement") for r in rr["rows"]}
    B = {r["qp"]: sv(r) for r in b1["rows"] if r.get("budget_reachable")}
    A = {r["qp"]: sv(r) for r in sg["rows"]
         if abs(r["budget_db"] - 0.1) < 1e-9 and r.get("budget_reachable")}
    qs = sorted(A)

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 1.72),
                           gridspec_kw={"width_ratios": [1.3, 1.0],
                                        "wspace": 0.3})
    tidy(ax[0]); tidy(ax[1])

    x = np.arange(len(qs)); w = 0.26
    ax[0].bar(x - w, [A[q] for q in qs], w, color="#08306b", lw=0,
              label="encoder search, 89 bits sent")
    ax[0].bar(x, [R[q] for q in qs], w, color=ns.ORANGE, lw=0,
              label="calibrated bit rule, nothing sent")
    ax[0].bar(x + w, [B[q] for q in qs], w, color="#9ecae1", lw=0,
              label="trained router, nothing sent")
    for i, q in enumerate(qs):
        ax[0].text(i, max(R[q], B[q]) + 0.7, f"+{R[q] - B[q]:.1f}",
                   ha="center", fontsize=ns.fs(6), color=ns.ORANGE)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"q{q}" for q in qs])
    ax[0].set_ylabel("decoder MACs saved (%)")
    ax[0].set_ylim(0, 34)
    ax[0].legend(frameon=False, fontsize=ns.fs(6), handlelength=1.1, borderpad=0,
                 loc="upper right")
    ns.panel(ax[0], "a")

    # b. agreement against what the agreement buys. The tempting version of
    # this panel compared the bit rule's agreement with the trained head's,
    # which is the stronger claim and not one the files support: the head's
    # agreement is recorded on held-out images at a fixed multiplier and the
    # rule's on the test set at the budget, so they are two different
    # measurements of a similarly named thing.
    ok = [q for q in qs if AG.get(q) is not None]
    xx = np.arange(len(ok))
    ax[1].bar(xx, [AG[q] for q in ok], 0.55, color="#9ecae1", lw=0)
    for i_, q in enumerate(ok):
        ax[1].text(i_, AG[q] + 0.015, f"{AG[q]:.2f}", ha="center",
                   fontsize=ns.fs(6), color=ns.INK2)
    ax[1].set_ylabel("tiles where the rule picks\nthe oracle's exit")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xticks(xx); ax[1].set_xticklabels([f"q{q}" for q in ok])
    a1b = ax[1].twinx()
    a1b.plot(xx, [100 * R[q] / A[q] for q in ok], "o-", ms=2.6, lw=0.9,
             color=ns.ORANGE)
    a1b.set_ylabel("% of the encoder search's saving", fontsize=ns.fs(6),
                   color=ns.ORANGE)
    a1b.tick_params(labelsize=ns.fs(6), length=2, width=0.5, colors=ns.ORANGE)
    a1b.set_ylim(0, 100)
    for sp in ("top",):
        a1b.spines[sp].set_visible(False)
    ax[1].text(0.5, 0.06, "picks a different exit on most tiles,\nand still "
               "captures three quarters of the search",
               transform=ax[1].transAxes, ha="center", fontsize=ns.fs(6),
               color=ns.INK2)
    ns.panel(ax[1], "b")
    _save(fig, "deciders.png")
    print("  wrote deciders.png")


# ------------------------------------------------------- 4. the mechanism
def mechanism():
    """How a multiplier turns a table of per-tile distortions into a map."""
    T = json.loads((RES / "tile_table.json").read_text())
    D = np.array(T["D"]); j, K = T["j"], T["K"]
    cost = np.array(T["cost"])
    near = min(T["sweep"], key=lambda s: abs(s["db"] - 0.1))
    lam = near["lam"]; chosen = np.array(near["map"])
    show = [0, 7, 18, 25, 33]                      # five representative tiles

    # Two by two at single-column width rather than a strip at double: a
    # full-width figure forces a page break in the two-column flow, and this
    # one does not earn a page.
    fig = plt.figure(figsize=(ns.W1, 2.35))
    gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.62)

    a0 = tidy(fig.add_subplot(gs[0, 0]))
    for n, t in enumerate(show):
        a0.plot(range(j, K), 10 * np.log10(D[t, j:] / D[t, K - 1]), "o-",
                ms=2.0, lw=0.8, color=RATE_COLS[n % len(RATE_COLS)],
                label=f"tile {t}")
    a0.set_xlabel("exit k"); a0.set_ylabel("D(t,k), dB")
    a0.set_xticks(range(j, K))
    a0.legend(frameon=False, fontsize=ns.fs(6), handlelength=1.1, borderpad=0)
    a0.set_title("what each tile loses", fontsize=ns.fs(6), color=ns.INK2,
                 loc="left", pad=3)
    ns.panel(a0, "a", dx=-0.34, dy=1.16)

    a1 = tidy(fig.add_subplot(gs[0, 1]))
    a1.step(range(j, K), cost[j:], where="mid", color=ns.INK2, lw=1.0)
    a1.set_xlabel("exit k"); a1.set_ylabel("c(k)")
    a1.set_xticks(range(j, K))
    a1.set_title("what each exit costs", fontsize=ns.fs(6), color=ns.INK2,
                 loc="left", pad=3)
    ns.panel(a1, "b", dx=-0.22, dy=1.16)

    a2 = tidy(fig.add_subplot(gs[1, 0]))
    for n, t in enumerate(show):
        L = D[t, j:] + lam * cost[j:]
        L = L / L.min()
        a2.plot(range(j, K), L, "o-", ms=2.0, lw=0.8,
                color=RATE_COLS[n % len(RATE_COLS)])
        kk = int(np.argmin(L)) + j
        a2.plot([kk], [1.0], "*", ms=5, color=RATE_COLS[n % len(RATE_COLS)])
    a2.set_xlabel("exit k")
    a2.set_ylabel("D + λc, scaled")
    a2.set_xticks(range(j, K))
    a2.set_title("add the price, take the argmin", fontsize=ns.fs(6), color=ns.INK2,
                 loc="left", pad=3)
    a2.text(0.97, 0.06, "★ the tile's exit", transform=a2.transAxes,
            ha="right", va="bottom", fontsize=ns.fs(6), color=ns.INK2)
    ns.panel(a2, "c", dx=-0.34, dy=1.16)

    a3 = fig.add_subplot(gs[1, 1])
    a3.set_xticks([]); a3.set_yticks([])
    for sp in a3.spines.values():
        sp.set_linewidth(0.4); sp.set_color(ns.GRID)
    grid = chosen.reshape(T["nh"], T["nw"])
    im = a3.imshow(grid, cmap="YlGnBu", vmin=j, vmax=K - 1)
    for r in range(T["nh"]):
        for c in range(T["nw"]):
            a3.text(c, r, str(grid[r, c]), ha="center", va="center",
                    fontsize=ns.fs(6),
                    color="white" if grid[r, c] >= K - 2 else ns.INK)
    a3.set_title(f"the map, λ = {lam:.2e}", fontsize=ns.fs(6), color=ns.INK2,
                 loc="left", pad=3)
    ns.panel(a3, "d", dx=-0.10, dy=1.16)
    _save(fig, "mechanism.png")
    print(f"  wrote mechanism.png   lambda {lam:.3e}, "
          f"exits {sorted(set(chosen.tolist()))}")


if __name__ == "__main__":
    window(); concentration(); deciders(); mechanism()
