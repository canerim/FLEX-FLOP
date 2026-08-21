"""The three figures of the supplement's complexity section.

Designed at Nature single-column width so that they are placed at roughly 1:1
in the supplement's column and the 5 to 7 pt type stays 5 to 7 pt. Every value
is read from results/; nothing is typed in.

    python scripts/supp_b_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()
# These figures are placed at the supplement's column width, which is very close
# to Nature's single-column 89 mm, so they are drawn at that size and land at
# roughly 1:1. Type is therefore set at the size it will print at, rather than
# at a size a later shrink will make illegible.
plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5,
})
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def J(name):
    return json.loads((RES / name).read_text())


MA = J("mac_audit.json")["1920x1088"]
AC = J("adapter_cost.json")
C = AC["trunk_channels"]
GH, GW = MA["layers"][1]["out_hw"]
FPX = GH * GW
TOT = MA["total_mac"]
M_BLK = AC["block"]["mac_per_px"] * FPX
M_UP = MA["parts"]["upsample"]["gmac"] * 1e9
M_HD = MA["parts"]["head"]["gmac"] * 1e9
M_REP = (9 * C + C * C) * FPX
M_A1 = AC["adapters"]["conv1x1"]["mac_per_px"] * FPX
M_AF = AC["adapters"]["ffn"]["mac_per_px"] * FPX
NB = MA["n_trunk_blocks"]


def derived():
    out = []
    for k in range(6):
        run = max(k, 2)
        ad = 0.0 if run == 5 else (M_AF if (5 - run) * 2 >= 4 else M_A1)
        out.append((M_UP + M_HD + M_REP + (run + 1) * 2 * M_BLK + ad) / TOT)
    return out


# ---------------------------------------------------------------- figure 1
def fig_cost():
    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 1.42),
                           gridspec_kw={"width_ratios": [1.35, 1.0],
                                        "wspace": 0.42})

    # (a) where the arithmetic of one decode is, module by module
    up = MA["parts"]["upsample"]["share"]
    blk = MA["parts"]["trunk"]["share"] / NB
    hd = MA["parts"]["head"]["share"]
    names = ["up"] + [str(i) for i in range(1, NB + 1)] + ["hd", "sr"]
    vals = [up] + [blk] * NB + [hd, M_REP / TOT]
    # always-on: upsample, the j*b = 4 blocks before the split, head, repair
    fixed = [True] + [i < 4 for i in range(NB)] + [True, True]
    added = [False] * (NB + 1) + [False, True]
    cols = [(ns.SKY if f else ns.BLUE) for f in fixed]
    x = np.arange(len(vals))
    ax[0].bar(x, [100 * v for v in vals], 0.82, color=cols, lw=0)
    for i, a in enumerate(added):
        if a:
            ax[0].bar(x[i], 100 * vals[i], 0.82, color="none",
                      edgecolor=ns.VERM, lw=0.7, hatch="////")
    ax[0].axvline(4.5, color=ns.INK2, lw=0.6, ls="--")
    ax[0].text(4.7, 8.2, "split, j = 2", fontsize=6, color=ns.INK2,
               ha="left", va="top")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, fontsize=6)
    ax[0].tick_params(axis="x", pad=1)
    ax[0].set_ylabel("% of a released decode", labelpad=1)
    ax[0].set_ylim(0, 9.2)
    ax[0].set_xlabel("module", labelpad=1)
    ns.panel(ax[0], "a", dx=-0.14)

    # (b) the per-exit cost vector under three prices
    D = derived()
    B = J("why_qp_PAPER.json")["cost"]
    A = J("why_qp.json")["cost"]
    ks = np.arange(2, 6)
    for v, c, m, lab in [(A, ns.GREEN, "^", "2C²"),
                         (B, ns.ORANGE, "s", "5C², reported"),
                         (D, ns.BLUE, "o", "hook count")]:
        ax[1].plot(ks, [100 * (1 - v[e]) for e in ks], m + "-", color=c,
                   ms=2.6, lw=0.9, label=lab)
    for v, c in [(A, ns.GREEN), (B, ns.ORANGE), (D, ns.BLUE)]:
        ax[1].axhline(100 * (1 - v[2]), color=c, lw=0.5, ls=":")
    ax[1].set_xticks(ks)
    ax[1].set_xlabel("exit k", labelpad=1)
    ax[1].set_ylabel("saved (%)")
    ax[1].set_ylim(-4, 46)
    ax[1].legend(frameon=False, fontsize=6, handlelength=1.2, borderpad=0,
                 labelspacing=0.2, loc="lower left")
    ns.panel(ax[1], "b", dx=-0.28)

    fig.savefig(OUT / "supp_b_cost.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_check():
    P = J("supp_power.json")
    G = J("signalled_RECIPE512_grid.json")
    D = derived()
    B = J("why_qp_PAPER.json")["cost"]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 1.42),
                           gridspec_kw={"wspace": 0.46})

    # (a) predicted against the hook count, six decodes
    meas, pd_, pb = [], [], []
    for r in P["rows"]:
        h, n = r["hist"], r["n_tiles"]
        t = float(sum(h))
        cnt = [int(round(v / t * n)) for v in h]
        cnt[-1] += n - sum(cnt)
        m = [cnt[0] + cnt[1] + cnt[2], cnt[3], cnt[4], cnt[5]]
        meas.append(r["mac_saving_pct"])
        pd_.append(100 * (1 - sum(c * D[e] for c, e in zip(m, (2, 3, 4, 5))) / n))
        pb.append(100 * (1 - sum(c * B[e] for c, e in zip(m, (2, 3, 4, 5))) / n))
    lo, hi = 15, 37
    ax[0].plot([lo, hi], [lo, hi], color=ns.GRID, lw=0.6, zorder=0)
    ax[0].plot(meas, pb, "s", color=ns.ORANGE, ms=2.6, lw=0,
               label="5C², reported")
    ax[0].plot(meas, pd_, "o", color=ns.BLUE, ms=2.6, lw=0, label="hook count")
    ax[0].set_xlabel("measured saving (%)", labelpad=1)
    ax[0].set_ylabel("predicted (%)")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].legend(frameon=False, fontsize=6, handlelength=1.0, borderpad=0,
                 labelspacing=0.2, loc="upper left")
    ns.panel(ax[0], "a", dx=-0.26)

    # (b) the residual over the whole budget grid
    qs = sorted({r["qp"] for r in G["rows"]})
    tab = {(r["budget_db"], r["qp"]): r for r in G["rows"]}
    bs = sorted({r["budget_db"] for r in G["rows"]})
    for i, q in enumerate(qs):
        xs = [b for b in bs
              if tab.get((b, q), {}).get("model_minus_measured") is not None]
        ys = [tab[(b, q)]["model_minus_measured"] for b in xs]
        ax[1].plot(xs, ys, "o-", color=ns.SERIES[i % len(ns.SERIES)], ms=1.8,
                   lw=0.8, label=f"q{q}")
    ceil = 100 * (D[2] - B[2])
    ax[1].axhline(ceil, color=ns.INK2, lw=0.6, ls="--")
    ax[1].text(0.5, ceil - 0.012, "exit-2 map", fontsize=6, color=ns.INK2,
               ha="right", va="top")
    ax[1].set_xlabel("budget (dB)", labelpad=1)
    ax[1].set_ylabel("model − hook count (pts)")
    ax[1].legend(frameon=False, fontsize=6, handlelength=1.1, borderpad=0,
                 labelspacing=0.15, loc="lower right", ncol=2,
                 columnspacing=0.7)
    ns.panel(ax[1], "b", dx=-0.28)

    fig.savefig(OUT / "supp_b_check.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_units():
    P = J("supp_power.json")
    Lb = J("supp_latency_batch_1920x1080.json")
    Lc = J("supp_latency_cpu_1920x1080.json")
    L1 = J("supp_latency_1920x1080.json")
    L7 = J("supp_latency_1280x720.json")

    fig, ax = plt.subplots(1, 3, figsize=(ns.W1, 1.34),
                           gridspec_kw={"wspace": 0.72})

    # (a) one map, four units
    rows = [r for r in P["rows"] if r["size"] == "2048x1280"]
    qps = [r["qp"] for r in rows]
    x = np.arange(len(qps))
    w = 0.27
    for i, (v, c, lab) in enumerate([
            ([r["mac_saving_pct"] for r in rows], ns.BLUE, "MACs"),
            ([r["time_saving_pct"] for r in rows], ns.ORANGE, "seconds"),
            ([r["energy_saving_pct"] for r in rows], ns.GREEN, "joules")]):
        ax[0].bar(x + (i - 1) * w, v, w, color=c, label=lab, lw=0)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"q{q}" for q in qps])
    ax[0].set_xlabel("rate index", labelpad=1)
    ax[0].set_ylabel("saved (%)", labelpad=1)
    ax[0].set_ylim(0, 44)
    ax[0].set_yticks([0, 10, 20, 30])
    ax[0].legend(frameon=False, fontsize=6, handlelength=1.0, borderpad=0,
                 labelspacing=0.2, loc="upper right")
    ns.panel(ax[0], "a", dx=-0.40, dy=1.14)

    # (b) the same map on two device classes
    g = {r["qp"]: r for r in Lb["rows"] if r["batch"] == 1}
    c_ = {r["qp"]: r for r in Lc["rows"]}
    qs = sorted(c_)
    x = np.arange(len(qs))
    ax[1].plot(x, [g[q]["predicted_saving_pct"] for q in qs], "k--", lw=0.7,
               ms=0, label="MAC model")
    ax[1].plot(x, [g[q]["realised_saving_pct"] for q in qs], "o-",
               color=ns.VERM, ms=2.6, lw=0.9, label="A6000")
    ax[1].plot(x, [c_[q]["realised_saving_pct"] for q in qs], "s-",
               color=ns.PURPLE, ms=2.6, lw=0.9, label="CPU")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"q{q}" for q in qs])
    ax[1].set_xlabel("rate index", labelpad=1)
    ax[1].set_ylabel("saved (%)", labelpad=1)
    ax[1].set_ylim(12, 46)
    ax[1].set_yticks([20, 30, 40])
    ax[1].legend(frameon=False, fontsize=6, handlelength=1.2, borderpad=0,
                 labelspacing=0.2, loc="upper right")
    ns.panel(ax[1], "b", dx=-0.40, dy=1.14)

    # (c) a tile costs more in a group that has fewer tiles left
    for L, col, mk, lab in [(L1, ns.BLUE, "o", "1080p"),
                            (L7, ns.ORANGE, "s", "720p")]:
        st = {s["stage"]: s["ms"] for s in L["stages"]}
        n, t = [], []
        for key, ms in st.items():
            if key.startswith("group "):
                n.append(int(key.split("(")[1].split()[0]))
                t.append(ms / n[-1])
        ax[2].plot(n, t, mk + "-", color=col, ms=2.6, lw=0.9, label=lab)
    ax[2].set_xlabel("tiles alive", labelpad=1)
    ax[2].set_ylabel("ms per tile", labelpad=1)
    ax[2].set_xscale("log")
    ax[2].set_xticks([3, 5, 10, 20, 40])
    ax[2].set_xticklabels(["3", "5", "10", "20", "40"])
    ax[2].set_xticks([], minor=True)
    ax[2].set_ylim(0.392, 0.462)
    ax[2].set_yticks([0.40, 0.42, 0.44, 0.46])
    ax[2].legend(frameon=False, fontsize=6, handlelength=1.2, borderpad=0,
                 labelspacing=0.2, loc="lower left")
    ns.panel(ax[2], "c", dx=-0.44, dy=1.14)

    fig.savefig(OUT / "supp_b_units.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    fig_cost()
    fig_check()
    fig_units()
    print("wrote", OUT / "supp_b_cost.png", OUT / "supp_b_check.png",
          OUT / "supp_b_units.png")
